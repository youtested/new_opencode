"""Session list: opencode-style session picker.

Groups persisted + live sessions into ``Today`` / ``Yesterday`` / older-day
sections (newest first). Up/Down navigate, Enter resumes the highlighted
session, type in the search box to filter, Esc closes. Ctrl+N renames the
highlighted session, Ctrl+D deletes it
(after a confirmation popup). Section headers are non-selectable separators —
navigation skips over them.

Rendering uses a single Textual ``OptionList`` widget instead of one ``ListItem``
per row. That makes the popup open several times faster on slow devices (the
old per-row widgets + per-row width measurement took seconds to mount).
"""

from __future__ import annotations

import datetime
from typing import Any, Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from ..session import group_sessions
from .theme import active_theme

# Titles longer than this are truncated so the row fits the popup width on one
# line (the previous build let them slide sideways instead; OptionList can't pan
# a single row past the viewport, so truncation is the fast substitute).
_TITLE_MAX = 40

# How often the popup polls its data provider for changes (running→done,
# new/deleted sessions) while it sits open.
_REFRESH_SECONDS = 2.0


def _clock(timestamp: float) -> str:
    """12h clock without strftime's locale pitfalls (`%p` on Android)."""
    try:
        dt = datetime.datetime.fromtimestamp(timestamp)
    except (OSError, OverflowError, ValueError):
        return ""
    hour = dt.hour % 12 or 12
    return f"{hour}:{dt.minute:02d}{'am' if dt.hour < 12 else 'pm'}"


def _clip(text: str, limit: int = _TITLE_MAX) -> str:
    """Truncate an already-escaped title to one display line."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _agent_badge(theme: Any, agent: str) -> str:
    """`▣ Build` / `▣ Plan` — badge with the agent's own color on the
    square, plain label text (mirrors the chat's `▣ Build · model` line)."""
    from rich.markup import escape as _escape

    name = str(agent or "build").strip() or "build"
    label = _escape(name.title())
    return f"[{theme.agent_color(name)}]▣[/] {label}"


class SessionSearchNav(Message):
    """The session search box wants the list to move/select while it keeps
    focus (mirrors the models picker's ModelsNav pattern, kept tiny)."""

    def __init__(self, action: str) -> None:
        super().__init__()
        self.action = action


class _SessionSearchInput(Input):
    """Single-row filter box; Up/Down/Enter/Escape drive the list instead of
    being consumed by the Input itself.

    Ctrl+E / Ctrl+D are shadowed too: Input natively binds them (end-of-line
    / delete-char), which would eat the popup's Select and Delete shortcuts
    while the box is focused — so they are re-posted to the popup instead.
    Ctrl+N / Ctrl+S have no native Input binding and bubble up untouched."""

    BINDINGS = [
        Binding("up", "nav_up", show=False),
        Binding("down", "nav_down", show=False),
        Binding("enter", "nav_select", show=False),
        Binding("escape", "nav_close", show=False),
        Binding("ctrl+e", "sess_select", show=False),
        Binding("ctrl+d", "sess_delete", show=False),
    ]

    async def _on_key(self, event: Key) -> None:
        # Space is printable, so Input._on_key eats it before ANY binding
        # (including a "space" binding here) is ever consulted. In Select
        # mode space must flip the checkbox, so intercept it first and
        # re-route as a nav message; otherwise let normal typing happen.
        if event.key == "space":
            try:
                select_mode = bool(getattr(self.screen, "select_mode", False))
            except Exception:
                select_mode = False
            if select_mode:
                event.stop()
                event.prevent_default()
                self.post_message(SessionSearchNav("space"))
                return
        await super()._on_key(event)

    def action_nav_up(self) -> None:
        self.post_message(SessionSearchNav("up"))

    def action_nav_down(self) -> None:
        self.post_message(SessionSearchNav("down"))

    def action_nav_select(self) -> None:
        self.post_message(SessionSearchNav("select"))

    def action_nav_close(self) -> None:
        self.post_message(SessionSearchNav("close"))

    def action_sess_select(self) -> None:
        self.post_message(SessionSearchNav("select_mode"))

    def action_sess_delete(self) -> None:
        self.post_message(SessionSearchNav("delete"))


class RenameDialog(ModalScreen[tuple[str, str]]):
    """Modal asking for a new session name; dismisses with (session_id, title),
    or ("", "") on Esc. ``on_rename`` runs before dismissal and may reject the
    title with an error message."""

    def __init__(
        self,
        session_id: str,
        current_title: str,
        on_rename: Callable[[str, str], str | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id
        self.current_title = current_title
        self.on_rename = on_rename

    def compose(self) -> ComposeResult:
        with Vertical(classes="cmd-popup"):
            yield Static("  Rename session  ", classes="cmd-popup-title")
            yield Input(
                value=self.current_title,
                id="rename-input",
                placeholder="New session name",
            )
            yield Static("Enter to save · Esc to cancel", classes="cmd-popup-usage")

    def on_mount(self) -> None:
        inp = self.query_one("#rename-input", Input)
        inp.focus()
        inp.cursor_position = len(inp.value)

    def on_input_submitted(self, event: Any) -> None:
        event.stop()
        title = event.value.strip()
        if not title:
            self.app.notify("Session name can't be empty.", severity="warning")
            return
        if self.on_rename:
            error = self.on_rename(self.session_id, title)
            if error:
                self.app.notify(error, severity="error")
                return
        self.dismiss((self.session_id, title))

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(("", ""))
            event.stop()


class ConfirmDeleteDialog(ModalScreen[tuple[str, bool]]):
    """Confirmation popup before deleting a session; dismisses with
    (session_id, confirmed). Esc / n / Cancel = no, y / Delete = yes."""

    def __init__(self, session_id: str, title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id
        # NOT `self.title`: that would shadow Screen.title (a Textual reactive)
        # with raw unescaped markup — inert today, a landmine if a header lands.
        self.session_title = title

    def compose(self) -> ComposeResult:
        with Vertical(classes="cmd-popup"):
            yield Static("  Delete session  ", classes="cmd-popup-title")
            yield Static(
                "This can't be undone.",
                classes="cmd-popup-usage",
            )
            with Horizontal(classes="cmd-popup-actions"):
                yield Button("Delete", id="del-yes", variant="error")
                yield Button("Cancel", id="del-no", variant="default")

    def on_mount(self) -> None:
        self.query_one("#del-no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id is not None:
            self.dismiss((self.session_id, event.button.id == "del-yes"))
            event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss((self.session_id, False))
            event.stop()
        elif event.key in ("y", "Y"):
            self.dismiss((self.session_id, True))
            event.stop()
        elif event.key in ("n", "N"):
            self.dismiss((self.session_id, False))
            event.stop()


class AgentsView(ModalScreen[str]):
    """In-parent agents list: every model launched inside one session.

    Opened from a parent's ``agents (N)`` row. Lists ONLY that parent's
    launched agents — Enter resumes one, Esc goes back. Dismisses with the
    chosen child session id (or None).
    """

    def __init__(
        self,
        parent_id: str,
        parent_title: str = "",
        agents: list[dict[str, Any]] | None = None,
        current: str = "",
        on_refresh: Callable[[], list[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.parent_id = parent_id
        self.parent_title = parent_title
        self.agents = list(agents or [])
        self.current = current
        self.on_refresh = on_refresh

    def _rows(self) -> list[tuple[str | None, str]]:
        theme = active_theme()
        rows: list[tuple[str | None, str]] = []
        for s in self.agents:
            title = _clip(escape(s.get("title") or "(untitled)"))
            agent = escape(str(s.get("agent") or "build"))
            status = s.get("status", "")
            state = ""
            if status == "running":
                state = f"[{theme.c('warning')}](running)[/]"
            elif status == "error":
                state = f"[{theme.c('error')}](error)[/]"
            mark = (
                f"[{theme.c('success')}]•[/]"
                if s.get("id") == self.current
                else " "
            )
            row = f"[{theme.c('text')}]  ↳ {mark} {title}[/]"
            row += f"  {_agent_badge(theme, agent)}"
            if state:
                row += f"  {state}"
            rows.append((s.get("id", ""), row))
        return rows

    def _options(self) -> list[Option]:
        options: list[Option] = []
        seen: set[str] = set()
        for sid, text in self._rows():
            if not sid or sid in seen:
                continue
            seen.add(sid)
            options.append(Option(text, id=sid))
        return options

    def compose(self) -> ComposeResult:
        theme = active_theme()
        with Vertical(classes="cmd-popup session-popup"):
            yield Static("  Agents  ", classes="cmd-popup-title")
            yield Static(
                f"[{theme.c('text_muted')}]launched inside: {escape(self.parent_title or self.parent_id[:12])}[/]",
                classes="cmd-popup-usage",
            )
            yield Static(
                f"[{theme.c('text_muted')}]↑/↓ navigate · Enter resume · Esc back[/]",
                classes="cmd-popup-usage",
            )
            yield OptionList(id="agents-list")

    def on_mount(self) -> None:
        self._populate()
        if self.on_refresh is not None:
            self.set_interval(_REFRESH_SECONDS, self._refresh_tick)

    def _populate(self) -> None:
        try:
            ol = self.query_one("#agents-list", OptionList)
        except Exception:
            return
        ol.clear_options()
        ol.add_options(self._options())
        idx = 0
        for j, (rid, _) in enumerate(self._rows()):
            if rid == self.current:
                idx = j
                break
        try:
            ol.highlighted = idx
            ol.scroll_to_highlight()
            ol.focus()
        except Exception:
            pass

    def _refresh_tick(self) -> None:
        if getattr(self, "_agents_refresh_busy", False):
            return
        self._agents_refresh_busy = True
        try:
            self.run_worker(self._agents_refresh_async(), exclusive=True)
        except Exception:
            self._agents_refresh_busy = False

    async def _agents_refresh_async(self) -> None:
        try:
            import asyncio as _asyncio

            fresh = await _asyncio.to_thread(self.on_refresh)
        except Exception:
            return
        finally:
            try:
                self._agents_refresh_busy = False
            except Exception:
                pass
        if not isinstance(fresh, list) or fresh == self.agents:
            return
        try:
            ol = self.query_one("#agents-list", OptionList)
            opt = ol.highlighted_option
            keep = str(opt.id) if opt is not None and opt.id else None
        except Exception:
            keep = None
        self.agents = fresh
        self._populate()
        if keep:
            try:
                ol = self.query_one("#agents-list", OptionList)
                for j, (rid, _) in enumerate(self._rows()):
                    if rid == keep:
                        ol.highlighted = j
                        ol.scroll_to_highlight()
                        break
            except Exception:
                pass

    def _picked(self, option: Any) -> str | None:
        if option is None or option.id is None:
            return None
        return str(option.id)

    def on_option_list_option_selected(self, event: Any) -> None:
        event.stop()
        picked = self._picked(getattr(event, "option", None))
        if picked is None:
            picked = getattr(event, "option_id", None)
            if picked is None:
                return
        self.dismiss(str(picked))

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
        elif event.key in ("left", "right"):
            event.stop()


class SessionList(ModalScreen[str]):
    """Modal popup listing sessions; dismisses with the chosen session id."""

    BINDINGS = [
        Binding("ctrl+n", "rename", "Rename"),
        Binding("ctrl+d", "delete", "Delete"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+e", "toggle_select", "Select"),
    ]

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        current: str = "",
        on_rename: Callable[[str, str], str | None] | None = None,
        on_delete: Callable[[str], bool] | None = None,
        on_save: Callable[[str], bool] | None = None,
        on_refresh: Callable[[], list[dict[str, Any]]] | None = None,
        on_agents: Callable[[str], list[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sessions = sessions
        self.current = current
        self.on_rename = on_rename
        self.on_delete = on_delete
        self.on_save = on_save
        # Optional live data provider: polled while the popup is open so the
        # list is not a frozen snapshot (running→done transitions, new/removed
        # sessions show up within a couple of seconds).
        self.on_refresh = on_refresh
        # Optional launched-agents provider: session_id -> its children.
        # When set, each parent row gets an inline "agents (N)" line; Enter
        # on it drills into the in-parent agents list.
        self.on_agents = on_agents
        self.on_open_agents: Callable[[str], None] | None = None
        self.selected: set[str] = set()
        self.select_mode = False
        self._query: str = ""

    def _agent_count(self, sid: str) -> int:
        if not sid or self.on_agents is None:
            return 0
        try:
            kids = self.on_agents(sid) or []
            return len(kids)
        except Exception:
            return 0

    def _rows(self) -> list[tuple[str | None, str]]:
        """Flat, display-ready rows: (id, text). ``id`` is None for the
        non-selectable section headers.

        Parents with launched agents carry an inline ``agents (N)`` row;
        Enter on it opens the dedicated in-parent Agents popup.
        """
        theme = active_theme()
        query = (self._query or "").strip().lower()
        groups = group_sessions(self.sessions)
        rows = []
        for label, items in groups:
            if query:
                items = [
                    s
                    for s in items
                    if query in str(s.get("title") or "").lower()
                    or query in str(s.get("agent") or "").lower()
                    or query in str(s.get("id") or "").lower()
                ]
                if not items:
                    continue
            rows.append(
                (
                    None,
                    f"[bold {theme.c('accent')}]{escape(label)}[/]",
                )
            )
            for s in items:
                title = _clip(escape(s.get("title") or "(untitled)"))
                # agent is session data too — it gets the same escaping as
                # titles (a poisoned value used to MarkupError-crash the popup)
                agent = escape(str(s.get("agent") or "build"))
                status = s.get("status", "")
                try:
                    created = float(s.get("created") or 0.0)
                except (TypeError, ValueError):
                    created = 0.0
                when = _clock(created) if label in ("Today", "Yesterday") else ""
                state = ""
                if status == "running":
                    state = f"[{theme.c('warning')}](running)[/]"
                elif status == "error":
                    state = f"[{theme.c('error')}](error)[/]"
                if self.select_mode:
                    mark = "[x]" if s.get("id") in self.selected else "[ ]"
                else:
                    mark = (
                        f"[{theme.c('success')}]•[/]"
                        if s.get("id") == self.current
                        else " "
                    )
                row = f"[{theme.c('text')}]{mark} {title}[/]"
                if when:
                    row += f"  [{theme.c('text_muted')}]{when}[/]"
                row += f"  {_agent_badge(theme, agent)}"
                if state:
                    row += f"  {state}"
                rows.append((s.get("id", ""), row))
        return rows

    def _options(self) -> list[Option]:
        """The OptionList content: disabled separator rows for section headers,
        then one selectable Option per session.

        Rows with a missing/empty id are skipped and ids are deduped — a
        duplicate Option id makes Textual raise DuplicateID, which the message
        pump swallows, leaving the popup silently EMPTY."""
        options: list[Option] = []
        seen: set[str] = set()
        for i, (sid, text) in enumerate(self._rows()):
            if sid is None:
                options.append(Option(text, id=f"__hdr__{i}", disabled=True))
                continue
            if not sid or sid in seen:
                continue
            seen.add(sid)
            options.append(Option(text, id=sid))
        return options

    def compose(self) -> ComposeResult:
        theme = active_theme()
        with Vertical(classes="cmd-popup session-popup"):
            yield Static("  Sessions  ", classes="cmd-popup-title")
            yield _SessionSearchInput(
                placeholder="Search sessions…",
                id="session-search",
            )
            yield Static(
                f"[{theme.c('text_muted')}]↑/↓ navigate · Enter resume · Esc close · Ctrl+E select · Ctrl+S save[/]",
                classes="cmd-popup-usage",
            )
            yield Static(
                f"[{theme.c('text_muted')}]type to filter · Ctrl+N rename · Ctrl+D delete · in Select: space/enter toggles[/]",
                classes="cmd-popup-usage",
            )
            # An empty OptionList is yielded first so the popup shell paints
            # instantly; the rows are added on the next refresh (_populate) so
            # opening the picker never waits on constructing every row widget.
            yield OptionList(id="session-list")
            with Horizontal(classes="cmd-popup-actions"):
                yield Button("Save", id="btn-save", variant="success")
                yield Button("Select", id="btn-select", variant="default")
                yield Button("Delete sel", id="btn-del-sel", variant="error")

    def on_mount(self) -> None:
        # The shell was already composed empty; filling the rows here keeps the
        # whole open fast (single lightweight widget) without ever blocking on
        # per-row widget construction.
        self._populate()
        self._focus_search()
        if self.on_refresh is not None:
            # NB: NOT named "_auto_refresh" — that identifier is Textual's own
            # DOMNode backing field for its auto-refresh reactive, and defining
            # a method with that name gets silently overwritten with None.
            self.set_interval(_REFRESH_SECONDS, self._refresh_tick)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """The header buttons were dead code until now: Save / Select /
        Delete-sel only worked via Ctrl+S / Ctrl+E / Ctrl+D. Clicking Select
        must actually enter select mode."""
        bid = event.button.id
        if bid == "btn-select":
            await self.action_toggle_select()
        elif bid == "btn-save":
            self.action_save()
            self._refocus_list()
        elif bid == "btn-del-sel":
            self.action_delete()
        else:
            return
        event.stop()

    def _refocus_list(self) -> None:
        self._focus_search()

    def _focus_search(self) -> None:
        try:
            self.query_one("#session-search", Input).focus()
        except Exception:
            pass

    def _refresh_tick(self) -> None:
        """Poll the live data provider and repaint only on actual change, so
        the popup is never a frozen snapshot (a session finishing mid-open
        used to keep its "(running)" badge forever).

        Disk work runs in a worker: the old inline on_refresh() did a full
        session-dir scan on the UI thread every 2s, freezing the whole
        screen on slow storage.
        """
        if getattr(self, "_refresh_busy", False):
            return
        self._refresh_busy = True
        try:
            self.run_worker(self._refresh_async(), exclusive=True)
        except Exception:
            self._refresh_busy = False

    async def _refresh_async(self) -> None:
        try:
            import asyncio as _asyncio

            fresh = await _asyncio.to_thread(self.on_refresh)
        except Exception:
            return
        finally:
            try:
                self._refresh_busy = False
            except Exception:
                pass
        if not isinstance(fresh, list) or fresh == self.sessions:
            return
        try:
            ol = self.query_one("#session-list", OptionList)
            opt = ol.highlighted_option
            keep = str(opt.id) if opt is not None and opt.id else None
        except Exception:
            keep = None
        self.sessions = fresh
        try:
            self._populate(focus=keep)
        except Exception:
            pass

    def _selectable_ids(self) -> set[str]:
        """Ids Enter may act on in the current view (excludes headers)."""
        ids: set[str] = set()
        for rid, _ in self._rows():
            if rid is None:
                continue
            if str(rid).startswith("__hdr__"):
                continue
            ids.add(str(rid))
        return ids

    def _populate(self, focus: str | None = None) -> None:
        ol = self.query_one("#session-list", OptionList)
        ol.clear_options()
        ol.add_options(self._options())
        want = focus or self.current
        if want and want not in self._selectable_ids():
            want = ""
        self._focus_sid(want, default=0)

    def on_input_changed(self, event: Input.Changed) -> None:
        try:
            if event.input.id != "session-search":
                return
        except Exception:
            return
        event.stop()
        self._query = event.value or ""
        self._rebuild_keep_focus()

    def _rebuild_keep_focus(self) -> None:
        try:
            ol = self.query_one("#session-list", OptionList)
            opt = ol.highlighted_option
            keep = str(opt.id) if opt is not None and opt.id else None
        except Exception:
            keep = None
        self._populate(focus=keep)
        self._focus_search()

    def _focus_sid(self, session_id: str, default: int = 0) -> None:
        ol = self.query_one("#session-list", OptionList)
        # headers are disabled, so pointing at the session's own row is safe;
        # if it isn't listed, land on the first selectable row (skip headers).
        rows = self._rows()
        index = default
        for j, (rid, _) in enumerate(rows):
            if rid == session_id:
                index = j
                break
        if index <= 0:
            index = next(
                (j for j, (rid, _) in enumerate(rows)
                 if rid is not None and not str(rid).startswith("__hdr__")),
                0,
            )
        try:
            ol.highlighted = index
            ol.scroll_to_highlight()
        except Exception:
            pass

    def _selected(self) -> dict[str, Any] | None:
        """The session dict under the highlight, or None for a header.

        Inside the agents view the children come from on_agents, not
        self.sessions, so look them up there first.
        """
        try:
            ol = self.query_one("#session-list", OptionList)
            opt = ol.highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        oid = str(opt.id)
        return next((s for s in self.sessions if str(s.get("id")) == oid), None)

    def _rebuild(self) -> None:
        """Re-render the list after a rename/delete/select toggle. Much cheaper
        than the old ListView swap: just clear and re-add options."""
        self._populate()

    def _picked(self, option: Any) -> str | None:
        """Map a highlighted Option to a session id; a header is a no-op."""
        if option is None or option.id is None:
            return None
        if str(option.id).startswith("__hdr__"):
            return None
        return str(option.id)

    def _toggle_selection(self, session_id: str) -> None:
        """Select-mode: flip a session's checkbox, keeping the highlight on it."""
        if session_id in self.selected:
            self.selected.discard(session_id)
        else:
            self.selected.add(session_id)
        self._rebuild()
        self._focus_sid(session_id)
        self._focus_search()

    def _move_highlight(self, direction: int) -> None:
        try:
            ol = self.query_one("#session-list", OptionList)
        except Exception:
            return
        rows = self._rows()
        selectable = [
            j
            for j, (rid, _) in enumerate(rows)
            if rid is not None and not str(rid).startswith("__hdr__")
        ]
        if not selectable:
            return
        try:
            cur = int(ol.highlighted)
        except Exception:
            cur = selectable[0]
        if cur not in selectable:
            ol.highlighted = selectable[0] if direction > 0 else selectable[-1]
        elif direction > 0:
            nxt = [j for j in selectable if j > cur]
            if nxt:
                ol.highlighted = nxt[0]
        else:
            prv = [j for j in selectable if j < cur]
            if prv:
                ol.highlighted = prv[-1]
        try:
            ol.scroll_to_highlight()
        except Exception:
            pass
        self._focus_search()

    def _activate_highlighted(self) -> None:
        sess = self._selected()
        if sess is None:
            return
        sid = str(sess.get("id") or "")
        if self.select_mode:
            self._toggle_selection(sid)
        else:
            self.dismiss(sid)

    def on_session_search_nav(self, event: SessionSearchNav) -> None:
        action = getattr(event, "action", "")
        if action == "up":
            self._move_highlight(-1)
        elif action == "down":
            self._move_highlight(1)
        elif action == "select":
            self._activate_highlighted()
        elif action == "space":
            if self.select_mode:
                sess = self._selected()
                if sess is not None:
                    self._toggle_selection(str(sess.get("id") or ""))
        elif action == "select_mode":
            self._toggle_select_sync()
        elif action == "delete":
            self.action_delete()
        elif action == "close":
            if self._query.strip():
                self._clear_search()
            elif self.select_mode:
                self.select_mode = False
                self.selected.clear()
                self._rebuild()
                self._focus_search()
            else:
                self.dismiss(None)

    def _clear_search(self) -> None:
        self._query = ""
        try:
            self.query_one("#session-search", Input).value = ""
        except Exception:
            pass
        self._rebuild_keep_focus()

    def on_option_list_option_selected(self, event: Any) -> None:
        event.stop()
        picked = self._picked(getattr(event, "option", None))
        if picked is None:
            picked = getattr(event, "option_id", None)
            if picked is None or str(picked).startswith("__hdr__"):
                return
        picked = str(picked)
        if picked.startswith("__agents__"):
            opener = getattr(self, "on_open_agents", None)
            if callable(opener):
                try:
                    opener(picked[len("__agents__"):])
                except Exception:
                    pass
            return
        if self.select_mode:
            self._toggle_selection(picked)
        else:
            self.dismiss(picked)

    def _search_focused(self) -> bool:
        try:
            return bool(self.query_one("#session-search", Input).has_focus)
        except Exception:
            return False

    async def on_key(self, event: Key) -> None:
        if event.key == "escape":
            if self._search_focused():
                # the search box's own Esc binding already posted a
                # SessionSearchNav("close") for this keypress (clear search
                # first, close second) — handling it here too would clear
                # AND close on a single press.
                return
            if self._query.strip():
                self._clear_search()
            elif self.select_mode:
                self.select_mode = False
                self.selected.clear()
                self._rebuild()
                self._focus_search()
            else:
                self.dismiss(None)
            event.stop()
        elif event.key == "space" and self.select_mode:
            # the help line promises "space/enter toggles": in Select mode
            # space flips the checkbox even though the search box is focused
            # (typing a literal space into the query can wait until Esc
            # leaves Select mode — a space-only query strips to "" anyway).
            sess = self._selected()
            if sess is not None:
                self._toggle_selection(str(sess.get("id") or ""))
            event.stop()
        elif event.key in ("left", "right"):
            # No per-row panning anymore (OptionList can't slide one row past
            # the viewport) — absorb the keys so they don't fall through to the
            # app's sub-agent navigation underneath the modal. EXCEPT when the
            # search box is focused: there they move the text cursor.
            if self._search_focused():
                return
            event.stop()

    # -- select mode / save ----------------------------------------------
    def _toggle_select_sync(self) -> None:
        """Ctrl+E handler shared by the screen binding and the search box's
        re-posted message (Textual can't await an async action from a
        message handler, so the message path needs a sync entry point)."""
        self.select_mode = not self.select_mode
        if not self.select_mode:
            self.selected.clear()
        self._rebuild()
        self._focus_search()

    async def action_toggle_select(self) -> None:
        self._toggle_select_sync()

    def action_save(self) -> None:
        if self.on_save is None:
            return
        # Save acts on the HIGHLIGHTED row (the old behaviour silently saved
        # whatever session was currently open, no matter which row you were on).
        sess = self._selected()
        if sess is None:
            self.app.notify("Highlight a session to save.", severity="warning")
            return
        self.on_save(str(sess.get("id") or ""))

    # -- rename / delete --------------------------------------------------
    def action_rename(self) -> None:
        sess = self._selected()
        if sess is None:
            return
        self.app.push_screen(
            RenameDialog(sess["id"], sess.get("title") or "", on_rename=self.on_rename),
            self._after_rename,
        )

    async def _after_rename(self, result: tuple[str, str] | None) -> None:
        try:
            if result and result[0]:
                sid, title = result
                for s in self.sessions:
                    if s.get("id") == sid:
                        s["title"] = title
                        break
                self._rebuild()
                self._focus_sid(sid)
            self._focus_search()
        except Exception:
            pass

    def action_delete(self) -> None:
        if self.select_mode:
            if not self.selected:
                self.app.notify("No sessions selected.", severity="warning")
                return
            self.app.push_screen(
                ConfirmDeleteDialog(
                    "batch",
                    f"{len(self.selected)} selected conversation(s)",
                ),
                self._after_batch_delete,
            )
            return
        sess = self._selected()
        if sess is None:
            return
        # every session — active or not — goes through the confirmation popup;
        # confirming an active session resets the workspace in the app callback.
        self.app.push_screen(
            ConfirmDeleteDialog(sess["id"], sess.get("title") or "(untitled)"),
            self._after_delete,
        )

    async def _after_batch_delete(self, result: tuple[str, bool] | None) -> None:
        if result and result[1]:
            failed: list[str] = []
            for sid in list(self.selected):
                if self.on_delete is not None and not self.on_delete(sid):
                    failed.append(sid)
            removed = len(self.selected) - len(failed)
            keep_ids = set(self.selected)
            self.sessions = [
                s
                for s in self.sessions
                if s.get("id") not in keep_ids or s.get("id") in failed
            ]
            self.selected = set(failed)
            if not self.selected:
                self.select_mode = False
            self._rebuild()
            msg = f"Deleted {removed} session{'s' if removed != 1 else ''}."
            if failed:
                msg += f" Kept {len(failed)} (still running)."
            self.app.notify(msg)
        self._focus_search()

    async def _after_delete(self, result: tuple[str, bool] | None) -> None:
        if result and result[1]:
            sid = result[0]
            if self.on_delete is not None and not self.on_delete(sid):
                self.app.notify("Couldn't delete that session.", severity="error")
            else:
                self.sessions = [s for s in self.sessions if s.get("id") != sid]
                self._rebuild()
        self._focus_search()