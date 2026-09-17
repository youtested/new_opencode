"""Agent permission editor: every live tool + current action for one agent.

Opened from the AgentPicker (`p` or the Permissions button). Row 0 is the
read-only switch (`r` flips); rows below show each tool the agent can see
(live registry names, so newly added tools appear automatically) with its
effective action; Enter / → steps allow → ask → deny, ← steps back, Esc
goes back. Type to filter the tool list (filter row shows matches,
Backspace clears, Esc on empty filter goes back). The
highlight stays on the same row after every change. Changes persist to
`opencode.json` under `agents.<name>.tools` via `on_set`.

Dismisses with None (the picker stays open underneath).
"""

from __future__ import annotations

from typing import Any, Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .theme import active_theme


def _color(theme: Any, action: str) -> str:
    action = str(action or "").strip().lower()
    if action == "allow":
        return theme.c("success")
    if action == "deny":
        return theme.c("error")
    if action == "ask":
        return theme.c("warning")
    return theme.c("text_muted")


_NEXT = {"allow": "ask", "ask": "deny", "deny": "allow"}
_PREV = {"allow": "deny", "ask": "allow", "deny": "ask"}
_ORDER = ("allow", "ask", "deny")


class AgentPermissionEditor(ModalScreen[None]):
    """Edit one agent's per-tool actions. Esc goes back."""

    # Screen-level arrows: fire even when focus sits on a button inside
    # the editor (on_key alone only runs when the list has focus — with
    # focus elsewhere the keys died silently and arrows "did nothing").
    BINDINGS = [
        Binding("left", "step_prev", "Prev", show=False),
        Binding("right", "step_next", "Next", show=False),
    ]

    CSS = """
    AgentPermissionEditor {
        align: center middle;
    }
    #agent-perm-box {
        width: 64;
        max-width: 92%;
        max-height: 86%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #agent-perm-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-bottom: 1;
    }
    #agent-perm-hint {
        color: $text-muted;
        height: 1;
        margin-top: 1;
    }
    #agent-perm-filter {
        color: $text-muted;
        height: 1;
    }
    #agent-perm-list {
        height: auto;
        max-height: 20;
        background: transparent;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
    }
    """

    def __init__(
        self,
        agent: str,
        tools: list[tuple[str, str]] | None = None,
        on_set: Callable[[str, str], None] | None = None,
        readonly: bool = False,
        readonly_locked: bool = False,
        scope: set[str] | None = None,
        on_toggle_readonly: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.tools = list(tools or [])
        self.on_set = on_set
        self.readonly = bool(readonly)
        self.readonly_locked = bool(readonly_locked)
        self.scope = set(scope) if scope is not None else None
        self.on_toggle_readonly = on_toggle_readonly
        self._filter = ""

    def _visible(self) -> list[tuple[str, str]]:
        if not self._filter:
            return list(self.tools)
        q = self._filter.lower()
        return [(tool, action) for tool, action in self.tools if q in tool.lower()]

    def _options(self) -> list[Option]:
        theme = active_theme()
        opts: list[Option] = []
        ro = "ON" if self.readonly else "OFF"
        ro_color = theme.c("warning") if self.readonly else theme.c("text_muted")
        lock = " (locked)" if self.readonly_locked else ""
        opts.append(
            Option(
                f"[{theme.c('text')}]  read-only[/]"
                f"  [{ro_color}]{ro}[/][dim]{lock}[/]",
                id="__readonly__",
            )
        )
        for tool, action in self._visible():
            color = _color(theme, action)
            opts.append(
                Option(
                    f"[{theme.c('text')}]  {escape(tool)}[/]"
                    f"  [{color}]{escape(action)}[/]",
                    id=f"__tool__{tool}",
                )
            )
        return opts

    def reload_state(
        self,
        tools: list[tuple[str, str]] | None = None,
        readonly: bool | None = None,
        scope: set[str] | None = None,
    ) -> None:
        """Refresh rows (and the readonly switch) keeping highlight + filter."""
        if tools is not None:
            self.tools = list(tools)
        if readonly is not None:
            self.readonly = bool(readonly)
        if scope is not None:
            self.scope = set(scope)
        self._rebuild_keep_focus()

    def reload_tools(self, tools: list[tuple[str, str]]) -> None:
        # keep the highlight on the same row across the rebuild (the
        # clear+re-add used to drop it, so the selection "disappeared"
        # after every Enter press).
        self.reload_state(tools=tools)

    def _rebuild_keep_focus(self) -> None:
        keep: str | None = None
        try:
            lst = self.query_one("#agent-perm-list", OptionList)
            opt = lst.highlighted_option
            oid = str(getattr(opt, "id", "") or "") if opt is not None else ""
            if oid == "__readonly__" or oid.startswith("__tool__"):
                keep = oid
        except Exception:
            keep = None
        try:
            lst = self.query_one("#agent-perm-list", OptionList)
            lst.clear_options()
            lst.add_options(self._options())
            if keep is not None:
                for i in range(lst.option_count):
                    try:
                        if str(lst.get_option_at_index(i).id or "") == keep:
                            lst.highlighted = i
                            break
                    except Exception:
                        continue
            lst.focus()
        except Exception:
            pass
        self._render_filter()

    def compose(self) -> ComposeResult:
        theme = active_theme()
        with Vertical(id="agent-perm-box"):
            yield Static(
                f"  {escape(self.agent.title())} permissions  ",
                id="agent-perm-title",
            )
            yield Static("", id="agent-perm-filter")
            yield OptionList(*self._options(), id="agent-perm-list")
            yield Static(
                f"[{theme.c('text_muted')}]Enter / → next · ← prev · r read-only · type to filter · Esc back[/]",
                id="agent-perm-hint",
            )

    def _render_filter(self) -> None:
        try:
            theme = active_theme()
            w = self.query_one("#agent-perm-filter", Static)
            if self._filter:
                n = len(self._visible())
                w.update(f"[{theme.c('accent')}]filter: {escape(self._filter)}[/][dim] ({n} match{'es' if n != 1 else ''}, ⌫ clears)[/]")
            else:
                w.update(f"[{theme.c('text_muted')}]type to filter tools…[/]")
        except Exception:
            pass

    def on_mount(self) -> None:
        try:
            self.query_one("#agent-perm-list", OptionList).focus()
        except Exception:
            pass
        self._render_filter()

    def _cycle(self, tool: str, forward: bool = True) -> None:
        cur = next((a for t, a in self.tools if t == tool), "allow")
        table = _NEXT if forward else _PREV
        nxt = table.get(str(cur).lower(), "allow")
        if self.on_set is not None:
            try:
                self.on_set(tool, nxt)
            except Exception:
                pass

    def _highlighted_tool(self) -> str | None:
        try:
            lst = self.query_one("#agent-perm-list", OptionList)
            opt = lst.highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        oid = str(opt.id)
        if oid.startswith("__tool__"):
            return oid[len("__tool__"):]
        return None

    def _highlighted_id(self) -> str | None:
        try:
            lst = self.query_one("#agent-perm-list", OptionList)
            opt = lst.highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        return str(opt.id)

    def _activate_highlighted(self, forward: bool = True) -> bool:
        oid = self._highlighted_id()
        if oid == "__readonly__":
            self._flip_readonly()
            return True
        if oid is not None and oid.startswith("__tool__"):
            self._cycle(oid[len("__tool__"):], forward=forward)
            return True
        return False

    def _flip_readonly(self) -> None:
        if self.on_toggle_readonly is None:
            return
        if self.readonly_locked:
            try:
                self.app.notify("Built-in agents keep their own mode.", severity="warning")
            except Exception:
                pass
            return
        try:
            self.on_toggle_readonly()
        except Exception:
            pass

    def on_option_list_option_selected(self, event: Any) -> None:
        opt = getattr(event, "option", None)
        oid = str(getattr(opt, "id", "") or "")
        event.stop()
        if oid == "__readonly__":
            self._flip_readonly()
            return
        if not oid.startswith("__tool__"):
            return
        self._cycle(oid[len("__tool__"):], forward=True)

    def on_key(self, event: Key) -> None:
        # Physical arrows always act; letters always filter — EXCEPT `r`
        # on the read-only row flips the switch (Enter does too). This way
        # typing "grep" never flips anything by accident.
        if event.key == "escape":
            if self._filter:
                self._filter = ""
                self._rebuild_keep_focus()
                event.stop()
                return
            self.dismiss(None)
            event.stop()
        elif event.key == "left":
            if self._highlighted_id() == "__readonly__":
                event.stop()
                return
            tool = self._highlighted_tool()
            if tool is not None:
                self._cycle(tool, forward=False)
                event.stop()
        elif event.key == "right":
            if self._activate_highlighted(forward=True):
                event.stop()
        elif event.key == "backspace":
            if self._filter:
                self._filter = self._filter[:-1]
                self._rebuild_keep_focus()
                event.stop()
        elif len(event.key) == 1 and event.key.isprintable():
            if event.key in ("r", "R") and self._highlighted_id() == "__readonly__":
                self._flip_readonly()
                event.stop()
                return
            self._filter += event.key
            self._rebuild_keep_focus()
            event.stop()

    def action_step_prev(self) -> None:
        if self._highlighted_id() == "__readonly__":
            return
        tool = self._highlighted_tool()
        if tool is not None:
            self._cycle(tool, forward=False)

    def action_step_next(self) -> None:
        if self._activate_highlighted(forward=True):
            return

    def action_toggle_readonly(self) -> None:
        # only via explicit callers; key handling lives in on_key so
        # typing "r" in the filter never flips the switch by accident.
        if self._highlighted_id() == "__readonly__":
            self._flip_readonly()
