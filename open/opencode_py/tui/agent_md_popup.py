"""Agent.md manager: shared AGENT.md + per-agent instruction files + prompt preview.

The Shared group sits on TOP: one AGENT.md with the common rules every
agent receives (Find-first loop, paging ban, ...). Below it, each agent
(builtins first, then customs — the Agents popup order) owns its own
files, then a `＋ Add agent.md file` row. Enter views a file, `e`/Edit
opens the content editor; Save stores it, Delete (with confirm) removes
it. Shared persists as `<config>/agents/AGENT.md`; per-agent files live
under `<config>/agents/<agent>/` with a marker in opencode.json. Prompt
injection is shared + own (see permission.agent_combined_md).

Dead-centered, keyboard-driven. Never touches engines or config itself:
all storage goes through callbacks the app wires.
"""

from __future__ import annotations

from typing import Any, Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from .theme import active_theme

_AGENT_MD_CSS = """
AgentMdPopup {
    align: center middle;
}
#agent-md-box {
    width: 96;
    max-width: 98%;
    max-height: 88%;
    height: auto;
    background: $surface;
    border: round $accent;
    padding: 1 2;
    align: center middle;
}
#agent-md-title {
    text-style: bold;
    color: $accent;
    height: 1;
    margin-bottom: 1;
}
#agent-md-hint {
    color: $text-muted;
    height: 1;
    margin-top: 1;
}
#agent-md-list {
    height: auto;
    max-height: 16;
    background: transparent;
    border: none;
    padding: 0;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
}
#agent-md-actions {
    height: 3;
    align: center middle;
    background: $surface;
    padding: 0 1;
    margin-top: 1;
}
#agent-md-actions Button {
    height: 3;
    min-width: 10;
    padding: 0 1;
    margin: 0;
    border: heavy $accent;
    background: transparent;
    color: $text;
}
#agent-md-actions Button:hover,
#agent-md-actions Button:focus {
    background: transparent;
    border: heavy $accent;
    color: $text;
    text-style: none;
    text-opacity: 1;
}
#agent-md-pick-agents {
    height: auto;
    max-height: 12;
    background: transparent;
    border: none;
    padding: 0;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
}
#agent-md-path, #agent-md-name {
    margin-bottom: 1;
    border: none;
    background: $background;
    padding: 0 1;
}
#agent-md-editor {
    height: 9;
    border: solid $accent-muted;
    background: $background;
}
#agent-md-view-path {
    color: $text-muted;
    height: 1;
    margin-bottom: 1;
}
#agent-md-view-body {
    height: auto;
    max-height: 18;
    background: $background;
    border: solid $accent-muted;
    padding: 0 1;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
}
"""


def md_entries_for_agent(spec: dict[str, Any] | None) -> list[tuple[str, str]]:
    """(filename, content) pairs stored for one agent. Never raises."""
    try:
        raw = (spec or {}).get("md")
        if isinstance(raw, dict):
            out = [(str(k), str(v)) for k, v in raw.items() if str(k).strip()]
            return sorted(out, key=lambda kv: kv[0].lower())
        if isinstance(raw, list):
            out = []
            for item in raw:
                if isinstance(item, dict) and item.get("name"):
                    out.append((str(item["name"]), str(item.get("content", ""))))
                elif isinstance(item, str) and item.strip():
                    out.append((item.strip(), ""))
            return sorted(out, key=lambda kv: kv[0].lower())
    except Exception:
        pass
    return []


class AgentPickPopup(ModalScreen[str | None]):
    """Step 1 of ＋ Add: pick which agent owns the new .md file.

    Rows follow the Agents popup order (builtins, then customs). Enter
    picks, Esc cancels. Dismisses with the agent name or None.
    """

    CSS = _AGENT_MD_CSS.replace("AgentMdPopup {", "AgentPickPopup {", 1)

    def __init__(self, agents: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agents = list(agents or [])

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-md-box"):
            yield Static("  Add agent.md — pick agent  ", id="agent-md-title")
            yield OptionList(
                *[Option(escape(n.title()), id=f"__pick__{n}") for n in self._agents],
                id="agent-md-pick-agents",
            )
            yield Static("↑/↓ move · Enter pick · Esc cancel", id="agent-md-hint")

    def on_mount(self) -> None:
        try:
            self.query_one("#agent-md-pick-agents", OptionList).focus()
        except Exception:
            pass

    def on_option_list_option_selected(self, event: Any) -> None:
        opt = getattr(event, "option", None)
        oid = str(getattr(opt, "id", "") or "")
        event.stop()
        if oid.startswith("__pick__"):
            self.dismiss(oid[len("__pick__"):])
            return
        self.dismiss(None)

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class AgentMdEditor(ModalScreen[tuple[str, str] | None]):
    """Step 2 of ＋ Add (and file edit): paste content or give a path.

    Shows the file name (editable for new files), a path row (＋ loads a
    file from disk into the editor), and the content area. Save stores,
    Cancel/Esc drops. Dismisses with (name, content) or None.
    """

    CSS = _AGENT_MD_CSS.replace("AgentMdPopup {", "AgentMdEditor {", 1)

    def __init__(
        self,
        agent: str,
        name: str = "",
        content: str = "",
        on_load_path: Callable[[str], str | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._agent = agent
        self._name = name
        self._content = content
        self._existing = bool(name)
        self.on_load_path = on_load_path

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-md-box"):
            yield Static(f"  {escape(self._agent.title())} agent.md  ", id="agent-md-title")
            yield Input(
                value=self._name,
                placeholder="File name (e.g. guide.md)",
                id="agent-md-name",
                disabled=self._existing,
            )
            if not self._existing:
                yield Input(
                    placeholder="＋ load from path (optional, Enter loads)",
                    id="agent-md-path",
                )
            yield TextArea(self._content, id="agent-md-editor")
            with Horizontal(id="agent-md-actions"):
                yield Button("Save", id="agent-md-save", variant="default")
                yield Button("Cancel", id="agent-md-cancel", variant="default")

    def on_mount(self) -> None:
        try:
            if self._existing:
                self.query_one("#agent-md-editor", TextArea).focus()
            else:
                self.query_one("#agent-md-name", Input).focus()
        except Exception:
            pass

    def _collect(self) -> tuple[str, str]:
        try:
            name = self.query_one("#agent-md-name", Input).value.strip()
        except Exception:
            name = ""
        try:
            content = self.query_one("#agent-md-editor", TextArea).text
        except Exception:
            content = ""
        # The path box is NOT auto-loaded: if the user typed a path but
        # never pressed Enter there, the editor is still empty and Save
        # would store an empty file. Load it now so Save does the
        # expected thing (warns when the path can't be read).
        if not (content or "").strip():
            try:
                path = self.query_one("#agent-md-path", Input).value.strip()
            except Exception:
                path = ""
            if path:
                self._load_from_path()
                try:
                    content = self.query_one("#agent-md-editor", TextArea).text
                except Exception:
                    pass
        return name, content

    def _load_from_path(self) -> None:
        try:
            path = self.query_one("#agent-md-path", Input).value.strip()
        except Exception:
            path = ""
        if not path or self.on_load_path is None:
            return
        try:
            content = self.on_load_path(path)
        except Exception:
            content = None
        if content is None:
            try:
                self.app.notify(f"Can't read '{path}'.", severity="warning")
            except Exception:
                pass
            return
        try:
            self.query_one("#agent-md-editor", TextArea).text = content
            self.query_one("#agent-md-editor", TextArea).focus()
        except Exception:
            pass
        # default the file name from the path when empty
        try:
            name_inp = self.query_one("#agent-md-name", Input)
            if not name_inp.value.strip():
                name_inp.value = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-md-save":
            name, content = self._collect()
            if not name:
                try:
                    self.app.notify("File name can't be empty.", severity="warning")
                except Exception:
                    pass
                return
            if not (content or "").strip():
                # still empty after the path auto-load attempt: saving
                # would mint an empty file — stop and say why instead.
                try:
                    self.app.notify("Nothing to save — paste content or load a readable path.", severity="warning")
                except Exception:
                    pass
                return
            self.dismiss((name, content))
        else:
            self.dismiss(None)
        event.stop()

    def on_input_submitted(self, event: Any) -> None:
        try:
            iid = getattr(event.input, "id", "")
        except Exception:
            return
        event.stop()
        if iid == "agent-md-path":
            self._load_from_path()
        elif iid in ("agent-md-name",):
            try:
                self.query_one("#agent-md-editor", TextArea).focus()
            except Exception:
                self._collect_and_save()

    def _collect_and_save(self) -> None:
        name, content = self._collect()
        if not name:
            try:
                self.app.notify("File name can't be empty.", severity="warning")
            except Exception:
                pass
            return
        if not (content or "").strip():
            try:
                self.app.notify("Nothing to save — paste content or load a readable path.", severity="warning")
            except Exception:
                pass
            return
        self.dismiss((name, content))

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class AgentMdViewer(ModalScreen[str | None]):
    """Read-only viewer (cat): the real file path on top, then the
    content in a scrollable block. Edit (button or `e`) dismisses with
    "edit" so the app can open the content editor; Esc/Close dismisses
    with None. Dead-centered."""

    CSS = _AGENT_MD_CSS.replace("AgentMdPopup {", "AgentMdViewer {", 1)

    def __init__(self, agent: str, name: str, path: str, content: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent = agent
        self._name = name
        self._path = path
        self._content = content

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll

        try:
            from ..permission import short_display_path as _short

            shown = _short(self._path)
        except Exception:
            shown = self._path
        with Vertical(id="agent-md-box"):
            yield Static(f"  {escape(self._name)}  ", id="agent-md-title")
            yield Static(escape(shown), id="agent-md-view-path")
            with VerticalScroll(id="agent-md-view-body"):
                yield Static(escape(self._content) if self._content.strip() else "(empty file)")
            with Horizontal(id="agent-md-actions"):
                yield Button("Edit", id="agent-md-view-edit", variant="default")
                yield Button("Close", id="agent-md-view-close", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#agent-md-view-edit", Button).focus()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-md-view-close":
            self.dismiss(None)
        elif event.button.id == "agent-md-view-edit":
            self.dismiss("edit")
        event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
        elif event.key in ("e", "E"):
            self.dismiss("edit")
            event.stop()


class AgentPromptPreview(ModalScreen[None]):
    """Byte-exact preview of the FULL system prompt one agent will send.

    Built by the SAME labeled_prompt_parts() the sender uses, so not one
    letter is missed. Blocks render in send order with their source label,
    then chars + ~tokens at the top. c copies the focused block (or all),
    Esc/Close goes back. Dead-centered, read-only."""

    CSS = _AGENT_MD_CSS.replace("AgentMdPopup {", "AgentPromptPreview {", 1)

    BINDINGS = [
        Binding("c", "copy_block", "Copy", show=False),
    ]

    def __init__(self, agent: str, blocks: list[tuple[str, str]] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent = agent
        self._blocks = list(blocks or [])

    def _counts(self) -> tuple[int, int]:
        total = sum(len(text) for _label, text in self._blocks)
        return total, max(1, total // 4)

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll

        total, toks = self._counts()
        with Vertical(id="agent-md-box"):
            yield Static(f"  {escape(self._agent.title())} prompt preview  ", id="agent-md-title")
            yield Static(
                f"{len(self._blocks)} blocks · {total:,} chars · ~{toks:,} tokens",
                id="agent-md-view-path",
            )
            with VerticalScroll(id="agent-md-view-body"):
                for i, (label, text) in enumerate(self._blocks):
                    yield Static(f"━━━ [{i + 1}/{len(self._blocks)}] {escape(label)} ━━━")
                    yield Static(escape(text) if text.strip() else "(empty block)")
            with Horizontal(id="agent-md-actions"):
                yield Button("Copy", id="agent-md-preview-copy", variant="default")
                yield Button("Close", id="agent-md-preview-close", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#agent-md-preview-close", Button).focus()
        except Exception:
            pass

    def _copy_text(self) -> str:
        try:
            from textual.widgets import OptionList as _OL  # noqa
        except Exception:
            pass
        return "\n\n".join(text for _label, text in self._blocks)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-md-preview-close":
            self.dismiss(None)
        elif event.button.id == "agent-md-preview-copy":
            try:
                self.app.copy_to_clipboard(self._copy_text())
                self.app.notify("Prompt copied to clipboard.")
            except Exception:
                try:
                    self.app.notify("Copy not available.", severity="warning")
                except Exception:
                    pass
        event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
        elif event.key in ("c", "C"):
            try:
                self.app.copy_to_clipboard(self._copy_text())
                self.app.notify("Prompt copied to clipboard.")
            except Exception:
                try:
                    self.app.notify("Copy not available.", severity="warning")
                except Exception:
                    pass
            event.stop()

    def action_copy_block(self) -> None:
        try:
            self.app.copy_to_clipboard(self._copy_text())
            self.app.notify("Prompt copied to clipboard.")
        except Exception:
            pass


class AgentMdPopup(ModalScreen[None]):
    """The manager: .md files grouped by agent (Agents-popup order),
    then `＋ Add agent.md file`. Enter views a file, `e`/Edit opens the
    content editor, Delete/Ctrl+D removes it (with confirm), Ctrl+N
    renames. Esc closes."""

    CSS = _AGENT_MD_CSS
    BINDINGS = [
        Binding("e", "edit_file", "Edit", show=False),
        Binding("p", "preview_prompt", "Preview", show=False),
        Binding("ctrl+n", "rename_file", "Rename", show=False),
        Binding("ctrl+d", "delete_file", "Delete", show=False),
    ]

    def __init__(
        self,
        groups: list[tuple[str, list[tuple[str, str]]]] | None = None,
        on_open: Callable[[str, str], None] | None = None,
        on_edit: Callable[[str, str], None] | None = None,
        on_add: Callable[[], None] | None = None,
        on_delete: Callable[[str, str], bool] | None = None,
        on_save_all: Callable[[], None] | None = None,
        on_rename: Callable[[str, str], None] | None = None,
        on_preview: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._groups = list(groups or [])
        self.on_open = on_open
        self.on_edit = on_edit
        self.on_add = on_add
        self.on_delete = on_delete
        self.on_save_all = on_save_all
        self.on_rename = on_rename
        self.on_preview = on_preview

    def _options(self) -> list[Option]:
        theme = active_theme()
        opts: list[Option] = []
        hdr = 0
        for agent, files in self._groups:
            try:
                from ..permission import SHARED_AGENT as _SA
            except Exception:
                _SA = "shared"
            if agent == _SA:
                title = "Shared — sent with EVERY agent"
            else:
                title = escape(agent.title())
            opts.append(
                Option(f"[bold {theme.c('accent')}]{title}[/]", id=f"__hdr__{hdr}", disabled=True)
            )
            hdr += 1
            for fname, _content in files:
                opts.append(
                    Option(
                        f"[{theme.c('text')}]    {escape(fname)}[/]",
                        id=f"__md__{agent}\x00{fname}",
                    )
                )
            if not files:
                opts.append(
                    Option(f"[dim]    (no .md files)[/]", id=f"__hdr__{hdr}", disabled=True)
                )
                hdr += 1
        opts.append(Option("＋ Add agent.md file", id="__add__"))
        return opts

    def reload(self, groups: list[tuple[str, list[tuple[str, str]]]]) -> None:
        """Refresh rows keeping the popup open. Keeps the highlight."""
        keep: str | None = None
        try:
            lst = self.query_one("#agent-md-list", OptionList)
            opt = lst.highlighted_option
            oid = str(getattr(opt, "id", "") or "") if opt is not None else ""
            if oid.startswith("__md__") or oid == "__add__":
                keep = oid
        except Exception:
            keep = None
        self._groups = list(groups)
        try:
            lst = self.query_one("#agent-md-list", OptionList)
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

    def _highlighted_file(self) -> tuple[str, str] | None:
        try:
            lst = self.query_one("#agent-md-list", OptionList)
            opt = lst.highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        oid = str(opt.id)
        if oid.startswith("__md__"):
            rest = oid[len("__md__"):]
            agent, _, fname = rest.partition("\x00")
            if agent and fname:
                return agent, fname
        return None

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-md-box"):
            yield Static("  Agent.md files  ", id="agent-md-title")
            yield OptionList(*self._options(), id="agent-md-list")
            yield Static(
                "↑/↓ move · Enter view · e edit · p preview prompt · Ctrl+N rename · Ctrl+D delete · Esc close",
                id="agent-md-hint",
            )
            with Horizontal(id="agent-md-actions"):
                yield Button("Save", id="agent-md-saveall", variant="default")
                yield Button("Preview", id="agent-md-preview", variant="default")
                yield Button("Edit", id="agent-md-edit", variant="default")
                yield Button("Delete", id="agent-md-del", variant="default")
                yield Button("Rename", id="agent-md-rename", variant="default")
                yield Button("Close", id="agent-md-close", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#agent-md-list", OptionList).focus()
        except Exception:
            pass

    def _open_highlighted(self) -> None:
        cur = self._highlighted_file()
        if cur is not None and self.on_open is not None:
            try:
                self.on_open(cur[0], cur[1])
            except Exception:
                pass

    def _edit_highlighted(self) -> None:
        cur = self._highlighted_file()
        if cur is not None and self.on_edit is not None:
            try:
                self.on_edit(cur[0], cur[1])
            except Exception:
                pass

    def _delete_highlighted(self) -> None:
        # bug 5 fix: destructive delete asks first. The confirm lives here
        # (popup layer) so the app callback stays synchronous.
        cur = self._highlighted_file()
        if cur is None or self.on_delete is None:
            return
        try:
            from .confirm_dialog import ConfirmDialog
        except Exception:
            try:
                self.on_delete(cur[0], cur[1])
            except Exception:
                pass
            return

        def _after(ok: bool | None) -> None:
            try:
                self.query_one("#agent-md-list", OptionList).focus()
            except Exception:
                pass
            if not ok:
                return
            try:
                self.on_delete(cur[0], cur[1])
            except Exception:
                pass

        try:
            self.app.push_screen(
                ConfirmDialog(
                    message=f"Delete '{cur[1]}' from {cur[0]}? This removes the file from disk.",
                    title="Delete file",
                ),
                _after,
            )
        except Exception:
            pass

    def _rename_highlighted(self) -> None:
        cur = self._highlighted_file()
        if cur is not None and self.on_rename is not None:
            try:
                self.on_rename(cur[0], cur[1])
            except Exception:
                pass

    def _preview_highlighted(self) -> None:
        if self.on_preview is None:
            return
        # file row -> its agent; anywhere else -> first agent group
        agent: str | None = None
        cur = self._highlighted_file()
        if cur is not None:
            agent = cur[0]
        else:
            try:
                if self._groups:
                    agent = self._groups[0][0]
            except Exception:
                agent = None
        if not agent:
            return
        try:
            self.on_preview(agent)
        except Exception:
            pass

    def on_option_list_option_selected(self, event: Any) -> None:
        opt = getattr(event, "option", None)
        oid = str(getattr(opt, "id", "") or "")
        event.stop()
        if oid == "__add__":
            if self.on_add is not None:
                try:
                    self.on_add()
                except Exception:
                    pass
            return
        if oid.startswith("__md__"):
            self._open_highlighted()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "agent-md-close":
            self.dismiss(None)
        elif bid == "agent-md-saveall":
            if self.on_save_all is not None:
                try:
                    self.on_save_all()
                except Exception:
                    pass
            try:
                self.query_one("#agent-md-list", OptionList).focus()
            except Exception:
                pass
        elif bid == "agent-md-preview":
            self._preview_highlighted()
        elif bid == "agent-md-del":
            self._delete_highlighted()
        elif bid == "agent-md-edit":
            self._edit_highlighted()
        elif bid == "agent-md-rename":
            self._rename_highlighted()
        event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
        elif event.key in ("p", "P"):
            self._preview_highlighted()
            event.stop()

    def action_preview_prompt(self) -> None:
        self._preview_highlighted()

    def action_edit_file(self) -> None:
        self._edit_highlighted()

    def action_rename_file(self) -> None:
        self._rename_highlighted()

    def action_delete_file(self) -> None:
        self._delete_highlighted()
