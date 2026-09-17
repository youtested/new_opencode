"""Agent picker popup: list agents, add/delete/rename customs, edit permissions.

Dead-centered, keyboard-driven (arrows + Enter, Esc closes):

- One row per agent: `● ▣ Build  full tools…` (current marked), builtins
  first then customs, then a `＋ Add agent` row.
- Enter on an agent row switches to it (dismisses with the name).
- Enter on `＋ Add agent` opens a name dialog (name + description +
  Save/Cancel buttons); rename reuses the same dialog.
- `Ctrl+D` deletes the highlighted CUSTOM agent (builtins protected).
- `Ctrl+N` renames the highlighted custom agent.
- Pressing `p` on a highlighted row opens the per-agent permission
  editor: every live tool with its current action (allow/ask/deny,
  exactly like /permissions shows); Enter cycles the action, Esc goes
  back. Changes save to `opencode.json` under `agents`.

Never touches engines or config files itself: all mutations go through
callbacks the app wires (`on_switch`, `on_add`, `on_delete`, `on_rename`,
`on_edit_description`). The permission editor (opened via `p`) owns the
read-only switch. Dismisses with the picked
agent name (or None).
"""

from __future__ import annotations

from typing import Any, Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from .theme import active_theme

_AGENT_PICKER_CSS = """
AgentPicker {
    align: center middle;
}
#agent-picker-box {
    width: 72;
    max-width: 94%;
    max-height: 88%;
    height: auto;
    background: $surface;
    border: round $accent;
    padding: 1 2;
}
#agent-picker-title {
    text-style: bold;
    color: $accent;
    height: 1;
    margin-bottom: 1;
}
#agent-picker-hint {
    color: $text-muted;
    height: 1;
    margin-top: 1;
}
#agent-picker-list {
    height: auto;
    max-height: 20;
    background: transparent;
    /* no container border: the popup box already frames it, and the
       list's own tall border draws half-block spikes on the sides. */
    border: none;
    padding: 0;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
}
#agent-picker-actions {
    height: 3;
    align: center middle;
    background: $surface;
    padding: 0 1;
    margin-top: 1;
}
#agent-picker-actions Button {
    height: 3;
    min-width: 12;
    padding: 0 2;
    margin: 0 1;
    border: heavy $accent;
    background: transparent;
    color: $text;
}
#agent-picker-actions Button:hover,
#agent-picker-actions Button:focus {
    background: transparent;
    border: heavy $accent;
    color: $text;
    text-style: none;
    text-opacity: 1;
}
#agent-name-input {
    margin: 1 0;
}
"""


def _badge(theme: Any, agent: str) -> str:
    name = str(agent or "").strip() or "?"
    return f"[{theme.agent_color(name)}]▣[/]"


class AgentNameDialog(ModalScreen[tuple[str, str] | None]):
    """Name + description dialog for add/rename. Dismisses with
    (name, description), or None on Esc/Cancel."""

    CSS = """
    AgentNameDialog {
        align: center middle;
    }
    #agent-name-box {
        width: 56;
        max-width: 92%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #agent-name-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-bottom: 1;
    }
    #agent-name-input, #agent-desc-input {
        margin-bottom: 1;
        /* no border: the input's own tall border draws half-block
           spikes on the sides inside the already-bordered dialog. */
        border: none;
        background: $background;
        padding: 0 1;
    }
    #agent-name-actions {
        height: 3;
        align: center middle;
        background: $surface;
    }
    #agent-name-actions Button {
        height: 3;
        min-width: 12;
        padding: 0 2;
        margin: 0 1;
        border: heavy $accent;
        background: transparent;
        color: $text;
    }
    #agent-name-actions Button:hover,
    #agent-name-actions Button:focus {
        background: transparent;
        border: heavy $accent;
        color: $text;
        text-style: none;
        text-opacity: 1;
    }
    """

    def __init__(
        self,
        old: str | None = None,
        title: str | None = None,
        hide_description: bool | None = None,
        name_placeholder: str = "Name (e.g. reviewer)",
        description: str = "",
        description_placeholder: str = "Short description (optional)",
        lock_name: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._old = old
        self._title = title
        # bug 2 fix: file renames reuse this dialog but must not say
        # "Rename agent". Callers pass title="Rename file".
        self._hide_desc = (old is not None) if hide_description is None else bool(hide_description)
        self._name_placeholder = name_placeholder
        self._desc_value = description or ""
        self._desc_placeholder = description_placeholder
        self._lock_name = bool(lock_name)

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal

        title = self._title or ("Rename agent" if self._old else "Add agent")
        with Vertical(id="agent-name-box"):
            yield Static(f"  {title}  ", id="agent-name-title")
            yield Input(
                value=self._old or "",
                placeholder=self._name_placeholder,
                id="agent-name-input",
                disabled=self._lock_name,
            )
            if not self._hide_desc:
                yield Input(
                    value=self._desc_value,
                    placeholder=self._desc_placeholder,
                    id="agent-desc-input",
                )
            with Horizontal(id="agent-name-actions"):
                yield Button("Save", id="agent-name-save", variant="default")
                yield Button("Cancel", id="agent-name-cancel", variant="default")

    def on_mount(self) -> None:
        try:
            if self._lock_name:
                inp = self.query_one("#agent-desc-input", Input)
            else:
                inp = self.query_one("#agent-name-input", Input)
            inp.focus()
            inp.cursor_position = len(inp.value)
        except Exception:
            pass

    def _done(self) -> None:
        try:
            name = self.query_one("#agent-name-input", Input).value.strip()
        except Exception:
            name = ""
        desc = ""
        if not self._hide_desc:
            try:
                desc = self.query_one("#agent-desc-input", Input).value.strip()
            except Exception:
                desc = ""
        if not name:
            self.app.notify("Agent name can't be empty.", severity="warning")
            return
        self.dismiss((name, desc))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-name-save":
            self._done()
        else:
            self.dismiss(None)
        event.stop()

    def on_input_submitted(self, event: Any) -> None:
        try:
            if getattr(event.input, "id", "") not in ("agent-name-input", "agent-desc-input"):
                return
        except Exception:
            return
        event.stop()
        self._done()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class AgentPicker(ModalScreen[str | None]):
    """Pick / manage agents. Dismisses with the agent name or None."""

    CSS = _AGENT_PICKER_CSS
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+d", "delete", "Delete", show=False),
        Binding("ctrl+n", "rename", "Rename", show=False),
    ]

    def __init__(
        self,
        agents: list[tuple[str, str, bool]] | None = None,
        current: str = "build",
        on_switch: Callable[[str], None] | None = None,
        on_add: Callable[[str, str], str | None] | None = None,
        on_delete: Callable[[str], bool] | None = None,
        on_rename: Callable[[str, str], str | None] | None = None,
        on_permissions: Callable[[str], None] | None = None,
        on_save: Callable[[], None] | None = None,
        on_agent_md: Callable[[], None] | None = None,
        on_edit_description: Callable[[str, str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._agents = list(agents or [])
        self._current = current
        self.on_switch = on_switch
        self.on_add = on_add
        self.on_delete = on_delete
        self.on_rename = on_rename
        self.on_permissions = on_permissions
        self.on_save = on_save
        self.on_agent_md = on_agent_md
        self.on_edit_description = on_edit_description

    # -- rows ------------------------------------------------------------
    def _options(self) -> list[Option]:
        theme = active_theme()
        opts: list[Option] = []
        for name, desc, _custom in self._agents:
            mark = "●" if name == self._current else " "
            opts.append(
                Option(
                    f"[{mark}] {_badge(theme, name)} {escape(name.title())}"
                    f"  [dim]{escape(desc)}[/]",
                    id=f"__agent__{name}",
                )
            )
        opts.append(Option("＋ Add agent", id="__add__"))
        return opts

    def reload_agents(self, agents: list[tuple[str, str, bool]], current: str = "") -> None:
        """Reload rows (after add/delete/rename) keeping the popup open."""
        self._agents = list(agents)
        if current:
            self._current = current
        try:
            lst = self.query_one("#agent-picker-list", OptionList)
            lst.clear_options()
            lst.add_options(self._options())
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-picker-box"):
            yield Static("  Agents — Enter switch · Esc close  ", id="agent-picker-title")
            yield OptionList(*self._options(), id="agent-picker-list")
            yield Static(
                "↑/↓ move · Enter switch · ＋ add · Ctrl+N rename · Ctrl+D delete · d edit desc · p permissions",
                id="agent-picker-hint",
            )
            with Horizontal(id="agent-picker-actions"):
                yield Button("Permissions", id="agent-perms", variant="default")
                yield Button("Edit", id="agent-edit", variant="default")
                yield Button("Save", id="agent-save", variant="default")
                yield Button("Agent.md", id="agent-md", variant="default")
                yield Button("Close", id="agent-close", variant="default")

    def on_mount(self) -> None:
        try:
            lst = self.query_one("#agent-picker-list", OptionList)
            for i in range(lst.option_count):
                try:
                    if lst.get_option_at_index(i).id == f"__agent__{self._current}":
                        lst.highlighted = i
                        break
                except Exception:
                    continue
            lst.focus()
        except Exception:
            pass

    # -- add/rename dialog -----------------------------------------------
    def _open_edit_dialog(self, name: str) -> None:
        """Push the description editor for an existing CUSTOM agent.

        Name locked, description prefilled; Save goes through on_rename
        with an unchanged name (the app updates just the description)."""
        if self.on_edit_description is None:
            return
        desc = next((d for n, d, _c in self._agents if n == name), "")
        try:
            self.app.push_screen(
                AgentNameDialog(
                    old=name,
                    title="Edit description",
                    hide_description=False,
                    name_placeholder="Name",
                    description=desc,
                    description_placeholder="Short description",
                    lock_name=True,
                ),
                lambda result: self._after_edit_dialog(name, result),
            )
        except Exception:
            pass

    def _after_edit_dialog(self, name: str, result: tuple[str, str] | None) -> None:
        """Edit dialog closed: description-only save via on_rename(name, name)."""
        try:
            self.query_one("#agent-picker-list", OptionList).focus()
        except Exception:
            pass
        if not result or self.on_edit_description is None:
            return
        try:
            self.on_edit_description(name, (result[1] or "").strip())
        except Exception:
            pass

    def _open_name_dialog(self, old: str | None = None) -> None:
        """Push the name+description dialog (add when old is None)."""
        try:
            self.app.push_screen(
                AgentNameDialog(old=old),
                lambda result: self._after_name_dialog(old, result),
            )
        except Exception:
            pass

    def _after_name_dialog(
        self, old: str | None, result: tuple[str, str] | None
    ) -> None:
        """Name dialog closed: add (old=None) or rename."""
        try:
            self.query_one("#agent-picker-list", OptionList).focus()
        except Exception:
            pass
        if not result:
            return
        name, desc = result
        name = (name or "").strip().lower().replace(" ", "-")
        if old is None:
            if not name:
                return
            if any(n == name for n, _d, _c in self._agents):
                self.app.notify(f"Agent '{name}' already exists.", severity="warning")
                return
            if self.on_add is not None:
                try:
                    self.on_add(name, (desc or "").strip())
                except Exception:
                    pass
            return
        if name and name != old and self.on_rename is not None:
            try:
                self.on_rename(old, name)
            except Exception:
                pass
            return
        # NOTE: same-name save here does nothing — the rename dialog hides
        # the description field, so desc="" means "not shown", not "empty".
        # Description edits go through _after_edit_dialog instead.

    def _highlighted_agent(self) -> tuple[str, bool] | None:
        """(name, is_custom) under the highlight, or None for ＋ Add."""
        try:
            lst = self.query_one("#agent-picker-list", OptionList)
            opt = lst.highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        oid = str(opt.id)
        if oid == "__add__":
            return None
        if oid.startswith("__agent__"):
            name = oid[len("__agent__"):]
            custom = any(n == name and c for n, _d, c in self._agents)
            return name, custom
        return None

    def _submit_name(self, value: str) -> None:
        # legacy inline path (unused now that add/rename use AgentNameDialog)
        self._refocus_list()

    def _refocus_list(self) -> None:
        try:
            self.query_one("#agent-picker-list", OptionList).focus()
        except Exception:
            pass

    def on_input_submitted(self, event: Any) -> None:
        return

    # -- selection --------------------------------------------------------
    def on_option_list_option_selected(self, event: Any) -> None:
        opt = getattr(event, "option", None)
        oid = str(getattr(opt, "id", "") or "")
        event.stop()
        if oid == "__add__":
            self._open_name_dialog(None)
            return
        if oid.startswith("__agent__"):
            name = oid[len("__agent__"):]
            if self.on_switch is not None:
                try:
                    self.on_switch(name)
                except Exception:
                    pass
            self.dismiss(name)
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "agent-close":
            self.dismiss(None)
        elif bid == "agent-save":
            if self.on_save is not None:
                try:
                    self.on_save()
                except Exception:
                    pass
            # Save leaves focus on the button, freezing ↑/↓ on the list —
            # hand focus straight back (same fix as the editor-Esc return).
            self._refocus_list()
        elif bid == "agent-perms":
            cur = self._highlighted_agent()
            if cur is not None and self.on_permissions is not None:
                try:
                    self.on_permissions(cur[0])
                except Exception:
                    pass
        elif bid == "agent-edit":
            cur = self._highlighted_agent()
            if cur is not None:
                name, custom = cur
                if not custom:
                    self.app.notify("Built-in descriptions can't be edited.", severity="warning")
                else:
                    self._open_edit_dialog(name)
        elif bid == "agent-md":
            if self.on_agent_md is not None:
                try:
                    self.on_agent_md()
                except Exception:
                    pass
        event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
        elif event.key in ("p", "P"):
            cur = self._highlighted_agent()
            if cur is not None and self.on_permissions is not None:
                try:
                    self.on_permissions(cur[0])
                except Exception:
                    pass
                event.stop()
        elif event.key in ("d", "D"):
            cur = self._highlighted_agent()
            if cur is not None:
                name, custom = cur
                if not custom:
                    self.app.notify("Built-in descriptions can't be edited.", severity="warning")
                else:
                    self._open_edit_dialog(name)
                event.stop()

    def action_delete(self) -> None:
        cur = self._highlighted_agent()
        if cur is None:
            return
        name, custom = cur
        if not custom:
            self.app.notify("Built-in agents can't be deleted.", severity="warning")
            return
        if self.on_delete is None:
            return
        # bug 5 fix: agents own permission rules + .md files — confirm
        # here (popup layer) so the app callback stays synchronous.
        try:
            from .confirm_dialog import ConfirmDialog
        except Exception:
            try:
                self.on_delete(name)
            except Exception:
                pass
            return

        def _after(ok: bool | None) -> None:
            try:
                self.query_one("#agent-picker-list", OptionList).focus()
            except Exception:
                pass
            if not ok:
                return
            try:
                self.on_delete(name)
            except Exception:
                pass

        try:
            self.app.push_screen(
                ConfirmDialog(
                    message=f"Delete agent '{name}' and its rules/files?",
                    title="Delete agent",
                ),
                _after,
            )
        except Exception:
            pass

    def action_rename(self) -> None:
        cur = self._highlighted_agent()
        if cur is None:
            return
        name, custom = cur
        if not custom:
            self.app.notify("Built-in agents can't be renamed.", severity="warning")
            return
        self._open_name_dialog(name)
