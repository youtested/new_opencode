"""MCP manager popup (bare /mcp): list, add, remove, test with arrows.

Thin screen: rows dismiss with an action string the app interprets, so this
screen never touches engines or config itself. Adding collects a run line in
two Inputs (name + command...); the app parses/saves via mcp_manager.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

_MCP_PICKER_CSS = """
McpPicker {
    align: center middle;
}
#mcp-picker-box {
    width: 72;
    max-height: 85%;
    height: auto;
    background: $surface;
    border: round #666;
    padding: 0 1;
}
#mcp-picker-title {
    text-style: bold;
    margin-bottom: 1;
}
#mcp-picker-list {
    height: auto;
    max-height: 14;
    background: transparent;
}
#mcp-picker-name, #mcp-picker-cmd {
    margin-top: 1;
}
#mcp-picker-hint {
    margin-top: 1;
}
"""


class McpPicker(ModalScreen[str | None]):
    """List servers + actions; Enter picks, Esc cancels."""

    CSS = _MCP_PICKER_CSS
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, servers: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.servers = dict(servers or {})

    def _rows(self) -> list[Option]:
        opts: list[Option] = [Option("[b]MCP servers[/]", disabled=True)]
        if self.servers:
            for name, spec in self.servers.items():
                if isinstance(spec, dict) and spec.get("command"):
                    cmdline = str(spec.get("command")) + "".join(
                        f" {a}" for a in (spec.get("args") or [])
                    )
                else:
                    cmdline = "(bad entry)"
                opts.append(Option(f"● {name}  [dim]{cmdline}[/]", id=f"__srv__{name}"))
        else:
            opts.append(Option("[dim](none configured)[/]", disabled=True))
        opts.append(Option("[b]Actions[/]", disabled=True))
        opts.append(Option("+ Add a server  [dim](fill name + command below, Enter to save)[/]", id="__add__"))
        for name in self.servers:
            opts.append(Option(f"⌧ Remove {name}", id=f"__remove__{name}"))
        opts.append(Option("⚙ Test all  [dim](up to ~15s each)[/]", id="__test__"))
        return opts

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-picker-box"):
            yield Static(
                "  MCP — arrows to move · Enter to apply · Esc to close",
                id="mcp-picker-title",
            )
            yield OptionList(*self._rows(), id="mcp-picker-list")
            yield Input(placeholder="name, e.g. files", id="mcp-picker-name")
            yield Input(
                placeholder="run line, e.g. python -m my_server  (or npx -y ...)",
                id="mcp-picker-cmd",
            )
            yield Static(
                "[dim]python -m ... = light, best for phones · "
                "npx ... = needs Node (heavy on 32-bit).[/]",
                id="mcp-picker-hint",
            )

    def on_mount(self) -> None:
        try:
            lst = self.query_one("#mcp-picker-list", OptionList)
            lst.focus()
            # Land on the first selectable row (with zero servers index 1 is
            # the disabled Actions header).
            try:
                for i in range(lst.option_count):
                    if not lst.get_option_at_index(i).disabled:
                        lst.highlighted = i
                        break
                else:
                    lst.highlighted = 1
            except Exception:
                lst.highlighted = 1
            lst.scroll_to_highlight()
        except Exception:
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = getattr(event.option, "id", None)
        if option_id == "__add__":
            try:
                name = self.query_one("#mcp-picker-name", Input).value.strip()
                cmdline = self.query_one("#mcp-picker-cmd", Input).value.strip()
            except Exception:
                name, cmdline = "", ""
            if not name or not cmdline:
                try:
                    self.query_one("#mcp-picker-name", Input).focus()
                except Exception:
                    pass
                self.notify("Fill in a name and a run line first.")
                return
            # \x1f separator: names may legally contain "__".
            self.dismiss(f"__add__\x1f{name}\x1f{cmdline}")
            return
        self.dismiss(option_id)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in ("mcp-picker-name", "mcp-picker-cmd"):
            try:
                name = self.query_one("#mcp-picker-name", Input).value.strip()
                cmdline = self.query_one("#mcp-picker-cmd", Input).value.strip()
            except Exception:
                name, cmdline = "", ""
            if name and cmdline:
                self.dismiss(f"__add__\x1f{name}\x1f{cmdline}")
            else:
                self.notify("Fill in a name and a run line first.")

    def action_cancel(self) -> None:
        self.dismiss(None)
