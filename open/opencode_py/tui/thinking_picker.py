"""Thinking picker popup (bare /thinking): current model + thought switch + effort.

Layout (arrow-navigable, one list):
  - header line: current `provider/model`
  - `Thinking  on/off` rows — whether thought bubbles render at all
  - one row per effort level the CURRENT model advertises in the live catalog
    (e.g. minimal/low/medium/high/xhigh for muse-spark; nothing extra for
    fixed thinkers) — future models appear automatically
  - `Show thoughts` / `Hide thoughts` convenience rows

Enter applies the highlighted row immediately (effort persists to config and
takes effect on the next turn); Esc cancels. Dismisses with an action string
the app interprets, so this screen never touches engines or config itself.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

_THINKING_PICKER_CSS = """
ThinkingPicker {
    align: center middle;
}
#thinking-picker-box {
    width: 60;
    max-height: 80%;
    height: auto;
    background: $surface;
    border: round #666;
    padding: 0 1;
}
#thinking-picker-title {
    text-style: bold;
    margin-bottom: 1;
}
#thinking-picker-list {
    height: auto;
    max-height: 22;
    background: transparent;
}
"""


class ThinkingPicker(ModalScreen[str]):
    """Pick thinking display + reasoning effort with the arrow keys."""

    CSS = _THINKING_PICKER_CSS
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(
        self,
        model: str = "",
        thinking_on: bool = True,
        effort: str = "",
        levels: list[str] | None = None,
        current_thoughts: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model or "?"
        self.thinking_on = bool(thinking_on)
        self.effort = (effort or "").strip().lower()
        self.levels = list(levels or [])
        self.current_thoughts = int(current_thoughts or 0)

    def _rows(self) -> list[Option]:
        opts: list[Option] = [Option(f"[b]Thinking — {self.model}[/]", disabled=True)]
        on_mark = "●" if self.thinking_on else " "
        off_mark = "●" if not self.thinking_on else " "
        opts.append(Option(f"[{on_mark}] Thinking on  [dim](thought bubbles show)[/]", id="__on__"))
        opts.append(Option(f"[{off_mark}] Thinking off  [dim](hide thought bubbles)[/]", id="__off__"))
        if self.levels:
            opts.append(Option("[b]Effort[/]", disabled=True))
            for lvl in self.levels:
                mark = "●" if lvl.lower() == self.effort else " "
                opts.append(Option(f"[{mark}] {lvl}  [dim](reasoning effort)[/]", id=f"__effort__{lvl}"))
        else:
            opts.append(
                Option("[dim]No effort levels — this model thinks at a fixed level[/]", disabled=True)
            )
        opts.append(Option("[b]Thoughts[/]", disabled=True))
        opts.append(
            Option(
                f"Show thoughts  [dim](expand {self.current_thoughts or 'all'})[/]",
                id="__show__",
            )
        )
        opts.append(Option("Hide thoughts  [dim](collapse all)[/]", id="__hide__"))
        return opts

    def compose(self) -> ComposeResult:
        with Vertical(id="thinking-picker-box"):
            yield Static(
                "  Thinking — arrows to move · Enter to apply · Esc to cancel",
                id="thinking-picker-title",
            )
            yield OptionList(*self._rows(), id="thinking-picker-list")

    def on_mount(self) -> None:
        lst = self.query_one("#thinking-picker-list", OptionList)
        lst.focus()
        # land on the current effort (or Thinking on) so arrows move from it
        try:
            want = f"__effort__{self.effort}" if self.effort else "__on__"
            for i in range(lst.option_count):
                if lst.get_option_at_index(i).id == want:
                    lst.highlighted = i
                    break
            else:
                lst.highlighted = 1
            lst.scroll_to_highlight()
        except Exception:
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = getattr(event.option, "id", None)
        self.dismiss(option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)
