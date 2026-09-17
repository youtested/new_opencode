"""Theme picker screen (bare /theme): arrow-navigable, dark themes first.

Each row shows three live color swatches sampled from that theme's own
palette (background / primary / accent), the command-style name, and a dim
one-line hint. Enter applies the highlighted theme immediately; Esc cancels.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .theme import LIGHT_THEMES, THEMES, get_theme

# Short per-theme briefing shown dimmed after the name.
HINTS: dict[str, str] = {
    "opencode": "opencode default dark",
    "dark": "black · orange & gray",
    "tokyo-night": "deep indigo · neon blue/purple",
    "blackout": "true-black OLED · amber/violet",
    "catppuccin-mocha": "soft pastel on dark cocoa",
    "dracula": "classic purple/pink night",
    "nord": "cool arctic blues",
    "gruvbox-dark": "warm retro amber/olive",
    "one-dark": "atom-style muted slate",
    "solarized": "classic low-contrast teal",
    "solarized-light": "paper-warm light",
    "catppuccin-latte": "pastel light",
    "github-light": "clean github white",
}

_THEME_PICKER_CSS = """
ThemePicker {
    align: center middle;
}
#theme-picker-box {
    width: 64;
    max-height: 80%;
    height: auto;
    background: $surface;
    border: round #666;
    padding: 0 1;
}
#theme-picker-title {
    text-style: bold;
    margin-bottom: 1;
}
#theme-picker-list {
    height: auto;
    max-height: 22;
    background: transparent;
}
"""


def _swatches(name: str) -> str:
    """Three blocks painted with the theme's own bg/primary/accent colors."""
    t = get_theme(name)
    return (
        f"[on {t.c('background')}]  [/]"
        f"[on {t.c('primary')}]  [/]"
        f"[on {t.c('accent')}]  [/]"
    )


def _row(name: str, current: str) -> str:
    marker = "●" if name == current else " "
    hint = HINTS.get(name, "")
    return f"{_swatches(name)} [{marker}] /{name}  [dim]{hint}[/]"


class ThemePicker(ModalScreen[str]):
    """Pick a theme with the arrow keys; Enter applies it live."""

    CSS = _THEME_PICKER_CSS
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, current: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.current_theme = current or "opencode"

    def compose(self) -> ComposeResult:
        with Vertical(id="theme-picker-box"):
            yield Static(
                f"  Theme — arrows to move · Enter to apply · Esc to cancel"
                f"  [dim](current: {self.current_theme})[/]",
                id="theme-picker-title",
            )
            options: list[Option] = [Option("[b]Dark themes[/]", disabled=True)]
            for name in THEMES:
                if name == "opencode-dark":
                    continue  # alias — `opencode` is the canonical entry
                if name == LIGHT_THEMES[0]:
                    options.append(
                        Option("", disabled=True)  # visual gap between sections
                    )
                    options.append(Option("[b]Light themes[/]", disabled=True))
                options.append(Option(_row(name, self.current_theme), id=name))
            yield OptionList(*options, id="theme-picker-list")

    def on_mount(self) -> None:
        lst = self.query_one("#theme-picker-list", OptionList)
        lst.focus()
        # highlight the current theme so Enter re-applies / arrows move from it
        for i in range(lst.option_count):
            if lst.get_option_at_index(i).id == self.current_theme:
                lst.highlighted = i
                break
        else:
            lst.highlighted = 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = getattr(event.option, "id", None)
        self.dismiss(option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)
