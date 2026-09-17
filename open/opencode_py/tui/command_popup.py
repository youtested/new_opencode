"""Centered command popup: shows a command's output (or help) and lets you act.

For read-only commands (e.g. /models, /help) the popup runs the command and
renders its output in the middle of the screen with a Close button. For
state-changing commands (e.g. /undo, /connect) it shows the description and a
Run button so the action only happens when explicitly confirmed.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from .theme import active_theme


class CommandPopup(ModalScreen[str]):
    """Modal that pops up in the middle of the screen for the chosen command."""

    def __init__(
        self,
        name: str,
        description: str,
        content: str | None = None,
        usage: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cmd_name = name
        self.cmd_description = description
        self.content = content
        self.usage = usage

    @property
    def previewed(self) -> bool:
        return self.content is not None

    def compose(self) -> ComposeResult:
        theme = active_theme()
        with Vertical(classes="cmd-popup"):
            yield Static(f"  /{self.cmd_name}  ", classes="cmd-popup-title")
            if self.previewed:
                with VerticalScroll(classes="cmd-popup-scroll", id="cmd-popup-scroll"):
                    yield Static(self.content or "(no output)", id="cmd-popup-body")
                with Horizontal(classes="cmd-popup-actions"):
                    yield Button("Close", id="cmd-close", variant="primary")
            else:
                yield Static(self.cmd_description or "(no description)", classes="cmd-popup-desc")
                # Only commands that actually take arguments get a usage line —
                # "Usage: /help [args]" on a no-arg command was fabricated noise.
                if self.usage:
                    yield Static(self.usage, classes="cmd-popup-usage")
                yield Static("", id="cmd-popup-note")
                with Horizontal(classes="cmd-popup-actions"):
                    yield Button("Run", id="cmd-run", variant="primary")
                    yield Button("Cancel", id="cmd-cancel", variant="default")

    def on_mount(self) -> None:
        theme = active_theme()
        if self.previewed:
            self.query_one("#cmd-close", Button).focus()
        else:
            self.query_one("#cmd-run", Button).focus()
            self.query_one("#cmd-popup-note", Static).update(
                f"[{theme.c('text_muted')}]Enter = Run    Esc = Cancel[/]"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "cmd-close":
            self.dismiss("close")
        elif bid == "cmd-run":
            self.dismiss("run")
        else:
            self.dismiss("cancel")

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss("cancel")
            event.stop()
        elif event.key == "enter" and not self.previewed:
            self.dismiss("run")
            event.stop()
