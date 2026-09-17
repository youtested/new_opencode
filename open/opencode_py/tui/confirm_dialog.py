from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


_CONFIRM_CSS = """
ConfirmDialog {
    align: center middle;
}
#confirm-box {
    width: 60;
    max-width: 92%;
    height: auto;
    background: $surface;
    border: round $accent;
    padding: 1 2;
}
#confirm-title {
    text-style: bold;
    color: $accent;
    height: 1;
    margin-bottom: 1;
}
#confirm-message {
    margin-bottom: 1;
}
#confirm-actions {
    height: 3;
    align: center middle;
    background: $surface;
}
#confirm-actions Button {
    height: 3;
    min-width: 12;
    padding: 0 2;
    margin: 0 1;
    border: heavy $accent;
    background: transparent;
    color: $text;
}
"""


class ConfirmDialog(ModalScreen[bool]):
    CSS = _CONFIRM_CSS
    def __init__(
        self,
        message: str = "Are you sure?",
        title: str = "Confirm",
        confirm_label: str = "Delete",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._message = message
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(f"  {self._title}  ", id="confirm-title")
            yield Label(self._message, id="confirm-message")
            with Horizontal(id="confirm-actions"):
                yield Button(self._confirm_label, id="confirm-yes", variant="error")
                yield Button("Cancel", id="confirm-no", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#confirm-no", Button).focus()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)
        event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(False)
            event.stop()
