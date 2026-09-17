"""Permission dialog: Allow / Always allow / Deny / Always deny per tool+input."""

from __future__ import annotations

from typing import Any, Callable

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class PermissionDialog(ModalScreen[str]):
    """Modal asking to approve a tool call.

    Choices: once / always / deny / always-deny. Esc = deny.
    """

    def __init__(
        self,
        description: str,
        on_decision: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.description = description
        self.on_decision = on_decision

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        with Vertical(classes="cmd-popup perm-popup"):
            yield Label("Permission required", classes="dialog-title")
            yield Label(self.description, classes="dialog-desc")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Allow once", id="perm-once", variant="primary")
                yield Button("Always allow", id="perm-always", variant="success")
                yield Button("Deny", id="perm-deny", variant="error")
                yield Button("Always deny", id="perm-always-deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decision = {
            "perm-once": "once",
            "perm-always": "always",
            "perm-deny": "deny",
            "perm-always-deny": "always-deny",
        }.get(event.button.id or "")
        if decision is None:
            return
        if self.on_decision:
            self.on_decision(decision)
        self.dismiss(decision)

    def on_key(self, event: Any) -> None:
        from textual.events import Key

        if isinstance(event, Key) and event.key == "escape":
            # Esc = deny; also report it so a blocked engine thread can resume.
            # If the hint is already armed (a 1st ESC happened), this is the
            # 2nd ESC: force-stop everything like the app handler does — the
            # dialog consumes the key so the app binding never fires.
            try:
                app = getattr(self, "app", None)
                if app is not None and getattr(app, "_esc_presses", 0) >= 1:
                    app._force_stop_all(pop_modal=False)
                    try:
                        app._disarm_interrupt_escape()
                    except Exception:
                        pass
            except Exception:
                pass
            if self.on_decision:
                self.on_decision("deny")
            self.dismiss("deny")
            event.stop()
