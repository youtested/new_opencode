"""Controller: maps engine events (text deltas, tool start/stop, permission,
usage) to TUI screen updates. Keeps the engine headless-testable.

The engine emits dict events via on_event. The TUI app installs handlers.
"""

from __future__ import annotations

from typing import Any, Callable

Handler = Callable[[dict[str, Any]], None]


class Controller:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}

    def on(self, kind: str, handler: Handler) -> None:
        self._handlers.setdefault(kind, []).append(handler)

    def emit(self, event: dict[str, Any]) -> None:
        kind = event.get("kind", "")
        for handler in self._handlers.get(kind, []):
            handler(event)

    def emit_event(self, event: dict[str, Any]) -> None:
        self.emit(event)

    def clear(self) -> None:
        self._handlers.clear()
