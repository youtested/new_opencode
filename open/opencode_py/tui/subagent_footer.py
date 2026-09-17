"""Subagent footer: opencode-style bar inside a sub-agent session.

Mirrors packages/tui/src/routes/session/subagent-footer.tsx. When the current
session is a child (it has a ``parent_id``), a bar renders beneath the chat:

    Build  (2 of 4)   12,345 (6%)     Prev ←    Next →

``Build`` is the child's agent label, ``(2 of 4)`` is its position among every
sub-agent the same parent spawned (sorted by creation time — the way parallel
agents are numbered), and the right cluster hops across the parallel siblings
(back to the parent is the ``↑`` key, scope-agnostic: empty prompt, chat focus
or footer).
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static

from .theme import active_theme


class NavRequested(Message):
    """A footer navigation button was clicked: parent / prev / next."""

    def __init__(self, action: str) -> None:
        super().__init__()
        self.action = action


class _NavButton(Static):
    """One clickable footer nav segment (``Prev ←`` / ``Next →``)."""

    def __init__(self, action: str, label: str, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._action = action
        self._label = label

    def on_mount(self) -> None:
        theme = active_theme()
        t = Text()
        t.append(self._label, style=theme.c("text"))
        self.update(t)

    def on_click(self, event: Any) -> None:
        self.post_message(NavRequested(self._action))
        event.stop()


class SubagentFooter(Horizontal):
    """Bottom bar rendered only while viewing a sub-agent session.

    The app calls :meth:`show` with the child's label, sibling index and token
    usage, or :meth:`hide` (equivalent to ``display = "none"``) for normal
    (parent) sessions.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.styles.display = "none"

    def compose(self) -> ComposeResult:
        yield Static("", id="subagent-info")
        with Horizontal(id="subagent-nav"):
            yield Static("   ", classes="subagent-gap")
            yield _NavButton("prev", "Prev ←")
            yield Static("   ", classes="subagent-gap")
            yield _NavButton("next", "Next →")

    def show(
        self,
        *,
        label: str,
        index: int,
        total: int,
        usage: dict[str, int] | None = None,
    ) -> None:
        theme = active_theme()
        t = Text()
        t.append(label or "Subagent", style=f"bold {theme.c('text')}")
        t.append(f"  ({index} of {total})", style=theme.c("text_muted"))
        usage_text = self._usage_text(usage)
        if usage_text:
            t.append(f"  {usage_text}", style=theme.c("text_muted"))
        try:
            self.query_one("#subagent-info", Static).update(t)
        except Exception:
            pass
        self.display = "block"

    def hide(self) -> None:
        self.display = "none"

    def _usage_text(self, usage: dict[str, int] | None) -> str:
        """`12,345 (6%)` — the same context hint as the status bar footer."""
        usage = usage or {}
        total = usage.get("total_tokens") or 0
        if not total:
            return ""
        ctx = usage.get("context_size") or 0
        text = f"{total:,}"
        if ctx:
            pct = min(100, round(total / ctx * 100))
            text += f" ({pct}%)"
        return text