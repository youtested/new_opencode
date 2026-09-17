"""Permissions popup: every effective permission rule, readable.

Shows the live permission MODE plus all rules grouped by tool in plain
words (`bash → allow`, `read *.env → ask`), so `/permissions` answers
"what can the agent do right now" instead of dumping raw `{}` config.
Read-only: view with arrows, Esc closes. Never raises.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .theme import active_theme


def _action_color(theme: Any, action: str) -> str:
    action = str(action or "").strip().lower()
    if action == "allow":
        return theme.c("success")
    if action == "deny":
        return theme.c("error")
    if action == "ask":
        return theme.c("warning")
    return theme.c("text_muted")


def build_permission_rows(
    mode: str,
    agent: str,
    user_permission: dict[str, Any] | None,
    cfg: Any = None,
) -> list[str]:
    """Renderable rows (rich markup) for the popup, engine-free.

    Merges the agent defaults under the user's config exactly like
    ``merge_permissions`` so the display matches what the engine
    enforces (last matching rule wins is evaluation-time; here every
    configured rule is listed).
    """
    from ..permission import merge_permissions

    theme = active_theme()
    rows: list[str] = []
    rows.append(
        f"[{theme.c('text_muted')}]mode:[/] [{theme.c('accent')}]{escape(str(mode or 'auto'))}[/]"
        f"  [{theme.c('text_muted')}]agent:[/] [{theme.c('accent')}]{escape(str(agent or 'build'))}[/]"
    )
    merged = merge_permissions(user_permission or {}, agent or "build", cfg)
    if not merged:
        rows.append(f"[{theme.c('text_muted')}](no rules — everything asks)[/]")
        return rows
    for perm in sorted(merged):
        value = merged[perm]
        if isinstance(value, str):
            color = _action_color(theme, value)
            rows.append(
                f"[{theme.c('text')}]  {escape(str(perm))}[/]"
                f"  [{color}]{escape(str(value))}[/]"
            )
        elif isinstance(value, dict):
            rows.append(f"[{theme.c('text')}]  {escape(str(perm))}[/]")
            for pattern in sorted(value, key=str):
                action = str(value[pattern])
                color = _action_color(theme, action)
                rows.append(
                    f"[{theme.c('text_muted')}]    {escape(str(pattern))}[/]"
                    f"  [{color}]{escape(action)}[/]"
                )
        else:
            rows.append(
                f"[{theme.c('text')}]  {escape(str(perm))}[/]"
                f"  [{theme.c('text_muted')}]{escape(str(value))}[/]"
            )
    return rows


class PermissionsPopup(ModalScreen[None]):
    """Read-only permissions viewer; Esc closes."""

    def __init__(
        self,
        mode: str = "auto",
        agent: str = "build",
        user_permission: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mode = mode
        self._agent = agent
        self._user_permission = dict(user_permission or {})

    def _options(self) -> list[Option]:
        opts: list[Option] = []
        for i, text in enumerate(
            build_permission_rows(self._mode, self._agent, self._user_permission)
        ):
            opts.append(Option(text, id=f"__perm__{i}", disabled=True))
        return opts

    def compose(self) -> ComposeResult:
        theme = active_theme()
        with Vertical(classes="cmd-popup session-popup"):
            yield Static("  Permissions  ", classes="cmd-popup-title")
            yield Static(
                f"[{theme.c('text_muted')}]↑/↓ scroll · Esc close[/]",
                classes="cmd-popup-usage",
            )
            yield OptionList(*self._options(), id="permissions-list")

    def on_mount(self) -> None:
        try:
            self.query_one("#permissions-list", OptionList).focus()
        except Exception:
            pass

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
