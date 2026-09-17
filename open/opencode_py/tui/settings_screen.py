"""Settings screen: interactive, arrow-navigable list of every app/tool setting.

Reached via ctrl+p / ctrl+s as a centered popup. All of the config + tool output
settings are listed; use Up/Down to move the selection, Enter to edit a value:

  - boolean settings toggle on Enter
  - enum settings (theme, agent, permission mode) cycle on Left/Right/Enter
  - numeric / free-text settings open an inline input to type a value
  - model settings open the model picker

Esc closes the popup (or cancels an in-progress edit). The Model and Close
buttons remain for mouse/touch users.

Rendering uses a single Textual ``OptionList`` (same pattern as the session
picker): the widget itself keeps the highlight scrolled into view, so moving
past the visible edge auto-scrolls instead of losing the selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from ..config import save_config
from .model_picker import ModelPicker
from .theme import active_theme, get_theme, set_active_theme, theme_names


@dataclass
class Row:
    label: str
    get: Callable[[], str]
    kind: str = "info"  # info | bool | enum | int | string | model
    choices: list[str] | None = None
    apply: Callable[[str], None] | None = None
    propagate: bool = True  # a model pick only retargets the app engine if True


_LABEL_W = 24
_VALUE_MAX = 40


def _fmt_value(row: Row, theme: Any) -> list[tuple[str, str]]:
    """[(text, style), ...] spans for a setting's value, styled by kind."""
    try:
        value = str(row.get())
    except Exception:
        value = "?"
    if row.kind == "bool":
        if value == "yes":
            return [("on", f"bold {theme.c('success')}")]
        return [("off", theme.c("text_muted"))]
    if row.kind == "enum":
        return [(value, "bold")]
    if row.kind == "model":
        return [(value, theme.c("secondary"))]
    return [(value, theme.c("text"))]


def _clip_value(text: str, limit: int = _VALUE_MAX) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_settings(
    rows: list[Row],
    index: int,
    width: int,
    theme: Any,
    editing: bool = False,
) -> tuple[list[Any], list[str]]:
    """Legacy pure-function renderer (kept for compatibility).

    The live popup now renders through ``OptionList`` (see
    ``SettingsScreen._options``); this helper preserves the old import surface
    for tests and external callers.
    """
    from rich.text import Text

    accent = theme.c("accent")
    muted = theme.c("text_muted")
    highlight = theme.c("primary")
    bar_bg = theme.c("background_element")

    lines: list[Any] = []
    plain: list[str] = []
    first_section = True
    for i, row in enumerate(rows):
        selected = i == index and row.kind != "info" and not editing
        if row.kind == "info" and row.label.startswith("—"):
            name = row.label.strip("— ").strip()
            if not first_section:
                lines.append(Text(""))
                plain.append("")
            title = f" {name} "
            rule_len = max(6, width - 5 - len(title))
            t = Text(f"──{title}", style=f"bold {accent}")
            t.append("─" * rule_len, style=accent)
            lines.append(t)
            plain.append(t.plain)
            first_section = False
            continue
        label = row.label[: _LABEL_W - 1].ljust(_LABEL_W)
        if row.kind == "info":
            t = Text("   ")
            t.append(label, style=muted)
            try:
                t.append(str(row.get()))
            except Exception:
                t.append("?")
            lines.append(t)
            plain.append(t.plain)
            continue
        vspans = _fmt_value(row, theme)
        if selected:
            t = Text("   ", style=f"on {bar_bg}")
            t.append(label, style=f"{highlight} on {bar_bg}")
            for s, st in vspans:
                t.append(s, style=f"{st} on {bar_bg}")
            used = 3 + len(label) + sum(len(s) for s, _ in vspans)
            t.append(" " * max(0, width - used - 2), style=f"on {bar_bg}")
        else:
            t = Text("   ")
            t.append(label, style=muted)
            for s, st in vspans:
                t.append(s, style=st)
        lines.append(t)
        plain.append(t.plain)
    return lines, plain


def _row_hint(row: Row) -> str:
    if row.kind == "bool":
        return "Enter toggles"
    if row.kind == "enum":
        return "change"
    if row.kind == "model":
        return "Enter picks"
    if row.kind in ("int", "string"):
        return "Enter edits"
    return ""


class SettingsScreen(ModalScreen[None]):
    """Centered popup listing every app/tool setting, keyboard navigable."""

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = kwargs.pop("cfg")
        self.engine = kwargs.pop("engine")
        self.auth = kwargs.pop("auth")
        self.session = kwargs.pop("session", None)
        self.on_model_change = kwargs.pop("on_model_change", None)
        self.on_apply = kwargs.pop("on_apply", None)
        super().__init__(**kwargs)
        self.rows: list[Row] = []
        self.index = 0
        self.editing = False
        self._edit_row: Row | None = None
        # legacy geometry caches (kept so old callers don't break; the live
        # popup no longer needs them because OptionList tracks visibility).
        self._row_ys: list[int] = []
        self._row_heights: list[int] = []
        self.body: Any = None

    # -- row model ---------------------------------------------------------
    def _build_rows(self) -> list[Row]:
        cfg = self.cfg
        engine = self.engine

        def _engine_agent() -> str:
            try:
                return str(getattr(engine, "agent", None) or "build")
            except Exception:
                return "build"

        def _engine_perm_mode() -> str:
            try:
                return str(getattr(getattr(engine, "permission", None), "mode", "auto"))
            except Exception:
                return "auto"

        rows: list[Row] = []
        rows.append(Row("— General —", lambda: "", kind="info"))
        rows.append(Row("provider", lambda: str(getattr(cfg, "provider", "?"))))
        rows.append(Row(
            "model",
            lambda: f"opencode/{getattr(cfg, 'model', '?')}",
            kind="model",
            apply=lambda v: setattr(cfg, "model", v),
        ))
        rows.append(Row(
            "small model",
            lambda: f"opencode/{getattr(cfg, 'small_model', '?')}",
            kind="model",
            apply=lambda v: setattr(cfg, "small_model", v),
            propagate=False,
        ))
        rows.append(Row(
            "default agent",
            lambda: str(getattr(cfg, "default_agent", "build")),
            kind="enum",
            choices=["build", "plan", "explore"],
            apply=lambda v: setattr(cfg, "default_agent", v),
        ))
        rows.append(Row("active agent", _engine_agent))
        rows.append(Row(
            "model read timeout (s)",
            lambda: str(getattr(cfg, "model_read_timeout", "")),
            kind="int",
            apply=lambda v: setattr(cfg, "model_read_timeout", float(v)),
        ))
        rows.append(Row(
            "save data",
            lambda: "yes" if getattr(cfg, "low_data", False) else "no",
            kind="enum",
            choices=["yes", "no"],
            apply=lambda v: setattr(cfg, "low_data", v == "yes"),
        ))
        rows.append(Row(
            "low ram",
            lambda: str(getattr(cfg, "low_ram", "auto")),
            kind="enum",
            choices=["auto", "on", "off"],
            apply=lambda v: setattr(cfg, "low_ram", v),
        ))
        rows.append(Row(
            "chat live window",
            lambda: str(getattr(cfg, "chat_live_window", 120)),
            kind="int",
            apply=lambda v: setattr(cfg, "chat_live_window", max(0, int(v))),
        ))
        rows.append(Row(
            "subagent depth",
            lambda: str(getattr(cfg, "subagent_depth", "")),
            kind="int",
            apply=lambda v: setattr(cfg, "subagent_depth", int(v)),
        ))
        rows.append(Row(
            "allow all permissions",
            lambda: "yes" if getattr(cfg, "permission_mode", "auto") in ("auto", "fully_auto") else "no",
            kind="enum",
            choices=["no", "yes"],
            apply=lambda v: setattr(
                cfg, "permission_mode", "auto" if v == "yes" else "ask"
            ),
        ))
        rows.append(Row(
            "permission mode",
            lambda: str(getattr(cfg, "permission_mode", "auto")),
            kind="enum",
            choices=["ask", "auto", "fully_auto"],
            apply=lambda v: setattr(cfg, "permission_mode", v),
        ))
        rows.append(Row(
            "reasoning effort",
            lambda: str(getattr(cfg, "reasoning_effort", "") or "default"),
            kind="enum",
            choices=["default", "minimal", "low", "medium", "high", "xhigh", "max", "none"],
            apply=lambda v: setattr(cfg, "reasoning_effort", "" if v == "default" else v),
        ))

        def _apply_theme(value: str) -> None:
            cfg.theme = value
            set_active_theme(value)

        rows.append(Row("— Appearance —", lambda: "", kind="info"))
        rows.append(Row(
            "theme",
            lambda: str(getattr(cfg, "theme", "opencode")),
            kind="enum",
            choices=theme_names(),
            apply=_apply_theme,
        ))

        rows.append(Row("— Tools & Output —", lambda: "", kind="info"))
        rows.append(Row(
            "bash timeout (s)",
            lambda: str(getattr(cfg, "bash_default_timeout", "")),
            kind="int",
            apply=lambda v: setattr(cfg, "bash_default_timeout", int(v)),
        ))
        rows.append(Row(
            "tool output max lines",
            lambda: str(getattr(cfg, "tool_output_max_lines", "")),
            kind="int",
            apply=lambda v: setattr(cfg, "tool_output_max_lines", int(v)),
        ))
        rows.append(Row(
            "tool output max bytes",
            lambda: str(getattr(cfg, "tool_output_max_bytes", "")),
            kind="int",
            apply=lambda v: setattr(cfg, "tool_output_max_bytes", int(v)),
        ))
        rows.append(Row(
            "context budget (tokens)",
            lambda: str(getattr(cfg, "context_budget", "")),
            kind="int",
            apply=lambda v: setattr(cfg, "context_budget", int(v)),
        ))

        rows.append(Row("— Diff —", lambda: "", kind="info"))
        rows.append(Row(
            "diff style",
            lambda: str(getattr(cfg, "diff_style", "split")),
            kind="enum",
            choices=["split", "stacked"],
            apply=lambda v: setattr(cfg, "diff_style", v),
        ))
        rows.append(Row(
            "diff wrap",
            lambda: str(getattr(cfg, "diff_wrap_mode", "word")),
            kind="enum",
            choices=["word", "none"],
            apply=lambda v: setattr(cfg, "diff_wrap_mode", v),
        ))
        rows.append(Row(
            "diff backgrounds",
            lambda: "off" if getattr(cfg, "suppress_backgrounds", False) else "on",
            kind="enum",
            choices=["on", "off"],
            apply=lambda v: setattr(cfg, "suppress_backgrounds", v == "off"),
        ))
        rows.append(Row(
            "subagent footer",
            lambda: "on" if getattr(cfg, "subagent_footer", False) else "off",
            kind="enum",
            choices=["off", "on"],
            apply=lambda v: setattr(cfg, "subagent_footer", v == "on"),
        ))

        rows.append(Row("— Conversation —", lambda: "", kind="info"))
        rows.append(Row(
            "auto continue",
            lambda: "yes" if getattr(cfg, "auto_continue", False) else "no",
            kind="bool",
            apply=lambda v: setattr(cfg, "auto_continue", v == "yes"),
        ))
        rows.append(Row(
            "auto compact",
            lambda: "yes" if getattr(cfg, "compaction_enabled", True) else "no",
            kind="bool",
            apply=lambda v: setattr(cfg, "compaction_enabled", v == "yes"),
        ))
        rows.append(Row(
            "compact tail turns",
            lambda: str(getattr(cfg, "compaction_tail_turns", "")),
            kind="int",
            apply=lambda v: setattr(cfg, "compaction_tail_turns", int(v)),
        ))

        rows.append(Row("— Voice (TTS) —", lambda: "", kind="info"))
        rows.append(Row(
            "speak",
            lambda: "yes" if getattr(cfg, "tts_enabled", False) else "no",
            kind="bool",
            apply=lambda v: setattr(cfg, "tts_enabled", v == "yes"),
        ))
        rows.append(Row(
            "auto voice",
            lambda: "yes" if getattr(cfg, "tts_auto", False) else "no",
            kind="bool",
            apply=lambda v: setattr(cfg, "tts_auto", v == "yes"),
        ))
        rows.append(Row(
            "speak engine",
            lambda: str(getattr(cfg, "tts_engine", "auto")),
            kind="enum",
            choices=["auto", "offline", "elevenlabs"],
            apply=lambda v: setattr(cfg, "tts_engine", v),
        ))
        rows.append(Row(
            "speak voice",
            lambda: str(getattr(cfg, "tts_voice", "") or "(default)"),
            kind="string",
            apply=lambda v: setattr(cfg, "tts_voice", "" if v == "(default)" else v),
        ))
        rows.append(Row(
            "speak language",
            lambda: str(getattr(cfg, "tts_language", "") or "(default)"),
            kind="string",
            apply=lambda v: setattr(cfg, "tts_language", "" if v == "(default)" else v),
        ))
        rows.append(Row(
            "speak rate",
            lambda: str(getattr(cfg, "tts_rate", 1.0)),
            kind="string",
            apply=lambda v: setattr(cfg, "tts_rate", min(4.0, max(0.25, float(v)))),
        ))
        rows.append(Row(
            "speak pitch",
            lambda: str(getattr(cfg, "tts_pitch", 1.0)),
            kind="string",
            apply=lambda v: setattr(cfg, "tts_pitch", min(2.0, max(0.5, float(v)))),
        ))
        rows.append(Row(
            "online voice id",
            lambda: str(getattr(cfg, "tts_voice_id", "") or "(default Rachel)"),
            kind="string",
            apply=lambda v: setattr(cfg, "tts_voice_id", "" if v == "(default Rachel)" else v),
        ))
        rows.append(Row(
            "online model",
            lambda: str(getattr(cfg, "tts_model", "eleven_multilingual_v2")),
            kind="string",
            apply=lambda v: setattr(cfg, "tts_model", v or "eleven_multilingual_v2"),
        ))

        rows.append(Row("— Permissions —", lambda: "", kind="info"))
        rows.append(Row(
            "permission mode",
            _engine_perm_mode,
            kind="enum",
            choices=["auto", "ask", "deny", "fully_auto"],
            apply=(lambda v: setattr(engine.permission, "mode", v)) if engine is not None and getattr(engine, "permission", None) is not None else None,
        ))
        try:
            perm_items = list((getattr(cfg, "permission", None) or {}).items())
        except Exception:
            perm_items = []
        for tool, action in perm_items:
            rows.append(Row(f"permission:{tool}", lambda a=action: str(a)))

        rows.append(Row("— Rotation —", lambda: "", kind="info"))
        try:
            from ..providers.rotation import live_opencode_lanes

            live = live_opencode_lanes(
                provider=str(getattr(cfg, "provider", "opencode") or "opencode"))
            live_ids = {str(l.get("model", "")).split("/", 1)[-1] for l in live}
        except Exception:
            live, live_ids = [], set()
        shown: list[tuple[str, str]] = []
        try:
            cfg_lanes = list(getattr(cfg, "rotation", None) or [])
        except Exception:
            cfg_lanes = []
        for lane in cfg_lanes:
            mid = str((lane or {}).get("model", "?"))
            bare = mid.split("/", 1)[-1]
            status = "" if bare in live_ids else " (removed)"
            shown.append((f"{(lane or {}).get('provider', '?')}/{mid}{status}", status))
        if not shown:
            # no pinned config: show what rotation ACTUALLY uses right now —
            # picked model first, then live failover best-capable first.
            pick = f"{getattr(cfg, 'provider', '?')}/{getattr(cfg, 'model', '?')}"
            shown.append((pick, " (current)"))
            for lane in live:
                mid = str(lane.get("model", ""))
                if mid.split("/", 1)[-1] == str(getattr(cfg, "model", "")).split("/", 1)[-1]:
                    continue
                shown.append((f"{lane.get('provider', '?')}/{mid}", ""))
        for text, _status in shown[:12]:
            rows.append(Row("lane", lambda t=text: t))
        if len(shown) > 12:
            rows.append(Row("lane", lambda: f"… +{len(shown) - 12} more"))

        rows.append(Row("— Plugins & MCP —", lambda: "", kind="info"))
        try:
            plugins = list((getattr(cfg, "raw", None) or {}).get("plugins") or [])
        except Exception:
            plugins = []
        for p in plugins:
            rows.append(Row("plugin", lambda p=p: str(p)))
        try:
            servers = list((getattr(cfg, "raw", None) or {}).get("mcpServers") or {})
        except Exception:
            servers = []
        for name in servers:
            rows.append(Row("mcp", lambda name=name: str(name)))
        rows.append(Row(
            "auth keys",
            lambda: ", ".join(sorted(self.auth.list())) if self.auth and self.auth.list() else "(none)",
        ))
        if self.session:
            rows.append(Row(
                "session",
                lambda: f"{str(getattr(self.session, 'id', '?'))[:12]} [{getattr(self.session, 'agent', '?')}] {getattr(self.session, 'model', None) or '?'}",
            ))
        return rows

    @property
    def _row(self) -> Row:
        return self.rows[self.index]

    # -- options -----------------------------------------------------------
    def _option_text(self, row: Row, theme: Any) -> str:
        """One-line markup for an OptionList row."""
        accent = theme.c("accent")
        muted = theme.c("text_muted")
        if row.kind == "info" and row.label.startswith("—"):
            name = escape(row.label.strip("— ").strip())
            return f"[bold {accent}]── {name} ──[/]"
        try:
            raw_value = str(row.get())
        except Exception:
            raw_value = "?"
        if row.kind == "info":
            label = escape(row.label[: _LABEL_W - 1])
            return f"[{muted}]  {label}: {escape(_clip_value(raw_value, 60))}[/]"
        label = escape(row.label[: _LABEL_W - 1].ljust(_LABEL_W))
        value = escape(_clip_value(raw_value))
        hint = escape(_row_hint(row))
        if row.kind == "bool":
            color = theme.c("success") if raw_value == "yes" else muted
            pill = "● on" if raw_value == "yes" else "○ off"
            return f"[{muted}]{label}[/]  [bold {color}]{escape(pill)}[/]  [{muted}]{hint}[/]"
        if row.kind == "enum":
            return f"[{muted}]{label}[/]  [bold]{value}[/]  [{muted}]{escape(_row_hint(row))}[/]"
        if row.kind == "model":
            return f"[{muted}]{label}[/]  [{theme.c('secondary')}]{value}[/]  [{muted}]{hint}[/]"
        return f"[{muted}]{label}[/]  {value}  [{muted}]{hint}[/]"

    def _options(self) -> list[Option]:
        try:
            theme = active_theme()
        except Exception:
            theme = get_theme("opencode")
        options: list[Option] = []
        for i, row in enumerate(self.rows):
            text = self._option_text(row, theme)
            if row.kind == "info":
                options.append(Option(text, id=f"__hdr__{i}", disabled=True))
            else:
                options.append(Option(text, id=f"srow-{i}"))
        return options

    # -- UI ----------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Vertical(classes="cmd-popup settings"):
            yield Static("Settings", classes="settings-title")
            yield Static("", id="settings-count", classes="settings-count")
            yield OptionList(id="settings-list")
            yield Input(
                id="settings-edit-input",
                classes="settings-edit",
                placeholder="enter a value",
            )
            with Horizontal(classes="cmd-popup-actions"):
                yield Button("Model", id="settings-model", variant="default")
                yield Button("Save", id="settings-save", variant="primary")
                yield Button("Close", id="settings-close", variant="default")
            yield Static(
                "↑/↓ select · Enter edit · ←/→ change · Esc close",
                id="settings-hint",
                classes="settings-hint",
            )

    def _list(self) -> Any | None:
        try:
            return self.query_one("#settings-list", OptionList)
        except Exception:
            return None

    def _scroll_container(self) -> Any:
        ol = self._list()
        if ol is not None:
            return ol
        # legacy fallback for harness-free unit tests (not attached)
        class _Dummy:
            def focus(self) -> None:
                pass

            def scroll_to(self, *a: Any, **k: Any) -> None:
                pass

            def scroll_to_highlight(self, *a: Any, **k: Any) -> None:
                pass

        return _Dummy()

    def on_mount(self) -> None:
        self.rows = self._build_rows()
        self.index = next((i for i, r in enumerate(self.rows) if r.kind != "info"), 0)
        self._set_edit_input_visible(False)
        self._toggle_row_visibility()
        self._populate()
        ol = self._list()
        if ol is not None:
            try:
                ol.can_focus = True
                ol.focus()
            except Exception:
                pass

    def _populate(self, focus: int | None = None) -> None:
        ol = self._list()
        if ol is None:
            return
        if focus is not None:
            self.index = focus
        try:
            ol.clear_options()
            ol.add_options(self._options())
        except Exception:
            return
        self._sync_highlight()
        self._update_count()

    def _sync_highlight(self) -> None:
        ol = self._list()
        if ol is None or not self.rows:
            return
        idx = max(0, min(self.index, len(self.rows) - 1))
        self.index = idx
        try:
            # one Option per row, so list position == row index
            ol.highlighted = idx
            ol.scroll_to_highlight()
        except Exception:
            pass

    def _update_count(self) -> None:
        try:
            total = sum(1 for r in self.rows if r.kind != "info")
            pos = sum(1 for r in self.rows[: self.index + 1] if r.kind != "info")
            self.query_one("#settings-count", Static).update(
                f"{self._row.label} · {pos}/{total}"
            )
        except Exception:
            pass
        try:
            self.query_one(".settings-title", Static).update(
                f"Settings · {len(self.rows)} items"
            )
        except Exception:
            pass

    def _keep_selection_visible(self) -> None:
        """Scroll the selection into view (OptionList-native)."""
        if not self.is_attached:
            return
        if self.editing:
            return
        try:
            ol = self._list()
            if ol is not None:
                ol.scroll_to_highlight()
        except Exception:
            pass

    def _set_edit_input_visible(self, visible: bool) -> None:
        try:
            inp = self.query_one("#settings-edit-input", Input)
            inp.display = visible
            if not visible:
                inp.value = ""
        except Exception:
            pass

    def _toggle_row_visibility(self) -> None:
        try:
            ol = self._list()
            if ol is not None:
                ol.display = not self.editing
        except Exception:
            pass

    # -- rendering ---------------------------------------------------------
    def _render_settings(self) -> None:
        if not self.is_attached:
            return
        self._populate()

    def _after_change(self) -> None:
        # persist the merged config so changes survive a restart
        try:
            save_config(self.cfg)
        except Exception:
            pass
        # let the app push startup-captured values into the RUNNING components
        # (bash caps, rotation timeouts) instead of waiting for a restart
        if self.on_apply is not None:
            try:
                self.on_apply()
            except Exception:
                pass
        self._populate()
        ol = self._list()
        if ol is not None:
            try:
                ol.focus()
            except Exception:
                pass

    # -- navigation / editing ---------------------------------------------
    def _move(self, delta: int) -> None:
        if self.editing:
            return
        n = len(self.rows)
        if n == 0:
            return
        new = self.index
        steps = 0
        moved = False
        while steps < n:
            new = (new + delta) % n
            if self.rows[new].kind != "info":
                self.index = new
                moved = True
                break
            steps += 1
        if not moved:
            return
        ol = self._list()
        if ol is None:
            return
        # refresh highlight text (values unchanged) then follow it
        try:
            ol.highlighted = self.index
            ol.scroll_to_highlight()
        except Exception:
            pass
        self._update_count()

    def _activate(self) -> None:
        if self.editing:
            return self._commit_edit()
        row = self._row
        if self.editing:
            return
        if row.kind in ("model",):
            self._open_model_picker(row.apply, propagate=row.propagate)
        elif row.kind == "bool":
            cur = row.get() == "yes"
            if row.apply:
                row.apply("no" if cur else "yes")
            self._after_change()
        elif row.kind == "enum":
            self._cycle(1)
        elif row.kind in ("int", "string"):
            self._begin_edit()
        # info rows do nothing

    def _cycle(self, delta: int) -> None:
        row = self._row
        if row.kind != "enum" or not row.choices:
            return
        choices = row.choices
        try:
            cur = choices.index(row.get())
        except ValueError:
            cur = -1
        nxt = (cur + delta) % len(choices)
        if row.apply:
            row.apply(choices[nxt])
        self._after_change()
        try:
            self.app.notify(f"{row.label}: {row.get()}")
        except Exception:
            pass

    def _begin_edit(self) -> None:
        row = self._row
        if row.kind not in ("int", "string"):
            return
        self.editing = True
        self._edit_row = row
        try:
            inp = self.query_one("#settings-edit-input", Input)
            inp.value = row.get()
            inp.display = True
            self._toggle_row_visibility()
            inp.focus()
        except Exception:
            pass

    def _commit_edit(self) -> None:
        row = self._edit_row
        if row and row.apply is not None:
            try:
                value = self.query_one("#settings-edit-input", Input).value.strip()
            except Exception:
                value = ""
            try:
                row.apply(value)
            except (ValueError, TypeError):
                try:
                    self.app.notify(f"Invalid value: {value!r}", severity="warning")
                except Exception:
                    pass
        self._cancel_edit()
        self._after_change()

    def _cancel_edit(self) -> None:
        self.editing = False
        self._edit_row = None
        self._set_edit_input_visible(False)
        self._toggle_row_visibility()
        ol = self._list()
        if ol is not None:
            try:
                ol.focus()
            except Exception:
                pass

    def _open_model_picker(self, apply: Callable[[str], None] | None, propagate: bool = True) -> None:
        def on_picked(choice: str | None) -> None:
            ol = self._list()
            if not choice or not apply:
                if ol is not None:
                    try:
                        ol.focus()
                    except Exception:
                        pass
                return
            provider, _, model = choice.partition("/")
            if provider and model:
                self.cfg.provider = provider
            apply(model)
            # only a real model pick on a row that owns the app model should
            # propagate to the app engine (the "small model" row must not)
            if propagate and self.on_model_change:
                self.on_model_change(model)
            self._after_change()

        self.app.push_screen(
            ModelPicker(current=self.cfg.model, cfg=self.cfg, auth=self.auth),
            on_picked,
        )

    # -- events ------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "settings-model":
            self._open_model_picker(lambda v: setattr(self.cfg, "model", v))
        elif bid == "settings-save":
            self._after_change()
            try:
                self.app.notify("Settings saved.")
            except Exception:
                pass
        elif bid == "settings-close":
            self.dismiss(None)

    def on_option_list_option_selected(self, event: Any) -> None:
        try:
            opt = getattr(event, "option", None)
            oid = str(getattr(opt, "id", "") or "")
        except Exception:
            return
        if oid.startswith("__hdr__"):
            try:
                event.stop()
            except Exception:
                pass
            return
        if oid.startswith("srow-"):
            try:
                self.index = int(oid[len("srow-"):])
            except Exception:
                pass
            try:
                event.stop()
            except Exception:
                pass
            self._activate()

    def on_key(self, event: Key) -> None:
        key = event.key
        if key == "escape":
            if self.editing:
                self._cancel_edit()
            else:
                self.dismiss(None)
            event.stop()
            return
        if self.editing:
            # only Enter commits / Esc cancels; everything else goes to the input
            if key == "enter":
                self._commit_edit()
                event.stop()
            return
        if key in ("down", "j"):
            self._move(1)
            event.stop()
        elif key in ("up", "k"):
            self._move(-1)
            event.stop()
        elif key == "enter" or key == "space":
            self._activate()
            event.stop()
        elif key in ("right", "left"):
            self._cycle(1 if key == "right" else -1)
            event.stop()
