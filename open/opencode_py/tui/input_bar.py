"""Input bar: opencode-style prompt with agent-colored accent, meta row, and
a busy spinner line, plus /command autocomplete + arrow-key navigation.

The prompt itself is a growing multi-line textarea (mirrors opencode's
`<textarea>` prompt component): long lines wrap and the box grows up to a
max-height cap instead of scrolling horizontally, so a long message is never
hidden behind the accent strip or the edge of the screen.

  - a solid 1-cell left accent strip in the current agent color
  - the input on the backgroundElement colour
  - a muted meta row underneath: `Agent auto · model provider`
  - a status line with an animated block spinner while the engine is busy

Type `/` and a dropdown of matching slash commands appears below the input.
Arrow keys move through it; Enter opens a centered CommandPopup for the
selected command. Enter submits; Shift+Enter inserts a newline. Up/Down move
the cursor through multi-line text, and fall back to prompt history at the
first/last line. Tab with an empty input toggles the agent.
"""

from __future__ import annotations

import math
import time
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.events import Key
from textual.message import Message
from textual.style import Style as ContentStyle
from textual.strip import Strip
from textual.widgets import Static, TextArea

from .theme import active_theme

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def format_duration(seconds: float) -> str:
    """Turn runtime, mirroring opencode's `Locale.duration`:
    `312ms`, `12.5s`, `1m 12s`, `1h 5m`, `2d 3h`."""
    ms = int(round(seconds * 1000))
    if ms < 1000:
        return f"{ms}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    if ms < 3600000:
        minutes = ms // 60000
        secs = (ms % 60000) // 1000
        return f"{minutes}m {secs}s"
    if ms < 86400000:
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        return f"{hours}h {minutes}m"
    days = ms // 86400000
    hours = (ms % 86400000) // 3600000
    return f"{days}d {hours}h"


class PromptSubmitted(Message):
    """User pressed Enter with a prompt."""

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


class AgentToggleRequested(Message):
    """User pressed Tab with an empty input; cycle agent."""

    def __init__(self) -> None:
        super().__init__()


class ModelsRequested(Message):
    """User pressed Ctrl+M; open the picker."""

    def __init__(self) -> None:
        super().__init__()


class RotationLockToggled(Message):
    """User clicked the model lock dot in the meta row; flip rotation on/off."""

    def __init__(self) -> None:
        super().__init__()


class CommandSelected(Message):
    """A slash-command suggestion was chosen; open its centered popup."""

    def __init__(self, name: str, description: str) -> None:
        super().__init__()
        self.name = name
        self.description = description


class SessionNavRequested(Message):
    """Arrow keys with an empty prompt: let the app route the key to the
    session navigation (`↑` parent, `←`/`→` siblings) or, when there is no
    navigation target, back to prompt-history recall."""

    def __init__(self, direction: str) -> None:
        super().__init__()
        self.direction = direction  # "up" | "down" | "left" | "right"


class PromptTextArea(TextArea):
    """Multi-line prompt input.

    Long lines soft-wrap and the widget grows up to a cap (like opencode's
    `<textarea minHeight=1 maxHeight=...>`), so the prompt never scrolls out of
    view. Enter submits; Shift+Enter inserts a newline. Up/Down are delegated to
    the parent InputBar while command suggestions or prompt history apply.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            show_cursor=True,
            tab_behavior="focus",
            **kwargs,
        )
        self._delegate_arrow: Any = None
        # Time of the last key event seen by _on_key, used to detect a paste
        # flowing through as raw key events (Termux/Android IME pastes often
        # bypass bracketed-paste mode, so Textual delivers each pasted newline
        # as an `enter` key instead of one Paste event). Pasted characters
        # arrive back-to-back (microseconds-to-ms gaps, all read from the
        # terminal in one burst); a deliberate Enter press comes well after the
        # user stops typing (humans are never faster than ~30ms between keys,
        # so 25ms cleanly separates the two).
        self._last_key_mono = 0.0
        self._last_key = ""
        self._paste_flow_gap = 0.025

    # -- compatibility with the old single-line Input ---------------------
    # The rest of the app (and the tests) talk to `bar.input.value` and
    # `bar.input.cursor_position`; map those onto the textarea.
    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.text = text
        self.resize_to_content()

    @property
    def cursor_position(self) -> int:
        return self.document.get_index_from_location(self.selection.end)

    @cursor_position.setter
    def cursor_position(self, index: int) -> None:
        self.move_cursor(self.document.get_location_from_index(index))

    # -- sizing ------------------------------------------------------------
    def _wrapped_lines(self) -> int:
        """Approximate number of visual lines once soft-wrapped to the width.

        Falls back to the screen width while the text area hasn't been given a
        concrete size yet (mounting / tests).
        """
        w = self.size.width or self.app.size.width
        width = max(1, w - 2)
        total = 0
        for line in self.text.split("\n"):
            if not line:
                total += 1
            else:
                total += max(1, math.ceil(cell_len(line) / width))
        return total

    def _max_height(self) -> int:
        # mirrors opencode: max(6, terminal height / 3)
        return max(6, self.app.size.height // 3)

    def resize_to_content(self) -> None:
        # min height 3 keeps the original prompt box look; grows as the prompt
        # wraps to multiple lines (like opencode's textarea). Skips the
        # full-app relayout when the line count didn't change — typing
        # within a line then costs nothing.
        try:
            lines = self._wrapped_lines()
            if lines == getattr(self, "_last_wrapped_lines", None):
                return
            self._last_wrapped_lines = lines
            self.styles.height = max(3, min(lines, self._max_height()))
        except Exception:
            try:
                self.styles.height = max(3, min(self._wrapped_lines(), self._max_height()))
            except Exception:
                pass

    def render_line(self, y: int) -> Strip:
        """Vertically center the placeholder in the (taller) prompt box.

        Textual's TextArea renders the placeholder at the top row and ignores
        `content-align`, so without this override "Ask anything..." hugs the top
        of the 3-row box instead of sitting in the middle of it.
        """
        if not self.text and self.placeholder:
            placeholder_lines = Content.from_text(self.placeholder).wrap(self.content_size.width)
            offset = max(0, (self.size.height - len(placeholder_lines)) // 2)
            idx = y - offset
            if 0 <= idx < len(placeholder_lines):
                style = self.get_visual_style("text-area--placeholder")
                content = placeholder_lines[idx].stylize(style)
                if self._draw_cursor and idx == 0:
                    theme = self._theme
                    cursor_style = theme.cursor_style if theme else None
                    if cursor_style:
                        content = content.stylize(ContentStyle.from_rich_style(cursor_style), 0, 1)
                return Strip(content.render_segments(self.visual_style), content.cell_length)
            return Strip.blank(self.size.width, self.visual_style.rich_style)
        return super().render_line(y)

    # -- keys --------------------------------------------------------------
    async def _on_key(self, event: Key) -> None:
        key = event.key
        now = time.monotonic()
        in_paste_flow = now - self._last_key_mono < self._paste_flow_gap
        prev_key = self._last_key
        self._last_key_mono = now
        self._last_key = key
        if key == "ctrl+m":
            # Ctrl+M opens the model picker (most terminals report a real
            # Ctrl+M as "enter" instead — the empty-enter branch below
            # opens the picker too, so both paths land here).
            event.stop()
            event.prevent_default()
            self.post_message(ModelsRequested())
            return
        if key == "enter":
            # plain Enter submits (mirrors opencode's onSubmit). An empty
            # Enter opens the model picker — Ctrl+M arrives as this same
            # byte in most terminals, so this IS the Ctrl+M path. But when
            # the key arrives inside a paste flowing through as raw key
            # events (Termux IME paste, no bracketed-paste markers), it's
            # a newline inside the pasted text — insert it instead.
            event.stop()
            event.prevent_default()
            if in_paste_flow:
                self._replace_via_keyboard("\n", *self.selection)
                return
            if not self.text.strip():
                self.post_message(ModelsRequested())
                return
            self.post_message(PromptSubmitted(self.text))
            return
        if key in ("shift+enter", "ctrl+enter", "alt+enter", "ctrl+j"):
            # opencode binds these to "insert newline"
            event.stop()
            event.prevent_default()
            if key == "ctrl+j" and in_paste_flow and prev_key == "enter":
                # the LF half of a CRLF line break inside a raw paste — the
                # CR already inserted the newline, skip the duplicate
                return
            self._replace_via_keyboard("\n", *self.selection)
            return
        if key in ("up", "down", "left", "right"):
            if not self.text.strip():
                # Empty prompt: `↑` goes to the parent session (or, with no
                # parent, recalls the previous prompt), `←`/`→` cycle the
                # parallel sub-agent siblings and `↓` recalls the next prompt —
                # official's session scope wins over the input scope while the
                # prompt isn't being edited. Route through the app so it can
                # decide between session navigation and history recall.
                event.stop()
                self.post_message(SessionNavRequested(key))
                return
            handler = self._delegate_arrow
            if handler is not None and handler(key):
                event.stop()
                return
        await super()._on_key(event)


class ModelMeta(Static):
    """Meta row under the input box; clicking the model dot toggles rotation."""

    def on_click(self, event: Any) -> None:
        self.post_message(RotationLockToggled())
        event.stop()


class InputBar(Vertical):
    """Prompt input with a left accent strip, meta row and status line."""

    def __init__(self, commands: list[dict[str, str]] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._history: list[str] = []
        self._hist_index: int = -1
        self._draft = ""
        self.commands = commands or []
        self._suggestions: list[str] = []
        self._sel: int = 0
        self._navigated = False
        # Set while _handle_arrow rewrites the input to the highlighted command:
        # that programmatic change must NOT re-run prefix matching, or the
        # candidate list collapses to the selected item and the arrows get
        # stuck (the menu you navigate is filtered by the selection itself).
        self._nav_programmatic = False
        self._busy = False
        self._spinner = 0
        self._timer: Any = None
        self._agents: list[str] = []
        self._compacting = False
        self._status_msg = ""
        self._dots = 0
        self.agent = "build"
        self.model = ""
        self.provider = ""
        self.permission_mode = "auto"
        self.rotation_locked = False
        self.reasoning_effort = ""
        self.last_duration = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="suggestions", classes="suggestions hidden"):
            yield Static("", id="suggestions-content")
        yield Static("", id="prompt-title")
        with Horizontal(classes="prompt-frame"):
            yield Static("", id="prompt-accent")
            with Vertical(classes="prompt-body"):
                yield PromptTextArea(placeholder="Ask anything...", id="prompt-input")
                yield ModelMeta("", id="prompt-meta")
        yield Static("", id="prompt-status")

    def on_mount(self) -> None:
        prompt = self.query_one(PromptTextArea)
        prompt._delegate_arrow = self._handle_arrow
        prompt.resize_to_content()
        self._sync_accent_height()
        prompt.focus()

    @property
    def input(self) -> PromptTextArea:
        return self.query_one("#prompt-input", PromptTextArea)

    def _sync_accent_height(self) -> None:
        """Keep the left accent strip the same height as the growing input."""
        try:
            prompt = self.input
            self.query_one("#prompt-accent", Static).styles.height = prompt.styles.height
        except Exception:
            pass

    # -- state --------------------------------------------------------------
    def set_header(
        self,
        *,
        agent: str,
        model: str,
        provider: str,
        permission_mode: str,
        rotation_locked: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        theme = active_theme()
        accent = theme.agent_color(agent)
        self.query_one("#prompt-accent", Static).styles.background = accent
        self.agent = agent
        self.model = model
        self.provider = provider
        self.permission_mode = permission_mode
        if reasoning_effort is not None:
            self.reasoning_effort = str(reasoning_effort or "")
        if rotation_locked is not None:
            self.rotation_locked = bool(rotation_locked)
        self._render_title()
        self._render_meta()

    def set_rotation_locked(self, locked: bool) -> None:
        """Flip the model-lock state and re-render the meta row's lock dot."""
        self.rotation_locked = bool(locked)
        self._render_meta()

    def _render_title(self) -> None:
        """Fixed `▣ Build · <picked model> · 1m 12s` line at the top of the prompt box.

        The trailing runtime mirrors opencode's per-message footer (`▣ build ·
        model · 1m 12s`), shown here after a turn finishes.
        """
        theme = active_theme()
        accent = theme.agent_color(self.agent)
        t = self.query_one("#prompt-title", Static)
        rich = Text()
        rich.append("▣ ", style=accent)
        rich.append(self.agent.title(), style=theme.c("text"))
        if self.model:
            rich.append(f" · {self.model}", style=theme.c("text_muted"))
        if self.last_duration:
            rich.append(f" · {self.last_duration}", style=theme.c("text_muted"))
        t.update(rich)

    def set_last_duration(self, duration: str) -> None:
        """Show the just-finished turn's runtime (`1m 12s`) on the mode line."""
        self.last_duration = duration or ""
        try:
            self._render_title()
        except Exception:
            pass

    def _render_meta(self) -> None:
        theme = active_theme()
        accent = theme.agent_color(self.agent)
        t = self.query_one("#prompt-meta", Static)
        label = self.agent.title()
        parts: list[tuple[str, str]] = [
            (label, f"bold {accent}"),
        ]
        if self.permission_mode == "auto":
            parts.append((" auto", theme.c("text_muted")))
        elif self.permission_mode == "fully_auto":
            parts.append((" fully_auto", theme.c("success")))
        parts.append((" ·", theme.c("text_muted")))
        if self.model:
            # lock dot before the model: red = pinned (no rotation), green =
            # unlocked (may fail over). Clicking the meta row toggles it.
            lock_color = theme.c("error") if self.rotation_locked else theme.c("success")
            parts.append(("• ", lock_color))
            parts.append((f"{self.model}", theme.c("text")))
            # official parity: active reasoning effort rides with the model
            # name in warning color (`Muse Spark · opencode high`).
            if getattr(self, "reasoning_effort", ""):
                parts.append((f" {self.reasoning_effort}", theme.c("warning")))
        if self.provider:
            parts.append((f" {self.provider}", theme.c("text_muted")))
        rich = Text()
        for text, style in parts:
            rich.append(text, style=style)
        t.update(rich)

    def set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        if busy and message:
            self._status_msg = message
        if not busy:
            self._status_msg = ""
        status = self.query_one("#prompt-status", Static)
        if busy:
            status.remove_class("hidden")
            if self._timer is None:
                self._timer = self.set_interval(0.08, self._tick_spinner)
            self._update_status_line(message)
        else:
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            status.update("")
            status.add_class("hidden")

    def set_compacting(self, compacting: bool) -> None:
        """Show the official opencode `Compacting conversation…` status while
        the session summarizes history (overrides the generic working… text)."""
        self._compacting = compacting
        if self._busy:
            self._update_status_line()

    def set_status(self, message: str = "") -> None:
        try:
            self._status_msg = message or ""
        except Exception:
            pass
        if self._busy:
            self._update_status_line()

    def set_running_agents(self, agents: list[str]) -> None:
        """Transient list of launched sub-agents shown in the status line while
        they run (opencode's `Delegating...` indicator)."""
        self._agents = agents
        if self._busy:
            self._update_status_line()

    def _tick_spinner(self) -> None:
        # Adaptive pace (same rule as chat bubbles): fresh stream activity ->
        # full spin; stalled net -> held frame. note_stream_activity stamps
        # _last_token_mono from ChatView appends via set_busy_tick.
        try:
            last_tok = float(getattr(self, "_last_token_mono", 0.0) or 0.0)
            if last_tok > 0.0:
                import time as _time

                age = _time.monotonic() - last_tok
                if age > 1.5:
                    return
                if age > 0.5:
                    self._tick_slow = not getattr(self, "_tick_slow", False)
                    if self._tick_slow:
                        return
        except Exception:
            pass
        self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
        self._dots = (int(getattr(self, "_dots", 0)) + 1) % 3
        self._update_status_line()

    def note_stream_activity(self, n_chars: int = 0) -> None:
        """Stamp token arrival so the busy spinner paces real speed."""
        try:
            import time as _time

            self._last_token_mono = _time.monotonic()
        except Exception:
            pass

    def _update_status_line(self, message: str = "") -> None:
        if not self._busy:
            return
        theme = active_theme()
        accent = theme.agent_color(self.agent)
        frame = SPINNER_FRAMES[self._spinner]
        if getattr(self, "_compacting", False):
            text = "Compacting conversation…"
        elif self._agents:
            text = self._agents[0]
            if len(self._agents) > 1:
                text += f" +{len(self._agents) - 1} more"
        else:
            base = message or getattr(self, "_status_msg", "") or "working..."
            try:
                base = str(base)
            except Exception:
                base = "working..."
            low = base.lower()
            if low.startswith("working") or low.startswith("reading") or low.startswith("testing") or low.startswith("editing") or low.startswith("fetching") or low.startswith("searching") or low.startswith("delegating") or low.startswith("running"):
                core = base.rstrip(".").rstrip()
                try:
                    n = (int(getattr(self, "_dots", 0)) % 3) + 1
                except Exception:
                    n = 3
                text = f"{core}{'.' * n}"
            else:
                text = base
        self.query_one("#prompt-status", Static).update(
            f"[{accent}]{frame}[/] [dim]{text}[/]"
        )

    # -- suggestions ------------------------------------------------------
    def _suggestion_box(self) -> VerticalScroll:
        return self.query_one("#suggestions", VerticalScroll)

    def _suggestion_content(self) -> Static:
        return self.query_one("#suggestions-content", Static)

    def _update_suggestions(self, value: str) -> None:
        if not value.startswith("/"):
            self._clear_suggestions()
            return
        query = value[1:].lower()
        matches = [
            c["name"]
            for c in self.commands
            if not c.get("hidden") and c["name"].lower().startswith(query)
        ]
        if not matches:
            self._clear_suggestions()
            return
        self._suggestions = matches
        # a freshly typed/edited query restarts the highlight at the top
        self._sel = 0
        self._render_suggestions()

    def _render_suggestions(self) -> None:
        """Redraw the dropdown from the CURRENT candidate list + selection.

        Used both after user typing (via _update_suggestions) and after an
        arrow move (which rewrites the input text but must keep the original
        candidates so the highlight can actually travel between them).

        The dropdown is a SCROLLABLE viewport (VerticalScroll, max-height 10):
        every candidate is rendered, and the view auto-scrolls so the ● row is
        always on screen. Reaching /review at the end really scrolls there."""
        matches = self._suggestions
        if not matches:
            return
        if self._sel >= len(matches):
            self._sel = 0
        box = self._suggestion_box()
        content = self._suggestion_content()
        # Thin scrollbar rail. (App CSS declares it too, but this Textual
        # version's cascade drops the rule for scroll containers — setting
        # the style directly is the reliable way.)
        try:
            if box.styles.scrollbar_size_vertical != 1:
                box.styles.scrollbar_size_vertical = 1
        except Exception:
            pass
        theme = active_theme()
        desc_by_name = {
            c["name"]: (c.get("description") or "").strip() for c in self.commands
        }
        lines = []
        for i, name in enumerate(matches):
            if i == self._sel:
                style = f"bold {theme.c('accent')}"
                marker = "● "
            else:
                style = theme.c("text")
                marker = "  "
            line = f"[{style}]{marker}/{name}[/]"
            # dim one-line briefing so the eye stays on the command names
            desc = desc_by_name.get(name, "")
            if desc:
                if len(desc) > 48:
                    desc = desc[:47].rstrip() + "…"
                line += f"  [{theme.c('text_muted')}]{desc}[/]"
            lines.append(line)
        content.update("\n".join(lines))
        box.remove_class("hidden")
        box.add_class("visible")
        # keep the highlighted row inside the viewport (one line per command)
        try:
            vh = box.size.height or 1
            top = int(box.scroll_y)
            if self._sel < top:
                box.scroll_to(y=self._sel, animate=False)
            elif self._sel >= top + vh:
                box.scroll_to(y=self._sel - vh + 1, animate=False)
        except Exception:
            pass

    def _clear_suggestions(self) -> None:
        self._suggestions = []
        self._navigated = False
        self._sel = 0
        self._nav_programmatic = False
        try:
            self._suggestion_content().update("")
        except Exception:
            pass
        box = self._suggestion_box()
        box.remove_class("visible")
        box.add_class("hidden")

    def _has_command(self, name: str) -> bool:
        return any(c["name"] == name for c in self.commands)

    def _select_name(self, name: str) -> None:
        self._suggestions = [name]
        self._sel = 0

    def _show_popup(self) -> None:
        name = self._suggestions[self._sel]
        desc = next((c["description"] for c in self.commands if c["name"] == name), "")
        self._clear_suggestions()
        self.input.value = ""
        self.post_message(CommandSelected(name, desc))

    # -- events -----------------------------------------------------------
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self._nav_programmatic:
            # value was set by _handle_arrow (highlight move): the candidate
            # list must stay as-is, so skip re-filtering entirely.
            self._nav_programmatic = False
        else:
            self._update_suggestions(event.text_area.text)
        event.text_area.resize_to_content()
        self._sync_accent_height()
        event.stop()

    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        value = event.value
        if not value.strip():
            event.stop()
            return
        if value.strip() == "/":
            # A lone "/" is command browsing, not a prompt — submitting it
            # used to send the literal "/" to the model as a user message.
            event.stop()
            return
        if value.startswith("/") and len(value.strip()) > 1:
            parts = value[1:].split(maxsplit=1)
            name = parts[0] if parts else ""
            has_args = len(parts) == 2 and bool(parts[1].strip())
            if self._suggestions:
                # dropdown active -> centered popup for the highlighted command
                event.stop()
                self._show_popup()
                return
            if self._has_command(name) and not has_args:
                # bare known command (e.g. "/models") -> popup with its output
                self._select_name(name)
                event.stop()
                self._show_popup()
                return
        self._navigated = False
        self.input.value = ""
        # a submitted prompt ends any draft from an earlier ↓-clear: the next
        # ↑ must browse only what was really sent, never the replaced text.
        self._draft = ""
        if value.strip():
            self._history.append(value)
            self._hist_index = len(self._history)
        # not stopped: the PromptSubmitted continues up to the app

    def recall_history(self, key: str) -> bool:
        """Empty-prompt fallback for ↑/↓: recall previous/next prompt (or restore
        the typed draft), mirroring opencode's input-scope history bindings."""
        if key == "up":
            if self._history and self.input.cursor_at_first_line:
                if self._hist_index == len(self._history) and not self.input.value and self._draft:
                    # the final ↓ cleared the box: come back to what we were
                    # typing (index stays at the end so the next ↑ walks into
                    # the recent history).
                    self.input.value = self._draft
                    self.input.cursor_position = len(self._draft)
                elif self._hist_index > 0:
                    if self._hist_index == len(self._history):
                        self._draft = self.input.value
                    self._hist_index -= 1
                    self.input.value = self._history[self._hist_index]
                    self.input.cursor_position = len(self.input.value)
                return True
            return False
        if key == "down":
            n = len(self._history)
            # walking forward through older→newer entries is always allowed
            if self._history and 0 <= self._hist_index < n - 1:
                self._hist_index += 1
                self.input.value = self._history[self._hist_index]
                self.input.cursor_position = len(self.input.value)
                return True
            # a long / multi-line prompt keeps ↓ for cursor movement to the
            # next line; clearing only applies on a short single line or once
            # the cursor sits at the very end of the last line (nothing below).
            multi_line = self.input._wrapped_lines() > 1
            can_move_down = (
                self.input.get_cursor_down_location() != self.input.cursor_location
            )
            if multi_line and can_move_down:
                return False
            # single line, or at the end of the last line: ↓ clears the whole
            # prompt so it can be replaced at once. The cleared text is saved
            # as the draft, so a following ↑ brings it straight back — unless a
            # new prompt has been submitted since (the submit discards it).
            if self._hist_index == -1 or 0 <= self._hist_index <= n:
                if self.input.value == "" and self._hist_index >= n:
                    return False  # already empty — nothing to clear
                if (self._hist_index == -1 or self._hist_index == n) and self.input.value:
                    # clearing the user's own freshly-typed prompt: remember it
                    # so ↑ brings it back. While browsing a history entry
                    # (index < n) the draft keeps the original typed text.
                    self._draft = self.input.value
                self._hist_index = n
                self.input.value = ""
                self.input.cursor_position = 0
                return True
            return False
        return False

    def _handle_arrow(self, key: str) -> bool:
        """Consume Up/Down for suggestions / history; False = let the cursor move."""
        if self._suggestions:
            n = len(self._suggestions)
            self._sel = (self._sel + (1 if key == "down" else -1)) % n
            self._navigated = True
            # Rewrite the input to the highlighted command, but mark the change
            # as programmatic: re-filtering on it collapsed the dropdown to a
            # single (stuck) item — the bug that made arrows useless here.
            self._nav_programmatic = True
            self.input.value = f"/{self._suggestions[self._sel]}"
            self.input.cursor_position = len(self.input.value)
            self._render_suggestions()
            return True
        if key in ("up", "down"):
            # typing a non-empty prompt: arrows stay with the input (history
            # recall / cursor movement), never the session navigation
            return self.recall_history(key)
        return False

    def on_key(self, event: Key) -> None:
        if event.key == "tab" and not self.input.value:
            self.post_message(AgentToggleRequested())
            event.stop()
        elif event.key == "escape" and self._suggestions:
            self._clear_suggestions()
            event.stop()
