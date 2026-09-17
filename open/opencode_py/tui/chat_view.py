"""Chat view: scrollable message list mirroring opencode's session screen.

Rendering mirrors opencode's TUI (packages/tui/src/routes/session/index.tsx):

  - User messages are a full-width block with a single left border strip in the
    agent accent color, a `backgroundPanel` fill and padding (no title), plus an
    optional ` QUEUED ` badge when the turn hasn't started yet.
  - Assistant text flows as plain markdown indented from the left with a block
    cursor (▍) while streaming, then a muted `▣ Build · model` mode line.
  - Reasoning streams as a spinner `Thinking...` and collapses to a clickable
    `+ Thought: <title>` line (opencode's hide mode); clicking toggles the body.
  - Tools render as compact inline rows (`{icon} {label}`, spinner while running)
    or, for tools that produce a result block (bash output, edit diff, todos,
    questions, apply-patch), a subtle left-bordered block on the panel background.
    Per-tool rendering mirrors opencode: Read shows `↳ Loaded <file>`, Glob/Grep
    show `(N matches)`, etc.
"""

from __future__ import annotations

import re
import time
from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from .theme import active_theme
from .logo import OPENCODE_LOGO, opencode_logo_text
from .markdown_renderer import render_markdown
from .diff_renderer import render_diff
from .input_bar import SessionNavRequested

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.08  # opencode uses 80ms


class OpenTaskSession(Message):
    """A completed task tool row was clicked; open that sub-session."""

    def __init__(self, sid: str) -> None:
        super().__init__()
        self.sid = sid

# tool -> icon. Mirrors opencode's InlineTool usage.
TOOL_ICONS = {
    "bash": "$",
    "shell": "$",
    "execute": "$",
    "background_task": "$",
    "read": "→",
    "write": "←",
    "glob": "✱",
    "grep": "✱",
    "find_symbols": "⎆",
    "screen_view": "◉",
    "device": "⚡",
    "history_search": "◷",
    "checkpoint": "⟲",
    "webfetch": "✦",
    "webfetch_many": "⟡",
    "websearch": "◈",
    "edit": "←",
    "apply_patch": "✦",
    "todowrite": "☰",
    "task": "│",
    "question": "→",
    "skill": "→",
    "lsp": "⌁",
    "speak": "♪",
    "mcp": "⊙",
    "notify": "·",
}
# tool -> label used in inline rows / titles.
TOOL_NAMES = {
    "bash": "Shell",
    "shell": "Shell",
    "execute": "Execute",
    "background_task": "Background",
    "read": "Read",
    "write": "Write",
    "glob": "Glob",
    "grep": "Grep",
    "find_symbols": "Symbols",
    "screen_view": "Screen",
    "device": "Device",
    "history_search": "History",
    "checkpoint": "Checkpoint",
    "webfetch": "WebFetch",
    "webfetch_many": "WebFetch Batch",
    "websearch": "WebSearch",
    "edit": "Edit",
    "apply_patch": "Apply Patch",
    "todowrite": "TodoWrite",
    "task": "Task",
    "skill": "Skill",
    "lsp": "LSP",
    "speak": "Speak",
}

# tools that have a dedicated renderer in opencode (everything else is "generic").
_TOOL_DISPLAYS = {
    "bash",
    "glob",
    "read",
    "grep",
    "find_symbols",
    "webfetch",
    "webfetch_many",
    "websearch",
    "write",
    "edit",
    "task",
    "apply_patch",
    "todowrite",
    "question",
    "skill",
    "execute",
    "lsp",
}


def _plain(content: Any, width: int | None = None) -> RenderableType:
    """Plain flowing markdown with no surrounding box (assistant text)."""
    return render_markdown(str(content), width=width)


def _render_diff(
    diff_text: str,
    filepath: str = "",
    width: int | None = None,
    opts: dict[str, Any] | None = None,
) -> RenderableType:
    """Render a unified diff the way opencode's `<diff>` edit block does.

    Includes a line-number gutter, +/- signs and syntax-highlighted content
    (all matched to the official opencode dark theme). ``opts`` mirrors the
    official diff config: ``diff_style`` (``"split"``/``"stacked"``),
    ``diff_wrap_mode`` (``"word"``/``"none"``) and ``suppress_backgrounds``.
    """
    opts = opts or {}
    style = opts.get("diff_style", "split")
    # opencode chooses split only when the terminal is wider than 120 cols and
    # the diff_style config has not forced "stacked".
    view = "auto" if style == "split" else "unified"
    return render_diff(
        diff_text,
        filename=filepath,
        view=view,
        width=width or 0,
        wrap=opts.get("diff_wrap_mode", "word"),
        suppress_backgrounds=opts.get("suppress_backgrounds", False),
    )


def collapse_tool_output(output: str, max_lines: int, max_chars: int) -> dict:
    """Mirror opencode's collapse-tool-output: cap lines and chars with '…'."""
    lines = output.split("\n")
    if len(lines) <= max_lines and len(output) <= max_chars:
        return {"output": output, "overflow": False}
    preview = "\n".join(lines[:max_lines])
    if len(preview) > max_chars:
        return {"output": preview[: max(0, max_chars - 1)] + "…", "overflow": True}
    return {"output": "\n".join(lines[:max_lines] + ["…"]), "overflow": True}


def _safe_input(run: dict[str, Any]) -> dict[str, Any]:
    inp = run.get("input") or {}
    if isinstance(inp, dict):
        return inp
    if isinstance(inp, str):
        try:
            import json as _j
            parsed = _j.loads(inp)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _format_input(inp: object, omit: tuple[str, ...] = ()) -> str:
    """Mirror opencode's `input()` helper: `[key=value, key2=value2]`."""
    if isinstance(inp, str):
        try:
            import json as _j
            parsed = _j.loads(inp)
            inp = parsed if isinstance(parsed, dict) else {}
        except Exception:
            return ""
    if not isinstance(inp, dict):
        return ""
    parts = []
    for key, value in inp.items():
        if key in omit:
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    if not parts:
        return ""
    return "[" + ", ".join(parts) + "]"


def format_subagent_title(agent: str, description: str, background: bool = False) -> str:
    """Mirror opencode's ``formatSubagentTitle``:
    ``Build Task — fix the login bug`` (+ ``(background)`` for background agents)."""
    label = str(agent).title() if str(agent).strip() else "General"
    return f"{label} Task{' (background)' if background else ''} — {description}"


def format_subagent_toolcalls(count: int) -> str:
    """Mirror opencode's ``formatSubagentToolcalls``: `1 toolcall` / `3 toolcalls`."""
    return f"{count} toolcall{'s' if count != 1 else ''}"


def format_subagent_retry(attempt: int, message: str) -> str:
    """Mirror opencode's ``formatSubagentRetry``:
    ``Retrying (attempt 2) · <message>``."""
    return f"Retrying (attempt {attempt}) · {message}"


def format_completed_subagent_detail(toolcalls: int, duration: str) -> str:
    """Mirror opencode's ``formatCompletedSubagentDetail`:
    ``3 toolcalls · 12.5s`` (just the duration when no tools ran)."""
    if toolcalls == 0:
        return duration
    if not duration:
        return format_subagent_toolcalls(toolcalls)
    return f"{format_subagent_toolcalls(toolcalls)} · {duration}"


# tool -> display name for the live `↳ Bash npm run test` line under a running
# task row (mirrors opencode's `titlecase(tool)` in the Subagent component).
_TOOL_TITLES = {
    "bash": "Bash",
    "shell": "Bash",
    "execute": "Execute",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "apply_patch": "Patch",
    "glob": "Glob",
    "grep": "Grep",
    "webfetch": "WebFetch",
    "webfetch_many": "WebFetch Batch",
    "websearch": "WebSearch",
    "todowrite": "TodoWrite",
    "task": "Task",
    "question": "Ask",
    "skill": "Skill",
    "lsp": "LSP",
    "mcp": "MCP",
    "notify": "Notify",
}


def tool_title(run: dict[str, Any]) -> str:
    """The title bit of a tool run (opencode's ``state.title``): Bash -> the
    command, Read/Edit/Write -> the file path, Glob/Grep -> the pattern, etc."""
    tool = str(run.get("tool", ""))
    inp = _safe_input(run)
    if tool in ("bash", "shell", "execute"):
        return str(inp.get("command", "")).strip()
    if tool in ("read", "write", "edit"):
        return str(inp.get("filePath", "")).strip()
    if tool == "apply_patch":
        return str(inp.get("command", "") or inp.get("filePath", "")).strip()
    if tool in ("glob", "grep"):
        pattern = str(inp.get("pattern", "")).strip()
        path = str(inp.get("path", "")).strip()
        title = pattern
        if pattern and path:
            title += f" in {path}"
        return title
    if tool == "webfetch":
        return str(inp.get("url", "")).strip()
    if tool == "webfetch_many":
        urls = inp.get("urls") or []
        if isinstance(urls, list) and urls:
            first = str(urls[0])[:80]
            return f"×{len(urls)} {first}"
        return str(inp.get("url", "")).strip()
    if tool == "websearch":
        return str(inp.get("query", "")).strip()
    if tool == "skill":
        return str(inp.get("name", "")).strip()
    if tool == "lsp":
        op = str(inp.get("operation", "")).strip()
        fp = str(inp.get("filePath", "")).strip()
        ln = str(inp.get("line", "")).strip()
        title = op
        if fp:
            title += f" {fp}"
            if ln:
                title += f":{ln}"
        return title.strip()
    return ""


def tool_label(run: dict[str, Any]) -> str:
    """Full `↳ Bash npm run test` style label for a tool run."""
    tool = str(run.get("tool", "?"))
    title = tool_title(run)
    base = _TOOL_TITLES.get(tool, tool)
    return f"{base} {title}".strip() if title else base


def reasoning_summary(text: str) -> dict:
    """Mirror opencode's reasoningSummary: extract a bold `**Title**` block."""
    content = str(text).strip()
    match = re.match(r"^\*\*([^*\n]+)\*\*(?:\r?\n\r?\n|$)", content)
    if not match:
        return {"title": None, "body": content}
    return {"title": match.group(1).strip(), "body": content[match.end():].strip()}


def _tool_display(tool: str) -> str:
    return tool if tool in _TOOL_DISPLAYS else "generic"


class MessageBubble(Static):
    """One chat element (user, assistant, reasoning, mode-line, meta) or a tool run."""

    queued: reactive = reactive(False)  # user message waiting for the turn to start
    streaming: reactive = reactive(False)
    expanded: reactive = reactive(False)  # reasoning body collapsed/expanded
    selected: reactive = reactive(False)  # this task row is the active sub-agent
    frozen: reactive = reactive(False)  # finished bubble: skip re-render (same pixels, ~0 cost)

    def __init__(
        self,
        role: str,
        content: Any = "",
        agent: str = "build",
        queued: bool = False,
        streaming: bool = False,
        directive: str = "",
        **kwargs: Any,
    ) -> None:
        self.role = role
        self.agent = agent
        self._message = content
        # Streaming deltas accumulate here and are joined once per flush
        # (append-only: `"".join(parts)` is O(n) per flush instead of an
        # O(n²) `content + text` rebuild of the growing string).
        self._stream_parts: list[str] = []
        # Optional header for the sub-agent "directive" block: the text of the
        # parent's instruction shown at the very top of a sub-agent's chat.
        self.directive = directive
        self._spinner = 0
        self._timer: Any = None
        self._thought_started: float | None = None
        self._thought_seconds: float | None = None
        super().__init__("", **kwargs)
        self.can_focus = role == "reasoning"
        self.set_reactive(MessageBubble.queued, queued)
        self.set_reactive(MessageBubble.streaming, streaming)
        self._refresh()

    def _diff_opts(self) -> dict[str, Any]:
        try:
            cfg = self.app.cfg
        except Exception:
            cfg = None
        if cfg is not None:
            try:
                return {
                    "diff_style": getattr(cfg, "diff_style", "split"),
                    "diff_wrap_mode": getattr(cfg, "diff_wrap_mode", "word"),
                    "suppress_backgrounds": getattr(cfg, "suppress_backgrounds", False),
                }
            except Exception:
                pass
        return {}

    def watch_queued(self, value: bool) -> None:
        self.unfreeze()
        self._refresh()

    def watch_streaming(self, value: bool) -> None:
        if value:
            self.unfreeze()
        self._refresh()

    def on_unmount(self) -> None:
        """Clean up spinner timer when bubble is removed to prevent timer leaks."""
        self._stop_spinner()

    def watch_expanded(self, value: bool) -> None:
        self.unfreeze()
        self._refresh()

    def watch_selected(self, value: bool) -> None:
        self.unfreeze()
        self._refresh()

    @property
    def content(self) -> Any:
        """Raw payload (assistant text or tool-run dict)."""
        return self._message

    # -- content ----------------------------------------------------------
    def _build_content(self) -> RenderableType:
        theme = active_theme()
        if self.role == "user":
            inner: list[RenderableType] = []
            if self.directive:
                header = Text()
                header.append("▣ ", style=theme.agent_color(self.agent or "build"))
                header.append(self.directive, style=f"bold {theme.c('text')}")
                header.append(" · ", style=theme.c("text_muted"))
                header.append("Directive", style=theme.c("text_muted"))
                inner.append(header)
            inner.append(Text(str(self._message), style=theme.c("user_bubble")))
            if self.queued:
                color = theme.c("queue_badge")
                inner.append(Text(" QUEUED ", style=f"bold {theme.c('background')} on {color}"))
            return Group(*inner)
        if self.role == "assistant":
            width = self.size.width if self.size else None
            text = str(self._message)
            if self.streaming:
                # Plain wrapped text for the ENTIRE streaming phase. Parsing the
                # whole accumulated markdown document on every delta flush was
                # O(reply-length²) and caused visible layout jumps mid-reply.
                # Streaming is append-only plain text; the final message
                # re-renders once as full markdown in end_stream.
                group: RenderableType = Text(text, style=theme.c("text"))
            else:
                group = _plain(self._message, width=width)
            if self.streaming:
                return Group(group, Text("▍", style=theme.c("streaming_cursor")))
            return group
        if self.role == "reasoning":
            return self._build_reasoning()
        if self.role == "assistant_mode":
            # e.g. `▣ Build · model`
            t = Text()
            t.append("▣ ", style=theme.agent_color(self.agent))
            t.append(self.agent.title(), style=theme.c("text"))
            if self._message:
                t.append(f" · {self._message}", style=theme.c("text_muted"))
            return t
        if self.role == "meta":
            return Text(str(self._message), style=theme.c("text_muted"))
        if self.role == "compaction":
            if self.streaming:
                # Live compaction: show the anchored summary as it streams in
                # (plain text + cursor), then re-render as the ` Compaction `
                # divider + markdown in end_compaction_stream.
                theme = active_theme()
                header = Text("▸ Compacted summary", style=theme.c("warning"))
                body = str(self._message or "")
                if body:
                    return Group(header, Text(body, style=theme.c("text")), Text("▍", style=theme.c("primary")))
                return Group(header, Text("▍", style=theme.c("primary")))
            return self._build_compaction()
        return self._render_tool(self._message) if isinstance(self._message, dict) else _plain(self._message, width=self.size.width if self.size else None)

    def _build_compaction(self) -> RenderableType:
        """Centered ` Compaction ` divider (opencode's compaction part,
        i18n `ui.messagePart.compaction`) with the anchored summary beneath —
        rendered as markdown so the headings (`## Objective`, `## Important
        Details`, `## Work State`, `## Next Move`, `## Relevant Files`) get the
        same colored styling as the official opencode summary message."""
        theme = active_theme()
        width = self.size.width if self.size else 80
        title = " Compaction "
        n = max(1, (width - len(title)) // 2)
        divider = Text("─" * n, style=theme.c("border_active"))
        divider.append(title, style=theme.c("border_active"))
        divider.append("─" * (width - len(title) - n), style=theme.c("border_active"))
        parts: list[RenderableType] = [divider]
        summary = str(self._message or "").strip()
        if summary:
            parts.append(render_markdown(summary, width=width))
        return Group(*parts)

    # -- reasoning (mirrors opencode's ReasoningPart, thinking mode "hide") --
    def _build_reasoning(self) -> RenderableType:
        """Collapsed by default: `+ Thought: <title>`. Streaming shows a spinner
        with `Thinking...`; clicking the header toggles the muted markdown body
        (also while the thought is still streaming)."""
        theme = active_theme()
        summary = reasoning_summary(self._message)
        title = summary["title"]
        body = summary["body"]
        prefix = "- " if self.expanded else "+ "

        if self.streaming:
            header = Text(
                f"{prefix}{SPINNER_FRAMES[self._spinner]} Thinking" + (f": {title}" if title else ""),
                style=theme.c("tool_running"),
            )
        else:
            header = Text(prefix, style=theme.c("tool_running"))
            header.append("Thought", style=theme.c("tool_running"))
            if self._thought_seconds is not None:
                header.append(f" for {self._thought_seconds:.1f}s", style=theme.c("thinking_time"))
            if title:
                header.append(f": {title}", style=theme.c("tool_running"))
        parts: list[RenderableType] = [header]
        if self.expanded and body:
            parts.append(Text(body, style=theme.c("text_muted")))
        return Group(*parts)

    # -- tool rendering (mirrors opencode's per-tool components) ----------
    def _render_tool(self, tool_run: dict[str, Any]) -> RenderableType:
        display = _tool_display(tool_run.get("tool", "?"))
        fn = getattr(self, f"_render_{display}", None)
        if fn is None:
            fn = self._render_generic
        return fn(tool_run)

    def _status(self, run: dict[str, Any]) -> str:
        return run.get("status", "pending")

    def _error(self, run: dict[str, Any]) -> str:
        return run.get("error") or ""

    def _denied(self, run: dict[str, Any]) -> bool:
        # permission denials may arrive as an error field or as output text
        err = run.get("error") or (run.get("output") or "")
        return any(
            m in err
            for m in (
                "QuestionRejectedError",
                "rejected permission",
                "permission denied",
                "denied by permission",
                "user dismissed",
                "specified a rule",
            )
        )

    def _failed(self, run: dict[str, Any]) -> bool:
        """Mirror opencode's InlineTool.failed: a real error that isn't a denial."""
        if self._denied(run):
            return False
        if run.get("status") == "error":
            return True
        return bool(run.get("error"))

    def _inline(
        self,
        icon: str,
        pending: str,
        label: str,
        *,
        spinner: bool = False,
        complete: bool | str | None = None,
        color: str | None = None,
    ) -> RenderableType:
        """Mirror opencode's InlineToolRow: ~ pending / spinner running / icon label.

        While running, once the tool input is known (complete truthy) we show the
        actual action (`← Edit /x.py`) instead of `~ Preparing edit...`, matching
        opencode. Colors mirror opencode: running -> primary, completed -> textMuted,
        failed -> error (red), denied -> strikethrough. ``color`` overrides the
        running/completed color (opencode passes ``theme.error`` for a retrying
        sub-agent row).
        """
        theme = active_theme()
        status = self._status(self._message)
        denied = self._denied(self._message)
        failed = self._failed(self._message)
        done = status in ("completed", "error") if complete is None else bool(complete)

        if status == "running":
            if spinner:
                return Text(f"{SPINNER_FRAMES[self._spinner]} {label}", style=color or theme.c("tool_running"))
            if complete:
                return Text(f"{icon} {label}", style=color or theme.c("tool_running"))
            return Text(f"~ {pending}", style=theme.c("text_muted"))
        if not done:
            return Text(f"~ {pending}", style=theme.c("text_muted"))

        main = color or (theme.c("tool_error") if failed else theme.c("tool_success"))
        style = f"{theme.c('tool_denied')} strike" if denied else main
        return Text(f"{icon} {label}", style=style)

    def _error_line(self, run: dict[str, Any]) -> RenderableType | None:
        """A red error line for a failed tool (None when not a real error)."""
        if not self._failed(run) or self._denied(run):
            return None
        err = (self._error(run) or "").strip()
        if not err:
            return None
        return Text(err, style=active_theme().c("tool_error"))

    def _render_bash(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        status = self._status(run)
        command = str((run.get("input") or {}).get("command", ""))
        output = (run.get("output") or "").strip()
        workdir = str((run.get("input") or {}).get("workdir") or "")
        if status == "running":
            return Text(f"{SPINNER_FRAMES[self._spinner]} {command}", style=theme.c("text"))
        if output:
            lines: list[RenderableType] = []
            if workdir and workdir != ".":
                lines.append(Text(f"# Running in {workdir}", style=theme.c("text_muted")))
            lines.append(Text(f"$ {command}", style=theme.c("text")))
            collapsed = self._tool_collapse(run)
            output_text = output if self.expanded else collapsed["output"]
            lines.append(Text(output_text, style=theme.c("text")))
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                lines.append(Text(label, style=theme.c("text_muted")))
            return Group(*lines)
        return self._inline("$", "Writing command...", command)

    def _render_read(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        status = self._status(run)
        filepath = str((run.get("input") or {}).get("filePath", ""))
        loaded = (run.get("metadata") or {}).get("loaded") or []
        extra = _format_input((run.get("input") or {}), omit=("filePath",))
        row = self._inline(
            "→",
            "Reading file...",
            f"Read {filepath}{extra}",
            spinner=status == "running",
            complete=bool(filepath or extra),
        )
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if status == "completed" and loaded:
            sub = [
                Text(f"↳ Loaded {p}", style=theme.c("text_muted"))
                for p in (loaded if isinstance(loaded, list) else [loaded])
            ]
            output = (run.get("output") or "").strip()
            if output:
                collapsed = self._tool_collapse(run)
                body = output if self.expanded else collapsed["output"]
                parts: list[RenderableType] = [row, *sub, Text(body, style=active_theme().c("text"))]
                if collapsed["overflow"]:
                    parts.append(Text("Click to collapse" if self.expanded else "Click to expand", style=active_theme().c("text_muted")))
                return Group(*parts)
            return Group(row, *sub)
        return row

    def _render_glob(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        pattern = str(inp.get("pattern", ""))
        path = str(inp.get("path", "")) if inp.get("path") else ""
        count = (run.get("metadata") or {}).get("count")
        label = f'Glob "{pattern}"'
        if path:
            label += f" in {path}"
        if isinstance(count, int):
            label += f" ({count} {'match' if count == 1 else 'matches'})"
        row = self._inline("✱", "Finding files...", label, complete=bool(pattern or path))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_grep(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        pattern = str(inp.get("pattern", ""))
        path = str(inp.get("path", "")) if inp.get("path") else ""
        count = (run.get("metadata") or {}).get("matches")
        label = f'Grep "{pattern}"'
        if path:
            label += f" in {path}"
        if isinstance(count, int):
            label += f" ({count} {'match' if count == 1 else 'matches'})"
        row = self._inline("✱", "Searching content...", label, complete=bool(pattern or path))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_find_symbols(self, run: dict[str, Any]) -> RenderableType:
        inp = _safe_input(run)
        meta = run.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        query = str(inp.get("query", ""))
        matches = meta.get("matches")
        label = f'Symbols "{query}"' if query else "Symbols"
        if isinstance(matches, int):
            label += f" ({matches} {'match' if matches == 1 else 'matches'})"
        row = self._inline("⎆", "Finding symbols...", label, spinner=self._status(run) == "running", complete=bool(query))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if self._status(run) == "completed":
            output = (run.get("output") or "").strip()
            if output:
                collapsed = self._tool_collapse(run)
                body = output if self.expanded else collapsed["output"]
                parts: list[RenderableType] = [row, Text(body, style=active_theme().c("text"))]
                if collapsed["overflow"]:
                    parts.append(Text("Click to collapse" if self.expanded else "Click to expand", style=active_theme().c("text_muted")))
                return Group(*parts)
        return row

    def _render_webfetch(self, run: dict[str, Any]) -> RenderableType:
        url = str((_safe_input(run)).get("url", ""))
        row = self._inline("✦", "Fetching from the web...", f"WebFetch {url}", spinner=self._status(run) == "running", complete=bool(url))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_webfetch_many(self, run: dict[str, Any]) -> RenderableType:
        inp = _safe_input(run)
        meta = run.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        urls = inp.get("urls") or []
        n = len(urls) if isinstance(urls, list) else 0
        done = meta.get("done", meta.get("progress_done"))
        total = meta.get("total", n or meta.get("count", 0))
        try:
            done_i = int(done) if done is not None else None
            total_i = int(total) if total else n
        except Exception:
            done_i, total_i = None, n
        if self._status(run) == "running":
            if isinstance(done_i, int) and total_i:
                label = f"WebFetch Batch {done_i}/{total_i}"
            elif total_i:
                label = f"WebFetch Batch ×{total_i}"
            else:
                label = "WebFetch Batch"
            row = self._inline("⟡", "Fetching batch...", label, spinner=True, complete=True)
        else:
            ok = meta.get("succeeded")
            fail = meta.get("failed")
            tail = f" ({ok} ok" + (f", {fail} failed" if fail else "") + ")" if isinstance(ok, int) else (f" ×{total_i}" if total_i else "")
            first = str(urls[0])[:60] if isinstance(urls, list) and urls else ""
            label = f"WebFetch Batch{tail} {first}".strip()
            row = self._inline("⟡", "Fetching batch...", label, complete=True)
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if self._status(run) == "completed":
            output = (run.get("output") or "").strip()
            if output:
                collapsed = self._tool_collapse(run)
                body = output if self.expanded else collapsed["output"]
                parts: list[RenderableType] = [row, Text(body, style=active_theme().c("text"))]
                if collapsed["overflow"]:
                    parts.append(Text("Click to collapse" if self.expanded else "Click to expand", style=active_theme().c("text_muted")))
                return Group(*parts)
        return row

    def _render_lsp(self, run: dict[str, Any]) -> RenderableType:
        inp = _safe_input(run)
        meta = run.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        op = str(inp.get("operation", ""))
        fp = str(inp.get("filePath", ""))
        ln = str(inp.get("line", "")) if inp.get("line") else ""
        matches = meta.get("matches", meta.get("issues", ""))
        label = op
        if fp:
            label += f" {fp}"
            if ln:
                label += f":{ln}"
        if isinstance(matches, int):
            label += f" ({matches})"
        row = self._inline("⌁", "Jumping...", label.strip() or "LSP", spinner=self._status(run) == "running", complete=bool(op or fp))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_websearch(self, run: dict[str, Any]) -> RenderableType:
        inp = _safe_input(run)
        meta = run.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        provider = str(meta.get("provider", ""))
        query = str(inp.get("query", ""))
        count = meta.get("numResults", meta.get("numresults"))
        base = f'{provider + " " if provider else ""}WebSearch "{query}"' if query else "WebSearch"
        label = base.strip()
        if isinstance(count, int):
            label += f" ({count} results)"
        row = self._inline("◈", "Searching web...", label.strip() or "WebSearch", spinner=self._status(run) == "running", complete=bool(query))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if self._status(run) == "completed":
            output = (run.get("output") or "").strip()
            if output:
                collapsed = self._tool_collapse(run)
                body = output if self.expanded else collapsed["output"]
                parts: list[RenderableType] = [row, Text(body, style=active_theme().c("text"))]
                if collapsed["overflow"]:
                    parts.append(Text("Click to collapse" if self.expanded else "Click to expand", style=active_theme().c("text_muted")))
                return Group(*parts)
        return row

    def _render_write(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        filepath = str((run.get("input") or {}).get("filePath", ""))
        written = (run.get("metadata") or {}).get("content")
        if self._status(run) == "completed" and written:
            content = str(written)
            collapsed = collapse_tool_output(content, 10, 10 * 80)
            output = content if self.expanded else collapsed["output"]
            rows: list[RenderableType] = [
                Text(f"# Wrote {filepath}", style=theme.c("text_muted")),
                Text(output, style=theme.c("text")),
            ]
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                rows.append(Text(label, style=theme.c("text_muted")))
            return Group(*rows)
        return self._inline("←", "Preparing write...", f"Write {filepath}", complete=bool(filepath))

    def _render_edit(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        filepath = str((run.get("input") or {}).get("filePath", ""))
        diff = (run.get("metadata") or {}).get("diff")
        if self._status(run) == "completed" and diff:
            return Group(
                Text(f"← Edit {filepath}", style=theme.c("text_muted")),
                _render_diff(diff, filepath, self.size.width if self.size else None, self._diff_opts()),
            )
        err = self._error_line(run)
        if err is not None:
            replace_all = _format_input((run.get("input") or {}), omit=("filePath", "oldString", "newString"))
            return Group(
                self._inline("←", "Preparing edit...", f"Edit {filepath}{replace_all}", complete=bool(filepath or replace_all)),
                err,
            )
        replace_all = _format_input((run.get("input") or {}), omit=("filePath", "oldString", "newString"))
        return self._inline("←", "Preparing edit...", f"Edit {filepath}{replace_all}", complete=bool(filepath or replace_all))

    def _render_apply_patch(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        files = (run.get("metadata") or {}).get("files") or []
        if self._status(run) == "completed" and files:
            lines: list[RenderableType] = []
            for f in files if isinstance(files, list) else [files]:
                if not isinstance(f, dict):
                    continue
                rel = f.get("relativePath", "")
                title = f"← Patched {rel}"
                if f.get("type") == "delete":
                    title = f"# Deleted {rel}"
                elif f.get("type") == "add":
                    title = f"# Created {rel}"
                lines.append(Text(title, style=theme.c("text_muted")))
                patch = f.get("patch")
                if patch:
                    fpath = f.get("filePath") or f.get("relativePath") or ""
                    lines.append(_render_diff(patch, fpath, self.size.width if self.size else None, self._diff_opts()))
            return Group(*lines)
        return self._inline("✦", "Preparing patch...", "Patch", complete=True)

    def _render_todowrite(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        todos = (run.get("metadata") or {}).get("todos") or (run.get("input") or {}).get("todos") or []
        if self._status(run) == "completed" and todos:
            lines: list[RenderableType] = [Text("# Todos", style=theme.c("text_muted"))]
            for todo in todos if isinstance(todos, list) else [todos]:
                if not isinstance(todo, dict):
                    continue
                status = todo.get("status", "pending")
                mark = "✓" if status == "completed" else ("•" if status == "in_progress" else " ")
                color = theme.c("tool_running") if status == "in_progress" else (theme.c("markdown_todo_done") if status == "completed" else theme.c("markdown_todo_open"))
                lines.append(Text(f"[{mark}] {todo.get('content', '')}", style=color))
            return Group(*lines)
        return self._inline("☰", "Updating todos...", "Updating todos...", complete=True)

    def _render_question(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        questions = (run.get("input") or {}).get("questions") or []
        answers = (run.get("metadata") or {}).get("answers")
        count = len(questions) if isinstance(questions, list) else 0
        if self._status(run) == "completed" and answers is not None:
            lines: list[RenderableType] = [Text("# Questions", style=theme.c("text_muted"))]
            for i, q in enumerate(questions if isinstance(questions, list) else []):
                if not isinstance(q, dict):
                    continue
                lines.append(Text(str(q.get("question", "")), style=theme.c("text_muted")))
                ans = answers[i] if isinstance(answers, list) and i < len(answers) else None
                lines.append(
                    Text(
                        ", ".join(ans) if isinstance(ans, list) else str(ans) if ans else "(no answer)",
                        style=theme.c("text"),
                    )
                )
            return Group(*lines)
        return self._inline("→", "Asking questions...", f"Asked {count} question{'s' if count != 1 else ''}", complete=True)

    def _render_task(self, run: dict[str, Any]) -> RenderableType:
        """Render a `task` tool row mirroring opencode's Subagent component
        (packages/tui/src/routes/session/index.tsx):

          running   : `│ Build Task — fix login` (spinner) + live `↳ Bash …`
          completed : `✓ Build Task — fix login` then `↳ 3 toolcalls · 12.5s`
          retrying  : the whole row turns error-red with `↳ Retrying (attempt ·)`
          hint      : a persistent `↓ ctrl+down view subagents` line

        Clicking the row opens the sub-agent's own session (see on_click).
        """
        theme = active_theme()
        status = self._status(run)
        inp = run.get("input") or {}
        meta = run.get("metadata") or {}
        # Resolved agent actually launched wins over the requested input:
        # read-only forcing or validation may change what ran, and the row
        # must show that — not the request. Falls back to input, then build.
        agent_label = (
            str(meta.get("agent") or "").strip()
            or str(inp.get("subagent_type", "")).strip()
            or "build"
        )
        description = str(inp.get("description", "")).strip()
        if not description:
            description = str(meta.get("title") or "")
        background = bool(meta.get("background"))
        sid = str(meta.get("sessionId") or "")
        retry = meta.get("retry") or {}
        title = format_subagent_title(agent_label, description, background)
        icon = "✓" if status == "completed" else "│"
        # the active sub-agent (the one selected / being viewed) gets an accent
        # marker on its row, mirroring opencode's highlighted current agent.
        if self.selected:
            icon = f"▸ {icon}"

        sub: list[RenderableType] = []
        if status == "running":
            runs = self._task_child_runs(sid)
            # official `current()`: the last running/completed child tool that
            # has a real `state.title` (`↳ Bash npm run test`); otherwise just
            # the toolcall count.
            current = next(
                (tool_label(r) for r in reversed(runs) if r.get("status") in ("running", "completed") and tool_title(r)),
                None,
            )
            if meta.get("bg_done"):
                # background worker finished, reply uncollected: frozen
                # final line instead of live ticking (Fix B stamped these).
                tc = meta.get("toolcalls")
                dur = str(meta.get("duration") or "")
                detail = format_completed_subagent_detail(
                    tc if isinstance(tc, int) and tc >= 0 else 0, dur
                )
                if detail:
                    sub.append(Text(f"↳ {detail} (done — task_read to collect)", style=theme.c("text_muted")))
                elif current:
                    sub.append(Text(f"↳ {current}", style=theme.c("text_muted")))
            elif current:
                sub.append(Text(f"↳ {current}", style=theme.c("text_muted")))
            elif runs:
                sub.append(Text(f"↳ {format_subagent_toolcalls(len(runs))}", style=theme.c("text_muted")))
            # live elapsed behind the running row (`↳ 3 toolcalls · 12.5s`)
            # so the agent's runtime is visible while it works, not only
            # after it finishes. Skipped once bg_done froze the final line.
            live_elapsed = str(meta.get("elapsed") or "")
            if live_elapsed and not meta.get("bg_done"):
                sub.append(Text(f"↳ {live_elapsed}", style=theme.c("text_muted")))
        elif status == "completed":
            # Official: `if (!isRunning() && completed)` ALWAYS pushes
            # `↳ N toolcalls · duration`, with duration derived from the
            # child session itself at render time — never gated on a
            # finalize event having fired. Same here: stamped metadata
            # first, then the child session (live map, else disk), so the
            # footer shows even on relinked/resumed rows whose finalize
            # never ran in this process.
            runs = self._task_child_runs(sid)
            toolcalls = self._task_render_toolcalls(sid, meta, runs)
            duration = self._task_render_duration(sid, meta)
            detail = format_completed_subagent_detail(toolcalls or 0, duration)
            if detail:
                sub.append(Text(f"↳ {detail}", style=theme.c("text_muted")))
            elif toolcalls is not None:
                sub.append(Text(f"↳ {format_subagent_toolcalls(toolcalls)}", style=theme.c("text_muted")))
        elif status == "error" and sid:
            err = (run.get("error") or "").strip()
            if err:
                sub.append(Text(f"↳ {err}", style=theme.c("error")))

        lines: list[RenderableType] = [
            self._inline(
                icon,
                "Delegating...",
                title,
                spinner=status == "running",
                complete=bool(description),
                color=theme.c("error") if retry.get("attempt") else None,
            )
        ]
        if retry.get("attempt"):
            lines.append(
                Text(
                    f"↳ {format_subagent_retry(int(retry['attempt']), str(retry.get('message') or ''))}",
                    style=theme.c("error"),
                )
            )
        lines += sub
        if sid and self._is_last_task_row():
            lines.append(Text("↓  ctrl+down  view subagents", style=theme.c("text_muted")))
        return Group(*lines)

    # -- sub-agent task row live state ------------------------------------
    def _task_parent_chat(self) -> ChatView | None:
        """The ChatView this task row is mounted in (the parent session's chat)."""
        parent = self.parent
        if isinstance(parent, ChatView):
            return parent
        return None

    def _is_last_task_row(self) -> bool:
        """True when this is the newest task row in the parent chat — the row
        under which opencode draws the `view subagents` hint (once, per message)."""
        chat = self._task_parent_chat()
        if chat is None:
            return False
        latest: MessageBubble | None = None
        try:
            nodes = list(chat.walk_children(MessageBubble))
        except Exception:
            nodes = [n for n in getattr(chat, "children", ()) if isinstance(n, MessageBubble)]
        for child in nodes:
            if child.role == "tool" and isinstance(child.content, dict) and child.content.get("tool") == "task":
                latest = child
        return latest is self

    def _task_child_session(self, session_id: str) -> Any | None:
        """The sub-agent's Session object (live map, else disk) — the source
        for render-time duration, mirroring official's message timestamps.

        Official computes duration from the child session's first-user and
        last-assistant message times at RENDER time, so the footer never
        depends on a finalize event firing. Ours stores session-level
        created/completed instead; same guarantee, cheaper lookup."""
        if not session_id:
            return None
        try:
            app = self.app
        except Exception:
            app = None
        if app is not None:
            try:
                sessions = getattr(app, "_sessions", None)
                if isinstance(sessions, dict) and sessions.get(session_id) is not None:
                    return sessions.get(session_id)
            except Exception:
                pass
        try:
            from ..session import load_session as _load

            return _load(session_id)
        except Exception:
            return None

    def _task_render_duration(self, session_id: str, meta: dict[str, Any]) -> str:
        """Duration string for a finished task row, official-style.

        Order: live `elapsed` ticker (running) → stamped/finalized `duration`
        → child session created/completed (works even when finalize never
        ran, e.g. relinked/resumed rows). '' only when nothing is known."""
        live = str(meta.get("elapsed") or "")
        if live:
            return live
        done = str(meta.get("duration") or "")
        if done:
            return done
        try:
            sess = self._task_child_session(session_id)
            if sess is None:
                return ""
            created = float(getattr(sess, "created", 0) or 0)
            completed = float(getattr(sess, "completed", 0) or 0)
            if completed > created:
                from .input_bar import format_duration

                return format_duration(completed - created)
        except Exception:
            pass
        return ""

    def _task_render_toolcalls(self, session_id: str, meta: dict[str, Any], runs: list[dict[str, Any]]) -> int | None:
        """Toolcall count for a finished task row, official-style.

        Order: stamped `toolcalls` → live child-chat runs → child session
        transcript tool messages (disk, resume-proof). None only when the
        child session can't be found at all."""
        stamped = meta.get("toolcalls")
        if isinstance(stamped, int) and stamped >= 0:
            return stamped
        if runs:
            return len([r for r in runs if r.get("status") == "completed"])
        try:
            sess = self._task_child_session(session_id)
            if sess is None:
                return None
            messages = getattr(sess, "messages", None) or []
            return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
        except Exception:
            return None

    def _task_child_chat(self, session_id: str) -> ChatView | None:
        """The sub-agent's own ChatView (opencode reads the child session's parts
        via ``sync.data.part[msg.id]``; here the child's tool bubbles are the
        equivalent live store). None when it isn't registered / was pruned."""
        if not session_id:
            return None
        try:
            app = self.app
        except Exception:
            # WidgetHost / off-pump tests, or the app is already torn down —
            # there's simply no child ChatView to read live state from.
            return None
        chats = getattr(app, "_chats", None)
        if not isinstance(chats, dict):
            return None
        return chats.get(session_id)

    def _task_child_runs(self, session_id: str) -> list[dict[str, Any]]:
        chat = self._task_child_chat(session_id)
        if chat is None:
            return []
        return [
            r
            for r in chat.tool_runs()
            if r.get("status") in ("running", "completed", "error")
        ]

    def _render_execute(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        status = self._status(run)
        icon = "✓" if status == "completed" else "│"
        row = self._inline(icon, "execute", "execute", spinner=status == "running")
        calls = (run.get("metadata") or {}).get("toolCalls") or []
        sub = []
        for c in calls if isinstance(calls, list) else []:
            if not isinstance(c, dict):
                continue
            name = c.get("tool", "")
            args = _format_input(c.get("input") or {})
            failed = " (failed)" if c.get("status") == "error" else ""
            sub.append(Text(f"↳ {name}{args}{failed}", style=theme.c("text_muted")))
        return Group(row, *sub) if sub else row

    def _render_skill(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        status = self._status(run)
        name = str((run.get("input") or {}).get("name", ""))
        output = (run.get("output") or "").strip()
        if status == "completed" and output and not name:
            lines: list[RenderableType] = [Text("# Skills", style=theme.c("text_muted"))]
            for line in output.split("\n"):
                if line.startswith("- "):
                    lines.append(Text(line[2:], style=theme.c("text")))
            return Group(*lines) if len(lines) > 1 else self._inline("→", "Loading skill...", "Skills", complete=True)
        row = self._inline("→", "Loading skill...", f'Skill "{name}"' if name else "Skills", complete=bool(name or status == "completed"))
        err = self._error_line(run)
        parts: list[RenderableType] = [row]
        if err is not None:
            parts.append(err)
        # Auditability: a loaded skill injects up to 30KB of instructions —
        # show the body collapsed (3 lines) with click-to-expand like generic
        # tools, instead of a bare one-liner.
        if status == "completed" and output and name:
            collapsed = self._tool_collapse({**run, "tool": "skill"})
            body = output if self.expanded else collapsed["output"]
            parts.append(Text(body, style=theme.c("text")))
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                parts.append(Text(label, style=theme.c("text_muted")))
        return Group(*parts) if len(parts) > 1 else row

    def _render_generic(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        tool = run.get("tool", "?")
        output = (run.get("output") or "").strip()
        if self._status(run) == "completed" and output:
            collapsed = self._tool_collapse(run)
            output_text = output if self.expanded else collapsed["output"]
            rows: list[RenderableType] = [
                Text(f"# {tool} {_format_input(run.get('input') or {})}".strip(), style=theme.c("text_muted")),
                Text(output_text, style=theme.c("text")),
            ]
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                rows.append(Text(label, style=theme.c("text_muted")))
            return Group(*rows)
        label = f"{tool} {_format_input(run.get('input') or {})}".strip()
        return self._inline("⚙", "Running...", label, complete=bool(label))

    # -- frame (border / background / padding) ----------------------------
    def _apply_frame(self) -> None:
        theme = active_theme()
        st = self.styles
        if self.role == "user":
            st.background = theme.c("background_panel")
            st.border_left = ("solid", theme.agent_color(self.agent))
            st.padding = (1, 1, 1, 2)
        elif self.role == "tool" and self._tool_block():
            # opencode BlockTool: left border in the panel background, panel fill.
            st.background = theme.c("background_panel")
            st.border_left = ("solid", theme.c("background"))
            st.padding = (1, 1, 1, 2)
        else:
            # indented plain text (assistant / mode-line / meta / inline tool).
            # use an invisible border so Textual never paints its default.
            st.background = "transparent"
            st.border_left = ("solid", theme.c("background"))
            st.padding = (0, 0, 0, 3)
        st.margin = (1, 0, 0, 0)

    def _tool_block(self) -> bool:
        """A completed tool run renders as a block iff it produced a result block."""
        if not isinstance(self._message, dict):
            return False
        run = self._message
        if run.get("status") != "completed":
            return False
        name = run.get("tool", "?")
        display = _tool_display(name)
        if display == "bash":
            return bool((run.get("output") or "").strip())
        if display == "edit":
            return bool((run.get("metadata") or {}).get("diff"))
        if display == "apply_patch":
            return bool((run.get("metadata") or {}).get("files"))
        if display == "todowrite":
            return bool((run.get("metadata") or {}).get("todos"))
        if display == "question":
            return (run.get("metadata") or {}).get("answers") is not None
        if display == "write":
            return bool((run.get("metadata") or {}).get("content"))
        if display == "generic":
            return bool((run.get("output") or "").strip())
        if display == "skill":
            return bool((run.get("output") or "").strip())
        if display in ("websearch", "webfetch", "webfetch_many", "find_symbols", "read"):
            return bool((run.get("output") or "").strip())
        return False

    def _tool_collapse(self, run: dict[str, Any]) -> dict:
        """Mirror opencode's per-tool collapse limits (bash 10 lines, generic 3)."""
        output = (run.get("output") or "").strip()
        if not output:
            return {"output": output, "overflow": False}
        display = _tool_display(run.get("tool", "?"))
        if display == "bash":
            return collapse_tool_output(output, 10, 10 * 80)
        if display in ("websearch", "webfetch", "webfetch_many", "find_symbols", "read"):
            return collapse_tool_output(output, 8, 8 * 80)
        return collapse_tool_output(output, 3, 3 * 80)

    def _tool_overflow(self) -> bool:
        return isinstance(self._message, dict) and self._tool_collapse(self._message)["overflow"]

    # -- updates -----------------------------------------------------------
    def _refresh(self) -> None:
        # Frozen bubbles are done: same pixels, skip the rebuild + layout.
        # Any state flip unfreezes first (see watchers below), so a frozen
        # bubble can never go stale.
        try:
            if bool(getattr(self, "frozen", False)) and not self.streaming:
                return
        except Exception:
            pass
        self._apply_frame()
        self.update(self._build_content())

    def freeze(self) -> None:
        """Mark this bubble finished: future _refresh calls are no-ops."""
        try:
            self.streaming = False
            self.frozen = True
        except Exception:
            pass

    def unfreeze(self) -> None:
        try:
            self.frozen = False
        except Exception:
            pass

    def watch_frozen(self, value: bool) -> None:
        # Entering frozen repaints once so the cached pixels match, then stays
        # silent. (The reactive setter already stored the value.)
        try:
            if value:
                self._apply_frame()
                self.update(self._build_content())
        except Exception:
            pass

    def _start_spinner(self) -> None:
        if self._timer is None:
            self._timer = self.set_interval(SPINNER_INTERVAL, self._tick)

    def _stop_spinner(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        # Background-session chats stay mounted but invisible: spinning them
        # at 12.5fps steals UI-thread time from the visible screen (very
        # noticeable on phone CPUs). Skip the re-render until visible again.
        try:
            if not (self.is_attached and self.screen.is_current):
                return
        except Exception:
            pass
        # Adaptive pace: the frame only advances on fresh tokens
        # (note_stream_activity stamps _last_token_mono). Fast net -> full
        # 12fps spin; slow net -> ~6fps; stalled (>1.5s) -> frame holds.
        # ponytail: naive arrival-age heuristic; upgrade path is provider-side
        # token timestamps if they ever exist.
        try:
            last_tok = float(getattr(self, "_last_token_mono", 0.0) or 0.0)
            if last_tok > 0.0:
                age = time.monotonic() - last_tok
                if age > 1.5:
                    return
                if age > 0.5:
                    self._tick_slow = not getattr(self, "_tick_slow", False)
                    if self._tick_slow:
                        return
        except Exception:
            pass
        if self.role == "tool" and self._message.get("status") == "running":
            self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
            self._refresh()
        elif self.role == "reasoning" and self.streaming:
            self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
            self._refresh()

    def note_stream_activity(self, n_chars: int = 0) -> None:
        """Stamp token arrival (age paces the spinner, not the frame)."""
        try:
            self._last_token_mono = time.monotonic()
            self._token_chars = int(getattr(self, "_token_chars", 0) or 0) + max(0, int(n_chars or 0))
        except Exception:
            pass

    def _task_sid(self) -> str:
        """Child session id for a task row, from every known location.

        Live rows carry metadata.sessionId; resumed rows re-attached by the
        relinker do too — but rows rendered before the relink (or from old
        saves mid-retry) may only have it in the bubble's fallback fields.
        """
        try:
            c = self.content or {}
            meta = c.get("metadata") or {}
            for key in ("sessionId", "session_id"):
                sid = meta.get(key)
                if sid:
                    return str(sid)
            for key in ("sessionId", "session_id", "sid"):
                sid = c.get(key)
                if sid:
                    return str(sid)
        except Exception:
            pass
        return ""

    def on_click(self, event: Any) -> None:
        if self.role == "reasoning":
            self.expanded = not self.expanded
        elif self.role == "tool":
            if self.content.get("tool") == "task":
                sid = self._task_sid()
                if sid:
                    self.post_message(OpenTaskSession(sid))
            elif self._tool_overflow():
                self.expanded = not self.expanded

    def on_key(self, event: Any) -> None:
        if self.role == "reasoning" and self.has_focus and event.key in ("enter", "space"):
            event.stop()
            self.expanded = not self.expanded
        elif self.role == "tool" and self.has_focus and event.key in ("enter", "space"):
            if self.content.get("tool") == "task":
                sid = self._task_sid()
                if sid:
                    event.stop()
                    self.post_message(OpenTaskSession(sid))

    def update_tool(self, tool_run: dict[str, Any]) -> None:
        self.unfreeze()
        self._message = tool_run
        if tool_run.get("status") == "running":
            self._start_spinner()
        else:
            self._stop_spinner()
            self.streaming = False
        self._refresh()
        if tool_run.get("status") not in ("running", "pending"):
            self.freeze()

    def set_tool_metadata(self, key: str, value: Any) -> None:
        """Merge one metadata entry into the tool run in place (keeps input,
        output and the other metadata), then re-render. The engine supplies the
        task row in a running -> completed sequence; this lets the TUI enrich it
        with live data (duration, toolcall count, retry state) without losing
        what the engine already wrote."""
        if not isinstance(self._message, dict):
            return
        self.unfreeze()
        meta = self._message.setdefault("metadata", {})
        meta[key] = value
        # _message and content are the SAME dict object (content property
        # returns self._message), so the click handler sees this immediately.
        # keep the parent chat's lookup index fresh (task rows gain sessionId
        # after mount; without this find_task rescans every event)
        try:
            parent = self.parent
            note = getattr(parent, "note_tool_metadata", None)
            if callable(note):
                note(self)
        except Exception:
            pass
        self._refresh()

    def pop_tool_metadata(self, key: str) -> None:
        if not isinstance(self._message, dict):
            return
        meta = self._message.get("metadata")
        if isinstance(meta, dict) and key in meta:
            self.unfreeze()
            del meta[key]
            self._refresh()

    def update_text(self, text: str) -> None:
        self.unfreeze()
        self._message = text
        self._refresh()
        self.freeze()

    def update_reasoning(self, text: str) -> None:
        self.unfreeze()
        self._message = text
        self._refresh()
        self.freeze()

    def append_stream(self, text: str) -> None:
        """Append one streamed chunk (amortized O(1) render, not per-token)."""
        if text:
            # Pace the spinner by real arrivals: fast net -> full spin,
            # stalled net -> held frame (see _tick).
            try:
                self.note_stream_activity(len(text))
            except Exception:
                pass
            self._stream_parts.append(text)
            # O(1) append: flush into _message every 16 chunks instead of
            # re-joining the whole reply per token (O(n^2) on long answers).
            if len(self._stream_parts) >= 16:
                try:
                    self._message = (self._message or "") + "".join(self._stream_parts) if isinstance(self._message, str) else "".join(self._stream_parts)
                except Exception:
                    self._message = "".join(self._stream_parts)
                del self._stream_parts[:]
            else:
                # cheap path: keep last chunk visible without full re-join
                try:
                    if isinstance(self._message, str):
                        self._message = (self._message or "") + text
                    else:
                        self._message = text
                except Exception:
                    pass
            self._refresh_stream()

    def _refresh_stream(self) -> None:
        # Streaming render at most ~10/s: tokens arriving faster coalesce
        # into one layout instead of thrashing per chunk (long answers
        # stuttered more and more). Final text always lands via end_stream.
        # Tail-only: the streaming body is plain Text (see _build_content),
        # so only the NEW tail needs paint — Textual still does a full
        # widget update, but the renderable itself is O(new) plain glyphs,
        # never an O(reply) markdown re-parse. Same pixels at the end.
        try:
            import time as _time

            now = _time.monotonic()
            last = getattr(self, "_last_stream_mono", 0.0)
            if now - last < 0.1:
                return
            self._last_stream_mono = now
        except Exception:
            pass
        self._refresh()

    def end_stream(self) -> None:
        # flush buffered chunks so the final text is exact (bubble-level;
        # ChatView.end_stream(text) below finalizes the whole turn)
        self.flush_stream()
        self.streaming = False
        try:
            self._stop_spinner()
        except Exception:
            pass
        self._refresh()
        self.freeze()

    def flush_stream(self) -> None:
        try:
            if getattr(self, "_stream_parts", None):
                try:
                    self._message = (self._message or "") + "".join(self._stream_parts) if isinstance(self._message, str) else "".join(self._stream_parts)
                except Exception:
                    self._message = "".join(self._stream_parts)
                del self._stream_parts[:]
        except Exception:
            pass

    def end_reasoning(self) -> None:
        self.streaming = False
        self._stop_spinner()
        if self._thought_started is not None:
            self._thought_seconds = time.monotonic() - self._thought_started
            self._thought_started = None
        # keep thoughts collapsed by default: show `+ Thought for 5.0s: <title>`
        # and let the user click / Enter / Ctrl+Shift+E to expand the full body
        self.expanded = False
        self._refresh()
        self.freeze()


class ChatView(VerticalScroll):
    """Scrollable list of message bubbles + streaming cursor."""

    messages: reactive = reactive([])
    streaming: reactive = reactive(False)
    streaming_text: reactive = reactive("")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stream_bubble: MessageBubble | None = None
        self._reasoning_bubble: MessageBubble | None = None
        self._compaction_bubble: MessageBubble | None = None
        # Queued user bubbles in submission order (mirrors opencode's FIFO
        # prompt queue). The engine clears them one-by-one as it folds each
        # prompt into the running turn.
        self._queued_bubbles: list[MessageBubble] = []
        # True while the user is reading the newest output (at the bottom).
        # When they scroll up to re-read history, we stop yanking the view back
        # down on every stream delta / tool update.
        self._follow_bottom = True
        # True when this chat belongs to a sub-agent session: the arrow keys
        # navigate the session tree instead of scrolling (so ↑ inside a
        # sub-agent always returns to the parent, even when the chat — not the
        # prompt — has focus).
        self._session_is_child = False
        # First-open welcome banner (ASCII opencode logo). Mounted only when
        # the chat has no messages/stream yet, removed once the conversation
        # starts.
        self._welcome_logo: Static | None = None
        # O(1) bubble lookups: the hot paths (find_tool per tool event,
        # find_task per sub-agent event, last_reasoning per toggle) used to
        # scan every mounted bubble on EVERY event — O(bubbles) per token-adjacent
        # update. Indexes are validated (still mounted) on hit, rebuilt on miss.
        self._tool_index: dict[str, MessageBubble] = {}
        self._task_index: dict[str, MessageBubble] = {}
        self._last_reasoning_bubble: MessageBubble | None = None
        self._history_pending: list = []
        self._history_loading: bool = False
        self._history_anchor: Any = None
        self._history_session_id: str = ""
        self._history_chunk: int = 80
        # Live-window hides (display:none, same pill as resume): oldest
        # bubbles past chat_live_window stay mounted-but-hidden so scroll
        # and layout stay O(window). Rebuilt by _trim_live_window.
        self._hidden_count: int = 0

    # -- first-open banner -------------------------------------------------
    def show_logo(self) -> None:
        """Show the opencode logo when the chat is empty (first launch)."""
        if self._welcome_logo is not None:
            return
        logo = Static(opencode_logo_text(), classes="chat-welcome-logo")
        self.mount(logo)
        self._welcome_logo = logo
        try:
            self.scroll_home(animate=False)
        except Exception:
            pass

    def _dismiss_logo(self) -> None:
        """Remove the first-open logo once the conversation begins."""
        logo = self._welcome_logo
        self._welcome_logo = None
        if logo is not None:
            try:
                logo.remove()
            except Exception:
                pass

    def welcome_empty(self) -> bool:
        """True while the first-open banner is still visible (no messages)."""
        return self._welcome_logo is not None

    # -- key routing -------------------------------------------------------
    async def on_key(self, event: Any) -> None:
        if getattr(self, "_session_is_child", False) and event.key in ("up", "left", "right"):
            event.stop()
            event.prevent_default()
            self.post_message(SessionNavRequested(event.key))
        # Not intercepted: let the key bubble to the App's binding dispatch
        # (scroll bindings, etc.). There is no base `on_key` message handler
        # to chain to, so just return.

    # -- scrolling ---------------------------------------------------------
    def _auto_scroll(self, throttled: bool = False) -> None:
        """Scroll to the newest message, unless the user scrolled up to read
        earlier conversation (opencode keeps your position while the model
        keeps streaming/tool-running below). Mid-stream calls pass
        throttled=True: scroll_end forces a full layout, so scrolls stay
        smooth and fast while tokens flow (final positions still land via
        the unthrottled end_stream/end_reasoning calls).

        Bottom-follow is sticky AND cheap: when already at the bottom we
        skip the layout when nothing grew (same max_scroll_y as last time),
        so 10 calls/s cost ~0 layouts instead of 10. ponytail: 0.05s floor
        + growth-gate; upgrade path is Textual's own smooth-scroll if it
        ever lands on phone builds.
        """
        if not self._follow_bottom:
            return
        if throttled:
            try:
                now = time.monotonic()
                last = getattr(self, "_last_scroll_mono", 0.0)
                if now - last < 0.05:
                    return
                try:
                    mx = float(self.max_scroll_y)
                except Exception:
                    mx = -1.0
                if mx >= 0 and mx == float(getattr(self, "_last_scroll_max", -1.0)):
                    return  # nothing grew: skip the layout entirely
                self._last_scroll_mono = now
                try:
                    self._last_scroll_max = mx
                except Exception:
                    pass
            except Exception:
                pass
        else:
            try:
                self._last_scroll_max = float(self.max_scroll_y)
            except Exception:
                pass
        self.scroll_end(animate=False)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        try:
            self._follow_bottom = new_value >= self.max_scroll_y - 1
        except Exception:
            self._follow_bottom = True
        try:
            self._maybe_prefetch()
        except Exception:
            pass

    def set_history_pending(self, pending: list, session_id: str = "", chunk: int = 80) -> None:
        self._history_pending = list(pending or [])
        self._history_loading = False
        self._history_session_id = session_id or ""
        self._history_chunk = int(chunk or 80)
        self._update_history_pill()

    def _history_pill_text(self) -> str:
        try:
            n = len(getattr(self, "_history_pending", []) or [])
        except Exception:
            n = 0
        try:
            hidden = int(getattr(self, "_hidden_count", 0) or 0)
        except Exception:
            hidden = 0
        # Live-trimmed bubbles outrank resume-pending: hidden are already
        # mounted (display:none, unhide is a style flip) while pending still
        # needs a disk load — but the pill text is shared.
        total = hidden if hidden > 0 else n
        if total <= 0:
            return ""
        return f"↑ {total} older messages — scroll up to load"

    def _history_count(self) -> int:
        """Pill number without touching the DOM (headless/test safe)."""
        try:
            n = len(getattr(self, "_history_pending", []) or [])
        except Exception:
            n = 0
        try:
            hidden = int(getattr(self, "_hidden_count", 0) or 0)
        except Exception:
            hidden = 0
        return hidden if hidden > 0 else n

    def _update_history_pill(self) -> None:
        try:
            anchor = getattr(self, "_history_anchor", None)
            n = len(getattr(self, "_history_pending", []) or [])
            try:
                hidden = int(getattr(self, "_hidden_count", 0) or 0)
            except Exception:
                hidden = 0
            if n <= 0 and hidden <= 0:
                if anchor is not None:
                    try:
                        anchor.remove()
                    except Exception:
                        pass
                    self._history_anchor = None
                return
            text = self._history_pill_text()
            if anchor is None or not getattr(anchor, "is_attached", False):
                bubble = MessageBubble("meta", text)
                try:
                    kids = list(self.children)
                    if kids:
                        self.mount(bubble, before=kids[0])
                    else:
                        self.mount(bubble)
                except Exception:
                    try:
                        self.mount(bubble)
                    except Exception:
                        return
                self._history_anchor = bubble
            else:
                try:
                    anchor._message = text
                    anchor._refresh()
                except Exception:
                    pass
        except Exception:
            pass

    def _live_window(self) -> int:
        """Max mounted message bubbles on this chat (0 = unlimited)."""
        try:
            cfg = getattr(getattr(self, "app", None), "cfg", None)
            w = int(getattr(cfg, "chat_live_window", 120) or 0)
            return max(0, w)
        except Exception:
            return 120

    def _visible_bubbles(self) -> list:
        try:
            return [c for c in self.query(MessageBubble) if getattr(c, "display", "") != "none"]
        except Exception:
            return []

    def _trim_live_window(self) -> None:
        """Hide oldest bubbles past the live window (display:none, not remove).

        Same pill/scroll-up flow as the resume path, but for live growth:
        mounting stays O(window), unhide is a style flip, and every hot path
        (find_tool/find_task/last_reasoning/tool_runs) already validates
        membership so a display:none bubble is skipped, never stale.
        """
        try:
            window = self._live_window()
            if window <= 0:
                return
            visible = self._visible_bubbles()
            over = len(visible) - window
            if over <= 0:
                return
            # Never hide the streaming / in-flight rows: newest-first scan
            # skips stream/reasoning bubbles and in-progress tool runs.
            hide: list = []
            for b in visible:
                if len(hide) >= over:
                    break
                try:
                    if getattr(b, "streaming", False):
                        continue
                    content = getattr(b, "content", None)
                    if isinstance(content, dict) and content.get("status") in ("running", "pending"):
                        continue
                except Exception:
                    pass
                hide.append(b)
                if len(hide) >= over:
                    break
            for b in hide:
                try:
                    b.display = "none"
                except Exception:
                    pass
            self._hidden_count = int(getattr(self, "_hidden_count", 0) or 0) + len(hide)
            self._update_history_pill()
        except Exception:
            pass

    def _unhide_older(self, count: int = 0) -> int:
        """Unhide hidden bubbles oldest-first (scroll-up loads them back)."""
        try:
            kids = [c for c in self.children if c.__class__.__name__ == "MessageBubble"
                    and getattr(c, "display", "") == "none"]
            if not kids:
                self._hidden_count = 0
                self._update_history_pill()
                return 0
            try:
                chunk = int(getattr(self, "_history_chunk", 80) or 80)
            except Exception:
                chunk = 80
            take = min(len(kids), count if count > 0 else chunk * 2)
            for b in kids[:take]:
                try:
                    b.display = "block"
                except Exception:
                    pass
            left = len(kids) - take
            self._hidden_count = max(0, left)
            self._update_history_pill()
            return take
        except Exception:
            return 0

    def _maybe_prefetch(self) -> None:
        # Live-hidden bubbles unhide first (already mounted, zero layout
        # cost); only when none remain does disk-backed pending load.
        try:
            if int(getattr(self, "_hidden_count", 0) or 0) > 0:
                self._unhide_older()
                return
        except Exception:
            pass
        pending = getattr(self, "_history_pending", None) or []
        if not pending or getattr(self, "_history_loading", False):
            return
        try:
            y = float(self.scroll_y)
            mx = float(self.max_scroll_y)
        except Exception:
            return
        if mx <= 0:
            return
        if y > 400 and y > mx * 0.3:
            return
        self._load_older()

    def _load_older(self) -> None:
        # Live-hidden bubbles (same session) unhide before disk pending loads.
        try:
            if int(getattr(self, "_hidden_count", 0) or 0) > 0:
                if self._unhide_older() > 0:
                    return
        except Exception:
            pass
        pending = getattr(self, "_history_pending", None) or []
        if not pending or getattr(self, "_history_loading", False):
            return
        self._history_loading = True
        try:
            old_max = float(self.max_scroll_y)
        except Exception:
            old_max = 0.0
        try:
            old_y = float(self.scroll_y)
        except Exception:
            old_y = 0.0
        try:
            chunk = int(getattr(self, "_history_chunk", 80) or 80)
        except Exception:
            chunk = 80
        take = min(len(pending), chunk * 2)
        if take <= 0:
            self._history_loading = False
            return
        batch = pending[:take]
        del pending[:take]
        self._history_pending = pending
        batch.reverse()
        try:
            anchor = getattr(self, "_history_anchor", None)
            kids = list(self.children)
            if anchor is not None and anchor in kids:
                idx = kids.index(anchor)
                first = kids[idx + 1] if idx + 1 < len(kids) else None
            else:
                first = kids[0] if kids else None
            if first is None:
                for fn in batch:
                    try:
                        fn(None)
                    except Exception:
                        continue
            else:
                for fn in batch:
                    try:
                        fn(first)
                    except Exception:
                        try:
                            fn(None)
                        except Exception:
                            continue
        finally:
            self._update_history_pill()
            try:
                def _fix() -> None:
                    try:
                        new_max = float(self.max_scroll_y)
                        self.scroll_to(y=old_y + max(0.0, new_max - old_max), animate=False)
                    except Exception:
                        pass
                    self._history_loading = False
                    try:
                        self._maybe_prefetch()
                    except Exception:
                        pass
                try:
                    self.call_after_refresh(_fix)
                except Exception:
                    self.call_later(_fix)
            except Exception:
                self._history_loading = False

    def append_user(self, text: str, agent: str = "build", queued: bool = False, before: Any = None, scroll: bool = True) -> MessageBubble:
        bubble = MessageBubble("user", text, agent=agent)
        bubble.queued = queued
        if queued:
            self._queued_bubbles.append(bubble)
        self._mount_bubble(bubble, before=before)
        if scroll:
            self._auto_scroll()
        return bubble

    def append_directive(self, text: str, title: str = "", agent: str = "") -> MessageBubble:
        """The parent's instruction rendered at the top of a sub-agent's chat
        (official opencode shows the task directive as the first message)."""
        self._dismiss_logo()
        bubble = MessageBubble("user", text, directive=title or "Task")
        if agent.strip():
            bubble.agent = agent.strip()
        bubble.add_class("directive")
        self.mount(bubble)
        self._auto_scroll()
        return bubble

    def promote_next_queued(self) -> str | None:
        """Clear the ` QUEUED ` badge on the oldest queued message.

        Called when the engine folds that prompt into the running turn
        (opencode's Session Drain) — the message keeps its place in the chat,
        just loses its badge. Returns the promoted text.
        """
        if not self._queued_bubbles:
            return None
        bubble = self._queued_bubbles.pop(0)
        try:
            bubble.queued = False
        except Exception:
            pass
        return str(bubble.content)

    def queued_count(self) -> int:
        return len(self._queued_bubbles)

    def _mount_bubble(self, bubble: Any, before: Any = None) -> Any:
        self._dismiss_logo()
        try:
            if before is not None and getattr(before, "is_attached", False):
                self.mount(bubble, before=before)
            else:
                self.mount(bubble)
        except Exception:
            try:
                self.mount(bubble)
            except Exception:
                pass
        try:
            if getattr(bubble, "role", "") == "tool":
                self._index_tool_bubble(bubble)
        except Exception:
            pass
        # Live chats that grow past the window trim here — same pill flow as
        # resume, so a 500-message session never holds 500 live layouts.
        try:
            if before is None:
                self._trim_live_window()
        except Exception:
            pass
        return bubble

    def clear(self) -> None:
        """Drop every bubble and reset the streaming cursors. Used when the
        main session is deleted and the workspace resets to a fresh one."""
        self._stream_bubble = None
        self._reasoning_bubble = None
        self._follow_bottom = True
        self._queued_bubbles = []
        self._history_pending = []
        self._history_loading = False
        self._history_anchor = None
        self._hidden_count = 0
        for child in list(self.children):
            try:
                child.remove()
            except Exception:
                pass
        try:
            self.scroll_home(animate=False)
        except Exception:
            pass
        # A removed-but-unflushed logo widget may still be in `children` (Textual
        # defers DOM removal), so drop the reference: the banner belongs to the
        # app's very first open, not to a workspace reset.
        self._welcome_logo = None

    def append_assistant(self, text: str, before: Any = None, scroll: bool = True) -> None:
        self._mount_bubble(MessageBubble("assistant", text), before=before)
        if scroll:
            self._auto_scroll()

    def append_meta(self, text: str, before: Any = None, scroll: bool = True) -> None:
        """Persistent command/system output (e.g. /models, /help, /config)."""
        self._mount_bubble(MessageBubble("meta", text), before=before)
        if scroll:
            self._auto_scroll()

    def append_compaction(self, summary: str, before: Any = None, scroll: bool = True) -> None:
        """Render a ` Compaction ` divider + markdown summary (opencode's
        compaction part).

        Ends any in-flight reasoning/stream bubble first so the divider lands
        between the finished text and whatever the model continues after the
        summarized history is replayed.
        """
        self.end_reasoning()
        self.remove_last_stream_bubble()
        self._mount_bubble(MessageBubble("compaction", summary), before=before)
        if scroll:
            self._auto_scroll()

    def begin_compaction_stream(self) -> None:
        """Start the live `▸ Compacted summary` bubble before the summary
        model streams; deltas append via stream_compaction_delta."""
        self._dismiss_logo()
        self.end_reasoning()
        self.remove_last_stream_bubble()
        bubble = MessageBubble("compaction", "", streaming=True)
        self._compaction_bubble = bubble
        self.mount(bubble)
        self._auto_scroll()

    def stream_compaction_delta(self, text: str) -> None:
        """Append one live summary token to the streaming compaction bubble."""
        bubble = self._compaction_bubble
        if bubble is None:
            self.begin_compaction_stream()
            bubble = self._compaction_bubble
        bubble.append_stream(text)
        self._auto_scroll()

    def end_compaction_stream(self, summary: str) -> None:
        """Finalize the live compaction bubble into the ` Compaction ` divider.

        The live bubble was rendering plain text while the summary streamed;
        on `compacted` we re-render the FULL final summary as markdown inside
        the divider (same look as the non-live path). An empty outcome is never
        silently dropped — a muted note keeps the failure visible."""
        bubble = self._compaction_bubble
        self._compaction_bubble = None
        if bubble is None:
            # No live bubble (compaction failed before the first delta) — fall
            # back to the static divider so the outcome is still visible.
            if summary:
                self.append_compaction(summary)
            return
        if not summary and not bubble.content:
            bubble.remove()
            self.mount(MessageBubble("meta", "(no summary returned)"))
            self._auto_scroll()
            return
        bubble._message = summary or bubble.content
        bubble.streaming = False  # re-render as the finished ` Compaction ` divider
        bubble._refresh()

    def append_tool(self, tool_run: dict[str, Any], before: Any = None, scroll: bool = True) -> None:
        bubble = MessageBubble("tool", tool_run)
        if tool_run.get("status") == "running":
            bubble._start_spinner()
        self._mount_bubble(bubble, before=before)
        if scroll:
            self._auto_scroll()

    def _thoughts_visible(self) -> bool:
        """Whether thought bubbles may render (Thinking-picker on/off).

        Reads live config off the running app so a toggle applies instantly
        without rebuilding chats. Defaults True when no app/config exists
        (headless tests, harness mounts) — existing behavior untouched.
        """
        try:
            cfg = getattr(getattr(self, "app", None), "cfg", None)
            if cfg is None:
                return True
            return bool(getattr(cfg, "show_thoughts", True))
        except Exception:
            return True

    def append_reasoning(self, text: str, seconds: float | None = None, before: Any = None, scroll: bool = True) -> None:
        """Render a finished `+ Thought` bubble (used when replaying saved
        history; live thoughts stream through begin_thinking/stream_reasoning)."""
        bubble = MessageBubble("reasoning", text)
        bubble._thought_seconds = seconds
        bubble.visible = self._thoughts_visible()
        self._mount_bubble(bubble, before=before)
        self._last_reasoning_bubble = bubble
        if scroll:
            self._auto_scroll()

    def set_thoughts_visible(self, visible: bool) -> int:
        """Show/hide every thought bubble in this chat. Returns count."""
        n = 0
        try:
            for b in self.reasoning_bubbles():
                try:
                    b.visible = visible
                    n += 1
                except Exception:
                    continue
        except Exception:
            pass
        return n

    def begin_stream(self) -> None:
        self._dismiss_logo()
        self._stream_bubble = MessageBubble("assistant", "")
        self._stream_bubble.streaming = True
        self.mount(self._stream_bubble)
        self._auto_scroll()

    def stream_delta(self, text: str) -> None:
        if self._stream_bubble is None:
            self.begin_stream()
        self._stream_bubble.append_stream(text)
        self._auto_scroll(throttled=True)

    def stream_reasoning_delta(self, text: str) -> None:
        self._dismiss_logo()
        if self._reasoning_bubble is None:
            bubble = MessageBubble("reasoning", "")
            bubble.streaming = True
            bubble._start_spinner()
            bubble._thought_started = time.monotonic()
            bubble.visible = self._thoughts_visible()
            # chronological order: each new thought mounts below the previous
            # tool runs (not above a stale empty stream bubble)
            self.mount(bubble)
            self._reasoning_bubble = bubble
            self._last_reasoning_bubble = bubble
        self._reasoning_bubble.append_stream(text)
        self._auto_scroll(throttled=True)

    def begin_thinking(self) -> None:
        """Mount an eager `Thinking...` bubble the moment a turn starts, before
        the first token arrives, so the UI reacts instantly to Enter (mirrors
        opencode). Real reasoning deltas stream into this same bubble; if no
        reasoning ever arrives, end_reasoning drops the empty placeholder."""
        self._dismiss_logo()
        if self._reasoning_bubble is not None:
            return
        bubble = MessageBubble("reasoning", "")
        bubble.streaming = True
        bubble._start_spinner()
        bubble._thought_started = time.monotonic()
        self.mount(bubble)
        self._reasoning_bubble = bubble
        self._last_reasoning_bubble = bubble
        self._auto_scroll()

    def end_reasoning(self) -> None:
        if self._reasoning_bubble is not None:
            if not self._reasoning_bubble.content:
                # the eager placeholder produced no actual reasoning — remove
                # it instead of leaving a misleading empty `Thought` line
                bubble = self._reasoning_bubble
                self._reasoning_bubble = None
                bubble._stop_spinner()
                try:
                    bubble.remove()
                except Exception:
                    pass
            else:
                self._reasoning_bubble.end_reasoning()
                self._reasoning_bubble = None
        self._auto_scroll()

    def end_stream(self, text: str = "") -> None:
        """Finalize the streaming cursor. An assistant bubble that never
        received any text is DROPPED instead of left as a blank line — e.g. a
        thought + tool turn whose final "summary" streamed an empty chunk."""
        if self._stream_bubble is not None:
            if text:
                try:
                    self._stream_bubble.flush_stream()
                except Exception:
                    pass
                self._stream_bubble.update_text(text)
            if self._stream_bubble.content:
                self._stream_bubble.streaming = False
                self._stream_bubble._refresh()
            else:
                try:
                    self._stream_bubble.remove()
                except Exception:
                    pass
            self._stream_bubble = None
        self._auto_scroll()

    def remove_last_stream_bubble(self) -> None:
        """Remove the empty streaming bubble left behind when there's no reply.

        A bubble that already holds partial text is KEPT (only its streaming
        cursor is ended) so a mid-stream error doesn't discard model output.
        """
        target = self._stream_bubble
        if target is None:
            target = None
            # The only empty assistant bubbles are streaming-cursor leftovers,
            # which always sit in the trailing assistant run at the bottom of
            # the chat. Walk just that run instead of a full container query
            # (O(call) → O(trailing run) per remove).
            for child in reversed(self.children):
                try:
                    role = child.role
                except Exception:
                    break
                if role != "assistant":
                    break
                if not child.content and not child.streaming:
                    target = child
                    break
        if target is not None:
            if target.content:
                target.streaming = False
            else:
                try:
                    target.remove()
                except Exception:
                    pass
        self._stream_bubble = None
        self._auto_scroll()

    def _indexed(self, bubble: MessageBubble | None) -> MessageBubble | None:
        """Return the bubble if still mounted here, else None (stale index)."""
        if bubble is None:
            return None
        try:
            return bubble if bubble.parent is self else None
        except Exception:
            return None

    def _index_tool_bubble(self, bubble: MessageBubble) -> None:
        try:
            content = bubble.content
            cid = content.get("call_id") if isinstance(content, dict) else ""
            if cid:
                self._tool_index[str(cid)] = bubble
            if isinstance(content, dict) and content.get("tool") == "task":
                meta = content.get("metadata")
                sid = meta.get("sessionId") if isinstance(meta, dict) else ""
                if sid:
                    self._task_index[str(sid)] = bubble
        except Exception:
            pass

    def note_tool_metadata(self, bubble: MessageBubble) -> None:
        """Re-index after set_tool_metadata (e.g. task row gains sessionId)."""
        self._index_tool_bubble(bubble)

    def find_tool(self, tool: str, call_id: str = "") -> MessageBubble | None:
        if call_id:
            hit = self._indexed(self._tool_index.get(str(call_id)))
            if hit is not None:
                return hit
        candidates = []
        for child in self.query(MessageBubble):
            if child.role == "tool" and child.content.get("tool") == tool:
                candidates.append(child)
        if not candidates:
            return None
        if not call_id:
            # no call id to match: the most RECENT row is the one still being
            # updated by the engine; candidates[0] stamped running/completed
            # status onto the OLDEST same-name row instead
            return candidates[-1]
        for child in candidates:
            if child.content.get("call_id") == call_id:
                self._tool_index[str(call_id)] = child
                return child
        return None

    def find_task(self, session_id: str) -> MessageBubble | None:
        """The task row whose metadata.sessionId matches a child session id.
        Mirrors the reverse lookup opencode does when a sub-agent's events
        bubble up to the parent chat."""
        hit = self._indexed(self._task_index.get(str(session_id)))
        if hit is not None:
            return hit
        for child in self.query(MessageBubble):
            if child.role != "tool" or child.content.get("tool") != "task":
                continue
            meta = child.content.get("metadata")
            if isinstance(meta, dict) and meta.get("sessionId") == session_id:
                self._task_index[str(session_id)] = child
                return child
        return None

    def tool_runs(self) -> list[dict[str, Any]]:
        """Every tool run currently shown in this chat (opencode's per-session
        tool-part store; the Subagent parent row reads the child's to show the
        live current tool / toolcall count)."""
        return [
            child.content
            for child in self.query(MessageBubble)
            if child.role == "tool" and isinstance(child.content, dict)
        ]

    def update_tool_bubble(self, tool_run: dict[str, Any]) -> None:
        bubble = self.find_tool(tool_run.get("tool", ""), tool_run.get("call_id", ""))
        if bubble:
            bubble.update_tool(tool_run)
            self._auto_scroll()
            return True
        return False

    def last_reasoning(self) -> MessageBubble | None:
        hit = self._indexed(self._last_reasoning_bubble)
        if hit is not None:
            return hit
        found = None
        for child in self.query(MessageBubble):
            if child.role == "reasoning":
                found = child
        self._last_reasoning_bubble = found
        return found

    def toggle_last_reasoning(self) -> None:
        bubble = self.last_reasoning()
        if bubble is not None:
            bubble.expanded = not bubble.expanded
            bubble.focus()

    def reasoning_bubbles(self) -> list[MessageBubble]:
        """Every thought bubble in this chat, oldest first. Never raises."""
        try:
            return [c for c in self.query(MessageBubble) if c.role == "reasoning"]
        except Exception:
            return []

    def set_all_reasoning(self, expanded: bool) -> int:
        """Expand/collapse every thought bubble. Returns the count changed."""
        n = 0
        try:
            for b in self.reasoning_bubbles():
                try:
                    if b.expanded != expanded:
                        b.expanded = expanded
                        n += 1
                except Exception:
                    continue
        except Exception:
            pass
        return n

    def last_reasoning_text(self) -> str:
        """Plain text of the newest thought (title + body), or ''."""
        try:
            bubble = self.last_reasoning()
            if bubble is None:
                return ""
            summary = reasoning_summary(bubble.content if isinstance(bubble.content, str) else bubble._message)
            title = str(summary.get("title") or "")
            body = str(summary.get("body") or summary.get("text") or bubble._message or "")
            text = (f"{title}\n{body}".strip() if title else body.strip())
            return text
        except Exception:
            return ""

    def watch_messages(self, value: list) -> None:
        self.refresh()
