"""Context compaction (mirrors upstream opencode's summary-based compaction).

Upstream opencode never lets a conversation die mid-turn from an overfilled
context window. Before the window fills it runs a **compaction**: it asks a
model to summarize the conversation (an "anchored summary"), keeps a small tail
of recent turns verbatim, and continues automatically. If the provider still
rejects with a context-length overflow mid-stream, the same compaction is used
to recover instead of surfacing a scary error.

Constants and wording mirror packages/core/src/session/compaction.ts.
"""

from __future__ import annotations

import json
from typing import Any

from ..util.truncate import cached_tokens, estimate_tokens

# Reserve space for the model's reply (input budget = context - this buffer).
COMPACTION_BUFFER = 20_000
# Heavy tool outputs become a placeholder inside the summary.
TOOL_OUTPUT_MAX_CHARS = 2_000
# Note: skill outputs compact like any other tool output (summarized head +
# verbatim tail). No special protection: the summary preserves the task
# context, and pinning 50KB skill bodies verbatim would defeat compaction.

# How many recent user turns are kept verbatim next to the summary.
DEFAULT_TAIL_TURNS = 2
# Budget for the preserved recent tail (proportional to usable context).
MIN_PRESERVE_RECENT_TOKENS = 2_000
MAX_PRESERVE_RECENT_TOKENS = 8_000

AUTO_CONTINUE_HINT = (
    "Continue if you have next steps, or stop and ask for clarification "
    "if you are unsure how to proceed."
)

# Matches upstream compaction.ts SUMMARY_TEMPLATE.
SUMMARY_TEMPLATE = """Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Important Details
- [constraints/preferences, decisions and why, important facts/assumptions, exact context needed to continue, or "(none)"]

## Work State
### Completed
- [finished work, verified facts, or changes made; otherwise "(none)"]

### Active
- [current work, partial changes, or investigation state; otherwise "(none)"]

### Blocked
- [blockers, failing commands, or unknowns; otherwise "(none)"]

## Next Move
1. [immediate concrete action, or "(none)"]
2. [next action if known, or "(none)"]

## Relevant Files
- [file or directory path: why it matters, or "(none)"]
</template>

Rules:
- Keep every section, even when empty.
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers when known.
- Do not mention the summary process or that context was compacted."""


def usable_context(context: int, output_limit: int = 0) -> int:
    """Tokens usable for conversation before compaction.

    Reserves output space so the model still has room to answer after the
    history fills the window. Upstream opencode reserves ``context - max(output,
    buffer)``, but a model's *declared* output limit (e.g. 128k) is the maximum
    it could ever emit, not what it typically writes — reserving all of it would
    starve the conversation (deepseek-v4-flash-free would compact at 36%).
    Instead the reserve is capped at `COMPACTION_BUFFER`: at least the buffer is
    reserved when the lane is unknown, and never more than a small-output lane's
    real limit. The post-overflow recovery path covers the rare genuinely long
    reply. A reserve that swallows the whole window returns 0 (the caller then
    skips proactive compaction and relies on that recovery path).
    """
    if context <= 0:
        return 0
    reserved = min(COMPACTION_BUFFER, output_limit) if output_limit > 0 else COMPACTION_BUFFER
    return max(0, context - reserved)


def estimate_request(messages: list[dict], tools: list[dict]) -> int:
    """Upstream `estimate({system, messages, tools})`: tokens in the request.

    The `messages` list already carries the system prompt (it was prepended by
    the loop), matching upstream counting ``system + messages + tools``. Tool
    schemas are counted too; they are static per agent and only add a few KB.
    """
    total = sum(cached_tokens(m, lambda m=m: estimate_tokens(msg_text(m))) for m in messages if isinstance(m, dict))
    if tools:
        try:
            total += estimate_tokens(json.dumps(tools, sort_keys=True))
        except (TypeError, ValueError):
            total += estimate_tokens(str(tools))
    return total


def is_overflow(
    context: int,
    request_tokens: int,
    output_limit: int = 0,
) -> bool:
    """Upstream `compactIfNeeded`: would a request of `request_tokens` fit?

    Compacts once the request's estimated size exceeds the usable window
    (``context - max(output, buffer)``). When the reserve swallows the whole
    window — a model whose declared output limit is >= its context — the
    request can't be judged proactively, so this returns False and the
    post-overflow recovery path in the loop handles the real overflow.
    """
    if context <= 0:
        return False
    usable = usable_context(context, output_limit)
    if usable <= 0:
        return False
    return request_tokens >= usable


def preserve_recent_budget(context: int, output_limit: int = 0) -> int:
    """Tail budget: keep at least a few thousand recent tokens verbatim."""
    usable = usable_context(context, output_limit)
    if usable <= 0:
        return MIN_PRESERVE_RECENT_TOKENS
    return min(MAX_PRESERVE_RECENT_TOKENS, max(MIN_PRESERVE_RECENT_TOKENS, int(usable * 0.25)))


def estimate_turns(messages: list[dict]) -> list[dict]:
    """Return {start, end} spans for each real user turn (skip compaction)."""
    turns: list[dict] = []
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        if m.get("compaction"):
            continue
        turns.append({"start": i, "end": len(messages), "id": m.get("id", "")})
    for i in range(len(turns) - 1):
        turns[i]["end"] = turns[i + 1]["start"]
    return turns


def select_tail(
    messages: list[dict],
    tail_turns: int = DEFAULT_TAIL_TURNS,
    budget: int | None = None,
    context: int | None = None,
    output_limit: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Split into (head-to-summarize, tail-to-keep-verbatim).

    Mirrors upstream `select()`: walk the last `tail_turns` user turns newest to
    oldest, keeping those that fit the recent-token budget. The kept tail plus a
    summarized head keeps continuity without blowing the window. Falls back to
    dropping everything except the last user turn when nothing fits.

    The tail budget derives from `context` (the active model's window) when a
    budget isn't given explicitly; upstream opencode sizes the preserved tail
    from the usable window of the selected model.
    """
    if tail_turns <= 0:
        return messages, []
    if budget is None:
        budget = preserve_recent_budget(context or 200_000, output_limit)
    all_turns = estimate_turns(messages)
    if not all_turns:
        return messages, []
    recent = all_turns[-tail_turns:]
    total = 0
    keep_start: int | None = None
    for turn in reversed(recent):
        turn_msgs = messages[turn["start"] : turn["end"]]
        size = sum(cached_tokens(m, lambda m=m: estimate_tokens(msg_text(m))) for m in turn_msgs if isinstance(m, dict))
        if total + size <= budget:
            total += size
            keep_start = turn["start"]
        else:
            break
    if keep_start is None:
        # Nothing fit the budget (even the newest turn alone is oversized).
        # Keep only the LAST user turn verbatim so the compacted history (and
        # the token counter the TUI recomputes from it) stays bounded.
        keep_start = recent[-1]["start"]
    if keep_start <= 0:
        # The whole conversation fits the recent-tail budget — there is nothing
        # worth summarizing, so keep it ALL verbatim. An empty head signals the
        # caller (the loop's `if not head: return None`) that no compaction is
        # needed; returning `(messages, [])` here would summarize the entire
        # conversation and drop the verbatim tail even though it would have fit.
        return [], messages
    return messages[:keep_start], messages[keep_start:]


def msg_text(message: dict[str, Any]) -> str:
    """Serialize one chat message to text for token estimation."""
    role = message.get("role")
    if role == "user":
        text = message.get("content", "")
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            return " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in text
            )
        return str(text)
    if role == "assistant":
        parts: list[str] = []
        text = message.get("content", "")
        if text:
            parts.append(str(text) if isinstance(text, str) else json.dumps(text, sort_keys=True))
        if message.get("reasoning_content"):
            parts.append(str(message["reasoning_content"]))
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
        return " ".join(parts)
    if role == "tool":
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
            )
        else:
            text = str(content)
        return f"{message.get('name', 'tool')}: {text}"
    return str(message.get("content", ""))


def serialize_message(m: dict[str, Any]) -> str:
    """One conversation message -> text for the summary request."""
    role = m.get("role", "")
    if role == "user":
        text = m.get("content", "")
        if isinstance(text, list):
            return " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in text
            )
        return str(text)
    if role == "assistant":
        parts: list[str] = []
        if m.get("content"):
            parts.append(str(m["content"]))
        if m.get("reasoning_content"):
            parts.append(f"[reasoning]: {m['reasoning_content']}")
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            parts.append(f"[tool]: {fn.get('name', '')}({fn.get('arguments', '')})")
        return "\n".join(parts)
    if role == "tool":
        content = m.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
            )
        else:
            text = str(content)
        if len(text) > TOOL_OUTPUT_MAX_CHARS:
            text = text[:TOOL_OUTPUT_MAX_CHARS] + "\n[truncated]"
        return f"[tool result]: {text}"
    return json.dumps(m, sort_keys=True, default=str)


def summarize_conversation_prompt(messages: list[dict], tail: list[dict] | None = None) -> str:
    """Build the user prompt handed to the summary model (upstream `buildPrompt`)."""
    head = messages if tail is None else messages
    body_msgs = list(head)
    if tail:
        body_msgs = body_msgs + tail
    body = "\n\n".join(f"[{m.get('role', '')}]: {serialize_message(m)}" for m in body_msgs)
    return (
        "Create a new anchored summary from the conversation above. Preserve "
        "still-true details, remove stale details, and record the current state "
        "and next steps.\n\n"
        + SUMMARY_TEMPLATE
        + "\n\nConversation:\n"
        + body
    )