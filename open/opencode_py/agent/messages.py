"""History building + context trimming."""

from __future__ import annotations

import json
from typing import Any

from ..util.truncate import cached_tokens, estimate_tokens


def build_messages(
    *,
    history: list[dict[str, Any]],
    user_text: str,
    reminder: str | None = None,
) -> list[dict[str, Any]]:
    """Build the OpenAI-style message payload.

    `history` is the prior conversation as OpenAI-style messages. `user_text` is
    the new turn. If `reminder` (plan/build-switch system-reminder) is given it is
    appended to the user message content (mirrors opencode's synthetic part).
    """
    user_content = user_text
    if reminder:
        user_content = f"{user_text}\n\n{reminder}"

    messages = list(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def repair_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make a request well-formed wrt. OpenAI tool-call pairing.

    An interrupted/force-killed turn can leave history ending in an assistant
    message that declares `tool_calls` with no following `tool` results. Strict
    backends (e.g. the Zen Console gateway) reject that payload with
    "insufficient tool messages following tool_calls". Walk the list and, for
    every assistant message whose declared call ids are not answered by the
    subsequent tool messages, insert placeholder tool results (before the next
    non-tool message, or at the end) so the request the provider sees is always
    valid. Runs on the local request copy only — history is untouched.
    """
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        return messages
    out: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []  # (call_id, tool name) still owed answers
    inserted = False

    def _close_pending() -> None:
        nonlocal inserted
        if not pending:
            return
        for call_id, name in pending:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name or "tool",
                    "content": "[interrupted — tool result missing]",
                }
            )
            inserted = True
        pending.clear()

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            if call_id:
                pending[:] = [p for p in pending if p[0] != call_id]
            out.append(msg)
            continue
        _close_pending()
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                pending.append((str(call.get("id") or ""), str(fn.get("name") or "")))
        out.append(msg)
    _close_pending()
    return out if inserted else messages


def trim_history(history: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Drop oldest messages until the token estimate fits the budget.

    Always keeps at least the last 2 messages (a user + assistant pair) unless
    the history is smaller. O(n): message sizes are computed once and dropped
    oldest-first by a running total — the previous implementation re-summed the
    whole list (and popped from the front) on every drop, going O(n²) and
    stalling every request ~7s on large near-capacity conversations.
    """
    if budget <= 0:
        return history
    n = len(history)
    if n <= 2:
        return history
    sizes = [cached_tokens(m, lambda m=m: estimate_tokens(_text(m))) for m in history]
    total = sum(sizes)
    if total <= budget:
        return history
    dropped = 0
    while total > budget and n - dropped > 2:
        total -= sizes[dropped]
        dropped += 1
    if dropped:
        return list(history[dropped:])
    return history


def _text(message: dict[str, Any]) -> str:
    """Serialize a message to text for token estimation.

    Counts content, reasoning, tool_calls (in assistant messages) and tool
    results so trimming under a tight context budget doesn't overtrim — or
    worse, underestimate real usage enough to overflow the window.
    """
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")
    text_parts: list[str] = []
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        text_parts.append(
            " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
            )
        )
    else:
        text_parts.append(str(content))
    text_parts.append(reasoning)

    # tool_calls embedded in assistant messages (large JSON blobs)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        try:
            text_parts.append(json.dumps(tool_calls, sort_keys=True))
        except (TypeError, ValueError):
            text_parts.append(str(tool_calls))

    # tool result metadata that carries substance (id, name)
    for key in ("tool_call_id", "name"):
        val = message.get(key)
        if val:
            text_parts.append(str(val))

    return " ".join(p for p in text_parts if p is not None and p != "")
