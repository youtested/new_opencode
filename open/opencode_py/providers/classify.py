"""Provider error classification (mirrors upstream opencode's provider-error.ts).

A single source of truth for deciding what a provider error *is* so every layer
(the SSE handler, the rotation wrapper, the agent loop) agrees:

- ``is_context_overflow``: the conversation exceeded the model's context window.
  Broad regex coverage of the messages real gateways send (OpenAI, Anthropic,
  OpenRouter, Google, the Zen router, deepseek, …) instead of a few substrings.
- ``is_rate_limit``: a real rate limit / quota exhaustion (the user's lane is
  genuinely done and rotation should move on), kept distinct from overflow.

Crucially ``is_context_overflow`` EXCLUDES rate-limit wording (``rate limit``,
``too many requests``, …) so a 429-style message can never be misread as an
overflow and wrongly trigger compaction instead of failover.
"""

from __future__ import annotations

import re

# Mirrors upstream opencode's context-overflow pattern list (packages/llm/
# src/provider-error.ts) so the same real-world gateway messages are caught.
_CONTEXT_OVERFLOW_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"prompt is too long",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds?(?: the)?(\s+the)? context window(?: of [\d,]+ tokens?)?",
        r"exceeds(?: the)? context size",
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))?",
        r"input token count.*exceeds(?: the)? max(?:imum)?",
        r"tokens in request more than max tokens allowed",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens",
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"context[_ ]?length[_ ]?exceeded",
        r"request entity too large",
        r"context length is only \d+ tokens",
        r"input length.*exceeds.*context length",
        r"prompt too long; exceeded (?:max )?context length",
        r"too large for model with \d+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens",
        r"model_context_window_exceeded",
        r"too many tokens",
        r"token limit exceeded",
        r"context_length_exceeded",
        r"context length",
        r"reduce_other_history",
    )
)

# Messages that merely LOOK like overflow but are really rate limiting.
_CONTEXT_OVERFLOW_EXCLUSIONS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(throttling error|service unavailable):",
        r"rate limit",
        r"too many requests",
        r"quota",
        r"insufficient",
    )
)

# Messages that identify an in-band provider error as a rate limit / quota
# exhaustion (the "reached your limit" case). Everything else is transient.
_RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    "rate",
    "quota",
    "limit",
    "429",
    "too many",
    "throttl",
    "insufficient",
    "tokens per minute",
    "requests per minute",
)


def is_context_overflow(message: str | None) -> bool:
    """True when a provider error means the history overflowed the window."""
    low = (message or "").lower()
    if not low:
        return False
    if any(p.search(low) for p in _CONTEXT_OVERFLOW_EXCLUSIONS):
        return False
    return any(p.search(low) for p in _CONTEXT_OVERFLOW_PATTERNS)


def is_rate_limit(message: str | None) -> bool:
    """True when an in-band error is a real rate limit / quota exhaustion."""
    low = (message or "").lower()
    return any(m in low for m in _RATE_LIMIT_PATTERNS)