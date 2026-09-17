"""Provider protocol + streaming events.

Every provider implements `stream_chat(messages, tools, on_event)`. Events
flow to a callback so the same engine drives both the TUI and headless mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, accumulated across deltas
    index: int = 0


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_total(u: Usage) -> int:
    """Official-style context count for one completion.

    Mirrors opencode's overflow math
    (``total || input + output + cache.read + cache.write``): a provider
    total already bundles thinking/cached tokens, so prefer it; otherwise
    sum the parts including Anthropic-style cache tokens (which ride
    outside input/output and were previously dropped, reading low).
    """
    raw = u.raw if isinstance(getattr(u, "raw", None), dict) else {}
    total = _safe_int(getattr(u, "total_tokens", 0)) or _safe_int(raw.get("total_tokens", 0))
    if total:
        return total
    cache_read = _safe_int(raw.get("cache_read_input_tokens", 0))
    cache_write = _safe_int(raw.get("cache_creation_input_tokens", 0))
    details = raw.get("prompt_tokens_details")
    if isinstance(details, dict):
        cache_read += _safe_int(details.get("cached_tokens", 0))
    return _safe_int(u.input_tokens) + _safe_int(u.output_tokens) + cache_read + cache_write


def usage_total_dict(u: dict[str, Any]) -> int:
    """Same official total for the loop's plain-dict `_usage_total` slot."""
    if not isinstance(u, dict):
        return 0
    return _safe_int(u.get("total_tokens", 0)) or (
        _safe_int(u.get("input_tokens", 0)) + _safe_int(u.get("output_tokens", 0))
    )


@dataclass
class ProviderEvent:
    kind: str  # text_delta | reasoning_delta | tool_call | usage | error | done
    text: str = ""
    tool_call: ToolCall | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    error: str = ""
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None,
                 network: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status = status
        # True for transport failures (disconnect, DNS, timeout) as opposed
        # to model/API errors — drives the TUI's auto-resume watcher.
        self.network = network


class RateLimitError(ProviderError):
    def __init__(self, message: str = "rate limit", *, retry_after: float | None = None, status: int | None = 429):
        super().__init__(message, retryable=True, status=status)
        self.retry_after = retry_after


class ContextOverflowError(ProviderError):
    """The model's context window was exceeded.

    Distinct from a hard provider error: the caller (agent loop) can trim the
    history and retry instead of surfacing a scary failure to the user.
    """

    def __init__(self, message: str = "context length exceeded", *, status: int | None = None):
        super().__init__(message, retryable=True, status=status)


class StreamInterrupted(Exception):
    """The user aborted an in-flight stream (e.g. Esc).

    Raised inside a provider's SSE iteration as soon as the interrupt flag
    flips. It is NOT a provider failure: the agent loop must not retry or
    rotate, it just ends the turn as interrupted.
    """


class Provider(Protocol):
    id: str
    name: str
    is_free: bool

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_event: Callable[[ProviderEvent], None],
        **kwargs: Any,
    ) -> None: ...


def sanitize_function_name(name: str) -> str:
    """Wire-safe function name for strict validators (Nemotron 400s on ``:``).

    Only ``a-z, A-Z, 0-9, _, -`` allowed, max 64 chars. Lenient models accept
    anything, strict ones reject the whole request — so sanitize on the wire.
    """
    import re

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name or ""))
    safe = safe.strip("_-") or "tool"
    return safe[:64]


def tool_to_openai_schema(tool) -> dict[str, Any]:
    """Convert a registry Tool to an OpenAI function-calling JSON schema."""
    try:
        from ..tools.registry import _provider_schema as _clean
    except Exception:  # pragma: no cover - import cycle fallback
        _clean = lambda s: s  # noqa: E731
    return {
        "type": "function",
        "function": {
            "name": sanitize_function_name(tool.name),
            "description": tool.description,
            "parameters": _clean(tool.parameters),
        },
    }
