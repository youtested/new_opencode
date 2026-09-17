"""Native Anthropic Messages API provider (SSE streaming).

Events: message_start / content_block_start / content_block_delta /
content_block_stop / message_delta / message_stop / ping / error.
"""

from __future__ import annotations

import json
from typing import Any, Callable, TYPE_CHECKING

from ..util.net import force_close_response
from ..util.sse import SSEDecoder
from .base import ProviderError, ProviderEvent, RateLimitError, StreamInterrupted, ToolCall, Usage

if TYPE_CHECKING:  # pragma: no cover - annotations only, httpx stays lazy at runtime
    import httpx

# httpx is heavy (~1s + ~20MB on slow 32-bit setups); mirror openai_compat's
# lazy pattern so importing this module never pays that on launch.
DEFAULT_TIMEOUT = None  # built lazily in _httpx()
_DEFAULT_TIMEOUT_CONFIG = dict(connect=10.0, read=90.0, write=30.0, pool=10.0)


def _httpx():
    """Lazily import httpx and build the shared default timeout."""
    global DEFAULT_TIMEOUT
    import httpx  # deferred: heavy import

    if DEFAULT_TIMEOUT is None:
        DEFAULT_TIMEOUT = httpx.Timeout(**_DEFAULT_TIMEOUT_CONFIG)
    return httpx


def __getattr__(name: str):
    """Expose ``httpx`` as a lazily-loaded module attribute (PEP 562)."""
    if name == "httpx":
        return _httpx()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

ANTHROPIC_HEADERS = {
    "anthropic-beta": "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
}


class AnthropicProvider:
    def __init__(
        self,
        *,
        id: str = "anthropic",
        name: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        model: str = "",
        is_free: bool = False,
        extra_headers: dict[str, str] | None = None,
        timeout: Any = None,
        reasoning_effort: str | None = None,
    ):
        # official variant parity: adaptive thinking + effort level
        self.reasoning_effort = (reasoning_effort or "").strip().lower() or None
        self.id = id
        self.name = name or id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.is_free = is_free
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        if self.timeout is None:
            self.timeout = _httpx().Timeout(**_DEFAULT_TIMEOUT_CONFIG)
        # Set while a stream is reading so the interrupt path can force-close
        # the connection and unblock a read that would otherwise wait for the
        # model's next token (idle "thinking" gaps never call interrupt_check).
        self._active_resp: Any | None = None

    def abort_stream(self) -> None:
        """Force-close the active HTTP response, if any.

        Called from the interrupt path so a blocked ``iter_bytes`` read wakes up
        immediately even while no chunks are arriving; the reader then rechecks
        the interrupt flag and surfaces ``StreamInterrupted`` instead of a
        network error.
        """
        resp = self._active_resp
        if resp is not None:
            force_close_response(resp)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "Accept": "text/event-stream",
        }
        headers.update(ANTHROPIC_HEADERS)
        headers.update(self.extra_headers)
        return headers

    def _url(self) -> str:
        return f"{self.base_url}/messages"

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Normalize the engine's OpenAI-style history into Anthropic's schema:
        # system -> `system` param; assistant tool_calls -> tool_use content
        # blocks (with their text); tool results -> `user` messages with
        # tool_result blocks that carry the original tool_call_id. Consecutive
        # tool results (the engine emits one message per call) are merged into a
        # single `user` message, because Anthropic rejects repeated same-role
        # messages and a tool_result must pair with its tool_use id.
        from ..agent.parse import anthropic_assistant_from_calls, parse_openai_tool_calls

        system_parts: list[str] = []
        body_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_parts.append(part.get("text", ""))
                continue
            if role == "tool":
                if isinstance(content, list):
                    text = " ".join(
                        str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
                    )
                else:
                    text = str(content or "")
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": str(msg.get("tool_call_id") or ""),
                    "content": text,
                }
                prev = body_messages[-1] if body_messages else None
                if (
                    prev is not None
                    and prev.get("role") == "user"
                    and isinstance(prev.get("content"), list)
                    and prev["content"]
                    and prev["content"][0].get("type") == "tool_result"
                ):
                    prev["content"].append(block)
                else:
                    body_messages.append({"role": "user", "content": [block]})
                continue
            if role == "assistant" and msg.get("tool_calls"):
                calls = parse_openai_tool_calls(msg)
                try:
                    from .base import sanitize_function_name as _san

                    for _c in calls:
                        if _c.get("name"):
                            _c["name"] = _san(_c["name"])
                except Exception:
                    pass
                converted = anthropic_assistant_from_calls(calls)
                text = msg.get("content") or ""
                if text:
                    converted["content"] = [{"type": "text", "text": str(text)}] + list(converted["content"])
                body_messages.append(converted)
                continue
            body_messages.append({"role": role, "content": content})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": body_messages,
            "stream": True,
            "max_tokens": 32000,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [_to_anthropic_tool(t) for t in tools]
            # No `tool_choice: auto` — it IS the default. 30B/request saved.
        if self.reasoning_effort:
            payload["thinking"] = {"type": "adaptive", "display": "summarized"}
            payload["effort"] = self.reasoning_effort
        payload.update(kwargs)
        return payload

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_event: Callable[[ProviderEvent], None] | None = None,
        **kwargs: Any,
    ) -> None:
        events: list[ProviderEvent] = []
        sink = on_event or (lambda e: events.append(e))
        # `is_interrupted` is an engine callback, not a request field: pop it
        # before build_payload so it never leaks into the JSON body (httpx
        # raises `TypeError: Object of type function is not JSON serializable`).
        is_interrupted = kwargs.pop("is_interrupted", None)
        payload = self.build_payload(messages, tools, **kwargs)
        decoder = SSEDecoder()
        tool_blocks: dict[int, ToolCall] = {}
        usage = Usage()
        stop_reason = ""
        done_emitted = [False]

        def interrupt_check() -> None:
            if is_interrupted is not None and is_interrupted():
                raise StreamInterrupted()

        from ..util.net import shared_client

        client = shared_client(self.timeout)
        try:
            with client.stream("POST", self._url(), json=payload, headers=self._headers()) as resp:
                # Register before the status check so an interrupt racing
                # connection setup can still force-close the response.
                self._active_resp = resp
                self._check_status(resp)
                try:
                    for chunk in resp.iter_bytes():
                        interrupt_check()
                        for evt in decoder.feed(chunk):
                            self._handle(evt, sink, tool_blocks, usage, done_emitted)
                    # flush any event that arrived without a trailing newline
                    for evt in decoder.close():
                        self._handle(evt, sink, tool_blocks, usage, done_emitted)
                except StreamInterrupted:
                    raise
                except Exception as e:
                    # a forced abort (abort_stream) closed the response while
                    # we were blocked reading — that's an interrupt, not a
                    # network failure
                    if is_interrupted is not None and is_interrupted():
                        raise StreamInterrupted() from e
                    raise
                finally:
                    self._active_resp = None
        except ProviderError:
            raise
        except StreamInterrupted:
            raise
        except _httpx().TimeoutException as e:
            if is_interrupted is not None and is_interrupted():
                raise StreamInterrupted() from e
            raise ProviderError(f"timeout talking to {self.name}: {e}", retryable=True, network=True) from e
        except _httpx().HTTPError as e:
            if is_interrupted is not None and is_interrupted():
                raise StreamInterrupted() from e
            raise ProviderError(f"network error talking to {self.name}: {e}", retryable=True, network=True) from e
        except Exception as e:
            # httpx stream errors (StreamClosed / StreamConsumed / a forced
            # close racing the interrupt flag) are RuntimeErrors, not
            # HTTPError subclasses — never let one leak raw out of here.
            if is_interrupted is not None and is_interrupted():
                raise StreamInterrupted() from e
            raise ProviderError(f"error talking to {self.name}: {e}", retryable=True) from e

        if tool_blocks:
            sink(ProviderEvent(kind="tool_call", tool_calls=list(tool_blocks.values())))
        if usage.raw or usage.total_tokens or usage.input_tokens or usage.output_tokens:
            sink(ProviderEvent(kind="usage", usage=usage))
        if not done_emitted[0]:
            sink(ProviderEvent(kind="done", finish_reason=stop_reason or "stop"))

    def _check_status(self, resp: "httpx.Response") -> None:
        if resp.status_code in (200, 201):
            return
        body = ""
        try:
            body = resp.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            try:
                ra = float(retry_after) if retry_after else None
            except ValueError:
                ra = None
            raise RateLimitError(f"{self.name} rate limited (429): {body}", retry_after=ra)
        if resp.status_code >= 500:
            raise ProviderError(
                f"{self.name} server error ({resp.status_code}): {body}", retryable=True, status=resp.status_code
            )
        raise ProviderError(f"{self.name} error ({resp.status_code}): {body}", status=resp.status_code)

    def _handle(
        self,
        evt: dict,
        sink: Callable[[ProviderEvent], None],
        tool_blocks: dict[int, ToolCall],
        usage: Usage,
        done_emitted: list[bool] | None = None,
    ) -> None:
        event_type = evt.get("event", "message")
        data = evt.get("data", "")
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return

        if event_type == "message_start":
            msg = obj.get("message", {})
            u = msg.get("usage", {})
            usage.input_tokens = int(u.get("input_tokens", 0) or 0)
            usage.output_tokens = int(u.get("output_tokens", 0) or 0)
            usage.total_tokens = usage.input_tokens + usage.output_tokens
            usage.raw = dict(u) if isinstance(u, dict) else {}
            return
        if event_type == "message_delta":
            delta = obj.get("delta", {})
            if delta.get("stop_reason"):
                if done_emitted is not None:
                    done_emitted[0] = True
                sink(ProviderEvent(kind="done", finish_reason=delta["stop_reason"]))
            u = obj.get("usage", {})
            if isinstance(u, dict) and u:
                usage.output_tokens = int(u.get("output_tokens", 0) or 0)
                usage.total_tokens = usage.input_tokens + usage.output_tokens
                merged = dict(usage.raw) if isinstance(usage.raw, dict) else {}
                merged.update(u)
                usage.raw = merged
            return
        if event_type == "content_block_start":
            block = obj.get("content_block", {})
            index = int(obj.get("index", 0))
            if block.get("type") == "tool_use":
                start_input = block.get("input") or {}
                tool_blocks[index] = ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    # When the input is streamed, content_block_start carries an
                    # EMPTY object and the real JSON arrives via input_json_delta.
                    # Seed with "" so the deltas accumulate into one clean JSON
                    # string instead of concatenating onto "{}" (which would
                    # make parse_arguments fail and tools receive garbage).
                    arguments="" if not start_input else json.dumps(start_input),
                    index=index,
                )
            return
        if event_type == "content_block_delta":
            index = int(obj.get("index", 0))
            delta = obj.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                sink(ProviderEvent(kind="text_delta", text=delta.get("text", "")))
            elif dtype == "thinking_delta":
                sink(ProviderEvent(kind="reasoning_delta", text=delta.get("thinking", "")))
            elif dtype == "input_json_delta" and index in tool_blocks:
                tool_blocks[index].arguments += delta.get("partial_json", "")
            return
        if event_type == "error":
            err = obj.get("error", {})
            message = err.get("message", "unknown error")
            sink(ProviderEvent(kind="error", error=message))


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI-style tool schema to Anthropic tool schema.

    Raises a NON-retryable ProviderError on malformed schemas so one bad
    tool definition surfaces immediately instead of burning retries.
    """
    try:
        from .base import sanitize_function_name as _san
    except Exception:  # pragma: no cover
        _san = lambda s: s  # noqa: E731
    fn = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(fn, dict):
        raise ProviderError(
            f"malformed tool schema (missing function object): {str(tool)[:120]}",
            retryable=False,
        )
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProviderError(
            f"malformed tool schema (missing function.name): {str(tool)[:120]}",
            retryable=False,
        )
    desc = fn.get("description", "")
    if not isinstance(desc, str):
        desc = "" if desc is None else str(desc)
    params = fn.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    return {
        "name": _san(name),
        "description": desc,
        "input_schema": params,
    }
