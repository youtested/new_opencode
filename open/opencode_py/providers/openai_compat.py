"""OpenAI-compatible chat completions provider with SSE streaming.

One class parameterized by base_url + api_key + model, reused by Groq, Cerebras,
OpenRouter, Google AI Studio, NVIDIA, Mistral, GitHub Models, Together, SambaNova,
OpenAI, DeepSeek, and local Ollama (/v1/chat/completions).

Streams `chat.completion.chunk` events; accumulates tool_calls deltas.
"""

from __future__ import annotations

import json
from typing import Any, Callable, TYPE_CHECKING

from ..util.net import force_close_response, shared_client
from ..util.sse import SSEDecoder
from .base import ContextOverflowError, ProviderError, ProviderEvent, RateLimitError, StreamInterrupted, ToolCall, Usage
from .classify import is_context_overflow as _classify_context_overflow

if TYPE_CHECKING:  # pragma: no cover - annotations only, httpx stays lazy at runtime
    import httpx

# httpx is a heavy import (~1s on slow 32-bit setups); the provider only needs
# it when a request actually streams, so it's loaded lazily on first use. The
# default timeout mirrors curl/httpx defaults.
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
    """Expose ``httpx`` as a lazily-loaded module attribute (PEP 562) so
    callers and tests can reference ``openai_compat.httpx`` without forcing
    the heavy import until a stream actually runs."""
    if name == "httpx":
        return _httpx()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _safe_int(value: object) -> int:
    try:
        n = int(value or 0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            n = int(float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
    return n if n > 0 else 0


def _content_to_text(content: Any) -> str:
    """Normalize OpenAI-compat delta.content (a string, or a list of content
    parts used by some gateways) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                else:
                    parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _is_context_overflow_message(message: str) -> bool:
    """True when a provider error means the history overflowed the window.

    Delegates to the shared classifier (mirrors upstream opencode's pattern
    list) so the SSE handler and the rotation wrapper stay in agreement.
    """
    return _classify_context_overflow(message or "")


class OpenAICompatProvider:
    """OpenAI /v1/chat/completions streaming provider."""

    def __init__(
        self,
        *,
        id: str = "openai",
        name: str | None = None,
        base_url: str,
        api_key: str | None = None,
        model: str = "",
        is_free: bool = False,
        extra_headers: dict[str, str] | None = None,
        timeout: "httpx.Timeout | None" = None,
        include_usage: bool = False,
        reasoning_passthrough: bool = False,
        reasoning_effort: str | None = None,
    ):
        # Reasoning effort (official variant parity): e.g. "high" for
        # muse-spark. Sent as reasoning.effort (+ reasoningEffort for
        # OpenAI-compat gateways); empty/None sends nothing.
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
        # Some OpenAI-compatible gateways reject the non-standard
        # `stream_options` field; disable it per-provider when needed.
        self.include_usage = include_usage
        # Thinking-mode gateways (e.g. Zen's Console router) require the
        # `reasoning_content` key present on every reassembled assistant
        # message. Strict endpoints (OpenAI official) instead REJECT unknown
        # message fields — so the default-on wire injection is opt-in.
        self.reasoning_passthrough = reasoning_passthrough
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
        force_close_response(self._active_resp)

    # -- helpers ---------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Strip internal-only keys (compaction marker, session ids, …) from the
        # messages before they hit the wire: they're bookkeeping for the local
        # history/TUI, not fields the API knows, and strict gateways reject
        # unknown fields on the request body.
        wire_messages = []
        for m in messages:
            wire = {k: v for k, v in m.items() if not k.startswith("_")}
            wire.pop("compaction", None)
            wire.pop("id", None)
            wire.pop("has_messages", None)
            # Local-only bookkeeping: never meaningful to the model, pure
            # upload bytes on every turn (task_children grows with every
            # sub-agent; session_id/sessionId ride every task result).
            wire.pop("task_children", None)
            wire.pop("session_id", None)
            wire.pop("sessionId", None)
            wire.pop("sid", None)
            if wire.get("role") == "tool":
                # `name` duplicates `tool_call_id` routing (gateway-verified:
                # tool results route by id alone; OpenAI ignores `name` on
                # tool messages). Kept in local history, dropped on the wire.
                wire.pop("name", None)
            if wire.get("role") == "assistant":
                # Zen-lane-only trims (live-verified 200s on the gateway):
                # - per-item `type: function` (OpenAI-official requires it,
                #   so strict non-Zen lanes keep it)
                # - empty `content: ""` on tool-call messages (means "no text",
                #   identical to a missing key on the wire)
                # `reasoning_content` is NEVER trimmed: thinking-mode lanes
                # 400 without it (see reasoning_passthrough).
                if self.id == "opencode":
                    if wire.get("content") == "" and wire.get("tool_calls"):
                        wire.pop("content", None)
                    for tc in wire.get("tool_calls") or []:
                        if isinstance(tc, dict) and tc.get("type") == "function":
                            tc.pop("type", None)
                # Strict validators (Nemotron 400 on "mcp:n") reject the whole
                # request for one bad history function name — sanitize replayed
                # names so an old session with a colon-named tool still sends.
                # Deep-copy the calls first: `wire` is a shallow dict copy, so
                # mutating the shared list would permanently rewrite history.
                if wire.get("tool_calls"):
                    import copy as _copy

                    wire["tool_calls"] = _copy.deepcopy(wire["tool_calls"])
                    for tc in wire.get("tool_calls") or []:
                        if isinstance(tc, dict):
                            fn = tc.get("function")
                            if isinstance(fn, dict) and fn.get("name"):
                                from .base import sanitize_function_name as _san

                                fn["name"] = _san(fn["name"])
            if wire.get("role") == "assistant":
                # Thinking-mode gateways (e.g. Zen's Console router) require the
                # `reasoning_content` field present on every reassembled
                # assistant message and reject history that lacks it
                # ("reasoning_content in the thinking mode must be passed
                # back") — those lanes set `reasoning_passthrough`. Strict
                # endpoints (OpenAI official) reject unknown message fields,
                # so everywhere else the key is only sent when the model
                # actually produced reasoning for it.
                if self.reasoning_passthrough:
                    wire.setdefault("reasoning_content", "")
                elif not wire.get("reasoning_content"):
                    wire.pop("reasoning_content", None)
            wire_messages.append(wire)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "stream": True,
        }
        if self.include_usage:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            # No `tool_choice: auto` — it IS the default (live-verified:
            # identical behavior without it). 23B saved per request.
        if self.reasoning_effort:
            # official variant merge: reasoning.effort (+ compat alias)
            payload["reasoning"] = {"effort": self.reasoning_effort}
            payload["reasoningEffort"] = self.reasoning_effort
        payload.update(kwargs)
        return payload

    # -- streaming -------------------------------------------------------
    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_event: Callable[[ProviderEvent], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Stream a chat completion; dispatch ProviderEvent to on_event.

        If on_event is None, collects and returns via a list (convenience).
        """
        events: list[ProviderEvent] = []
        sink = on_event or (lambda e: events.append(e))
        self._stream(messages, tools, sink, **kwargs)
        if on_event is None:
            return events  # type: ignore[return-value]

    def _interrupt_check(self, is_interrupted: Callable[[], bool] | None) -> None:
        if is_interrupted is not None and is_interrupted():
            raise StreamInterrupted()

    def _stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        sink: Callable[[ProviderEvent], None],
        **kwargs: Any,
    ) -> None:
        # `is_interrupted` is an engine callback, not a request field: pull it
        # out before build_payload or it leaks into the JSON body and httpx
        # fails to serialize it (`TypeError: Object of type function is not
        # JSON serializable`), killing the whole stream.
        is_interrupted = kwargs.pop("is_interrupted", None)
        payload = self.build_payload(messages, tools, **kwargs)
        decoder = SSEDecoder()
        tool_calls: dict[int, ToolCall] = {}
        tool_ids: dict[str, int] = {}
        usage: Usage = Usage()
        done_emitted = [False]
        httpx = _httpx()

        # Pooled client shared process-wide per timeout shape: parallel
        # agents reuse keep-alive connections instead of paying N TLS
        # handshakes. Never closed here (the pool owns it); abort still
        # force-closes only the in-flight response.
        client = shared_client(self.timeout)
        try:
            with client.stream("POST", self._url(), json=payload, headers=self._headers()) as resp:
                # Register before the status check so an interrupt racing
                # connection setup can still force-close the response.
                self._active_resp = resp
                self._check_status(resp)
                try:
                    for chunk in resp.iter_bytes():
                        self._interrupt_check(is_interrupted)
                        for evt in decoder.feed(chunk):
                            self._handle_event(evt, sink, tool_calls, usage, done_emitted, tool_ids=tool_ids)
                    # flush any event that arrived without a trailing newline,
                    # otherwise the final chunk's content is silently dropped
                    for evt in decoder.close():
                        self._handle_event(evt, sink, tool_calls, usage, done_emitted, tool_ids=tool_ids)
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
        except httpx.TimeoutException as e:
            if is_interrupted is not None and is_interrupted():
                raise StreamInterrupted() from e
            raise ProviderError(f"timeout talking to {self.name}: {e}", retryable=True, network=True) from e
        except httpx.HTTPError as e:
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

        if tool_calls:
            sink(ProviderEvent(kind="tool_call", tool_calls=list(tool_calls.values())))
        if usage.raw or usage.total_tokens or usage.input_tokens or usage.output_tokens:
            sink(ProviderEvent(kind="usage", usage=usage))
        if not done_emitted[0]:
            sink(ProviderEvent(kind="done", finish_reason="stop"))

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
        if _is_context_overflow_message(body):
            raise ContextOverflowError(f"{self.name} context overflow: {body}", status=resp.status_code)
        raise ProviderError(f"{self.name} error ({resp.status_code}): {body}", status=resp.status_code)

    def _handle_event(
        self,
        evt: dict,
        sink: Callable[[ProviderEvent], None],
        tool_calls: dict[int, ToolCall],
        usage: Usage,
        done_emitted: list[bool] | None = None,
        tool_ids: dict[str, int] | None = None,
    ) -> None:
        ids = tool_ids if tool_ids is not None else {}
        data = evt.get("data", "")
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        if obj.get("object") == "error" or "error" in obj:
            err = obj.get("error", {})
            message = err.get("message", "unknown error") if isinstance(err, dict) else str(err)
            code = err.get("code", "") if isinstance(err, dict) else ""
            if code == "context_length_exceeded" or _is_context_overflow_message(message):
                sink(ProviderEvent(kind="error", error=f"context overflow: {message}"))
                raise ContextOverflowError(f"{self.name} context overflow: {message}")
            if code in ("insufficient_quota", "server_is_overloaded", "server_error"):
                sink(ProviderEvent(kind="error", error=f"{code}: {message}"))
            else:
                sink(ProviderEvent(kind="error", error=message))
            return

        # usage chunk. Some gateways (e.g. the Zen router) attach a usage
        # object to *every* SSE chunk, not just the final one, so we must NOT
        # return here — the same chunk can also carry a content delta. Only
        # record usage and continue; a pure usage/cost ping has no choices and
        # falls through to the `if not choices` guard below.
        if "usage" in obj and isinstance(obj.get("usage"), dict) and obj.get("usage"):
            u = obj["usage"]
            usage.input_tokens = _safe_int(u.get("prompt_tokens", 0))
            usage.output_tokens = _safe_int(u.get("completion_tokens", 0))
            usage.total_tokens = _safe_int(u.get("total_tokens", 0))
            usage.raw = u

        choices = obj.get("choices") or []
        if not choices:
            # possible cost ping: data: {"choices":[],"cost":...}
            return
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        delta = choice.get("delta") or {}

        if delta.get("role"):
            pass  # first chunk role marker
        if delta.get("content"):
            content = delta["content"]
            if not isinstance(content, str):
                content = _content_to_text(content)
            if content:
                sink(ProviderEvent(kind="text_delta", text=content))
        if delta.get("reasoning_content"):
            sink(ProviderEvent(kind="reasoning_delta", text=delta["reasoning_content"]))
        if delta.get("reasoning"):
            reason = delta["reasoning"]
            if isinstance(reason, str):
                text = reason
            else:
                text = reason.get("content") or reason.get("text") or ""
            if text:
                sink(ProviderEvent(kind="reasoning_delta", text=text))
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                # Slot resolution, strictest-first: an explicit `index` wins
                # (the OpenAI streaming contract); otherwise a provider call
                # id already seen claims its own slot; otherwise a NEW id opens
                # a new slot. This matters because some gateways stream
                # PARALLEL tool calls without any `index` field — keying those
                # by 0 merged every call into one (names concatenated,
                # arguments JSON fused into garbage). A bare continuation chunk
                # (no id AND no index) still appends to the most recent slot.
                raw_index = tc.get("index")
                tid = str(tc.get("id") or "")
                if isinstance(raw_index, int):
                    index = raw_index
                elif tid and tid in ids:
                    index = ids[tid]
                elif tid:
                    index = (max(tool_calls) + 1) if tool_calls else 0
                    ids[tid] = index
                else:
                    index = max(tool_calls) if tool_calls else 0
                call = tool_calls.setdefault(index, ToolCall(id="", name="", arguments="", index=index))
                if tid:
                    call.id = tid
                    ids[tid] = index
                fn = tc.get("function") or {}
                if fn.get("name"):
                    chunk = fn["name"]
                    if not call.name.endswith(chunk):
                        if call.name and chunk.startswith(call.name):
                            # A later chunk re-sent the FULL name with the
                            # accumulated prefix already included (some gateways
                            # re-send cumulative state) — adopt the superset
                            # instead of doubling ("get" + "get_weather" must
                            # yield "get_weather", not "getget_weather").
                            call.name = chunk
                        else:
                            call.name += chunk
                if fn.get("arguments"):
                    call.arguments += fn["arguments"]
        if finish_reason:
            if done_emitted is not None:
                done_emitted[0] = True
            sink(ProviderEvent(kind="done", finish_reason=finish_reason))

    # -- non-streaming convenience --------------------------------------
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Non-streaming completion, returns the JSON response."""
        payload = self.build_payload(messages, tools)
        payload["stream"] = False
        client = shared_client(self.timeout)
        resp = client.post(self._url(), json=payload, headers=self._headers())
        self._check_status(resp)
        try:
            return resp.json()
        except ValueError as e:
            body = ""
            try:
                body = resp.text[:500]
            except Exception:
                pass
            raise ProviderError(
                f"{self.name} returned a non-JSON response: {body or e}", status=resp.status_code
            ) from e
