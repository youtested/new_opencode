"""OpenAI Responses API transport (mirrors the official opencode client).

The official client talks to Zen via ``POST {base}/responses`` (not
``/chat/completions``): ``input`` items, ``store: false``, SSE events like
``response.output_text.delta``. Some Zen models ONLY serve one of the two
APIs (muse-spark needs /responses, big-pickle needs /chat), so neither
 endpoint alone covers present — or future — models. This module implements
the Responses side; ``ZenProvider`` tries the cached-preferred endpoint and
falls back to the other one, remembering what worked per model.
"""

from __future__ import annotations

import json
from typing import Any, Callable, TYPE_CHECKING

from ..util.net import force_close_response
from ..util.sse import SSEDecoder
from .base import ContextOverflowError, ProviderError, ProviderEvent, RateLimitError, StreamInterrupted, ToolCall, Usage
from .classify import is_context_overflow as _classify_context_overflow

if TYPE_CHECKING:  # pragma: no cover - annotations only, httpx stays lazy at runtime
    import httpx


class TransportIncompatible(Exception):
    """This model/endpoint pair doesn't speak this API (HTTP 400/404/405/422
    or an in-band unsupported-model error before any output).

    Internal signal only: the caller must silently try the other transport,
    never surface this to the user or the retry banner.
    """


_INCOMPATIBLE_STATUS = (400, 404, 405, 422)

_INCOMPATIBLE_PATTERNS: tuple[str, ...] = (
    "unsupported",
    "not supported",
    "unknown model",
    "does not support",
    "no such model",
    "model not found",
)


def _is_incompatible_message(message: str) -> bool:
    low = (message or "").lower()
    return any(p in low for p in _INCOMPATIBLE_PATTERNS)


def _text_content(content: Any) -> str:
    """Normalize message content (string or parts list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("input_text", "text", "output_text"):
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def build_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style history to Responses ``input`` items.

    - system → ``developer`` role (the API has no system role)
    - user/assistant text → same roles, plain-text content
    - assistant ``tool_calls`` → ``function_call`` items (replay, like the
      official SDK does for prior turns)
    - ``tool`` results → ``function_call_output`` items keyed by call id
    - internal-only keys (``_*``, compaction markers, ``reasoning_content``,
      ``task_children``/``session_id``/``sessionId`` local bookkeeping)
      are dropped: strict gateways reject unknown fields.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            text = _text_content(m.get("content", ""))
            if text:
                out.append({"role": "developer", "content": text})
            continue
        if role == "user":
            text = _text_content(m.get("content", ""))
            out.append({"role": "user", "content": text})
            continue
        if role == "assistant":
            text = _text_content(m.get("content", ""))
            calls = m.get("tool_calls") or []
            if text:
                out.append({"role": "assistant", "content": text})
            for tc in calls:
                from .base import sanitize_function_name as _san

                fn = tc.get("function") or {}
                item: dict[str, Any] = {
                    "type": "function_call",
                    "call_id": tc.get("id") or tc.get("call_id") or "",
                    "name": _san(fn.get("name") or tc.get("name") or ""),
                    "arguments": fn.get("arguments") or tc.get("arguments") or "{}",
                }
                if not item["call_id"]:
                    item.pop("call_id")
                out.append(item)
            continue
        if role == "tool":
            output = _text_content(m.get("content", ""))
            item_out: dict[str, Any] = {
                "type": "function_call_output",
                "call_id": m.get("tool_call_id") or m.get("call_id") or "",
                "output": output,
            }
            if not item_out["call_id"]:
                item_out.pop("call_id")
            out.append(item_out)
            continue
    return out


def build_responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI function schemas to Responses function tools."""
    from .base import sanitize_function_name as _san

    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            out.append(
                {
                    "type": "function",
                    "name": _san(fn.get("name", "")),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        elif isinstance(t, dict) and t.get("type") == "function" and "name" in t:
            out.append(
                {
                    "type": "function",
                    "name": _san(t.get("name", "")),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                }
            )
    return out or None


def build_responses_payload(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    session_key: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Request body mirroring the official client's Responses call."""
    payload: dict[str, Any] = {
        "model": model,
        "input": build_responses_input(messages),
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }
    if session_key:
        payload["prompt_cache_key"] = session_key
    rtools = build_responses_tools(tools)
    if rtools:
        payload["tools"] = rtools
        # No `tool_choice: auto` — it IS the default (live-verified on both
        # /responses and /chat/completions: identical 200s without it).
    effort = str(kwargs.pop("reasoning_effort", "") or "").strip().lower()
    if effort:
        # official parity: the Responses path carries reasoning.effort
        payload["reasoning"] = {"effort": effort}
    payload.update(kwargs)
    return payload


# -- endpoint preference cache (per-model adaptation) --------------------

ENDPOINT_TTL_SECONDS = 24 * 60 * 60

def _endpoint_cache_path():
    from pathlib import Path

    from ..globals import Path as GPath

    return Path(GPath.cache) / "model-endpoint.json"


def get_preferred_endpoint(model_id: str) -> str:
    """Cached working transport for a model: ``"responses"`` or ``"chat"``.

    Entries older than ENDPOINT_TTL_SECONDS are ignored so a model that
    flips transports recovers by itself. Defaults to ``"responses"``
    (the official client's transport). Never raises.
    """
    try:
        import json as _json
        import time as _time

        data = _json.loads(_endpoint_cache_path().read_text(encoding="utf-8"))
        ep = (data or {}).get(str(model_id).split("/", 1)[-1])
        if isinstance(ep, dict):
            ts = ep.get("ts", 0)
            try:
                fresh = _time.time() - float(ts) < ENDPOINT_TTL_SECONDS
            except (TypeError, ValueError):
                fresh = False
            if fresh and ep.get("endpoint") in ("responses", "chat"):
                return ep["endpoint"]
        elif ep in ("responses", "chat"):
            return ep
    except Exception:
        pass
    return "responses"


def set_preferred_endpoint(model_id: str, endpoint: str) -> None:
    """Remember which transport served a model. Never raises."""
    if endpoint not in ("responses", "chat"):
        return
    try:
        import json as _json
        import time as _time

        path = _endpoint_cache_path()
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[str(model_id).split("/", 1)[-1]] = {"endpoint": endpoint, "ts": _time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(data), encoding="utf-8")
    except Exception:
        pass


# -- streaming -----------------------------------------------------------

def _check_responses_status(resp: "httpx.Response", name: str) -> None:
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
        raise RateLimitError(f"{name} rate limited (429): {body}", retry_after=ra)
    if resp.status_code in _INCOMPATIBLE_STATUS:
        raise TransportIncompatible(f"{name}: responses API rejected ({resp.status_code}): {body}")
    if resp.status_code >= 500:
        raise ProviderError(
            f"{name} server error ({resp.status_code}): {body}", retryable=True, status=resp.status_code
        )
    if _classify_context_overflow(body):
        raise ContextOverflowError(f"{name} context overflow: {body}", status=resp.status_code)
    raise ProviderError(f"{name} error ({resp.status_code}): {body}", status=resp.status_code)


class _FunctionCallSlots:
    """Accumulate streaming function_call items (deltas + done payloads)."""

    def __init__(self) -> None:
        self.by_item: dict[str, ToolCall] = {}
        self.order: list[str] = []

    def _slot(self, key: str) -> ToolCall:
        call = self.by_item.get(key)
        if call is None:
            call = ToolCall(id="", name="", arguments="", index=len(self.order))
            self.by_item[key] = call
            self.order.append(key)
        return call

    def arguments_delta(self, item_id: str, delta: str) -> None:
        if delta:
            self._slot(item_id or f"#anon-{len(self.order)}").arguments += delta

    def item_done(self, item: dict[str, Any]) -> None:
        """Merge a completed function_call item (carries full state)."""
        key = str(item.get("call_id") or item.get("id") or "")
        call = self._slot(key or f"#done-{len(self.order)}")
        if item.get("call_id") or item.get("id"):
            call.id = str(item.get("call_id") or item.get("id"))
        if item.get("name"):
            call.name = str(item["name"])
        args = item.get("arguments")
        if isinstance(args, dict):
            args = json.dumps(args)
        if args and len(str(args)) > len(call.arguments):
            # the done payload is cumulative — adopt the superset, mirroring
            # the chat transport's cumulative-name handling
            call.arguments = str(args)

    def values(self) -> list[ToolCall]:
        return [self.by_item[k] for k in self.order if self.by_item[k].name]


def handle_responses_event(
    evt: dict,
    sink: Callable[[ProviderEvent], None],
    slots: _FunctionCallSlots,
    usage: Usage,
    done_emitted: list[bool],
    state: dict[str, Any],
) -> None:
    """Map one Responses SSE event to provider events.

    ``state`` carries ``had_output``/``had_text``/``had_error``/``error_message``
    flags (same contract as the chat transport's rotation wrapper needs).
    Raises TransportIncompatible / ContextOverflowError like the chat handler.
    """
    name = str(evt.get("event", ""))
    try:
        obj = json.loads(evt.get("data", ""))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return
    if not isinstance(obj, dict):
        return

    otype = str(obj.get("type", ""))
    if otype == "error":
        err = obj.get("error", {})
        message = err.get("message", "unknown error") if isinstance(err, dict) else str(err)
        code = err.get("code", "") if isinstance(err, dict) else ""
        state["had_error"] = True
        state["error_message"] = message
        if code == "context_length_exceeded" or _classify_context_overflow(message):
            sink(ProviderEvent(kind="error", error=f"context overflow: {message}"))
            raise ContextOverflowError(f"context overflow: {message}")
        if not state.get("had_output") and _is_incompatible_message(f"{code} {message}"):
            raise TransportIncompatible(f"responses API: {message}")
        sink(ProviderEvent(kind="error", error=message))
        return

    if "output_text.delta" in name or otype == "response.output_text.delta":
        delta = obj.get("delta", "")
        if isinstance(delta, str) and delta:
            state["had_output"] = True
            state["had_text"] = True
            sink(ProviderEvent(kind="text_delta", text=delta))
        return

    if "reasoning" in name or "reasoning" in otype:
        delta = obj.get("delta", "") or obj.get("text", "")
        if isinstance(delta, str) and delta:
            state["had_output"] = True
            sink(ProviderEvent(kind="reasoning_delta", text=delta))
        return

    if otype == "response.function_call_arguments.delta":
        item_id = str(obj.get("item_id") or "")
        slots.arguments_delta(item_id, str(obj.get("delta") or ""))
        state["had_output"] = True
        return

    if otype == "response.output_item.done":
        item = obj.get("item") or {}
        if isinstance(item, dict) and item.get("type") == "function_call":
            slots.item_done(item)
            state["had_output"] = True
        return

    if otype == "response.output_item.added":
        item = obj.get("item") or {}
        if isinstance(item, dict) and item.get("type") == "function_call":
            key = str(item.get("call_id") or item.get("id") or "")
            call = slots._slot(key or f"#added-{len(slots.order)}")
            if item.get("call_id") or item.get("id"):
                call.id = str(item.get("call_id") or item.get("id"))
            if item.get("name"):
                call.name = str(item["name"])
            state["had_output"] = True
        return

    if otype in ("response.completed", "response.failed", "response.incomplete"):
        resp = obj.get("response") or {}
        status = str(resp.get("status") or "")
        u = resp.get("usage") or {}
        if u:
            try:
                usage.input_tokens = int(u.get("input_tokens", 0) or 0)
                usage.output_tokens = int(u.get("output_tokens", 0) or 0)
                usage.total_tokens = int(
                    u.get("total_tokens", 0) or (usage.input_tokens + usage.output_tokens)
                )
                usage.raw = dict(u)
            except (TypeError, ValueError):
                pass
        if otype != "response.completed" or status not in ("completed", ""):
            err = resp.get("error") or obj.get("error") or status or otype
            message = err.get("message") if isinstance(err, dict) else str(err)
            state["had_error"] = True
            state["error_message"] = message
            sink(ProviderEvent(kind="error", error=message))
            return
        if not done_emitted[0]:
            done_emitted[0] = True
            sink(ProviderEvent(kind="done", finish_reason="stop"))
        return


def stream_responses(
    *,
    base_url: str,
    headers: dict[str, str],
    timeout: Any,
    model: str,
    name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    sink: Callable[[ProviderEvent], None],
    session_key: str | None = None,
    is_interrupted: Callable[[], bool] | None = None,
    extra_payload: dict[str, Any] | None = None,
    active_slot: list | None = None,
) -> None:
    """Stream one Responses API turn, same event contract as the chat transport.

    Raises TransportIncompatible (try the other API), RateLimitError,
    ContextOverflowError, StreamInterrupted, or retryable ProviderError.
    Emits a terminal tool_call/usage/done synthesis like the chat transport.
    """
    import httpx as _httpx_mod

    from ..util.net import shared_client

    payload = build_responses_payload(model, messages, tools, session_key, **(extra_payload or {}))
    decoder = SSEDecoder()
    slots = _FunctionCallSlots()
    usage = Usage()
    done_emitted = [False]
    state: dict[str, Any] = {"had_output": False, "had_text": False, "had_error": False, "error_message": ""}
    active: list[Any] = active_slot if active_slot is not None else [None]

    def _check_interrupt() -> None:
        if is_interrupted is not None and is_interrupted():
            raise StreamInterrupted()

    try:
        with shared_client(timeout).stream("POST", base_url.rstrip("/") + "/responses", json=payload, headers=headers) as resp:
            active[0] = resp
            _check_responses_status(resp, name)
            try:
                for chunk in resp.iter_bytes():
                    _check_interrupt()
                    for evt in decoder.feed(chunk):
                        handle_responses_event(evt, sink, slots, usage, done_emitted, state)
                for evt in decoder.close():
                    handle_responses_event(evt, sink, slots, usage, done_emitted, state)
            except (StreamInterrupted, TransportIncompatible, ContextOverflowError):
                raise
            except ProviderError:
                raise
            except Exception as e:
                if is_interrupted is not None and is_interrupted():
                    raise StreamInterrupted() from e
                raise
            finally:
                active[0] = None
    except (ProviderError, StreamInterrupted, TransportIncompatible):
        raise
    except _httpx_mod.TimeoutException as e:
        if is_interrupted is not None and is_interrupted():
            raise StreamInterrupted() from e
        raise ProviderError(f"timeout talking to {name}: {e}", retryable=True, network=True) from e
    except _httpx_mod.HTTPError as e:
        if is_interrupted is not None and is_interrupted():
            raise StreamInterrupted() from e
        raise ProviderError(f"network error talking to {name}: {e}", retryable=True, network=True) from e
    except Exception as e:
        if is_interrupted is not None and is_interrupted():
            raise StreamInterrupted() from e
        raise ProviderError(f"error talking to {name}: {e}", retryable=True) from e

    for call in slots.values():
        sink(ProviderEvent(kind="tool_call", tool_calls=[call]))
    if usage.raw or usage.total_tokens or usage.input_tokens or usage.output_tokens:
        sink(ProviderEvent(kind="usage", usage=usage))
    if not done_emitted[0]:
        sink(ProviderEvent(kind="done", finish_reason="stop"))


def probe_responses(model_id: str, headers: dict[str, str]) -> bool:
    """Minimal non-streaming Responses probe: True when the model answers.

    Raises TransportIncompatible when this model doesn't speak the API (so
    the caller can try Chat Completions); returns False on dead lanes.
    Transport errors propagate (the health layer fail-opens on total outage).
    """
    import httpx as _httpx_mod

    resp = _httpx_mod.post(
        "https://opencode.ai/zen/v1/responses",
        headers=headers,
        json={"model": model_id, "input": [{"role": "user", "content": "hi"}], "store": False, "stream": False},
        timeout=15,
        follow_redirects=True,
    )
    if resp.status_code in (200, 429):
        return True
    if resp.status_code in _INCOMPATIBLE_STATUS:
        raise TransportIncompatible(f"{model_id}: responses probe {resp.status_code}")
    return False
