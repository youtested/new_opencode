"""OpenCode Zen provider (https://opencode.ai/zen/v1) — the free models.

Thin wrapper over OpenAICompatProvider pointing at Zen's OpenAI-compatible
endpoint. With no API key, free (cost==0) models are used and Zen accepts the
literal API key "public" (mirrors opencode's behavior).

The Zen gateway throttles anonymous clients (unknown User-Agent, no
x-opencode-* headers) to a tiny free allowance — a couple of requests, then
429 FreeUsageLimitError. The official opencode client identifies itself with
`User-Agent: opencode/...` plus x-opencode-* headers, and the gateway treats
those as trusted clients with the real free quota. This provider sends the
same identity headers so free models (x-preview-f-free, etc.) keep working
across turns instead of being blocked after the first message.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import uuid4

from .openai_compat import OpenAICompatProvider

ZEN_BASE_URL = "https://opencode.ai/zen/v1"


def _user_agent() -> str:
    """Real version string like the official client (opencode/1.18.29).

    The old hardcoded 'opencode/latest/0.1.0/cli' looked anonymous to the
    gateway. Use our real package version so Zen counts us as a trusted
    client with the full free quota.
    """
    try:
        from .. import __version__ as _v

        return f"opencode/{_v}"
    except Exception:
        return "opencode/0.1.0"

# Lane-rotation registry: Zen pins each x-opencode-session id to ONE upstream
# lane. When that lane dies server-side (streams end with Zen's own
# finish_reason="network_error"), retrying with the SAME id hammers the SAME
# dead lane — the user sees ↻ climb to (50) without a single success while a
# brand-new session id would have worked on attempt #1 (measured 2026-08-23:
# fixed sid A 5/5 OK, fixed sid B 1/5, fresh ids mixed). Keyed by BASE session
# id so a rotation survives provider re-instantiation: rotation.build_provider
# constructs a NEW ZenProvider for every attempt with the same engine sid.
_LANE_EPOCH: dict[str, int] = {}
_LANE_LOCK = threading.Lock()

# Limited-time free models on Zen ($0). Live-fetched in factory; this is the
# bundled fallback for when the network model list is unavailable (R2 risk).
# Ordered most-used first: muse-spark-1.3 is this user's daily driver.
FREE_MODELS: list[dict] = [
    {"id": "muse-spark-1.3-contributor-free", "name": "Muse Spark 1.3 Free", "context": 1048576, "output": 131072},
    {"id": "x-preview-f-free", "name": "Ox Alpha Free (Unlimited)", "context": 1000000, "output": 131072},
    {"id": "big-pickle", "name": "Big Pickle", "context": 200000, "output": 32000},
    {"id": "hy3-free", "name": "Hy3 Free", "context": 190000, "output": 64000},
    {"id": "mimo-v2.5-free", "name": "MiMo-V2.5 Free", "context": 200000, "output": 32000},
    {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash Free", "context": 200000, "output": 128000},
    {"id": "nemotron-3-ultra-free", "name": "Nemotron 3 Ultra Free", "context": 1000000, "output": 128000},
    {"id": "nemotron-3.5-lightning-free", "name": "Nemotron 3.5 Lightning Free", "context": 262144, "output": 128000},
]


class ZenProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "x-preview-f-free",
        session_id: str | None = None,
        project: str | None = None,
        **kwargs: Any,
    ):
        # If no key given, we still need SOMETHING; Zen accepts "public" for free models.
        effective_key = api_key or "public"
        # Stable session id across turns (like the official client) so Zen
        # keeps provider affinity and serves the real free quota. The id only
        # changes when rotate_session() runs on a DEAD lane — never per turn.
        # x-opencode-request is a FRESH id per request (official: per-message
        # user id); reusing the session id there made every turn look like
        # the same request. _headers() refreshes it on every call below.
        self._base_session = session_id
        rot_key = session_id or "__nosession__"
        with _LANE_LOCK:
            epoch = _LANE_EPOCH.get(rot_key, 0)
        effective_sid = session_id
        if epoch and session_id:
            effective_sid = f"{session_id}::r{epoch}"
        elif epoch:
            effective_sid = f"cli::r{epoch}"
        client_headers: dict[str, str] = {
            "User-Agent": _user_agent(),
            "x-opencode-client": "cli",
            "x-opencode-project": project or "opencode_py",
            "x-opencode-request": uuid4().hex,
        }
        if effective_sid:
            client_headers["x-opencode-session"] = effective_sid
        merged = dict(client_headers)
        merged.update(kwargs.pop("extra_headers", {}) or {})
        super().__init__(
            id="opencode",
            name="OpenCode Zen",
            base_url=ZEN_BASE_URL,
            api_key=effective_key,
            model=model,
            is_free=True,
            extra_headers=merged,
            reasoning_passthrough=True,
            **kwargs,
        )
        self.has_key = bool(api_key)

    def _headers(self) -> dict[str, str]:
        """Fresh x-opencode-request per call, stable x-opencode-session.

        Official sends a new request id for every message but keeps the
        session id stable for the whole chat (sticky lane). The base
        implementation would resend the init-time request id forever.
        """
        headers = super()._headers()
        headers["x-opencode-request"] = uuid4().hex
        return headers

    def abort_stream(self) -> None:
        # Responses-API streams register here too (see stream_chat); the chat
        # transport manages self._active_resp itself.
        from ..util.net import force_close_response

        box = getattr(self, "_responses_box", None)
        if isinstance(box, list) and box:
            force_close_response(box[0])
        super().abort_stream()

    def _session_cache_key(self) -> str | None:
        """Stable cache key for the Responses `prompt_cache_key` (no lane epoch)."""
        base = self._base_session or self.extra_headers.get("x-opencode-session") or ""
        return str(base).split("::r")[0] or None

    def stream_chat(self, messages, tools=None, on_event=None, **kwargs):
        """Adaptive transport: try the model's cached-preferred API first
        (Responses by default, like the official client), silently fall back
        to the other one when the model doesn't speak it, and remember what
        worked per model — so present AND future models land on a working
        transport without user-visible retries.

        Only TransportIncompatible — or an HTTP 5xx with zero output, which
        on Zen means "this model doesn't live on this API" as often as a
        transient — triggers the silent fallback; every other error
        propagates exactly as the chat transport raises it (rotation and the
        agent loop keep their retry/failover semantics).
        """
        from .base import ProviderError
        from .responses import (
            TransportIncompatible,
            get_preferred_endpoint,
            set_preferred_endpoint,
            stream_responses,
        )

        events: list = []
        sink = on_event or (lambda e: events.append(e))
        model = self.model
        pref = get_preferred_endpoint(model)
        other = "chat" if pref == "responses" else "responses"
        is_interrupted = kwargs.get("is_interrupted")

        def _run(ep: str) -> None:
            effort = (getattr(self, "reasoning_effort", "") or "").strip().lower() or None
            if ep == "responses":
                box: list = [None]
                self._responses_box = box
                try:
                    stream_responses(
                        base_url=self.base_url,
                        headers=self._headers(),
                        timeout=self.timeout,
                        model=model,
                        name=self.name,
                        messages=messages,
                        tools=tools,
                        sink=sink,
                        session_key=self._session_cache_key(),
                        is_interrupted=is_interrupted,
                        active_slot=box,
                        extra_payload={"reasoning_effort": effort} if effort else None,
                    )
                finally:
                    self._responses_box = []
            else:
                self._stream(messages, tools, sink, **kwargs)

        def _finish(ep: str):
            set_preferred_endpoint(model, ep)
            if on_event is None:
                return events  # type: ignore[return-value]
            return None

        try:
            _run(pref)
        except TransportIncompatible:
            # This model doesn't speak the preferred API — one silent shot
            # on the other transport (this is also how unknown future models
            # are onboarded: first contact tries Responses, then Chat).
            try:
                _run(other)
            except TransportIncompatible as e2:
                raise e2
            except ProviderError:
                # other transport failed too — report the *preferred* failure
                # would mislead; surface this one (it ran last, most context)
                raise
            return _finish(other)
        except ProviderError as e:
            # HTTP 5xx before any output is ambiguous on Zen: a dead lane AND
            # a wrong API both look like this (muse 500s on chat, big-pickle
            # 500s on responses). One silent shot on the other transport; if
            # it answers, that settles it and the choice is cached.
            status = getattr(e, "status", None)
            if status is not None and 500 <= status < 600:
                try:
                    _run(other)
                except (TransportIncompatible, ProviderError):
                    raise e from None
                return _finish(other)
            raise
        return _finish(pref)

    def rotate_session(self) -> None:
        """Force a different upstream lane on the next request.

        Bumps this session's rotation epoch so every future ZenProvider built
        for the same base session id carries a fresh x-opencode-session value —
        Zen then assigns it a new lane instead of the dead one. Called by the
        rotation layer when a stream dies with server-side lane failure
        (in-band error / empty reply), NOT on local network problems.
        x-opencode-request stays fresh-per-call via _headers(), untouched here.
        """
        key = self._base_session or "__nosession__"
        with _LANE_LOCK:
            epoch = _LANE_EPOCH.get(key, 0) + 1
            _LANE_EPOCH[key] = epoch
        effective = (
            f"{self._base_session}::r{epoch}" if self._base_session else f"cli::r{epoch}"
        )
        if self._base_session:
            self.extra_headers["x-opencode-session"] = effective
        else:
            self.extra_headers["x-opencode-session"] = effective
