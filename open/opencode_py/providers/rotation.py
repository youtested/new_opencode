"""Provider factory + registry + failover rotation.

rotation list: [{"provider": "zen", "model": "..."}, {"provider": "groq", "model": "..."}, ...]
On a real rate limit (429 / "limit reached") the engine tries the next lane.
Transient hiccups (timeout, 5xx, overload, empty reply) keep the current model.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from ..config import Config
from .base import ContextOverflowError, ProviderError, ProviderEvent, RateLimitError

if TYPE_CHECKING:  # pragma: no cover - annotations only, httpx stays lazy at runtime
    import httpx

from .tables import FREE_DEFAULT_MODELS, FREE_PROVIDERS, PAID_PROVIDERS


# Keep the historical local name (used across the rotation code) but delegate
# to the shared classifier so detection is identical everywhere.
from .classify import is_context_overflow as _is_context_overflow_message  # noqa: E402


def _is_rate_limit_message(message: str) -> bool:
    from .classify import is_rate_limit

    return is_rate_limit(message)


def _fail_message(provider_id: str, error: Exception) -> str:
    """Message for a surfaced failure; keeps prior lane failures for context."""
    return f"{provider_id}: {error}"


def build_read_timeout(read_seconds: float | None = None, model: str = "") -> httpx.Timeout:
    """httpx timeout with a long configurable read window (streaming).

    The read timeout bounds the gap *between* SSE chunks. Mirrors the
    official client's inter-chunk window (provider.ts chunkTimeout):
    non-thinking models fail fast (45s) so a dead lane surfaces quickly;
    true reasoning models get 300s so long silent thinks still complete.
    Users can raise/lower it via `model_read_timeout` in settings.

    Reasoning models (deepseek-reasoner, gpt-5, claude-sonnet-4, gemini-2.5,
    ...) stream `reasoning_content` and can sit silently for minutes
    between tokens mid-thought. Give those lanes 300s (official parity) so
    a long silent think isn't killed. Non-reasoning models get 45s
    fail-fast. The gap is capped at 1800s. Detection is official-style:
    live catalog `reasoning` flag first, observed reasoning second,
    name fallback last.
    """
    user = 0.0
    try:
        user = float(read_seconds) if read_seconds else 0.0
    except (TypeError, ValueError):
        user = 0.0
    if is_reasoning_model(model):
        base = user if user else 300.0
        read = max(base, 300.0)
    else:
        if user and abs(user - 300.0) > 1e-6:
            read = user
        else:
            read = 45.0
    read = min(read, 1800.0)
    import httpx

    return httpx.Timeout(connect=10.0, read=read, write=30.0, pool=10.0)


_REASONING_MODEL_PATTERN = re.compile(
    r"deepseek-reasoner|deepseek-r1|reasoner|thinking|(?:^|[/\-_])(?:o1|o3|r1)(?:$|[^0-9a-z])|"
    r"gpt-5|claude-sonnet-4|claude-opus|gemini-2\.[5-9]|grok-3-mini|"
    r"x-preview|k2-thinking|k2p",
    re.IGNORECASE,
)

_OBSERVED_REASONING_MODELS: set[str] = set()
_REASONING_FLAG_CACHE: dict[str, bool | None] = {}


def note_reasoning_support(model_id: str) -> None:
    try:
        if model_id:
            _OBSERVED_REASONING_MODELS.add(str(model_id).split("/", 1)[-1].lower())
            _OBSERVED_REASONING_MODELS.add(str(model_id).lower())
    except Exception:
        pass


def _catalog_reasoning_flag(model_id: str) -> bool | None:
    try:
        key = str(model_id).lower()
        if key in _REASONING_FLAG_CACHE:
            return _REASONING_FLAG_CACHE[key]
    except Exception:
        pass
    found: bool | None = None
    try:
        bare = str(model_id).split("/", 1)[-1]
        cands = [bare]
        for suffix in ("-contributor-free", "-free", ":free"):
            if bare.endswith(suffix):
                cands.append(bare[: -len(suffix)])
        catalog = (fetch_catalog().get("opencode") or {}).get("models", {})
        for cand in cands:
            if cand in catalog:
                val = catalog[cand].get("reasoning", None)
                if isinstance(val, bool):
                    found = val
                    break
        if found is None:
            low = bare.lower()
            for mid, entry in catalog.items():
                try:
                    if low.startswith(str(mid).lower()) or str(mid).lower() in low:
                        val = entry.get("reasoning", None)
                        if isinstance(val, bool):
                            found = val
                            break
                except Exception:
                    continue
    except Exception:
        pass
    try:
        _REASONING_FLAG_CACHE[str(model_id).lower()] = found
    except Exception:
        pass
    return found


def is_reasoning_model(model_id: str) -> bool:
    """True for models that stream a reasoning/thinking phase.

    Official parity (provider.ts capabilities.reasoning): catalog flag first,
    observed reasoning_content second (future-proof, no edits), name regex
    fallback last for unknown models.
    """
    if not model_id:
        return False
    try:
        low_full = str(model_id).lower()
        bare_low = str(model_id).split("/", 1)[-1].lower()
        if low_full in _OBSERVED_REASONING_MODELS or bare_low in _OBSERVED_REASONING_MODELS:
            return True
    except Exception:
        pass
    flag = _catalog_reasoning_flag(model_id)
    if isinstance(flag, bool):
        return flag
    return bool(model_id and _REASONING_MODEL_PATTERN.search(model_id))


class Rotation:
    """Try lanes in order on rate limits / dead lanes.

    `session_id` is the stable conversation id sent to Zen as
    `x-opencode-session` (and reused for x-opencode-request). It must stay
    constant across turns so the Zen gateway keeps provider affinity and serves
    the real free quota for a trusted client. Mutable: the engine rebinds it
    when a session is created/switched, and each newly-built provider picks it
    up from here.
    """

    def __init__(self, lanes: list[dict[str, str]], make_provider: Callable[[str, str], Any], session_id: str | None = None):
        self.lanes = lanes
        self.make_provider = make_provider
        self.session_id = session_id or uuid4().hex
        # the provider currently reading a stream — the interrupt path closes it
        # so a blocked read wakes up immediately (idle "thinking" gap)
        self._active_provider: Any | None = None

    def abort(self) -> None:
        """Force-close the active provider stream, if one is being read.

        Lets the interrupt path abort instantly even when the model is silent
        (no chunk arrives to trigger the per-chunk interrupt check)."""
        provider = self._active_provider
        if provider is not None:
            try:
                provider.abort_stream()
            except Exception:
                pass

    def new_turn(self) -> None:
        """Keep the SAME Zen session id across turns (official parity).

        The old code bumped the lane epoch on EVERY turn to draw a fresh
        upstream lane. That broke the server's sticky routing: the official
        client reuses one session id for the whole chat so Zen pins it to
        one healthy lane after the first 200. Fresh ids each turn forced a
        re-pick every turn and raised the chance of landing on a throttled
        lane. Now a no-op: lane changes happen only on real dead lanes via
        _rotate_lane()/rotate_session(). Never raises.
        """

    @staticmethod
    def _rotate_lane(provider: Any) -> None:
        """Ask the provider to drop its current upstream lane assignment.

        Zen pins each session id to one lane; when a stream dies SERVER-SIDE
        (in-band error / empty reply with finish_reason='network_error'),
        retrying with the same identity re-hits the same corpse. Providers
        that expose rotate_session() (ZenProvider) get a fresh identity so
        the next attempt lands on a different lane. No-op elsewhere."""
        fn = getattr(provider, "rotate_session", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    @property
    def first(self) -> Any | None:
        if not self.lanes:
            return None
        l = self.lanes[0]
        return self.make_provider(l.get("provider", "zen"), l.get("model", FREE_DEFAULT_MODELS["zen"]))

    def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        on_event: Callable[[ProviderEvent], None],
        on_notice: Callable[[str, str, str], None] | None = None,
        locked: bool = False,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Stream across lanes; returns (provider_id, model_id) that succeeded.

        `on_notice(provider_id, model, reason)` is called when a lane other
        than the first succeeds (a failover happened), with a short reason for
        the switch so the UI can announce it accurately.

        `locked=True` pins the stream to the FIRST lane only (the model the
        user selected): a rate limit or hard failure surfaces as an error
        instead of failing over to another model — the TUI's red lock dot.
        Retries (transient failures) still run on the same locked lane.

        A lane fails over when it genuinely can't serve: a real rate limit
        (429 or an in-band "limit reached" error), a hard error (bad model
        id, bad key, dead endpoint), or an empty reply (dead handshake —
        the official client never retries empty turns either). Transient
        failures on the user's chosen lane (timeout, 5xx, overload) raise so
        the agent loop's own backoff retry (5 attempts, official parity) can
        wait it out on the same model instead of silently routing elsewhere.
        Temporarily-down backup lanes are skipped so a dead free model never
        blocks the chain.
        """
        errors: list[str] = []
        saw_rate_limit = False
        saw_other = False
        last_reason = ""
        lanes = self.lanes[:1] if locked else self.lanes
        for index, lane in enumerate(lanes):
            provider_id = lane.get("provider", "zen")
            model = lane.get("model", FREE_DEFAULT_MODELS.get(provider_id, FREE_DEFAULT_MODELS["zen"]))
            had_output = False
            had_text = False
            had_error = False
            error_message = ""
            live_streamed = False
            buffered: list[ProviderEvent] = []

            def wrapped(evt: ProviderEvent) -> None:
                nonlocal had_output, had_text, had_error, error_message, live_streamed
                # text, reasoning, or a real tool call = usable output.
                # Reasoning counts too: a reasoning model cut off before any
                # content must not be treated as "empty" and silently dropped.
                if evt.kind in ("text_delta", "reasoning_delta", "tool_call"):
                    had_output = True
                    if evt.kind == "text_delta":
                        had_text = True
                elif evt.kind == "error":
                    had_error = True
                    error_message = evt.error or error_message
                    if live_streamed and had_text:
                        # Real ANSWER TEXT already reached the screen, then the
                        # lane hit a hard in-band error (e.g. free-quota
                        # exhaustion mid-reply). Commit the partial answer and
                        # stop the turn clearly — do NOT replay a generic error,
                        # retry, or fail over (which would glue a backup's
                        # answer onto the partial text). Marked `partial` so the
                        # handler below doesn't re-wrap the message.
                        err = ProviderError(
                            f"{provider_id}: {evt.error or 'error response'}"
                            " — the reply was cut off by an error",
                            retryable=False,
                        )
                        err.partial = True
                        raise err
                    # Reasoning-only partials fall through to the post-stream
                    # classification below: no answer text is on screen yet, so
                    # nothing user-visible is lost by failing this lane and
                    # retrying / rotating (fixes "stops mid-thinking").
                if evt.kind in ("text_delta", "reasoning_delta"):
                    if evt.kind == "reasoning_delta":
                        try:
                            note_reasoning_support(model)
                        except Exception:
                            pass
                    # Live-stream visible tokens so the UI isn't a frozen
                    # spinner until the whole response finishes. Unknown
                    # success = later events (tool calls, usage, done) replay,
                    # but visible content has already reached the user.
                    live_streamed = True
                    on_event(evt)
                    return
                # Buffer non-visible events (tool calls, usage, done, errors)
                # and only replay them once this lane is known to have
                # completed successfully — a failed lane must not leak them.
                buffered.append(evt)

            try:
                provider = self.make_provider(provider_id, model)
                self._active_provider = provider
                try:
                    provider.stream_chat(messages, tools, wrapped, **kwargs)
                finally:
                    self._active_provider = None
                if had_error and not had_text:
                    # the lane errored before producing any answer TEXT — pure
                    # thinking included, since committing it would end the turn
                    # mid-reasoning ("stops in the thinking part") when the
                    # lane could simply be retried or rotated instead
                    self._rotate_lane(provider)
                    if _is_context_overflow_message(error_message):
                        raise ContextOverflowError(f"{provider_id}: {error_message or 'context overflow'}")
                    if _is_rate_limit_message(error_message):
                        raise RateLimitError(f"{provider_id}: {error_message or 'rate limited'}")
                    raise ProviderError(
                        f"{provider_id}: {error_message or 'error response'}",
                        retryable=True,
                    )
                if not had_output:
                    # The lane "succeeded" but produced nothing usable — Zen's
                    # flaky upstream dying mid-handshake looks exactly like
                    # this (finish_reason='network_error', zero events).
                    # Like the official client (which never retries empty
                    # turns), don't burn a visible same-lane retry here: move
                    # on to the next lane silently. Only the last lane raises,
                    # so a single-lane setup still surfaces instead of hanging.
                    self._rotate_lane(provider)
                    if not locked and index < len(lanes) - 1:
                        errors.append(f"{provider_id}: empty response")
                        last_reason = "empty response"
                        saw_other = True
                        continue
                    raise ProviderError(f"{provider_id}: empty response", retryable=True)
                for evt in buffered:
                    on_event(evt)
                if provider_id == "opencode":
                    # Only lanes that actually respond stay pinned/visible:
                    # a real answer outweighs any probe result.
                    try:
                        pin_zen_model_alive(model)
                    except Exception:
                        pass
                if index > 0 and on_notice:
                    on_notice(provider_id, model, last_reason or "provider error")
                return provider_id, model
            except ContextOverflowError as e:
                # History overflowed the window — every lane shares the same
                # oversized history, so rotating would just hit the same wall.
                # Propagate so the caller (agent loop) trims history and retries.
                raise
            except RateLimitError as e:
                if had_text:
                    # Visible ANSWER TEXT already reached the screen, then the
                    # lane hit a limit. Failover would glue the backup's answer
                    # onto the partial text (duplicate response), so commit and
                    # surface. Reasoning-only partials fall through and rotate.
                    raise ProviderError(
                        f"{provider_id}: {e}\n\n(partial answer already shown —"
                        " the reply was cut off by a rate limit)",
                        retryable=False,
                    ) from e
                errors.append(f"{provider_id}: rate limited ({e})")
                saw_rate_limit = True
                last_reason = "rate limited"
                if locked:
                    # locked: never leave the user's selected model
                    raise
                continue
            except ProviderError as e:
                if getattr(e, "partial", False):
                    # already a clear "reply cut off after partial output" — the
                    # live-streamed text stays; never re-wrap or rotate
                    raise
                # Only permanently broken lanes (bad model id / bad key / dead
                # endpoint) rotate. Transient failures (timeout, 5xx, overload,
                # empty reply) must NOT silently move the user off the model
                # they picked, especially the primary lane.
                if had_text:
                    # Visible ANSWER TEXT already reached the screen, then the
                    # lane failed. Retrying or rotating would duplicate the
                    # partial answer, so commit: keep the partial text and
                    # surface the real cause. A lane that only streamed THINKING
                    # falls through — the agent loop's auto-retry ("keep going")
                    # or the next lane takes over instead of the turn dying
                    # silently mid-thought (the reported "stops in the thinking
                    # part" on x-preview-f-free).
                    raise ProviderError(
                        f"{provider_id}: {e}\n\n(partial answer already shown —"
                        " the reply was cut off)",
                        retryable=False,
                    ) from e
                if e.retryable:
                    if index == 0:
                        # the user's chosen model hiccuped — surface the real
                        # cause instead of routing them elsewhere. Keep
                        # retryable=True so the agent loop's own backoff retry
                        # (auto_retry_count) can wait it out on the same model.
                        raise ProviderError(
                            f"{provider_id}: {e}\n\nHint: this looks like a"
                            " temporary issue — add another provider to your"
                            " 'rotation' list to fail over, or wait and retry.",
                            retryable=True,
                        )
                    # a backup lane that's temporarily down: skip it, the next
                    # one may still answer
                    errors.append(f"{provider_id}: {e}")
                    last_reason = (e.message or str(e))[:120]
                    saw_other = True
                    continue
                # hard (non-retryable) error: model id gone, bad key, dead
                # endpoint — this lane can never answer, rotate on
                errors.append(f"{provider_id}: {e}")
                last_reason = (e.message or str(e))[:120]
                if locked:
                    # locked: never leave the user's selected model
                    raise
                continue
        message = (
            "all providers failed:\n" + "\n".join(errors)
            + "\n\nHint: add another provider to your 'rotation' list to fail over,"
            + " or wait and retry."
        )
        if saw_rate_limit and not saw_other:
            # every failure was a rate limit -> surface as a retryable rate-limit error
            raise RateLimitError(message)
        raise ProviderError(message)


def build_provider(
    cfg: Config,
    provider_id: str | None = None,
    model: str | None = None,
    auth=None,
    session_id: str | None = None,
) -> Any:
    """Build a provider instance from config + auth."""
    explicit_provider = provider_id is not None
    provider_id = provider_id or cfg.provider
    model = model or cfg.model
    if not explicit_provider and "/" in model:
        # 'provider/model' shorthand only when the provider isn't already known;
        # avoids breaking model ids that legitimately contain '/' (e.g. OpenRouter).
        provider_id, model = model.split("/", 1)
    elif explicit_provider and model.startswith(provider_id + "/"):
        # Strip the 'provider/' prefix already folded in by _parse_model; it must
        # not be sent to the API (e.g. Zen rejects "opencode/deepseek-v4-flash-free").
        model = model.split("/", 1)[1]
    try:
        key = auth.get(provider_id, cfg.providers) if auth else None
    except TypeError:
        key = auth.get(provider_id) if auth else None
    if key is None:
        try:
            custom = cfg.providers.get(provider_id)
        except Exception:
            custom = None
        if isinstance(custom, dict):
            key = custom.get("api_key") or custom.get("apiKey")

    timeout = build_read_timeout(getattr(cfg, "model_read_timeout", None), model)
    # Per-model reasoning effort (official variant parity): only sent when the
    # level is valid for THIS model per the live catalog — future models with
    # new levels work with zero code changes; fixed thinkers get nothing.
    effort: str | None = None
    try:
        want = str(getattr(cfg, "reasoning_effort", "") or "").strip().lower()
        if want and want in model_effort_levels(model, provider_id):
            effort = want
    except Exception:
        effort = None

    if provider_id == "opencode":
        from .zen import ZenProvider

        return ZenProvider(api_key=key, model=model, timeout=timeout, session_id=session_id,
                         reasoning_effort=effort) if effort else ZenProvider(api_key=key, model=model, timeout=timeout, session_id=session_id)
    if provider_id == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(model=model, timeout=timeout)
    if provider_id == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(api_key=key, model=model, timeout=timeout,
                                reasoning_effort=effort) if effort else AnthropicProvider(api_key=key, model=model, timeout=timeout)
    if provider_id == "openai":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            id="openai",
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=key,
            model=model,
            is_free=False,
            timeout=timeout,
            reasoning_effort=effort,
        )
    if provider_id in FREE_PROVIDERS:
        from .openai_compat import OpenAICompatProvider

        info = FREE_PROVIDERS[provider_id]
        return OpenAICompatProvider(
            id=provider_id,
            name=info["name"],
            base_url=info["base_url"],
            api_key=key,
            model=model,
            is_free=True,
            extra_headers=info.get("headers", {}),
            timeout=timeout,
            reasoning_effort=effort,
        )
    if provider_id in PAID_PROVIDERS:
        from .openai_compat import OpenAICompatProvider

        info = PAID_PROVIDERS[provider_id]
        return OpenAICompatProvider(
            id=provider_id,
            name=info["name"],
            base_url=info["base_url"],
            api_key=key,
            model=model,
            is_free=False,
            extra_headers=info.get("headers", {}),
            timeout=timeout,
            reasoning_effort=effort,
        )
    # custom provider from config providers.<id>
    custom = cfg.providers.get(provider_id)
    if custom and isinstance(custom, dict):
        from .openai_compat import OpenAICompatProvider

        base_url = custom.get("base_url") or custom.get("api")
        api_key = custom.get("api_key") or key
        if not base_url:
            raise ProviderError(f"provider {provider_id}: no base_url configured")
        return OpenAICompatProvider(
            id=provider_id,
            name=custom.get("name", provider_id),
            base_url=base_url,
            api_key=api_key,
            model=model,
            extra_headers=custom.get("headers", {}) or {},
            timeout=timeout,
            reasoning_effort=effort,
        )
    raise ProviderError(f"unknown provider: {provider_id}")


def _has_openrouter_key(auth) -> bool:
    """True if the user has an OpenRouter API key (env or auth.json)."""
    import os

    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    return auth is not None and bool(auth.get("openrouter"))


def model_effort_levels(model_id: str, provider_id: str = "opencode") -> list[str]:
    """Reasoning-effort levels a model accepts, from the live catalog.

    Reads the model's ``reasoning_options`` (``{type: effort, values: [...]}``)
    so present AND future models work with zero code changes: a new model
    advertising ``[low, high, max]`` is honored the day it lands. Models with
    no effort option (fixed thinkers like big-pickle/nemotron, toggles,
    budget-tokens) return []. Paid models work identically — options come
    from the same catalog record. Never raises.
    """
    try:
        bare = str(model_id).split("/", 1)[-1]
    except Exception:
        return []
    try:
        catalog = (fetch_catalog().get(provider_id) or {}).get("models", {})
        entry = catalog.get(bare) or catalog.get(str(model_id)) or {}
        for opt in entry.get("reasoning_options") or []:
            if isinstance(opt, dict) and opt.get("type") == "effort":
                vals = [str(v) for v in (opt.get("values") or []) if str(v)]
                seen: list[str] = []
                for v in vals:
                    if v not in seen:
                        seen.append(v)
                return seen
    except Exception:
        pass
    return []


def effort_payload_fragment(model_id: str, effort: str, provider_id: str = "opencode") -> dict[str, Any]:
    """Provider-native request fragment for an effort level.

    Mirrors official opencode's variant mapping (provider/transform.ts):
    OpenAI-family (Responses + chat) takes ``reasoning: {effort}`` /
    ``reasoningEffort``; Anthropic takes adaptive ``thinking`` + effort;
    Google takes ``thinkingLevel``; Zen's gateway translates the OpenAI shape
    to whatever the upstream needs. Unknown models fall back to the OpenAI
    shape (harmless: gateways ignore unknown fields, strict ones get it only
    when the catalog advertised the level). Never raises.
    """
    effort = str(effort or "").strip().lower()
    if not effort:
        return {}
    try:
        bare = str(model_id).split("/", 1)[-1].lower()
    except Exception:
        bare = ""
    if "claude" in bare or provider_id == "anthropic":
        return {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": effort}
    if "gemini" in bare or "gemma" in bare or provider_id == "google":
        return {"thinkingLevel": effort, "includeThoughts": True}
    # Responses API shape (official zen path sends reasoning.effort)
    return {"reasoning": {"effort": effort}, "reasoningEffort": effort}


def _capability_score(model: dict) -> tuple:
    """Power ranking for free Zen models, most capable first.

    Sort key (tuple = lexicographic priority):
      1. alive first (a dead giant helps nobody)
      2. tool_call (agentic work needs tools)
      3. reasoning (coding models reason)
      4. attachment (images/pdf for real tasks)
      5. structured_output (reliable tool args)
      6. modalities breadth (text+image+audio+video > text-only)
      7. context + 4x output (raw window power)
      8. release_date (newer wins ties)
    Higher tuple wins; caller sorts reverse=True. Never raises.
    """
    raw = model.get("_raw") or {}
    limit = raw.get("limit") or {}
    modalities = raw.get("modalities") or {}
    in_mod = modalities.get("input") or []

    def _num(*vals: Any) -> float:
        for v in vals:
            try:
                return float(v or 0)
            except (TypeError, ValueError):
                continue
        return 0.0

    ctx = _num(limit.get("context"), model.get("context"))
    out = _num(limit.get("output"), model.get("output"))
    return (
        bool(model.get("alive", True)),
        bool(raw.get("tool_call", True)),
        bool(raw.get("reasoning", False)),
        bool(raw.get("attachment", False)),
        bool(raw.get("structured_output", False)),
        len(in_mod) if isinstance(in_mod, list) else 0,
        ctx + out * 4.0,
        str(raw.get("release_date") or model.get("release_date") or ""),
    )


def live_opencode_lanes(provider: str = "opencode") -> list[dict]:
    """Failover lanes from the LIVE models list, most powerful first.

    Takes every FREE opencodezen model (same source the picker uses:
    catalog -> endpoint -> bundle), asks the DEFAULT catalog record how
    capable it is (tools, reasoning, attachments, modalities, window), and
    sorts most-powerful first. Dead (probe-red) models are dropped; removed
    upstream vanishes; new upstream appears. Your picked model is NOT
    special-cased here — build_rotation keeps it lane 0 and appends these
    behind it. Never raises: falls back to the bundled list on any failure.
    """
    try:
        models = fetch_zen_models()
    except Exception:
        models = []
    live = [
        m for m in (models or [])
        if m.get("free") and m.get("status", "active") == "active"
        and m.get("alive", True)
    ]
    if not live:
        try:
            from .zen import FREE_MODELS

            live = [dict(m, free=True, alive=True) for m in FREE_MODELS]
        except Exception:
            live = []
    live = sorted(live, key=_capability_score, reverse=True)
    seen: set[str] = set()
    lanes: list[dict] = []
    for m in live:
        mid = str(m.get("id") or "")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        lanes.append({"provider": provider, "model": mid})
    return lanes


_warm_once_lock = threading.Lock()
_warm_done: set[str] = set()


def warm_startup(cfg=None, auth=None) -> None:
    """Best-effort background warm: catalog memo, health snapshot kick, lane
    list, and selected-model context — so the first real turn finds hot
    caches. Idempotent per process (keyed by provider/model), daemon thread,
    never raises, never blocks the caller."""
    try:
        key = f"{getattr(cfg, 'provider', '')}\x00{getattr(cfg, 'model', '')}" if cfg is not None else "none"
    except Exception:
        return
    with _warm_once_lock:
        if key in _warm_done:
            return
        _warm_done.add(key)

    def _run() -> None:
        try:
            fetch_catalog()
        except Exception:
            pass
        try:
            if cfg is not None:
                from .zen import FREE_MODELS as _FM
                ids = [str(m.get("id") or "") for m in _FM if m.get("id")]
                if ids:
                    # snapshot served instantly; bg thread probes behind
                    check_zen_model_health(ids)
        except Exception:
            pass
        try:
            if cfg is not None:
                live_opencode_lanes(provider=getattr(cfg, "provider", "opencode") or "opencode")
        except Exception:
            pass
        try:
            if cfg is not None:
                model_context_size(getattr(cfg, "provider", ""), getattr(cfg, "model", ""), auth)
                model_output_limit(getattr(cfg, "provider", ""), getattr(cfg, "model", ""))
        except Exception:
            pass

    try:
        import threading as _th

        _th.Thread(target=_run, name="opencode_py-warm", daemon=True).start()
    except Exception:
        with _warm_once_lock:
            _warm_done.discard(key)


def build_rotation(cfg: Config, auth=None, session_id: str | None = None) -> Rotation:
    """Build a rotation with the picked model as the primary lane.

    - An explicit `rotation` list in config is failover order, not a veto:
      the picked model is always lane 0.
    - Stale lanes (model removed upstream) are dropped, so rotation never
      gets stuck on ghosts.
    - Failover behind the pick comes from the LIVE models list, best-capable
      first — new models appear automatically, removed ones vanish.
    - If the user has an OpenRouter API key, the OpenRouter default free model
      is added as a final failover lane.
    """
    lanes = list(cfg.rotation)

    # The model the user PICKED is always lane 0 — an explicit `rotation` list
    # in config is failover order, not a veto over the picker selection.
    selection = {"provider": cfg.provider, "model": cfg.model}

    def _bare(model_id: str) -> str:
        return str(model_id).split("/", 1)[-1]

    def _same(lane: dict, sel: dict) -> bool:
        return (
            lane.get("provider") == sel["provider"]
            and _bare(lane.get("model", "")) == _bare(sel["model"])
        )

    if lanes and not any(_same(l, selection) for l in lanes):
        lanes.insert(0, dict(selection))

    # Drop lanes whose model no longer exists upstream (catalog source of
    # truth): a removed id would just burn a turn on a guaranteed error.
    # Covers BOTH "opencode" and legacy "zen" provider keys.
    try:
        catalog_ids = {
            mid
            for mid, m in (fetch_catalog().get("opencode") or {}).get("models", {}).items()
            if m.get("status", "active") == "active"
        }
        if catalog_ids:
            lanes = [
                l
                for l in lanes
                if l.get("provider") not in ("opencode", "zen")
                or _bare(l.get("model", "")) in catalog_ids
                or "/" in str(l.get("model", ""))
            ]
    except Exception:
        pass

    if not lanes:
        lanes = [dict(selection)]
    else:
        lanes = [l for l in lanes if l]

    if not any(_same(l, selection) for l in lanes):
        lanes.insert(0, dict(selection))

    if _has_openrouter_key(auth) and cfg.provider != "openrouter" and not cfg.rotation:
        lanes.append({"provider": "openrouter", "model": FREE_DEFAULT_MODELS["openrouter"]})

    if len(lanes) == 1 and cfg.provider in ("opencode", "zen"):
        # Failover from the LIVE list, best-capable first — never the frozen
        # bundle, never stale config ghosts.
        current = cfg.model.split("/", 1)[-1]
        lanes += [
            lane for lane in live_opencode_lanes(provider=cfg.provider)
            if _bare(lane["model"]) != current and not _same(lane, selection)
        ]
    rotation = Rotation(
        lanes,
        lambda pid, m: build_provider(cfg, pid, m, auth, rotation.session_id),
        session_id=session_id,
    )
    return rotation


# -- models.dev catalog (mirrors official opencode's ModelsDev service) ------
# The official client lists models from the models.dev catalog
# (https://models.opencode.ai/api.json), NOT from a provider /models endpoint:
# names, context limits, pricing and add/removes all come from it. It caches
# the raw api.json on disk (atomic write), treats it as fresh for 6 hours,
# and re-fetches every 6h in the background so models added or removed upstream
# appear/disappear automatically. Env overrides match opencode:
#   OPENCODE_MODELS_URL          alternate catalog source
#   OPENCODE_MODELS_PATH         read the catalog from this file instead
#   OPENCODE_DISABLE_MODELS_FETCH=1  never fetch; use cache/bundled only

_MODELS_SOURCE = "https://models.opencode.ai"
_MODELS_FALLBACK_SOURCE = "https://models.dev/api.json"
_CATALOG_FRESH_SECONDS = 6 * 60 * 60
_CATALOG_REFRESH_SECONDS = 6 * 60 * 60

_catalog_refresher_lock = threading.Lock()
_catalog_refresher_started = False
# Wake event for the refresher: fetch_catalog() sets it instead of fetching
# synchronously, so a stale catalog never blocks the calling thread (which
# used to be the TUI's UI thread at end of turn — a hard screen freeze).
_catalog_kick = threading.Event()


def _catalog_cache_file():
    from ..globals import Path as GPath

    source = os.environ.get("OPENCODE_MODELS_URL")
    if source and source != _MODELS_SOURCE:
        import hashlib

        return GPath.cache / f"models-catalog-{hashlib.sha1(source.encode()).hexdigest()[:8]}.json"
    override = os.environ.get("OPENCODE_MODELS_PATH")
    if override:
        from pathlib import Path as _P

        return _P(override)
    return GPath.catalog_file()


def _catalog_fresh(path) -> bool:
    import time

    try:
        return time.time() - path.stat().st_mtime < _CATALOG_FRESH_SECONDS
    except OSError:
        return False


def _fetch_catalog_text() -> str | None:
    urls = []
    override = os.environ.get("OPENCODE_MODELS_URL")
    if override:
        urls.append(override.rstrip("/") + "/api.json")
    else:
        urls.extend([f"{_MODELS_SOURCE}/api.json", _MODELS_FALLBACK_SOURCE])
    for url in urls:
        try:
            import httpx

            try:
                from .zen import _user_agent as _ua
            except Exception:
                _ua = lambda: "opencode/0.1.0"  # noqa: E731
            resp = httpx.get(
                url,
                headers={"User-Agent": _ua()},
                timeout=10,
                follow_redirects=True,
            )
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
        except Exception:
            continue
    return None


def _write_catalog_atomic(path, text: str) -> None:
    """tmp-file + rename, like the official client's fetchAndWrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(text)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _start_catalog_refresher() -> None:
    """6-hour background refresh (models rarely appear/disappear; health
    probes on a 30min TTL catch dead lanes between catalog refreshes)."""
    global _catalog_refresher_started

    if os.environ.get("OPENCODE_DISABLE_MODELS_FETCH"):
        return
    with _catalog_refresher_lock:
        if _catalog_refresher_started:
            return
        _catalog_refresher_started = True

    def _loop() -> None:
        while True:
            _catalog_kick.clear()
            if not _catalog_readonly():
                try:
                    path = _catalog_cache_file()
                    text = _fetch_catalog_text()
                    if text is not None and text != _read_catalog_text(path):
                        _write_catalog_atomic(path, text)
                except Exception:
                    pass
            _catalog_kick.wait(_CATALOG_REFRESH_SECONDS)

    threading.Thread(target=_loop, name="opencode_py-models-refresh", daemon=True).start()


def _read_catalog_text(path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _catalog_readonly() -> bool:
    """True when the catalog source is a user-pinned local file
    (OPENCODE_MODELS_PATH): serve it read-only — the background
    refresher used to OVERWRITE the pinned file with network data."""
    return bool(os.environ.get("OPENCODE_MODELS_PATH"))


_catalog_memo: dict = {"mtime": None, "size": None, "data": None}
_catalog_memo_lock = threading.Lock()


def fetch_catalog() -> dict:
    """The models.dev provider catalog {provider_id: {..., models: {...}}}.

    Cache-first with a 6-hour freshness TTL (stale data is still served when
    the network is down — listing never blocks or fails); a daemon thread
    re-fetches every 6h so live changes propagate while the app runs.
    The 4.6MB file is parsed ONCE per mtime (in-process memo) — without this
    every build_rotation + zen-list + context lookup re-parsed it (~0.4s each).
    """
    path = _catalog_cache_file()

    if not os.environ.get("OPENCODE_DISABLE_MODELS_FETCH") and not _catalog_readonly():
        # NEVER top up synchronously: a stale cache used to trigger blocking
        # httpx calls right here, freezing whatever thread called this — the
        # TUI's UI thread at turn end (rebuild_rotation) being the worst case
        # (full connect timeout x2 URLs). Serve stale immediately and kick the
        # background refresher instead; fresh data lands seconds later with
        # nobody blocking.
        if path.exists() and not _catalog_fresh(path):
            _catalog_kick.set()
        _start_catalog_refresher()

    try:
        st = path.stat()
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None:
        with _catalog_memo_lock:
            if _catalog_memo["mtime"] == key[0] and _catalog_memo["size"] == key[1] \
                    and isinstance(_catalog_memo["data"], dict):
                return _catalog_memo["data"]
    text = _read_catalog_text(path)
    if text:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                if key is not None:
                    with _catalog_memo_lock:
                        _catalog_memo.update(mtime=key[0], size=key[1], data=data)
                return data
        except json.JSONDecodeError:
            pass
    return {}


def refresh_catalog_sync() -> bool:
    """Synchronously pull the models.dev catalog and cache it (official parity).

    Like the official client's fetchAndWrite: new models appear, removed ones
    vanish. Blocking by design — call only from a worker thread (the model
    picker's fetch worker), never the UI/engine thread. Respects
    OPENCODE_DISABLE_MODELS_FETCH / OPENCODE_MODELS_PATH. Returns True when
    the cache was updated.
    """
    if os.environ.get("OPENCODE_DISABLE_MODELS_FETCH") or _catalog_readonly():
        return False
    try:
        path = _catalog_cache_file()
        text = _fetch_catalog_text()
        if text is not None and text != _read_catalog_text(path):
            try:
                data = json.loads(text)
                if not isinstance(data, dict):
                    return False
            except json.JSONDecodeError:
                return False
            _write_catalog_atomic(path, text)
            return True
    except Exception:
        pass
    return False


# -- zen model health (so /models only lists models that actually answer) ----
# The catalog lists every free-tier model, but several need a Zen account key
# or are currently down upstream — picking them just yields 400/401/503.
# A tiny "are you alive" probe per model (1 token, parallel) filters the list;
# results are cached for HEALTH_TTL_SECONDS so refreshes stay cheap.

HEALTH_TTL_SECONDS = 30 * 60


def _probe_headers(session: str | None = None) -> dict:
    import os
    from uuid import uuid4

    # Per-model session: Zen pins a session id to ONE upstream lane, so all
    # probes sharing "health" would hammer the same lane and misreport every
    # other model. A stable per-model id keeps probes independent.
    sid = session or "health"
    try:
        from .zen import _user_agent as _ua
    except Exception:
        _ua = lambda: "opencode/0.1.0"  # noqa: E731
    headers = {
        "User-Agent": _ua(),
        "x-opencode-client": "cli",
        "x-opencode-project": "opencode_py",
        # Fresh request id per probe (official parity); session stays stable
        # per model so probes do not collide on one lane.
        "x-opencode-request": uuid4().hex,
    }
    # Authenticate exactly like ZenProvider does for keyless free lanes:
    # without this the gateway could 401/403 the anonymous ping and mark a
    # model dead that answers fine in real chat. Env key first, then the
    # stored auth.json key (contributor-free models need a real key — probing
    # anonymous "public" alone false-negatives them).
    key = os.environ.get("OPENCODE_API_KEY") or _stored_opencode_key()
    headers["Authorization"] = f"Bearer {key or 'public'}"
    headers["x-opencode-session"] = sid
    return headers


def _stored_opencode_key() -> str | None:
    """Best-effort read of the stored opencode key (auth.json), if any."""
    try:
        from ..auth import Auth
        from ..globals import Path as GPath

        return Auth(auth_file=GPath.auth_file()).get("opencode")
    except Exception:
        return None


def pin_zen_model_alive(model_id: str) -> None:
    """Pin a model as working because it actually responded in real chat.

    Probes can false-negative (anonymous key, dead probe lane, transient 500)
    for models that answer fine with the real session/key — so a lane that
    produces real output is pinned here and stays visible in /models forever
    (until removed from the catalog). Never raises.
    """
    if not model_id:
        return
    try:
        import time
        from pathlib import Path

        from ..globals import Path as GPath

        cache_path = Path(GPath.cache) / "model-health.json"
        data = _load_health_cache(cache_path)
        pinned: dict = dict(data.get("pinned") or {})
        health: dict = dict(data.get("health") or {})
        bare = str(model_id).split("/", 1)[-1]
        pinned[bare] = time.time()
        health[bare] = True
        fails = data.get("fails") or {}
        if isinstance(fails, dict):
            fails.pop(bare, None)
            data["fails"] = fails
        data["pinned"] = pinned
        data["health"] = health
        data["ts"] = data.get("ts") or time.time()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _probe_zen_model(model_id: str) -> bool:
    """True when the gateway actually serves this model right now.

    Probes the model's preferred transport first (Responses by default, like
    the official client — some models only answer there, others only on Chat
    Completions), falling back to the other API before calling it dead. 200 =
    alive; 429 = alive but throttled (don't hide a working model during a
    burst); anything else (401 no-key / 400 / 503 dead upstream) = not usable
    on either transport. Transport errors propagate so the health layer can
    fail open on a total outage.
    """
    from .responses import (
        TransportIncompatible,
        get_preferred_endpoint,
        probe_responses,
        set_preferred_endpoint,
    )

    pref = get_preferred_endpoint(model_id)
    order = [pref, "chat" if pref == "responses" else "responses"]
    seen: set[str] = set()
    for ep in order:
        if ep in seen:
            continue
        seen.add(ep)
        try:
            if ep == "responses":
                ok = probe_responses(model_id, _probe_headers(session=f"health-{model_id}"))
            else:
                ok = _probe_chat_model(model_id)
        except TransportIncompatible:
            continue
        if ok:
            set_preferred_endpoint(model_id, ep)
            return True
    return False


def _probe_chat_model(model_id: str) -> bool:
    """Legacy Chat Completions probe (one leg of _probe_zen_model)."""
    try:
        import httpx

        resp = httpx.post(
            f"{ZEN_CHAT_URL}/chat/completions",
            headers=_probe_headers(session=f"health-{model_id}"),
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=12,
        )
        return resp.status_code in (200, 429)
    except Exception:
        raise


def _load_health_cache(cache_path) -> dict:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


_health_lock = threading.Lock()
_health_bg_keys: set[str] = set()


def _health_cached_verdicts(model_ids: list[str], ttl_seconds: int) -> tuple[dict[str, bool], bool]:
    """Cached alive/dead per model + whether the window is fresh.

    Zero network: unknown models read alive (fail open — visible until a
    probe says otherwise). Never raises."""
    import time
    from pathlib import Path

    from ..globals import Path as GPath

    try:
        cache_path = Path(GPath.cache) / "model-health.json"
    except Exception:
        return {m: True for m in model_ids}, True
    data = _load_health_cache(cache_path)
    try:
        fresh = time.time() - data.get("ts", 0) < ttl_seconds
    except Exception:
        fresh = False
    cached: dict = data.get("health", {}) or {}
    try:
        pinned: dict = data.get("pinned") or {}
    except Exception:
        pinned = {}
    results: dict[str, bool] = {}
    for mid in model_ids:
        if mid in cached:
            results[mid] = bool(cached.get(mid))
        elif isinstance(pinned, dict) and pinned.get(mid):
            results[mid] = True
        else:
            results[mid] = True
    return results, fresh


def _health_probe_store(model_ids: list[str]) -> dict[str, bool]:
    """One blocking probe pass + pin/strike bookkeeping + cache write.

    This is the OLD check_zen_model_health body verbatim (freshness gate
    removed — the background thread always probes, the caller never waits).
    Runs on a daemon thread; callers never call this directly."""
    import time
    from pathlib import Path

    from ..globals import Path as GPath

    cache_path = Path(GPath.cache) / "model-health.json"
    data = _load_health_cache(cache_path)
    pinned: dict[str, float] = {
        k: v for k, v in (data.get("pinned") or {}).items() if isinstance(v, (int, float))
    }
    fails: dict[str, int] = {
        k: int(v) for k, v in (data.get("fails") or {}).items() if isinstance(v, int) and v > 0
    }
    cached: dict = data.get("health", {}) or {}

    known = set(model_ids)
    for mid in [m for m in pinned if m not in known]:
        del pinned[mid]  # the provider removed it from the catalog — unpin
    for mid in [m for m in fails if m not in known]:
        del fails[mid]

    results: dict[str, bool] = {}
    to_probe: list[str] = []
    for mid in model_ids:
        # bg pass: probe everything not known-good (pins re-verified too —
        # a pin only survives actual answers)
        if pinned.get(mid) and mid in cached and bool(cached.get(mid)):
            results[mid] = True
            to_probe.append(mid)
        elif mid not in cached:
            to_probe.append(mid)  # unknown: probe now
        else:
            results[mid] = bool(cached.get(mid))
            to_probe.append(mid)  # stale-safe: re-verify behind

    probed: dict[str, bool] = {}
    if to_probe:
        try:
            from concurrent.futures import ThreadPoolExecutor

            workers = max(1, min(4, (len(to_probe) + 1) // 2))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_probe_zen_model, mid): mid for mid in to_probe}
                for fut, mid in futures.items():
                    try:
                        probed[mid] = bool(fut.result())
                    except Exception:
                        # transport error (DNS/TLS/connection) — not a verdict
                        # on the model; the fail-open below keeps it visible
                        probed[mid] = False
        except Exception:
            probed = {mid: True for mid in to_probe}

        # fail-open: nothing proven alive (network down, gateway unreachable)
        # -> don't trust the failures, keep everything visible WITHOUT counting
        # strikes or pinning (an outage must neither hide nor pin anything).
        if to_probe and not any(probed.values()):
            for mid in to_probe:
                results[mid] = True
        else:
            for mid, ok in probed.items():
                if ok:
                    results[mid] = True
                    fails.pop(mid, None)
                    if mid not in pinned:
                        pinned[mid] = time.time()  # responded -> pinned
                else:
                    strikes = fails.get(mid, 0) + 1
                    if strikes >= 2 or mid not in pinned:
                        # dead: never-pinned models hide at once; pinned ones get
                        # one tolerated blip, then unpin on the second strike.
                        results[mid] = False
                        fails.pop(mid, None)
                        pinned.pop(mid, None)
                    else:
                        fails[mid] = strikes
                        results[mid] = True  # first strike: keep showing

    merged = dict(results)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"ts": time.time(), "health": merged, "pinned": pinned, "fails": fails}),
            encoding="utf-8",
        )
    except OSError:
        pass
    return merged


def _health_kick_bg(model_ids: list[str]) -> None:
    """Single-flight background probe: many callers, one worker per model set."""
    try:
        key = ",".join(sorted(set(model_ids)))
    except Exception:
        return
    if not key:
        return
    with _health_lock:
        if key in _health_bg_keys:
            return
        _health_bg_keys.add(key)

    def _run() -> None:
        try:
            _health_probe_store(model_ids)
        except Exception:
            pass
        finally:
            with _health_lock:
                _health_bg_keys.discard(key)

    try:
        import threading as _th

        _th.Thread(target=_run, name="opencode_py-health", daemon=True).start()
    except Exception:
        with _health_lock:
            _health_bg_keys.discard(key)


def _health_drain(timeout: float = 10.0) -> bool:
    """Wait until all background health probes finish (tests only)."""
    import time as _t
    end = _t.monotonic() + max(0.1, timeout)
    while _t.monotonic() < end:
        with _health_lock:
            if not _health_bg_keys:
                return True
        _t.sleep(0.02)
    with _health_lock:
        return not _health_bg_keys


def check_zen_model_health(model_ids: list[str], ttl_seconds: int = HEALTH_TTL_SECONDS) -> dict[str, bool]:
    """{model_id: alive} — serves the cache instantly, never blocks on network.

    Stale windows are served (listing never blocks or fails) while a
    single-flight background thread probes + stores fresh verdicts for the
    next call. Timing (TTL) is unchanged — only the waiting is gone.
    Fail-open preserved: unknown models read alive.
    """
    results, fresh = _health_cached_verdicts(model_ids, ttl_seconds)
    if not fresh:
        _health_kick_bg(model_ids)
    return results


ZEN_CHAT_URL = "https://opencode.ai/zen/v1"


def _zen_models_from_catalog(catalog: dict) -> list[dict]:
    """Flatten the catalog's `opencode` provider entry into model dicts.

    Keeps the full capability record in ``_raw`` so power ranking can see
    tools/reasoning/attachments/modalities (not just ctx/out).
    """
    models = (catalog.get("opencode") or {}).get("models") or {}
    out: list[dict] = []
    for mid, m in models.items():
        limit = m.get("limit") or {}
        cost = m.get("cost") or {}
        free = cost.get("input", -1) == 0 and cost.get("output", -1) == 0
        out.append(
            {
                "id": mid,
                "name": m.get("name") or mid,
                "context": int(limit.get("context") or 0),
                "output": int(limit.get("output") or 0),
                "free": bool(free),
                "status": m.get("status", "active"),
                "release_date": m.get("release_date") or "",
                "_raw": {
                    "limit": dict(limit),
                    "modalities": dict(m.get("modalities") or {}),
                    "tool_call": m.get("tool_call"),
                    "reasoning": m.get("reasoning"),
                    "attachment": m.get("attachment"),
                    "structured_output": m.get("structured_output"),
                    "release_date": m.get("release_date") or "",
                },
            }
        )
    return out


def fetch_zen_models(cache_file=None, ttl_hours: int = 1) -> list[dict]:
    """OpenCode Zen model list, exactly like the official client.

    Primary source: the models.dev catalog (names like "Big Pickle", real
    context limits, $0 pricing → FREE badge). Every active catalog entry is
    listed — none skipped. Models added or removed upstream show up/vanish on
    their own via the cached catalog + sync refresh on picker open + 6-hour
    background refresh. Within the list, free models that actually respond
    (probe/pin health) sort first. Fallbacks: Zen's own /models endpoint,
    then the bundled free-model list.
    """
    models = _zen_models_from_catalog(fetch_catalog())
    if models:
        # Official dialog-model.tsx: filter deprecated, disable (hide) -nano.
        # Everything else active is listed — even free models whose probe is
        # currently red (they sort below the responding ones, never hidden).
        models = [m for m in models if m.get("status", "active") != "deprecated"]
        models = [m for m in models if "-nano" not in m.get("id", "")]
        # Liveness only orders the list (responding free first); paid ones
        # just need a key, so they aren't probed.
        alive: set[str] = set()
        free_ids = [m["id"] for m in models if m["free"]]
        if free_ids:
            try:
                health = check_zen_model_health(free_ids)
                alive = {mid for mid, ok in health.items() if ok}
            except Exception:
                alive = set()
        for m in models:
            m["_alive"] = (not m["free"]) or (m["id"] in alive) or (not alive and not free_ids)
            # _raw already carries the full catalog capability record from
            # _zen_models_from_catalog — leave it intact for power ranking.
        return _normalize_models(models)

    models = _fetch_zen_endpoint_models(cache_file, ttl_hours)
    if models:
        return models

    from .zen import FREE_MODELS

    return list(FREE_MODELS)


def _fetch_zen_endpoint_models(cache_file=None, ttl_hours: int = 1) -> list[dict]:
    """Fallback: https://opencode.ai/zen/v1/models with a cached fallback."""
    import time
    from pathlib import Path

    from ..globals import Path as GPath

    cache_file = cache_file or GPath.models_file()
    cache_path = Path(cache_file)
    models: list[dict] = []

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            age = time.time() - data.get("ts", 0)
            models = data.get("models", [])
            if age < ttl_hours * 3600 and models:
                return _normalize_models(models)
        except (OSError, json.JSONDecodeError):
            pass

    try:
        import httpx

        from .zen import ZEN_BASE_URL

        resp = httpx.get(f"{ZEN_BASE_URL}/models", timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", []) if isinstance(data, dict) else data
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"ts": time.time(), "models": models}), encoding="utf-8")
            return _normalize_models(models)
    except Exception:
        pass

    return []


def fetch_openrouter_models() -> list[dict]:
    """Live-fetch OpenRouter's public model list filtered to free models.

    Returns [{id, name, context, free, provider='openrouter'}] sorted by id.
    The chosen default free model is always included even if the live list is
    unavailable or it happens to be missing.
    """
    import time

    out: list[dict] = []
    default = FREE_DEFAULT_MODELS["openrouter"]
    seen: set[str] = set()
    live_ok = False

    try:
        import httpx

        resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            live_ok = True
            data = resp.json()
            raw = data.get("data", []) if isinstance(data, dict) else data
            for m in raw:
                mid = m.get("id", "")
                if not mid.endswith(":free"):
                    continue
                seen.add(mid)
                out.append(
                    {
                        "id": mid,
                        "name": m.get("name") or mid,
                        "context": (m.get("context_length") or 0) // 1000,
                        "free": True,
                        "provider": "openrouter",
                    }
                )
    except Exception:
        pass

    # Only show the bundled default when the live list actually responded —
    # otherwise we'd display a model that didn't respond (hide dead lanes).
    if live_ok and default not in seen:
        out.append(
            {
                "id": default,
                "name": default,
                "context": (1000 if "550b" in default else 0),
                "free": True,
                "provider": "openrouter",
            }
        )

    return sorted(out, key=lambda d: d["id"])


def fetch_live_models(
    provider_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
    api_kind: str = "openai",
) -> list[dict]:
    """Live-fetch a provider's `GET /models` list.

    Handles OpenAI-compatible endpoints (Bearer auth) and Anthropic
    (x-api-key + anthropic-version). Returns
    ``[{id, name, context, free, provider}]`` (``free=False``) or ``[]`` when
    there is no key, the endpoint fails, or nothing usable comes back. The
    free-focused fetchers for opencode/openrouter are handled separately.
    """
    if not api_key:
        return []
    base_url = (
        base_url
        or FREE_PROVIDERS.get(provider_id, {}).get("base_url")
        or PAID_PROVIDERS.get(provider_id, {}).get("base_url")
    )
    if not base_url:
        return []
    headers = (
        {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        if api_kind == "anthropic"
        else {"Authorization": f"Bearer {api_key}"}
    )
    try:
        import httpx

        resp = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=8, follow_redirects=True)
        if resp.status_code != 200:
            return []
        data = resp.json()
        raw = data.get("data", []) if isinstance(data, dict) else (data or [])
        out: list[dict] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or ""
            if not mid:
                continue
            if provider_id == "openai" and not (
                mid.startswith("gpt-") or mid.startswith("o") or mid.startswith("chatgpt-")
            ):
                continue
            if provider_id == "anthropic" and not mid.startswith("claude-"):
                continue
            if provider_id == "deepseek" and mid not in ("deepseek-chat", "deepseek-reasoner"):
                continue
            ctx = m.get("context_length") or m.get("context") or 0
            try:
                ctx = int(ctx)
            except (TypeError, ValueError):
                ctx = 0
            out.append(
                {
                    "id": mid,
                    "name": m.get("display_name") or m.get("name") or mid,
                    "context": ctx,
                    "free": False,
                    "provider": provider_id,
                }
            )
        return out
    except Exception:
        return []


_context_cache: dict[tuple[str, str], int] = {}

# Per-provider model list (id + real context), fetched once so the context-window
# lookup hits the network at most once per provider per process.
_model_list_cache: dict[str, list[dict]] = {}
_model_list_cache_lock = threading.Lock()


def _ctx_disk_path():
    try:
        from ..globals import Path as GPath
        return Path(GPath.cache) / "model-context.json"
    except Exception:
        return None


def _ctx_disk_load() -> None:
    """Load persisted context windows (cold starts skip network). Never raises."""
    try:
        fp = _ctx_disk_path()
        if fp is None or not fp.is_file():
            return
        import json as _j
        import time as _t
        data = _j.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        if _t.time() - float(data.get("ts", 0)) > 7 * 24 * 3600:
            return  # a week old: re-verify from network
        for k, v in (data.get("sizes") or {}).items():
            try:
                prov, _, mid = str(k).partition("\x00")
                if prov and mid and int(v) > 0:
                    _context_cache[(prov, mid)] = int(v)
            except (ValueError, TypeError):
                continue
    except Exception:
        pass


def _ctx_disk_save() -> None:
    """Persist known context windows (best-effort, atomic). Never raises."""
    try:
        fp = _ctx_disk_path()
        if fp is None or not _context_cache:
            return
        import json as _j
        import time as _t
        sizes = {f"{p}\x00{m}": s for (p, m), s in _context_cache.items() if s}
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_name(f"{fp.name}.{os.getpid()}.tmp")
        tmp.write_text(_j.dumps({"ts": _t.time(), "sizes": sizes}), encoding="utf-8")
        os.replace(tmp, fp)
    except Exception:
        pass


_ctx_disk_load()

# Known context windows for the default free-provider models (failover lanes).
# Used when a provider's live /models list isn't available.
KNOWN_CONTEXT: dict[str, int] = {
    "opencode": 131072,
    "groq": 131072,
    "cerebras": 131072,
    "google": 1048576,
    "openrouter": 1000000,
    "nvidia": 1000000,
    "mistral": 256000,
    "github": 128000,
    "sambanova": 131072,
    "togetherai": 131072,
    "anthropic": 200000,
    "openai": 128000,
    "deepseek": 128000,
    "xai": 256000,
    "deepinfra": 128000,
    "ollama": 131072,
}

# Exact context windows for specific paid models (documented by the vendor).
# `OPencode /models` and the paid `/models` endpoints rarely report a context
# window, so these prevent the TUI from showing a 0% or a per-provider guess.
# Explicitly documented sizes only — no invented numbers.
_KNOWN_CONTEXTS_BY_MODEL: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "o3": 200000,
    "claude-3-7-sonnet-20250219": 200000,
    "claude-3-7-sonnet-latest": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-sonnet-4-20250514": 200000,
    "deepseek-chat": 65536,
    "deepseek-reasoner": 65536,
    "deepseek-v3": 65536,
    "deepseek-v3.2": 65536,
    "deepseek-coder": 65536,
    "grok-2": 131072,
    "grok-2-1212": 131072,
    "grok-beta": 131072,
    "grok-3": 256000,
    "grok-3-mini": 256000,
    "nemotron-3-ultra-550b-a55b:free": 1000000,
}


def _known_context(provider_id: str, model_id: str) -> int:
    """Context size from the bundled Zen free-model list (opencode only)."""
    if provider_id == "opencode":
        from .zen import FREE_MODELS

        for m in FREE_MODELS:
            if m["id"] == _bare_model_id(provider_id, model_id):
                return int(m.get("context") or 0)
    # A hardcoded per-provider guess is never the *selected model's* real
    # window, so it must NOT short-circuit the live provider lookup below.
    return 0


def _provider_model_list(provider_id: str, auth=None) -> list[dict]:
    """Real model list (id + context) for a provider, fetched once and cached.

    Returns the provider's live `/models` data so the status-bar percentage is
    the REAL context window of the selected model (mirrors opencode, which
    derives the percentage from `model.limit.context`). opencode/openrouter get
    their dedicated fetchers; other providers (openai, anthropic, the free
    BYOK providers, custom ones) use `fetch_live_models` with the configured
    API key. Falls back to the bundled free-model list when no key exists.
    """
    cached = _model_list_cache.get(provider_id)
    if cached is not None:
        return cached
    models: list[dict] = []
    if provider_id == "opencode":
        # Context-size lookups only need ids/windows, NOT liveness — skip
        # fetch_zen_models()'s health probes here (12s timeout per model,
        # previously run on the ENGINE thread at every usage event, stalling
        # the moment the stream finished). Raw catalog + bundled fallback.
        models = _normalize_models(_zen_models_from_catalog(fetch_catalog()))
        if not models:
            from .zen import FREE_MODELS

            models = list(FREE_MODELS)
    elif provider_id == "openrouter":
        # fetch_openrouter_models reports context in thousands; normalize to tokens
        models = [
            {**m, "context": int(m.get("context") or 0) * 1000}
            for m in fetch_openrouter_models()
        ]
    else:
        meta = FREE_PROVIDERS.get(provider_id) or PAID_PROVIDERS.get(provider_id) or {}
        try:
            key = auth.get(provider_id, {}) if auth else None
        except TypeError:
            key = auth.get(provider_id) if auth else None
        if meta:
            models = fetch_live_models(
                provider_id,
                key,
                meta.get("base_url"),
                meta.get("api_kind", "openai"),
            )
    _model_list_cache[provider_id] = models or []
    return models or []


def _bare_model_id(provider_id: str, model_id: str) -> str:
    """Strip a folded `provider/` prefix so lookup matches the bare model id."""
    if model_id.startswith(provider_id + "/"):
        return model_id.split("/", 1)[1]
    return model_id


def model_context_size(provider_id: str, model_id: str, auth=None) -> int:
    """Real context-window size (in tokens) for the selected provider/model.

    Resolution order: bundled known sizes (no network, opencode free models)
    -> the provider's live model list (uses the configured API key, caches the
    list once) -> 0 when genuinely unknown (the UI then omits the percentage).
    Returns the selected model's actual context, not a hardcoded per-provider
    guess, so the TUI's `12,345 (6%)` is the truth for openai/openrouter/etc.
    """
    key = (provider_id, model_id)
    if key in _context_cache:
        return _context_cache[key]
    if not _context_cache:
        _ctx_disk_load()
        if key in _context_cache:
            return _context_cache[key]
    bare = _bare_model_id(provider_id, model_id)
    # opencode bundled free models are exact and need no network
    size = _known_context(provider_id, model_id)
    if not size:
        # real per-model window from the provider's model list
        for m in _provider_model_list(provider_id, auth):
            if m.get("id") == bare and m.get("context"):
                size = int(m["context"])
                break
    if not size:
        # documented per-model windows (vendor-published, not guesses) for
        # providers whose /models endpoint doesn't report a context window
        size = _KNOWN_CONTEXTS_BY_MODEL.get(bare, 0)
    if not size:
        # last resort: a provider-wide default (only for the bundled free
        # providers whose whole catalogue shares one documented window)
        size = KNOWN_CONTEXT.get(provider_id, 0)
    _context_cache[key] = size
    if size:
        try:
            _ctx_disk_save()
        except Exception:
            pass
    return size


def _known_output(provider_id: str, model_id: str) -> int:
    """Output-token limit from the bundled Zen free-model list."""
    if provider_id == "opencode":
        from .zen import FREE_MODELS

        for m in FREE_MODELS:
            if m["id"] == _bare_model_id(provider_id, model_id):
                return int(m.get("output") or 0)
    return 0


def model_output_limit(provider_id: str, model_id: str) -> int:
    """Best-effort max output tokens for a provider/model lane (0 when unknown)."""
    if not model_id:
        return 0
    return _known_output(provider_id, model_id)


def _normalize_models(raw: list[dict]) -> list[dict]:
    from .zen import FREE_MODELS

    fallback = {f["id"]: f for f in FREE_MODELS}
    out = []
    for m in raw:
        cost = m.get("cost") or {}
        limit = m.get("limit") or {}
        fb = fallback.get(m.get("id"), {})
        is_free = (
            (isinstance(cost.get("input"), (int, float)) and cost["input"] == 0)
            or m.get("id") in fallback
            or m.get("free") is True
            or str(m.get("id", "")).endswith("-free")
        )
        out.append(
            {
                "id": m.get("id", ""),
                "name": m.get("name") or fb.get("name") or m.get("id", ""),
                "context": limit.get("context", 0) or m.get("context", 0) or fb.get("context", 0),
                "output": limit.get("output", 0) or m.get("output", 0) or fb.get("output", 0),
                "free": bool(is_free),
                "status": m.get("status", "active"),
                "release_date": m.get("release_date") or fb.get("release_date") or "",
                "alive": bool(m.get("_alive", True)),
                "provider": "opencode",
                "_raw": dict(m.get("_raw") or {}),
            }
        )
    # Official sortModelOptions: Free first, then release_date desc, then title.
    # Responding free models (_alive) sort above the rest of the free tier so
    # the working ones are on top but none are ever hidden.
    # Stable multi-pass sort (last key first) to mix asc/desc correctly.
    out.sort(key=lambda x: str(x.get("name") or x["id"]).lower())
    out.sort(key=lambda x: str(x.get("release_date") or ""), reverse=True)
    out.sort(key=lambda x: (not x.get("alive", True)))
    out.sort(key=lambda x: (not x["free"]))
    return out


def sort_model_options(options: list[dict], newest_first: bool = False) -> list[dict]:
    """Python port of official sortModelOptions (dialog-model.tsx).

    Default: Free first, then release_date desc, then title.
    newest_first (single-provider view): release_date desc, then title.
    Options are dicts with optional free/release_date/name/id keys.
    """
    opts = list(options)
    opts.sort(key=lambda o: str(o.get("name") or o.get("title") or o.get("id") or "").lower())
    opts.sort(key=lambda o: str(o.get("release_date") or ""), reverse=True)
    if not newest_first:
        opts.sort(key=lambda o: (not bool(o.get("free") or o.get("footer") == "Free")))
    return opts


def check_provider(cfg: Config, auth) -> dict[str, Any]:
    """Ping a provider; used by --check. Returns status dict."""
    result: dict[str, Any] = {}
    lanes = cfg.rotation or [{"provider": cfg.provider, "model": cfg.model}]
    for lane in lanes:
        pid = lane.get("provider", "zen")
        model = lane.get("model", FREE_DEFAULT_MODELS.get(pid, FREE_DEFAULT_MODELS["zen"]))
        try:
            provider = build_provider(cfg, pid, model, auth)
            if not provider.api_key and pid != "opencode" and pid != "ollama":
                result[pid] = {"ok": False, "model": model, "error": "no API key (see /connect or env)"}
                continue
            # lightweight models list ping (GET /models) for openai-compat
            try:
                import httpx

                meta = FREE_PROVIDERS.get(pid) or PAID_PROVIDERS.get(pid) or {}
                api_kind = meta.get("api_kind", "openai")
                if api_kind == "anthropic":
                    # Anthropic's /models endpoint needs x-api-key +
                    # anthropic-version, not a Bearer token (Bearer => 401).
                    headers = {
                        "x-api-key": provider.api_key or "",
                        "anthropic-version": "2023-06-01",
                    }
                else:
                    headers = {"Authorization": f"Bearer {provider.api_key}"}
                resp = httpx.get(f"{provider.base_url}/models", headers=headers, timeout=10, follow_redirects=True)
                ok = resp.status_code == 200
                result[pid] = {"ok": ok, "model": model, "status": resp.status_code}
            except Exception as e:
                result[pid] = {"ok": False, "model": model, "error": str(e)}
        except Exception as e:
            result[pid] = {"ok": False, "model": model, "error": str(e)}
    return result
