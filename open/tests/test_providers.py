"""Tests for the OpenAI-compatible provider: content-part deltas, the
stream_options flag, and flushing the SSE tail on stream end."""

import json
from unittest import mock

import pytest

from opencode_py.providers.base import ContextOverflowError, ProviderEvent, Usage
from opencode_py.providers.openai_compat import (
    OpenAICompatProvider,
    _content_to_text,
    _is_context_overflow_message,
)


class FakeResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self):
        yield from self._chunks


class FakeClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, *a, **k):
        return FakeResponse(self._chunks)


def provider(**kwargs):
    return OpenAICompatProvider(base_url="https://example.com", api_key="k", model="m", **kwargs)


def test_abort_stream_wakes_blocked_read_as_interrupt():
    """abort_stream() must unblock a read that's stalled on an idle model gap
    (no chunks arriving), and the interrupt flag must turn the wake-up
    exception into StreamInterrupted — never a network error."""
    import threading

    from opencode_py.providers.base import StreamInterrupted

    p = provider()

    class BlockedResponse:
        status_code = 200

        def __init__(self):
            self.started = threading.Event()
            self.closed = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_bytes(self):
            # simulate protracting: the model "thinks" silently, the read blocks
            # forever on the socket until abort_stream() forces close()
            self.started.set()
            if not self.closed.wait(timeout=10):
                raise RuntimeError("abort did not wake the read")
            raise OSError("connection closed by abort")

        def close(self):
            self.closed.set()

    class BlockingClient:
        def __init__(self):
            self.resp = BlockedResponse()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, *a, **k):
            return self.resp

    bc = BlockingClient()
    flag = {"value": False}
    raised = []

    def run():
        try:
            p.stream_chat([], [], lambda e: None, is_interrupted=lambda: flag["value"])
        except BaseException as e:  # noqa: BLE001 - captured to assert below
            raised.append(e)

    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=bc):
        t = threading.Thread(target=run)
        t.daemon = True
        t.start()
        # wait until the provider is genuinely blocked reading the stream
        assert bc.resp.started.wait(timeout=5), "reader never entered iter_bytes"
        # interrupt lands while the model is silent: no chunk will arrive, so
        # only an explicit abort of the stream can wake the read up
        flag["value"] = True
        p.abort_stream()
        t.join(timeout=5)
        assert not t.is_alive(), "abort did not wake the blocked stream read"
    assert len(raised) == 1
    assert isinstance(raised[0], StreamInterrupted)


def test_abort_stream_noop_when_idle():
    """abort_stream() with nothing in flight must be a harmless no-op."""
    p = provider()
    p.abort_stream()


def test_rotation_abort_forwards_to_active_provider():
    from opencode_py.providers.rotation import Rotation

    calls = []

    class P:
        def abort_stream(self):
            calls.append("aborted")

    rot = Rotation(lanes=[{"provider": "a", "model": "m"}], make_provider=lambda pid, m: P())
    rot._active_provider = P()
    rot.abort()
    assert calls == ["aborted"]

    # no active provider -> no-op, no error
    rot2 = Rotation(lanes=[], make_provider=lambda pid, m: P())
    rot2.abort()


def test_handle_content_string():
    p = provider()
    events = []
    evt = {"data": json.dumps({"choices": [{"delta": {"content": "hello"}}]})}
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "text_delta"] == ["hello"]


def test_handle_content_list_parts():
    p = provider()
    events = []
    evt = {
        "data": json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "text", "text": "hello "},
                                {"type": "text", "text": "world"},
                            ]
                        }
                    }
                ]
            }
        )
    }
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "text_delta"] == ["hello world"]


def test_content_to_text_variants():
    assert _content_to_text("plain") == "plain"
    assert _content_to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _content_to_text([{"text": "x"}]) == "x"
    assert _content_to_text(["raw"]) == "raw"
    assert _content_to_text({"unexpected": True}) == "{'unexpected': True}"


def test_handle_reasoning_content():
    p = provider()
    events = []
    evt = {"data": json.dumps({"choices": [{"delta": {"reasoning_content": "think..."}}]})}
    p._handle_event(evt, events.append, {}, None)
    assert [e.text for e in events if e.kind == "reasoning_delta"] == ["think..."]


def test_usage_in_every_chunk_keeps_content():
    # Some gateways (e.g. the Zen router) attach a usage object to every SSE
    # chunk alongside the content delta; the content must not be dropped.
    p = provider()
    events = []
    usage = Usage()
    for data in [
        {"choices": [{"delta": {"reasoning_content": "think"}}], "usage": {"total_tokens": 1}},
        {"choices": [{"delta": {"content": "hi"}}], "usage": {"total_tokens": 2}},
    ]:
        p._handle_event({"data": json.dumps(data)}, events.append, {}, usage)
    assert [e.text for e in events if e.kind == "reasoning_delta"] == ["think"]
    assert [e.text for e in events if e.kind == "text_delta"] == ["hi"]
    assert usage.total_tokens == 2


def test_usage_only_ping_no_content():
    p = provider()
    events = []
    usage = mock.MagicMock()
    usage.input_tokens = usage.output_tokens = usage.total_tokens = 0
    evt = {"data": json.dumps({"choices": [], "cost": "0", "usage": {"total_tokens": 5}})}
    p._handle_event(evt, events.append, {}, usage)
    assert events == []
    assert usage.total_tokens == 5


def test_build_payload_stream_options_on_by_default():
    p = provider(include_usage=True)
    payload = p.build_payload([{"role": "user", "content": "hi"}])
    assert payload["stream_options"] == {"include_usage": True}


def test_build_payload_stream_options_disableable():
    p = provider()
    payload = p.build_payload([{"role": "user", "content": "hi"}])
    assert "stream_options" not in payload


def test_build_payload_reasoning_content_opt_in():
    """Thinking-mode gateways (Zen Console) require `reasoning_content` on every
    reassembled assistant message and 400 with 'reasoning_content must be passed
    back' when it's missing — those lanes set `reasoning_passthrough=True`.
    Strict endpoints (OpenAI official) reject unknown message fields, so by
    default the key is only sent when the model actually produced reasoning."""
    p = provider(reasoning_passthrough=True)
    payload = p.build_payload(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "plain reply"},
            {"role": "assistant", "content": "", "reasoning_content": "thoughts", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "grep", "content": "out"},
        ]
    )
    asst = [m for m in payload["messages"] if m["role"] == "assistant"]
    # passthrough lane: always present, defaulted to ""
    assert asst[0].get("reasoning_content") == ""
    assert asst[1].get("reasoning_content") == "thoughts"
    # internal-only keys never leak to the wire
    assert all("_pending" not in m for m in payload["messages"])

    strict = provider()
    payload2 = strict.build_payload(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "plain reply"},
            {"role": "assistant", "content": "", "reasoning_content": "thoughts"},
        ]
    )
    asst2 = [m for m in payload2["messages"] if m["role"] == "assistant"]
    # strict lane: no invented key; real reasoning still passes back untouched
    assert "reasoning_content" not in asst2[0]
    assert asst2[1].get("reasoning_content") == "thoughts"

def test_zen_provider_opts_into_reasoning_passthrough():
    from opencode_py.providers.zen import ZenProvider

    zen = ZenProvider(model="x-preview-f-free")
    assert zen.reasoning_passthrough is True


def test_handle_tool_call_name_not_doubled_on_repeated_chunks():
    """Some gateways repeat the function name on later chunks; the accumulated
    name must not become 'grepgrep' (which would call a bogus tool)."""
    import json as _json

    p = provider()
    tool_calls: dict = {}
    sink = []
    for i, delta in enumerate(
        [
            {"index": 0, "id": "c1", "function": {"name": "grep", "arguments": '{"pat'}},
            {"index": 0, "function": {"name": "grep", "arguments": 'tern": "x"}'}},
        ]
    ):
        p._handle_event(
            {"data": _json.dumps({"choices": [{"delta": {"tool_calls": [delta]}}]})},
            sink.append,
            tool_calls,
            Usage(),
            done_emitted=[False],
        )
    calls = list(tool_calls.values())
    assert len(calls) == 1 and calls[0].name == "grep"
    assert calls[0].arguments == '{"pattern": "x"}'


def test_stream_flushes_tail_without_newline():
    p = provider()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',  # no trailing newline
    ]
    events = []
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=FakeClient(chunks)):
        p.stream_chat([], [], events.append)
    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text == "hello"


def test_stream_tail_done_sentinel_without_newline():
    p = provider()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]",  # no trailing newline
    ]
    events = []
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=FakeClient(chunks)):
        p.stream_chat([], [], events.append)
    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text == "hi"


def test_context_overflow_marker_detection():
    assert _is_context_overflow_message("context_length_exceeded: too long")
    assert _is_context_overflow_message("this model's maximum context length is 8192 tokens")
    assert _is_context_overflow_message("reduce_other_history to keep the conversation under the token limit")
    assert not _is_context_overflow_message("rate limit reached")
    assert not _is_context_overflow_message("")


def test_context_overflow_broad_real_world_messages():
    from opencode_py.providers.classify import is_context_overflow

    assert is_context_overflow(
        "prompt was truncated because it exceeded the max context length of 128000 tokens"
    )
    assert is_context_overflow(
        "This model's maximum context length is 8192 tokens. However, your messages resulted in about 9000 tokens."
    )
    assert is_context_overflow("400 Bad Request: request_too_large")
    assert is_context_overflow("exceeds the context window")
    assert is_context_overflow("exceeds context window 200k")
    assert is_context_overflow("input token count exceeds max")
    assert is_context_overflow("input token count exceeds the maximum")
    assert is_context_overflow("exceeds the maximum allowed input length of 131072 tokens")
    assert is_context_overflow("error: {'code': 'model_context_window_exceeded'}")
    # rate-limit wording must NEVER be classified as overflow
    assert not is_context_overflow("throttling error: rate limit reached")
    assert not is_context_overflow("rate_limit_exceeded: too many requests this minute")
    assert not is_context_overflow("quota exceeded")
    assert not is_context_overflow("insufficient_quota")
    assert not is_context_overflow("")


def test_stream_error_event_context_overflow_raises():
    p = provider()
    events = []
    # context_length_exceeded must raise ContextOverflowError (not sink as error)
    with pytest.raises(ContextOverflowError):
        p._handle_event(
            {"data": json.dumps({"error": {"message": "budget too long", "code": "context_length_exceeded"}})},
            events.append,
            {},
            None,
        )


def test_context_overflow_by_message_text_raises():
    p = provider()
    events = []
    # some gateways only report the overflow in the message, without a code
    with pytest.raises(ContextOverflowError):
        p._handle_event(
            {"data": json.dumps({"error": {"message": "this model's maximum context length is 4000 tokens"}})},
            events.append,
            {},
            None,
        )


def test_non_context_error_still_sinks_error_event():
    p = provider()
    events = []
    p._handle_event(
        {"data": json.dumps({"error": {"message": "server hiccup", "code": "server_error"}})},
        events.append,
        {},
        None,
    )
    assert any(e.kind == "error" for e in events)


class RecordingClient:
    """Fake httpx.Client that captures the request JSON (like a real one,
    where json.dumps would reject non-serializable payload values)."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.request_payload = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, *a, json=None, **k):
        self.request_payload = json
        return FakeResponse(self._chunks)


def test_is_interrupted_not_leaked_into_request_payload():
    # Regression: the engine passes `is_interrupted` (a callback) down to the
    # provider. It must be popped before build_payload, or it lands in the JSON
    # body and real httpx fails with `TypeError: function is not JSON
    # serializable`, killing the stream.
    p = provider()
    client = RecordingClient([b"data: [DONE]\n\n"])
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=client):
        p.stream_chat(
            [{"role": "user", "content": "hi"}],
            [],
            events := [].append,
            is_interrupted=lambda: False,
        )
    assert "is_interrupted" not in client.request_payload
    json.dumps(client.request_payload)  # must be serializable


def test_is_interrupted_mid_stream_raises_stream_interrupted():
    from opencode_py.providers.base import StreamInterrupted

    p = provider()
    flag = {"interrupted": False}

    def on_event(evt):
        if evt.kind == "text_delta":
            flag["interrupted"] = True

    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    with mock.patch("opencode_py.providers.openai_compat.httpx.Client", return_value=FakeClient(chunks)):
        with pytest.raises(StreamInterrupted):
            p.stream_chat(
                [{"role": "user", "content": "hi"}],
                [],
                on_event,
                is_interrupted=lambda: flag["interrupted"],
            )


def test_anthropic_is_interrupted_not_leaked_into_request_payload():
    from opencode_py.providers.anthropic import AnthropicProvider

    p = AnthropicProvider(base_url="https://api.anthropic.com/v1", api_key="k", model="claude")
    client = RecordingClient([b"data: [DONE]\n\n"])
    with mock.patch("opencode_py.providers.anthropic.httpx.Client", return_value=client):
        p.stream_chat(
            [{"role": "user", "content": "hi"}],
            [],
            [].append,
            is_interrupted=lambda: False,
        )
    assert "is_interrupted" not in client.request_payload
    json.dumps(client.request_payload)  # must be serializable


def test_anthropic_is_interrupted_mid_stream_raises_stream_interrupted():
    from opencode_py.providers.anthropic import AnthropicProvider
    from opencode_py.providers.base import StreamInterrupted

    p = AnthropicProvider(base_url="https://api.anthropic.com/v1", api_key="k", model="claude")
    flag = {"interrupted": False}

    def on_event(evt):
        if evt.kind == "text_delta":
            flag["interrupted"] = True

    chunks = [
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}\n\n',
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}\n\n',
        b"data: [DONE]\n\n",
    ]
    with mock.patch("opencode_py.providers.anthropic.httpx.Client", return_value=FakeClient(chunks)):
        with pytest.raises(StreamInterrupted):
            p.stream_chat(
                [{"role": "user", "content": "hi"}],
                [],
                on_event,
                is_interrupted=lambda: flag["interrupted"],
            )


def test_anthropic_payload_converts_tool_conversation():
    """OpenAI-style history with tool calls must map to Anthropic's schema:
    system -> `system` param, assistant tool_calls -> tool_use content blocks,
    and consecutive tool results merged into one user message with tool_result
    blocks that keep their tool_use_id."""
    from opencode_py.providers.anthropic import AnthropicProvider

    p = AnthropicProvider(base_url="https://api.anthropic.com/v1", api_key="k", model="claude")
    client = RecordingClient([b"data: [DONE]\n\n"])
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "let me check",
            "reasoning_content": "thinking...",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "read", "arguments": "{\"filePath\": \"a.py\"}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "out1"},
        {"role": "tool", "tool_call_id": "c2", "name": "read", "content": "out2"},
    ]
    with mock.patch("opencode_py.providers.anthropic.httpx.Client", return_value=client):
        p.stream_chat(history, [], [].append)
    payload = client.request_payload
    assert payload["system"] == "sys"
    assert payload["messages"][0] == {"role": "user", "content": "hi"}
    asst = payload["messages"][1]
    assert asst["role"] == "assistant"
    types = [b.get("type") for b in asst["content"]]
    assert types == ["text", "tool_use", "tool_use"]
    ids = [b.get("id") for b in asst["content"] if b.get("type") == "tool_use"]
    assert ids == ["c1", "c2"]
    # both tool results merged into ONE user message with matching ids
    assert payload["messages"][2]["role"] == "user"
    blocks = payload["messages"][2]["content"]
    assert all(b["type"] == "tool_result" for b in blocks)
    assert [b["tool_use_id"] for b in blocks] == ["c1", "c2"]
    assert [b["content"] for b in blocks] == ["out1", "out2"]
    json.dumps(payload)  # must be serializable


def test_indexless_parallel_tool_calls_split_by_id():
    """Indexless parallel tool-call streams must stay separate calls."""
    import json as _json

    from opencode_py.providers.base import Usage
    from opencode_py.providers.openai_compat import OpenAICompatProvider, SSEDecoder

    tc_a1 = {"id": "call_a", "function": {"name": "read", "arguments": "{" + '"file'}}
    tc_a2 = {"function": {"arguments": _JSON_A1}}
    tc_b = {"id": "call_b", "function": {"name": "write", "arguments": "{}"}}
    payloads = [
        {"choices": [{"delta": {"tool_calls": [tc_a1]}}]},
        {"choices": [{"delta": {"tool_calls": [tc_a2]}}]},
        {"choices": [{"delta": {"tool_calls": [tc_b]}}]},
    ]
    raw = "".join("data: " + _json.dumps(p) + "\n\n" for p in payloads)
    raw += "data: [DONE]\n\n"
    prov = OpenAICompatProvider(base_url="https://x", model="m")
    sink, calls, usage, done = [], {}, Usage(), [False]
    for evt in SSEDecoder().feed(raw.encode()) + SSEDecoder().close():
        prov._handle_event(evt, sink.append, calls, usage, done)
    assert len(calls) == 2, calls
    by_id = {c.id: c for c in calls.values()}
    assert by_id["call_a"].name == "read"
    assert by_id["call_a"].arguments == _JSON_A_FULL
    assert by_id["call_b"].name == "write"


_JSON_A1 = 'Path": "a.py"}'
_JSON_A_FULL = '{"filePath": "a.py"}'
