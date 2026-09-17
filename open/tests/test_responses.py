"""Tests for the Responses-API transport + adaptive endpoint selection.

Mirrors the official client (POST /responses, `input` items); covers history
mapping, tool schema conversion, SSE event parsing, per-model endpoint
preference, and silent fallback between transports.
"""

from opencode_py.providers.base import ProviderEvent, Usage
from opencode_py.providers.responses import (
    _FunctionCallSlots,
    build_responses_input,
    build_responses_payload,
    build_responses_tools,
    get_preferred_endpoint,
    handle_responses_event,
    set_preferred_endpoint,
)


def _state():
    return {"had_output": False, "had_text": False, "had_error": False, "error_message": ""}


# -- history mapping --------------------------------------------------------

def test_system_becomes_developer_and_internals_dropped():
    out = build_responses_input([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi", "_session": "x", "compaction": True, "id": "1"},
        {"role": "assistant", "content": "yo", "reasoning_content": "think", "id": "2"},
    ])
    assert out[0] == {"role": "developer", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2] == {"role": "assistant", "content": "yo"}


def test_tool_calls_replay_as_function_calls_and_outputs():
    out = build_responses_input([
        {
            "role": "assistant",
            "content": "running",
            "reasoning_content": "",
            "tool_calls": [
                {"id": "call_0", "type": "function",
                 "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "name": "bash", "content": "a\nb"},
    ])
    assert {"role": "assistant", "content": "running"} in out
    assert {"type": "function_call", "call_id": "call_0", "name": "bash",
            "arguments": '{"command": "ls"}'} in out
    assert {"type": "function_call_output", "call_id": "call_0", "output": "a\nb"} in out


def test_empty_assistant_text_with_calls_omits_bare_message():
    out = build_responses_input([
        {"role": "assistant", "content": "", "reasoning_content": "",
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "glob", "arguments": "{}"}}]},
    ])
    assert {"role": "assistant", "content": ""} not in out
    assert any(i.get("type") == "function_call" for i in out)


# -- tool schemas -----------------------------------------------------------

def test_openai_schemas_convert_to_responses_tools():
    tools = [
        {"type": "function", "function": {"name": "bash", "description": "run",
                                          "parameters": {"type": "object", "properties": {}}}},
    ]
    out = build_responses_tools(tools)
    assert out == [{"type": "function", "name": "bash", "description": "run",
                    "parameters": {"type": "object", "properties": {}}}]
    assert build_responses_tools(None) is None
    assert build_responses_tools([]) is None


def test_payload_matches_official_shape():
    p = build_responses_payload("m", [{"role": "user", "content": "hi"}], None, session_key="ses_1")
    assert p["model"] == "m"
    assert p["store"] is False
    assert p["stream"] is True
    assert p["include"] == ["reasoning.encrypted_content"]
    assert p["prompt_cache_key"] == "ses_1"
    assert "tools" not in p
    p2 = build_responses_payload("m", [], None)
    assert "prompt_cache_key" not in p2


# -- event parsing ----------------------------------------------------------

def _feed(raw_events):
    got = []
    slots = _FunctionCallSlots()
    usage = Usage()
    done = [False]
    state = _state()
    for name, data in raw_events:
        import json

        handle_responses_event({"event": name, "data": json.dumps(data)}, got.append,
                               slots, usage, done, state)
    for call in slots.values():
        got.append(ProviderEvent(kind="tool_call", tool_calls=[call]))
    return got, usage, done, state


def test_text_delta_and_done():
    got, usage, done, state = _feed([
        ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "PONG"}),
        ("response.completed", {"type": "response.completed",
                                "response": {"status": "completed",
                                             "usage": {"input_tokens": 10, "output_tokens": 2,
                                                       "total_tokens": 12}}}),
    ])
    assert [e.text for e in got if e.kind == "text_delta"] == ["PONG"]
    assert state["had_output"] and state["had_text"]
    assert usage.total_tokens == 12
    assert any(e.kind == "done" for e in got)


def test_reasoning_delta():
    got, _, _, state = _feed([
        ("response.reasoning_summary_text.delta",
         {"type": "response.reasoning_summary_text.delta", "delta": "hmm"}),
    ])
    assert [e.text for e in got if e.kind == "reasoning_delta"] == ["hmm"]
    assert state["had_output"] and not state["had_text"]


def test_function_call_assembled_from_delta_and_done():
    got, _, _, state = _feed([
        ("response.output_item.added",
         {"type": "response.output_item.added",
          "item": {"type": "function_call", "id": "fc_1", "call_id": "call_0", "name": "bash"}}),
        ("response.function_call_arguments.delta",
         {"type": "response.function_call_arguments.delta", "item_id": "call_0",
          "delta": '{"command": "ls"}'}),
        ("response.output_item.done",
         {"type": "response.output_item.done",
          "item": {"type": "function_call", "id": "fc_1", "call_id": "call_0",
                   "name": "bash", "arguments": '{"command": "ls"}'}}),
    ])
    calls = [e.tool_calls[0] for e in got if e.kind == "tool_call"]
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].id == "call_0"
    assert "ls" in calls[0].arguments
    assert state["had_output"]


def test_failed_response_sinks_error():
    got, _, _, state = _feed([
        ("response.failed", {"type": "response.failed",
                             "response": {"status": "failed",
                                          "error": {"message": "upstream blew up"}}}),
    ])
    assert state["had_error"]
    assert "upstream blew up" in state["error_message"]
    assert any(e.kind == "error" for e in got)


def test_incompatible_model_raises_before_output():
    from opencode_py.providers.responses import TransportIncompatible

    got = []
    try:
        handle_responses_event(
            {"event": "message",
             "data": '{"type": "error", "error": {"message": "model does not support responses"}}'},
            got.append, _FunctionCallSlots(), Usage(), [False], _state())
        raise AssertionError("expected TransportIncompatible")
    except TransportIncompatible:
        pass


# -- endpoint preference ----------------------------------------------------

def test_endpoint_preference_defaults_responses_and_roundtrips(tmp_path, monkeypatch):
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "cache", tmp_path)
    assert get_preferred_endpoint("some-future-model") == "responses"
    set_preferred_endpoint("some-future-model", "chat")
    assert get_preferred_endpoint("some-future-model") == "chat"
    set_preferred_endpoint("some-future-model", "bogus")  # ignored
    assert get_preferred_endpoint("some-future-model") == "chat"


def test_zen_falls_back_to_chat_when_responses_refuses(tmp_path, monkeypatch):
    """A model that 404s on /responses must silently stream via chat."""
    from opencode_py.globals import Path as GPath
    from opencode_py.providers import responses as R
    from opencode_py.providers.zen import ZenProvider

    monkeypatch.setattr(GPath, "cache", tmp_path)
    monkeypatch.setattr(R, "stream_responses", lambda **kw: (_ for _ in ()).throw(
        R.TransportIncompatible("nope")))
    seen = {}

    def fake_chat(self, messages, tools, sink, **kw):
        seen["chat"] = True
        sink(ProviderEvent(kind="text_delta", text="via chat"))

    monkeypatch.setattr("opencode_py.providers.openai_compat.OpenAICompatProvider._stream",
                        fake_chat)
    got = []
    p = ZenProvider(api_key=None, model="chat-only-model", session_id="s")
    p.stream_chat([{"role": "user", "content": "hi"}], None, got.append)
    assert seen.get("chat") is True
    assert [e.text for e in got if e.kind == "text_delta"] == ["via chat"]
    assert R.get_preferred_endpoint("chat-only-model") == "chat"


def test_zen_uses_cached_chat_without_touching_responses(tmp_path, monkeypatch):
    from opencode_py.globals import Path as GPath
    from opencode_py.providers import responses as R
    from opencode_py.providers.zen import ZenProvider

    monkeypatch.setattr(GPath, "cache", tmp_path)
    R.set_preferred_endpoint("chat-fan", "chat")

    def boom(**kw):
        raise AssertionError("responses must not be tried")

    monkeypatch.setattr(R, "stream_responses", boom)
    monkeypatch.setattr("opencode_py.providers.openai_compat.OpenAICompatProvider._stream",
                        lambda self, m, t, sink, **kw: sink(ProviderEvent(kind="text_delta", text="ok")))
    got = []
    ZenProvider(api_key=None, model="chat-fan", session_id="s").stream_chat(
        [{"role": "user", "content": "hi"}], None, got.append)
    assert [e.text for e in got if e.kind == "text_delta"] == ["ok"]
