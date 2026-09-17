"""Tests for the AgentLoop: reasoning-only turns, empty-reply history, and the
degenerate tool-call guard."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

from opencode_py.agent.loop import AgentLoop
from opencode_py.config import Config
from opencode_py.providers.base import ProviderEvent, ToolCall
from opencode_py.tools import build_registry


class FakeRotation:
    """Replays a script of steps; each step is an event-callable or an exception."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        step = self.script[idx]
        if isinstance(step, BaseException):
            raise step
        step(on_event)
        return "opencode", "deepseek-v4-flash-free"


def make_loop(rotation):
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    return AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=rotation,
        agent="build",
    )


def test_normal_text_answer():
    rot = FakeRotation([lambda on: on(ProviderEvent(kind="text_delta", text="hi there"))])
    loop = make_loop(rot)
    result = loop.run_turn("hello")
    assert result.text == "hi there"
    assert not result.error


def test_queued_prompt_folds_into_same_turn():
    """Prompts queued while a turn runs join the SAME drain (opencode's Session
    Drain): after the first text answer, the engine folds the queued prompt in
    at the next provider-turn boundary instead of ending the turn."""
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="text_delta", text="first answer")),
        lambda on: on(ProviderEvent(kind="text_delta", text="second answer")),
    ])
    loop = make_loop(rot)
    assert loop.queue_prompt("second") == 1
    result = loop.run_turn("first")
    # both provider turns ran under the one run_turn call
    assert rot.calls == 2
    assert result.text == "second answer"
    assert not result.error
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert users == ["first", "second"]
    assert loop.prompt_pending() == 0


def test_queued_prompts_drain_in_order_in_one_turn():
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="text_delta", text="a")),
        lambda on: on(ProviderEvent(kind="text_delta", text="b")),
        lambda on: on(ProviderEvent(kind="text_delta", text="c")),
    ])
    loop = make_loop(rot)
    assert loop.queue_prompt("second") == 1
    assert loop.queue_prompt("third") == 2
    result = loop.run_turn("first")
    assert rot.calls == 3
    assert result.text == "c"
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert users == ["first", "second", "third"]
    assert loop.prompt_pending() == 0


def test_queued_prompt_folds_after_tool_round_mid_drain():
    """opencode's chat-mid-working behavior: a prompt typed while the model is
    running tools is folded into the SAME context as soon as the tool results
    settle (the next provider-turn boundary) — the model sees the tool output
    AND the queued chat together and reasons about it as a continuation (it can
    keep working or stop, as asked). No fresh turn, no idle gap."""
    tc = ToolCall(id="c1", name="glob", arguments=json.dumps({"pattern": "*.py"}))
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=[tc])),
        lambda on: on(ProviderEvent(kind="text_delta", text="stopping now")),
    ])
    loop = make_loop(rot)
    # queue arrives while the model is mid tool-round; run_turn is already busy
    assert loop.queue_prompt("immediately stop") == 1
    result = loop.run_turn("find py files")
    # tools settle and the queued prompt join the same provider request
    assert rot.calls == 2
    assert result.text == "stopping now"
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert users == ["find py files", "immediately stop"]
    roles = [m.get("role") for m in loop.get_history()]
    assert "tool" in roles
    assert loop.prompt_pending() == 0


def test_queue_prompts_cleared_on_reset():
    loop = make_loop(FakeRotation([]))
    loop.queue_prompt("old")
    loop.clear_prompts()
    assert loop.prompt_pending() == 0


def test_reasoning_only_turn_keeps_nonempty_history():
    rot = FakeRotation([lambda on: on(ProviderEvent(kind="reasoning_delta", text="thinking..."))])
    loop = make_loop(rot)
    result = loop.run_turn("hello")
    assert result.text == ""
    assert result.reasoning == "thinking..."
    assistant = [m for m in loop.get_history() if m.get("role") == "assistant"]
    assert assistant and assistant[-1].get("reasoning_content") == "thinking..."
    # the reasoning signal lives in `reasoning_content`, not duplicated as
    # `content` (replay must match what the model actually returned)
    assert assistant[-1].get("content") == ""


def test_empty_turn_appends_nonempty_assistant():
    rot = FakeRotation([lambda on: None])
    loop = make_loop(rot)
    result = loop.run_turn("hello")
    assert result.text == ""
    history = loop.get_history()
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] != ""


def test_degenerate_tool_call_ends_turn_with_error_no_spin():
    tc = ToolCall(id="1", name="", arguments="{}")
    rot = FakeRotation([lambda on: on(ProviderEvent(kind="tool_call", tool_calls=[tc]))])
    loop = make_loop(rot)
    result = loop.run_turn("hi")
    assert result.error and "invalid tool call" in result.error
    assert rot.calls == 1


def test_valid_tool_call_runs_and_history_has_tool_role():
    tc = ToolCall(id="c1", name="glob", arguments=json.dumps({"pattern": "*.py"}))
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=[tc])),
        lambda on: on(ProviderEvent(kind="text_delta", text="done")),
    ])
    loop = make_loop(rot)
    result = loop.run_turn("find py files")
    assert result.text == "done"
    roles = [m.get("role") for m in loop.get_history()]
    assert "tool" in roles


def test_tool_call_with_missing_id_gets_fallback():
    tc = ToolCall(id="", name="glob", arguments=json.dumps({"pattern": "*.py"}))
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=[tc])),
        lambda on: on(ProviderEvent(kind="text_delta", text="done")),
    ])
    loop = make_loop(rot)
    result = loop.run_turn("find py files")
    assert result.text == "done"
    assert not result.error


def test_assistant_tool_calls_message_content_not_none():
    from opencode_py.agent.parse import assistant_message_from_calls

    msg = assistant_message_from_calls([{"id": "c1", "name": "glob", "arguments": "{}"}])
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["function"]["name"] == "glob"


def test_inband_rate_limit_failover_emits_rotated_with_reason():
    """End-to-end: a primary lane that returns an in-band rate-limit error must
    fail over to the next lane AND the 'rotated' event must report the reason."""
    from opencode_py.providers.rotation import Rotation

    class FakeProvider:
        def __init__(self, events):
            self.events = events

        def stream_chat(self, messages, tools, on_event, **kwargs):
            for e in self.events:
                on_event(e)

    queue = iter([
        FakeProvider([ProviderEvent(kind="error", error="rate limit: try again later")]),
        FakeProvider([ProviderEvent(kind="text_delta", text="fallback answer")]),
    ])
    rot = Rotation(
        lanes=[{"provider": "opencode", "model": "deepseek-v4-flash-free"}, {"provider": "opencode", "model": "big-pickle"}],
        make_provider=lambda pid, m: next(queue),
    )
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    events = []
    loop = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=rot,
        agent="build",
        on_event=events.append,
    )
    result = loop.run_turn("hello")
    assert result.text == "fallback answer"
    rotated = [e for e in events if e.get("kind") == "rotated"]
    assert rotated
    assert rotated[0]["provider"] == "opencode"
    assert "rate" in rotated[0]["reason"]


def test_primary_overload_does_not_fail_over():
    """A transient overload on the chosen model must NOT rotate to a backup;
    it is retried on the SAME model, and the real cause surfaces only after
    retries are exhausted (never routing the user to a different model)."""
    from opencode_py.providers.rotation import Rotation

    class AlwaysOverloaded:
        def __init__(self):
            self.calls = 0

        def stream_chat(self, messages, tools, on_event, **kwargs):
            self.calls += 1
            on_event(ProviderEvent(kind="error", error="server_is_overloaded: retry"))

    class Backup:
        def stream_chat(self, messages, tools, on_event, **kwargs):
            on_event(ProviderEvent(kind="text_delta", text="fallback answer"))

    primary = AlwaysOverloaded()
    backup_called = []

    def make(pid, m):
        if pid == "opencode-backup":
            backup_called.append(m)
            return Backup()
        return primary

    rot = Rotation(
        lanes=[
            {"provider": "opencode", "model": "deepseek-v4-flash-free"},
            {"provider": "opencode-backup", "model": "big-pickle"},
        ],
        make_provider=make,
    )
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    cfg.auto_retry_count = 2
    events = []
    loop = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=rot,
        agent="build",
        on_event=events.append,
    )
    loop._retry_delay = lambda e, a: 0.0
    result = loop.run_turn("hello")
    # the overload was retried on the same lane (auto_retry loop) instead of
    # being surfaced immediately; the backup lane was NEVER consulted
    assert "server_is_overloaded" in result.error
    assert backup_called == []
    assert not [e for e in events if e.get("kind") == "rotated"]


# --------------------------------------------------------------------------
# Nested sub-agent event routing (A5): a grandchild's events must keep their
# own session_id through the parent's bridge, not be re-tagged with the
# direct child's id.
# --------------------------------------------------------------------------

def test_nested_subagent_events_keep_own_session_id():
    cfg = Config()
    events = []
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        on_event=events.append,
        provider=SimpleNamespace(),
    )
    bridge = parent._subagent_bridge("child1")
    # an event already tagged by a deeper bridge keeps its own session id
    bridge({"kind": "text_delta", "session_id": "grandchild9", "text": "hi"})
    assert events[-1]["session_id"] == "grandchild9"
    # an untagged event from the direct child gets the child's id
    bridge({"kind": "text_delta", "text": "yo"})
    assert events[-1]["session_id"] == "child1"


def test_subagent_start_keeps_own_id_through_parent_bridge():
    cfg = Config()
    events = []
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        on_event=events.append,
        provider=SimpleNamespace(),
    )
    bridge = parent._subagent_bridge("child1")
    bridge({"kind": "subagent_start", "session_id": "grandchild9", "agent": "build", "title": "t"})
    evt = events[-1]
    assert evt["session_id"] == "grandchild9"
    assert evt["title"] == "t"


# --------------------------------------------------------------------------
# A failed sub-agent must still emit `subagent_done` (ok=False) so the TUI
# clears its busy state / running indicator.
# --------------------------------------------------------------------------

def test_subagent_run_failure_emits_subagent_done_ok_false(monkeypatch):
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    events = []
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        on_event=events.append,
        provider=SimpleNamespace(),
    )
    parent.provider_factory = lambda: SimpleNamespace()

    def boom(self, prompt):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(AgentLoop, "run_turn", boom)
    result = parent.spawn_task({"prompt": "do it", "description": "sub", "subagent_type": "build"})
    done = [e for e in events if e.get("kind") == "subagent_done"]
    assert done and done[-1]["ok"] is False
    assert done[-1]["session_id"] != parent.session_id
    assert result["error"]
    assert "kaboom" in result["output"]


# --------------------------------------------------------------------------
# tool_denied must carry the tool input so the TUI can render what was denied
# (even when no tool_call event preceded it).
# --------------------------------------------------------------------------

def test_tool_denied_emits_input_arguments():
    cfg = Config()
    events = []
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        on_event=events.append,
        provider=SimpleNamespace(),
    )
    fake_perm = SimpleNamespace()
    fake_perm.evaluate = lambda *a: "deny"
    parent.permission = fake_perm
    parent.check_permission(
        "write", "{}", "display", call_id="c1", arguments={"filePath": "x.py"}
    )
    evt = events[-1]
    assert evt["kind"] == "tool_denied"
    assert evt["input"] == {"filePath": "x.py"}
    assert evt["call_id"] == "c1"


# --------------------------------------------------------------------------
# Context overflow (bug 5 adjacent): the loop must compact history and retry
# once instead of surfacing a hard error.
# --------------------------------------------------------------------------

def test_context_overflow_retries_trimmed_then_succeeds():
    from opencode_py.providers.base import ContextOverflowError

    calls = []
    captured = []

    class OverflowThenOk:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            calls.append(len(messages))
            captured.append(list(messages))
            # compaction request: summary prompt, no tools -> succeed and pick
            # a body back; main step: overflow on first attempt only.
            if not tools:
                on_event(ProviderEvent(kind="text_delta", text="summarized"))
                return "opencode", "deepseek-v4-flash-free"
            if sum(1 for c in calls if tools) == 1:
                raise ContextOverflowError("context length exceeded")
            on_event(ProviderEvent(kind="text_delta", text="recovered"))
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(OverflowThenOk())
    loop.cfg.compaction_enabled = True
    # seed some history so there is something to compact
    loop._history = [
        {"role": "user", "content": "old q1"}, {"role": "assistant", "content": "old a1"},
        {"role": "user", "content": "old q2"}, {"role": "assistant", "content": "old a2"},
        {"role": "user", "content": "old q3"}, {"role": "assistant", "content": "old a3"},
    ]
    # proactive compaction: with no usage there's no overflow, so it must not fire
    result = loop.run_turn("new question")
    assert result.text == "recovered"
    assert result.error in ("", "context overflow: context length exceeded")
    # one main step overflowed, compaction ran, then the retry succeeded
    assert sum(1 for c in calls if True) >= 3
    # the last (successful) request includes the anchored summary marker
    assert any(m.get("compaction") is True for m in captured[-1])


def test_context_overflow_persistent_reports_error():
    from opencode_py.providers.base import ContextOverflowError

    class AlwaysOverflow:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            raise ContextOverflowError("context length exceeded")

    loop = make_loop(AlwaysOverflow())
    loop.cfg.compaction_enabled = True
    result = loop.run_turn("hello")
    # compaction runs (a nested stream call) but keeps failing → surfaced
    assert result.error
    assert "context" in result.error


def test_proactive_compaction_skips_error_when_history_too_large():
    """When the request about to be sent already fills the usable window, the
    loop must summarize history BEFORE sending, so a provider context-length
    error never surfaces."""
    from opencode_py.providers.base import ContextOverflowError

    calls = []

    class CompactionThenOk:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            # compaction request has no tools and a user summary prompt
            if not tools and any(m.get("role") == "user" for m in messages):
                calls.append("summary")
                on_event(ProviderEvent(kind="text_delta", text="summarized state"))
                return "opencode", "deepseek-v4-flash-free"
            calls.append("main")
            on_event(ProviderEvent(kind="text_delta", text="done"))
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(CompactionThenOk())
    loop.cfg.compaction_enabled = True
    # deepseek-v4-flash-free: 200k context, output reserve capped at 20k →
    # 180k usable; make the history genuinely exceed it (>720k chars ≈ 180k).
    big = "x" * 400_000
    loop._history = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": big},
    ]
    # the TUI shows `Compacting conversation…` on compaction_start, then the
    # ` Session compacted ` divider on compacted — assert the event order
    emitted = []
    loop.on_event = lambda evt: emitted.append(evt.get("kind"))
    result = loop.run_turn("next step")
    assert "summary" in calls
    assert result.text == "done"
    assert not result.error
    # history was compacted: it now starts with the summary marker
    assert loop._history[0].get("compaction") is True
    assert loop._compaction_summary != ""
    assert "compaction_start" in emitted
    assert "compacted" in emitted
    assert emitted.index("compaction_start") < emitted.index("compacted")


def test_proactive_compaction_works_for_any_known_model():
    """The compaction trigger must use the SELECTED model's window, not a
    hardcoded deepseek/200k assumption — e.g. groq llama-3.3-70b-versatile
    (131072 context, unknown output → 111072 usable)."""
    calls = []

    class CompactionThenOk:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            if not tools and any(m.get("role") == "user" for m in messages):
                calls.append("summary")
                on_event(ProviderEvent(kind="text_delta", text="summarized state"))
                return "groq", "llama-3.3-70b-versatile"
            calls.append("main")
            on_event(ProviderEvent(kind="text_delta", text="done"))
            return "groq", "llama-3.3-70b-versatile"

    cfg = Config()
    cfg.provider = "groq"
    cfg.model = "llama-3.3-70b-versatile"
    loop = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=CompactionThenOk(),
        agent="build",
    )
    loop.cfg.compaction_enabled = True
    # groq llama: 131072 - 20000 = 111072 usable → history needs > ~444k chars
    big = "x" * 300_000
    loop._history = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": big},
    ]
    result = loop.run_turn("next step")
    assert "summary" in calls
    assert result.text == "done"
    assert not result.error


def test_unknown_context_model_falls_back_without_crashing():
    """A model with no known window can't be judged proactively — the loop must
    fall back to the configured hard budget and still run the turn normally."""
    calls = []

    class PlainOk:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            calls.append("main")
            on_event(ProviderEvent(kind="text_delta", text="ok"))
            return "custom", "mystery-model"

    cfg = Config()
    cfg.provider = "custom"
    cfg.model = "mystery-model"
    loop = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=PlainOk(),
        agent="build",
    )
    loop.cfg.compaction_enabled = True
    loop._history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
    ]
    result = loop.run_turn("hello")
    assert result.text == "ok"
    assert not result.error
    assert calls == ["main"]


def test_compaction_tail_budget_uses_selected_model_window():
    """The preserved tail must be sized from the active model's context window,
    not a hardcoded 200k assumption, and reserve the lane's output budget."""
    from opencode_py.agent import compaction as compact_mod

    # a big-window model keeps a proportionally larger recent tail
    small_window = compact_mod.preserve_recent_budget(16_000, output_limit=8_000)
    big_window = compact_mod.preserve_recent_budget(200_000, output_limit=128_000)
    assert big_window > small_window
    # the reserve is capped at the buffer: an 8k-output lane reserves 8k, so
    # its usable window is larger than an unknown-output lane (20k reserve)
    assert compact_mod.usable_context(200_000, 8_000) > compact_mod.usable_context(200_000, 0)


def test_select_tail_fallback_keeps_only_last_turn():
    """When no recent turn fits the preserve budget, the tail must collapse to
    the LAST user turn only — the compacted history (and the token counter the
    TUI recomputes from it) stays bounded instead of keeping every oversized
    recent turn."""
    from opencode_py.agent import compaction as compact_mod

    messages = [
        {"role": "user", "content": "q" * 200_000}, {"role": "assistant", "content": "a" * 200_000},
        {"role": "user", "content": "q2" * 200_000}, {"role": "assistant", "content": "a2" * 200_000},
        {"role": "user", "content": "GIANT PASTE " + "z" * 400_000},
    ]
    head, tail = compact_mod.select_tail(messages, tail_turns=2, context=200_000, output_limit=128_000)
    assert len(tail) == 1
    assert tail[0]["role"] == "user"
    assert tail[0] is messages[-1]
    assert head == messages[:-1]


def test_usable_context_caps_output_reserve_at_buffer():
    """The declared output limit is never reserved past the compaction buffer —
    a 128k 'max output' shouldn't starve a 200k conversation (36% usable)."""
    from opencode_py.agent import compaction as compact_mod

    assert compact_mod.usable_context(200_000, 0) == 180_000          # unknown → buffer
    assert compact_mod.usable_context(200_000, 8_000) == 192_000      # small lane → 8k only
    assert compact_mod.usable_context(200_000, 128_000) == 180_000    # huge output → capped at buffer
    # a buffer larger than the whole window swallows it entirely → 0
    assert compact_mod.usable_context(16_000, 0) == 0


def test_is_overflow_compares_request_to_usable_window():
    from opencode_py.agent import compaction as compact_mod

    usable = compact_mod.usable_context(200_000, 0)  # 180k
    assert not compact_mod.is_overflow(200_000, usable - 1, output_limit=0)
    assert compact_mod.is_overflow(200_000, usable, output_limit=0)


def test_is_overflow_skips_when_reserve_swallows_window():
    """usable == 0 means the request can't be judged proactively — skip instead
    of firing compaction every turn (no infinite compact loop). The post-overflow
    recovery path handles the real overflow."""
    from opencode_py.agent import compaction as compact_mod

    assert compact_mod.usable_context(16_000, 0) == 0
    assert not compact_mod.is_overflow(16_000, 10_000_000, output_limit=0)


def test_auto_compacts_between_steps_when_actual_usage_fills_window():
    """Mid-turn auto-compaction (mirrors upstream opencode): after a step whose
    ACTUAL provider-reported usage fills the usable window, the loop must
    compact BEFORE the next step instead of running to 100% and stalling. The
    usable window comes from the selected model (deepseek: 200k, output capped
    at buffer → 180k usable)."""
    from opencode_py.providers.base import Usage

    calls = []

    class ToolLoopThenOverflowingUsage:
        def __init__(self):
            self.step = 0

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            self.step += 1
            # compaction summary request
            if not tools and any(m.get("role") == "user" for m in messages):
                calls.append("summary")
                on_event(ProviderEvent(kind="text_delta", text="anchored summary"))
                return "opencode", "deepseek-v4-flash-free"
            if self.step == 1:
                # first main step: a tool call; usage reports the window is full
                calls.append("main1")
                tc = ToolCall(id="c1", name="glob", arguments=json.dumps({"pattern": "*.py"}))
                on_event(ProviderEvent(kind="tool_call", tool_calls=[tc]))
                on_event(
                    ProviderEvent(
                        kind="usage",
                        usage=Usage(input_tokens=180_000, output_tokens=1_000, total_tokens=181_000),
                    )
                )
                return "opencode", "deepseek-v4-flash-free"
            calls.append("main2")
            on_event(ProviderEvent(kind="text_delta", text="done"))
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(ToolLoopThenOverflowingUsage())
    loop.cfg.compaction_enabled = True
    loop._history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
    ]
    result = loop.run_turn("search")
    assert result.text == "done"
    assert not result.error
    # step 1 ran the tool, then compaction ran BEFORE step 2
    assert "main1" in calls
    assert "summary" in calls
    assert "main2" in calls
    assert calls.index("summary") < calls.index("main2")
    # history was compacted mid-turn and continues with the summary
    assert loop._history[0].get("compaction") is True


def test_auto_compaction_skips_when_usage_below_window():
    """Mid-turn auto-compaction must NOT fire when actual usage is below the
    usable window — a small conversation shouldn't be summarized."""
    from opencode_py.providers.base import Usage

    calls = []

    class ToolLoopSmallUsage:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            calls.append("main")
            tc = ToolCall(id="c1", name="glob", arguments=json.dumps({"pattern": "*.py"}))
            on_event(ProviderEvent(kind="tool_call", tool_calls=[tc]))
            on_event(
                ProviderEvent(kind="usage", usage=Usage(input_tokens=1_000, output_tokens=100, total_tokens=1_100))
            )
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(ToolLoopSmallUsage())
    loop.cfg.compaction_enabled = True
    loop._history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
    ]
    result = loop.run_turn("search")
    assert "main" in calls
    assert "summary" not in calls


def test_turn_start_compacts_on_prior_actual_usage():
    """A turn whose previous completion's ACTUAL usage already filled the usable
    window must compact before sending the next request — the estimate can
    undercount tool-heavy conversations, so the provider-reported usage from the
    last turn (persists across turns on the loop) is the authoritative signal."""
    from opencode_py.providers.base import Usage

    calls = []

    class PriorUsageFull:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            if not tools and any(m.get("role") == "user" for m in messages):
                calls.append("summary")
                on_event(ProviderEvent(kind="text_delta", text="anchored summary"))
                return "opencode", "deepseek-v4-flash-free"
            calls.append("main")
            on_event(ProviderEvent(kind="text_delta", text="done"))
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(PriorUsageFull())
    loop.cfg.compaction_enabled = True
    loop._history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
    ]
    # last turn's real usage already at the window: deepseek 200k, usable 180k
    loop._usage_total = {"input_tokens": 180_000, "output_tokens": 1_000, "total_tokens": 181_000}
    result = loop.run_turn("next")
    assert "summary" in calls
    assert result.text == "done"
    # history was compacted at the start of the turn
    assert loop._history[0].get("compaction") is True


def test_estimate_request_counts_messages_and_tools():
    """Upstream `estimate({system, messages, tools})`: the request payload only."""
    from opencode_py.agent import compaction as compact_mod

    messages = [
        {"role": "system", "content": "s" * 100},
        {"role": "user", "content": "u" * 400},
    ]
    tools = [{"function": {"name": "read", "description": "r" * 200, "parameters": {}}}]
    total = compact_mod.estimate_request(messages, tools)
    # messages alone: (100 + 400) / 4 = 125 tokens
    assert total > 125
    assert total < 125 + 1000  # tools add a bounded few hundred tokens


def test_force_compact_uses_ai_summary_like_official():
    """/compact must summarize with the model (not just drop history)."""
    calls = []

    class SummarizingProvider:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            calls.append({"messages": list(messages), "tools": list(tools)})
            on_event(ProviderEvent(kind="text_delta", text="anchored summary"))
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(SummarizingProvider())
    loop.cfg.compaction_enabled = True
    loop._history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"},
    ]
    summary = loop.force_compact()
    assert summary == "anchored summary"
    assert loop._compaction_summary == "anchored summary"
    # the summary request went out with no tools
    assert calls and calls[0]["tools"] == []
    # history now leads with the anchored summary marker
    assert loop.get_history()[0].get("compaction") is True


def test_compaction_summary_is_user_message_keeping_alternation():
    """The compaction checkpoint must be a USER message (upstream opencode's
    semantics) — never a fabricated assistant message, which strict
    thinking-mode gateways reject because it lacks `reasoning_content`. The
    summary folds into the tail's first (user) turn so `assistant` messages are
    never invented and role alternation survives, so Anthropic-style providers
    don't see a bare `user,user` run."""
    events = []

    class SummarizingProvider:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            on_event(ProviderEvent(kind="text_delta", text="anchored summary"))
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(SummarizingProvider())
    loop.cfg.compaction_enabled = True
    loop._history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"}, {"role": "assistant", "content": "a4"},
    ]
    loop.force_compact()
    hist = loop.get_history()
    # the checkpoint rides on a real user turn, not a synthetic assistant turn
    roles = [m["role"] for m in hist]
    assert roles[0] == "user", f"compaction must not invent an assistant msg: {roles}"
    no_consecutive_same = all(hist[i]["role"] != hist[i + 1]["role"] for i in range(len(hist) - 1))
    assert no_consecutive_same, f"alternation broken: {roles}"
    # marker preserved for the TUI divider + retry replay
    assert any(m.get("compaction") for m in hist)


def test_transient_timeout_retries_once_then_succeeds():
    """A retryable transient failure (streaming read timeout) must retry before
    surfacing — a slow free model should not kill a mid-conversation turn.
    Events from the failed attempt stay buffered (nothing visible), so the
    retry is clean."""
    from opencode_py.providers.base import ProviderError

    class TimeoutThenOk:
        def __init__(self):
            self.calls = 0

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("timeout talking to OpenCode Zen: read timed out", retryable=True)
            on_event(ProviderEvent(kind="text_delta", text="recovered"))
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(TimeoutThenOk())
    result = loop.run_turn("hello")
    assert result.text == "recovered"
    assert not result.error
    assert loop.rotation.calls == 2


def test_transient_timeout_retry_exhausted_surfaces_error():
    """If every retry fails, the transient error surfaces (retryable) instead
    of hanging forever."""
    import opencode_py.agent.loop as loop_mod
    from opencode_py.providers.base import ProviderError

    class AlwaysTimeout:
        def __init__(self):
            self.calls = 0

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            self.calls += 1
            raise ProviderError("timeout talking to OpenCode Zen: read timed out", retryable=True)

    rot = AlwaysTimeout()
    loop = make_loop(rot)
    loop.cfg.auto_retry_count = 3
    original_sleep = time.sleep
    loop_mod.time.sleep = lambda x: None
    try:
        result = loop.run_turn("hello")
    finally:
        loop_mod.time.sleep = original_sleep
    assert result.error
    assert "timeout" in result.error.lower()
    # the primary attempt plus up to auto_retry_count retries
    assert rot.calls == loop.cfg.auto_retry_count + 1


def test_auto_retry_disabled_surfaces_immediately():
    """With auto_retry off, a transient failure surfaces on the first attempt
    instead of waiting out the backoff."""
    from opencode_py.providers.base import ProviderError

    class CountingTimeout:
        def __init__(self):
            self.calls = 0

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            self.calls += 1
            raise ProviderError("timeout talking to OpenCode Zen: read timed out", retryable=True)

    rot = CountingTimeout()
    loop = make_loop(rot)
    loop.cfg.auto_retry = False
    result = loop.run_turn("hello")
    assert result.error
    assert "timeout" in result.error.lower()
    assert rot.calls == 1


def test_retry_honors_retry_after_header():
    """A RateLimitError carrying Retry-After must wait that long (not the
    exponential backoff) before retrying, mirroring upstream opencode."""
    from opencode_py.providers.base import RateLimitError

    class RateLimitedThenOk:
        def __init__(self):
            self.calls = 0
            self.sleeps: list[float] = []

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("rate limited (429)", retry_after=3.0)
            on_event(ProviderEvent(kind="text_delta", text="back"))
            return "opencode", "deepseek-v4-flash-free"

    rot = RateLimitedThenOk()
    loop = make_loop(rot)
    import opencode_py.agent.loop as loop_mod

    def fake_sleep(seconds):
        rot.sleeps.append(seconds)

    loop_mod.time.sleep = fake_sleep
    try:
        result = loop.run_turn("hello")
    finally:
        loop_mod.time.sleep = time.sleep
    assert result.text == "back"
    # interruptible backoff sleeps in 0.1s slices so ESC wakes it: the total
    # must still honor Retry-After exactly (3.0s), not the exponential curve
    assert abs(sum(rot.sleeps) - 3.0) < 1e-9


def test_rate_limit_without_retry_after_retries_with_backoff():
    """A 429 with no Retry-After (Zen free tier never sends one — live-verified)
    is transient burst, not a dead quota: mirror the official client and retry
    the SAME model with exponential backoff instead of dying mid-chat."""
    from opencode_py.providers.base import RateLimitError

    class AlwaysLimited:
        def __init__(self):
            self.calls = 0

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            self.calls += 1
            raise RateLimitError("FreeUsageLimitError: rate limit exceeded")

    rot = AlwaysLimited()
    loop = make_loop(rot)
    loop.cfg.auto_retry_count = 5
    import opencode_py.agent.loop as loop_mod

    import time as _time

    loop_mod.time.sleep = lambda s: None
    try:
        result = loop.run_turn("hello")
    finally:
        loop_mod.time.sleep = _time.sleep
    assert result.error
    assert "rate limit" in result.error.lower()
    assert "FreeUsageLimitError" in result.error
    # full retry budget spent on the same lane (initial + 5 retries)
    assert rot.calls == 6


def test_rate_limit_without_retry_after_recovers_on_retry():
    """One burst 429 then success: the turn must recover with no visible error."""
    from opencode_py.providers.base import RateLimitError

    class LimitedThenOk:
        def __init__(self):
            self.calls = 0

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("rate limit exceeded")
            on_event(ProviderEvent(kind="text_delta", text="back"))
            return "opencode", "muse-spark-1.3-contributor-free"

    rot = LimitedThenOk()
    loop = make_loop(rot)
    import opencode_py.agent.loop as loop_mod

    import time as _time

    loop_mod.time.sleep = lambda s: None
    try:
        result = loop.run_turn("hello")
    finally:
        loop_mod.time.sleep = _time.sleep
    assert result.text == "back"
    assert not result.error
    assert rot.calls == 2


def test_rotation_session_id_follows_engine_session():
    """Rebinding the engine's session id must update the rotation so every
    provider built for the conversation reports the same x-opencode-session."""
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=Path("."), agent="build")
    assert loop.rotation.session_id == loop.session_id
    loop.session_id = "switched-session"
    assert loop.rotation.session_id == "switched-session"


def test_tool_step_preserves_text_and_reasoning_in_history():
    """When a step streams text + reasoning AND makes a tool call, the assistant
    message written to history must carry the text and reasoning alongside the
    tool_calls — one faithful assistant turn, not a split/empty message (which
    makes reasoning models lose thread on the next tool-loop step)."""
    tc = ToolCall(id="c1", name="glob", arguments=json.dumps({"pattern": "*.py"}))
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="reasoning_delta", text="thinking...")) or
        on(ProviderEvent(kind="text_delta", text="Let me look.")) or
        on(ProviderEvent(kind="tool_call", tool_calls=[tc])),
        lambda on: on(ProviderEvent(kind="text_delta", text="done")),
    ])
    loop = make_loop(rot)
    result = loop.run_turn("find py files")
    assert result.text == "done"
    calls_msg = [m for m in loop.get_history() if m.get("tool_calls")]
    assert len(calls_msg) == 1
    assert "Let me look." in calls_msg[0]["content"]
    assert calls_msg[0]["reasoning_content"] == "thinking..."
    # no stray text-only assistant message splitting the tool step
    assistants = [m for m in loop.get_history() if m.get("role") == "assistant"]
    assert all("done" not in m.get("content", "") or not m.get("tool_calls") for m in assistants[:-1])


def test_retry_nudge_falls_back_to_full_context():
    """After every keep-going nudge is used, a retry re-sends the full context
    unchanged instead of piling on more nudge noise (which reasoning models
    read as a mid-thought interrupt)."""
    from opencode_py.agent.loop import TurnResult

    loop = make_loop(object())
    messages = [{"role": "user", "content": "original"}]
    result = TurnResult(tool_calls_made=2)
    nudges = len(loop._RETRY_NUDGES)
    for attempt in range(1, nudges + 1):
        rebuilt = loop._retry_messages(messages, result, attempt)
        assert rebuilt[-1]["role"] == "user"
        assert "keep going" in rebuilt[-1]["content"] or "continue" in rebuilt[-1]["content"]
    # the nudge after the last one must be a plain full-context re-send
    rebuilt = loop._retry_messages(messages, result, nudges + 1)
    assert rebuilt == messages


def test_rotation_lock_flag_forwarded_to_stream():
    """rotation_locked must reach the rotation's stream() call (and default off)."""
    seen = {}

    class CapturingRotation:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            seen["locked"] = kwargs.get("locked", False)
            on_event(ProviderEvent(kind="text_delta", text="ok"))
            return "opencode", "deepseek-v4-flash-free"

    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=Path("."), provider=CapturingRotation(), agent="build")
    assert loop.rotation_locked is False
    result = loop.run_turn("hi")
    assert not result.error
    assert seen["locked"] is False
    # lock it: the next turn pins the selected model (locked=True)
    loop.rotation_locked = True
    loop.run_turn("hi again")
    assert seen["locked"] is True


def test_rotation_lock_reads_config():
    """The engine picks up rotation_lock from the config at construction."""
    cfg = Config()
    cfg.rotation_lock = True
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=Path("."), agent="build")
    assert loop.rotation_locked is True


def test_retry_backoff_grows_with_attempt():
    """Backoff doubles per attempt with additive +25% jitter (2s base, cap 30s)."""
    from opencode_py.providers.base import ProviderError

    error = ProviderError("boom", retryable=True)
    loop = make_loop(object())
    d1 = loop._retry_delay(error, 0)
    d2 = loop._retry_delay(error, 1)
    assert 2.0 <= d1 <= 2.5
    assert 4.0 <= d2 <= 5.0
    assert d2 >= d1


def test_nothing_done_retry_resends_original_prompt():
    """If the model failed with no tools run yet, the retry re-sends the
    original prompt unchanged (the user's 're-run works' case)."""
    from opencode_py.providers.base import ProviderError

    sent = []

    class DropThenOk:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            sent.append(list(messages))
            if len(sent) == 1:
                raise ProviderError("network error talking to OpenCode Zen: Server disconnected", retryable=True)
            on_event(ProviderEvent(kind="text_delta", text="environment report"))
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(DropThenOk())
    result = loop.run_turn("run tools and give me a small report")
    assert result.text == "environment report"
    assert not result.error
    # retry sent the exact same prompt — no nudges appended
    assert [m.get("role") for m in sent[0]] == [m.get("role") for m in sent[1]]
    assert all(m1.get("content") == m2.get("content") for m1, m2 in zip(sent[0], sent[1]))


def test_mid_operation_retry_sends_keep_going_same_model():
    """If the model stops mid-operation (it had already run tools), the retry
    must nudge the SAME model to keep going — never rotate to another lane,
    and never replay the whole tool task from scratch."""
    import opencode_py.agent.loop as loop_mod
    from opencode_py.providers.base import ProviderError

    class ToolThenDropThenResume:
        def __init__(self):
            self.calls = 0
            self.last_messages = None

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # first step: the model performs a tool call
                on_event(
                    ProviderEvent(
                        kind="tool_call",
                        tool_calls=[ToolCall(id="c1", name="glob", arguments=json.dumps({"pattern": "*.py"}))],
                    )
                )
                return "opencode", "deepseek-v4-flash-free"
            if self.calls == 2:
                # second step: the model stops mid-operation with a dropped conn
                raise ProviderError(
                    "network error talking to OpenCode Zen: Server disconnected without sending a response",
                    retryable=True,
                )
            # retry succeeds — the SAME model continues
            self.last_messages = list(messages)
            on_event(ProviderEvent(kind="text_delta", text="environment report finished"))
            return "opencode", "deepseek-v4-flash-free"

    rot = ToolThenDropThenResume()
    loop = make_loop(rot)
    loop_mod.time.sleep = lambda x: None
    try:
        result = loop.run_turn("run tools and give me a small report about the environment")
    finally:
        loop_mod.time.sleep = time.sleep
    assert result.text == "environment report finished"
    assert not result.error
    assert rot.calls == 3
    # the resumed request carried a keep-going nudge (not a full replay)
    assert rot.last_messages is not None
    nudge = rot.last_messages[-1]
    assert nudge.get("role") == "user"
    assert "keep going" in nudge.get("content", "").lower() or "continue" in nudge.get("content", "").lower()


def test_interrupt_forwarded_to_provider_mid_stream():
    """ESC/Ctrl+C must abort an IN-FLIGHT stream, not just between steps.

    The interrupt callable must reach the provider during streaming (so the
    SSE loop raises StreamInterrupted), and that must end the turn as an
    'interrupted' event — never a generic error."""
    from opencode_py.providers.base import StreamInterrupted

    class InterruptableProvider:
        def __init__(self, flag):
            self.flag = flag

        def stream_chat(self, messages, tools, on_event, **kwargs):
            seen = kwargs.get("is_interrupted")
            assert seen is not None, "is_interrupted must be forwarded to the provider"
            # simulate the provider's own interrupt check firing mid-stream
            self.flag["value"] = True
            if seen():
                raise StreamInterrupted()

    class InterruptableRotation:
        def __init__(self, flag):
            self.flag = flag

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            assert kwargs.get("is_interrupted") is not None
            provider = InterruptableProvider(self.flag)
            provider.stream_chat(messages, tools, on_event, **kwargs)

    events = []
    flag = {"value": False}
    loop = make_loop(InterruptableRotation(flag))
    loop.interrupt = lambda: flag["value"]
    loop.on_event = events.append
    result = loop.run_turn("hello")
    assert not result.error, "an interrupt is not an error"
    assert any(e.get("kind") == "interrupted" for e in events)


def test_interrupt_mid_stream_preserves_partial_text():
    """Partial text streamed before the interrupt must survive (upstream
    opencode keeps what was already shown on screen)."""
    from opencode_py.providers.base import StreamInterrupted

    class PartialThenInterrupt:
        def __init__(self):
            self.flag = {"value": False}

        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            on_event(ProviderEvent(kind="text_delta", text="partial answer"))
            # user presses Esc (2nd press) mid-stream — flip the shared flag
            self.flag["value"] = True
            if kwargs.get("is_interrupted") and kwargs["is_interrupted"]():
                raise StreamInterrupted()
            on_event(ProviderEvent(kind="text_delta", text=" rest"))
            return "opencode", "deepseek-v4-flash-free"

    rot = PartialThenInterrupt()
    events = []
    loop = make_loop(rot)
    loop.interrupt = lambda: rot.flag["value"]
    loop.on_event = events.append
    result = loop.run_turn("hello")
    assert result.text == "partial answer"
    assert not result.error
    assert any(e.get("kind") == "interrupted" for e in events)


def _assert_no_orphan_tool_calls(history):
    """Every assistant `tool_calls` message must be followed by a tool message
    for each declared call id (strict backends reject "insufficient tool
    messages following tool_calls")."""
    pending = []
    for m in history:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid in pending:
                pending.remove(cid)
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                pending.append(str(c.get("id")))
        else:
            assert not pending, f"orphan tool_calls {pending} before {m.get('role')}"
    assert not pending, f"orphan tool_calls at end of history: {pending}"


def test_assistant_tool_calls_message_keeps_reasoning_content():
    """assistant_message_from_calls must always carry `reasoning_content`
    ('' when none) so thinking-mode gateways that demand it on every history
    assistant message don't reject the reassembled tool-call declaration."""
    from opencode_py.agent.parse import assistant_message_from_calls

    with_reasoning = assistant_message_from_calls([{"id": "c1", "name": "grep", "arguments": "{}"}], reasoning="think", content="")
    assert with_reasoning["reasoning_content"] == "think"

    without_reasoning = assistant_message_from_calls([{"id": "c2", "name": "read", "arguments": "{}"}], reasoning="", content="")
    assert without_reasoning.get("reasoning_content") == ""
    assert without_reasoning["content"] == ""


def test_interrupt_during_tool_run_never_orphans_tool_calls():
    """Esc pressed right after the model emitted a tool call but before the
    tool ran must NOT leave an assistant tool_calls message in history with no
    tool result — that history is persisted and re-sent, and strict backends
    reject it ('insufficient tool messages following tool_calls')."""
    from opencode_py.providers.base import ToolCall

    tc = ToolCall(id="c1", name="grep", arguments="{}")
    flag = {"value": False}

    class InterruptAfterCall:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            on_event(ProviderEvent(kind="tool_call", tool_calls=[tc]))
            flag["value"] = True  # user hits Esc right here
            return "opencode", "deepseek-v4-flash-free"

    rot = InterruptAfterCall()
    events = []
    loop = make_loop(rot)
    loop.interrupt = lambda: flag["value"]
    loop.on_event = events.append
    result = loop.run_turn("do something")
    assert any(e.get("kind") == "interrupted" for e in events)
    _assert_no_orphan_tool_calls(loop.get_history())
    tool_msgs = [m for m in loop.get_history() if m.get("role") == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["tool_call_id"] == "c1"


def test_interrupt_during_parallel_tools_never_orphans():
    """Same guarantee for the parallel fan-out path: every declared call id in
    the assistant message is answered by a tool result even when interrupted."""
    from opencode_py.providers.base import ToolCall

    calls = [
        ToolCall(id="a1", name="grep", arguments="{}"),
        ToolCall(id="b1", name="glob", arguments="{}"),
    ]
    flag = {"value": False}

    class ParallelInterrupt:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            on_event(ProviderEvent(kind="tool_call", tool_calls=calls))
            flag["value"] = True
            return "opencode", "deepseek-v4-flash-free"

    loop = make_loop(ParallelInterrupt())
    loop.interrupt = lambda: flag["value"]
    loop.run_turn("fan out")
    _assert_no_orphan_tool_calls(loop.get_history())
    tool_msgs = [m for m in loop.get_history() if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"a1", "b1"}


def test_abort_reaches_rotation_and_subagents():
    """abort() must forward to the active rotation AND every sub-agent's, so
    the TUI can close all blocked streams instantly on interrupt."""
    calls = []

    class R(FakeRotation):
        def abort(self):
            calls.append("rotation")

    class FakeSub:
        def abort(self):
            calls.append("sub")

    loop = make_loop(R([]))
    loop.subagents["s1"] = FakeSub()
    loop.abort()
    assert calls == ["rotation", "sub"]


def test_abort_is_safe_noop_without_active_stream():
    """abort() on an idle engine (no stream in flight) must be a no-op."""
    loop = make_loop(FakeRotation([]))
    loop.abort()  # must not raise


def test_run_tool_resolves_relative_paths_against_directory(tmp_path, monkeypatch):
    """Relative filePaths must resolve against the session directory, not
    Path.cwd(): launching from a subdirectory of the git worktree must not make
    write/edit/read act on cwd-relative paths while the undo snapshot uses
    self.directory."""
    import os
    from opencode_py.tools import build_registry

    base = tmp_path
    workdir = base / "project"
    workdir.mkdir()
    monkeypatch.chdir(base)  # cwd = base, loop directory = base/project

    (workdir / "file.txt").write_text("original")
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    loop = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=workdir,
        agent="build",
    )

    r = loop.run_tool("read", {"filePath": "file.txt"})
    assert r.get("error") is None, r
    r = loop.run_tool("write", {"filePath": "file.txt", "content": "changed", "force": True})
    assert r.get("error") is None, r
    assert (workdir / "file.txt").read_text() == "changed"
    assert not (base / "file.txt").exists(), "must not write to cwd"


def test_run_tool_keeps_absolute_paths_unchanged(tmp_path):
    from opencode_py.tools import build_registry

    target = tmp_path / "abs.txt"
    target.write_text("a")
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    loop = AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=tmp_path,
        agent="build",
    )
    r = loop.run_tool("read", {"filePath": str(target)})
    assert r.get("error") is None, r


# --------------------------------------------------------------------------
# Parallel tool calls: official opencode launches multiple `task` calls from
# one reply concurrently. Results must land in history in the ORIGINAL call
# order so the assistant tool_calls -> tool-result pairing stays valid.
# --------------------------------------------------------------------------

def test_multi_tool_calls_run_concurrently_and_history_stays_ordered(tmp_path):
    import threading

    loop = make_loop(FakeRotation([]))

    # A barrier sized to the call count PROVES the calls overlap: sequential
    # execution would time out waiting for siblings to arrive. (Not wall-clock
    # based, so a leaked time.sleep patch in another test can't fake it.)
    barrier = threading.Barrier(3)

    def slow_run(arguments):
        barrier.wait(timeout=5)
        return {"output": f"ok:{arguments.get('tag', '')}"}

    fake_tool = SimpleNamespace(run=slow_run)
    loop.registry.get = lambda name: fake_tool

    tcs = [
        ToolCall(id=f"c{i}", name="slowtool", arguments=json.dumps({"tag": str(i)}))
        for i in range(3)
    ]
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=tcs)),
        lambda on: on(ProviderEvent(kind="text_delta", text="all done")),
    ])
    loop.rotation = rot

    result = loop.run_turn("run them")
    assert result.text == "all done"
    # all three calls entered slow_run together -> genuinely parallel
    tool_msgs = [m for m in loop.get_history() if m.get("role") == "tool"]
    # history keeps the original call order c0, c1, c2
    assert [t.get("tool_call_id") for t in tool_msgs] == ["c0", "c1", "c2"]
    assert [m.get("content") for m in tool_msgs] == ["ok:0", "ok:1", "ok:2"]


def test_single_tool_call_still_runs_sequentially():
    loop = make_loop(FakeRotation([]))
    loop.registry.get = lambda name: SimpleNamespace(run=lambda a: {"output": "done"})
    tc = ToolCall(id="c1", name="t", arguments="{}")
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=[tc])),
        lambda on: on(ProviderEvent(kind="text_delta", text="ok")),
    ])
    loop.rotation = rot
    result = loop.run_turn("x")
    assert result.text == "ok"
    tool_msgs = [m for m in loop.get_history() if m.get("role") == "tool"]
    assert tool_msgs[-1].get("content") == "done"


def test_two_task_calls_spawn_subagents_concurrently(tmp_path, monkeypatch):
    """End-to-end fan-out: two `task` calls from one reply spawn two REAL
    sub-agents at the same time (official opencode parallel agents). Both
    finish ok, both subagent_done events fire and the parent history holds both
    tool results."""
    import threading

    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    barrier = threading.Barrier(2)

    def child_factory():
        barrier.wait(timeout=5)  # only satisfies if both spawns overlap
        return FakeRotation([lambda on: on(ProviderEvent(kind="text_delta", text="hi from child"))])

    loop = make_loop(FakeRotation([]))
    loop.provider_factory = child_factory
    events: list[dict] = []
    loop.on_event = events.append

    tcs = [
        ToolCall(id="t1", name="task", arguments=json.dumps({"prompt": "task A", "description": "Agent A", "subagent_type": "build"})),
        ToolCall(id="t2", name="task", arguments=json.dumps({"prompt": "task B", "description": "Agent B", "subagent_type": "build"})),
    ]
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=tcs)),
        lambda on: on(ProviderEvent(kind="text_delta", text="all done")),
    ])
    loop.rotation = rot

    result = loop.run_turn("launch parallel agents")
    assert result.text == "all done"
    starts = [e for e in events if e.get("kind") == "subagent_start"]
    dones = [e for e in events if e.get("kind") == "subagent_done"]
    assert len(starts) == 2, f"starts={len(starts)}"
    assert len(dones) == 2, f"dones={len(dones)}"
    assert all(d.get("ok") is True for d in dones), dones
    assert len({d["session_id"] for d in dones}) == 2  # two distinct sessions
    tool_msgs = [m for m in loop.get_history() if m.get("role") == "tool"]
    assert len(tool_msgs) == 2


def test_six_tool_calls_can_run_at_once():
    """The parallel fan-out pool supports 6 concurrent calls (the model may
    launch 6 sub-agents in one reply)."""
    import threading

    loop = make_loop(FakeRotation([]))
    N = 6
    barrier = threading.Barrier(N)

    def slow_run(arguments):
        barrier.wait(timeout=5)
        return {"output": f"ok:{arguments.get('tag', '')}"}

    loop.registry.get = lambda name: SimpleNamespace(run=slow_run)
    tcs = [ToolCall(id=f"c{i}", name="slowtool", arguments=json.dumps({"tag": str(i)})) for i in range(N)]
    loop.rotation = FakeRotation([
        lambda on: on(ProviderEvent(kind="tool_call", tool_calls=tcs)),
        lambda on: on(ProviderEvent(kind="text_delta", text="all done")),
    ])
    result = loop.run_turn("run six")
    assert result.text == "all done"
    tool_msgs = [m for m in loop.get_history() if m.get("role") == "tool"]
    assert len(tool_msgs) == N
    assert [t.get("tool_call_id") for t in tool_msgs] == [f"c{i}" for i in range(N)]


def test_output_limit_stop_surfaces_error_instead_of_silence():
    """finish_reason=length right after a long thinking phase must surface a
    clear output-limit error — not end the turn silently ('stops in the
    thinking part'). The reasoning already shown stays in history."""

    class ThinkingCutByLength:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            if not tools and messages and messages[-1]["role"].startswith("user"):
                pass
            on_event(ProviderEvent(kind="reasoning_delta", text="long thinking..."))
            on_event(ProviderEvent(kind="done", finish_reason="length"))
            return "opencode", "x-preview-f-free"

    loop = make_loop(ThinkingCutByLength())
    result = loop.run_turn("do the task")
    assert "output-token limit" in result.error
    assert result.finish_reason == "length"
    # the visible thinking is kept as the assistant message; the error must
    # NOT be appended into history as a permanent [system] user message
    assert not any(
        str(m.get("content", "")).startswith("[system]") for m in loop._history
    )
    asst = [m for m in loop._history if m.get("role") == "assistant"][-1]
    assert asst.get("reasoning_content") == "long thinking..."
    assert "(no response)" not in str(asst.get("content"))


def test_normal_done_finish_reason_does_not_error():
    class PlainAnswer:
        def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
            on_event(ProviderEvent(kind="text_delta", text="all done"))
            on_event(ProviderEvent(kind="done", finish_reason="stop"))
            return "opencode", "x-preview-f-free"

    loop = make_loop(PlainAnswer())
    result = loop.run_turn("hello")
    assert result.error == ""
    assert result.text == "all done"


def test_subagent_depth_limit_enforced():
    """cfg.subagent_depth bounds task nesting: a child at the limit is told to
    do the work directly instead of recursing further."""
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    cfg.subagent_depth = 1
    parent = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        provider=SimpleNamespace(),
        agent="build",
    )
    parent.provider_factory = lambda: SimpleNamespace()

    child = AgentLoop(
        cfg=cfg,
        registry=SimpleNamespace(),
        directory=Path("."),
        provider=SimpleNamespace(),
        agent="build",
        depth=1,
    )
    blocked = child.spawn_task({"prompt": "go deeper", "description": "lvl2"})
    assert blocked.get("error") and "depth limit" in blocked["output"]

    # parent at depth 0 still spawns fine (monkeypatch run_turn to no-op fast)
    import types

    from opencode_py.agent.loop import TurnResult

    def fake_run(self, prompt):
        r = TurnResult()
        r.text = "ok"
        return r

    orig = AgentLoop.run_turn
    AgentLoop.run_turn = fake_run
    try:
        ok = parent.spawn_task({"prompt": "do it", "description": "lvl1"})
    finally:
        AgentLoop.run_turn = orig
    assert not ok.get("error"), ok


def test_parallel_drain_join_is_interruptible():
    """The final result join must not block forever on a hung tool: with the
    interrupt flag set, pending futures resolve to interrupted rows (2nd ESC
    is never ignored)."""
    import threading
    import time

    loop = make_loop(object())
    started = threading.Event()

    def hang(name, arguments, call_id=""):
        started.set()
        time.sleep(30.0)
        return {"output": "never"}

    loop.run_tool = hang  # type: ignore[assignment]
    loop.interrupt = lambda: True
    prepared = [({"id": "c1"}, "bash", {"command": "sleep 30"})]
    t0 = time.monotonic()
    out = loop._run_tools_parallel(prepared)
    dt = time.monotonic() - t0
    assert dt < 10.0
    assert out[0][2]["error"] is True
    assert out[0][2].get("stopped") is True


def test_auto_continue_off_by_default_and_roundtrips():
    cfg = Config()
    assert cfg.auto_continue is False
    cfg2 = Config.from_dict({"auto_continue": True})
    assert cfg2.auto_continue is True
    assert cfg2.as_dict()["auto_continue"] is True


def test_auto_continue_off_stops_on_text():
    rot = FakeRotation([lambda on: on(ProviderEvent(kind="text_delta", text="hi there"))])
    loop = make_loop(rot)
    assert loop.cfg.auto_continue is False
    result = loop.run_turn("hi")
    assert rot.calls == 1
    assert result.text == "hi there"
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert users == ["hi"]


def test_auto_continue_on_nudges_with_exact_words(monkeypatch):
    import opencode_py.agent.loop as loop_mod

    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_WAIT_S", 0.0)
    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_MAX_NUDGES", 2)
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="text_delta", text="a")),
        lambda on: on(ProviderEvent(kind="text_delta", text="b")),
        lambda on: on(ProviderEvent(kind="text_delta", text="c")),
    ])
    loop = make_loop(rot)
    loop.cfg.auto_continue = True
    loop.run_turn("hi")
    assert loop_mod.AUTO_CONTINUE_MSG == "keep going dont stop keep going never stop"
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert users[0] == "hi"
    assert len(users) > 1
    assert all(u == loop_mod.AUTO_CONTINUE_MSG for u in users[1:])
    assert rot.calls >= 2


def test_auto_continue_silent_after_tools_nudges_then_finishes(monkeypatch):
    """Your whoami shape: tools ran, then a silent (empty-text) stop.
    auto_continue must send the nudge so the final summary always lands."""
    import opencode_py.agent.loop as loop_mod

    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_WAIT_S", 0.0)
    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_MAX_NUDGES", 3)

    def toolbash(on):
        on(ProviderEvent(kind="tool_call", tool_calls=[
            ToolCall(id="c1", name="bash", arguments='{"command":"echo hi"}', index=0)]))

    rot = FakeRotation([toolbash, lambda on: None, lambda on: on(ProviderEvent(kind="text_delta", text="you are u0_a1164"))])
    loop = make_loop(rot)
    loop.cfg.auto_continue = True
    result = loop.run_turn("run who am i")
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert loop_mod.AUTO_CONTINUE_MSG in users
    assert result.text == "you are u0_a1164"


def test_auto_continue_stops_after_work(monkeypatch):
    import opencode_py.agent.loop as loop_mod

    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_WAIT_S", 0.0)
    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_MAX_NUDGES", 5)

    def toolbash(on):
        on(ProviderEvent(kind="tool_call", tool_calls=[
            ToolCall(id="c9", name="bash", arguments='{"command":"echo WORKED"}', index=0)]))

    rot = FakeRotation([toolbash, lambda on: on(ProviderEvent(kind="text_delta", text="all done"))])
    loop = make_loop(rot)
    loop.cfg.auto_continue = True
    result = loop.run_turn("hi")
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert sum(1 for u in users if u == loop_mod.AUTO_CONTINUE_MSG) == 0
    assert result.tool_calls_made == 1
    assert result.text == "all done"


def test_auto_continue_repeat_chatter_stops(monkeypatch):
    """New-session safety: identical chatter after a nudge stops, not 25-spam."""
    import opencode_py.agent.loop as loop_mod

    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_WAIT_S", 0.0)
    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_MAX_NUDGES", 25)
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="text_delta", text="hello there")),
        lambda on: on(ProviderEvent(kind="text_delta", text="hello there")),
        lambda on: on(ProviderEvent(kind="text_delta", text="hello there")),
        lambda on: on(ProviderEvent(kind="text_delta", text="hello there")),
    ])
    loop = make_loop(rot)
    loop.cfg.auto_continue = True
    loop.run_turn("hi")
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert users.count(loop_mod.AUTO_CONTINUE_MSG) <= 2
    assert rot.calls <= 3


def test_auto_continue_queued_user_wins(monkeypatch):
    import opencode_py.agent.loop as loop_mod

    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_WAIT_S", 0.0)
    monkeypatch.setattr(loop_mod, "AUTO_CONTINUE_MAX_NUDGES", 5)
    rot = FakeRotation([
        lambda on: on(ProviderEvent(kind="text_delta", text="a")),
        lambda on: on(ProviderEvent(kind="text_delta", text="b")),
        lambda on: on(ProviderEvent(kind="text_delta", text="c")),
    ])
    loop = make_loop(rot)
    loop.cfg.auto_continue = True
    loop.queue_prompt("REAL-USER-MSG")
    loop.run_turn("hi")
    users = [m.get("content") for m in loop.get_history() if m.get("role") == "user"]
    assert users[1] == "REAL-USER-MSG"
