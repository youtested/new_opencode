"""Tests for network-failure tagging + auto-resume (reconnect watcher).

A network-killed turn must be resumable without duplicating the prompt, and
only transport failures (not model/API errors) may trigger the watcher.
"""

from pathlib import Path

from opencode_py.agent.loop import AgentLoop, TurnResult
from opencode_py.config import Config
from opencode_py.providers.base import ProviderError, ProviderEvent


class FakeRotation:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
        self.calls += 1
        if not self.script:
            on_event(ProviderEvent(kind="text_delta", text="ok"))
            return "opencode", "m"
        step = self.script.pop(0)
        return step(on_event)


def make_loop(rotation):
    from opencode_py.tools import build_registry

    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "big-pickle"
    return AgentLoop(
        cfg=cfg,
        registry=build_registry(cfg),
        directory=Path("."),
        provider=rotation,
        agent="build",
    )


def test_network_error_tags_result():
    def boom(on_event):
        raise ProviderError("network error talking to X: down", retryable=True, network=True)

    loop = make_loop(FakeRotation([boom]))
    loop.cfg.auto_retry = False
    result = loop.run_turn("hello")
    assert result.error
    assert result.network_failed is True


def test_model_error_does_not_tag_network():
    def boom(on_event):
        raise ProviderError("bad model id", retryable=False)

    loop = make_loop(FakeRotation([boom]))
    result = loop.run_turn("hello")
    assert result.error
    assert result.network_failed is False


def test_resume_reruns_without_duplicating_prompt():
    seen = []

    def fail_once(on_event):
        seen.append("try")
        raise ProviderError("network error talking to X: down", retryable=True, network=True)

    loop = make_loop(FakeRotation([fail_once]))
    loop.cfg.auto_retry = False
    first = loop.run_turn("do the thing")
    assert first.network_failed is True
    users = [m for m in loop._history if m.get("role") == "user"]
    assert len(users) == 1

    loop.rotation = FakeRotation([])
    second = loop.resume_turn()
    assert not second.error, second.error
    assert second.text == "ok"
    users = [m for m in loop._history if m.get("role") == "user" and m.get("content") == "do the thing"]
    assert len(users) == 1  # no duplicate prompt


def test_resume_drops_partial_assistant_and_keeps_context():
    loop = make_loop(FakeRotation([]))
    loop._history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "done before"},
        {"role": "user", "content": "now this"},
        {"role": "assistant", "content": "par", "reasoning_content": "half thought"},
    ]
    result = loop.resume_turn()
    assert not result.error, result.error
    roles = [m.get("role") for m in loop._history]
    assert roles.count("user") == 2
    assert loop._history[-1].get("role") == "assistant"
    assert "par" not in (loop._history[2].get("content") or "")


def test_resume_with_no_user_message_errors():
    loop = make_loop(FakeRotation([]))
    loop._history = []
    result = loop.resume_turn()
    assert result.error == "nothing to resume"


def test_provider_error_network_defaults_false():
    e = ProviderError("x")
    assert e.network is False
    assert TurnResult().network_failed is False
