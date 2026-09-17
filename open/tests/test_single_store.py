"""Single-store history: spill big old tool bodies, rehydrate on demand."""

import json
import os


def _patch_spill(monkeypatch, tmp_path):
    """spill_body/read_spill call module-global _spill_dir: patch it there."""
    import opencode_py.agent.trim as T

    monkeypatch.setattr(T, "_spill_dir",
                        lambda sid="": tmp_path / "spill" / (sid or "d"))


def test_spill_and_rehydrate_roundtrip(tmp_path, monkeypatch):
    from opencode_py.agent import trim as T

    _patch_spill(monkeypatch, tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read",
                     "content": "X" * 20000})
    full_bytes = sum(len(str(m.get("content") or "")) for m in msgs)
    slim, n = T.spill_old_tools(msgs, session_id="s1", keep_turns=2)
    assert n > 0
    slim_bytes = sum(len(str(m.get("content") or "")) for m in slim)
    assert slim_bytes < full_bytes // 2
    # receipts carry spill keys
    keys = [m["spill_meta"]["key"] for m in slim
            if isinstance(m, dict) and isinstance(m.get("spill_meta"), dict)]
    assert len(keys) == n
    # rehydrate restores byte-exact
    back = T.rehydrate_spills(slim)
    assert sum(len(str(m.get("content") or "")) for m in back) == full_bytes
    assert all("spill_meta" not in m for m in back if isinstance(m, dict))


def test_spill_keeps_recent_turns_verbatim(tmp_path, monkeypatch):
    from opencode_py.agent import trim as T

    _patch_spill(monkeypatch, tmp_path)
    msgs = [{"role": "user", "content": "q0"},
            {"role": "tool", "tool_call_id": "c0", "name": "read", "content": "Y" * 20000},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "mid"},
            {"role": "user", "content": "q2"},
            {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "Z" * 20000}]
    slim, n = T.spill_old_tools(msgs, session_id="s2", keep_turns=1)
    # last turn stays full, older spills
    assert n == 1
    assert slim[1]["content"] == "Y" * 20000 or "spill_meta" in slim[1]
    assert slim[5]["content"] == "Z" * 20000


def test_spill_never_breaks_pairing(tmp_path, monkeypatch):
    from opencode_py.agent import trim as T

    _patch_spill(monkeypatch, tmp_path)
    msgs = [{"role": "user", "content": "a"},
            {"role": "assistant", "content": "b",
             "tool_calls": [{"id": "c9", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c9", "name": "read", "content": "W" * 20000},
            {"role": "user", "content": "c"}]
    slim, _ = T.spill_old_tools(msgs, session_id="s3", keep_turns=1)
    tool = [m for m in slim if m.get("role") == "tool"][0]
    assert tool["tool_call_id"] == "c9"
    assert tool["name"] == "read"
    back = T.rehydrate_spills(slim)
    assert [m for m in back if m.get("role") == "tool"][0]["content"] == "W" * 20000


def test_read_spill_bad_key_is_none():
    from opencode_py.agent.trim import read_spill

    assert read_spill("") is None
    assert read_spill("nope") is None
    assert read_spill("spill:x:999999") is None


def test_loop_rehydrates_on_get_history(tmp_path, monkeypatch):
    from pathlib import Path as _P

    from opencode_py.agent.loop import AgentLoop
    from opencode_py.config import Config
    from opencode_py.tools import build_registry

    _patch_spill(monkeypatch, tmp_path)
    cfg = Config.from_dict({})
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg),
                     directory=_P("."), agent="build")
    loop.session_id = "s4"
    for i in range(8):
        loop._history.append({"role": "user", "content": f"q{i}"})
        loop._history.append({"role": "assistant", "content": f"a{i}"})
        loop._history.append({"role": "tool", "tool_call_id": f"c{i}",
                              "name": "read", "content": "Q" * 20000})
    full = sum(len(str(m.get("content") or "")) for m in loop._history)
    assert len(loop._history) > 20
    from opencode_py.agent import trim as T
    slim2, n2 = T.spill_old_tools(loop._history, session_id="s4", keep_turns=2)
    assert n2 > 0
    loop._history = slim2
    got = loop.get_history()
    assert sum(len(str(m.get("content") or "")) for m in got) == full
