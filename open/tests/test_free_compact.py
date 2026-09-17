"""Free-first compact + self-recall (no model calls, no network)."""

import json
import time


def _msgs(ntools=6, size=20000):
    msgs = [{"role": "user", "content": "hi"}]
    for i in range(ntools):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read",
                     "content": "X" * size})
    return msgs


def test_free_compact_cuts_without_model():
    from opencode_py.agent import trim as T

    msgs = _msgs()
    full = sum(len(str(m.get("content") or "")) for m in msgs)
    cut, n = T.free_compact(msgs, session_id="zz", keep_turns=2)
    assert n > 0
    slim = sum(len(str(m.get("content") or "")) for m in cut)
    assert slim < full // 2
    notes = [m for m in cut if isinstance(m.get("content"), str)
             and m["content"].startswith(T.RECALL_PREFIX)]
    assert len(notes) == n
    assert all("recall:zz:" in m["content"] for m in notes)
    # user + assistant text untouched
    assert [m["content"] for m in cut if m.get("role") == "user"] == \
           [m["content"] for m in msgs if m.get("role") == "user"]


def test_free_compact_never_touches_pairs_or_summaries():
    from opencode_py.agent import trim as T

    msgs = [{"role": "user", "content": "a"},
            {"role": "assistant", "content": "b",
             "tool_calls": [{"id": "c9", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c9", "name": "read", "content": "W" * 20000},
            {"role": "user", "content": "c"}]
    cut, n = T.free_compact(msgs, session_id="zz", keep_turns=1)
    tool = [m for m in cut if m.get("role") == "tool"][0]
    assert tool["tool_call_id"] == "c9"  # pairing intact
    asst = [m for m in cut if m.get("role") == "assistant"][0]
    assert asst["tool_calls"][0]["id"] == "c9"
    # compaction summaries immune
    msgs2 = [{"role": "user", "content": "x", "compaction": True},
             {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "Y" * 20000},
             {"role": "user", "content": "y"}]
    cut2, _ = T.free_compact(msgs2, session_id="zz", keep_turns=5)
    assert cut2[0]["content"] == "x"


def test_rehydrate_recalls_restores_from_full():
    from opencode_py.agent import trim as T

    msgs = _msgs()
    cut, n = T.free_compact(msgs, session_id="zz", keep_turns=2)
    assert n > 0
    back = T.rehydrate_recalls(cut, msgs)
    assert sum(len(str(m.get("content") or "")) for m in back) == \
           sum(len(str(m.get("content") or "")) for m in msgs)


def test_recall_tool_roundtrip(tmp_path, monkeypatch):
    from opencode_py.globals import Path as G
    from opencode_py.session import Session, delete_session, save_session
    from opencode_py.tools.history_search import tool as hs_tool

    monkeypatch.setattr(G, "sessions_dir", staticmethod(lambda: tmp_path))
    from opencode_py import session as S
    S._session_cache.clear()
    body = "BODY-" + "V" * 5000
    s = Session({"id": "rr1", "title": "r", "created": time.time(),
                 "messages": [{"role": "user", "content": "q"},
                              {"role": "tool", "tool_call_id": "c0",
                               "name": "read", "content": body}]})
    save_session(s)
    S._session_cache.clear()
    t = hs_tool()
    r = t.run({"action": "recall", "recall_key": "recall:rr1:1"})
    assert r.get("error") is not True
    assert r["output"] == body
    assert r["metadata"]["chars"] == len(body)
    assert t.run({"action": "recall", "recall_key": "nope"}).get("error") is True
    assert t.run({"action": "recall", "recall_key": "recall:rr1:99"}).get("error") is True
    assert delete_session("rr1") is True
    S._session_cache.clear()


def test_recall_tool_serves_spill_keys(tmp_path, monkeypatch):
    from opencode_py.agent import trim as T
    from opencode_py.tools.history_search import tool as hs_tool

    monkeypatch.setattr(T, "_spill_dir", lambda sid="": tmp_path / "spill" / (sid or "d"))
    key = T.spill_body("sp1", 3, "SPILLED-" + "W" * 3000)
    assert key.startswith("spill:")
    r = hs_tool().run({"action": "recall", "recall_key": key})
    assert r.get("error") is not True
    assert r["output"] == "SPILLED-" + "W" * 3000
