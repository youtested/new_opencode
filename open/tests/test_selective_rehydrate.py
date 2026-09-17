"""Selective rehydrate: request path restores only survivors (save stays full)."""

from opencode_py.agent import trim as T


def _slim(ntools=6, size=20000):
    msgs = [{"role": "user", "content": "hi"}]
    for i in range(ntools):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read",
                     "content": "X" * size})
    slim, n = T.spill_old_tools(msgs, session_id="sel1", keep_turns=2)
    assert n > 0
    return slim, n


def test_request_history_skips_rehydrate_when_shrink_recuts(tmp_path, monkeypatch):
    import opencode_py.agent.trim as TM

    monkeypatch.setattr(TM, "_spill_dir", lambda sid="": tmp_path / "spill" / (sid or "d"))
    from pathlib import Path as _P

    from opencode_py.agent.loop import AgentLoop
    from opencode_py.config import Config
    from opencode_py.tools import build_registry

    slim, n = _slim()
    cfg = Config.from_dict({})
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg),
                     directory=_P("."), agent="build")
    loop.session_id = "sel1"
    loop._history = slim
    req = loop.request_history()
    # same bytes the old path sends (shrink output identical, same cfg)...
    # NB: default cfg is low-RAM auto (trim 1/200 on this 1.7GB phone), so
    # compare against the LOOP's own trim values, not hardcoded 2/500.
    kt = int(getattr(loop.cfg, "trim_keep_turns", 2) or 2)
    mc = int(getattr(loop.cfg, "trim_max_chars", 500) or 500)
    old = T.shrink_tool_history(T.rehydrate_spills(list(slim)), kt, mc)
    assert sum(len(str(m.get("content") or "")) for m in req) == \
        sum(len(str(m.get("content") or "")) for m in old)
    # ...but zero spilled bodies restored (all re-cut by the shrink)
    assert not any(isinstance(m, dict) and isinstance(m.get("spill_meta"), dict)
                   and not str(m.get("content") or "").startswith(T.RECEIPT_PREFIX)
                   for m in req)


def test_request_history_restores_survivors(tmp_path, monkeypatch):
    import opencode_py.agent.trim as TM

    monkeypatch.setattr(TM, "_spill_dir", lambda sid="": tmp_path / "spill" / (sid or "d"))
    from pathlib import Path as _P

    from opencode_py.agent.loop import AgentLoop
    from opencode_py.config import Config
    from opencode_py.tools import build_registry

    # keep_turns high: shrink keeps everything, so spilled survivors restore
    msgs = [{"role": "user", "content": "q0"},
            {"role": "tool", "tool_call_id": "c0", "name": "read", "content": "Y" * 20000},
            {"role": "user", "content": "q1"}]
    slim, n = T.spill_old_tools(msgs, session_id="sel2", keep_turns=5)
    cfg = Config.from_dict({"trim": {"keep_turns": 5, "max_chars": 50000}})
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg),
                     directory=_P("."), agent="build")
    loop.session_id = "sel2"
    loop._history = slim
    req = loop.request_history()
    # with a generous trim nothing is cut: bodies come back full
    assert any(str(m.get("content") or "") == "Y" * 20000 for m in req)


def test_get_history_still_saves_full(tmp_path, monkeypatch):
    import opencode_py.agent.trim as TM

    monkeypatch.setattr(TM, "_spill_dir", lambda sid="": tmp_path / "spill" / (sid or "d"))
    from pathlib import Path as _P

    from opencode_py.agent.loop import AgentLoop
    from opencode_py.config import Config
    from opencode_py.tools import build_registry

    slim, n = _slim()
    cfg = Config.from_dict({})
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg),
                     directory=_P("."), agent="build")
    loop.session_id = "sel1"
    loop._history = slim
    full = loop.get_history()
    assert sum(len(str(m.get("content") or "")) for m in full) > \
        sum(len(str(m.get("content") or "")) for m in loop.request_history())
