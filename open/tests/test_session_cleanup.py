"""Session cleanup: squeeze old big bodies, pins, vacuum rules (tmp_path only)."""

import gzip
import json
import time

from opencode_py import session as S
from opencode_py.globals import Path as G


def _mk(monkeypatch, tmp_path, sid, age_days, nbytes, title="t", msgs=True):
    monkeypatch.setattr(G, "sessions_dir", staticmethod(lambda: tmp_path))
    S._session_cache.clear()
    body = {
        "id": sid, "title": title, "created": time.time() - age_days * 86400,
        "messages": [{"role": "user", "content": "x" * 100}] if msgs else [],
    }
    text = json.dumps(body)
    if len(text) < nbytes:
        body["messages"] = [{"role": "user", "content": "x" * nbytes}]
        text = json.dumps(body)
    (tmp_path / f"{sid}.json").write_text(text, encoding="utf-8")
    S._session_cache.clear()
    return tmp_path / f"{sid}.json"


def test_squeeze_old_big_and_rehydrate(tmp_path, monkeypatch):
    p = _mk(monkeypatch, tmp_path, "big1", age_days=10, nbytes=600 * 1024)
    S._session_cache.clear()
    rep = S.squeeze_sessions()
    assert rep["squeezed"] == 1
    assert rep["bytes_saved"] > 0
    assert not p.exists()
    gz = tmp_path / "big1.json.gz"
    assert gz.exists()
    # same open path: picker lists it, load rehydrates it
    S._session_cache.clear()
    ids = [s.id for s in S._list_sessions_all()]
    assert "big1" in ids
    loaded = S.load_session("big1")
    assert loaded is not None and loaded.id == "big1"
    assert loaded.messages and loaded.messages[0]["role"] == "user"


def test_squeeze_skips_young_small_live_pinned(tmp_path, monkeypatch):
    _mk(monkeypatch, tmp_path, "young", age_days=1, nbytes=600 * 1024)
    _mk(monkeypatch, tmp_path, "small", age_days=10, nbytes=1000)
    _mk(monkeypatch, tmp_path, "live1", age_days=10, nbytes=600 * 1024)
    _mk(monkeypatch, tmp_path, "pin1", age_days=10, nbytes=600 * 1024)
    assert S.set_pinned("pin1", True) is True
    S._session_cache.clear()
    rep = S.squeeze_sessions(live_ids={"live1"})
    assert rep["squeezed"] == 0
    for sid in ("young", "small", "live1", "pin1"):
        assert (tmp_path / f"{sid}.json").exists()
    # unpin re-arms (live1 also squeezes now — it was only protected by live_ids)
    assert S.set_pinned("pin1", False) is True
    S._session_cache.clear()
    rep2 = S.squeeze_sessions()
    assert rep2["squeezed"] == 2
    assert (tmp_path / "pin1.json.gz").exists()
    assert (tmp_path / "live1.json.gz").exists()


def test_squeeze_never_touches_children(tmp_path, monkeypatch):
    _mk(monkeypatch, tmp_path, "parent", age_days=10, nbytes=600 * 1024)
    monkeypatch.setattr(G, "sessions_dir", staticmethod(lambda: tmp_path))
    S._session_cache.clear()
    (tmp_path / "kid.json").write_text(json.dumps({
        "id": "kid", "title": "k", "created": time.time() - 10 * 86400,
        "parent_id": "parent", "messages": [{"role": "user", "content": "y" * 600 * 1024}],
    }), encoding="utf-8")
    S._session_cache.clear()
    S.squeeze_sessions()
    assert (tmp_path / "kid.json").exists()


def test_vacuum_respects_pins_and_kills_junk(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "sessions_dir", staticmethod(lambda: tmp_path))
    S._session_cache.clear()
    for i in range(4):
        (tmp_path / f"old{i}.json").write_text(json.dumps(
            {"id": f"old{i}", "title": f"t{i}", "created": 1000 + i,
             "messages": [{"role": "user", "content": "x"}]}))
    (tmp_path / "junk.json").write_text(json.dumps(
        {"id": "junk", "title": "", "created": time.time(), "messages": []}))
    assert S.set_pinned("old0", True) is True
    S._session_cache.clear()
    rep = S.vacuum_sessions(2)
    assert (tmp_path / "old0.json").exists()  # pinned survives
    assert not (tmp_path / "junk.json").exists()  # junk goes even when young
    assert rep["pruned"] >= 2
    S._session_cache.clear()


def test_delete_removes_gz_too(tmp_path, monkeypatch):
    _mk(monkeypatch, tmp_path, "gone", age_days=10, nbytes=600 * 1024)
    S._session_cache.clear()
    S.squeeze_sessions()
    assert (tmp_path / "gone.json.gz").exists()
    assert S.delete_session("gone") is True
    assert not (tmp_path / "gone.json.gz").exists()
    S._session_cache.clear()


def test_cleanup_and_pin_commands(tmp_path, monkeypatch):
    from opencode_py.commands import build_registry
    from opencode_py.config import Config

    _mk(monkeypatch, tmp_path, "c1", age_days=10, nbytes=600 * 1024)
    monkeypatch.chdir(tmp_path)
    reg = build_registry()
    assert reg.get("cleanup") is not None
    assert reg.get("pin") is not None
    seen = []

    class Ctx:
        config = Config.from_dict({"session_file_cap": 0})
        preview_only = True
        get_session = None

        def reply(self, text):
            seen.append(text)

    reg.get("cleanup").handler(Ctx(), "")
    assert seen and "Dry run" in seen[0]
