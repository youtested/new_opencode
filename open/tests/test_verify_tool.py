"""Tests for the verify tool: auto-tracking from edit/write/apply_patch and
the syntax/parse checkers themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencode_py.globals import Path as GPath
from opencode_py.tools.verify import (
    _clear_tracked,
    tool,
    track,
    tracked,
    untrack,
)


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path)
    _clear_tracked()
    yield
    _clear_tracked()


def _run(**kw) -> dict:
    return tool().run(kw)


# ------------------------------------------------------------------ checkers


def test_python_ok_and_broken(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("def f():\n    return 1\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n    pass\n", encoding="utf-8")
    res = _run(action="check", paths=[str(good), str(bad)])
    assert res["error"] is True
    lines = res["output"].splitlines()
    assert any(l.startswith("✅") and "good.py" in l for l in lines)
    fail_line = next(l for l in lines if "bad.py" in l)
    assert "❌" in fail_line
    # exact location given so the model can fix without re-reading the file
    assert "line 1" in fail_line
    assert res["metadata"]["failed"] == 1


def test_json_checker(tmp_path):
    ok = tmp_path / "ok.json"
    ok.write_text('{"a": 1}', encoding="utf-8")
    bad = tmp_path / "nope.json"
    bad.write_text("{a: 1,}", encoding="utf-8")
    res = _run(action="check", paths=[str(ok), str(bad)])
    out = res["output"]
    assert "valid JSON" in out
    assert "invalid JSON" in out


def test_unknown_type_reported_unchecked(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello", encoding="utf-8")
    res = _run(action="check", paths=[str(p)])
    assert "no checker" in res["output"]
    assert res["error"] is not True


def test_missing_explicit_path_fails(tmp_path):
    res = _run(action="check", paths=[str(tmp_path / "ghost.py")])
    assert res["error"] is True
    assert "missing" in res["output"]


# ------------------------------------------------------- auto-track lifecycle


def test_write_tracks_and_green_check_clears(tmp_path):
    from opencode_py.tools.write import _write

    target = tmp_path / "tracked.py"
    _write(str(target), "x = 1\n")
    assert str(target.resolve()) in [p for p in tracked()] or any(
        p.endswith("tracked.py") for p in tracked()
    )
    res = _run(action="check")  # implicit: uses tracker
    assert res.get("error") is not True
    assert "tracked list cleared" in res["output"]
    assert tracked() == []


def test_edit_failure_keeps_list_for_retry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from opencode_py.tools.edit import _edit

    f = tmp_path / "mod.py"
    f.write_text("value = 1\n", encoding="utf-8")
    r = _edit(str(f), "value = 1", "value = (\n")  # broken replacement
    assert r.get("error") is not True  # edit itself succeeded (writes broken code!)
    res = _run(action="check")
    assert res["error"] is True
    # failing run must NOT clear the tracker — retry loop needs it
    assert any(p.endswith("mod.py") for p in tracked())
    # fix it via edit, verify goes green and clears
    f.write_text("value = 2\n", encoding="utf-8")
    r2 = _edit(str(f), "value = 2", "value = 3\n")
    assert r2.get("error") is not True
    res2 = _run(action="check")
    assert res2.get("error") is not True
    assert tracked() == []


def test_apply_patch_tracks_creates_and_untracks_deletes(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    from opencode_py.tools.apply_patch import tool as ap_tool

    diff = (
        "--- /dev/null\n+++ b/made.py\n@@ -0,0 +1,1 @@\n+print(1)\n"
        "--- a/old.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-old\n"
    )
    (tmp_path / "old.py").write_text("old\n", encoding="utf-8")
    # pre-seed old.py as tracked to prove deletion untracks it
    track(tmp_path / "old.py", "write")

    res = ap_tool().run({"diff": diff})
    assert res.get("error") is not True, res["output"]

    paths = tracked()
    assert any(p.endswith("made.py") for p in paths)
    assert not any(p.endswith("old.py") for p in paths)

    # verify sees only the created file now
    chk = _run(action="check")
    assert "made.py" in chk["output"]
    assert "old.py" not in chk["output"]


# --------------------------------------------------------------------- misc


def test_reset_action():
    track("/tmp/whatever_a.py", "write")
    track("/tmp/whatever_b.py", "edit")
    res = _run(action="reset")
    assert "cleared" in res["output"]
    assert tracked() == []


def test_empty_check_reports_nothing_to_do():
    res = _run(action="check")
    assert res.get("error") is not True
    assert "Nothing to verify" in res["output"]


def test_registered_in_registry():
    from opencode_py.tools import build_registry

    reg = build_registry()
    t = reg.get("verify")
    assert t is not None
    assert t.permission == "verify"


def test_tui_label_present():
    # label optional; just ensure no crash when absent/present
    from opencode_py.tui.chat_view import TOOL_NAMES

    assert TOOL_NAMES.get("verify") in (None, "Verify")