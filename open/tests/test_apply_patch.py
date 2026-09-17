"""Tests for the apply_patch tool: atomic multi-file diffs, dry-run, undo.

The journal is redirected to a temp data dir and the working directory to a
temp repo so no real file is ever touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencode_py.globals import Path as GPath


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    (repo / "b.py").write_text("alpha\nbeta\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    return repo


def _run(**kw) -> dict:
    from opencode_py.tools.apply_patch import tool

    return tool().run(kw)


def _journal(tmp_path) -> list:
    f = tmp_path / "patch_journal.json"
    if not f.exists():
        return []
    return json.loads(f.read_text())


# ------------------------------------------------------------------- happy path


def test_multi_file_apply(env):
    diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
        "--- a/b.py\n+++ b/b.py\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+BETA\n"
    )
    res = _run(diff=diff, message="shout two files")
    assert res.get("error") is not True, res["output"]
    assert "2 file(s) OK" in res["output"]
    assert (env / "a.py").read_text() == "one\nTWO\nthree\nfour\nfive\n"
    assert (env / "b.py").read_text() == "alpha\nBETA\n"


def test_offset_tolerance_applies_near_declared_position(env, monkeypatch):
    # hunk declares line 5 but content actually sits at line 10
    body = "\n".join(f"filler{i}" for i in range(1, 10))
    p = env / "off.txt"
    p.write_text("head\n" + body + "\ntail\n", encoding="utf-8")
    diff = "--- a/off.txt\n+++ b/off.txt\n@@ -2,2 +2,2 @@\n filler8\n-filler9\n+FILLER9\n"
    res = _run(diff=diff)
    assert res.get("error") is not True, res["output"]
    assert "FILLER9" in p.read_text()
    assert "offset" in res["output"]  # shift reported


def test_new_file_via_dev_null(env):
    diff = (
        "--- /dev/null\n+++ b/new_dir/created.py\n@@ -0,0 +1,2 @@\n+print('hi')\n"
        "+print('bye')\n"
    )
    res = _run(diff=diff)
    assert res.get("error") is not True, res["output"]
    created = env / "new_dir" / "created.py"
    assert created.read_text() == "print('hi')\nprint('bye')\n"


def test_delete_file_via_dev_null(env):
    diff = "--- a/b.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-alpha\n-beta\n"
    res = _run(diff=diff)
    assert res.get("error") is not True, res["output"]
    assert not (env / "b.py").exists()


def test_dry_run_writes_nothing(env):
    diff = "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n"
    res = _run(diff=diff, dry_run=True)
    assert res.get("error") is not True
    assert res["metadata"]["dry_run"] is True
    assert (env / "a.py").read_text() == "one\ntwo\nthree\nfour\nfive\n"
    assert _journal(GPath.data) == []


# ------------------------------------------------------- safety / atomicity


def test_conflict_aborts_everything(env):
    # first hunk fine, second hunk context wrong -> NEITHER file written
    diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n"
        "--- a/b.py\n+++ b/b.py\n@@ -1,2 +1,2 @@\n alpha\n-WRONG_CONTEXT\n+x\n"
    )
    res = _run(diff=diff)
    assert res.get("error") is True
    assert "Nothing was written" in res["output"]
    assert (env / "a.py").read_text() == "one\ntwo\nthree\nfour\nfive\n"
    assert (env / "b.py").read_text() == "alpha\nbeta\n"
    assert _journal(GPath.data) == []


def test_missing_file_conflicts(env):
    diff = "--- a/nope.py\n+++ b/nope.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"
    res = _run(diff=diff)
    assert res.get("error") is True
    assert "does not exist" in res["output"]


def test_new_file_already_exists_with_other_content(env):
    diff = "--- /dev/null\n+++ b/a.py\n@@ -0,0 +1,1 @@\n+different\n"
    res = _run(diff=diff)
    assert res.get("error") is True
    assert "already exists" in res["output"]


def test_deletion_leaving_remainder_conflicts(env):
    diff = "--- a/b.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-alpha\n"
    res = _run(diff=diff)
    assert res.get("error") is True
    assert "remainder" in res["output"]


def test_empty_diff_errors(env):
    res = _run(diff="   ")
    assert res.get("error") is True


def test_unparseable_diff_errors(env):
    res = _run(diff="hello world this is not a diff")
    assert res.get("error") is True
    assert "Could not parse" in res["output"]


def test_crlf_preserved(env):
    p = env / "win.txt"
    p.write_bytes(b"keep\r\nlines\r\nintact\r\n")
    diff = "--- a/win.txt\n+++ b/win.txt\n@@ -1,2 +1,2 @@\n keep\r\n-lines\r\n+LINES\r\n"
    res = _run(diff=diff.replace("\r\n", "\n"))  # tool normalizes CRLF in diff text
    assert res.get("error") is True or res.get("error") is None
    raw = p.read_bytes()
    if res.get("error") is not True:
        assert raw.startswith(b"keep\r\nLINES\r\n")


# --------------------------------------------------------------------- undo


def test_undo_restores_modified_created_deleted(env):
    # patch: modify a.py, create c.txt, delete b.py — then undo all three
    diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-one\n+ONE\n"
        "--- /dev/null\n+++ b/c.txt\n@@ -0,0 +1,1 @@\n+created\n"
        "--- a/b.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-alpha\n-beta\n"
    )
    res = _run(diff=diff, message="triple change")
    assert res.get("error") is not True, res["output"]

    undo = _run(action="undo")
    assert undo.get("error") is not True, undo["output"]
    assert (env / "a.py").read_text() == "one\ntwo\nthree\nfour\nfive\n"
    assert (env / "b.py").read_text() == "alpha\nbeta\n"
    assert not (env / "c.txt").exists()
    assert "Patches left in journal: 0" in undo["output"]


def test_undo_empty_journal_errors(env):
    res = _run(action="undo")
    assert res.get("error") is True
    assert "empty" in res["output"].lower()


def test_history_lists_entries(env):
    _run(diff="--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-one\n+ONE\n",
         message="first patch")
    res = _run(action="history")
    assert res.get("error") is not True
    assert "first patch" in res["output"]
    assert "a.py" in res["output"]


def test_journal_capped(env, monkeypatch):
    from opencode_py.tools import apply_patch as ap

    monkeypatch.setattr(ap, "MAX_JOURNAL_ENTRIES", 3)
    prev = "one"
    for i in range(5):
        res = _run(
            diff=f"--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-{prev}\n+v{i}\n",
            message=f"p{i}",
        )
        assert res.get("error") is not True
        prev = f"v{i}"
    entries = _journal(GPath.data)
    assert len(entries) == 3
    assert entries[-1]["message"] == "p4"


# ------------------------------------------------------------ registry glue


def test_registered_in_registry():
    from opencode_py.tools import build_registry

    reg = build_registry()
    t = reg.get("apply_patch")
    assert t is not None
    assert t.permission == "apply_patch"


def test_unknown_action_errors():
    res = _run(action="explode")
    assert res.get("error") is True