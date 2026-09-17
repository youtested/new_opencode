"""Tests for the `remember` and `summarize_file` tools.

`remember` persists notes under the data dir (patched to a temp dir here), and
`summarize_file` returns a structural map of a single text file. Also verifies
that saved notes are injected into the system prompt so they survive restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencode_py.agent.system import build_system_prompt, _load_memory
from opencode_py.config import Config
from opencode_py.globals import Path as GPath
from opencode_py.tools import build_registry
from opencode_py.tools.remember import _add, _clear, _delete, _list, _project


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path)
    return tmp_path


def _run_action(action: str, **kwargs) -> dict:
    from opencode_py.tools.remember import tool as remember_tool

    return remember_tool().run({**(kwargs or {}), "action": action})


def _run_summarize(filePath: str) -> dict:
    from opencode_py.tools.summarize_file import tool as summarize_tool

    return summarize_tool().run({"filePath": filePath})


def _make_py(tmp_path) -> Path:
    p = tmp_path / "engine.py"
    p.write_text(
        '"""The engine."""\n'
        "\n"
        "import json\n"
        "\n"
        "class SessionStore:\n"
        '    """Session cache wrapper."""\n'
        "    def get(self):\n"
        "        pass\n"
        "    def put(self):\n"
        "        pass\n"
        "\n"
        "def run():\n"
        '    """Entry point."""\n'
        "    store = SessionStore()\n"
        "    return store.get()\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------- remember


def test_remember_add_persists(data_dir):
    res = _run_action("add", text="always use /opt/bin/python")
    assert res.get("error") is not True
    assert "always use /opt/bin/python" in res["output"]
    assert (data_dir / "memory.json").exists()
    raw = json.loads((data_dir / "memory.json").read_text(encoding="utf-8"))
    assert len(raw["entries"]) == 1
    assert raw["entries"][0]["text"] == "always use /opt/bin/python"


def test_remember_add_rejects_empty(data_dir):
    res = _run_action("add", text="   ")
    assert res["error"] is True
    assert "empty" in res["output"].lower()


def test_remember_list_shows_saved(data_dir):
    _run_action("add", text="note one")
    _run_action("add", text="note two")
    res = _run_action("list")
    assert "note one" in res["output"]
    assert "note two" in res["output"]


def test_remember_delete_by_id(data_dir):
    _run_action("add", text="note one")
    _run_action("add", text="note two")
    res = _run_action("delete", id=1)
    assert res.get("error") is not True
    remaining = _run_action("list")
    assert "note one" not in remaining["output"]
    assert "note two" in remaining["output"]


def test_remember_delete_missing_id(data_dir):
    res = _run_action("delete", id=999)
    assert res["error"] is True


def test_remember_clear_project(data_dir, monkeypatch):
    _run_action("add", text="remove me")
    res = _run_action("clear")
    assert res.get("error") is not True
    listed = _run_action("list")
    assert "remove me" not in listed["output"]


def test_remember_scopes_by_project(data_dir, monkeypatch):
    proj_a = "project_a"
    proj_b = "project_b"
    with monkeypatch.context() as m:
        m.setattr("opencode_py.tools.remember._project", lambda: proj_a)
        _run_action("add", text="a-note")
    with monkeypatch.context() as m:
        m.setattr("opencode_py.tools.remember._project", lambda: proj_b)
        _run_action("add", text="b-note")
    with monkeypatch.context() as m:
        m.setattr("opencode_py.tools.remember._project", lambda: proj_a)
        res_a = _run_action("list")
    assert "a-note" in res_a["output"]
    assert "b-note" not in res_a["output"]


def test_remember_unhappy_action(data_dir):
    res = _run_action("explode")
    assert res["error"] is True


# ------------------------------------------------------- summarize_file


def test_summarize_file_python(tmp_path):
    p = _make_py(tmp_path)
    res = _run_summarize(str(p))
    assert res.get("error") is not True
    assert "engine.py" in res["output"]
    assert "SessionStore" in res["output"]
    assert "run" in res["output"]
    assert "lines" in res["output"]


def test_summarize_file_python_has_line_numbers(tmp_path):
    p = _make_py(tmp_path)
    res = _run_summarize(str(p))
    assert "line 5" in res["output"]  # class SessionStore
    assert "line 12" in res["output"]  # def run


def test_summarize_file_markdown_headings(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Title\n\n## Section One\n\nsome text\n\n### Deep\n", encoding="utf-8")
    res = _run_summarize(str(p))
    assert "# Title" in res["output"]
    assert "Section One" in res["output"]


def test_summarize_file_json_keys(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"name": "o", "items": [1, 2], "ok": true}', encoding="utf-8")
    res = _run_summarize(str(p))
    assert "name" in res["output"]
    assert "items" in res["output"]
    assert "object" in res["output"]


def test_summarize_file_missing(tmp_path):
    res = _run_summarize(str(tmp_path / "nope.txt"))
    assert res["error"] is True


def test_summarize_file_directory(tmp_path):
    res = _run_summarize(str(tmp_path))
    assert res["error"] is True
    assert "one file" in res["output"].lower()


def test_summarize_file_binary(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01\x02\xff\xfe\x00")
    res = _run_summarize(str(p))
    assert res["error"] is True


def test_summarize_file_image_hint(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\r")
    res = _run_summarize(str(p))
    assert res["error"] is True
    assert "read" in res["output"]


def test_summarize_file_text_outline(tmp_path):
    p = tmp_path / "app.log"
    p.write_text("INFO start\nWARN disk full\nINFO done\n", encoding="utf-8")
    res = _run_summarize(str(p))
    assert res.get("error") is not True
    assert "3 lines" in res["output"]


def test_summarize_file_relative_path(tmp_path, monkeypatch):
    p = _make_py(tmp_path)
    monkeypatch.chdir(tmp_path)
    res = _run_summarize("engine.py")
    assert res.get("error") is not True
    assert "SessionStore" in res["output"]


# ------------------------------------------------- registry + prompt glue


def test_registry_has_both_tools():
    registry = build_registry()
    assert "remember" in registry.names()
    assert "summarize_file" in registry.names()


def test_system_prompt_includes_memory(data_dir, tmp_path):
    _add("use tabs not spaces", project="")
    cfg = Config()
    prompt = build_system_prompt(
        directory=tmp_path,
        worktree=tmp_path,
        provider_id="test",
        model_id="test-model",
        cfg=cfg,
    )
    assert "use tabs not spaces" in prompt


def test_load_memory_projects_out_other_project(data_dir):
    _add("this-project-rule", project="zero")
    _add("other-project-rule", project="one")
    block = _load_memory("zero")
    assert "this-project-rule" in block
    assert "other-project-rule" not in block


def test_load_memory_global_loads_everywhere(data_dir):
    _add("global-rule", project="")
    assert "global-rule" in _load_memory("anything")