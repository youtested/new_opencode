"""Tests for the find_symbols tool and its index package.

Covers: the Python AST indexer (defs/methods/callers/imports/locality), the
per-language heuristic indexers (JS, C, Go, Rust, bash), the disk cache
(creation + incremental refresh on edit), query-verb parsing, and registry/TUI
integration. The cache dir is patched to a temp path so tests never touch the
real ~/.cache/opencode_py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencode_py.globals import Path as GPath


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "cache", tmp_path)
    return tmp_path


@pytest.fixture
def repo(tmp_path):
    """A small polyglot repo: python pkg + js + c + go + rust + bash files."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)

    (root / "app.py").write_text(
        "import os\n"
        "from pkg.helpers import fmt_name\n"
        "\n"
        "MAX_LEN = 10\n"
        "\n"
        "class Greeter:\n"
        "    def greet(self, who):\n"
        "        return fmt_name(who, MAX_LEN)\n"
        "\n"
        "def main():\n"
        "    g = Greeter()\n"
        "    print(g.greet('world'))\n",
        encoding="utf-8",
    )
    (root / "pkg" / "helpers.py").write_text(
        "def fmt_name(name, limit):\n"
        "    return name[:limit]\n",
        encoding="utf-8",
    )
    (root / "util.js").write_text(
        'const fs = require("fs");\n'
        "function shout(s) { return s.toUpperCase(); }\n"
        "class Box { }\n"
        "export default shout;\n",
        encoding="utf-8",
    )
    (root / "mem.c").write_text(
        '#include <stdlib.h>\n'
        '#include "mem.h"\n'
        "#define POOL_SIZE 1024\n"
        "struct Pool { int used; };\n"
        "void pool_init(void) { }\n",
        encoding="utf-8",
    )
    (root / "main.go").write_text(
        'package main\n'
        '\n'
        'import (\n'
        '\t"fmt"\n'
        '\t"example.com/proj/util"\n'
        ')\n'
        '\n'
        'func Run() {\n'
        '\tfmt.Println(util.Name())\n'
        '}\n',
        encoding="utf-8",
    )
    (root / "lib.rs").write_text(
        "pub struct Widget;\n"
        "\n"
        "impl Widget {\n"
        "    pub fn draw(&self) {}\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "run.sh").write_text(
        "#!/bin/sh\n"
        "start_service() {\n"
        '  echo started\n'
        "}\n",
        encoding="utf-8",
    )
    return root


def _ask(root: Path, q: str, **kw) -> dict:
    from opencode_py.index.engine import query

    return query(q, root=root, **kw)


# ---------------------------------------------------------------- python AST


def test_def_finds_function_with_signature(repo, cache_dir):
    res = _ask(repo, "def fmt_name")
    assert res.get("error") is not True
    assert "pkg/helpers.py:1" in res["output"]
    assert "def fmt_name(name, limit)" in res["output"]


def test_def_finds_class_and_method_with_container(repo, cache_dir):
    res = _ask(repo, "symbols app.py")
    out = res["output"]
    assert "class Greeter" in out
    assert "(method in Greeter.)" in out  # greet is inside class Greeter


def test_callers_finds_exact_call_sites(repo, cache_dir):
    res = _ask(repo, "callers fmt_name")
    assert res.get("error") is not True
    # called once in Greeter.greet (line 8); the def/import lines are NOT calls
    items = res["metadata"]["items"]
    assert all(i["role"] == "call" for i in items)
    assert any(i["file"] == "app.py" and i["line"] == 8 for i in items)


def test_refs_include_plain_uses(repo, cache_dir):
    res = _ask(repo, "refs MAX_LEN")
    lines = [i["line"] for i in res["metadata"]["items"]]
    assert 8 in lines  # `return fmt_name(who, MAX_LEN)`


def test_deps_local_vs_stdlib(repo, cache_dir):
    res = _ask(repo, "deps app.py")
    meta = res["metadata"]
    assert "pkg.helpers" in meta["local"]
    assert "os" in meta["external"]
    # local dep maps back to a file path
    assert "pkg/helpers.py" in res["output"]


def test_deps_relative_import_is_local(repo, cache_dir):
    (repo / "pkg" / "extra.py").write_text(
        "from .helpers import fmt_name\n", encoding="utf-8"
    )
    res = _ask(repo, "deps pkg/extra.py")
    assert "fmt_name" in "".join(res["metadata"]["local"]) or \
        any(".helpers" in m for m in res["metadata"]["local"])


def test_imports_reverse_lookup(repo, cache_dir):
    res = _ask(repo, "imports helpers")
    assert "app.py" in res["output"]
    assert any(i["file"] == "app.py" for i in res["metadata"]["items"])


def test_module_constant_indexed_as_constant(repo, cache_dir):
    res = _ask(repo, "def MAX_LEN", kind="constant")
    assert res["metadata"]["items"][0]["kind"] == "constant"


# ------------------------------------------------------------- heuristic langs


def test_js_declarations_and_import(repo, cache_dir):
    res = _ask(repo, "symbols util.js")
    names = {i["name"]: i["kind"] for i in res["metadata"]["items"]}
    assert names.get("shout") == "function"
    assert names.get("Box") == "class"


def test_c_macro_include_struct_function(repo, cache_dir):
    res = _ask(repo, "symbols mem.c")
    names = {i["name"]: i["kind"] for i in res["metadata"]["items"]}
    assert names.get("POOL_SIZE") == "macro"
    assert names.get("Pool") == "struct"
    assert names.get("pool_init") == "function"
    deps = _ask(repo, "deps mem.c")["metadata"]["external"]
    assert "stdlib.h" in deps
    assert "mem.h" in deps


def test_go_func_and_import_block(repo, cache_dir):
    res = _ask(repo, "symbols main.go")
    names = {i["name"]: i["kind"] for i in res["metadata"]["items"]}
    assert names.get("Run") == "function"
    imports = set(_ask(repo, "deps main.go")["metadata"]["external"])
    assert "example.com/proj/util" in imports


def test_rust_struct_and_fn(repo, cache_dir):
    res = _ask(repo, "symbols lib.rs")
    names = {i["name"]: i["kind"] for i in res["metadata"]["items"]}
    assert names.get("Widget") == "struct"
    assert names.get("draw") == "function"


def test_bash_function(repo, cache_dir):
    res = _ask(repo, "symbols run.sh")
    names = {i["name"]: i["kind"] for i in res["metadata"]["items"]}
    assert names.get("start_service") == "function"


# --------------------------------------------------------------------- cache


def test_cache_file_created_under_patched_cache_dir(repo, cache_dir):
    _ask(repo, "def main")
    symbols_dirs = list((cache_dir / "symbols").glob("*"))
    assert len(symbols_dirs) == 1
    assert (symbols_dirs[0] / "index.json").exists()
    data = json.loads((symbols_dirs[0] / "index.json").read_text())
    assert data["version"] == 3
    paths = {f["path"] for f in data["files"]}
    assert "app.py" in paths


def test_incremental_refresh_picks_up_edit(repo, cache_dir):
    _ask(repo, "def main")  # cold build
    helpers = repo / "pkg" / "helpers.py"
    helpers.write_text(
        "def fmt_name(name, limit):\n"
        "    return name[:limit]\n"
        "\n"
        "def brand_new_helper():\n"
        "    return 42\n",
        encoding="utf-8",
    )
    res = _ask(repo, "def brand_new_helper")
    assert res["metadata"]["matches"] >= 1
    assert "brand_new_helper" in res["output"]


def test_fresh_rebuild_still_correct(repo, cache_dir):
    _ask(repo, "def main")
    res = _ask(repo, "def main", force=True)
    assert "def main()" in res["output"]


# -------------------------------------------------------------- query parsing


def test_bare_name_defaults_to_def(repo, cache_dir):
    res = _ask(repo, "fmt_name")
    assert "def fmt_name(name, limit)" in res["output"]


def test_unknown_name_reports_empty(repo, cache_dir):
    res = _ask(repo, "def no_such_thing_anywhere")
    assert res["metadata"]["matches"] == 0
    assert "No def found" in res["output"]


def test_substring_match_falls_back(repo, cache_dir):
    res = _ask(repo, "def fmt_")
    assert res["metadata"]["matches"] >= 1  # startswith match on fmt_name


def test_unresolvable_deps_file_errors(repo, cache_dir):
    res = _ask(repo, "deps nope/missing.py")
    assert res.get("error") is True


# ------------------------------------------------------------ tool + registry


def test_tool_registered_in_registry():
    from opencode_py.tools import build_registry

    registry = build_registry()
    assert "find_symbols" in registry.names()


def test_tool_run_via_registry_input(tmp_path, cache_dir, repo, monkeypatch):
    monkeypatch.chdir(repo)
    from opencode_py.tools.find_symbols import tool

    res = tool().run({"query": "callers fmt_name"})
    assert res.get("error") is not True
    assert res["metadata"]["mode"] == "callers"


def test_tool_empty_query_error(cache_dir):
    from opencode_py.tools.find_symbols import tool

    res = tool().run({"query": "   "})
    assert res.get("error") is True


def test_tui_labels_registered():
    from opencode_py.tui.chat_view import TOOL_ICONS, TOOL_NAMES

    assert TOOL_NAMES.get("find_symbols") == "Symbols"
    assert "find_symbols" in TOOL_ICONS