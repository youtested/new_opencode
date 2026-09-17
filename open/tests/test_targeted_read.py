"""Targeted reads: symbol=/pattern=/context= kill blind offset paging.

- read symbol= returns one exact def/class block (AST for .py, header scan otherwise)
- read pattern= returns match groups with surroundings
- grep context=N prints code with matches (legacy tokens preserved)
- find_symbols def rows carry start-end + read symbol= hint
"""

from pathlib import Path

import pytest

from opencode_py.index.engine import query as index_query
from opencode_py.tools import context_ledger as cl
from opencode_py.tools.find_symbols import tool as sym_tool
from opencode_py.tools.grep import tool as grep_tool
from opencode_py.tools.read import tool as read_tool


@pytest.fixture(autouse=True)
def _clean_ledger():
    cl.clear()
    yield
    cl.clear()


def rread(p, **kw):
    return read_tool().run({"filePath": str(p), **kw})


def make_mod(tmp_path: Path) -> Path:
    p = tmp_path / "m.py"
    p.write_text(
        "import os\n\n"
        "class Foo:\n"
        "    def bar(self, x):\n"
        "        y = x + 1\n"
        "        return y\n\n"
        "def top():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    return p


def test_read_symbol_returns_exact_block(tmp_path):
    p = make_mod(tmp_path)
    out = rread(p, symbol="bar")["output"]
    assert "def bar(self, x):" in out
    assert "return y" in out
    assert "def top" not in out


def test_read_symbol_dotted_and_class(tmp_path):
    p = make_mod(tmp_path)
    out = rread(p, symbol="Foo.bar")["output"]
    assert "def bar" in out
    out2 = rread(p, symbol="Foo")["output"]
    assert "class Foo:" in out2


def test_read_symbol_miss_suggests(tmp_path):
    p = make_mod(tmp_path)
    res = rread(p, symbol="nope")
    assert res.get("error") or "No symbol" in res["output"]
    assert "outline" in res["output"]


def test_read_pattern_groups(tmp_path):
    p = make_mod(tmp_path)
    out = rread(p, pattern="return", context=1)["output"]
    assert "return y" in out and "return 1" in out
    res = rread(p, pattern="zzz-no-hit")
    assert "No matches" in res["output"]
    res = rread(p, pattern="([")
    assert res.get("error") and "Invalid regex" in res["output"]


def test_read_pattern_nonpython(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# T\n\n## A\n\ntext\n\n## B\n\nmore\n", encoding="utf-8")
    out = rread(p, pattern="##", context=1)["output"]
    assert "## A" in out and "## B" in out


def test_grep_context_default_and_zero(tmp_path):
    p = make_mod(tmp_path)
    g = grep_tool().run({"pattern": "return", "path": str(p)})
    assert "context=2" in g["output"]
    assert "Line 6" in g["output"] or "return y" in g["output"]
    g0 = grep_tool().run({"pattern": "return", "path": str(p), "context": 0})
    assert "Line 6" in g0["output"]


def test_find_symbols_def_carries_range_and_hint(tmp_path, monkeypatch):
    from opencode_py import globals as G

    monkeypatch.setattr(G.Path, "config", tmp_path / "cfg", raising=False)
    (tmp_path / "w.py").write_text("def hello(x):\n    return x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    res = sym_tool().run({"query": "def hello"})
    assert "read symbol='hello'" in res["output"]
    assert ":1-" in res["output"] or ":1" in res["output"]


def test_grep_context_rg_first_path_once_and_anchors(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("import os\n\n\ndef target_fn(x):\n    return x + 1\n", encoding="utf-8")
    g = grep_tool().run({"pattern": "target_fn", "path": str(p)})
    out = g["output"]
    assert "context=2" in out
    assert "read offset=" in out and "limit=" in out
    assert "read symbol='target_fn'" in out
    # no per-line filepath repeat (path-once rendering)
    assert not any(l.startswith(str(tmp_path)) for l in out.splitlines())


def test_grep_parallel_queries_one_call(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("ALPHA = 1\n", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("BETA = 2\n", encoding="utf-8")
    r = grep_tool().run({"queries": ["ALPHA", {"pattern": "BETA", "path": str(b)}]})
    assert "Batch grep results (2 patterns" in r["output"]
    assert "ALPHA" in r["output"] and "BETA" in r["output"]
    assert r["metadata"]["count"] == 2 and r["metadata"]["succeeded"] == 2


def test_grep_parallel_rejects_empty_and_guards_pattern():
    r = grep_tool().run({"queries": []})
    assert r.get("error") or "non-empty" in r["output"]
    r = grep_tool().run({})
    assert r.get("error") and "queries" in r["output"]


def test_grep_ignores_build_mirror_by_default(tmp_path):
    src = tmp_path / "x.py"
    src.write_text("MARKER = 1\n", encoding="utf-8")
    mirror = tmp_path / "build" / "lib" / "x.py"
    mirror.parent.mkdir(parents=True)
    mirror.write_text("MARKER = 1\n", encoding="utf-8")
    g = grep_tool().run({"pattern": "MARKER", "path": str(tmp_path)})
    assert "x.py" in g["output"]
    assert "build/lib/x.py" not in g["output"] and "build/**" not in g["output"]
    # explicit search inside still works
    g2 = grep_tool().run({"pattern": "MARKER", "path": str(mirror.parent)})
    assert "MARKER" in g2["output"]


def test_grep_flood_is_capped(tmp_path):
    p = tmp_path / "big.py"
    p.write_text("".join(f"def f_{i}():\n    return {i}\n" for i in range(2000)), encoding="utf-8")
    g = grep_tool().run({"pattern": "def ", "path": str(p)})
    assert g["metadata"]["truncated"] is True
    assert len(g["output"]) < 50 * 1024 + 2000


# ---------------------------------------------------------------------------
# any-language exact landing: 1 try, no guessing
# ---------------------------------------------------------------------------

LANG_CASES = {
    "a.js": (
        "export default async function load(url) {\n  const r = 1;\n  return r;\n}\n"
        "class U {\n  field = \"}\";\n  getName() {\n    return 1;\n  }\n}\n",
        [("load", 1, 4), ("U", 5, 10), ("getName", 7, 9)],
    ),
    "c.ts": (
        "export namespace N {\n  export interface P {\n    x: string;\n  }\n}\n",
        [("N", 1, 5)],
    ),
    "d.go": (
        "package main\n\nfunc add(a int, b int) int {\n\tif a > 0 {\n\t\treturn a + b\n\t}\n\treturn b\n}\n",
        [("add", 3, 8)],
    ),
    "e.rs": (
        "pub fn add(a: i32, b: i32) -> i32 {\n    let x = 1;\n    x\n}\n",
        [("add", 1, 4)],
    ),
    "f.java": (
        "public class Main {\n    public static int add(int a, int b) {\n        return a + b;\n    }\n}\n",
        [("Main", 1, 5), ("add", 2, 4)],
    ),
    "g.c": (
        "typedef struct {\n    int x;\n} Point;\n",
        [("Point", 1, 3)],
    ),
    "h.c": (
        "#include <stdio.h>\n\nint add(int a, int b) {\n    return a + b;\n}\n",
        [("add", 3, 5)],
    ),
    "i.php": (
        "<?php\nfunction add($a, $b) {\n    return $a + $b;\n}\n",
        [("add", 2, 4)],
    ),
    "j.rb": (
        "class U\n  def hi\n    1\n  end\nend\n",
        [("U", 1, 5), ("hi", 2, 4)],
    ),
    "k.lua": (
        "function m.add(a)\n  return a\nend\n",
        [("m.add", 1, 3)],
    ),
    "m.cc": (
        "std::string Foo::bar(int x) {\n    return \"\";\n}\n",
        [("bar", 1, 3)],
    ),
    "n.swift": (
        "func greet(name: String) -> String {\n    return name\n}\n",
        [("greet", 1, 3)],
    ),
}


def _nums(out: str) -> set:
    import re

    n = set()
    for m in re.finditer(r"(?m)^\s*(?:>\s*)?(?:Line\s+)?(\d+):", out):
        n.add(int(m.group(1)))
    return n


def test_any_language_symbol_lands_first_try(tmp_path):
    from opencode_py.tools import context_ledger as cl

    for fname, (src, wants) in LANG_CASES.items():
        p = tmp_path / fname
        p.write_text(src, encoding="utf-8")
        for name, s, e in wants:
            cl.clear()
            out = read_tool().run({"filePath": str(p), "symbol": name})["output"]
            assert set(range(s, e + 1)) <= _nums(out), f"{fname} {name}: {out[:200]}"


def test_spans_unit_brace_string_comment_traps():
    from opencode_py.index.spans import block_span

    assert block_span(['function f() {', '  const s = "}{"; // }', "  return s;", "}"], 1, "js") == (1, 4)
    assert block_span(["def f():", "    x = 1", "    return x"], 1, "python") == (1, 3)
    assert block_span(["def hi", "  puts 1", "end"], 1, "ruby") == (1, 3)
    assert block_span(["int proto(int a);", "int x;"], 1, "c") == (1, 1)


def test_heuristic_indexer_stores_real_end_lines(tmp_path):
    from opencode_py.index.heuristic_indexer import index_file
    from opencode_py.index.model import language_for

    p = tmp_path / "a.js"
    src = "export function greet(n) {\n  return n;\n}\n"
    p.write_text(src, encoding="utf-8")
    fi = index_file(tmp_path, "a.js", src, len(src), 0.0, language_for("a.js"))
    got = [(s.line, s.end_line) for s in fi.symbols if s.name == "greet"]
    assert got == [(1, 3)], got


def test_shared_agent_md_prepended_to_every_agent(tmp_path, monkeypatch):
    from opencode_py import globals as G
    from opencode_py import permission as P
    from opencode_py.config import Config

    monkeypatch.setattr(G.Path, "config", tmp_path, raising=False)
    (tmp_path / "agents").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agents" / "AGENT.md").write_text("SHARED-RULE", encoding="utf-8")
    (tmp_path / "agents" / "build").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agents" / "build" / "workflow.md").write_text("BUILD-ONLY", encoding="utf-8")
    cfg = Config()
    cfg.agents = {}
    groups = P.agent_md_groups(cfg)
    assert groups[0][0] == "shared" and [f for f, _ in groups[0][1]] == ["AGENT.md"]
    combined = P.agent_combined_md(cfg, "build")
    assert combined.index("SHARED-RULE") < combined.index("BUILD-ONLY")
    # future custom agent with no files still gets shared
    cfg.agents = {"newbot": {"description": "x"}}
    assert "SHARED-RULE" in P.agent_combined_md(cfg, "newbot")


def test_shared_agent_md_empty_is_harmless(tmp_path, monkeypatch):
    from opencode_py import globals as G
    from opencode_py import permission as P
    from opencode_py.config import Config

    monkeypatch.setattr(G.Path, "config", tmp_path, raising=False)
    cfg = Config()
    cfg.agents = {}
    assert P.agent_combined_md(cfg, "build") == ""
    assert P.agent_md_groups(cfg)[0] == ("shared", [])


def test_only_active_agent_is_sent_never_inactive_ones(tmp_path, monkeypatch):
    """The prompt for agent X holds shared + X's files only.

    build/plan/explore/custom markers are all distinct — any leak of an
    inactive agent's content fails the test."""
    from pathlib import Path as _P

    from opencode_py import globals as G
    from opencode_py.agent import system as S
    from opencode_py.config import Config

    monkeypatch.setattr(G.Path, "config", tmp_path, raising=False)
    base = tmp_path / "agents"
    base.mkdir(parents=True, exist_ok=True)
    (base / "AGENT.md").write_text("MARK-SHARED")
    files = {
        "build": ("workflow.md", "MARK-BUILD"),
        "plan": ("rules.md", "MARK-PLAN"),
        "explore": ("rules.md", "MARK-EXPLORE"),
        "mybot": ("notes.md", "MARK-MINE"),
    }
    for agent, (fname, mark) in files.items():
        d = base / agent
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(mark)
    cfg = Config()
    cfg.agents = {"mybot": {"description": "x"}}
    for agent, (_fname, own) in files.items():
        sp = S.build_system_prompt(
            directory=_P("."), worktree=_P("."), provider_id="p", model_id="m", cfg=cfg, agent=agent,
        )
        assert "MARK-SHARED" in sp, agent
        assert own in sp, agent
        for other_agent, (_f2, other_mark) in files.items():
            if other_agent != agent:
                assert other_mark not in sp, f"{agent} prompt leaked {other_agent}'s rules"


# ---------------------------------------------------------------------------
# agent editor: description edit + readonly toggle + tool filter
# ---------------------------------------------------------------------------


def test_agent_name_dialog_edit_mode_prefills_and_locks():
    import asyncio

    from opencode_py.tui.agent_picker import AgentNameDialog

    d = AgentNameDialog(
        old="mybot", title="Edit description", hide_description=False,
        description="old desc", lock_name=True,
    )
    assert d._desc_value == "old desc" and d._lock_name is True
    d2 = AgentNameDialog()
    assert d2._desc_value == "" and d2._lock_name is False


def test_perm_editor_readonly_row_scope_marks_and_filter():
    from opencode_py.tui.agent_permission_editor import AgentPermissionEditor

    ed = AgentPermissionEditor(
        agent="x", tools=[("bash", "allow"), ("read", "ask")],
        readonly=False, readonly_locked=False, scope={"bash"},
    )
    opts = ed._options()
    assert opts[0].id == "__readonly__"
    assert "×" not in str(opts[1].prompt) and "×" not in str(opts[2].prompt)
    ed._filter = "bas"
    assert ed._visible() == [("bash", "allow")]
    ed.reload_state(readonly=True, scope={"read"})
    assert ed.readonly is True and ed.scope == {"read"}
    opts = ed._options()
    assert "ON" in str(opts[0].prompt)


def test_agent_edit_description_and_readonly_persist():
    from opencode_py.config import Config
    from opencode_py.tui import app as appmod

    cfg = Config()
    cfg.agents = {"mybot": {"description": "old", "tools": {}}}
    app = appmod.OpenCodeTUI.__new__(appmod.OpenCodeTUI)
    app.cfg = cfg
    app.notify = lambda *a, **k: None
    app._refresh_agent_picker = lambda: None
    app._persist_agents = lambda: None

    class Eng:
        agent = "build"

    app._active_engine = lambda: Eng()
    app._set_agent = lambda *a, **k: None
    app._agent_edit_description("mybot", "new desc")
    assert cfg.agents["mybot"]["description"] == "new desc"
    app._agent_edit_description("build", "x")  # builtin: protected
    assert "description" not in (cfg.agents.get("build") or {})
    app._agent_toggle_readonly("mybot")
    assert cfg.agents["mybot"]["readonly"] is True
    app._agent_toggle_readonly("mybot")
    assert cfg.agents["mybot"]["readonly"] is False
    ro, locked, _scope = app._agent_perm_state("plan")
    assert ro is True and locked is True


# ---------------------------------------------------------------------------
# turn speed: trim receipts + stable prefix + early summary + vacuum
# ---------------------------------------------------------------------------


def test_trim_shrinks_old_tools_keeps_last_turns_full():
    from opencode_py.agent import trim as T

    def hist(n):
        msgs = []
        for i in range(n):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}",
                         "tool_calls": [{"id": f"c{i}", "function": {"name": "read", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read", "content": "Z" * 8000})
        return msgs

    msgs = hist(5)
    out = T.shrink_tool_history(msgs, keep_turns=2, max_chars=500)
    assert len(out) == len(msgs)
    # last 2 turns verbatim (6 msgs), older tool bodies are receipts
    assert out[-1]["content"] == "Z" * 8000
    assert out[2]["content"].startswith("[trimmed for sending:")
    assert "read" in out[2]["content"] and "8000 chars" in out[2]["content"]
    # user + assistant text untouched
    assert out[0]["content"] == "q0" and out[1]["content"] == "a0"
    # tool_call declarations untouched (pairing intact)
    assert out[1]["tool_calls"][0]["id"] == "c0"
    # idempotent: second pass changes nothing
    again = T.shrink_tool_history(out, keep_turns=2, max_chars=500)
    assert [m.get("content") for m in again] == [m.get("content") for m in out]


def test_trim_never_touches_multimodal_or_compaction():
    from opencode_py.agent import trim as T

    msgs = [
        {"role": "user", "content": "old q"},
        {"role": "tool", "tool_call_id": "c", "name": "read",
         "content": [{"type": "text", "text": "big " * 500}]},
        {"role": "user", "content": "[Summary of earlier conversation]\nnotes", "compaction": True},
        {"role": "user", "content": "new q"},
        {"role": "tool", "tool_call_id": "d", "name": "read", "content": "short"},
    ]
    out = T.shrink_tool_history(msgs, keep_turns=1, max_chars=100)
    assert out[1]["content"] == msgs[1]["content"]  # multimodal intact
    assert out[2]["content"] == msgs[2]["content"]  # compaction intact
    assert out[4]["content"] == "short"  # last turn intact


def test_request_prefix_key_stable_across_schema_order():
    from opencode_py.agent import trim as T

    tools_a = [{"function": {"name": n}} for n in ["read", "grep", "bash"]]
    tools_b = [{"function": {"name": n}} for n in ["bash", "read", "grep"]]
    a = T.request_prefix_key("SYS", sorted(tools_a, key=lambda s: s["function"]["name"]))
    b = T.request_prefix_key("SYS", sorted(tools_b, key=lambda s: s["function"]["name"]))
    assert a == b and len(a) == 16
    assert T.request_prefix_key("SYS2", tools_a) != a


def test_vacuum_dry_run_and_stats(tmp_path, monkeypatch):
    from opencode_py import session as S

    monkeypatch.setattr(S.GPath, "sessions_dir", lambda: tmp_path)
    (tmp_path / "a.json").write_text('{"id":"a"}')
    stats = S.session_disk_stats()
    assert stats["files"] >= 0 and "bytes" in stats
    rep = S.vacuum_sessions(0)  # dry run: report only, delete nothing
    assert rep["pruned"] == 0
    assert (tmp_path / "a.json").exists()


def test_early_summary_adopt_and_stale_reject():
    from pathlib import Path as _P

    from opencode_py.agent import compaction as C
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.config import Config
    from opencode_py.tools import build_registry

    cfg = Config()
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=_P("."), agent="build")
    for i in range(4):
        loop._history.append({"role": "user", "content": f"q{i}", "id": f"u{i}"})
        loop._history.append({"role": "assistant", "content": f"a{i}"})
        loop._history.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read", "content": "Z" * 3000})
    head, _tail = C.select_tail(
        [m for m in loop._history if not m.get("compaction")],
        tail_turns=2, context=200000, output_limit=32000,
    )

    def _mid(m):
        return str(m.get("id") or "") + "|" + str(m.get("role") or "") + "|" + str(m.get("content", ""))[:120]

    loop._early_summary_text = "EARLY SUMMARY TEXT"
    loop._early_summary_at = len(loop._history)
    loop._early_summary_head = {"len": len(head), "first": _mid(head[0]), "last": _mid(head[-1])}
    out = loop._adopt_early_summary("SYS")
    assert out is not None and out[0]["role"] == "system"
    assert "EARLY SUMMARY TEXT" in loop._history[0]["content"]
    # stale snapshot never adopted
    loop._early_summary_text = "STALE"
    loop._early_summary_head = {"len": 999, "first": "x", "last": "y"}
    assert loop._adopt_early_summary("SYS") is None


def test_vacuum_prunes_old_keeps_recent_and_children(tmp_path, monkeypatch):
    import json
    import time

    from opencode_py import session as S
    from opencode_py.globals import Path as G

    monkeypatch.setattr(G, "sessions_dir", staticmethod(lambda: tmp_path))
    S._session_cache.clear()
    for i in range(5):
        (tmp_path / f"old{i}.json").write_text(json.dumps(
            {"id": f"old{i}", "title": f"t{i}", "created": 1000 + i,
             "messages": [{"role": "user", "content": "x"}]}))
    (tmp_path / "new.json").write_text(json.dumps(
        {"id": "new", "title": "new", "created": time.time(),
         "messages": [{"role": "user", "content": "x"}]}))
    S._session_cache.clear()
    rep = S.vacuum_sessions(3)
    assert rep["pruned"] >= 2 and rep["kept"] <= 4
    assert (tmp_path / "new.json").exists()
    S._session_cache.clear()


def test_models_search_rank_best_match_first():
    from opencode_py.tui.model_picker import _match_rank

    assert _match_rank("muse-spark-1.3", "Muse Spark", "muse-spark-1.3", "Zen")[0] == 0
    assert _match_rank("muse", "Muse Spark", "muse-spark-1.3", "Zen")[0] == 1
    assert _match_rank("spark", "Muse Spark", "muse-spark-1.3", "Zen")[0] == 2
    assert _match_rank("msprk", "Muse Spark", "muse-spark-1.3", "Zen")[0] == 3
    assert _match_rank("zzz", "Muse Spark", "abc", "Zen")[0] == 9


def test_models_row_always_shows_provider_ctx_free():
    from opencode_py.tui.model_picker import _model_row_label

    lbl = _model_row_label("opencode/x", {"id": "x", "name": "X", "free": True, "context": 1000000},
                           "", provider_name="Zen", favorite=False)
    s = str(lbl.content if hasattr(lbl, "content") else lbl._content)
    assert "Zen" in s and "FREE" in s and "1,000,000" in s


def test_models_snapshot_roundtrip_and_stale_renders(tmp_path, monkeypatch):
    import json
    import time

    from opencode_py.tui import model_picker as M

    monkeypatch.setattr(M, "_snapshot_path", lambda: tmp_path / "model-list.json")
    assert M._load_snapshot() is None
    M._save_snapshot({"opencode": [{"id": "m1", "name": "M1", "context": 1000, "free": True}]})
    snap = M._load_snapshot()
    assert snap["providers"]["opencode"][0]["id"] == "m1"
    old = json.loads((tmp_path / "model-list.json").read_text())
    old["ts"] = time.time() - 99 * 3600
    (tmp_path / "model-list.json").write_text(json.dumps(old))
    assert M._load_snapshot() is not None  # stale still renders instantly


def test_models_snapshot_skips_junk(tmp_path, monkeypatch):
    from opencode_py.tui import model_picker as M

    monkeypatch.setattr(M, "_snapshot_path", lambda: tmp_path / "model-list.json")
    M._save_snapshot({})
    assert M._load_snapshot() is None
    M._save_snapshot({"x": [{"name": "no-id"}]})
    assert M._load_snapshot() is None


def test_prompt_preview_equals_sender_byte_for_byte():
    """labeled_prompt_parts joined == build_system_prompt (not one letter missed)."""
    from pathlib import Path as _P

    from opencode_py.agent import system as S
    from opencode_py.config import Config

    cfg = Config()
    for agent in ["build", "plan", "explore"]:
        sent = S.build_system_prompt(
            directory=_P("."), worktree=_P("."), provider_id="p", model_id="m", cfg=cfg, agent=agent)
        blocks = S.labeled_prompt_parts(
            directory=_P("."), worktree=_P("."), provider_id="p", model_id="m", cfg=cfg, agent=agent)
        assert sent == "\n\n".join(t for _, t in blocks)
        labels = [label for label, _ in blocks]
        assert labels[0] == "base.md" and labels[1] == "environment"
        assert any("Agent instructions" in label for label in labels)


def test_prompt_preview_counts_and_copy():
    from opencode_py.tui.agent_md_popup import AgentPromptPreview

    blocks = [("base.md", "hello world"), ("Agent instructions (build.md)", "do the thing")]
    pop = AgentPromptPreview(agent="build", blocks=blocks)
    total, toks = pop._counts()
    assert total == len("hello world") + len("do the thing") and toks == total // 4
    assert pop._copy_text() == "hello world\n\ndo the thing"


def test_grep_rank_def_before_comment(tmp_path):
    from opencode_py.tools import grep as G

    p = tmp_path / "m.py"
    p.write_text(
        "# zebra is mentioned here first\n"
        "# another zebra comment\n"
        "x = 'zebra string'\n"
        "\n"
        "def zebra(arg):\n"
        "    return arg\n",
        encoding="utf-8",
    )
    out = G.tool().run({"pattern": "zebra", "path": str(p), "context": 0})["output"]
    assert out.index("def zebra") < out.index("# zebra")


def test_grep_rank_exact_def_first(tmp_path):
    from opencode_py.tools import grep as G

    p = tmp_path / "s.py"
    p.write_text(
        "def session_cache():\n    pass\n\n\n"
        "def session_path(x):\n    return x\n",
        encoding="utf-8",
    )
    out = G.tool().run({"pattern": "session", "path": str(p), "context": 0})["output"]
    assert out.index("def session_path") < out.index("def session_cache")
    ctx = G.tool().run({"pattern": "session", "path": str(p), "context": 2})["output"]
    assert "def session_path" in ctx and "def session_cache" in ctx  # one merged group


def test_grep_rank_tiers_unit():
    from opencode_py.tools.grep import _hit_kind

    assert _hit_kind("def zebra(x):", "zebra")[0] == 0
    assert _hit_kind("def zebra_cache():", "zebra")[0] == 1
    assert _hit_kind("def other():", "zebra")[0] == 2
    assert _hit_kind("    return zebra(x)", "zebra")[0] == 3
    assert _hit_kind("    # zebra comment", "zebra")[0] == 5
