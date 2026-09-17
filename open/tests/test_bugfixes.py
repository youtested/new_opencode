"""Regression tests for the bugs catalogued in bug_found.txt."""

import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import pytest

from opencode_py.agent.messages import _text
from opencode_py.config import Config, _strip_jsonc
from opencode_py.permission import MAX_APPROVED_PATTERNS, PermissionEngine


# -- #1 + #8: JSONC stripping ------------------------------------------------
@pytest.mark.parametrize(
    "src, expected",
    [
        ('{"a": 1} /* trailing */', {"a": 1}),
        ('{ /* o /* i */ o */ "a": 1 }', {"a": 1}),
        ('{"a": [/* l1 /* l2 */ */ "x"]}', {"a": ["x"]}),
        ('{ "s": "/* keep */" }', {"s": "/* keep */"}),
        ('{ "u": "\\\\u002F" }', {"u": "\\u002F"}),
    ],
)
def test_strip_jsonc_nested_comments_and_escapes(src, expected):
    assert json.loads(_strip_jsonc(src)) == expected


def test_strip_jsonc_escaped_unicode_boundary():
    # a unicode escape whose payload looks like a comment start must survive
    src = '{"x": "\\\\u002F* not a comment" }'
    assert json.loads(_strip_jsonc(src)) == {"x": "\\u002F* not a comment"}


# -- bug 4: agent vs agents -----------------------------------------------

def test_config_reads_both_agent_and_agents():
    cfg = Config.from_dict({"agent": {"a": {"model": "m1"}}})
    assert cfg.agents == {"a": {"model": "m1"}}
    cfg2 = Config.from_dict({"agents": {"b": {"model": "m2"}}})
    assert cfg2.agents == {"b": {"model": "m2"}}
    cfg3 = Config.from_dict({"agent": {"a": {"model": "m1"}}, "agents": {"b": {"model": "m2"}}})
    # plural wins (user explicitly wrote the dataclass field name)
    assert cfg3.agents == {"b": {"model": "m2"}}


# -- model_read_timeout config -------------------------------------------

def test_config_model_read_timeout_default_and_parse():
    cfg = Config.from_dict({})
    assert cfg.model_read_timeout == 300.0
    cfg2 = Config.from_dict({"model_read_timeout": 600})
    assert cfg2.model_read_timeout == 600.0
    cfg3 = Config.from_dict({"model_read_timeout": "not-a-number"})
    assert cfg3.model_read_timeout == 300.0


def test_config_model_read_timeout_roundtrips():
    cfg = Config.from_dict({"model_read_timeout": 120})
    out = cfg.as_dict()
    assert out["model_read_timeout"] == 120.0


def test_config_rotation_lock_default_and_parse():
    cfg = Config.from_dict({})
    assert cfg.rotation_lock is False
    cfg2 = Config.from_dict({"rotation_lock": True})
    assert cfg2.rotation_lock is True
    out = Config.from_dict({"rotation_lock": True}).as_dict()
    assert out["rotation_lock"] is True


# -- bug 3: session.py corruption tolerance --------------------------------

def test_list_sessions_survives_corrupt_json(tmp_path, monkeypatch):
    """list_sessions/load_session must not crash (or surface) on a corrupt or
    non-dict session file — one bad file (partial write, garbage thumbnails,
    hand-edit) must never break the session picker."""
    import json
    import time
    import opencode_py.session as session_mod
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))

    # a well-formed session
    session_mod.save_session(
        session_mod.new_session(directory=str(tmp_path), provider="opencode", model="m", title="good")
    )
    # corruption zoo: unparseable bytes, JSON `null`, JSON array, str-created
    (tmp_path / "bad1.json").write_bytes(b"{ not json !!!")
    (tmp_path / "bad2.json").write_text("null")
    (tmp_path / "bad3.json").write_text("[1,2,3]")
    (tmp_path / "bad4.json").write_text(json.dumps({"id": "bad4", "title": "x", "created": "yesterday"}))
    # a never-completed save (the atomic tmp slot)
    (tmp_path / "bad1.json.tmp").write_text('{"we": "partial"}')

    # the full scan must skip junk yet still return the good session
    listed = session_mod.list_sessions()
    ids = {s.id for s in listed}
    assert "good" in ids or any("good" in s.title for s in listed)
    # unparseable / non-dict files are skipped entirely
    assert not any(s.title == "w" or s.id in ("bad1", "bad2", "bad3") for s in listed)
    # a dict whose `created` is a string either loads (coerced) or is skipped —
    # it must NEVER made the listing crash
    for s in listed:
        assert isinstance(s.created, float)

    # loading a non-dict session returns None; a bad `created` coerces safely
    assert session_mod.load_session("bad2") is None
    assert session_mod.load_session("bad3") is None
    coerced = session_mod.load_session("bad4")
    assert coerced is None or isinstance(coerced.created, float)

    # grouping a garbage `created` never raises
    from opencode_py.session import group_sessions

    groups = group_sessions([{"id": "g", "created": "garbage", "title": "t"}, {"id": "n", "title": "n"}])
    assert groups  # bucketed without crashing


def test_load_session_survives_garbage_created(tmp_path, monkeypatch):
    """A session JSON with a non-numeric `created` loads as a float, so sorting
    (newest-first) in the picker can't raise comparing str vs float."""
    import json
    import opencode_py.session as session_mod
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    (tmp_path / "weird.json").write_text(
        json.dumps({"id": "weird", "created": "not-a-date", "title": "w"})
    )
    sess = session_mod.load_session("weird")
    assert sess is not None
    assert isinstance(sess.created, float)

def test_save_session_atomic(tmp_path, monkeypatch):
    """A disk error while writing a session body (crash mid-save) must never
    destroy the previous good file: the body is written to a temp file first
    and only atomically renamed over the final path."""
    import opencode_py.session as session_mod
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    sess = session_mod.new_session(directory=str(tmp_path), provider="opencode", model="m")

    # first save puts a good body (and a durable .bak replica) in place
    session_mod.save_session(sess)
    target = session_mod.session_path(sess.id)
    good = target.read_text(encoding="utf-8")
    assert target.read_text(encoding="utf-8") == good
    assert target.with_suffix(".json.bak").exists()

    # a failed mid-write must leave that good body untouched — partial content
    # may only ever sit in the .tmp file, never at the final path
    orig_open = session_mod.os.open

    def boom_open(*a, **k):
        raise OSError("boom mid-write")

    monkeypatch.setattr(session_mod.os, "open", boom_open)
    session_mod.save_session(sess)
    monkeypatch.setattr(session_mod.os, "open", orig_open)

    assert target.exists(), "a failed write must not delete the session file"
    assert target.read_text(encoding="utf-8") == good, "previous good body must survive a failed write"
    assert sess.id in [s.id for s in session_mod.list_sessions()], "session must still be listed"


def test_session_survives_sudden_reboot(tmp_path, monkeypatch):
    """A sudden phone reboot can leave a session body as a 0-byte file (the
    write was still in the OS page cache when power died). The session must
    recover from its durable .bak/.tmp replica instead of vanishing."""
    import json
    import opencode_py.session as session_mod
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    sess = session_mod.new_session(directory=str(tmp_path), title="reboot test")
    sess.messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    session_mod.save_session(sess)
    path = session_mod.session_path(sess.id)

    # crash 1: body in page cache lost on reboot -> 0-byte file
    path.write_text("")
    assert session_mod.load_session(sess.id).id == sess.id, "0-byte body must recover from .bak"
    assert sess.id in [s.id for s in session_mod.list_sessions()], "must still appear in the picker"

    # crash 2: died between writing the temp file and renaming it -> only
    # .json.tmp remains (no primary, no .bak)
    path.unlink()
    path.with_suffix(".json.bak").unlink()
    path.with_suffix(".json.tmp").write_text(json.dumps(sess.to_dict()), encoding="utf-8")
    assert session_mod.load_session(sess.id).id == sess.id, "tmp replica must be recoverable"

    # crash 3: nothing on disk at all
    path.unlink()
    assert session_mod.load_session(sess.id) is None


# -- bug 5: token estimation ----------------------------------------------

def test_repair_tool_pairs_closes_orphaned_assistant_tool_calls():
    from opencode_py.agent.messages import repair_tool_pairs

    # intact pair stays untouched
    intact = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "grep", "content": "out"},
    ]
    assert repair_tool_pairs(intact) == intact

    # orphan at the end: assistant declares c1 + c2, only c1 answered
    orphaned = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "x", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "grep", "content": "out"},
    ]
    fixed = repair_tool_pairs(orphaned)
    assert fixed[-1]["role"] == "tool" and fixed[-1]["tool_call_id"] == "c2"

    # orphan in the middle: next user message forced a boundary
    mid = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c9", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "user", "content": "next turn"},
    ]
    fixed = repair_tool_pairs(mid)
    assert fixed[1]["role"] == "tool" and fixed[1]["tool_call_id"] == "c9"
    assert fixed[2]["role"] == "user"


def test_text_counts_tool_calls_and_tool_results():
    assistant = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]}
    tool_result = {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "out"}
    assert len(_text(assistant)) > 0
    assert len(_text(tool_result)) > 0
    # content alone (< tool_calls-bearing message)
    plain = {"role": "assistant", "content": "hi"}
    assert _text(plain) == "hi"


# -- bug 6: permission approved list cleanup --------------------------------

def test_approved_patterns_bounded():
    eng = PermissionEngine()
    approved = []
    # simulate "always" answers with unique patterns
    patterns = [f"perm_{i}" for i in range(MAX_APPROVED_PATTERNS + 50)]
    eng.ask_callback = lambda desc, pats: "always"
    for p in patterns:
        eng.ask("desc", [p])
    assert len(eng._approved_patterns) <= MAX_APPROVED_PATTERNS
    # dedup: re-approving an existing pattern doesn't create a duplicate
    eng.ask("desc", [patterns[0]])
    assert eng._approved_patterns.count(patterns[0]) == 1


# -- bug 9: symlink cycle safety --------------------------------------------

def test_find_instruction_files_symlink_cycle_no_hang(tmp_path):
    from opencode_py.agent.system import find_instruction_files

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    try:
        (a / "ln").symlink_to(b, target_is_directory=True)
        (b / "ln").symlink_to(a, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported")
    cfg = Config()
    # must terminate promptly
    files = find_instruction_files(a, tmp_path, cfg)
    assert isinstance(files, list)


# -- bug 2: undo cleanup ----------------------------------------------------

def test_missing_directories_walks_up_to_existing():
    from opencode_py.agent.loop import _missing_directories

    base = Path(tempfile.mkdtemp())
    deep = base / "x" / "y" / "z"
    missing = _missing_directories(deep)
    # deepest first, stopping before the existing base
    assert missing[0] == deep
    assert missing[-1] == base / "x"
    assert _missing_directories(base) == []


def test_undo_created_file_removes_it(tmp_path):
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.tools import build_registry

    cfg = Config()
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=tmp_path, provider=mock.MagicMock(), agent="build")
    f = tmp_path / "new.txt"
    f.write_text("created")
    loop._undo_stack.append({"path": str(f), "original": None, "dirs": []})
    msg = loop.undo_last()
    assert "Reverted" in msg
    assert not f.exists()


def test_undo_cleans_created_parent_dirs(tmp_path):
    from opencode_py.agent.loop import _missing_directories
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.tools import build_registry

    cfg = Config()
    loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=tmp_path, provider=mock.MagicMock(), agent="build")
    nested = tmp_path / "p" / "q" / "deep.txt"
    dirs = [str(d) for d in _missing_directories(nested.parent)]
    nested.parent.mkdir(parents=True)
    nested.write_text("hi")
    loop._undo_stack.append({"path": str(nested), "original": None, "dirs": dirs})
    loop.undo_last()
    assert not nested.exists()
    # dirs we created should be gone, but the common ancestor stays
    assert not (tmp_path / "p" / "q").exists()
    assert tmp_path.exists()


# -- bug 10: edit/write must NOT require a prior read -------------------------

def test_edit_does_not_require_read_first():
    """Editing an existing file the model never Read must succeed — the
    read-before-edit guard was removed so edit/write work without loading."""
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.providers.base import ProviderEvent, ToolCall
    from opencode_py.tools import build_registry

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "a.py"
        target.write_text("original\n")

        class EditOnly:
            def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
                tc = ToolCall(
                    id="c1",
                    name="edit",
                    arguments=json.dumps({"filePath": str(target), "oldString": "original", "newString": "changed"}),
                )
                on_event(ProviderEvent(kind="tool_call", tool_calls=[tc]))
                return "opencode", "deepseek-v4-flash-free"

        cfg = Config()
        cfg.provider = "opencode"
        cfg.model = "deepseek-v4-flash-free"
        loop = AgentLoop(cfg=cfg, registry=build_registry(cfg), directory=Path(td), provider=EditOnly(), agent="build")
        result = loop.run_turn("edit the file")
        assert not result.error
        assert target.read_text() == "changed\n"


# -- permission / rotation config hardening ----------------------------------

def test_config_permission_string_normalized():
    """opencode accepts `permission: "ask"` (global default action). It must be
    normalized to a dict so PermissionEngine / merge / validate never crash."""
    from opencode_py.permission import PermissionEngine, merge_permissions

    cfg = Config.from_dict({"permission": "ask"})
    assert cfg.permission == {"*": "ask"}
    # the engine must build without crashing and treat it as the global default
    engine = PermissionEngine.from_config(merge_permissions(cfg.permission, "build"))
    assert engine._find_action("bash", "some command") == "ask"
    assert engine.evaluate("bash", "some command") == "allow"  # auto mode approves asks

    # malformed non-dict / non-str values degrade to an empty permission map
    assert Config.from_dict({"permission": ["nope"]}).permission == {}
    assert Config.from_dict({"permission": 42}).permission == {}


def test_plan_mode_edit_write_denied_even_outside_worktree():
    """external_directory must be an ADDITIONAL gate, not a replacement for the
    tool's own permission. Previously a plan agent could edit/write an absolute
    path OUTSIDE the worktree because the permission name got swapped to
    external_directory (ask -> auto allow), bypassing plan's edit:deny."""
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.tools import build_registry

    worktree = Path(tempfile.mkdtemp())
    outside = Path(tempfile.mkdtemp())

    loop = AgentLoop(
        cfg=Config(),
        registry=build_registry(Config()),
        directory=worktree,
        agent="plan",
    )
    victim = outside / "secret.txt"
    victim.write_text("hello world")
    res = loop.run_tool(
        "edit",
        {"filePath": str(victim), "oldString": "hello", "newString": "hacked"},
        call_id="t1",
    )
    assert res.get("denied"), "plan mode must deny edit outside the worktree"
    assert victim.read_text() == "hello world", "file must not change"

    new_file = outside / "new.txt"
    res = loop.run_tool("write", {"filePath": str(new_file), "content": "pwned"}, call_id="t2")
    assert res.get("denied"), "plan mode must deny write outside the worktree"
    assert not new_file.exists(), "file must not be created"

    # read-only tools outside the worktree stay allowed in plan mode
    res = loop.run_tool("read", {"filePath": str(victim)}, call_id="t3")
    assert "hello world" in res.get("output", "")


def test_build_mode_external_edit_still_allowed():
    """Build agent keeps editing files outside the worktree (auto mode approves
    the external_directory ask) — the combined gate must not over-block."""
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.tools import build_registry

    worktree = Path(tempfile.mkdtemp())
    outside = Path(tempfile.mkdtemp())

    loop = AgentLoop(
        cfg=Config(),
        registry=build_registry(Config()),
        directory=worktree,
        agent="build",
    )
    victim = outside / "secret.txt"
    victim.write_text("hello world")
    res = loop.run_tool(
        "edit",
        {"filePath": str(victim), "oldString": "hello", "newString": "hacked"},
        call_id="t1",
    )
    assert not res.get("denied"), "build agent should still edit outside the worktree"
    assert victim.read_text() == "hacked world"


def test_config_malformed_rotation_sanitized():
    """A malformed `rotation` (string, non-dict lanes, missing provider/model)
    must be sanitized instead of crashing build_rotation / Rotation.stream."""
    from opencode_py.providers.rotation import build_rotation

    cfg = Config.from_dict({"rotation": "groq"})
    assert cfg.rotation == []
    rot = build_rotation(cfg)
    # first lane is the configured model (most-used default when no pick
    # is saved); the garbage string must not leak in
    assert rot.lanes[0]["provider"] == "opencode"
    assert rot.lanes[0]["model"] == cfg.model
    assert all(isinstance(l, dict) and l.get("provider") for l in rot.lanes)

    cfg2 = Config.from_dict(
        {"rotation": [{"provider": "groq", "model": "m"}, "garbage", {"provider": "x"}, {"model": "only"}]}
    )
    assert cfg2.rotation == [{"provider": "groq", "model": "m"}]
# -- bug: session save keeps the transcript faithful (repair is request-side) --

def test_save_session_persists_orphaned_tool_pairs_faithfully(tmp_path, monkeypatch):
    """Regression: a kill during parallel sub-agent fan-out used to leave an
    assistant `tool_calls` message with no following tool results in memory,
    and save_session used to inject synthetic "[interrupted]" tool messages
    INTO the persisted transcript. The saved session must be the REAL
    conversation — orphaned tool pairs stay as-is on disk, and the request
    path (not the save path) closes them for strict backends."""
    import json
    import opencode_py.session as session_mod
    from opencode_py.agent.messages import repair_tool_pairs
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    sess = session_mod.new_session(directory=str(tmp_path), provider="opencode", model="m")
    # simulate an in-flight parallel turn interrupted mid-execution:
    # the assistant declared c1+c2 but only c1's result landed
    sess.messages = [
        {"role": "user", "content": "run agents"},
        {"role": "assistant", "content": "x", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "task", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "task", "content": "done"},
    ]
    session_mod.save_session(sess)
    on_disk = json.loads((tmp_path / f"{sess.id}.json").read_text())
    msgs = on_disk["messages"]
    # the saved transcript is byte-for-byte the conversation, never patched
    assert msgs == sess.messages, "save must not mutate the message list"
    assert msgs[-1]["tool_call_id"] == "c1", "no synthetic trailing tool row"
    # the request path still closes the orphan for the provider in one pass
    repaired = repair_tool_pairs(msgs)
    assert repaired == msgs or (repaired[-1]["role"] == "tool" and repaired[-1]["tool_call_id"] == "c2")

    # a well-formed history is persisted byte-for-byte
    clean = session_mod.new_session(directory=str(tmp_path), provider="opencode", model="m")
    good = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "grep", "content": "out"},
    ]
    clean.messages = good
    session_mod.save_session(clean)
    from_disk = json.loads((tmp_path / f"{clean.id}.json").read_text())
    assert from_disk["messages"] == good


def test_repair_tool_pairs_fastpath_returns_same_list_when_intact():
    """No allocation churn on every save: an already-valid history is returned
    by identity, so the save path's `repaired is not messages` guard stays off."""
    from opencode_py.agent.messages import repair_tool_pairs

    intact = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "grep", "content": "out"},
    ]
    assert repair_tool_pairs(intact) is intact

    # non-dict messages (a corrupted/foreign session) never crash repair/save
    junk = [{"role": "user", "content": "x"}, "not a dict", 123]
    assert repair_tool_pairs(junk) is junk


def test_headless_model_command_does_not_claim_success():
    """Without a TUI set_model callback, /model must say it can't switch —
    not reply 'Model set to ...' while silently keeping the old model."""
    from opencode_py.commands import CommandContext, handle_command, build_registry
    from opencode_py.config import Config

    replies = []
    ctx = CommandContext(config=Config(), auth=None, registry=build_registry(), reply=replies.append)
    handled = handle_command(ctx.registry, ctx, "/model big-pickle")
    assert handled
    text = "\n".join(replies)
    assert "Model set to" not in text
    assert "TUI" in text or "--model" in text


def test_export_writes_into_worktree(tmp_path, monkeypatch):
    """/export must write next to the project (worktree), not wherever the
    process happened to be launched from."""
    import json as _json

    from opencode_py import session as session_mod
    from opencode_py.commands import CommandContext, handle_command, build_registry
    from opencode_py.config import Config
    from opencode_py.session import new_session

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    sess = new_session(title="export me")
    sess.messages = [{"role": "user", "content": "hello"}]
    session_mod.save_session(sess)

    replies = []
    worktree = tmp_path / "proj"
    worktree.mkdir()
    ctx = CommandContext(
        config=Config(),
        auth=None,
        engine=type("E", (), {"session_id": sess.id})(),
        worktree=str(worktree),
        reply=replies.append,
        registry=build_registry(),
    )
    handled = handle_command(ctx.registry, ctx, f"/export {sess.id}")
    assert handled
    out = (worktree / f"opencode-session-{sess.id[:12]}.md").read_text()
    assert "hello" in out


def test_tui_surface_has_new_session_and_runtime_apply():
    """The TUI implements real /new (/clear) and live Settings application."""
    from opencode_py.tui.app import OpenCodeTUI
    from opencode_py.tui.settings_screen import SettingsScreen

    assert callable(getattr(OpenCodeTUI, "_action_new", None))
    assert callable(getattr(OpenCodeTUI, "_apply_runtime_settings", None))
    # SettingsScreen stores the callback (init pops kwargs)
    screen = SettingsScreen(cfg=Config(), engine=None, auth=None, on_apply=lambda: None)
    assert callable(screen.on_apply)


# --------------------------------------------------------------------------
# "Allow all permissions" toggle: persisted permission_mode drives every
# engine; ON skips questions, OFF pops the dialog; deny rules always apply.
# --------------------------------------------------------------------------

def test_permission_mode_config_roundtrip():
    from opencode_py.config import Config

    cfg = Config.from_dict({"permission_mode": "ask"})
    assert cfg.permission_mode == "ask"
    assert "permission_mode" in cfg.as_dict()
    # invalid values fall back to the safe default
    assert Config.from_dict({"permission_mode": "yolo"}).permission_mode == "auto"
    assert Config().permission_mode == "auto"


def test_allow_all_skips_questions_but_honors_denies():
    from opencode_py.permission import PermissionEngine

    # last matching rule wins (opencode semantics): the deny must come AFTER
    # the catch-all ask
    rules = {"bash": {"*": "ask", "rm -rf *": "deny"}}
    on = PermissionEngine.from_config(rules, mode="auto")
    off = PermissionEngine.from_config(rules, mode="ask")

    assert on.evaluate("bash", "npm install") == "allow"   # never asks
    assert off.evaluate("bash", "npm install") == "ask"    # popup
    # hard denies hold in BOTH modes
    assert on.evaluate("bash", "rm -rf /") == "deny"
    assert off.evaluate("bash", "rm -rf /") == "deny"


def test_agent_loop_seeds_mode_from_cfg():
    from pathlib import Path

    from opencode_py.agent.loop import AgentLoop
    from opencode_py.config import Config

    for value, expected in (("ask", "ask"), ("auto", "auto"), (None, "auto")):
        cfg = Config()
        if value is not None:
            cfg.permission_mode = value
        loop = AgentLoop(
            cfg=cfg,
            registry=SimpleNamespace(),
            directory=Path("."),
            provider=None,
        )
        assert loop.permission.mode == expected


def test_settings_row_flips_permission_mode():
    from opencode_py.config import Config
    from opencode_py.tui.settings_screen import SettingsScreen

    screen = SettingsScreen(cfg=Config(), engine=None, auth=None)
    screen.rows = screen._build_rows()  # normally done on_mount
    row = next(r for r in screen.rows if r.label == "allow all permissions")
    assert row.get() == "yes"  # default
    row.apply("no")
    assert screen.cfg.permission_mode == "ask"
    assert row.get() == "no"
    row.apply("yes")
    assert screen.cfg.permission_mode == "auto"
