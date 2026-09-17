import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencode_py.config import Config
from opencode_py.permission import PermissionEngine, merge_permissions
from opencode_py.tools import build_registry
from opencode_py.tools import skill as skill_mod


def _write(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


GOOD = """---
name: git-release
description: Create consistent releases and changelogs
---
## What I do
- Draft release notes
"""


def test_parse_and_validate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".opencode/skills/git-release/SKILL.md", GOOD)
    skill_mod.clear_cache()
    skills = skill_mod.list_skills(fresh=True)
    assert [s.name for s in skills] == ["git-release"]
    assert skills[0].description.startswith("Create consistent")


def test_bad_skills_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".opencode/skills/Bad_Name/SKILL.md", GOOD)
    _write(tmp_path, ".opencode/skills/mismatch/SKILL.md", GOOD.replace("git-release", "other"))
    _write(tmp_path, ".opencode/skills/nofront/SKILL.md", "no frontmatter here")
    _write(tmp_path, ".opencode/skills/nodesc/SKILL.md", "---\nname: nodesc\n---\nbody here\n")
    skill_mod.clear_cache()
    assert skill_mod.list_skills(fresh=True) == []


def test_project_beats_global_and_claude_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".opencode/skills/dup/SKILL.md", GOOD.replace("git-release", "dup").replace("Create consistent releases and changelogs", "project one"))
    _write(tmp_path, ".claude/skills/dup/SKILL.md", GOOD.replace("git-release", "dup").replace("Create consistent releases and changelogs", "claude one"))
    _write(tmp_path, ".claude/skills/only-claude/SKILL.md", GOOD.replace("git-release", "only-claude").replace("Create consistent releases and changelogs", "claude only"))
    skill_mod.clear_cache()
    skills = {s.name: s for s in skill_mod.list_skills(fresh=True)}
    assert skills["dup"].description == "project one"
    assert skills["only-claude"].description == "claude only"


def test_tool_load_and_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".opencode/skills/git-release/SKILL.md", GOOD)
    skill_mod.clear_cache()
    t = skill_mod.tool()
    out = t.run({"name": "git-release"})
    assert not out.get("error")
    assert "Draft release notes" in out["output"]
    bad = t.run({"name": "nope"})
    assert bad.get("error")
    listed = t.run({})
    assert "git-release" in listed["output"]


def test_deny_hides_from_list_and_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".opencode/skills/a/SKILL.md", GOOD.replace("git-release", "a").replace("Create consistent releases and changelogs", "desc a"))
    _write(tmp_path, ".opencode/skills/b/SKILL.md", GOOD.replace("git-release", "b").replace("Create consistent releases and changelogs", "desc b"))
    skill_mod.clear_cache()
    engine = PermissionEngine.from_config(merge_permissions({"skill": {"b": "deny"}}))
    assert [s.name for s in skill_mod.visible_skills(engine)] == ["a"]
    block = skill_mod.skills_block(engine)
    assert "<name>a</name>" in block and "<name>b</name>" not in block


def test_registered_and_system_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".opencode/skills/git-release/SKILL.md", GOOD)
    skill_mod.clear_cache()
    reg = build_registry(Config())
    assert reg.get("skill") is not None
    from opencode_py.agent import system as system_mod

    prompt = system_mod.build_system_prompt(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m", cfg=Config(),
    )
    assert "<available_skills>" in prompt and "git-release" in prompt


def test_disabled_by_tools_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".opencode/skills/git-release/SKILL.md", GOOD)
    skill_mod.clear_cache()
    cfg = Config()
    cfg.raw = {"tools": {"skill": False}}
    reg = build_registry(cfg)
    assert reg.get("skill") is None
    from opencode_py.agent import system as system_mod

    prompt = system_mod.build_system_prompt(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m", cfg=cfg,
    )
    assert "<available_skills>" not in prompt
