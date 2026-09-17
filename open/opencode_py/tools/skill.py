from __future__ import annotations

"""skill tool: load SKILL.md files on demand.

RULES (full guidance for maintainers; the sent schema is short):
- Menu of name+description lives in the system prompt; call with a name
  only when the task matches, then follow the loaded body.
- Empty name lists skills. Denied skills stay hidden. Bodies capped.
"""

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import Tool, schema_with

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESC = 1024
MAX_BODY = 50 * 1024
# Shared display budget: prompt block + tool description + list output.
SKILL_LIST_LIMIT = 40
SKILL_ERROR_LIST_LIMIT = 20
# Wire cap for one loaded skill body (history replays it every step).
SKILL_OUTPUT_MAX_CHARS = 30 * 1024

SKILL_DIRS = (".opencode/skills", ".claude/skills", ".agents/skills")

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, list["Skill"]]] = {}
_CACHE_TTL = 5.0
_CACHE_MAX_KEYS = 32


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source: str = ""


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line or line[:1] in (" ", "\t"):
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        if key in ("name", "description"):
            meta[key] = val.strip().strip("\"'")
    return meta, "\n".join(lines[end + 1:])


def _valid(name: str, desc: str, dirname: str) -> bool:
    return validate_skill(name, desc, dirname, None)[0]


def validate_skill(name: str, desc: str, dirname: str, body: str) -> tuple[bool, str]:
    """Validate a skill, returning (ok, reason). Reason powers /skills validate."""
    if not name:
        return False, "missing frontmatter `name`"
    if len(name) > MAX_NAME:
        return False, f"name too long ({len(name)} > {MAX_NAME})"
    if not NAME_RE.match(name):
        return False, "name must be lowercase alphanumeric with single dashes"
    if name != dirname:
        return False, f"name {name!r} != directory {dirname!r}"
    if not desc:
        return False, "missing frontmatter `description`"
    if len(desc) > MAX_DESC:
        return False, f"description too long ({len(desc)} > {MAX_DESC})"
    if body is not None and not body.strip():
        return False, "empty body"
    return True, "ok"


def _read_skill_file(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    meta, body = _parse_frontmatter(text)
    name = meta.get("name", "")
    desc = meta.get("description", "")
    if not _valid(name, desc, path.parent.name):
        return None
    body = body.strip()
    if not body:
        return None
    raw = body.encode("utf-8")
    if len(raw) > MAX_BODY:
        # Truncate at a UTF-8 char boundary (slicing bytes can split a
        # multibyte char into U+FFFD).
        cut = raw[:MAX_BODY]
        body = cut.decode("utf-8", "ignore")
    return Skill(name=name, description=desc, body=body, source=str(path))


def _worktree() -> Path:
    try:
        from ..globals import resolve_worktree

        return resolve_worktree(Path.cwd())
    except OSError:
        return Path.cwd()


def _scan() -> list[Skill]:
    found: dict[str, Skill] = {}
    wt = _worktree().resolve()
    # Walk up to the git root only. Without a repo the old code scanned every
    # ancestor to `/` (home + system skill dirs unintentionally + iterdir
    # storm). No .git anywhere → project skills come from the worktree alone.
    # The launch directory leads: running from a subdir (e.g. open/ inside
    # a monorepo) still finds that subdir's own .opencode/skills first.
    # (Appended AFTER the for/else: the else resets the chain when no git
    # root is found walking up, which would otherwise wipe this entry.)
    chain: list[Path] = [wt]
    for parent in wt.parents:
        chain.append(parent)
        if (parent / ".git").exists():
            break
    else:
        chain = [wt]
    try:
        cwd = Path.cwd().resolve()
        if cwd != wt:
            chain.insert(0, cwd)
    except OSError:
        pass
    for base in chain:
        for sub in SKILL_DIRS:
            root = base / sub
            try:
                kids = sorted(p for p in root.iterdir() if p.is_dir())
            except OSError:
                continue
            for kid in kids:
                if kid.name in found:
                    continue
                skill = _read_skill_file(kid / "SKILL.md")
                if skill is not None:
                    found[kid.name] = skill
    try:
        from ..globals import Path as GPath

        # Legacy compat: ~/.config/opencode_py/skills (undocumented but
        # previously the only opencode_py global). The dead doubly-nested
        # opencode_py/opencode/skills path is gone.
        for bucket in (GPath.config / "skills",):
            try:
                kids = sorted(p for p in bucket.iterdir() if p.is_dir())
            except OSError:
                continue
            for kid in kids:
                if kid.name in found:
                    continue
                skill = _read_skill_file(kid / "SKILL.md")
                if skill is not None:
                    found[kid.name] = skill
    except Exception:
        pass
    try:
        home = Path.home()
        buckets: list[Path] = [
            home / sub for sub in (".claude/skills", ".agents/skills", ".opencode/skills")
        ]
        # README documents global ~/.config/opencode/skills.
        try:
            import os as _os

            _xdg = _os.environ.get("XDG_CONFIG_HOME")
            _cfg_home = Path(_xdg) if _xdg else home / ".config"
            buckets.append(_cfg_home / "opencode" / "skills")
        except Exception:
            pass
        for root in buckets:
            try:
                kids = sorted(p for p in root.iterdir() if p.is_dir())
            except OSError:
                continue
            for kid in kids:
                if kid.name in found:
                    continue
                skill = _read_skill_file(kid / "SKILL.md")
                if skill is not None:
                    found[kid.name] = skill
    except Exception:
        pass
    return sorted(found.values(), key=lambda s: s.name)


def list_skills(*, fresh: bool = False) -> list[Skill]:
    try:
        key = str(_worktree().resolve())
    except OSError:
        key = str(_worktree())
    now = __import__("time").monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and not fresh:
            ts, skills = hit
            if now - ts < _CACHE_TTL:
                return list(skills)
    skills = _scan()
    with _LOCK:
        _CACHE[key] = (now, skills)
        while len(_CACHE) > _CACHE_MAX_KEYS:
            _CACHE.pop(next(iter(_CACHE)), None)
    return list(skills)


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def visible_skills(permission=None) -> list[Skill]:
    skills = list_skills()
    if permission is None:
        return skills
    try:
        from ..permission import PermissionEngine

        match = PermissionEngine.match
    except Exception:
        return skills
    out: list[Skill] = []
    for s in skills:
        action = "allow"
        try:
            action = permission.evaluate("skill", s.name)
        except Exception:
            action = "allow"
        if action == "deny":
            continue
        if action == "ask" and getattr(permission, "mode", "auto") == "deny":
            continue
        out.append(s)
    return out


def skills_block(permission=None, limit: int = SKILL_LIST_LIMIT) -> str:
    """Skills menu rendered per skills.md (no hardcoded XML here)."""
    from ..agent.system import _block_template as _block

    skills = visible_skills(permission)[: max(0, limit)]
    if not skills:
        return ""
    entries = "\n".join(
        f"<skill>\n<name>{s.name}</name>\n<description>{s.description}</description>\n</skill>"
        for s in skills
    )
    return _block("skills.md", "{entries}", entries)


def _load(name: str, permission=None) -> Skill | None:
    want = (name or "").strip()
    if not want or not NAME_RE.match(want):
        return None
    if permission is not None:
        try:
            if permission.evaluate("skill", want) == "deny":
                return None
        except Exception:
            pass
    for s in list_skills():
        if s.name == want:
            return s
    return None


def tool(registry=None) -> Tool:
    description = "Load a SKILL.md by name when the task matches. Empty name lists skills."

    def run(arguments: dict) -> dict:
        perm = getattr(registry, "_skill_permission", None)
        name = str(arguments.get("name", "")).strip()
        if not name:
            skills = visible_skills(perm)
            if not skills:
                return {"output": "No skills installed. Create .opencode/skills/<name>/SKILL.md."}
            lines = ["Available skills:"]
            for s in skills[:SKILL_LIST_LIMIT]:
                lines.append(f"- {s.name}: {s.description}")
            lines.append('Load one with skill({"name": "<name>"}).')
            return {"output": "\n".join(lines)}
        if perm is not None:
            try:
                if perm.evaluate("skill", name) == "deny":
                    return {"output": f'Skill "{name}" is disabled by permission.', "error": True}
            except Exception:
                pass
        skill = _load(name, perm)
        if skill is None:
            skills = visible_skills(perm)
            names = ", ".join(s.name for s in skills[:SKILL_ERROR_LIST_LIMIT]) or "(none installed)"
            return {"output": f'Unknown skill "{name}". Available: {names}', "error": True}
        body = skill.body
        if len(body.encode("utf-8")) > SKILL_OUTPUT_MAX_CHARS:
            cut = body.encode("utf-8")[:SKILL_OUTPUT_MAX_CHARS].decode("utf-8", "ignore")
            body = cut + "\n\n…(truncated to wire cap)"
        return {
            "output": f"# Skill: {skill.name}\n{skill.description}\n\n{body}",
            "metadata": {"skill": skill.name, "source": skill.source},
        }

    return Tool(
        name="skill",
        description=description,
        parameters=schema_with(
            {
                "name": {
                    "type": "string",
                    "description": "Skill name (empty = list)",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="skill",
    )
