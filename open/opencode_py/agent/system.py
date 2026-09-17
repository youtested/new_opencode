"""System prompt assembly.

Mirrors opencode: base prompt (default.txt) + environment block + AGENTS.md
instructions + user system override. Plan/build differences are injected as
<system-reminder> text parts appended to the user message (in loop.py).
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from ..config import Config
from ..globals import resolve_worktree

_PROMPT_DIR = Path(__file__).resolve().parent
_base_cache: str | None = None


def _read_prompt_file(name: str) -> str:
    """Read a prompt .md next to this module. Raises FileNotFoundError
    with a clear message when missing — no hardcoded prompt text lives
    in code anymore."""
    try:
        text = (_PROMPT_DIR / name).read_text(encoding="utf-8").strip()
    except OSError as e:
        raise FileNotFoundError(
            f"prompt file missing: {_PROMPT_DIR / name} ({e})"
        ) from e
    if not text:
        raise ValueError(f"prompt file empty: {_PROMPT_DIR / name}")
    return text


def _base_text() -> str:
    """Base prompt from base.md (cached)."""
    global _base_cache
    if _base_cache is None:
        _base_cache = _read_prompt_file("base.md")
    return _base_cache


def clear_prompt_cache() -> None:
    """Forget the cached base.md (tests / live reload)."""
    global _base_cache
    _base_cache = None

# NOTE: agent behavior rules used to live here as hardcoded reminder
# constants (PLAN/EXPLORE/BUILD_SWITCH/KEEP_GOING) injected into every
# user message. They now live ONLY in the agents' .md files
# (build/workflow.md, plan/rules.md, explore/rules.md via the Agent.md
# manager) — keeping both sent every rule twice.


def find_instruction_files(directory: Path, worktree: Path, cfg: Config) -> list[Path]:
    """Find AGENTS.md/CLAUDE.md files: global then project (walk up to worktree)."""
    files: list[Path] = []

    global_home = Path(os.path.expanduser("~/.config/opencode/AGENTS.md"))
    if global_home.exists():
        files.append(global_home)

    d = directory.resolve()
    seen_dirs: set[Path] = set()
    while True:
        if d in seen_dirs:
            # symlink cycle (defense-in-depth): never walk a path twice
            break
        seen_dirs.add(d)
        # NOTE: workflow rules live in the build agent's workflow.md
        # (Agent.md manager) — the single source of truth, no duplicate file.
        for name in ("AGENTS.md", "CLAUDE.md"):
            p = d / name
            if p.exists():
                files.append(p)
                break
        if d == worktree:
            break
        if d.parent == d:
            break
        d = d.parent

    for pattern in cfg.instructions:
        try:
            matches = worktree.glob(pattern)
            for m in matches:
                if m.is_file():
                    files.append(m)
        except ValueError:
            pass

    # dedupe, preserve order
    seen: set[Path] = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def build_environment(directory: Path, worktree: Path, provider_id: str, model_id: str) -> str:
    """One-line env facts. Format lives in environment.md (template docs);
    the builder fills it — identical output with or without the file."""
    is_git = (worktree / ".git").exists()
    platform = os.uname().sysname if hasattr(os, "uname") else "?"
    return (
        f"Model: {provider_id}/{model_id} | dir: {directory} | root: {worktree} | "
        f"git: {'yes' if is_git else 'no'} | {platform} | "
        f"{datetime.date.today().isoformat()}"
    )


def labeled_prompt_parts(
    *,
    directory: Path,
    worktree: Path,
    provider_id: str,
    model_id: str,
    cfg: Config,
    agent: str = "build",
) -> list[tuple[str, str]]:
    """Every system-prompt block as (label, text) — the SINGLE builder the
    sender AND the preview both use, so the preview misses not one letter.
    Labels name the source file exactly as injected below."""
    blocks: list[tuple[str, str]] = []
    try:
        base = _base_text()
        if getattr(cfg, "low_data", False):
            base = _slim_prompt(base)
    except Exception as e:
        base = f"[base.md unreadable: {e}]"
    blocks.append(("base.md", base))
    try:
        blocks.append(("environment", build_environment(directory, worktree, provider_id, model_id)))
    except Exception as e:
        blocks.append(("environment", f"[environment unreadable: {e}]"))
    try:
        found = find_instruction_files(directory, worktree, cfg)
    except Exception:
        found = []
    for path in found:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            content = f"[unreadable: {e}]"
        blocks.append((f"Instructions from: {path}", content))
    # per-agent .md files (Agent.md manager): stored under
    # agents.<name>.md, injected only for the active agent.
    try:
        from ..permission import agent_combined_md as _agent_md

        md_text = _agent_md(cfg, agent)
        if md_text.strip():
            blocks.append((f"Agent instructions ({agent}.md)", md_text))
    except Exception:
        pass
    try:
        memory = _load_memory(str(worktree))
    except Exception:
        memory = ""
    if memory:
        try:
            blocks.append(("memory", _memory_block(memory)))
        except Exception:
            blocks.append(("memory", memory))
    try:
        skills = _load_skills(cfg, agent)
    except Exception:
        skills = ""
    if skills:
        blocks.append(("skills", skills))
    try:
        override = cfg.system_prompt or ""
    except Exception:
        override = ""
    if override:
        blocks.append(("system_prompt (config override)", override))
    return [(label, text) for label, text in blocks if text]


def build_system_prompt(
    *,
    directory: Path,
    worktree: Path,
    provider_id: str,
    model_id: str,
    cfg: Config,
    agent: str = "build",
) -> str:
    blocks = labeled_prompt_parts(
        directory=directory, worktree=worktree, provider_id=provider_id,
        model_id=model_id, cfg=cfg, agent=agent,
    )
    return "\n\n".join(text for _label, text in blocks)


def _block_template(name: str, placeholder: str, content: str) -> str:
    """Render a prompt block from its .md template file.

    The file holds a fenced example showing the shape; the fence body is
    the template. No hardcoded copy lives here — the file is the source.
    """
    template = _read_prompt_file(name)
    body = template
    lines = template.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == "```"), None)
    end = next(
        (i for i, l in enumerate(lines) if l.strip() == "```" and i != start),
        None,
    ) if start is not None else None
    if start is not None and end is not None and end > start:
        body = "\n".join(lines[start + 1 : end])
    return body.replace(placeholder, content)


def _memory_block(notes: str) -> str:
    """Memory notes wrapped per memory.md (no hardcoded prefix here)."""
    return _block_template("memory.md", "{content}", notes)


def _load_memory(worktree: str, limit: int = 50) -> str:
    """Return the stored `remember` notes for this project (plus global ones)
    as a bulleted block, or '' when nothing is saved. Safely tolerant of a
    corrupt/absent memory file."""
    from ..globals import Path as GPath

    try:
        raw = (GPath.data / "memory.json").read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return ""
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return ""
    notes = [e for e in entries if e.get("text") and (not e.get("project") or e.get("project") == worktree)][-limit:]
    if not notes:
        return ""
    lines = [f"- {str(e.get('text')).replace(chr(10), ' ')}" for e in notes]
    return "\n".join(lines)


def _load_skills(cfg: Config, agent: str = "build") -> str:
    try:
        from ..permission import PermissionEngine, merge_permissions
        from ..tools.skill import skills_block

        raw = getattr(cfg, "raw", None) or {}
        tools_cfg = raw.get("tools") if isinstance(raw, dict) else None
        if isinstance(tools_cfg, dict) and tools_cfg.get("skill") is False:
            return ""
        agents_cfg = getattr(cfg, "agents", None) or {}
        if isinstance(agents_cfg, dict):
            spec = agents_cfg.get(agent) or {}
            if isinstance(spec, dict):
                tools = spec.get("tools")
                if isinstance(tools, dict) and tools.get("skill") is False:
                    return ""
        perm = getattr(cfg, "permission", None) or {}
        mode = str(getattr(cfg, "permission_mode", "") or "auto").lower()
        if mode not in ("auto", "ask", "deny", "fully_auto"):
            mode = "auto"
        engine = PermissionEngine.from_config(merge_permissions(perm, agent), mode=mode)
        return skills_block(engine)
    except Exception:
        return ""


def _slim_prompt(prompt: str) -> str:
    """Low-data variant: the base is already short; only strip any
    <example> blocks if present. Everything left is directives."""
    import re as _re

    slim = _re.sub(r"<example>.*?</example>", "", prompt, flags=_re.DOTALL)
    slim = _re.sub(r"\n{3,}", "\n\n", slim).strip()
    return slim or prompt


def agent_reminder(agent: str, was_plan: bool, cfg: Config | None = None) -> None:
    """Retired: agent rules now live ONLY in the agents' .md files
    (build/workflow.md, plan/rules.md, explore/rules.md via the Agent.md
    manager). Kept as a no-op so old callers don't break; always None so
    nothing is ever appended to the user message."""
    return None
