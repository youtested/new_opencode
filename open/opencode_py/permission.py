"""Permission engine: ask / allow / deny with pattern rules.

Mirrors opencode: rule = {permission, pattern, action}; LAST matching rule wins;
default action is "ask". Patterns: * -> .*, ? -> ., anchored, "/" normalization.
Headless default: auto-approve unless the rule denies (opencode --auto) OR a mode
is explicitly set to always-ask.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# how many identical tool calls with identical input before it's a "doom loop"
DOOM_LOOP_THRESHOLD = 3
# cap on remembered "always allow" patterns (LRU-evicted, newest wins)
MAX_APPROVED_PATTERNS = 256
# compiled wildcard-regex cache: match() rebuilt the regex per rule per input
_MATCH_CACHE: dict[str, Any] = {}
_MATCH_CACHE_MAX = 512
_MATCH_CACHE_LOCK = threading.Lock()
# inputs longer than this are prefix-checked (a rule either matches or fails
# within the head; giant tool outputs never need full-regex scans)
MAX_MATCH_INPUT = 32 * 1024


@dataclass
class Rule:
    permission: str
    pattern: str
    action: str  # allow | ask | deny


@dataclass
class PermissionEngine:
    rules: list[Rule] = field(default_factory=list)
    mode: str = "auto"  # auto | ask | deny
    # callback(user-facing description, always_patterns) -> "once"|"always"|"reject"
    ask_callback: Callable[[str, list[str]], str] | None = None
    _approved_patterns: list[str] = field(default_factory=list)
    _last_calls: list[tuple[str, str]] = field(default_factory=list)
    # Parent and parallel sub-agents share one engine: guard the mutable
    # ledgers so concurrent evaluate/ask/reset can't interleave into false
    # doom-loop detections or lost approvals. Never held across the modal
    # callback (siblings must keep streaming while one dialog is open).
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        mode: str = "auto",
        ask_callback: Callable[[str, list[str]], str] | None = None,
    ) -> "PermissionEngine":
        rules: list[Rule] = []
        for perm, value in config.items():
            if perm == "_comment":
                continue
            if isinstance(value, str):
                rules.append(Rule(permission=perm, pattern="*", action=value))
            elif isinstance(value, dict):
                for pattern, action in value.items():
                    if isinstance(action, str):
                        rules.append(Rule(permission=perm, pattern=pattern, action=action))
        engine = cls(rules=rules, mode=mode, ask_callback=ask_callback)
        engine.apply_defaults()
        return engine

    def apply_defaults(self) -> None:
        """opencode defaults: * allow, doom_loop ask, question deny."""
        defaults: dict[str, Any] = {
            "*": "allow",
            "doom_loop": "ask",
            "question": "deny",
            "plan_enter": "deny",
            "plan_exit": "deny",
            "read": {"*": "allow", "*.env": "ask", "*.env.*": "ask", "*.env.example": "allow"},
        }
        for perm, value in defaults.items():
            if self._has_permission(perm):
                continue
            if isinstance(value, str):
                self.rules.append(Rule(permission=perm, pattern="*", action=value))
            else:
                for pattern, action in value.items():
                    self.rules.append(Rule(permission=perm, pattern=pattern, action=action))

    def _has_permission(self, perm: str) -> bool:
        return any(r.permission == perm for r in self.rules)

    # -- pattern matching -------------------------------------------------
    @staticmethod
    def match(pattern: str, value: str) -> bool:
        """Wildcard match: * -> .*, ? -> ., anchored. '/' normalized."""
        pattern = pattern.replace("\\", "/")
        value = value.replace("\\", "/")
        if len(value) > MAX_MATCH_INPUT:
            value = value[:MAX_MATCH_INPUT]
        # trailing ' *' matches "tool" or "tool <anything>"
        if pattern.endswith(" *"):
            base = pattern[:-2]
            base = re.escape(base).replace(r"\*", ".*").replace(r"\?", ".")
            regex = "^" + base + "( .*)?$"
        else:
            escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
            regex = "^" + escaped + "$"
        try:
            rx = _MATCH_CACHE.get(regex)
            if rx is None:
                rx = re.compile(regex)
                with _MATCH_CACHE_LOCK:
                    if len(_MATCH_CACHE) >= _MATCH_CACHE_MAX:
                        _MATCH_CACHE.clear()
                    _MATCH_CACHE[regex] = rx
            return rx.match(value) is not None
        except re.error:
            return False

    def evaluate(self, permission: str, input_value: str = "") -> str:
        """Return the action for a permission+input: allow | ask | deny."""
        if self.mode == "fully_auto":
            # fully_auto: the model does everything with zero popups. Only an
            # explicit deny rule can still block (deny always wins); every ask
            # (incl. *.env, external_directory, doom_loop) becomes allow.
            action = self._find_action(permission, input_value)
            return "deny" if action == "deny" else "allow"
        # doom loop detection (bash tools repeat with identical input)
        if permission in ("bash", "write", "edit"):
            with self._lock:
                self._last_calls.append((permission, input_value))
                if len(self._last_calls) > DOOM_LOOP_THRESHOLD:
                    self._last_calls.pop(0)
                recent = list(self._last_calls[-DOOM_LOOP_THRESHOLD:])
            if len(recent) == DOOM_LOOP_THRESHOLD and len({c for c in recent}) == 1:
                r = self._find_action("doom_loop", input_value)
                if r != "allow":
                    return r

        action = self._find_action(permission, input_value)
        if action == "ask":
            if self.mode == "deny":
                return "deny"
            if self.mode == "auto":
                return "allow"  # headless default: auto-approve (unless denied)
            return "ask"
        return action

    def _find_action(self, permission: str, input_value: str) -> str:
        # check explicit rules (last match wins), then "*" rule, then "ask"
        explicit = [r for r in self.rules if r.permission == permission]
        for r in reversed(explicit):
            if self.match(r.pattern, input_value):
                return r.action
        wildcard = [r for r in self.rules if r.permission == "*"]
        for r in reversed(wildcard):
            if self.match(r.pattern, input_value):
                return r.action
        return "ask"

    def ask(self, description: str, always_patterns: list[str] | None = None) -> bool:
        """Interactively ask the user; returns True if allowed."""
        always_patterns = always_patterns or []
        # if an always-allow pattern already approved, skip. Approved entries are
        # exact "permission input" strings; check literal equality FIRST so a
        # user-supplied command containing wildcard characters ("rm /*") can't
        # accidentally approve a broader pattern via the glob matcher below.
        with self._lock:
            approved_snapshot = list(self._approved_patterns)
        for approved in approved_snapshot:
            if any(p == approved for p in always_patterns):
                return True
            if any(self.match(approved, p) or self.match(p, approved) for p in always_patterns):
                return True
        if self.ask_callback is None:
            return True
        reply = self.ask_callback(description, always_patterns)
        if reply == "always":
            with self._lock:
                for p in always_patterns:
                    # don't store megabyte-sized tool inputs as approval patterns
                    if len(p) > 512:
                        continue
                    if p in self._approved_patterns:
                        # refresh LRU position by re-appending (dedup first)
                        self._approved_patterns.remove(p)
                    self._approved_patterns.append(p)
                # bounded memory: evict the oldest "always allow" approvals so a
                # long session's approval list never grows without limit.
                while len(self._approved_patterns) > MAX_APPROVED_PATTERNS:
                    self._approved_patterns.pop(0)
            return True
        return reply == "once"

    def reset_doom_tracking(self) -> None:
        with self._lock:
            self._last_calls = []


BUILTIN_AGENTS: dict[str, str] = {
    "build": "Full tools — edits files, runs commands, does the work.",
    "plan": "Read-only — makes a plan, changes nothing.",
    "explore": "Read-only — answers questions about the code.",
}

READONLY_TOOLS = ("edit", "write", "apply_patch", "bash")


def agent_spec(cfg: Any, agent: str) -> dict[str, Any]:
    """The stored spec for a custom agent ({} for builtins/unknown)."""
    try:
        agents = getattr(cfg, "agents", None) or {}
        spec = agents.get(agent)
        return dict(spec) if isinstance(spec, dict) else {}
    except Exception:
        return {}


def is_custom_agent(cfg: Any, agent: str) -> bool:
    return agent not in BUILTIN_AGENTS and bool(agent_spec(cfg, agent))


def agent_description(cfg: Any, agent: str) -> str:
    """One-line description for the picker (builtins fixed, customs stored)."""
    if agent in BUILTIN_AGENTS:
        return BUILTIN_AGENTS[agent]
    return str(agent_spec(cfg, agent).get("description") or "Custom agent.")


def agent_readonly(cfg: Any, agent: str) -> bool:
    """True when the agent may not mutate (plan/explore, or a custom with readonly)."""
    if agent in ("plan", "explore"):
        return True
    if agent == "build":
        return False
    spec = agent_spec(cfg, agent)
    if spec.get("readonly") is True:
        return True
    # a custom that denies every mutating tool is read-only in effect
    try:
        tools = spec.get("tools")
        if isinstance(tools, dict) and tools:
            denied = {
                t for t, v in tools.items()
                if (v == "deny" if isinstance(v, str)
                    else isinstance(v, dict) and v.get("*") == "deny"
                    and len(v) == 1)
            }
            if all(t in denied for t in READONLY_TOOLS):
                return True
    except Exception:
        pass
    return False


def list_agents(cfg: Any) -> list[tuple[str, str, bool]]:
    """Every agent: (name, description, is_custom). Builtins first, then
    customs alphabetically — the picker order.

    Never raises (TUI pickers call this on every open); a non-dict agents
    value is treated as no customs — from_dict already rejects it, and the
    headless /agent command reports the config error explicitly.
    """
    out = [(name, BUILTIN_AGENTS[name], False) for name in BUILTIN_AGENTS]
    agents = getattr(cfg, "agents", None) or {}
    if not isinstance(agents, dict):
        return out
    try:
        for name in sorted(agents):
            if name in BUILTIN_AGENTS:
                continue
            out.append((str(name), agent_description(cfg, str(name)), True))
    except Exception:
        pass
    return out


def agents_config_error(cfg: Any) -> str | None:
    """Human-readable config error when agents is malformed, else None."""
    agents = getattr(cfg, "agents", None)
    if agents is None:
        return None
    if not isinstance(agents, dict):
        return f"agents must be a dict of agent specs, got {type(agents).__name__}"
    return None


def agent_effective_tools(
    cfg: Any, agent: str, registry: Any = None,
) -> list[str]:
    """All tool names for the permission editor (new tools appear
    automatically). ALWAYS the full list — out-of-scope tools simply
    show as deny (see agent_tool_action); nothing is ever hidden, for
    present and future agents alike. Sorted for stable display.
    """
    try:
        names = list(registry.names()) if registry is not None else []
    except Exception:
        names = []
    if not names:
        try:
            from .tools import TOOL_NAMES as _NAMES

            names = list(_NAMES)
        except Exception:
            names = []
    return sorted(names)


def agent_md_dir(agent: str) -> Any:
    """Real directory for one agent's .md files:
    `<config>/agents/<agent>/` (e.g. `~/.config/opencode_py/agents/build/`).
    Created on write; never raises."""
    from pathlib import Path as _P

    from .globals import Path as _G

    try:
        base = _G.config
    except Exception:
        base = _P.home() / ".config" / "opencode_py"
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(agent or "build"))
    return base / "agents" / (safe or "build")


def agent_md_path(agent: str, name: str) -> Any:
    """Real file path for one stored .md file. Never raises."""
    from pathlib import Path as _P

    safe = _P(str(name or "notes.md")).name or "notes.md"
    if not safe.lower().endswith(".md"):
        safe += ".md"
    return agent_md_dir(agent) / safe


def _safe_rel(path: Any, base: Any) -> str:
    """Short display path (`~/.config/...`) — never leaks temp dirs."""
    try:
        s = str(path)
        home = str(base)
        import os as _os

        h = _os.path.expanduser("~")
        if s.startswith(h):
            return "~" + s[len(h):]
        return s
    except Exception:
        return str(path)


def short_display_path(path: str, limit: int = 60) -> str:
    """Home-shortened path, middle-truncated (`~/.../agents/build/x.md`)
    when longer than limit. Display only — the real path is untouched."""
    try:
        import os as _os

        s = str(path)
        h = _os.path.expanduser("~")
        if s.startswith(h):
            s = "~" + s[len(h):]
        if len(s) <= limit:
            return s
        head, sep, tail = s.partition("/agents/")
        if sep and tail:
            keep = limit - len(head) - len("/.../agents/") - 12
            if keep > 8:
                return head + "/.../agents/" + tail[-keep:]
        if len(s) > limit:
            return s[:12] + "..." + s[-(limit - 15):]
        return s
    except Exception:
        return str(path)


_MD_MIGRATED: set[str] = set()


def md_entries_for_agent_cfg(cfg: Any, agent: str) -> list[tuple[str, str]]:
    """(filename, content) pairs stored for one agent (builtins included).

    Reads REAL files from `<config>/agents/<agent>/`, with a one-time
    migration of legacy config-embedded entries (`agents.<name>.md`).
    Never raises.
    """
    try:
        from .tui.agent_md_popup import md_entries_for_agent as _entries

        legacy = _entries(agent_spec(cfg, agent))
    except Exception:
        legacy = []
    out: dict[str, str] = {}
    # 1. legacy config entries (migrated to files on first read).
    # Keys normalized through agent_md_path so "guide" and "guide.md"
    # collapse to one row (bug 3: unnormalized keys duplicated rows).
    for fname, content in legacy:
        try:
            key = str(agent_md_path(agent, str(fname)).name)
        except Exception:
            key = str(fname)
        out[key] = str(content)
    # 2. real files win over same-named legacy entries
    try:
        d = agent_md_dir(agent)
        if d.is_dir():
            for p in sorted(d.glob("*.md")):
                try:
                    if p.is_file():
                        out[p.name] = p.read_text(encoding="utf-8", errors="replace")[:100_000]
                except OSError:
                    continue
    except Exception:
        pass
    # bug 4: the "one-time" migration ran disk IO on EVERY list call
    # (agent_md_groups calls this per agent per refresh). Guard per-agent
    # per-process so files are flushed once; disk remains authoritative.
    if legacy and agent not in _MD_MIGRATED:
        _MD_MIGRATED.add(agent)
        try:
            for fname, content in legacy:
                fp = agent_md_path(agent, fname)
                if not fp.exists():
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(str(content), encoding="utf-8")
        except Exception:
            pass
    return sorted(out.items(), key=lambda kv: kv[0].lower())


SHARED_AGENT = "shared"
SHARED_FILE = "AGENT.md"


def shared_md_dir() -> Any:
    """Real directory for the shared file: `<config>/agents/` (the parent
    of every per-agent dir). Created on write; never raises."""
    return agent_md_dir("__shared__").parent


def shared_md_path() -> Any:
    """Real file path of the shared AGENT.md. Never raises."""
    return shared_md_dir() / SHARED_FILE


def shared_md_text(cfg: Any = None) -> str:
    """The shared AGENT.md content (disk file wins, legacy config `md`
    under a `shared` agent spec migrates once). Never raises."""
    try:
        fp = shared_md_path()
        if fp.is_file():
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")[:100_000]
            except OSError:
                text = ""
            if text.strip():
                return text
    except Exception:
        pass
    if cfg is not None:
        try:
            for n, c in md_entries_for_agent_cfg(cfg, SHARED_AGENT):
                if str(n).lower() == SHARED_FILE.lower() and str(c).strip():
                    return str(c)
            for n, c in md_entries_for_agent_cfg(cfg, SHARED_AGENT):
                if str(c).strip():
                    return str(c)
        except Exception:
            pass
    return ""


def agent_md_groups(cfg: Any) -> list[tuple[str, list[tuple[str, str]]]]:
    """Shared group first, then all agents in Agents-popup order
    (builtins first, then customs alphabetically)."""
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    try:
        text = shared_md_text(cfg)
    except Exception:
        text = ""
    groups.append((SHARED_AGENT, [(SHARED_FILE, text)] if text.strip() else []))
    for name, _desc, _custom in list_agents(cfg):
        if name == SHARED_AGENT:
            continue
        groups.append((name, md_entries_for_agent_cfg(cfg, name)))
    return groups


def agent_combined_md(cfg: Any, agent: str) -> str:
    """Shared AGENT.md FIRST, then the agent's own files — joined for prompt
    injection. So build gets shared+workflow, plan gets shared+rules, and
    every future agent gets shared+its own automatically."""
    parts: list[str] = []
    try:
        shared = shared_md_text(cfg)
    except Exception:
        shared = ""
    if shared.strip():
        parts.append(shared)
    parts.extend(c for _n, c in md_entries_for_agent_cfg(cfg, agent) if c.strip())
    return "\n\n".join(parts)


def default_permissions(agent: str = "build") -> dict[str, Any]:
    """Mirror opencode's per-agent permission configs."""
    base: dict[str, Any] = {
        "*": "allow",
        "external_directory": {"*": "ask"},
        "read": {"*": "allow", "*.env": "ask", "*.env.*": "ask", "*.env.example": "allow"},
    }
    if agent == "build":
        base["question"] = "allow"
        base["plan_enter"] = "allow"
        base["browser"] = "allow"
        base["phone"] = "allow"
    elif agent == "plan":
        base["question"] = "allow"
        base["plan_exit"] = "allow"
        base["task"] = {"general": "deny"}
        # plan agent is read-only: deny every mutating tool (defense-in-depth;
        # the loop also filters these from the model's tool schemas)
        base["edit"] = {"*": "deny"}
        base["write"] = {"*": "deny"}
        base["bash"] = {"*": "deny"}
        base["apply_patch"] = {"*": "deny"}
        # browser: read-only actions only (status/tabs/snapshot/screenshot).
        # NOTE: "*" goes FIRST so the specific allow rules (checked last,
        # last-match-wins) override it; mutating actions fall to deny.
        base["browser"] = {
            "*": "deny",
            "status *": "allow",
            "tabs *": "allow",
            "snapshot *": "allow",
            "screenshot *": "allow",
        }
        base["phone"] = {
            "*": "deny",
            "status *": "allow",
            "screen *": "allow",
            "ui *": "allow",
            "focused *": "allow",
            "app_list *": "allow",
        }
    elif agent == "explore":
        # explore agent is a pure retrieval agent: same read-only walls as
        # plan, and it may only spawn further READ-ONLY sub-agents — never
        # build/general, which could mutate files.
        base["task"] = {"*": "deny", "plan": "allow", "explore": "allow"}
        base["edit"] = {"*": "deny"}
        base["write"] = {"*": "deny"}
        base["bash"] = {"*": "deny"}
        base["apply_patch"] = {"*": "deny"}
        base["browser"] = {
            "*": "deny",
            "status *": "allow",
            "tabs *": "allow",
            "snapshot *": "allow",
            "screenshot *": "allow",
        }
        base["phone"] = {
            "*": "deny",
            "status *": "allow",
            "screen *": "allow",
            "ui *": "allow",
            "focused *": "allow",
            "app_list *": "allow",
        }
    return base


def merge_permissions(
    user: dict[str, Any], agent: str = "build", cfg: Any = None,
) -> dict[str, Any]:
    """Merge user config permission over the agent defaults.

    Per-agent specs in cfg.agents layer their own per-tool rules on top —
    for custom agents AND builtins with stored overrides (the permission
    editor writes agents.<any>.tools): builtin defaults -> spec rules ->
    user global rules, so an agent's editor choices always win for it.
    """
    defaults = default_permissions(agent)
    if cfg is not None:
        try:
            spec = agent_spec(cfg, agent)
            tools = spec.get("tools")
            if isinstance(tools, dict):
                for perm, value in tools.items():
                    if isinstance(value, dict) and isinstance(defaults.get(perm), dict):
                        d = dict(defaults[perm])
                        d.update(value)
                        defaults[perm] = d
                    else:
                        defaults[perm] = value
            if spec.get("readonly") is True:
                for tool in READONLY_TOOLS:
                    defaults[tool] = {"*": "deny"}
        except Exception:
            pass
    merged = dict(defaults)
    for perm, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(perm), dict):
            d = dict(merged[perm])
            d.update(value)
            merged[perm] = d
        else:
            merged[perm] = value
    return merged
