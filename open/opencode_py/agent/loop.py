"""Main agentic loop.

Flow per turn:
  1. build messages (system + trimmed history + latest user turn + agent reminder)
  2. stream model -> emit text/reasoning/tool_call events to on_event
  3. if tool calls: permission check -> run tool -> append tool result; loop again
  4. cap iterations (safety) and honor interrupt

build agent: full tools. plan agent: edit/write/bash denied by permissions.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..globals import resolve_worktree
from ..permission import PermissionEngine, merge_permissions
from ..providers import ContextOverflowError, ProviderError, RateLimitError, StreamInterrupted, build_rotation
from ..providers.base import usage_total, usage_total_dict
from ..session import new_session, save_session
from ..tools import glob as glob_mod
from ..tools.registry import Registry
from . import compaction as compact_mod
from . import messages as msg_mod
from . import parse as parse_mod
from . import system as system_mod
from . import trim as trim_mod

MAX_STEPS = 50
MAX_UNDO = 20
AUTO_CONTINUE_MSG = "keep going dont stop keep going never stop"
AUTO_CONTINUE_MAX_NUDGES = 25
AUTO_CONTINUE_WAIT_S = 2.0


def _missing_directories(path: Path) -> list[Path]:
    """Return the chain of directories (deepest-first) that don't exist yet,
    walking from `path` upward. Used to undo mkdir(parents=True) side effects."""
    missing: list[Path] = []
    p = path
    while str(p) not in ("", ".") and not p.exists():
        missing.append(p)
        parent = p.parent
        if parent == p:
            break
        p = parent
    return missing


class _BgAbandoned(Exception):
    """Internal: bg summary worker abandons on user interrupt."""


@dataclass
class TurnResult:
    text: str = ""
    reasoning: str = ""
    tool_calls_made: int = 0
    usage: dict[str, int] | None = None
    provider_id: str = ""
    model_id: str = ""
    finish_reason: str = ""
    error: str = ""
    # True when the turn died on a network/transport failure (disconnect,
    # DNS, timeout) as opposed to a model/API error — the TUI uses this to
    # offer automatic resume once connectivity returns.
    network_failed: bool = False


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


class AgentLoop:
    def __init__(
        self,
        *,
        cfg: Config,
        registry: Registry,
        directory: Path,
        provider=None,
        auth=None,
        permission_engine: PermissionEngine | None = None,
        question_service: Any = None,
        on_event: Callable[[dict], None] | None = None,
        agent: str = "build",
        provider_id: str = "",
        model_id: str = "",
        interrupt: Callable[[], bool] | None = None,
        session_id: str | None = None,
        provider_factory: Callable[[], Any] | None = None,
        ask_lock: threading.Lock | None = None,
        depth: int = 0,
    ):
        self.cfg = cfg
        self.registry = registry
        # nesting level of THIS engine: main agent = 0, its sub-agents = 1, ...
        # bounded by cfg.subagent_depth in spawn_task
        self._depth = max(0, int(depth))
        self.directory = directory
        self.worktree = resolve_worktree(directory)
        self.auth = auth
        self.agent = agent
        self._prev_agent: str = agent
        self._turn_agent: str = agent
        # Fall back to the configured provider/model so the active lane is the
        # source of truth (a hardcoded "opencode" default would resolve the
        # wrong context window / output limit for any other provider).
        self.provider_id = provider_id or cfg.provider
        self.model_id = model_id or cfg.model
        # `interrupt` is a property: assigning it also re-points the registry's
        # `interrupt_check` hook (which bash/webfetch read at call time). The
        # TUI wires the live callback AFTER construction (`_wire_engine`), so a
        # plain attribute would leave the tools frozen on the init default and
        # ESC/Ctrl+C could never abort a running command.
        self.interrupt = interrupt or (lambda: False)
        self._session_id = session_id or uuid.uuid4().hex
        # factory used to build rotations for spawned sub-agents (override in
        # tests); default matches the parent's own rotation construction.
        self.provider_factory = provider_factory or (lambda: build_rotation(cfg, auth, self._session_id))

        self.rotation = provider or build_rotation(cfg, auth, self._session_id)
        # When True the selected model is pinned: rate limits / hard failures
        # surface instead of failing over to another lane (the TUI lock dot).
        self.rotation_locked = bool(getattr(cfg, "rotation_lock", False))

        # mode comes from cfg so the Settings "allow all permissions" toggle
        # is the single source of truth ("ask" -> popups; "auto"/"fully_auto" ->
        # never ask; explicit deny rules apply in ALL modes)
        raw_mode = str(getattr(cfg, "permission_mode", "auto") or "auto").lower()
        perm_mode = raw_mode if raw_mode in ("ask", "deny", "fully_auto") else "auto"
        self.permission = permission_engine or PermissionEngine.from_config(
            merge_permissions(cfg.permission, agent, cfg),
            mode=perm_mode,
        )
        self.on_event = on_event
        self._history: list[dict] = []
        self._call_seq = 0
        self._pending_calls: list[dict] = []
        # FIFO of prompts submitted while a turn was running. Mirroring
        # opencode, these are consumed INSIDE the running turn (a single
        # "Session Drain") at the next provider-turn boundary — the turn keeps
        # working straight into the next prompt instead of ending and starting
        # a fresh turn.
        self._prompt_queue: list[str] = []
        # Lazy rotation rebuild: build_rotation() can hit the network (model
        # catalogs). UI-side callers mark dirty instead of rebuilding on the
        # UI thread; run_turn rebuilds here on the engine thread.
        self._rotation_dirty = False
        self._prompt_lock = threading.Lock()
        # Serializes permission/question prompts for parallel tool calls WITHIN
        # this agent only: the TUI shows exactly one dialog at a time. Each
        # sub-agent gets its OWN lock (see spawn_task) — siblings keep
        # streaming while one waits on its modal, and the app's dialog queue
        # displays their prompts one by one instead of stacking screens.
        self._ask_lock = ask_lock or threading.Lock()
        # Persistent worker pool reused across steps (creating a pool per step
        # churned threads on every tool-loop iteration). Shut down in close().
        self._tool_pool: Any = None
        self._tool_pool_lock = threading.Lock()
        # Bounds simultaneously running sub-agents (see _MAX_CONCURRENT_SUBAGENTS).
        self._subagent_sem = threading.Semaphore(self._MAX_CONCURRENT_SUBAGENTS)
        # Per-thread id of the tool call currently spawning a sub-agent. With
        # parallel execution each worker thread sets its own call id, so a
        # subagent_start event can be linked back to the exact tool row that
        # launched it (official opencode keys sub-agents by their task call).
        self._spawn_ctx = threading.local()
        self._undo_stack: list[dict] = []
        # Background sub-agents: at most ONE live detached child (RAM +
        # context budget on phones). Keyed by session id while running;
        # finished results stay readable until collected or replaced.
        self._bg_agents: dict[str, dict[str, Any]] = {}
        self._bg_lock = threading.Lock()
        self._usage_total: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self._compaction_summary: str = ""
        # Early-summary worker (fix 2): a daemon summary pass started at ~70%
        # of the window, so the freeze-turn never happens. Keyed by history
        # length so a newer turn invalidates a stale worker's result.
        self._early_summary_text: str = ""
        self._early_summary_at: int = -1
        self._early_summary_head: dict = {}
        self._early_summary_lock = threading.Lock()
        self.subagents: dict[str, "AgentLoop"] = {}
        # the task tool looks this up lazily so sub-agents can nest
        self.registry.task_spawner = self.spawn_task
        # Registry.reads `interrupt_check` at call time so ESC/Ctrl+C aborts a
        # running command/fetch. The `interrupt` property keeps it in sync
        # whenever the TUI wires the live callback after construction.
        self.registry.interrupt_check = self.interrupt
        # the question tool asks the user through this service (TUI attaches an
        # ask_callback that surfaces a modal and blocks until answered). Built
        # after permissions so the per-agent "question" rule is already loaded.
        from ..question import QuestionService

        self.question_service = question_service or QuestionService()
        if perm_mode == "fully_auto":
            # fully_auto: the question tool never opens a dialog. Auto-answer
            # with the model's own recommended (first) option so the turn
            # continues toward the goal with zero popups. A `notice` event
            # tells the TUI (which shows a small toast) so the user knows
            # why no question appeared.
            def _fully_auto_ask(questions: list) -> list[list[str]]:
                try:
                    self._emit("notice", text=(
                        "fully_auto: question auto-answered — "
                        "switch permission mode to see questions"
                    ))
                except Exception:
                    pass
                out: list[list[str]] = []
                for q in questions:
                    opts = getattr(q, "options", None) or []
                    first = getattr(opts[0], "label", "") if opts else ""
                    out.append([first] if first else [])
                return out
            self.registry.question_asker = _fully_auto_ask
        else:
            self.registry.question_asker = self.question_service.ask
        # Live permission for skill listing/enforcement (skill tool reads
        # registry._skill_permission; without this it always saw None = all).
        try:
            self.registry._skill_permission = self.permission
        except Exception:
            pass

    @property
    def interrupt(self) -> Callable[[], bool]:
        return self._interrupt

    @interrupt.setter
    def interrupt(self, value: Callable[[], bool]) -> None:
        self._interrupt = value
        registry = getattr(self, "registry", None)
        if registry is not None:
            try:
                registry.interrupt_check = value
            except Exception:
                pass

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Bind a session id, keeping the rotation's Zen identity in sync.

        The TUI assigns the persisted session id after construction; the
        rotation's `x-opencode-session` must follow so every provider built
        for the current conversation reports the same id.
        """
        self._session_id = value
        rotation = getattr(self, "rotation", None)
        if rotation is not None and getattr(rotation, "session_id", None) is not None:
            rotation.session_id = value

    def find_subagent(self, session_id: str) -> "AgentLoop | None":
        """Depth-first search for a sub-agent by session id (nested included)."""
        stack = [self]
        while stack:
            loop = stack.pop()
            if loop.session_id == session_id:
                return loop
            stack.extend(loop.subagents.values())
        return None

    # -- event plumbing --------------------------------------------------
    def _emit(self, kind: str, **kwargs: Any) -> None:
        event = {"kind": kind, "session_id": self._session_id, **kwargs}
        if self.on_event:
            self.on_event(event)

    # -- prompt queue (opencode's queue-and-promote, in-turn drain) -------
    MAX_PROMPT_QUEUE = 10
    MAX_PROMPT_CHARS = 32 * 1024

    def queue_prompt(self, text: str) -> int:
        """Add a prompt submitted while a turn is running.

        It sits in the FIFO until the running turn reaches its next
        provider-turn boundary, then the engine folds it into the SAME turn
        (opencode's Session Drain). Returns the queue depth after adding.

        Bounded: at most MAX_PROMPT_QUEUE prompts are kept (oldest dropped
        first) and each prompt is truncated to MAX_PROMPT_CHARS, so typing
        during a long turn cannot grow memory without bound.
        """
        text = str(text or "")
        if len(text) > self.MAX_PROMPT_CHARS:
            text = text[: self.MAX_PROMPT_CHARS]
        with self._prompt_lock:
            while len(self._prompt_queue) >= self.MAX_PROMPT_QUEUE:
                self._prompt_queue.pop(0)
            self._prompt_queue.append(text)
            return len(self._prompt_queue)

    def prompt_pending(self) -> int:
        with self._prompt_lock:
            return len(self._prompt_queue)

    def prompt_peek(self) -> str:
        with self._prompt_lock:
            return self._prompt_queue[0] if self._prompt_queue else ""

    def pop_prompt(self) -> str | None:
        """Pop the oldest queued prompt (used by the TUI for the leftover after a
        drain ends; run_turn itself uses ``_next_prompt`` internally)."""
        return self._next_prompt()

    def _next_prompt(self) -> str | None:
        """Pop the oldest queued prompt (engine thread, run_turn)."""
        with self._prompt_lock:
            if not self._prompt_queue:
                return None
            return self._prompt_queue.pop(0)

    def clear_prompts(self) -> None:
        with self._prompt_lock:
            self._prompt_queue.clear()

    def rebuild_rotation(self) -> None:
        """Rebuild the failover lanes from the current config.

        The rotation is built once at startup, so picking a different model or
        provider at runtime would otherwise keep using the old lanes. Call this
        whenever `cfg.model` / `cfg.provider` change so the next turn uses the
        newly picked model.
        """
        self.rotation = build_rotation(self.cfg, self.auth, self._session_id)

    def mark_rotation_dirty(self) -> None:
        """Defer rebuild_rotation() to the start of the next turn (engine thread).

        build_rotation() can hit the network (catalog refresh), so callers on
        the UI thread (turn done, model pick, settings) flag it instead of
        rebuilding inline — an inline rebuild there froze the entire screen
        whenever the models cache was stale and the network slow.
        """
        self._rotation_dirty = True

    def _emit_tool(self, kind: str, tool: str, **kwargs: Any) -> None:
        self._emit(kind, tool=tool, **kwargs)

    # -- permission ------------------------------------------------------
    def check_permission(self, tool: str, input_value: str, display: str, call_id: str = "", arguments: dict | None = None, permission: str | None = None, action: str | None = None) -> bool:
        permission_name = permission or tool
        if tool in ("write", "edit", "apply_patch") and permission is None:
            permission_name = "edit"
        if action is None:
            action = self.permission.evaluate(permission_name, input_value)
        if action == "allow":
            return True
        kwargs: dict[str, Any] = {"reason": "denied by permission", "call_id": call_id}
        if action == "deny":
            if arguments is not None:
                kwargs["input"] = arguments
            self._emit_tool("tool_denied", tool, **kwargs)
            return False
        # ask
        # "Always" scope: the specific permission+input being approved, NOT a
        # universal "*" — one approval granted to `bash npm install` must not
        # silently auto-approve every future ask for any tool/command.
        always_patterns: list[str] = [f"{permission_name} {input_value}".strip()]
        # Cross-thread safe: when a step runs several tools in parallel this
        # lock guarantees only one permission/question modal is visible at a
        # time (the TUI shows exactly one dialog at once).
        with self._ask_lock:
            allowed = self.permission.ask(display, always_patterns)
        if not allowed:
            kwargs = {"reason": "rejected by user", "call_id": call_id}
            if arguments is not None:
                kwargs["input"] = arguments
            self._emit_tool("tool_denied", tool, **kwargs)
        return allowed

    # -- tools ------------------------------------------------------------
    def run_tool(self, name: str, arguments: dict, call_id: str = "") -> dict[str, Any]:
        if name == "skill" and not self._skill_enabled():
            return {"output": "Skill tool is disabled for this agent.", "error": True}
        tool = self.registry.get(name)
        if tool is None:
            # Wire names are sanitized (mcp:n -> mcp_n); an old session or a
            # custom plugin may reply with the sanitized form while the
            # registry holds the raw name (or vice versa). Resolve by
            # sanitized comparison before giving up.
            try:
                from ..tools.registry import _sanitize_wire_name as _san

                want = _san(name)
                for cand in self.registry.list():
                    if _san(cand.name) == want:
                        tool = cand
                        break
            except Exception:
                tool = None
            if tool is None:
                return {"output": f"Unknown tool: {name}", "error": True}
        # Relative paths are resolved against the session directory (git
        # worktree root), NOT Path.cwd(): launching opencode from a
        # subdirectory would otherwise make read/write/edit/bash act on the
        # wrong files while the undo snapshots use self.directory. Normalize
        # here so every component agrees. `resolved_path` keeps the canonical
        # target used both for permission matching and worktree confinement.
        resolved_path = None
        for key in ("filePath", "path", "workdir"):
            if key in arguments and isinstance(arguments[key], str) and arguments[key]:
                p = Path(arguments[key])
                if not p.is_absolute():
                    p = self.directory / p
                p = p.resolve()
                arguments[key] = str(p)
                resolved_path = str(p)
                break

        # Permission match value: path-based tools match their rules (e.g.
        # "*.env" -> ask) against the RESOLVED PATH, not the JSON blob of all
        # arguments (which ends in `}` and never matches a path pattern). task
        # matches on the sub-agent type, mirroring opencode. bash matches on the
        # COMMAND STRING — rules like {"bash": {"npm install *": "deny"}} must
        # actually fire, and the JSON dump of arguments would never match any
        # command pattern (it's "{...}", not "npm install ...").
        input_value = json.dumps(arguments, sort_keys=True)
        if name == "task":
            # opencode's task input id is the sub-agent type, so a plan
            # agent's `task: {grant: deny}` rule actually matches.
            input_value = str(arguments.get("subagent_type", "build"))
        elif name == "bash":
            input_value = str(arguments.get("command", ""))
        elif name in ("browser", "phone"):
            # Permission patterns like {"browser": {"snapshot *": ...}} must
            # match the ACTION NAME, not the JSON blob '{"action": ...}'.
            # The trailing ' *' matcher also covers "browser" bare.
            action_name = str(arguments.get("action", "") or "").strip().lower()
            input_value = f"{action_name} {json.dumps(arguments, sort_keys=True)}".strip()
        elif name in ("skill",):
            # Permission patterns like {"skill": {"internal-*": "deny"}} must
            # match the SKILL NAME, not the JSON blob '{"name": ...}' (which
            # never matches a name pattern, so denies silently never fired).
            input_value = str(arguments.get("name", "") or "")
        elif resolved_path is not None and name in ("read", "write", "edit", "apply_patch", "grep", "glob"):
            input_value = resolved_path

        permission = name
        if name in ("write", "edit", "apply_patch"):
            permission = "edit"
        # Worktree confinement: a path-based tool targeting a file OUTSIDE the
        # canonical worktree root goes through the `external_directory`
        # permission (default: ask). The rule existed in the config defaults
        # but was never evaluated, so every absolute path was silently allowed.
        # NOTE: external_directory is an ADDITIONAL gate, NOT a replacement for
        # the tool's own permission — otherwise a plan agent's edit/write "deny"
        # could be bypassed by pointing the tool at an absolute path outside the
        # worktree. Combine both with the strictest action (deny > ask > allow)
        # and gate exactly once, so there is a single permission dialog.
        is_external = (
            resolved_path is not None
            and (
                name in ("read", "write", "edit", "apply_patch", "grep", "glob", "bash", "background_task")
                or name.startswith("mcp__")
            )
            and not Path(resolved_path).is_relative_to(self.worktree)
        )
        if is_external:
            combined = [
                self.permission.evaluate(permission, input_value),
                self.permission.evaluate("external_directory", resolved_path),
            ]
            if getattr(self.permission, "mode", "auto") == "fully_auto":
                action: str | None = "deny" if "deny" in combined else "allow"
            else:
                action = "deny" if "deny" in combined else ("ask" if "ask" in combined else "allow")
        else:
            action = None

        display = f"{name} {input_value[:120]}"
        if is_external:
            # Jail visibility: the tool result names the outside path so the
            # user always sees when the model reached outside the project.
            # Enforcement stays in the permission gate above (ask by default,
            # deny with an explicit rule) — see default_permissions.
            try:
                self._emit(
                    "notice",
                    text=f"{name} outside worktree: {resolved_path}",
                )
            except Exception:
                pass

        if not self.check_permission(name, input_value, display, call_id=call_id, arguments=arguments, permission=permission, action=action):
            return {
                "output": f"Permission denied for {name}. Tell the user what to do differently.",
                "error": True,
                "denied": True,
            }

        mutates = name in ("edit", "write", "apply_patch")
        file_path = arguments.get("filePath")
        snapshot: bytes | None = None
        snapshot_path: Path | None = None
        created_dirs: list[Path] = []
        if mutates and file_path:
            p = Path(file_path)
            if not p.is_absolute():
                p = self.directory / p
            snapshot_path = p
            snapshot = p.read_bytes() if p.exists() else None
            # record which parent directories would be newly created by the
            # write's mkdir(parents=True) so undo can clean them back up.
            if snapshot is None:
                created_dirs = _missing_directories(p.parent)

        # The glob tool resolves a missing `path` against this thread's
        # session worktree (each worker thread gets its own), so parallel
        # globs never fall back to the process CWD.
        glob_mod.set_worktree(self.worktree)

        self._emit_tool("tool_start", name, input=arguments, status="running", call_id=call_id)
        try:
            _pctx = getattr(self.registry, "_progress_ctx", None)
            if _pctx is not None:
                def _emit_progress(done: int, total: int, _n=name, _c=call_id) -> None:
                    try:
                        self._emit_tool("tool_progress", _n, call_id=_c, done=int(done), total=int(total))
                    except Exception:
                        pass
                _pctx.call_id = call_id
                _pctx.emitter = _emit_progress
        except Exception:
            pass
        try:
            if name == "task":
                # let spawn_task tag its subagent_start event with this call id
                self._spawn_ctx.call_id = call_id
                action = str(arguments.get("action") or "start").strip().lower()
                if action in ("status", "read", "stop", "list"):
                    if action == "status":
                        result = self.task_status(str(arguments.get("session_id") or ""))
                    elif action == "read":
                        result = self.task_read(str(arguments.get("session_id") or ""))
                    elif action == "stop":
                        result = self.task_stop(str(arguments.get("session_id") or ""))
                    else:
                        result = self.task_status("")
                else:
                    result = tool.run(arguments)
            else:
                result = tool.run(arguments)
        except Exception as e:
            result = {"output": f"{name} failed: {e}", "error": True}
        if mutates and snapshot_path is not None:
            self._undo_stack.append(
                {
                    "path": str(snapshot_path),
                    "original": snapshot,
                    "dirs": [str(d) for d in created_dirs],
                }
            )
            if len(self._undo_stack) > MAX_UNDO:
                self._undo_stack.pop(0)
        try:
            _pctx2 = getattr(self.registry, "_progress_ctx", None)
            if _pctx2 is not None:
                _pctx2.call_id = ""
                _pctx2.emitter = None
        except Exception:
            pass
        self._emit_tool(
            "tool_complete",
            name,
            input=arguments,
            status="error" if result.get("error") else "completed",
            output=result.get("output", ""),
            metadata=result.get("metadata", {}),
            call_id=call_id,
        )
        return result

    # -- sub-agents -------------------------------------------------------
    def close(self) -> None:
        """Release OS resources held by this engine (MCP server processes).

        Called when an engine (or a finished sub-agent) is done; the parent
        keeps its own refcount on shared servers, so a child's close only
        drops the reference the sub-agent acquired.
        """
        try:
            self.task_stop("")
        except Exception:
            pass
        self._shutdown_tool_pool()
        registry = getattr(self, "registry", None)
        close = getattr(registry, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass

    def abort(self) -> None:
        """Force-close the active provider stream(s) for this engine and every
        sub-agent it spawned.

        Used right after the shared interrupt flag is flipped so a blocked
        stream read wakes up immediately — even during an idle "thinking" gap
        where no chunk arrives to trigger the per-chunk interrupt check."""
        rotation = getattr(self, "rotation", None)
        if getattr(rotation, "abort", None) is not None:
            try:
                rotation.abort()
            except Exception:
                pass
        # Force-close any webfetch responses this engine has in flight, so a
        # blocked HTTP read surfaces the interrupt right away instead of letting
        # the fetch run to its timeout after the user pressed ESC twice.
        registry = getattr(self, "registry", None)
        abort_fetches = getattr(registry, "abort_fetches", None)
        if callable(abort_fetches):
            try:
                abort_fetches()
            except Exception:
                pass
        abort_servers = getattr(registry, "abort_servers", None)
        if callable(abort_servers):
            try:
                abort_servers()
            except Exception:
                pass
        # detached workers check the parent flag via their own override, so
        # abort them explicitly too (their streams would otherwise keep
        # burning tokens until the next chunk check).
        try:
            with self._bg_lock:
                for entry in list(self._bg_agents.values()):
                    try:
                        entry.get("stop_flag", {})["stopped"] = True
                    except Exception:
                        pass
                    try:
                        rot = getattr((entry.get("bundle") or {}).get("sub"), "rotation", None)
                        if getattr(rot, "abort", None) is not None:
                            rot.abort()
                    except Exception:
                        pass
        except Exception:
            pass
        subagents = getattr(self, "subagents", {}) or {}
        for sub in tuple(subagents.values()):
            try:
                sub.abort()
            except Exception:
                pass

    def _subagent_bridge(self, sub_id: str) -> Callable[[dict], None]:
        """Forward a sub-agent's events to our own on_event, tagged with the
        sub-session id so the UI can route them to the right chat view.

        Nested sub-agents already carry their own `session_id` (tagged by the
        deeper bridge); keep that id so a grandchild's events reach its own chat
        instead of being re-tagged with the direct child's id.
        """

        def forward(event: dict[str, Any]) -> None:
            kind = event.get("kind", "")
            sid = event.get("session_id") or sub_id
            payload = {k: v for k, v in event.items() if k not in ("kind", "session_id")}
            self._emit(kind, session_id=sid, **payload)

        return forward

    @staticmethod
    def _child_rotation(provider_factory: Callable[[], Any], session_id: str) -> Any:
        """Build a child's rotation, rebound to the CHILD session id.

        The default factory closes over the parent session; without rebinding,
        every parallel child shares one Zen upstream lane. Rotations built by
        custom factories (tests) are rebound too when they expose session_id.
        Never raises: falls back to a fresh rotation on factory failure.
        """
        try:
            rotation = provider_factory()
        except Exception:
            rotation = None
        if rotation is not None and getattr(rotation, "session_id", None) is not None:
            try:
                rotation.session_id = session_id
                return rotation
            except Exception:
                pass
            return rotation
        return rotation

    def _prepare_task(
        self, arguments: dict[str, Any]
    ) -> tuple[dict | None, dict | None]:
        """Validate + build a child AgentLoop WITHOUT running it.

        Returns (bundle, error): exactly one is not None. Shared by the
        inline path (run now via _finish_task) and the background path
        (run on a worker thread, poll later).
        """
        from ..tools import build_registry

        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return None, {"output": "task: no prompt provided.", "error": True}
        description = str(arguments.get("description", "")).strip() or "sub-agent"
        requested_type = str(arguments.get("subagent_type", "")).strip() or "build"
        # Launch what was asked: an unknown name is a model mistake, not a
        # build request. Fail loudly with the valid names so the model
        # retries correctly — silently running build with full tools when
        # a scoped specialist was intended is the dangerous wrong choice.
        try:
            from ..permission import list_agents as _la

            known = {n for n, _d, _c in _la(self.cfg)} | {"general"}
        except Exception:
            known = {"build", "plan", "explore", "general"}
        if requested_type not in known:
            return None, {
                "output": (
                    f"Unknown subagent_type '{requested_type}'. "
                    f"Valid: {', '.join(sorted(known))}."
                ),
                "error": True,
            }
        subagent_type = requested_type

        # Enforce cfg.subagent_depth: the number of sub-agent LEVELS allowed
        # below the main agent (default 1). Without this the setting was dead:
        # a model could recurse task -> task -> task without bound, each level
        # minting a new session + rotation + registry on a low-RAM phone.
        child_depth = self._depth + 1
        allowed = max(0, int(getattr(self.cfg, "subagent_depth", 1)))
        if child_depth > allowed:
            return None, {
                "output": (
                    f"sub-agent depth limit reached (cfg.subagent_depth={allowed}):"
                    " this agent is already at the deepest allowed level."
                    " Do the work directly instead of delegating."
                ),
                "error": True,
            }

        # A read-only agent (plan/explore, or a custom with readonly) must
        # not spawn a mutating sub-agent: the child would inherit
        # read-write tools. Force the child to stay read-only too.
        from ..permission import agent_readonly as _is_ro2

        try:
            _ro = bool(_is_ro2(self.cfg, self.agent))
        except Exception:
            _ro = self.agent in ("plan", "explore")
        if _ro and subagent_type not in ("plan", "explore"):
            try:
                _child_ro = bool(_is_ro2(self.cfg, subagent_type))
            except Exception:
                _child_ro = False
            if not _child_ro:
                subagent_type = "plan" if self.agent == "plan" else "explore"

        # Bound simultaneously running children (phone RAM): overflow tells
        # the model to do the work directly instead of OOMing.
        if not self._subagent_sem.acquire(blocking=False):
            return None, {
                "output": (
                    f"too many sub-agents running (max {self._MAX_CONCURRENT_SUBAGENTS}):"
                    " wait for one to finish, or do the work directly instead of delegating."
                ),
                "error": True,
            }

        sub_session = new_session(
            directory=str(self.directory),
            provider=self.provider_id or self.cfg.provider,
            model=self.model_id or self.cfg.model,
            agent=subagent_type,
            title=description,
            parent_id=self.session_id,
        )
        # No save here: the sub-agent session has no conversation yet and must
        # not mint an empty file. It is persisted only once it has content (the
        # completion saves below), or by the app's exit/close save-all.

        sub = AgentLoop(
            cfg=self.cfg,
            registry=build_registry(self.cfg),
            directory=self.directory,
            # The child gets its OWN Zen session identity: sharing the
            # parent's id pinned every parallel child to the same upstream
            # lane (correlated 429s), and one child's rotation poisoned the
            # siblings' lane via the global epoch. The factory seam stays
            # intact for tests — we just rebind the built rotation to the
            # child session. Same for the grandchild factory, so nesting
            # derives from the child, not the grandparent.
            provider=self._child_rotation(self.provider_factory, sub_session.id),            auth=self.auth,
            permission_engine=self.permission,
            question_service=self.question_service,
            on_event=self._subagent_bridge(sub_session.id),
            ask_lock=self._ask_lock,
            agent=subagent_type,
            provider_id=self.provider_id,
            model_id=self.model_id,
            interrupt=self.interrupt,
            session_id=sub_session.id,
            provider_factory=lambda: build_rotation(self.cfg, self.auth, sub_session.id),
            depth=child_depth,
        )
        self.subagents[sub_session.id] = sub
        bundle = {
            "sub": sub,
            "session": sub_session,
            "prompt": prompt,
            "description": description,
            "agent": subagent_type,
            "started": time.time(),
        }
        self._emit(
            "subagent_start",
            session_id=sub_session.id,
            agent=subagent_type,
            title=description,
            prompt=prompt,
            call_id=getattr(self._spawn_ctx, "call_id", None) or "",
        )
        return bundle, None

    def spawn_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a sub-agent in its own session; returns the sub-agent's reply.

        The sub-agent gets its own history/session but shares the parent's
        config, auth, permission engine, directory, and interrupt flag. Events
        stream out tagged with the sub-session id. Nested `task` calls work:
        sub-agents build their own registry, which lazily resolves the same hook.

        With background=true the child runs detached: this returns a session
        id immediately and the parent keeps working; use task_status /
        task_read / task_stop (same verbs as background_task) to poll,
        collect, or kill it. At most ONE background agent at a time.
        """
        bg = arguments.get("background")
        if bg is True or str(bg or "").strip().lower() in ("1", "true", "yes", "on"):
            return self._spawn_task_background(arguments)
        bundle, error = self._prepare_task(arguments)
        if error is not None:
            return error
        assert bundle is not None
        return self._finish_task(bundle)

    # -- background sub-agents (fire-and-keep-working) -------------------
    # At most ONE detached child: each holds history + registry + rotation
    # in RAM and burns its own context window. The model starts it, keeps
    # working, and collects the reply later with task_read (or task_status
    # to peek, task_stop to kill). Unread results are folded into history
    # at turn end so nothing is silently lost.
    _MAX_BG_AGENTS = 1

    def _release_task_slot(self, bundle: dict) -> None:
        """Free one _subagent_sem slot exactly once per bundle.

        Inline and background paths both end in _finish_task; a killed
        background worker still completes through _finish_task later.
        Without the flag, stop-then-finish would release twice and grow
        the semaphore past its cap, silently allowing extra agents.
        """
        try:
            if bundle.get("_slot_released"):
                return
            bundle["_slot_released"] = True
            self._subagent_sem.release()
        except Exception:
            pass

    def _bg_live(self) -> dict[str, dict[str, Any]] | None:
        """The running detached entry, or None. Holds _bg_lock."""
        with self._bg_lock:
            for sid, entry in self._bg_agents.items():
                if entry.get("running"):
                    return entry
        return None

    def _spawn_task_background(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Start a child detached; return its session id immediately."""
        if self._bg_live() is not None:
            return {
                "output": (
                    "A background agent is already running. Collect it with "
                    "task_read (or stop it with task_stop) before starting another."
                ),
                "error": True,
            }
        bundle, error = self._prepare_task(arguments)
        if error is not None:
            return error
        assert bundle is not None
        sub = bundle["sub"]
        sub_session = bundle["session"]
        # A detached child needs its OWN interrupt flag: the parent's flag
        # would kill it on the next user Ctrl+C even though the user only
        # meant to stop the parent. task_stop flips this one instead.
        stop_flag = {"stopped": False}
        parent_interrupt = self.interrupt
        sub.interrupt = lambda: bool(stop_flag["stopped"]) or parent_interrupt()
        entry: dict[str, Any] = {
            "sid": sub_session.id,
            "bundle": bundle,
            "running": True,
            "result": None,
            "stop_flag": stop_flag,
            "started": time.time(),
            "thread": None,
        }
        with self._bg_lock:
            self._bg_agents[sub_session.id] = entry

        def _run() -> None:
            try:
                entry["result"] = self._finish_task(bundle)
            except Exception as e:  # never leave a zombie entry
                entry["result"] = {"output": f"background agent failed: {e}", "error": True}
            finally:
                entry["running"] = False

        thread = threading.Thread(target=_run, name=f"bg-agent-{sub_session.id[:8]}", daemon=True)
        entry["thread"] = thread
        thread.start()
        return {
            "output": (
                f"Started background agent {sub_session.id} ({bundle['description']}).\n"
                f"Keep working — poll with task_status, collect with task_read, "
                f"kill with task_stop."
            ),
            "metadata": {"sessionId": sub_session.id, "background": True},
        }

    def _bg_get(self, session_id: str) -> dict[str, Any] | None:
        with self._bg_lock:
            return self._bg_agents.get((session_id or "").strip())

    def _bg_collect(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Blocking collect: wait for the worker, fold reply into history
        as a tool result would, drop the entry. Returns the child result."""
        thread = entry.get("thread")
        if thread is not None:
            while thread.is_alive():
                if self.interrupt():
                    break
                # A newly typed user message breaks the wait (see
                # wake_bg_wait): the turn ends, the message starts its own
                # turn at once instead of QUEUED, the agent keeps running.
                wake = getattr(self, "_bg_wake", None)
                if wake is not None:
                    try:
                        if wake.is_set():
                            wake.clear()
                            return {
                                "output": (
                                    "Background agent still running — "
                                    "collect it later with task_read."
                                ),
                                "metadata": {
                                    "sessionId": entry.get("sid", ""),
                                    "background": True,
                                    "still_running": True,
                                },
                            }
                    except Exception:
                        pass
                thread.join(timeout=0.2)
        with self._bg_lock:
            self._bg_agents.pop(entry.get("sid", ""), None)
        result = entry.get("result") or {"output": "(no reply from sub-agent)", "error": True}
        return result

    def wake_bg_wait(self) -> bool:
        """Wake a parent parked in task_read so a fresh user message can
        start its own turn immediately. Returns True if a wait was broken
        (the bg agent keeps running; its reply folds in later).

        Also arms end-of-step: the current tool step completes, then the
        drain exits instead of starting new provider steps.
        """
        try:
            wake = getattr(self, "_bg_wake", None)
            if wake is None:
                import threading as _th

                wake = _th.Event()
                self._bg_wake = wake
            with self._bg_lock:
                parked = any(e.get("running") for e in self._bg_agents.values())
            if not parked:
                return False
            wake.set()
            try:
                self._bg_end_turn = True
            except Exception:
                pass
            return True
        except Exception:
            return False

    def task_status(self, session_id: str = "") -> dict[str, Any]:
        """Peek at detached agent(s): running/finished, runtime, title."""
        with self._bg_lock:
            entries = list(self._bg_agents.values())
            if session_id:
                entries = [e for e in entries if e.get("sid") == session_id.strip()]
        if not entries:
            return {"output": "No background agents."}
        lines = []
        for e in entries:
            state = "RUNNING" if e.get("running") else "FINISHED"
            lines.append(
                f"  {e.get('sid', '?')}  {state:<9} {time.time() - e.get('started', time.time()):6.1f}s  "
                f"{str((e.get('bundle') or {}).get('description', ''))[:60]}"
            )
        return {"output": f"{len(entries)} background agent(s):\n" + "\n".join(lines)}

    def task_read(self, session_id: str = "") -> dict[str, Any]:
        """Collect a detached agent's reply (waits if still running)."""
        entry = None
        if session_id:
            entry = self._bg_get(session_id)
            if entry is None:
                return {"output": f"No such background agent {session_id!r}.", "error": True}
        else:
            entry = self._bg_live()
            if entry is None:
                with self._bg_lock:
                    leftovers = list(self._bg_agents.values())
                if not leftovers:
                    return {"output": "No background agents."}
                entry = leftovers[0]
        return self._bg_collect(entry)

    def task_stop(self, session_id: str = "") -> dict[str, Any]:
        """Kill a detached agent (or all) without collecting."""
        with self._bg_lock:
            if session_id:
                targets = [self._bg_agents.get(session_id.strip())]
                targets = [t for t in targets if t is not None]
            else:
                targets = list(self._bg_agents.values())
        if not targets:
            return {"output": "No background agents."}
        stopped = 0
        for entry in targets:
            try:
                entry.get("stop_flag", {})["stopped"] = True
            except Exception:
                pass
            thread = entry.get("thread")
            if thread is not None:
                thread.join(timeout=5)
            with self._bg_lock:
                self._bg_agents.pop(entry.get("sid", ""), None)
            stopped += 1
        # NOTE: no _subagent_sem.release() here — the worker still owns its
        # slot until _finish_task runs (or _release_task_slot fires once via
        # the bundle flag). Releasing here too would over-release the cap.
        return {"output": f"Stopped {stopped} background agent(s)."}

    def _fold_finished_bg_agents(self) -> None:
        """Fold finished-but-uncollected detached replies into history.

        Runs at turn end: a background agent that completed while the
        parent kept working must not vanish silently. Its reply is
        appended as an assistant note (capped length) so the NEXT turn
        sees it even if the model never called task_read. Still-running
        agents are left alone.
        """
        with self._bg_lock:
            done = [e for e in self._bg_agents.values() if not e.get("running")]
        for entry in done:
            self._fold_one_bg_agent(entry)

    def _fold_one_bg_agent(self, entry: dict, result: dict | None = None) -> None:
        """Fold a single finished entry into history + emit bg_collected.

        `result` overrides entry["result"]: at worker-completion time the
        entry doesn't hold it yet (it's still being computed), so the
        caller passes the fresh dict. Turn-end fold calls without it.
        """
        result = result if result is not None else (entry.get("result") or {})
        text = str(result.get("output", "") or "")[:2000]
        title = str((entry.get("bundle") or {}).get("description", "sub-agent"))
        try:
            self._history.append({
                "role": "assistant",
                "content": f"[background agent finished: {title}]\n{text}",
            })
        except Exception:
            pass
        with self._bg_lock:
            self._bg_agents.pop(entry.get("sid", ""), None)
        self._emit("bg_collected", session_id=entry.get("sid", ""), title=title)

    def _finish_task(self, bundle: dict) -> dict:
        """Run a prepared child NOW and return its reply (inline path)."""
        sub = bundle["sub"]
        sub_session = bundle["session"]
        prompt = bundle["prompt"]
        description = bundle["description"]
        subagent_type = bundle["agent"]
        task_started = bundle["started"]
        try:
            result = sub.run_turn(prompt)
        except BaseException as e:
            # never leave the sub-agent session dangling: persist and report done.
            # BaseException (not just Exception) so Ctrl+C/SystemExit also
            # releases the semaphore, pops the child, and closes server refs —
            # otherwise 4 interrupts exhaust the pool and every future task
            # fails with "too many sub-agents".
            try:
                sub_session.messages = sub.get_history()
            except Exception:
                pass
            sub_session.completed = time.time()
            try:
                if sub_session.messages:
                    save_session(sub_session)
            except Exception:
                pass
            try:
                sub.close()
            except Exception:
                pass
            self._emit(
                "subagent_done",
                session_id=sub_session.id,
                agent=subagent_type,
                title=description,
                ok=False,
            )
            self._release_task_slot(bundle)
            self.subagents.pop(sub_session.id, None)
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            return {
                "output": f"sub-agent failed: {e}",
                "error": True,
                "metadata": {
                    "sessionId": sub_session.id,
                    "title": description,
                    "status": "error",
                    "agent": subagent_type,
                },
            }
        sub_session.messages = sub.get_history()
        sub_session.completed = time.time()
        try:
            if sub_session.messages:
                save_session(sub_session)
        except Exception:
            pass
        sub.close()

        # Stamp the child link onto the pending assistant message NOW (not
        # only at history-append): an interrupt between spawn and append used
        # to leave the row permanently unlinked for all future resumes.
        try:
            for m in reversed(self._history):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    m.setdefault("task_children", {})[str(sub_session.id)] = {
                        "title": description, "agent": subagent_type}
                    break
        except Exception:
            pass
        # Completion detail travels WITH the task result (engine-level truth):
        # wall-clock runtime + the child's real toolcall count. The TUI used
        # to derive both from live widgets (start-time map + child chat) that
        # were already popped/unmounted when it read them — so finished rows
        # showed nothing but the `↓ ctrl+down` hint. Stamping here makes the
        # `↳ N toolcalls · Xs` footer race-free and resume-proof.
        try:
            child_calls = int(getattr(result, "tool_calls_made", 0) or 0)
        except Exception:
            child_calls = 0
        task_duration = max(0.0, time.time() - task_started)
        # Was this child detached? Snapshot BEFORE any fold/collect pops
        # the entry: the TUI needs it to decide row teardown vs "done,
        # unread" (Fix B). bg_collected mirrors whether the reply was
        # already folded/collected at emit time.
        try:
            with self._bg_lock:
                _bg_entry_now = self._bg_agents.get(sub_session.id)
            _was_bg = _bg_entry_now is not None
        except Exception:
            _was_bg = False
        self._emit(
            "subagent_done",
            session_id=sub_session.id,
            agent=subagent_type,
            title=description,
            ok=not result.error,
            duration=task_duration,
            toolcalls=child_calls,
            was_background=_was_bg,
        )
        text = result.text or result.error or "(no reply from sub-agent)"
        self._release_task_slot(bundle)
        # Free the finished child (history + registry + rotation): the TUI
        # keeps its own transcript; holding it here too doubles peak RAM per
        # fan-out on low-memory phones.
        self.subagents.pop(sub_session.id, None)
        # Background worker completion: if the parent has NO turn running
        # (the idle case — the whole point of background agents), fold the
        # reply into history NOW so the model learns it done immediately
        # instead of waiting for a turn end that may never come. When a
        # parent turn IS active, turn-end fold owns it (avoids mid-step
        # history mutation under the running loop). Pass the fresh result:
        # entry["result"] isn't set yet (we're still computing it).
        try:
            with self._bg_lock:
                bg_entry = self._bg_agents.get(sub_session.id)
            if bg_entry is not None and not getattr(self, "_turn_active", False):
                self._fold_one_bg_agent(bg_entry, {
                    "output": text,
                    "error": bool(result.error and not result.text),
                })
        except Exception:
            pass
        return {
            "output": text,
            "error": bool(result.error and not result.text),
            "metadata": {
                "sessionId": sub_session.id,
                "title": description,
                "status": "error" if result.error else "completed",
                "duration_s": task_duration,
                "toolcalls": child_calls,
                # resolved agent actually launched (may differ from the
                # requested input after read-only forcing) so the row
                # renders what ran, not what was asked.
                "agent": subagent_type,
            },
        }

    _PARALLEL_TOOL_WORKERS = 6
    _TOOL_TIMEOUT = 120.0
    _MUTATING_TOOLS = ("edit", "write", "apply_patch")
    # Cap on simultaneously RUNNING sub-agents per parent: each live child
    # holds a full history + registry + rotation + (in the TUI) a chat view,
    # so unbounded model fan-out OOMs low-RAM phones. Overflow degrades to a
    # guidance error (do the work directly) instead of a crash.
    _MAX_CONCURRENT_SUBAGENTS = 4

    def _tool_pool_locked(self):
        """Persistent worker pool shared by all steps of this agent."""
        from concurrent.futures import ThreadPoolExecutor

        with self._tool_pool_lock:
            pool = self._tool_pool
            if pool is None:
                pool = ThreadPoolExecutor(
                    max_workers=self._PARALLEL_TOOL_WORKERS,
                    thread_name_prefix="opencode-tools",
                )
                self._tool_pool = pool
            return pool

    def _shutdown_tool_pool(self) -> None:
        with self._tool_pool_lock:
            pool, self._tool_pool = self._tool_pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _run_tools_parallel(self, prepared):
        """Run every tool call of a step concurrently and return their results
        in the original call order.

        `prepared` is a list of ``(call, name, arguments)`` triples. Reads and
        other independent calls run concurrently on worker threads — `task` calls
        therefore spawn and run their sub-agents at the same time (official
        opencode's parallel agent fan-out). MUTATING tools (edit/write/apply_patch)
        are kept strictly sequential, in call order, so two writes to the same
        file can't interleave their read-modify-write and their undo snapshots
        can't race. Permission prompts stay serialized via ``_ask_lock`` so the
        TUI never shows stacked dialogs.
        """
        from concurrent.futures import TimeoutError as FuturesTimeout
        from concurrent.futures import wait

        def execute(index: int, call: dict, name: str, arguments: dict):
            if self.interrupt():
                return index, call, name, {"output": "(interrupted)", "error": True, "stopped": True}
            try:
                tool_result = self.run_tool(name, arguments, call_id=call.get("id", ""))
            except Exception as e:  # run_tool already guards, but be safe
                tool_result = {"output": f"{name} failed: {e}", "error": True}
            return index, call, name, tool_result

        ordered: list[tuple[dict, str, dict] | None] = [None] * len(prepared)
        pool = self._tool_pool_locked()
        if pool is None:
            # pool exhausted at shutdown: run inline, sequentially
            for i, (call, name, arguments) in enumerate(prepared):
                _, _, _, tool_result = execute(i, call, name, arguments)
                ordered[i] = (call, name, tool_result)
            return [o for o in ordered if o is not None]
        try:
            futures = {}
            for i, (call, name, arguments) in enumerate(prepared):
                if name not in self._MUTATING_TOOLS:
                    futures[pool.submit(execute, i, call, name, arguments)] = i
            # Mutating calls execute on this thread one at a time, in original
            # order (strictly serialized; never concurrent with one another).
            for i, (call, name, arguments) in enumerate(prepared):
                if name in self._MUTATING_TOOLS:
                    _, _, _, tool_result = execute(i, call, name, arguments)
                    ordered[i] = (call, name, tool_result)
            pending = set(futures)
            while pending:
                if self.interrupt():
                    for f in pending:
                        f.cancel()
                    # Mark remaining as interrupted (like original behavior)
                    for f, idx in futures.items():
                        if ordered[idx] is None:
                            # Don't call result() on cancelled futures - construct directly
                            call = prepared[idx][0]  # original call dict
                            name = prepared[idx][1]  # original tool name
                            ordered[idx] = (call, name, {"output": "(interrupted)", "error": True, "stopped": True})
                    break
                done, pending = wait(pending, timeout=0.5)
                for fut in done:
                    index, call, name, tool_result = fut.result()
                    ordered[index] = (call, name, tool_result)
            for fut, idx in futures.items():
                if ordered[idx] is None:
                    # Interruptible join: a bare fut.result() here blocked
                    # forever on a hung tool, making ESC look ignored.
                    while True:
                        if self.interrupt():
                            call = prepared[idx][0]
                            name = prepared[idx][1]
                            ordered[idx] = (call, name, {"output": "(interrupted)", "error": True, "stopped": True})
                            break
                        try:
                            index, call, name, tool_result = fut.result(timeout=0.5)
                        except FuturesTimeout:
                            continue
                        ordered[index] = (call, name, tool_result)
                        break
        except RuntimeError:
            # pool shut down mid-step (engine closed): finish inline
            for i, (call, name, arguments) in enumerate(prepared):
                if ordered[i] is None:
                    _, _, _, tool_result = execute(i, call, name, arguments)
                    ordered[i] = (call, name, tool_result)
        return [o for o in ordered if o is not None]

    # -- main turn --------------------------------------------------------
    def run_turn(self, user_text: str) -> TurnResult:
        self._history.append({"role": "user", "content": user_text})
        self._turn_active = True
        try:
            return self._run_turn_body(user_text)
        finally:
            try:
                self._turn_active = False
            except Exception:
                pass

    def resume_turn(self) -> TurnResult:
        """Re-run the last user prompt after a failed turn (e.g. the network
        died mid-stream and is back now).

        Drops everything after the last user message (partial assistant text,
        tool results of the dead turn — the model regenerates them cleanly)
        and re-runs WITHOUT appending a duplicate prompt. Returns an error
        result when there is no user message to resume.
        """
        idx = -1
        for i, m in enumerate(self._history):
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    idx = i
        if idx < 0:
            result = TurnResult()
            result.error = "nothing to resume"
            return result
        user_text = self._history[idx]["content"]
        del self._history[idx + 1:]
        return self._run_turn_body(user_text)

    def _run_turn_body(self, user_text: str) -> TurnResult:
        result = TurnResult()

        # Honor a deferred rotation rebuild (model pick / settings change /
        # end-of-turn lane switch): the network-touching rebuild stays OFF
        # the UI thread by running here, before anything streams.
        if getattr(self, "_rotation_dirty", False):
            self._rotation_dirty = False
            try:
                self.rebuild_rotation()
            except Exception:
                pass

        # Stable session across turns (official parity): Rotation.new_turn()
        # is now a no-op that keeps the same x-opencode-session so Zen holds
        # its sticky healthy lane. Dead lanes still rotate via
        # _rotate_lane()/rotate_session() on real failures.
        try:
            rotation = getattr(self, "rotation", None)
            new_turn = getattr(rotation, "new_turn", None)
            if callable(new_turn):
                new_turn()
        except Exception:
            pass

        reset = getattr(self.permission, "reset_doom_tracking", None)
        if reset:
            reset()

        # agent reminder (plan/build-switch)
        reminder = system_mod.agent_reminder(self.agent, self._was_plan(), self.cfg)
        system_prompt = system_mod.build_system_prompt(
            directory=self.directory,
            worktree=self.worktree,
            provider_id=self.provider_id,
            model_id=self.model_id,
            cfg=self.cfg,
            agent=self.agent,
        )

        history = self.request_history()
        messages = msg_mod.build_messages(history=history[:-1], user_text=user_text, reminder=reminder)
        messages = self._prepend_system(messages, system_prompt)

        # Stable tool-schema order (fix 3): deterministic names order so the
        # request prefix (system + schemas) is byte-stable across turns for
        # providers with prefix caching. No behavior change otherwise.
        tools = self._active_tool_schemas()
        try:
            tools = sorted(tools, key=lambda s: str((s.get("function") or {}).get("name", "")))
        except Exception:
            pass

        # The selected model's real window drives both compaction and the
        # trimming safety net, so this works for ANY model — not just the
        # bundled free ones. An unknown window (0) falls back to the configured
        # hard budget and lets the post-overflow recovery path compact.
        ctx = self._model_context_size()
        output_limit = self._model_output_limit()
        usable = compact_mod.usable_context(ctx, output_limit) if ctx > 0 else 0

        # Fix 1: shrink OLD tool outputs for the request copy (history on disk
        # stays full). The last keep_turns go verbatim; older tool bodies
        # become one-line receipts. Pairing/ids untouched — safe to replay.
        try:
            keep_turns = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
        except (TypeError, ValueError):
            keep_turns = 2
        try:
            trim_chars = int(getattr(self.cfg, "trim_max_chars", 500) or 500)
        except (TypeError, ValueError):
            trim_chars = 500
        if keep_turns > 0:
            try:
                messages = trim_mod.shrink_tool_history(messages, keep_turns, trim_chars)
            except Exception:
                pass

        # Fix 2: adopt a ready early summary (built quietly behind at ~70%).
        # It replaces the SAME head the overflow path would summarize, so the
        # turn continues with zero extra model calls instead of freezing.
        try:
            adopted = self._adopt_early_summary(system_prompt)
            if adopted is not None:
                messages = adopted
        except Exception:
            pass

        # Proactive compaction (mirrors upstream `compactIfNeeded`): estimate the
        # request about to be sent — system + messages + tools — and compact it
        # before it ever reaches the provider's length limit. This runs on the
        # UNTRIMMED history so compaction (which preserves the conversation via
        # the anchored summary) gets first pick; the trim below is only a
        # last-resort safety net that may drop old turns. The estimate alone can
        # undercount tool-heavy conversations, so ALSO honor the actual usage
        # reported by the provider for the last completion (persists across
        # turns on the loop) — mirrors opencode's `lastFinished.tokens` check.
        if self.cfg.compaction_enabled and usable > 0:
            overflow = compact_mod.is_overflow(ctx, compact_mod.estimate_request(messages, tools), output_limit)
            if not overflow and self._usage_total:
                actual = self._usage_total.get("total_tokens") or usage_total_dict(self._usage_total)
                overflow = actual >= usable
            if overflow:
                compacted = self._compact_context(system_prompt)
                if compacted is not None:
                    messages = compacted

        # Kick the early summary for the NEXT turn while this one streams
        # (fix 2): at >=70% of usable it pre-builds the summary behind, so a
        # later overflow adopts it instead of freezing the turn.
        try:
            self._maybe_kick_early_summary(system_prompt, messages, tools, ctx, output_limit, usable)
        except Exception:
            pass

        # Last-resort cap: never send past the model's usable window (or its raw
        # context when the reserve can't be sized, or the configured budget when
        # the window is unknown). After compaction this is normally a no-op.
        # Runs on the SHRUNK request so the estimate reflects what's sent.
        trim_budget = usable if usable > 0 else (ctx if ctx > 0 else self.cfg.context_budget)
        messages = msg_mod.trim_history(messages, trim_budget)

        turn_start_calls = int(getattr(result, "tool_calls_made", 0) or 0)
        for step in range(MAX_STEPS):
            if self.interrupt():
                self._emit("interrupted")
                break

            # A queued prompt invalidates any pre-built summary (it was made
            # for the old history) — the worker's length key won't match, and
            # we clear here too so nothing stale can be adopted.
            # opencode's queue-and-promote: if a prompt was queued while a
            # provider turn ran, fold it into this SAME drain at the next
            # provider-turn boundary — the generation "pauses for a beat", the
            # chat enters context as a continuation, then the model reasons
            # about it (keep working / stop, whatever it asks). Mirror the
            # official runLoop, which only exits when the last assistant's
            # parent IS the last user message, so a promoted prompt keeps the
            # drain going. Only after the first turn (step>0) — the initial
            # prompt is already in `messages`, and folding before the first
            # stream would merge two user messages.
            if step > 0:
                queued = self._next_prompt()
                if queued is not None:
                    self._history.append({"role": "user", "content": queued})
                    self._emit("prompt_promoted", text=queued)
                    try:
                        with self._early_summary_lock:
                            self._early_summary_text = ""
                            self._early_summary_at = -1
                            self._early_summary_head = {}
                    except Exception:
                        pass
                    # shrink the promoted rebuild too (same receipt rules)
                    try:
                        _kt = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
                        _mc = int(getattr(self.cfg, "trim_max_chars", 500) or 500)
                        messages = trim_mod.shrink_tool_history(list(self._history), _kt, _mc)
                    except Exception:
                        messages = list(self._history)

            self._emit("step", step=step)
            # stream
            self._stream(messages, tools, result, system_prompt)

            if result.error:
                break

            # collect tool calls
            if self._pending_calls:
                calls = self._pending_calls
                self._pending_calls = []
                # Drop degenerate calls (missing name); a model that only emits
                # empty tool calls would otherwise spin a silent 50-step loop.
                calls = [c for c in calls if c.get("name")]
                if not calls:
                    result.error = "model produced an invalid tool call (missing name)"
                    self._emit("error", error=result.error)
                    break
            elif getattr(self.cfg, "auto_continue", False) and not self.interrupt() and step < MAX_STEPS - 1 and self.prompt_pending() == 0 and sum(1 for m in self._history if isinstance(m, dict) and m.get("content") == AUTO_CONTINUE_MSG) < AUTO_CONTINUE_MAX_NUDGES:
                # auto_continue ON: text-only stop is not the end.
                # Rule: real work + real answer -> stop. Anything idle -> nudge.
                # - tools ran this turn AND this step has non-empty text:
                #   the final summary landed (your whoami fix target) -> stop.
                # - empty text (model went silent, incl. right after tools):
                #   wait a beat, send the nudge, run another provider step in
                #   THIS same drain so the summary always lands.
                # - no tools at all + chatter: idle -> nudge (never-stop mode).
                # Real queued message wins; OFF stops as before.
                worked = int(getattr(result, "tool_calls_made", 0) or 0) > turn_start_calls
                final_text = (getattr(result, "text", "") or "").strip()
                if worked and final_text:
                    if self.prompt_pending() == 0:
                        break
                    continue
                # Repeat guard (new-session safety): model echoing the identical
                # text as its PREVIOUS turn is chatting, not working — stop
                # instead of burning all 25 nudges on repeats. Compares the two
                # most recent assistant texts (the current reply is already in
                # history, so one match alone is not a repeat).
                if final_text:
                    try:
                        found: list[str] = []
                        for m in reversed(self._history):
                            if isinstance(m, dict) and m.get("role") == "assistant":
                                c = m.get("content")
                                if isinstance(c, str) and c.strip():
                                    found.append(c.strip())
                                    if len(found) >= 2:
                                        break
                        if len(found) >= 2 and found[0] == found[1] == final_text:
                            if self.prompt_pending() == 0:
                                break
                            continue
                    except Exception:
                        pass
                try:
                    self._emit("auto_continue", text=AUTO_CONTINUE_MSG)
                except Exception:
                    pass
                deadline = time.monotonic() + AUTO_CONTINUE_WAIT_S
                while time.monotonic() < deadline:
                    if self.interrupt() or self.prompt_pending() > 0:
                        break
                    time.sleep(0.1)
                if self.interrupt():
                    break
                if self.prompt_pending() > 0:
                    continue
                self._history.append({"role": "user", "content": AUTO_CONTINUE_MSG})
                messages = list(self._history)
                continue
            else:
                # The model replied with text but made no tool call — a
                # complete answer for the current prompt. If a prompt was
                # queued meanwhile, don't exit: the loop-top fold on the next
                # iteration promotes it into this same drain (the stream "pauses
                # a beat", the chat enters context, then the model reasons).
                if self.prompt_pending() == 0:
                    break
                continue

            # Some models emit tool calls without an id. Assign a stable
            # fallback (must match assistant_message_from_calls) so the
            # assistant declaration and the following tool-result messages use
            # the same id; otherwise strict OpenAI-compatible backends reject
            # with "insufficient tool messages following tool_calls". The
            # counter keeps ids unique across every step and turn so the UI can
            # reliably match tool rows by call_id (per-step indices collide and
            # cause duplicate/incorrect tool rows).
            for call in calls:
                if not call.get("id"):
                    self._call_seq += 1
                    call["id"] = f"call_{self._call_seq}"

            # append assistant message with calls to history — keep the model's
            # own text and reasoning in the same message so the next tool-loop
            # request is a faithful replay (reasoning models lose thread when
            # their previous message is stored empty).
            self._history.append(
                parse_mod.assistant_message_from_calls(
                    calls,
                    reasoning=result.reasoning,
                    content=result.text,
                )
            )
            messages = list(self._history)

            # Execute the step's tool calls. A step with several calls runs them
            # concurrently (official opencode launches sub-agents in parallel
            # when the model emits multiple `task` calls in one reply). Each
            # call runs on its own worker; results come back on this thread and
            # history is written in the ORIGINAL call order so the
            # assistant_message -> tool_result pairing stays valid for replay.
            # No early-return may happen between committing the assistant
            # message and writing its tool results: strict backends reject a
            # request whose assistant tool_calls are not all answered by tool
            # messages ("insufficient tool messages following tool_calls"), and
            # the poisoned history would then be persisted and re-sent forever.
            # An interrupt here instead yields placeholder "(interrupted)"
            # results so the pair always closes.
            prepared = []
            for call in calls:
                name = call.get("name", "")
                try:
                    arguments = parse_mod.parse_arguments(call.get("arguments", "{}"))
                except Exception:
                    arguments = {"arguments": call.get("arguments", "{}")}
                prepared.append((call, name, arguments))

            for call, name, arguments in prepared:
                self._emit_tool("tool_call", name, arguments=arguments, call_id=call.get("id", ""))

            if len(prepared) > 1:
                outcomes = self._run_tools_parallel(prepared)
            else:
                outcomes = []
                for call, name, arguments in prepared:
                    if self.interrupt():
                        outcomes.append(
                            (call, name, {"output": "(interrupted)", "error": True, "stopped": True})
                        )
                        break
                    tool_result = self.run_tool(name, arguments, call_id=call.get("id", ""))
                    outcomes.append((call, name, tool_result))

            for call, name, tool_result in outcomes:
                result.tool_calls_made += 1
                msg = parse_mod.tool_result_message(
                    call.get("id", ""),
                    name,
                    tool_result.get("output", ""),
                    error=bool(tool_result.get("error")),
                )
                # Persist the child's session link INSIDE the saved transcript:
                # without this a resumed parent's task rows have no sessionId
                # (it only lived in the live tool-result metadata), so
                # clicking them after a restart went nowhere.
                try:
                    meta = (tool_result.get("metadata") or {})
                    sid = meta.get("sessionId") if isinstance(meta, dict) else None
                    if name == "task" and sid:
                        msg["session_id"] = sid
                        msg["sessionId"] = sid
                except Exception:
                    pass
                self._history.append(msg)
            messages = list(self._history)

            if self.interrupt():
                self._emit("interrupted")
                return result
            # A fresh user message arrived while parked in task_read (the
            # TUI broke the wait via wake_bg_wait): end the drain after
            # this step instead of starting new provider steps, so the
            # message starts its own turn immediately — no QUEUED wait.
            # The bg agent keeps running; its reply folds in later.
            try:
                if getattr(self, "_bg_end_turn", False):
                    self._bg_end_turn = False
                    break
            except Exception:
                pass

            # Auto-compaction between steps (mirrors upstream opencode): after a
            # step completes, compact as soon as the ACTUAL provider-reported
            # usage fills the usable window — not only when the request is
            # estimated to overflow at the start of a turn. Tool loops grow the
            # context faster than the turn-start estimate predicts, so without
            # this a long session runs to 100% and stalls. Uses the SELECTED
            # model's window, so it works for any provider/model lane.
            if self.cfg.compaction_enabled:
                compacted = self._maybe_compact_by_usage(system_prompt)
                if compacted is not None:
                    messages = compacted

        # NOTE: errors are NOT appended to history. A permanent "[system]"
        # user message would re-send every past failure to the provider on
        # every future turn — polluting the context window and confusing the
        # model with stale noise. Errors reach the user through
        # TurnResult.error / the "error" event instead.
        # Track agent state for build-switch detection
        self._prev_agent = self.agent
        self._fold_finished_bg_agents()
        # Single-store pass (end of turn): big OLD tool bodies move to spill
        # files; live RAM keeps receipts, disk saves stay full (get_history
        # rehydrates before save — see below). Only when history is long
        # enough to matter; never touches the last keep_turns.
        try:
            if len(self._history) > 20:
                _kt = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
                _sid = str(getattr(self, "_session_id", "") or getattr(self, "session_id", "") or "")
                slim, _n = trim_mod.spill_old_tools(
                    self._history,
                    session_id=_sid,
                    keep_turns=_kt,
                )
                if _n > 0:
                    self._history = slim
        except Exception:
            pass
        return result

    # -- streaming --------------------------------------------------------
    def _stream(self, messages, tools, result: TurnResult, system_prompt: str) -> None:
        self._pending_calls = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Mid-turn effort switch: an effort change saved while streaming (the
        # thinking popup stays usable mid-turn) rebuilds the lanes HERE, so
        # the very next provider step — retry, failover, or tool-loop step —
        # uses the new effort instead of waiting for the next user turn.
        try:
            if getattr(self, "_rotation_dirty", False):
                self._rotation_dirty = False
                try:
                    self.rebuild_rotation()
                except Exception as e:
                    # Never stream silently on stale lanes: tell the user the
                    # model switch failed (the turn continues on the previous
                    # lanes, now visible).
                    try:
                        self._emit("notice", text=f"model switch failed ({e}); using previous model")
                    except Exception:
                        pass
        except Exception:
            pass
        # The assistant reply being streamed is kept LIVE in self._history as it
        # grows, so the app's periodic autosave (get_history) persists the exact
        # conversation up to the last token even if the app is killed suddenly
        # mid-stream — the session resumes where the user left off.
        live_assistant: dict | None = None

        def _ensure_live_assistant() -> dict:
            nonlocal live_assistant
            if live_assistant is None:
                live_assistant = {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "",
                }
                self._history.append(live_assistant)
            return live_assistant

        content_buf: list[str] = []
        reasoning_buf: list[str] = []

        def _live_append(live: dict, field: str, buf: list, text: str) -> None:
            # Amortized O(1) streaming: buffer chunks in a closure-local list,
            # flush into the live message every 64 chunks. The old
            # `live["content"] += text` copied the whole reply per token
            # (O(n^2) on long answers: each token slower than the last).
            # Buffers stay OUT of the live dict so autosave/sessions on disk
            # never persist internal keys; the settle path flushes exactly.
            buf.append(text)
            if len(buf) >= 64:
                live[field] = (live.get(field) or "") + "".join(buf)
                del buf[:]

        def _drop_live_assistant() -> None:
            nonlocal live_assistant
            if live_assistant is None:
                return
            idx = next((i for i, m in enumerate(self._history) if m is live_assistant), None)
            if idx is not None:
                self._history.pop(idx)
            live_assistant = None

        # Tool-loop requests are rebuilt from raw history (which never holds the
        # system prompt); re-prepend it so every request is well-formed.
        if not messages or messages[0].get("role") != "system":
            messages = self._prepend_system(messages, system_prompt)
        # An interrupted/force-killed turn can leave the request with an
        # assistant message declaring tool_calls but no following tool results;
        # strict backends reject that payload. Repair the local request copy so
        # the provider always sees a well-formed conversation.
        messages = msg_mod.repair_tool_pairs(messages)

        def on_event(evt) -> None:
            kind = evt.kind
            if kind == "text_delta":
                text_parts.append(evt.text)
                _live_append(_ensure_live_assistant(), "content", content_buf, evt.text)
                self._emit("text_delta", text=evt.text)
            elif kind == "reasoning_delta":
                reasoning_parts.append(evt.text)
                _live_append(_ensure_live_assistant(), "reasoning_content", reasoning_buf, evt.text)
                self._emit("reasoning_delta", text=evt.text)
            elif kind == "tool_call":
                for tc in evt.tool_calls or []:
                    self._pending_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
            elif kind == "usage":
                u = evt.usage
                # Context-window usage = the last completion's full footprint
                # (official: total || input+output+cache.read+cache.write).
                # Summing across streams would double-count history on
                # multi-step tool loops.
                total = usage_total(u)
                self._usage_total["input_tokens"] = u.input_tokens
                self._usage_total["output_tokens"] = u.output_tokens
                self._usage_total["total_tokens"] = total
                result.usage = dict(self._usage_total)
                usage_evt = dict(self._usage_total)
                ctx = self._model_context_size()
                if ctx:
                    result.usage["context_size"] = ctx
                    usage_evt["context_size"] = ctx
                self._emit("usage", usage=usage_evt)
            elif kind == "error":
                result.error = evt.error
                self._emit("error", error=evt.error)
            elif kind == "done":
                result.finish_reason = getattr(evt, "finish_reason", "") or result.finish_reason

        try:
            # Automatic retry for transient failures (streaming timeouts, 5xx,
            # overload, short rate limits) — mirrors upstream opencode's retry
            # policy, which waits out transient errors with exponential backoff
            # instead of surfacing them. Mirrors the official behavior:
            #   BASE_DELAY_MS=500, factor 2, jitter ±20%, cap 10s,
            #   honoring Retry-After when the provider sends it.
            # Events stay buffered inside rotation until a lane completes, so a
            # failed attempt emits nothing visible and any retry is clean.
            # Smart resume: if the model was mid-operation (it already ran
            # tools this turn), a plain re-send would replay every tool call
            # from scratch. Instead nudge the SAME model to continue; if
            # nothing happened yet, re-send the prompt as-is.
            for attempt in range(self.cfg.auto_retry_count + 1):
                try:
                    provider_id, model_id = self.rotation.stream(
                        self._retry_messages(messages, result, attempt),
                        tools,
                        on_event,
                        on_notice=self._on_rotated,
                        is_interrupted=self.interrupt,
                        locked=self.rotation_locked,
                    )
                    break
                except ContextOverflowError:
                    # has its own compaction recovery path below — never retry
                    raise
                except RateLimitError as e:
                    # Burst 429s (Zen free tier sends NO Retry-After header —
                    # live-verified: retry-after=None on every 429) are
                    # transient: the official client waits 2s/4s/8s... and
                    # retries the SAME model up to 5 times instead of dying
                    # mid-chat. Mirror that: backoff (honors Retry-After when
                    # present, else exponential) and only surface after the
                    # full retry budget is spent.
                    last_error = e
                    if not self.cfg.auto_retry or attempt >= self.cfg.auto_retry_count:
                        raise
                    if self.interrupt():
                        raise
                    remaining = self.cfg.auto_retry_count - attempt
                    self._emit(
                        "retry",
                        attempt=attempt + 1,
                        total=self.cfg.auto_retry_count,
                        message=f"↻ rate limited — retrying ({remaining} left)…",
                    )
                    self._sleep_interruptible(self._retry_delay(e, attempt))
                except ProviderError as e:
                    last_error = e
                    if not self.cfg.auto_retry or not e.retryable or attempt >= self.cfg.auto_retry_count:
                        raise
                    if self.interrupt():
                        raise
                    # Like official's session.status retry event: show the
                    # provider's own cause plus the attempt, not one static
                    # string for every failure kind.
                    remaining = self.cfg.auto_retry_count - attempt
                    cause = (getattr(e, "message", "") or str(e)).split("\n")[0][:100]
                    self._emit(
                        "retry",
                        attempt=attempt + 1,
                        total=self.cfg.auto_retry_count,
                        message=f"↻ {cause} — retrying ({remaining} left)…" if cause else f"↻ retrying ({remaining} left)…",
                    )
                    self._sleep_interruptible(self._retry_delay(e, attempt))
            else:
                # loop exhausted without success (the `break` never ran)
                raise last_error
            result.provider_id = provider_id
            result.model_id = model_id
        except ContextOverflowError as e:
            # The history overflowed the model's window even after budget
            # trimming (estimates are cheap). Recover by summarizing the
            # conversation into an anchored summary and keeping the recent tail
            # verbatim (upstream opencode's compaction), then retry once. This
            # fixes the turn instead of surfacing a hard error mid-conversation.
            if self.cfg.compaction_enabled:
                compacted = self._compact_context(system_prompt)
            else:
                compacted = None
            if compacted is not None:
                messages = compacted
                try:
                    provider_id, model_id = self.rotation.stream(
                        messages, tools, on_event, on_notice=self._on_rotated, is_interrupted=self.interrupt,
                        locked=self.rotation_locked,
                    )
                    result.provider_id = provider_id
                    result.model_id = model_id
                    result.error = ""
                except ContextOverflowError as e2:
                    result.error = f"context overflow (even after compaction): {e2}"
                    self._emit("error", error=result.error, retryable=True)
                    if not result.usage:
                        self._emit_request_usage(messages, tools)
                except ProviderError as e2:
                    result.error = str(e2)
                    self._emit("error", error=result.error, retryable=bool(e2.retryable))
                    if not result.usage:
                        self._emit_request_usage(messages, tools)
                except StreamInterrupted:
                    self._emit_request_usage(messages, tools, output_text="".join(text_parts) + "".join(reasoning_parts))
                    raise
                except Exception as e2:
                    result.error = str(e2)
                    self._emit("error", error=result.error)
                    if not result.usage:
                        self._emit_request_usage(messages, tools)
                if result.error:
                    pass  # surfaced above
            else:
                result.error = f"context overflow: {e}"
                self._emit("error", error=result.error, retryable=True)
                if not result.usage:
                    self._emit_request_usage(messages, tools)
        except RateLimitError as e:
            result.error = f"rate limit: {e}"
            self._emit("error", error=result.error, retryable=True)
            if not result.usage:
                self._emit_request_usage(messages, tools)
        except ProviderError as e:
            result.error = str(e)
            result.network_failed = bool(getattr(e, "network", False))
            self._emit("error", error=result.error, retryable=bool(e.retryable))
            if not result.usage:
                self._emit_request_usage(messages, tools)
        except StreamInterrupted:
            # User aborted mid-stream (Esc pressed twice / Ctrl+C): NOT an
            # error — end the turn as interrupted. Partial text already
            # streamed stays on screen, like upstream opencode's abort.
            # The request WAS fully sent, so count it; otherwise the footer
            # freezes on the older lower number as if nothing was spent.
            self._emit_request_usage(messages, tools, output_text="".join(text_parts) + "".join(reasoning_parts))
            self._emit("interrupted")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result.error = str(e)
            self._emit("error", error=result.error)

        result.text = "".join(text_parts)
        result.reasoning = "".join(reasoning_parts)
        if live_assistant is not None:
            # flush buffered live chunks (2.2 O(1) streaming keeps up to 63
            # unflushed) so history/autosave see the exact final text
            for _field, _buf in (("content", content_buf), ("reasoning_content", reasoning_buf)):
                try:
                    if _buf:
                        live_assistant[_field] = (live_assistant.get(_field) or "") + "".join(_buf)
                        del _buf[:]
                except Exception:
                    pass
        if (
            not self._pending_calls
            and not result.text
            and result.finish_reason == "length"
            and not result.error
        ):
            # Thinking burned the whole output budget before any answer text.
            # Without this the turn ends silently right after the visible
            # thinking phase — the reported "stops in the thinking part".
            _drop_live_assistant()
            result.error = (
                "the model hit its output-token limit before answering"
                " (finish_reason=length) — usually right after a long thinking"
                " phase. Retry, split the task smaller, or pick a lane with a"
                " larger output limit."
            )
            self._emit("error", error=result.error)
        if self._pending_calls:
            # run_turn builds ONE assistant message carrying text + reasoning +
            # tool_calls from result.text/result.reasoning — drop the live
            # text-only message first so one assistant turn isn't split (strict
            # backends and reasoning models lose thread on the next step).
            _drop_live_assistant()
        elif result.text:
            # keep the live message (it already holds every streamed delta);
            # sync it to the final buffers in case no delta path weirdly fired
            if live_assistant is None:
                _ensure_live_assistant()
            live_assistant["content"] = result.text
            live_assistant["reasoning_content"] = result.reasoning
        elif result.reasoning:
            # reasoning-only reply: replay it faithfully as what the model
            # actually returned — empty content plus a separate reasoning
            # signal. Storing the thinking also as `content` produces a message
            # that ever after disagrees with the model's real output, which
            # strict thinking-mode backends reject on the next request
            # ("reasoning_content must be passed back").
            if live_assistant is None:
                _ensure_live_assistant()
            live_assistant["content"] = ""
            live_assistant["reasoning_content"] = result.reasoning
        elif result.error:
            # nothing streamed — don't leave an empty assistant message behind
            _drop_live_assistant()
        else:
            # keep role alternation valid and avoid an empty-content message
            self._history.append(
                {"role": "assistant", "content": "(no response)", "reasoning_content": ""}
            )

    def undo_last(self) -> str:
        """Revert the most recent edit/write tool call (file-level snapshot)."""
        if not self._undo_stack:
            return "Nothing to undo."
        entry = self._undo_stack.pop()
        path = Path(entry["path"])
        try:
            if entry["original"] is None:
                if path.exists():
                    path.unlink()
                dirs = entry.get("dirs") or []
                for d in reversed(dirs):
                    d = Path(d)
                    # only remove directories we created and that are now empty
                    try:
                        if d.is_dir() and not any(d.iterdir()):
                            d.rmdir()
                    except OSError:
                        pass
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry["original"])
        except OSError as e:
            return f"Undo failed for {path}: {e}"
        self._emit("undo", path=str(path))
        return f"Reverted {path}."

    def _on_rotated(self, provider_id: str, model_id: str, reason: str = "") -> None:
        """A failover lane succeeded; announce it (with the switch reason)."""
        self._emit("rotated", provider=provider_id, model=model_id, reason=reason)

    def _model_context_size(self) -> int:
        """Context-window size of the active lane (0 when unknown)."""
        from ..providers import model_context_size

        pid = self.provider_id or self.cfg.provider
        mid = self.model_id or self.cfg.model
        return model_context_size(pid, mid, auth=self.auth)

    def _model_output_limit(self) -> int:
        """Max output tokens of the active lane (0 when unknown)."""
        from ..providers import model_output_limit

        pid = self.provider_id or self.cfg.provider
        mid = self.model_id or self.cfg.model
        return model_output_limit(pid, mid)

    def _estimate_request_tokens(self, system_prompt: str) -> int:
        """Request-size footprint of the current history.

        Same math the engine sends: system prompt + history + tool schemas
        (mirrors upstream `estimate({system, messages, tools})`). Shared by
        every place that repaints the footer's `12,345 (6%)` without a fresh
        provider usage event (compaction, interrupts, resume estimate).
        """
        try:
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(self._history)
            return compact_mod.estimate_request(msgs, self._active_tool_schemas())
        except Exception:
            return 0

    def estimate_history_tokens(self) -> int:
        """Request-size footprint of the current history, system prompt built
        in. Used by the TUI to paint a true count when reopening a session."""
        try:
            system_prompt = system_mod.build_system_prompt(
                directory=self.directory,
                worktree=self.worktree,
                provider_id=self.provider_id,
                model_id=self.model_id,
                cfg=self.cfg,
                agent=self.agent,
            )
        except Exception:
            system_prompt = ""
        return self._estimate_request_tokens(system_prompt)

    def _emit_request_usage(self, messages: list, tools: list, output_text: str = "") -> None:
        """Paint the footer after a turn that produced no provider usage event
        (interrupt / failure): the request WAS sent, so count the request
        estimate plus any partial output. Callers skip this when a real
        usage already landed for the turn."""
        try:
            sent = compact_mod.estimate_request(messages, tools)
            partial = compact_mod.estimate_tokens(output_text) if output_text else 0
            total = sent + partial
            self._usage_total = {"input_tokens": sent, "output_tokens": partial, "total_tokens": total}
            usage_evt = dict(self._usage_total)
            ctx = self._model_context_size()
            if ctx:
                usage_evt["context_size"] = ctx
            self._emit("usage", usage=usage_evt)
        except Exception:
            pass

    # continuation nudges sent to the SAME model when it stops mid-operation.
    # A plain re-send would replay every tool call from scratch; nudging lets
    # the model continue where it left off. Never rotates to another lane.
    _RETRY_NUDGES = (
        "keep going",
        "don't stop, keep going",
        "continue working, don't stop",
        "keep going, finish the task",
    )

    def _retry_messages(self, messages: list, result: TurnResult, attempt: int) -> list:
        """Rebuild the request for a retry on the same model.

        - If the turn already made progress (tools executed), the model was
          mid-operation: don't replay from scratch — append a short continuation
          nudge so it picks up where it was cut off. Each retry uses the next
          nudge in the list.
        - After every nudge is used, fall back to re-sending the full context
          unchanged: a fresh continuation beats piling on more "keep going"
          noise (which reasoning models read as a mid-thought interrupt and
          answer with nothing).
        - If nothing has happened yet, re-send the original prompt unchanged.
        """
        nudges = self._RETRY_NUDGES
        if result.tool_calls_made > 0 and attempt > 0:
            if attempt <= len(nudges):
                nudge = nudges[(attempt - 1) % len(nudges)]
                return [*messages, {"role": "user", "content": nudge}]
            return list(messages)
        return list(messages)

    def _sleep_interruptible(self, seconds: float) -> None:
        # Backoff that wakes the moment ESC is pressed: poll the interrupt
        # flag in 0.1s slices instead of one blocking time.sleep (which left
        # the app looking frozen for up to 30s). Raises StreamInterrupted so
        # the turn ends as interrupted, not as a retry.
        remaining = max(0.0, float(seconds or 0.0))
        while remaining > 0:
            if self.interrupt():
                from ..providers import StreamInterrupted as _SI

                raise _SI()
            step = 0.1 if remaining > 0.1 else remaining
            time.sleep(step)
            remaining -= step

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        """Backoff before a retry, mirroring upstream opencode's retry policy
        (session/retry.ts): 2s * 2^attempt plus up to +25% additive jitter.

        Honors ``Retry-After`` (seconds) when the provider sent it — header
        waits bypass the idle cap, like official. Without headers the wait is
        capped at 30s.
        """
        import random

        retry_after = getattr(error, "retry_after", None)
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except (TypeError, ValueError):
                pass
        base = 2.0 * (2**attempt)
        cap = min(base, 30.0)
        return round(cap * (1.0 + random.random() * 0.25), 3)

    def _skill_enabled(self) -> bool:
        """Honor tools.skill=false and agents.<agent>.tools.skill=false."""
        try:
            raw = getattr(self.cfg, "raw", None) or {}
            tools_cfg = raw.get("tools") if isinstance(raw, dict) else None
            if isinstance(tools_cfg, dict) and tools_cfg.get("skill") is False:
                return False
            agents_cfg = getattr(self.cfg, "agents", None) or {}
            if isinstance(agents_cfg, dict):
                spec = agents_cfg.get(self.agent) or {}
                if isinstance(spec, dict):
                    tools = spec.get("tools")
                    if isinstance(tools, dict) and tools.get("skill") is False:
                        return False
        except Exception:
            pass
        return True

    def _active_tool_schemas(self) -> list[dict]:
        schemas = self.registry.schemas()
        from ..permission import agent_readonly as _is_ro

        try:
            readonly = bool(_is_ro(self.cfg, self.agent))
        except Exception:
            readonly = self.agent in ("plan", "explore")
        if readonly:
            schemas = [
                s
                for s in schemas
                if s["function"]["name"] not in ("bash", "write", "edit", "apply_patch")
            ]
        # specialist scope: an agent whose spec names an explicit `scope`
        # list sees exactly those tools (intersected with what exists).
        # This keeps slim agents slim — web-agent never sees edit schemas.
        # Agents without a scope list see everything (minus readonly
        # stripping above). The permission editor manages allow/ask/deny
        # within whatever is visible.
        try:
            from ..permission import agent_spec as _spec

            spec = _spec(self.cfg, self.agent) or {}
            scope = spec.get("scope")
            if isinstance(scope, list) and scope:
                want = {str(t) for t in scope}
                schemas = [s for s in schemas if s["function"]["name"] in want]
        except Exception:
            pass
        if not self._skill_enabled():
            schemas = [s for s in schemas if s["function"]["name"] != "skill"]
        for s in schemas:
            if s["function"]["name"] == "skill":
                try:
                    from ..tools.skill import visible_skills

                    skills = visible_skills(self.permission)
                    if skills:
                        if getattr(self.cfg, "low_data", False):
                            # Save data: the system prompt already carries the
                            # full <available_skills> block — don't duplicate
                            # it (~N KB) inside the tool description too.
                            s["function"]["description"] = (
                                "Load a skill: reusable instructions from SKILL.md files. "
                                "Names are listed in the system prompt <available_skills>; "
                                "call skill({name}) to load the full content when the task matches."
                            )
                        else:
                            from ..tools.skill import SKILL_LIST_LIMIT

                            entries = "".join(
                                f"<skill><name>{sk.name}</name><description>{sk.description}</description></skill>"
                                for sk in skills[:SKILL_LIST_LIMIT]
                            )
                            s["function"]["description"] = (
                                "Load a skill: reusable instructions from SKILL.md files. "
                                f"Available skills: <available_skills>{entries}</available_skills> "
                                "Call skill({name}) to load the full content when the task matches, then follow it. "
                                'Call with no name to list them. Example: skill({"name": "git-release"}).'
                            )
                except Exception:
                    pass
                break
        return schemas

    def _was_plan(self) -> bool:
        return self._prev_agent == "plan"

    def _prepend_system(self, messages: list[dict], system_prompt: str) -> list[dict]:
        return [{"role": "system", "content": system_prompt}] + messages

    # -- compaction --------------------------------------------------------
    def force_compact(self) -> str:
        """Manually compact the conversation now (the `/compact` command).

        Mirrors upstream opencode: `/compact` runs the same AI compaction as the
        automatic overflow path. Returns the anchored summary ("" when nothing
        was compacted). Raises on provider failure.
        """
        system_prompt = system_mod.build_system_prompt(
            directory=self.directory,
            worktree=self.worktree,
            provider_id=self.provider_id,
            model_id=self.model_id,
            cfg=self.cfg,
            agent=self.agent,
        )
        messages = self._compact_context(system_prompt)
        if messages is None:
            return ""
        return self._compaction_summary or ""

    def _maybe_compact_by_usage(self, system_prompt: str) -> list[dict] | None:
        """Compact after a step when actual usage has filled the usable window.

        Mirrors upstream opencode's `compaction.isOverflow({tokens, model})`,
        which runs after every completed step: once the provider-reported token
        count (input + output of the last completion) reaches the usable window
        (context minus the reserve), the conversation is summarized so the next
        request never hits the provider's hard length limit. The window comes
        from the SELECTED model, so this works for any provider/model lane.
        """
        ctx = self._model_context_size()
        if ctx <= 0:
            return None
        output_limit = self._model_output_limit()
        usable = compact_mod.usable_context(ctx, output_limit)
        if usable <= 0:
            return None
        count = self._usage_total.get("total_tokens") or usage_total_dict(self._usage_total)
        if count < usable:
            return None
        return self._compact_context(system_prompt)

    # -- early background summary (fix 2) --------------------------------
    # At >=70% of the usable window a daemon pass summarizes the head BEHIND
    # the running turn. A later overflow then adopts the ready text with ZERO
    # extra model calls instead of freezing the turn on a sync summary.
    # Same head/tail split + prompt + application as _compact_context, so the
    # result is interchangeable. Disabled with compaction, single-flight per
    # history length, invalidated by interrupts/new prompts.
    def _early_summary_threshold(self, usable: int) -> float:
        try:
            frac = float(getattr(self.cfg, "early_summary_at", 0.7) or 0.7)
        except (TypeError, ValueError):
            frac = 0.7
        return min(0.95, max(0.5, frac))

    def _maybe_kick_early_summary(self, system_prompt, messages, tools, ctx, output_limit, usable) -> None:
        if not self.cfg.compaction_enabled:
            return
        if usable <= 0 or ctx <= 0:
            return
        try:
            est = compact_mod.estimate_request(messages, tools)
        except Exception:
            return
        if est < int(usable * self._early_summary_threshold(usable)):
            return
        if compact_mod.is_overflow(ctx, est, output_limit):
            return  # the sync path handles it this turn; no point pre-building
        with self._early_summary_lock:
            if self._early_summary_at == len(self._history):
                return  # already building/built for this history
            self._early_summary_at = len(self._history)
            self._early_summary_text = ""
            self._early_summary_head = {}
        hist = [m for m in self._history if not m.get("compaction")]
        if len(hist) < 4:
            with self._early_summary_lock:
                self._early_summary_at = -1
            return
        try:
            head, tail = compact_mod.select_tail(
                hist, tail_turns=self.cfg.compaction_tail_turns,
                context=ctx, output_limit=output_limit,
            )
        except Exception:
            with self._early_summary_lock:
                self._early_summary_at = -1
            return
        if not head:
            with self._early_summary_lock:
                self._early_summary_at = -1
            return
        try:
            prompt = compact_mod.summarize_conversation_prompt(head, tail)
        except Exception:
            with self._early_summary_lock:
                self._early_summary_at = -1
            return
        at = len(self._history)
        try:
            def _mid2(m):
                if not isinstance(m, dict):
                    return ""
                return str(m.get("id") or "") + "|" + str(m.get("role") or "") + "|" + str(m.get("content", ""))[:120]
            snap = {"len": len(head), "first": _mid2(head[0]), "last": _mid2(head[-1])}
        except Exception:
            snap = {}
        with self._early_summary_lock:
            self._early_summary_head = snap
        try:
            import threading as _th

            worker = _th.Thread(
                target=self._early_summary_run,
                args=(prompt, at),
                name="opencode_py-early-summary", daemon=True,
            )
            worker.start()
        except Exception:
            with self._early_summary_lock:
                self._early_summary_at = -1
                self._early_summary_head = {}

    def _early_summary_run(self, prompt: str, at: int) -> None:
        """Background summary on a FRESH provider — never self.rotation.

        rotation.stream mutates shared state (_active_provider, lane epochs,
        sticky sessions): sharing it with the live turn would corrupt the
        interrupt path and lane affinity. A fresh one-shot provider costs one
        extra lane pick and touches nothing shared.
        """
        texts: list[str] = []
        try:
            from ..providers import build_provider as _bp
            provider = _bp(self.cfg, self.provider_id or self.cfg.provider,
                           self.model_id or self.cfg.model, self.auth,
                           getattr(self.rotation, "session_id", None))
        except Exception:
            with self._early_summary_lock:
                if self._early_summary_at == at:
                    self._early_summary_at = -1
                    self._early_summary_head = {}
            return

        def _bg_event(evt) -> None:
            try:
                if evt.kind == "text_delta":
                    texts.append(evt.text)
            except Exception:
                pass
            # user hit Esc mid-turn: abandon the worker fast instead of
            # burning a summary nobody will adopt (length key won't match,
            # but fail fast rather than wasting the lane + battery).
            try:
                if self.interrupt():
                    raise _BgAbandoned()
            except _BgAbandoned:
                raise
            except Exception:
                pass

        try:
            provider.stream_chat([{"role": "user", "content": prompt}], [], _bg_event)
        except _BgAbandoned:
            with self._early_summary_lock:
                if self._early_summary_at == at:
                    self._early_summary_at = -1
                    self._early_summary_head = {}
            return
        except Exception:
            with self._early_summary_lock:
                if self._early_summary_at == at:
                    self._early_summary_at = -1
                    self._early_summary_head = {}
            return
        summary = "".join(texts).strip()
        with self._early_summary_lock:
            if self._early_summary_at == at and summary:
                self._early_summary_text = summary
            elif self._early_summary_at == at:
                self._early_summary_at = -1

    def _adopt_early_summary(self, system_prompt: str) -> list[dict] | None:
        """Use the ready background summary if its head still matches.

        The kick snapshots the head it summarized (not just a length): the
        turn keeps appending after the kick, so a length key could never
        match. Adoption verifies the snapshotted head is still the prefix
        of this history — same split, same first/last head ids.
        """
        with self._early_summary_lock:
            text = self._early_summary_text
            snap = dict(getattr(self, "_early_summary_head", None) or {})
        if not text or not snap:
            return None
        history = [m for m in self._history if not m.get("compaction")]
        if len(history) < 4:
            return None
        try:
            ctx = self._model_context_size()
            head, tail = compact_mod.select_tail(
                history, tail_turns=self.cfg.compaction_tail_turns,
                context=ctx, output_limit=self._model_output_limit(),
            )
        except Exception:
            return None
        if not head:
            return None
        # head-identity check: same length, same boundary message ids
        try:
            def _mid(m):
                if not isinstance(m, dict):
                    return ""
                return str(m.get("id") or "") + "|" + str(m.get("role") or "") + "|" + str(m.get("content", ""))[:120]
            if len(head) != int(snap.get("len", -1)):
                return None
            if _mid(head[0]) != snap.get("first", None) or _mid(head[-1]) != snap.get("last", None):
                return None
        except Exception:
            return None
        self._compaction_summary = text
        summary_text = f"[Summary of earlier conversation]\n{text}"
        if tail and tail[0].get("role") == "user":
            tail0 = dict(tail[0])
            content = tail0.get("content")
            if isinstance(content, str):
                tail0["content"] = f"{summary_text}\n\n{content}"
            elif isinstance(content, list):
                tail0["content"] = [{"type": "text", "text": summary_text}] + list(content)
            tail0["compaction"] = True
            new_history = [tail0] + tail[1:]
        else:
            new_history = [{"role": "user", "content": summary_text, "compaction": True}] + tail
        self._history = new_history
        with self._early_summary_lock:
            self._early_summary_text = ""
            self._early_summary_at = -1
            self._early_summary_head = {}
        try:
            from ..tools.context_ledger import reset_for_compaction
            reset_for_compaction()
        except Exception:
            pass
        self._emit("compacted", summary=text)
        try:
            total = self._estimate_request_tokens(system_prompt)
            self._usage_total = {"input_tokens": total, "output_tokens": 0, "total_tokens": total}
            # The adopted-summary path never emitted usage, so the footer
            # kept showing the pre-compaction high number until the next
            # reply — repaint it now like the other compaction paths.
            usage_evt = dict(self._usage_total)
            if ctx:
                usage_evt["context_size"] = ctx
            self._emit("usage", usage=usage_evt)
        except Exception:
            pass
        return self._prepend_system(list(self._history), system_prompt)

    def _compact_context(self, system_prompt: str) -> list[dict] | None:
        """Summarize the conversation into an anchored summary and continue.

        Mirrors upstream opencode's compaction: split the history into a head
        (to summarize) and a recent tail (kept verbatim), ask a model to write
        an anchored summary, then rebuild the request from summary + tail so the
        turn continues instead of erroring out. Returns the rebuilt request
        messages, or None if compaction can't run (no history / model failure).

        Free-first: before spending a model call, try the free rule-cut
        (trim.free_compact). When it frees enough to fit, the turn continues
        with ZERO model calls — same tail, recall notes in place of cut
        bodies, self-recall via history_search action=recall. The model
        summary below runs only when the free cut was NOT enough.
        """
        if not self.cfg.compaction_enabled:
            return None
        history = [m for m in self._history if not m.get("compaction")]
        if len(history) < 4:
            return None
        # Size the preserved tail from the active lane's context window (the
        # model that is actually answering), not a hardcoded 200k assumption.
        ctx = self._model_context_size()
        head, tail = compact_mod.select_tail(
            history,
            tail_turns=self.cfg.compaction_tail_turns,
            context=ctx,
            output_limit=self._model_output_limit(),
        )
        if not head:
            return None
        # Free-first: rule-cut the head; if the result fits, skip the model.
        try:
            _sid = str(getattr(self, "_session_id", "") or getattr(self, "session_id", "") or "")
            _keep = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
            _usable = compact_mod.usable_context(ctx, self._model_output_limit()) if ctx > 0 else 0
            _cut_hist, _ncut = trim_mod.free_compact(list(history), session_id=_sid, keep_turns=_keep)
            if _ncut > 0:
                _req = compact_mod.estimate_request(
                    self._prepend_system(_cut_hist, system_prompt), self._active_tool_schemas())
                if _usable <= 0 or _req < _usable:
                    self._history = _cut_hist
                    from ..tools.context_ledger import reset_for_compaction

                    reset_for_compaction()
                    self._emit("compacted", summary=f"Free compact: cut {_ncut} old tool outputs (no model call). Re-fetch any with history_search action=recall.")
                    total = self._estimate_request_tokens(system_prompt)
                    self._usage_total = {"input_tokens": total, "output_tokens": 0, "total_tokens": total}
                    usage_evt = dict(self._usage_total)
                    if ctx:
                        usage_evt["context_size"] = ctx
                    self._emit("usage", usage=usage_evt)
                    messages = self._prepend_system(list(self._history), system_prompt)
                    return messages
        except Exception:
            pass
        prompt = compact_mod.summarize_conversation_prompt(head, tail)
        summary_texts: list[str] = []

        # Emit BEFORE the summary stream so the TUI can show the official
        # opencode "Compacting conversation…" indicator while the model works.
        self._emit("compaction_start", reason="auto")

        def on_summary(evt) -> None:
            if evt.kind == "text_delta":
                summary_texts.append(evt.text)
                # Stream the summary live so the TUI shows it being written
                # instead of a static "Compacting…" spinner that dumps the whole
                # block at the end.
                self._emit("summary_delta", text=evt.text)

        try:
            self.rotation.stream(
                [{"role": "user", "content": prompt}],
                [],
                on_summary,
                on_notice=self._on_rotated,
                locked=self.rotation_locked,
            )
        except Exception as e:
            # `compaction_start` was already emitted; a failure must still emit a
            # terminal event or the TUI's "Compacting…" spinner never clears
            # (the InputBar only resets on `compacted`). Report it there so both
            # the state clears and the user sees what went wrong.
            self._emit("compacted", summary=f"Compaction failed: {e}")
            return None
        summary = "".join(summary_texts).strip()
        if not summary:
            self._emit("compacted", summary="Compaction produced no summary")
            return None
        self._compaction_summary = summary
        # Upstream opencode emits the compaction checkpoint as a USER message
        # (`<conversation-checkpoint>` in to-llm-message.ts), NOT an assistant
        # message. Fabricating an assistant message here would be replayed to the
        # API in thinking mode without a `reasoning_content`, which strict
        # gateways reject with "reasoning_content in thinking mode must be passed
        # back". The tail already starts with a user turn, so fold the summary
        # into that first message to keep role alternation intact (a bare
        # `user,user` sequence is rejected by Anthropic).
        summary_text = f"[Summary of earlier conversation]\n{summary}"
        if tail and tail[0].get("role") == "user":
            tail0 = dict(tail[0])
            content = tail0.get("content")
            if isinstance(content, str):
                tail0["content"] = f"{summary_text}\n\n{content}"
            elif isinstance(content, list):
                # Multimodal user turn: keep the original parts verbatim but
                # PREPEND the summary as a text part — the old code replaced
                # the list unchanged, quietly losing the entire anchored
                # summary (the conversation history evaporated on the next
                # compaction/overflow).
                tail0["content"] = [{"type": "text", "text": summary_text}] + list(content)
            tail0["compaction"] = True
            summary_msg = tail0
            new_history = [summary_msg] + tail[1:]
        else:
            summary_msg = {
                "role": "user",
                "content": summary_text,
                "compaction": True,
            }
            new_history = [summary_msg] + tail
        self._history = new_history
        # The pre-compaction bodies (file reads, tool outputs) are gone from
        # the model's head now — the read tool's dedup ledger must forget what
        # was delivered, or re-reads would return "already sent" stubs for
        # content the model can no longer see.
        from ..tools.context_ledger import reset_for_compaction

        reset_for_compaction()
        self._emit("compacted", summary=summary)
        # Recompute the context estimate so the TUI's `12,345 (6%)` reflects
        # the compacted conversation, not the pre-summary size that triggered
        # the overflow (mirrors opencode recomputing tokens after compaction).
        total = self._estimate_request_tokens(system_prompt)
        self._usage_total = {"input_tokens": total, "output_tokens": 0, "total_tokens": total}
        usage_evt = dict(self._usage_total)
        if ctx:
            usage_evt["context_size"] = ctx
        self._emit("usage", usage=usage_evt)
        messages = self._prepend_system(list(self._history), system_prompt)
        return messages

    # -- session glue -----------------------------------------------------
    def set_history(self, history: list[dict]) -> None:
        self._history = list(history)
        if not history:
            self._usage_total = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    def get_history(self) -> list[dict]:
        # Disk saves + exports must stay FULL: rehydrate spilled bodies so
        # the persisted transcript never loses a byte (live RAM keeps the
        # slim receipts; see the end-of-turn spill above).
        try:
            return trim_mod.rehydrate_spills(list(self._history))
        except Exception:
            return list(self._history)

    def request_history(self) -> list[dict]:
        """Slim request history: rehydrate ONLY bodies the request trim keeps.

        The request path shrink_tool_history re-cuts every OLD spilled body
        back to a receipt — rehydrating all 24 first (53ms) just to re-shrink
        24 (22ms) is pure waste. Trim on the slim receipts, then restore only
        the survivors (measured: 0 of 24 survive). Same bytes sent, ~0 rehydrate.
        Falls back to full rehydrate on any error (never lose content).
        """
        try:
            keep_turns = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
        except (TypeError, ValueError):
            keep_turns = 2
        try:
            trim_chars = int(getattr(self.cfg, "trim_max_chars", 500) or 500)
        except (TypeError, ValueError):
            trim_chars = 500
        try:
            slim = trim_mod.shrink_tool_history(list(self._history), keep_turns, trim_chars)
            need = any(isinstance(m, dict) and isinstance(m.get("spill_meta"), dict)
                       and not str(m.get("content") or "").startswith(trim_mod.RECEIPT_PREFIX)
                       for m in slim)
            if not need:
                return slim
            return trim_mod.rehydrate_spills(slim)
        except Exception:
            try:
                return trim_mod.rehydrate_spills(list(self._history))
            except Exception:
                return list(self._history)

    def live_history(self) -> list[dict]:
        """Slim in-RAM history (spilled bodies stay as receipts)."""
        try:
            return list(self._history)
        except Exception:
            return []

    def add_placeholder_tool_message(self, output: str) -> None:
        self._history.append({"role": "assistant", "content": output})
