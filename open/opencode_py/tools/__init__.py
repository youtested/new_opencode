"""Tools package: registry builder mirroring opencode's builtin tool order."""

from __future__ import annotations

import atexit
import threading
from typing import Any

from ..config import Config
from .registry import Registry, Tool, schema_with

# ponytail: tool modules import INSIDE their factory (not at top), so
# importing this package costs ~0ms of tool code. Each factory is one
# importlib.import_module + one tool() call, cached by the registry.
def _lazy(mod_name: str, attr: str = "tool", **kwargs: Any) -> Any:
    def _factory() -> Any:
        import importlib

        mod = importlib.import_module(f"opencode_py.tools.{mod_name}")
        return getattr(mod, attr)(**kwargs)

    return _factory

TOOL_NAMES = ["bash", "read", "glob", "grep", "find_symbols", "lsp", "edit", "write", "apply_patch", "background_task", "browser", "phone", "screen_view", "device", "history_search", "checkpoint", "verify", "quick_calc", "speak", "webfetch", "webfetch_many", "websearch", "todowrite", "task", "question", "skill", "remember", "summarize_file"]

# Process-wide cache of running MCP servers keyed by (name, command, args).
# build_registry() runs once per engine AND once per spawned sub-agent; without
# sharing, N sub-agents would spawn N+1 server processes per server that were
# never reaped. A cache makes every registry reuse ONE process per server, and
# MCPServer's reference counting terminates it when the last registry closes it.
_MCP_SERVER_CACHE: dict[tuple, Any] = {}
_MCP_CACHE_LOCK = threading.Lock()


_MCP_CACHE_MAX = 32


def _mcp_timeouts(cfg: Config | None) -> tuple[float, float]:
    """Handshake / tool-call budgets (slow 32-bit phones need room).

    Config knobs (opencode.json): ``mcpTimeout`` (default 20s),
    ``mcpToolTimeout`` (default 60s). Env ``OPENCODE_MCP_TIMEOUT`` /
    ``OPENCODE_MCP_TOOL_TIMEOUT`` override. Never raises.
    """
    import os as _os

    t, tt = 20.0, 60.0
    try:
        raw = (getattr(cfg, "raw", None) or {}) if cfg else {}
        if isinstance(raw.get("mcpTimeout"), (int, float)):
            t = float(raw["mcpTimeout"])
        if isinstance(raw.get("mcpToolTimeout"), (int, float)):
            tt = float(raw["mcpToolTimeout"])
    except Exception:
        pass
    for env, slot in (("OPENCODE_MCP_TIMEOUT", "t"), ("OPENCODE_MCP_TOOL_TIMEOUT", "tt")):
        try:
            v = _os.environ.get(env)
            if v:
                if slot == "t":
                    t = float(v)
                else:
                    tt = float(v)
        except (TypeError, ValueError):
            pass
    t = min(max(t, 5.0), 120.0)
    tt = min(max(tt, t), 300.0)
    return t, tt


def _mcp_needs_trust(command: str) -> bool:
    # Shell-capable launchers can run arbitrary code from a repo config;
    # plain interpreters with explicit module args are safe by default.
    base = str(command or "").split("/")[-1].lower()
    return base in ("sh", "bash", "zsh", "fish", "dash", "cmd", "powershell", "pwsh", "curl", "wget")


def _mcp_server_get(name: str, command: str, args: list, timeout: float = 20.0, tool_timeout: float | None = None) -> Any:
    """Return a cached (running or respawnable) MCP server, creating it on
    first use."""
    from .mcp import MCPServer

    key = (str(name), command, tuple(args))
    with _MCP_CACHE_LOCK:
        server = _MCP_SERVER_CACHE.get(key)
        if server is not None:
            # Widen budgets when a later config asks for more room.
            try:
                if timeout and getattr(server, "timeout", 0) < timeout:
                    server.timeout = timeout
                if tool_timeout and getattr(server, "tool_timeout", 0) < tool_timeout:
                    server.tool_timeout = tool_timeout
            except Exception:
                pass
        if server is None:
            # Bound growth across add/remove cycles: evict a closed entry
            # first, else the oldest key. Stale closed entries respawn on
            # next use anyway, so eviction is always safe.
            while len(_MCP_SERVER_CACHE) >= _MCP_CACHE_MAX:
                for k, v in list(_MCP_SERVER_CACHE.items()):
                    try:
                        idle = getattr(v, "proc", None) is None
                    except Exception:
                        idle = True
                    if idle:
                        _MCP_SERVER_CACHE.pop(k, None)
                        break
                else:
                    _MCP_SERVER_CACHE.pop(next(iter(_MCP_SERVER_CACHE)), None)
            server = MCPServer(name=str(name), command=command, args=args,
                               timeout=timeout, tool_timeout=tool_timeout)
            _MCP_SERVER_CACHE[key] = server
        return server


def _mcp_cache_evict_closed() -> None:
    """Drop closed/idle entries (called after refresh/remove)."""
    with _MCP_CACHE_LOCK:
        for k, v in list(_MCP_SERVER_CACHE.items()):
            try:
                if getattr(v, "proc", None) is None and getattr(v, "_users", 0) <= 0:
                    _MCP_SERVER_CACHE.pop(k, None)
            except Exception:
                pass


def _mcp_server_close_all() -> None:
    """Terminate every cached MCP server process (atexit + engine teardown)."""
    with _MCP_CACHE_LOCK:
        servers = list(_MCP_SERVER_CACHE.values())
        _MCP_SERVER_CACHE.clear()
    for server in servers:
        try:
            server.close()
        except Exception:  # pragma: no cover - best effort at process exit
            pass


atexit.register(_mcp_server_close_all)


def build_registry(cfg: Config | None = None) -> Registry:
    cfg = cfg or Config()
    registry = Registry()
    state: dict = {}

    # Hot path stays eager (bash/read/edit/grep/glob/lsp/find_symbols):
    # zero first-use latency mid-work. Everything else loads lazily on
    # first get()/run — one ~0.05s hit, then cached forever.
    from . import bash as bash_mod
    from . import edit as edit_mod
    from . import find_symbols as find_symbols_mod
    from . import glob as glob_mod
    from . import grep as grep_mod
    from . import lsp as lsp_mod
    from . import read as read_mod

    registry.register(bash_mod.tool(
        max_lines=cfg.tool_output_max_lines,
        max_bytes=cfg.tool_output_max_bytes,
        default_timeout=cfg.bash_default_timeout,
        registry=registry,
    ))
    registry.register(read_mod.tool(cfg))
    registry.register(glob_mod.tool())
    registry.register(grep_mod.tool(cfg))
    registry.register(find_symbols_mod.tool())
    registry.register(lsp_mod.tool())
    registry.register(edit_mod.tool())
    registry.register_lazy("write", _lazy("write"))
    registry.register_lazy("apply_patch", _lazy("apply_patch"))
    registry.register_lazy("background_task", _lazy("background", registry=registry))
    registry.register_lazy("browser", _lazy("browser"))
    registry.register_lazy("phone", _lazy("phone"))
    registry.register_lazy("screen_view", _lazy("screen_view"))
    registry.register_lazy("device", _lazy("device"))
    registry.register_lazy("history_search", _lazy("history_search"))
    registry.register_lazy("checkpoint", _lazy("checkpoint"))
    registry.register_lazy("verify", _lazy("verify"))
    registry.register_lazy("quick_calc", _lazy("quick_calc"))
    registry.register_lazy("speak", _lazy("speak", cfg=cfg))
    registry.register_lazy("webfetch", _lazy("webfetch", registry=registry))
    registry.register_lazy("webfetch_many", _lazy("webfetch", attr="batch_tool", registry=registry))
    registry.register_lazy("websearch", _lazy("websearch", registry=registry))
    registry.register_lazy("todowrite", _lazy("todo", state=state))
    registry.register_lazy("task", _lazy("task", registry=registry))
    registry.register_lazy("question", _lazy("question", registry=registry))
    registry.register_lazy("skill", _lazy("skill", registry=registry))
    registry.register_lazy("remember", _lazy("remember"))
    registry.register_lazy("summarize_file", _lazy("summarize_file"))

    # config-driven tool toggles: tools.<name> = false removes it (opencode behavior)
    enabled: dict | None = None
    if cfg and cfg.raw:
        enabled = cfg.raw.get("tools")
    if enabled:
        for name in list(registry.names()):
            if enabled.get(name) is False:
                registry._tools.pop(name, None)
                try:
                    registry._lazy.pop(name, None)
                except Exception:
                    pass

    _load_plugins(registry, cfg)
    _load_mcp_servers(registry, cfg)

    return registry


def _load_plugins(registry: Registry, cfg: Config | None) -> None:
    """Plugin-lite: config key "plugins": ["my.tools.module"] where the module
    exposes TOOLS = [{name, description, parameters, run}, ...]."""
    raw = (cfg and cfg.raw) or {}
    import importlib
    import sys
    from pathlib import Path

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    for mod_path in raw.get("plugins", []) or []:
        try:
            mod = importlib.import_module(str(mod_path))
            for tool_def in getattr(mod, "TOOLS", []) or []:
                registry.register(
                    Tool(
                        name=tool_def["name"],
                        description=tool_def.get("description", ""),
                        parameters=tool_def.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                        run=tool_def["run"],
                    )
                )
        except Exception as e:
            registry.register(
                Tool(
                    name=_sanitize_tool_name(f"plugin__{mod_path}__error"),
                    description="plugin failed to load",
                    parameters={"type": "object", "properties": {}},
                    run=lambda args, err=e, mod=mod_path: {
                        "output": f"plugin {mod} failed to load: {err}",
                        "error": True,
                    },
                )
            )


def _load_mcp_servers(registry: Registry, cfg: Config | None) -> None:
    """MCP-lite: config key "mcpServers": {name: {command, args}} -> tools named
    mcp__<name>__<tool>.

    Server processes are SHARED process-wide (see the module cache above): a
    parent registry and every sub-agent registry reuse the SAME process instead
    of spawning one per registry build, and each registry holds a reference via
    ``server.acquire()`` that ``Registry.close()`` returns so the process dies
    when its last user is done.
    """
    raw = (cfg and cfg.raw) or {}
    servers = raw.get("mcpServers", {}) or {}
    if not servers:
        return
    from .mcp import MCPError

    trusted = raw.get("trustedServers") or []
    if not isinstance(trusted, list):
        trusted = []
    timeout, tool_timeout = _mcp_timeouts(cfg)
    for sname, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        command = spec.get("command")
        args = spec.get("args") or []
        if not command or not isinstance(args, list):
            continue
        if _mcp_needs_trust(str(command)) and str(sname) not in trusted:
            # Untrusted shell-capable server: register a placeholder that
            # explains how to approve instead of spawning arbitrary code
            # from a repo config on first use.
            registry.register(
                Tool(
                    name=_sanitize_tool_name(f"mcp__{sname}__error"),
                    description=f"mcp server {sname} needs approval",
                    parameters={"type": "object", "properties": {}},
                    run=lambda args, _s=str(sname): {
                        "output": (
                            f"mcp server '{_s}' is not trusted: it runs a shell-capable "
                            f"command. Add its name to `trustedServers` in opencode.json "
                            f"to approve it, or use a python/node module server."
                        ),
                        "error": True,
                    },
                )
            )
            continue
        server = _mcp_server_get(sname, command, args, timeout, tool_timeout)
        server.acquire()
        registry.mcp_servers.append(server)
        try:
            remote_tools = server.list_tools()
        except Exception as e:
            server.release()
            registry.mcp_servers.remove(server)
            registry.register(
                Tool(
                    name=_sanitize_tool_name(f"mcp__{sname}__error"),
                    description=f"mcp server {sname} failed to start",
                    parameters={"type": "object", "properties": {}},
                    run=lambda args, err=str(e): {"output": f"mcp {sname}: {err}", "error": True},
                )
            )
            continue
        for t in remote_tools:
            if not isinstance(t, dict):
                continue
            # `tools/list` is server-controlled: a malformed item (missing
            # "name") used to raise KeyError here and crash build_registry —
            # i.e. every engine and every sub-agent — for a config typo on one
            # remote server. Skip nameless tools instead of taking down startup.
            remote_name = t.get("name")
            if not remote_name:
                continue
            params = t.get("inputSchema") or {"type": "object", "properties": {}}
            if not isinstance(params, dict):
                continue
            registry.register(
                Tool(
                    name=_sanitize_tool_name(f"mcp__{sname}__{remote_name}"),
                    description=t.get("description") or f"{sname}: {remote_name}",
                    parameters=params,
                    run=_mcp_run(server, remote_name),
                )
            )


def _sanitize_tool_name(name: str) -> str:
    """Make a tool name safe for strict OpenAI-compatible validators.

    Single source of truth lives in ``providers.base.sanitize_function_name``;
    this wrapper keeps the historical import path working.
    """
    from ..providers.base import sanitize_function_name as _san

    return _san(name)


def _mcp_run(server, remote_name: str):
    def run(arguments: dict) -> dict[str, Any]:
        try:
            return server.run_tool(remote_name, arguments)
        except Exception as e:
            return {"output": f"mcp tool {remote_name} failed: {e}", "error": True}

    return run


__all__ = [
    "Registry",
    "Tool",
    "schema_with",
    "build_registry",
    "TOOL_NAMES",
]
