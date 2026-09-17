"""MCP server manager: list/add/remove/test with 32-bit Termux in mind.

Config shape (project ./opencode.json or ~/.config/opencode_py/opencode.json):
  {"mcpServers": {"name": {"command": "python", "args": ["-m", "my_server"]}}}

Only stdio servers (command + args). Both `python -m ...` (preferred on a
32-bit Termux phone: tiny, no Node needed) and `npx -y ...` (needs Node, often
missing/heavy on armv7) are accepted; setup warns when the binary is missing.
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


def list_servers(cfg: Any) -> dict[str, dict]:
    raw = (getattr(cfg, "raw", None) or {})
    servers = raw.get("mcpServers", {}) or {}
    if not isinstance(servers, dict):
        return {}
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in servers.items()}


def parse_add_args(args: str) -> tuple[str, str, list[str], bool]:
    """Parse `/mcp add <name> [--global] -- <command...>` (shlex).

    Returns (name, command, cmd_args, is_global). Raises ValueError on misuse.
    """
    tokens = shlex.split(args)
    if tokens and tokens[0].lower() == "add":
        tokens = tokens[1:]
    is_global = False
    if "--global" in tokens:
        is_global = True
        tokens = [t for t in tokens if t != "--global"]
    if "--" in tokens:
        i = tokens.index("--")
        head, cmd = tokens[:i], tokens[i + 1:]
    else:
        head, cmd = tokens[:1], tokens[1:]
    if not head:
        raise ValueError("Usage: /mcp add <name> -- <command...>")
    if not cmd:
        raise ValueError(
            f"Usage: /mcp add {head[0]} -- <command...>\n"
            "  e.g. /mcp add files -- python -m my_server\n"
            "  e.g. /mcp add files -- npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
    return head[0], cmd[0], cmd[1:], is_global


def check_command(command: str, args: list[str]) -> list[str]:
    """Phone-friendly warnings for a server command. Empty = looks fine."""
    warnings: list[str] = []
    if not command:
        return ["empty command"]
    resolved = shutil.which(command)
    if "/" in command:
        p = Path(command)
        if not p.exists():
            warnings.append(f"'{command}' not found on this device.")
        elif not p.is_file():
            warnings.append(f"'{command}' is not a file.")
    elif resolved is None:
        if command in ("npx", "node", "npm", "bun", "deno"):
            warnings.append(
                f"'{command}' not found. On Termux: pkg install nodejs "
                "(heavy, ~100MB+, may be missing on 32-bit) — "
                "prefer a Python server: python -m my_server."
            )
        elif command == "python":
            warnings.append(f"'{command}' not found. Try '{sys.executable}'.")
        else:
            warnings.append(f"'{command}' not found in PATH.")
    if command in ("python", "python3", sys.executable) or (
        resolved and Path(resolved).name.startswith("python")
    ):
        if args[:1] == ["-m"] and len(args) >= 2:
            mod = args[1]
            try:
                import importlib.util as _ilu

                if _ilu.find_spec(mod) is None:
                    warnings.append(f"python module '{mod}' not importable here.")
            except Exception:
                pass
        elif args[:1] == ["-m"]:
            warnings.append("'python -m' needs a module name after it.")
    return warnings


def test_server(name: str, command: str, args: list[str], timeout: float = 20.0) -> str:
    """Start the server, run tools/list, close it. Returns a human line."""
    from .tools.mcp import MCPError, MCPServer

    server = MCPServer(name=name, command=command, args=list(args), timeout=timeout)
    try:
        tools = server.list_tools()
    except MCPError as e:
        return f"{name}: FAILED — {e}"
    except Exception as e:  # pragma: no cover - defensive
        return f"{name}: FAILED — {e}"
    finally:
        try:
            server.close()
        except Exception:
            pass
    names = [t.get("name", "?") for t in tools if isinstance(t, dict)][:5]
    extra = f" (+{len(tools) - 5} more)" if len(tools) > 5 else ""
    shown = ", ".join(names) if names else "no tools"
    return f"{name}: OK — {len(tools)} tool(s): {shown}{extra}"


def config_path(worktree: str = "", is_global: bool = False) -> Path:
    from .globals import Path as GPath

    if is_global:
        return GPath.config / "opencode.json"
    base = Path(worktree) if worktree else Path.cwd()
    return base / "opencode.json"


def save_server(name: str, command: str, args: list[str], worktree: str = "", is_global: bool = False) -> Path:
    """Write one server into opencode.json (preserving other keys)."""
    from .config import _load_json, _strip_jsonc

    path = config_path(worktree, is_global)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = _load_json(path)
        except Exception:
            try:
                data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
            except Exception:
                data = {}
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        data["mcpServers"] = servers
    servers[name] = {"command": command, "args": list(args)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def remove_server(name: str, worktree: str = "", is_global: bool = False) -> Path | None:
    """Delete one server from opencode.json. Returns path, or None if absent."""
    from .config import _load_json, _strip_jsonc

    path = config_path(worktree, is_global)
    if not path.exists():
        return None
    try:
        data = _load_json(path)
    except Exception:
        try:
            data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
        except Exception:
            return None
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or name not in servers:
        return None
    del servers[name]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def remove_server_everywhere(name: str, worktree: str = "") -> list[Path]:
    """Delete a server from BOTH project and global files.

    The merged config can't tell which scope provided the entry, so a
    single-scope delete resurrects on restart. Returns removed paths.
    """
    removed: list[Path] = []
    for is_global in (False, True):
        try:
            p = remove_server(name, worktree, is_global)
        except Exception:
            p = None
        if p is not None:
            removed.append(p)
    try:
        from .tools import _mcp_cache_evict_closed
    except Exception:
        pass
    else:
        try:
            _mcp_cache_evict_closed()
        except Exception:
            pass
    return removed


def refresh_engine_mcp(engine: Any, cfg: Any) -> str:
    """Reload MCP tools into a live engine registry (next turn uses them).

    Drops stale mcp__*/mcp:* tools, releases old server refs, re-runs the
    shared _load_mcp_servers() against the updated cfg.
    """
    if engine is None or getattr(engine, "registry", None) is None:
        return "saved (applies on next session start)."
    registry = engine.registry
    lock = getattr(registry, "_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        for tool_name in list(registry.names()):
            if tool_name.startswith("mcp__") or tool_name.startswith("mcp:"):
                try:
                    registry._tools.pop(tool_name, None)
                except Exception:
                    pass
        try:
            registry.close()
        except Exception:
            pass
        from .tools import _load_mcp_servers, _mcp_cache_evict_closed

        _load_mcp_servers(registry, cfg)
        try:
            _mcp_cache_evict_closed()
        except Exception:
            pass
    except Exception as e:
        return f"saved (live reload failed: {e} — applies on next session start)."
    finally:
        try:
            if lock is not None:
                lock.release()
        except Exception:
            pass
    names = [n for n in registry.names() if n.startswith("mcp__")]
    return f"live now ({len(names)} MCP tool(s) loaded)."
