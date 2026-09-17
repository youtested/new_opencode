"""Tool registry: name -> Tool dataclass with JSON schema + run().

Mirrors opencode's Tool.define pattern. Each tool declares a name, description,
parameter JSON schema, an optional permission key, and a run(input) -> dict.
The run result dict carries `output` (text), plus optional metadata the TUI
renders (e.g. edit diff, bash exit code).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], dict[str, Any]]
    permission: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # Lazy tool factories: name -> () -> Tool. build_registry registers
        # these instead of importing every tool module up front; the module
        # loads on first get()/run (one ~0.05s hit, then cached forever).
        # ponytail: plain dict + RLock, no fancy loader; upgrade path is
        # importlib lazy loading if the tool count ever grows 10x.
        self._lazy: dict[str, Any] = {}
        self.mcp_servers: list[Any] = []
        # Guards _tools/mcp_servers against refresh-while-busy: /mcp add
        # runs on the UI thread mid-turn while workers read schemas.
        self._lock = threading.RLock()
        # In-flight HTTP responses opened by the webfetch tools on this engine.
        # The interrupt path (engine.abort) calls abort_fetches() to force-close
        # them, waking a blocked socket read immediately so ESC/Ctrl+C aborts a
        # running fetch without waiting for its timeout.
        self._fetch_lock = threading.Lock()
        self._active_fetches: list[Any] = []
        # Per-call live progress for long tools (webfetch_many 2/5 counts):
        # run_tool() stamps this thread's (call_id, emitter) here; tools read
        # it at run() start and call emitter(done, total) per finished unit.
        # Thread-local so parallel tools never race each other.
        self._progress_ctx = threading.local()

    def register(self, tool: Tool) -> Tool:
        # Sanitized-collision guard: two raw names (e.g. "a:b" vs "a_b")
        # would otherwise silently overwrite each other and the wrong tool
        # executes. Disambiguate with a numeric suffix instead.
        try:
            from ..providers.base import sanitize_function_name as _san

            want = _san(tool.name)
            for existing in self._tools.values():
                if _san(existing.name) == want and existing.name != tool.name:
                    base = tool.name[:56]
                    i = 2
                    while f"{base}__{i}" in self._tools or any(
                        _san(n) == _san(f"{base}__{i}") for n in self._tools
                    ):
                        i += 1
                    tool.name = f"{base}__{i}"
                    break
        except Exception:
            pass
        self._tools[tool.name] = tool
        return tool

    def register_lazy(self, name: str, factory: Any) -> None:
        """Register a tool factory loaded on first use (see _materialize)."""
        try:
            with self._lock:
                if name not in self._tools:
                    self._lazy[name] = factory
        except Exception:
            pass

    def _materialize(self, name: str) -> Any | None:
        """Build + register a lazy tool on first get/run. Never raises."""
        try:
            with self._lock:
                if name in self._tools:
                    return self._tools[name]
                factory = self._lazy.pop(name, None)
            if factory is None:
                return None
            tool = factory()
            if tool is not None:
                try:
                    self.register(tool)
                except Exception:
                    pass
                return tool
            return None
        except Exception:
            return None

    def _materialize_all(self) -> None:
        """Build every lazy tool (schemas/list paths need them all)."""
        try:
            names = list(self._lazy.keys())
        except Exception:
            return
        for name in names:
            try:
                self._materialize(name)
            except Exception:
                continue

    def get(self, name: str) -> Tool | None:
        tool = self._tools.get(name)
        if tool is None:
            tool = self._materialize(name)
        return tool

    def list(self) -> list[Tool]:
        self._materialize_all()
        return list(self._tools.values())

    def names(self) -> list[str]:
        try:
            with self._lock:
                return list(self._tools.keys()) + [n for n in self._lazy if n not in self._tools]
        except Exception:
            return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas for all registered tools."""
        self._materialize_all()
        out = []
        for tool in self._tools.values():
            out.append(
                {
                    "type": "function",
                    "function": {
                        # Strict validators (Nemotron 400 on "mcp:n") reject the
                        # whole request for one bad name — sanitize here (the
                        # actual wire path; base.tool_to_openai_schema is the
                        # legacy helper). Registry lookup still uses the raw
                        # registered name; the agent loop maps the sanitized
                        # reply back (see AgentLoop.run_tool fallback).
                        "name": _sanitize_wire_name(tool.name),
                        "description": tool.description,
                        "parameters": _provider_schema(tool.parameters),
                    },
                }
            )
        return out

    def register_fetch(self, resp: Any) -> None:
        """Track an in-flight webfetch response so an interrupt can close it."""
        with self._fetch_lock:
            self._active_fetches.append(resp)

    def unregister_fetch(self, resp: Any) -> None:
        """Stop tracking a fetch response (called when its read finishes)."""
        with self._fetch_lock:
            try:
                self._active_fetches.remove(resp)
            except ValueError:
                pass

    def abort_fetches(self) -> None:
        """Force-close every in-flight fetch this engine is running.

        Mirrors the provider-stream abort: ``socket.shutdown(SHUT_RDWR)`` wakes
        a reader blocked in ``iter_bytes`` so the tool sees the interrupt (the
        interrupt flag is already set by the caller) and returns immediately.
        """
        from ..util.net import force_close_response

        with self._fetch_lock:
            responses = list(self._active_fetches)
            self._active_fetches.clear()
        for resp in responses:
            try:
                force_close_response(resp)
            except Exception:
                pass

    def abort_servers(self) -> None:
        """Abort in-flight MCP calls (ESC path): wake blocked reads fast."""
        for server in list(self.mcp_servers):
            try:
                abort = getattr(server, "abort", None)
                if callable(abort):
                    abort()
            except Exception:
                pass

    def close(self) -> None:
        """Release the MCP server processes this registry holds.

        Servers are shared process-wide and reference-counted: this drops each
        one's reference, terminating it when the last holding registry (or a
        finished sub-agent) releases it. Safe to call more than once.
        """
        servers = list(self.mcp_servers)
        self.mcp_servers.clear()
        for server in servers:
            release = getattr(server, "release", None)
            if release is not None:
                try:
                    release()
                except Exception:  # pragma: no cover - best effort at teardown
                    pass


def _param(
    type_: str,
    description: str,
    required: bool = True,
    enum: list[str] | None = None,
    default: Any = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": type_, "description": description}
    if enum is not None:
        schema["enum"] = enum
    if default is not None:
        schema["default"] = default
    if not required:
        schema["optional"] = True
    return schema


def schema_with(params: dict[str, dict], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": params, "required": required}


def _sanitize_wire_name(name: str) -> str:
    """Wire-safe function name — delegates to the single shared helper."""
    try:
        from ..providers.base import sanitize_function_name as _san

        return _san(name)
    except Exception:  # pragma: no cover - import cycle fallback
        import re as _re

        safe = _re.sub(r"[^a-zA-Z0-9_-]", "_", str(name or ""))
        safe = safe.strip("_-") or "tool"
        return safe[:64]


def _provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a tool schema for a provider, dropping non-standard keywords.

    ``schema_with``/``_param`` add an ``"optional": True`` hint for the TUI
    (and tool builders); it is NOT part of JSON Schema, and strict provider
    tool-schema validators (Anthropic, some OpenAI-compatible gateways) reject
    unknown keywords. Return a copy so the stored tool definition keeps its
    hint while the wire format stays valid.
    """
    if not isinstance(schema, dict):
        return schema
    clean: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "optional":
            continue
        if key == "properties" and isinstance(value, dict):
            clean[key] = {
                name: _provider_schema(prop) if isinstance(prop, dict) else prop
                for name, prop in value.items()
            }
        else:
            clean[key] = value
    return clean
