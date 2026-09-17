"""find_symbols tool: code navigation (definitions, callers, deps).

RULES (full guidance for maintainers; the sent schema is short):
- Verbs: def/callers/refs/deps/imports/symbols (+ bare word = def).
- Indexed once, cached, refreshed incrementally — faster than grep+read.
- Use BEFORE editing: jump to definition, check callers.
"""
from __future__ import annotations

from pathlib import Path

from .registry import Tool, schema_with

# ponytail: index.engine (~80ms: ast walkers + ctags probe) loads on first
# find_symbols call, not at registry build. _DEFAULT_LIMIT mirrors
# engine.MAX_RESULTS but stays import-light; the engine import below is the
# single source of truth at call time.
_DEFAULT_LIMIT = 60


def tool() -> Tool:
    description = """Name search across files (def/callers/refs/deps/imports/symbols). SECOND choice: if you have file+line use lsp first (0.4ms exact). Use this for name-only search, then read symbol= for the block. grep is plain text only."""

    def run(input: dict) -> dict:
        q = str(input.get("query") or "").strip()
        if not q:
            return {
                "output": (
                    "Empty query. Try: 'def run', 'callers _atomic_write', "
                    "'deps tools/edit.py', 'imports write', 'symbols main.py'."
                ),
                "error": True,
            }
        root = input.get("root")
        kind = str(input.get("kind") or "")
        try:
            limit = int(input.get("limit") or _DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = _DEFAULT_LIMIT
        ignore = input.get("ignore")
        if isinstance(ignore, str):
            ignore = [ignore]
        force = bool(input.get("fresh", False))
        from ..index.engine import query as _query
        return _query(
            q,
            root=Path(root) if root else None,
            kind=kind,
            limit=max(1, min(limit, 200)),
            ignore_extra=[str(x) for x in (ignore or [])],
            force=force,
        )

    return Tool(
        name="find_symbols",
        description=description,
        parameters=schema_with(
            {
                "query": {
                    "type": "string",
                    "description": (
                        'What to find, with an optional leading verb: '
                        '"def run", "callers _atomic_write", "refs Registry", '
                        '"deps open/opencode_py/tools/edit.py", '
                        '"imports opencode_py.tools.write", "symbols bash.py".'
                    ),
                },
                "root": {
                    "type": "string",
                    "description": (
                        "Project root to index (default: the current worktree)."
                    ),
                    "optional": True,
                },
                "kind": {
                    "type": "string",
                    "description": (
                        'Optional definition-kind filter for "def" queries: '
                        "function, method, class, variable, constant."
                    ),
                    "optional": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to show (default 60).",
                    "optional": True,
                },
                "ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra directory names to skip while indexing.",
                    "optional": True,
                },
                "fresh": {
                    "type": "boolean",
                    "description": "Force a full rebuild of the index (default false).",
                    "optional": True,
                },
            },
            ["query"],
        ),
        run=run,
        permission="find_symbols",
    )