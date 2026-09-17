"""todowrite tool: in-session todo list.

RULES (full guidance for maintainers; the sent schema is short):
- 3+ steps or user-asked lists only; single trivial tasks skip it.
- Exactly ONE in_progress; complete only when truly done + verified.
- Specific actionable items; blocked -> follow-up todo; keep commands verbatim.
"""

from __future__ import annotations

import json

from .registry import Tool, schema_with

VALID_STATUS = {"pending", "in_progress", "completed", "cancelled"}
VALID_PRIORITY = {"high", "medium", "low"}


def _todowrite(todos: list[dict], state: dict) -> dict:
    normalized = []
    for t in todos:
        if not isinstance(t, dict):
            continue
        status = t.get("status", "pending")
        priority = t.get("priority", "medium")
        if status not in VALID_STATUS:
            status = "pending"
        if priority not in VALID_PRIORITY:
            priority = "medium"
        normalized.append(
            {"content": t.get("content", ""), "status": status, "priority": priority}
        )
    state["todos"] = normalized
    remaining = sum(1 for t in normalized if t["status"] not in ("completed", "cancelled"))
    return {
        "output": json.dumps(normalized, indent=2),
        "metadata": {"todos": normalized, "remaining": remaining},
    }


def tool(state: dict | None = None) -> Tool:
    state = state or {}
    description = """Task list for multi-step work. One in_progress at a time; complete only when truly done."""

    def run(input: dict) -> dict:
        return _todowrite(input.get("todos", []), state)

    return Tool(
        name="todowrite",
        description=description,
        parameters=schema_with(
            {
                "todos": {
                    "type": "array",
                    "description": "Todo list",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Task"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                            },
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["content", "status", "priority"],
                    },
                }
            },
            ["todos"],
        ),
        run=run,
        permission="todowrite",
    )
