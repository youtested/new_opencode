"""Task tool: sub-agent in its own session (mirrors opencode task).

RULES (full guidance for maintainers; the sent schema is short):
- Lazy spawner from the registry; nests naturally; headless gives an error.
- Give an isolated, self-contained prompt + short title. Long/complex or
  parallelizable work only. Types: build/plan/explore/general + any custom
  specialist (web-agent, search-agent, test-agent, phone-agent, edit-agent).
- background=true: the child runs detached (max ONE at a time); the call
  returns a session id at once and you keep working. task_status peeks,
  task_read collects (waits), task_stop kills. Finished-but-unread replies
  fold into history at turn end, so nothing is lost. NEVER start+wait
  back-to-back: start, do other work, THEN collect.
"""

from __future__ import annotations

from .registry import Registry, Tool, schema_with


def tool(registry: Registry) -> Tool:
    def run(arguments: dict) -> dict:
        spawner = getattr(registry, "task_spawner", None)
        if spawner is None:
            return {
                "output": "task tool is unavailable here (no sub-agent runtime).",
                "error": True,
            }
        return spawner(arguments)

    return Tool(
        name="task",
        description=(
            "Launch a sub-agent (own session) for long/complex/parallel work. "
            "Self-contained prompt + short title. Types: build/plan/explore/general "
            "or specialists (web-agent, search-agent, test-agent, phone-agent, edit-agent). "
            "background=true runs it detached (max 1): returns a session id at once, "
            "keep working, then task_status/task_read/task_stop."
        ),
        parameters=schema_with(
            {
                "prompt": {
                    "type": "string",
                    "description": "Task for the sub-agent",
                },
                "description": {
                    "type": "string",
                    "description": "Short title",
                },
                "subagent_type": {
                    "type": "string",
                    "description": "build, plan, explore, general, or a specialist name",
                    "enum": ["build", "plan", "explore", "general",
                             "web-agent", "search-agent", "test-agent",
                             "phone-agent", "edit-agent"],
                },
                "background": {
                    "type": "boolean",
                    "description": "Run detached; returns session id at once (max 1)",
                    "optional": True,
                },
                "action": {
                    "type": "string",
                    "description": "status, read, stop, or list detached agents",
                    "enum": ["status", "read", "stop", "list"],
                    "optional": True,
                },
                "session_id": {
                    "type": "string",
                    "description": "Detached agent id (default: the running one)",
                    "optional": True,
                },
            },
            required=["prompt"],
        ),
        run=run,
    )
