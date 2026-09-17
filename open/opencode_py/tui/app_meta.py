"""Shared TUI app constants and startup helpers (split from app.py).

Pure module-level values only — no app state. Imported back into
``opencode_py.tui.app`` so every existing ``app.<name>`` reference keeps working.
"""

from __future__ import annotations

_MAX_LINES_CAP = 400  # screen_view text capture: rows before truncation
_MAX_WIDGET_LINES = 250  # screen_view widgets capture: tree rows before truncation
_DIALOG_TIMEOUT = 30.0  # seconds to wait for permission/question dialog before defaulting


def _prewarm_heavy_deps() -> None:
    """Import the engine chain + provider internals on a background thread so
    they don't sit on the first-turn path (best effort — lazy imports re-run
    normally if any of this fails)."""
    try:
        import opencode_py.agent.loop  # noqa: F401
        import opencode_py.agent.compaction  # noqa: F401
        import opencode_py.commands  # noqa: F401
        import opencode_py.tools  # noqa: F401
        import opencode_py.providers.zen  # noqa: F401
        import opencode_py.providers.openai_compat  # noqa: F401
        import opencode_py.providers.anthropic  # noqa: F401
    except Exception:  # pragma: no cover - best-effort warm-up
        pass


# Read-only slash commands allowed while a turn is running. Everything else
# (undo/clear/compact/agent/model/theme/...) mutates engine state or the running
# turn and is blocked until the current request finishes. `thinking` is safe:
# on/off/show/hide/last are pure UI, and an effort change only saves config +
# marks rotation dirty — the engine picks it up at the next provider step
# (see _stream), never mid-chunk.
_SAFE_WHILE_BUSY = {
    "help",
    "config",
    "permissions",
    "sessions",
    "ls",
    "models",
    "review",
    "connect",
    "resume",
    "continue",  # /resume's alias — the pair must behave identically
    "thinking",
    "mcp",  # list/test are read-only; add/remove save config + reload next turn
}

# Honest usage lines for arg-taking commands (the popup used to fabricate
# "Usage: /x [args]" even for no-arg commands like /help).
_COMMAND_USAGE = {
    "resume": "Usage: /resume <session-id>",
    "export": "Usage: /export [session-id]",
    "model": "Usage: /model <model-id>",
    "agent": "Usage: /agent build|plan|explore",
    "theme": "Usage: /theme <name>",
    "config": "Usage: /config print|validate",
    "connect": "Usage: /connect [provider]",
    "mcp": "Usage: /mcp [list|add <name> -- <command...>|remove <name>|test [name]]",
    "thinking": "Usage: /thinking [show|hide|last]",
}


def _probe_online() -> bool:
    """True when the provider route is reachable (reconnect watcher).

    Tiny raced probes (~3s max total): ANY HTTP status (even 500) means the
    route is alive — only transport exceptions mean still offline. Two hosts
    race so one dead provider can't fake an outage. Never raises.
    """
    urls = [
        "https://opencode.ai/zen/v1/models",
        "https://api.groq.com/openai/v1/models",
    ]
    try:
        import httpx
    except ImportError:
        return False

    def _hit(url: str) -> bool:
        try:
            httpx.get(url, timeout=2.5, follow_redirects=True)
            return True
        except Exception:
            return False

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(_hit, u) for u in urls]
            try:
                for fut in as_completed(futs, timeout=3.0):
                    try:
                        if fut.result():
                            return True
                    except Exception:
                        continue
            except Exception:
                pass
    except Exception:
        pass
    return False
