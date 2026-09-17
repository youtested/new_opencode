"""device tool: Termux phone controls.

RULES (full guidance for maintainers; the sent schema is short):
- wake_lock before multi-minute jobs (survives doze), wake_unlock after.
- battery (sysfs, no API needed) before long jobs; vibrate to signal.
- Missing pieces report install commands instead of failing silently.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading

from .registry import Tool, schema_with

_SYSFS_POWER = "/sys/class/power_supply"

_WAKE_HELD = threading.Event()

_ACTIONS = ("wake_lock", "wake_unlock", "battery", "vibrate")


def _run_cmd(args: list[str], timeout: float = 15.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return False, f"{args[0]} not found"
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or f"{args[0]} exited {proc.returncode}"
    return True, proc.stdout


def _missing_api_hint(binary: str) -> dict:
    pkg = {
        "termux-wake-lock": "termux-tools (usually preinstalled)",
        "termux-battery-status": "termux-api",
        "termux-vibrate": "termux-api",
    }.get(binary, "termux-api")
    return {
        "output": (
            f"{binary} is not installed. Fix: `pkg install {pkg.split(' ')[0]}`"
            + (" and install the Termux:API app" if pkg.startswith("termux-api") else "")
            + "."
        ),
        "error": True,
    }


def _action_wake(hold: bool) -> dict:
    binary = "termux-wake-lock" if hold else "termux-wake-unlock"
    ok, msg = _run_cmd([binary])
    if not ok:
        return _missing_api_hint(binary)
    if hold:
        _WAKE_HELD.set()
    else:
        _WAKE_HELD.clear()
    state = "HELD — screen-off no longer freezes background jobs" if hold else \
        "released — Android may doze when idle"
    return {
        "output": f"Wake lock {'acquired' if hold else 'released'}: {state}.",
        "metadata": {"held": _WAKE_HELD.is_set()},
    }


def _battery_from_sysfs() -> dict | None:
    """Kernel fallback: /sys/class/power_supply/battery/{capacity,status}."""
    import os
    from pathlib import Path

    base = Path(_SYSFS_POWER)
    if not base.is_dir():
        return None
    best: dict | None = None
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return None
    for node in entries:
        if not node.is_dir():
            continue
        cap_f = node / "capacity"
        if not cap_f.exists():
            continue
        try:
            pct = int(cap_f.read_text().strip())
        except (OSError, ValueError):
            continue
        status = ""
        status_f = node / "status"
        if status_f.exists():
            try:
                status = status_f.read_text().strip()
            except OSError:
                status = ""
        best = {"percentage": pct, "status": status or "unknown", "plugged": "", "source": "sysfs"}
    return best


def _action_battery() -> dict:
    if shutil.which("termux-battery-status"):
        ok, out = _run_cmd(["termux-battery-status"])
        if ok:
            try:
                data = json.loads(out.strip() or "{}")
            except ValueError:
                data = {}
            pct = data.get("percentage")
            status = str(data.get("status") or "unknown")
            plugged = str(data.get("plugged") or "")
            if isinstance(pct, int):
                line = f"Battery: {pct}% ({status}"
                if plugged and plugged.lower() not in ("unplugged", "none"):
                    line += f", {plugged}"
                line += ")"
                return {
                    "output": line,
                    "metadata": {"percentage": pct, "status": status,
                                 "plugged": plugged, "source": "termux-api"},
                }
    fallback = _battery_from_sysfs()
    if fallback is not None:
        pct, status = fallback["percentage"], fallback["status"]
        return {
            "output": f"Battery: {pct}% ({status})",
            "metadata": {"percentage": pct, "status": status,
                         "source": fallback.get("source", "sysfs")},
        }
    return _missing_api_hint("termux-battery-status")


def _action_vibrate(duration_ms: int) -> dict:
    duration_ms = max(50, min(int(duration_ms or 300), 5000))
    if not shutil.which("termux-vibrate"):
        return _missing_api_hint("termux-vibrate")
    ok, msg = _run_cmd(["termux-vibrate", "-d", str(duration_ms)])
    if not ok:
        return {"output": f"vibrate failed: {msg}", "error": True}
    return {"output": f"Vibrated for {duration_ms} ms.",
            "metadata": {"duration_ms": duration_ms}}


def tool() -> Tool:
    description = """Phone controls: wake_lock/wake_unlock (survive doze), battery, vibrate."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "").strip().lower()
        if action not in _ACTIONS:
            return {
                "output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True,
            }
        if action == "wake_lock":
            return _action_wake(True)
        if action == "wake_unlock":
            return _action_wake(False)
        if action == "battery":
            return _action_battery()
        try:
            duration = int(input.get("duration") or 300)
        except (TypeError, ValueError):
            duration = 300
        return _action_vibrate(duration)

    return Tool(
        name="device",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "wake_lock, wake_unlock, battery, vibrate",
                },
                "duration": {
                    "type": "integer",
                    "description": "Vibrate ms (default 300)",
                    "optional": True,
                },
            },
            ["action"],
        ),
        run=run,
        permission="device",
    )