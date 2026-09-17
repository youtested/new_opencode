"""verify tool: syntax-check edited files before claiming done.

RULES (full guidance for maintainers; the sent schema is short):
- Tracks edit/write/apply_patch touches; one check covers all. Python is
  compiled, JSON/TOML/YAML parsed, shell gets bash -n — never executed.
- Green implicit check clears tracking; failures persist for fix + re-check.
- Only say done when green.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from .registry import Tool, schema_with

_LOCK = threading.Lock()
# absolute path -> which tool touched it ("edit"/"write"/"apply_patch")
_TRACKED: dict[str, str] = {}
_TRACK_ORDER: list[str] = []

MAX_TRACKED = 200


def track(path: str | Path, source: str) -> None:
    """Record a file the model just wrote/edited (called by other tools)."""
    try:
        p = Path(path).resolve()
    except OSError:
        return
    with _LOCK:
        if p.as_posix() not in _TRACKED:
            _TRACK_ORDER.append(p.as_posix())
            while len(_TRACK_ORDER) > MAX_TRACKED:
                old = _TRACK_ORDER.pop(0)
                _TRACKED.pop(old, None)
        _TRACKED[p.as_posix()] = source


def untrack(path: str | Path) -> None:
    """Forget a file (e.g. apply_patch deleted it)."""
    with _LOCK:
        key = Path(path).resolve().as_posix()
        _TRACKED.pop(key, None)
        if key in _TRACK_ORDER:
            _TRACK_ORDER.remove(key)


def tracked() -> list[str]:
    with _LOCK:
        return list(_TRACK_ORDER)


def _clear_tracked() -> int:
    with _LOCK:
        n = len(_TRACK_ORDER)
        _TRACKED.clear()
        _TRACK_ORDER.clear()
        return n


# ---------------------------------------------------------------------------
# per-file checks
# ---------------------------------------------------------------------------

def _check_python(path: Path) -> tuple[bool, str]:
    """Compile-only syntax check: catches broken code WITHOUT running it."""
    try:
        src = path.read_bytes()
    except OSError as e:
        return False, f"unreadable: {e}"
    try:
        compile(src, str(path), "exec")
    except SyntaxError as e:
        where = f"line {e.lineno}" + (f", col {e.offset}" if e.offset else "")
        return False, f"syntax error at {where}: {e.msg}"
    except ValueError as e:
        return False, f"compile error: {e}"
    except (OverflowError, TypeError) as e:  # pragma: no cover - exotic
        return False, f"compile error: {e}"
    return True, "syntax OK"


def _check_json(path: Path) -> tuple[bool, str]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as e:
        return False, f"not valid UTF-8: {e}"
    except ValueError as e:
        return False, f"invalid JSON: {e}"
    except OSError as e:
        return False, f"unreadable: {e}"
    return True, "valid JSON"


def _check_toml(path: Path) -> tuple[bool, str]:
    try:
        import tomllib
    except ImportError:
        return True, "not checked (tomllib unavailable)"
    try:
        with path.open("rb") as fh:
            tomllib.load(fh)
    except ValueError as e:
        return False, f"invalid TOML: {e}"
    except OSError as e:
        return False, f"unreadable: {e}"
    return True, "valid TOML"


def _check_yaml(path: Path) -> tuple[bool, str]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return True, "not checked (PyYAML not installed)"
    try:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    except ValueError as e:  # yaml raises its own errors subclassing ValueError
        return False, f"invalid YAML: {e}"
    except OSError as e:
        return False, f"unreadable: {e}"
    return True, "valid YAML"


def _check_shell(path: Path) -> tuple[bool, str]:
    bash = shutil.which("bash")
    if not bash:
        return True, "not checked (bash not found)"
    try:
        proc = subprocess.run(
            [bash, "-n", str(path)], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as e:
        return True, f"not checked ({e})"
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = msg[-1] if msg else f"exit {proc.returncode}"
        return False, f"syntax error: {detail}"
    return True, "syntax OK"


def _check_one(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {"file": path_str, "ok": False, "note": "missing (deleted?)"}
    if not path.is_file():
        return {"file": path_str, "ok": False, "note": "not a regular file"}
    ext = path.suffix.lower()
    if ext == ".py":
        ok, note = _check_python(path)
        kind = "python"
    elif ext == ".json":
        ok, note = _check_json(path)
        kind = "json"
    elif ext == ".toml":
        ok, note = _check_toml(path)
        kind = "toml"
    elif ext in (".yaml", ".yml"):
        ok, note = _check_yaml(path)
        kind = "yaml"
    elif ext in (".sh", ".bash"):
        ok, note = _check_shell(path)
        kind = "shell"
    else:
        ok, note, kind = True, "no checker for this type", "other"
    return {"file": str(path), "ok": ok, "note": note, "kind": kind}


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _action_check(paths: list[str] | None) -> dict:
    if paths:
        targets = [str(p) for p in paths]
    else:
        targets = tracked()
    if not targets:
        return {
            "output": (
                "Nothing to verify yet: no files have been edited/written "
                "since the last clean verify (or pass explicit paths)."
            ),
            "metadata": {"checked": 0},
        }
    results = [_check_one(p) for p in targets]
    failed = [r for r in results if not r["ok"]]
    lines = []
    for r in results:
        mark = "✅" if r["ok"] else "❌"
        lines.append(f"{mark} {r['file']}  ({r.get('kind', '')}: {r['note']})")
    summary = f"{len(results)} checked: {len(results) - len(failed)} pass, {len(failed)} fail"
    lines.append(summary)

    cleared = 0
    if not failed and not paths:
        # only auto-clear after a fully green implicit check
        cleared = _clear_tracked()

    out = "\n".join(lines)
    if failed:
        out += "\nFix the failing files, then verify again."
    elif cleared:
        out += f"\nAll green — tracked list cleared ({cleared} file(s))."
    return {
        "output": out,
        "error": bool(failed),
        "metadata": {
            "checked": len(results),
            "failed": len(failed),
            "files": results,
        },
    }


def tool() -> Tool:
    description = """Syntax-check edited files (never executed). check (default) or reset. Green before done."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "check").strip().lower()
        raw_paths = input.get("paths")
        if isinstance(raw_paths, str):
            paths = [raw_paths]
        elif isinstance(raw_paths, list):
            paths = [str(p) for p in raw_paths]
        else:
            paths = None
        if action == "reset":
            n = _clear_tracked()
            return {"output": f"Tracked list cleared ({n} file(s)).",
                    "metadata": {"cleared": n}}
        if action != "check":
            return {"output": f"Unknown action {action!r} (want check or reset).",
                    "error": True}
        return _action_check(paths)

    return Tool(
        name="verify",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": ["check", "reset"],
                    "description": "check (default) or reset",
                    "optional": True,
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit files",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="verify",
    )