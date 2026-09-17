"""file_wait tool: the patient watcher.

Blocks until a FILE reaches a state, then reports — completing the waiting
trio around background_task (start a job -> wait_for its output -> file_wait
for files OTHER processes write: logs from dev servers, exports, downloads,
pip's build artifacts...).

Modes (`until`):
- ``contains`` (default): wait until the file's content matches `pattern`
  (plain substring by default; set `is_regex` for a regex). New bytes are
  checked incrementally so huge growing logs stay cheap.
- ``change``  : wait until size or mtime differs from when the call started.
- ``quiet``   : wait until the file has been unchanged for `stable_seconds`
                ("the download finished writing").

Interruptible via the engine's interrupt hook (ESC), bounded by `timeout`
(ms). Missing files error immediately unless `wait_for_creation=true`.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .registry import Registry, Tool, schema_with

POLL_S = 0.25
DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000
MAX_TAIL_BYTES = 512 * 1024          # how much of a grown file we keep scanning
SNIPPET_RADIUS = 120


def _stat(path: Path) -> tuple[float, int] | None:
    try:
        st = path.stat()
        return st.st_mtime, st.st_size
    except OSError:
        return None


def _tail(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _snippet(text: str, index: int, length: int) -> str:
    start = max(0, index - SNIPPET_RADIUS)
    end = min(len(text), index + length + SNIPPET_RADIUS)
    piece = text[start:end].replace("\n", " ⏎ ")
    return ("…" if start > 0 else "") + piece + ("…" if end < len(text) else "")


def tool(registry: Registry | None = None) -> Tool:
    description = """Blocks until a FILE reaches a state you describe, then reports. The patient watcher for files written by OTHER processes: server logs, exports/downloads, build artifacts.

Modes (via `until`):
- contains (default): wait until content matches `pattern` (plain text unless is_regex=true). Checked incrementally as the file grows.
- change: wait until the file's size or modification time changes at all.
- quiet: wait until the file has been unchanged for `stable_seconds` — "the writer is done".

Rules:
- Missing file fails immediately unless wait_for_creation=true (then it waits for it to appear).
- Default timeout 120s (max 600s); on timeout you get the file's last known state.
- ESC-interruptible like bash."""

    def run(input: dict) -> dict:
        path_str = str(input.get("path") or "").strip()
        if not path_str:
            return {"output": "No path given.", "error": True}
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path

        until = str(input.get("until") or "contains").strip().lower()
        if until not in ("contains", "change", "quiet"):
            return {"output": f"Unknown until={until!r} (contains/change/quiet).",
                    "error": True}

        try:
            timeout_ms = int(input.get("timeout") or DEFAULT_TIMEOUT_MS)
        except (TypeError, ValueError):
            timeout_ms = DEFAULT_TIMEOUT_MS
        timeout_ms = max(500, min(timeout_ms, MAX_TIMEOUT_MS))
        try:
            stable_s = float(input.get("stable_seconds") or 2.0)
        except (TypeError, ValueError):
            stable_s = 2.0
        stable_s = max(0.5, min(stable_s, 60.0))

        pattern = str(input.get("pattern") or "")
        is_regex = bool(input.get("is_regex", False))
        rx: re.Pattern | None = None
        if until == "contains":
            if not pattern:
                return {"output": "contains mode needs a pattern.", "error": True}
            if is_regex:
                try:
                    rx = re.compile(pattern)
                except re.error as e:
                    return {"output": f"Invalid regex: {e}", "error": True}

        wait_for_creation = bool(input.get("wait_for_creation", False))
        start = time.monotonic()
        deadline = start + timeout_ms / 1000.0

        # initial state / creation gate
        stat = _stat(path)
        if stat is None and not wait_for_creation:
            return {
                "output": f"{path} does not exist yet "
                "(pass wait_for_creation=true to wait for it).",
                "error": True,
            }
        initial = stat or (0.0, 0)

        checker = getattr(registry, "interrupt_check", None) if registry else None
        scan_offset = 0           # incremental contains-scanning
        carried = ""              # partial line tail between polls
        quiet_since: float | None = None
        last_stat = stat

        while True:
            if checker is not None:
                try:
                    if checker():
                        return {
                            "output": f"Wait interrupted; {path.name} never reached the state.",
                            "error": True,
                        }
                except Exception:
                    pass

            now_stat = _stat(path)
            if now_stat is not None:
                mtime, size = now_stat
                if until == "change":
                    if now_stat != initial:
                        return {
                            "output": (
                                f"{path} changed: size {initial[1]} -> {size}, "
                                f"waited {_elapsed(deadline)}s"
                            ),
                            "metadata": {"size": size, "mode": "change"},
                        }
                elif until == "quiet":
                    if last_stat != now_stat or quiet_since is None:
                        quiet_since = time.monotonic()
                        last_stat = now_stat
                    elif time.monotonic() - quiet_since >= stable_s:
                        return {
                            "output": (
                                f"{path} quiet for {stable_s:g}s "
                                f"(size {size}), waited {_elapsed(deadline)}s"
                            ),
                            "metadata": {"size": size, "mode": "quiet"},
                        }
                else:  # contains — incremental scan of new bytes only
                    if size > scan_offset or scan_offset == 0:
                        with _read_lock(path, scan_offset, size):
                            pass
                        new_text = _read_from(path, min(scan_offset, size))
                        scan_offset = size
                        if new_text:
                            carried += new_text
                            # keep only the tail in case the pattern straddles
                            if len(carried) > MAX_TAIL_BYTES:
                                carried = carried[-MAX_TAIL_BYTES // 2:]
                            hay = carried
                            found_at = -1
                            needle = ""
                            if rx is not None:
                                # Timed search: a tricky pattern on a fast-growing
                                # tail must not spike CPU per poll — 0.2s budget,
                                # then keep waiting for more (cheaper) data.
                                import time as _time

                                _t0 = _time.monotonic()
                                try:
                                    m = rx.search(hay)
                                except Exception:
                                    m = None
                                if _time.monotonic() - _t0 > 0.2:
                                    carried = hay[-(4096):]
                                    last_stat = now_stat
                                    continue
                                if m:
                                    found_at = m.start()
                                    needle = m.group(0)
                            else:
                                found_at = hay.find(pattern)
                                needle = pattern
                            if found_at >= 0:
                                return {
                                    "output": (
                                        f"{path.name}: pattern found after "
                                        f"{_elapsed(deadline)}s:\n"
                                        f"…{_snippet(hay, found_at, len(needle))}…"
                                    ),
                                    "metadata": {
                                        "size": size,
                                        "mode": "contains",
                                        "match": needle[:200],
                                    },
                                }
                            # keep only the possible straddle region
                            keep = max(len(needle) - 1, 0) if needle else 0
                            carried = hay[-(keep * 2 + 4096):]
                last_stat = now_stat
            elif wait_for_creation and initial == (0.0, 0) and _stat(path) is not None:
                continue  # loop again; next iteration handles it as existing

            if time.monotonic() >= deadline:
                tail_note = ""
                if until == "contains" and now_stat is not None and now_stat[1]:
                    tail = _tail(path, 300)
                    tail_note = f"\nLast bytes: …{tail.strip()[-200:]}"
                elif now_stat is not None:
                    tail_note = f"\nCurrent size: {now_stat[1]} bytes."
                return {
                    "output": (
                        f"Timeout after {timeout_ms / 1000:.0f}s waiting for "
                        f"'{until}' on {path}.{tail_note}"
                    ),
                    "error": True,
                    "metadata": {"timed_out": True},
                }
            time.sleep(POLL_S)

    return Tool(
        name="file_wait",
        description=description,
        parameters=schema_with(
            {
                "path": {"type": "string", "description": "File to watch."},
                "until": {
                    "type": "string",
                    "enum": ["contains", "change", "quiet"],
                    "description": "State to wait for (default contains).",
                    "optional": True,
                },
                "pattern": {
                    "type": "string",
                    "description": "Substring/regex to wait for (contains mode).",
                    "optional": True,
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Treat pattern as regex.",
                    "optional": True,
                },
                "timeout": {
                    "type": "number",
                    "description": "Milliseconds to wait (default 120000, max 600000).",
                    "optional": True,
                },
                "stable_seconds": {
                    "type": "number",
                    "description": "Unchanged duration for quiet mode (default 2s).",
                    "optional": True,
                },
                "wait_for_creation": {
                    "type": "boolean",
                    "description": "Wait for a not-yet-existing file instead of failing.",
                    "optional": True,
                },
            },
            ["path"],
        ),
        run=run,
        permission="file_wait",
    )


class _read_lock:
    """No-op context kept for interface symmetry (single-threaded polling)."""

    def __init__(self, *_a, **_k) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _read_from(path: Path, offset: int) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(MAX_TAIL_BYTES)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _elapsed(deadline: float) -> str:
    total = getattr(_elapsed, "_start", None)
    return "?.?"
