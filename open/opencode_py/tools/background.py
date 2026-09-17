"""background_task: long-running jobs and dev servers that outlive a tool call.

RULES (full guidance for maintainers; the sent schema is short):
- bash kills everything at call end (120s cap); THESE persist across turns
  (servers stay up, installs finish) and die with the app (atexit reaps).
- start returns a task id immediately; a reader thread drains output to a
  bounded ring buffer so the child never blocks on a full pipe.
- read = NEW output since last read; wait capped at 30s per call (job
  keeps running); stop kills the process group; status/list report.
- NEVER start+wait back-to-back: start, do other work, THEN poll.
"""

from __future__ import annotations

import atexit
import os
import select
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..globals import Path as GPath
from .bash import _kill_proc_tree
from .registry import Registry, Tool, schema_with

MAX_RUNNING = 8          # concurrent live tasks before start() refuses
MAX_BUFFER = 256 * 1024  # per-task output kept (oldest dropped beyond this)
READ_CHUNK = 65536
MAX_READ_RETURN = 50 * 1024
DEFAULT_WAIT_MS = 120_000


@dataclass
class Task:
    id: str
    command: str
    workdir: str
    proc: Any = None
    pgid: int | None = None
    started: float = field(default_factory=time.time)
    finished_at: float | None = None
    exit_code: int | None = None
    stopped: bool = False
    buffer: bytearray = field(default_factory=bytearray)
    dropped: int = 0        # bytes evicted from the front of `buffer`
    total: int = 0          # total bytes the task has produced
    read_cursor: int = 0    # absolute offset the consumer last read up to
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None
    reserved: bool = False  # slot held pre-spawn; counts against MAX_RUNNING
    # Set exactly once when the task reaches a terminal state (natural exit
    # in _drain, or stop()). wait() blocks on this instead of polling
    # running() every 100ms — instant wakeup, zero busy-loop on the worker.
    done: threading.Event = field(default_factory=threading.Event)

    def runtime_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


_TASKS: dict[str, Task] = {}
_REG_LOCK = threading.RLock()
_COUNTER = 0


def _new_id() -> str:
    global _COUNTER
    with _REG_LOCK:
        _COUNTER += 1
        return f"bg{_COUNTER}"


def _running_count() -> int:
    with _REG_LOCK:
        items = list(_TASKS.values())
    n = 0
    for t in items:
        try:
            if t.running() or getattr(t, "reserved", False):
                n += 1
        except Exception:
            pass
    return n


# ---------------------------------------------------------------------------
# reader thread
# ---------------------------------------------------------------------------

def _drain(task: Task) -> None:
    proc = task.proc
    stdout = proc.stdout if proc else None
    if stdout is None:
        task.done.set()
        return
    try:
        fd = stdout.fileno()
    except (OSError, ValueError):
        fd = None
    try:
        if fd is not None:
            os.set_blocking(fd, False)
    except OSError:
        pass
    while True:
        if fd is None:
            break
        try:
            ready, _, _ = select.select([fd], [], [], 0.2)
        except (OSError, ValueError):
            break
        if not ready:
            if proc.poll() is not None:
                if task.stopped:
                    break
                try:
                    data = os.read(fd, READ_CHUNK)
                except (BlockingIOError, OSError):
                    break
                if not data:
                    break
                with task.lock:
                    task.buffer.extend(data)
                    task.total += len(data)
                    overflow = len(task.buffer) - MAX_BUFFER
                    if overflow > 0:
                        del task.buffer[:overflow]
                        task.dropped += overflow
                continue
            continue
        try:
            data = os.read(fd, READ_CHUNK)
        except BlockingIOError:
            continue
        except OSError:
            break
        if not data:
            break
        with task.lock:
            task.buffer.extend(data)
            task.total += len(data)
            overflow = len(task.buffer) - MAX_BUFFER
            if overflow > 0:
                del task.buffer[:overflow]
                task.dropped += overflow
    code = proc.poll() if proc else None
    if code is not None:
        task.exit_code = code
        task.finished_at = time.time()
    task.done.set()


# ---------------------------------------------------------------------------
# lifecycle helpers
# ---------------------------------------------------------------------------

def _redact_command(command: str) -> str:
    # Never echo raw commands: they carry tokens/keys (--token X, api_key=Y)
    # that would leak into the transcript, status screen, and saved session.
    try:
        import re as _re

        red = _re.sub(r"(?i)(token|key|secret|password|passwd|auth)[=:]\s*\S+", r"\1=<redacted>", command)
        red = _re.sub(r"(?i)(--token|--api-key|--apikey|--password|--secret)\s+\S+", r"\1 <redacted>", red)
        red = _re.sub(r"(?i)\b(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|xox[bpas]-[A-Za-z0-9-]{8,})\b", "<redacted>", red)
        return red
    except Exception:
        return command


def start(command: str, workdir: str | None = None) -> dict[str, Any]:
    command = (command or "").strip()
    if not command:
        return {"output": "No command given.", "error": True}
    # Reserve the slot BEFORE spawning: the old check-then-Popen window let
    # parallel starts exceed MAX_RUNNING (too many processes, RAM death).
    # A placeholder task holds the slot; spawn failure releases it.
    with _REG_LOCK:
        if _running_count() >= MAX_RUNNING:
            return {
                "output": (
                    f"Refusing to start: {MAX_RUNNING} background tasks are "
                    "already running. Stop or wait for one first."
                ),
                "error": True,
            }
        slot = Task(id=_new_id(), command=command, workdir=str(workdir or ""), reserved=True)
        _TASKS[slot.id] = slot

    cwd = Path(workdir).resolve() if workdir else Path.cwd()
    if not cwd.exists():
        with _REG_LOCK:
            _TASKS.pop(slot.id, None)
        return {"output": f"Workdir does not exist: {cwd}", "error": True}

    shell = os.environ.get("SHELL", "/bin/sh")
    import subprocess

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=shell,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=os.environ.copy(),
            start_new_session=os.name == "posix",
        )
    except Exception as e:
        with _REG_LOCK:
            _TASKS.pop(slot.id, None)
        return {"output": f"Failed to start: {e}", "error": True}

    slot.command = command
    slot.workdir = str(cwd)
    slot.proc = proc
    slot.reserved = False
    task = slot
    if os.name == "posix" and hasattr(os, "getpgid"):
        try:
            task.pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError, OSError):
            task.pgid = None
    task.thread = threading.Thread(
        target=_drain, args=(task,), name=f"bgtask-{task.id}", daemon=True
    )
    task.thread.start()

    with _REG_LOCK:
        # prune old finished tasks so the registry never grows unbounded
        finished = [t for t in _TASKS.values() if not t.running()]
        for old in sorted(finished, key=lambda t: t.finished_at or 0)[:-32]:
            _TASKS.pop(old.id, None)
        _TASKS[task.id] = task

    return {
        "output": (
            f"Started background task {task.id} (pid {proc.pid}).\n"
            f"Command: {_redact_command(command)}\n"
            f"Use action=read task_id={task.id} for new output,\n"
            f"action=status / action=wait / action=stop likewise."
        ),
        "metadata": {"task_id": task.id, "pid": proc.pid},
    }


def adopt(proc, pgid, command, workdir, seed: bytes = b"", started: float | None = None) -> dict[str, Any]:
    """Adopt a live Popen (e.g. bash timeout) so it keeps running. No re-run."""
    with _REG_LOCK:
        if _running_count() >= MAX_RUNNING:
            return {
                "output": (
                    f"Refusing to adopt: {MAX_RUNNING} background tasks are "
                    "already running. Stop or wait for one first."
                ),
                "error": True,
            }
    try:
        pid = proc.pid
    except Exception:
        return {"output": "Cannot adopt: process gone.", "error": True}
    try:
        if proc.poll() is not None:
            return {"output": "Cannot adopt: process already exited.", "error": True}
    except Exception:
        return {"output": "Cannot adopt: process gone.", "error": True}
    task = Task(
        id=_new_id(),
        command=(command or "(adopted)").strip() or "(adopted)",
        workdir=str(workdir or Path.cwd()),
        proc=proc,
        pgid=pgid,
        started=started or time.time(),
    )
    if seed:
        raw_seed = bytes(seed)
        keep = raw_seed[-MAX_BUFFER:] if len(raw_seed) > MAX_BUFFER else raw_seed
        task.buffer.extend(keep)
        task.total = len(raw_seed)
        task.dropped = len(raw_seed) - len(keep)
        task.read_cursor = len(raw_seed)
    task.thread = threading.Thread(
        target=_drain, args=(task,), name=f"bgtask-{task.id}", daemon=True
    )
    task.thread.start()
    with _REG_LOCK:
        finished = [t for t in _TASKS.values() if not t.running()]
        for old in sorted(finished, key=lambda t: t.finished_at or 0)[:-32]:
            _TASKS.pop(old.id, None)
        _TASKS[task.id] = task
    return {
        "output": f"Adopted live process as background task {task.id} (pid {pid}).",
        "metadata": {"task_id": task.id, "pid": pid},
    }


def _get(task_id: str) -> Task | None:
    with _REG_LOCK:
        return _TASKS.get((task_id or "").strip())


def status(task_id: str) -> dict[str, Any]:
    task = _get(task_id)
    if task is None:
        known = ", ".join(sorted(_TASKS)) or "(none)"
        return {"output": f"No such task {task_id!r}. Known: {known}", "error": True}
    lines = [
        f"Task {task.id}: {'RUNNING' if task.running() else 'FINISHED'}",
        f"Command: {_redact_command(task.command)}",
        f"Runtime: {task.runtime_s():.1f}s",
    ]
    if task.exit_code is not None:
        lines.append(f"Exit code: {task.exit_code}")
    elif not task.running():
        lines.append("Exit code: (shell gone, reaping)")
    lines.append(f"Output so far: {task.total} bytes ({len(task.buffer)} buffered)")
    return {"output": "\n".join(lines), "metadata": {
        "task_id": task.id,
        "running": task.running(),
        "exit_code": task.exit_code,
        "total_bytes": task.total,
    }}


TAIL_DEFAULT = 8 * 1024  # tail mode default: last 8KB (status line, errors, failures live here)


def tail(task_id: str, limit_bytes: int = TAIL_DEFAULT) -> dict[str, Any]:
    """Last N bytes of the whole buffer (no history re-send); advances cursor to end."""
    task = _get(task_id)
    if task is None:
        known = ", ".join(sorted(_TASKS)) or "(none)"
        return {"output": f"No such task {task_id!r}. Known: {known}", "error": True}
    with task.lock:
        buf_start_abs = task.dropped
        end_abs = buf_start_abs + len(task.buffer)
        want = max(1024, int(limit_bytes))
        start_abs = max(buf_start_abs, end_abs - want)
        data = bytes(task.buffer[start_abs - buf_start_abs:])
        omitted = start_abs - buf_start_abs
        task.read_cursor = end_abs
    text = data.decode("utf-8", errors="replace")
    head = [f"(... {omitted} older buffered bytes omitted — tail only)"] if omitted > 0 else []
    state = "still running" if task.running() else f"finished, exit code {task.exit_code}"
    body = text if text.strip() else "(no output yet)"
    return {"output": ("\n".join(head) + "\n" if head else "") + body + f"\n[task {task.id}: {state}, {task.total} bytes total]",
            "metadata": {"task_id": task.id, "bytes_returned": len(data),
                         "running": task.running(), "exit_code": task.exit_code,
                         "total_bytes": task.total, "mode": "tail"}}


def read(task_id: str, limit_bytes: int = MAX_READ_RETURN) -> dict[str, Any]:
    task = _get(task_id)
    if task is None:
        known = ", ".join(sorted(_TASKS)) or "(none)"
        return {"output": f"No such task {task_id!r}. Known: {known}", "error": True}
    with task.lock:
        buf_start_abs = task.dropped
        avail_start = max(task.read_cursor, buf_start_abs)
        skipped = avail_start - task.read_cursor
        end_abs = buf_start_abs + len(task.buffer)
        chunk_end = min(end_abs, avail_start + max(1024, int(limit_bytes)))
        data = bytes(task.buffer[avail_start - buf_start_abs : chunk_end - buf_start_abs])
        task.read_cursor = chunk_end
        remaining = end_abs - chunk_end
    text = data.decode("utf-8", errors="replace")
    head = []
    if skipped > 0:
        head.append(f"(skipped {skipped} older bytes no longer buffered)")
    state = "still running" if task.running() else (
        f"finished, exit code {task.exit_code}"
    )
    body = text if text.strip() else "(no new output)"
    tail_note = "" if remaining <= 0 else (
        f"\n({remaining} more bytes buffered — call read again)"
    )
    result_text = ("\n".join(head) + "\n" if head else "") + body + tail_note + (
        f"\n[task {task.id}: {state}]"
    )
    return {"output": result_text, "metadata": {
        "task_id": task.id,
        "bytes_returned": len(data),
        "remaining_buffered": max(0, remaining),
        "running": task.running(),
        "exit_code": task.exit_code,
    }}


WAIT_SOFT_CAP_MS = 30_000


def wait(task_id: str, timeout_ms: int = DEFAULT_WAIT_MS,
         is_interrupted=None) -> dict[str, Any]:
    task = _get(task_id)
    if task is None:
        known = ", ".join(sorted(_TASKS)) or "(none)"
        return {"output": f"No such task {task_id!r}. Known: {known}", "error": True}
    # Long agent-side waits freeze the turn: the model sits idle instead of
    # doing other work. Cap a single wait at 30s so a `start` + `wait` pattern
    # returns control quickly; the caller can `read`/`wait` again if needed.
    # The task itself keeps running — only this blocking call is shortened.
    asked_ms = max(0.1, timeout_ms / 1000.0) * 1000.0
    capped_ms = min(timeout_ms, WAIT_SOFT_CAP_MS)
    was_capped = capped_ms < timeout_ms
    deadline = time.monotonic() + max(0.1, capped_ms / 1000.0)
    interrupted = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        # Event wait wakes the instant the task finishes (no 100ms poll
        # granularity); short slices keep ESC responsive.
        if task.done.wait(timeout=min(0.2, remaining)):
            break
        if is_interrupted is not None:
            try:
                if is_interrupted():
                    interrupted = True
                    break
            except Exception:
                pass
    if interrupted:
        return {"output": f"Wait interrupted; task {task.id} still running.",
                "metadata": {"task_id": task.id, "running": True}}
    res = status(task_id)
    if task.running():
        hint = ""
        if was_capped:
            hint = (f"\nNote: wait was capped at {capped_ms / 1000:.0f}s "
                    f"(asked {asked_ms / 1000:.0f}s) so the turn stays responsive — "
                    f"use action=read task_id={task.id} to poll output, or wait again.")
        res["output"] = (
            f"Timeout after {min(timeout_ms, capped_ms) / 1000:.0f}s; task {task.id} STILL RUNNING.\n"
            + res["output"] + hint
        )
        res["metadata"]["timed_out"] = True
        return res
    # finished: include the last chunk of output as a tail
    with task.lock:
        tail = bytes(task.buffer[-4096:]).decode("utf-8", errors="replace")
    res["output"] += "\n--- last output ---\n" + (tail.strip() or "(empty)")
    return res


def stop(task_id: str) -> dict[str, Any]:
    task = _get(task_id)
    if task is None:
        known = ", ".join(sorted(_TASKS)) or "(none)"
        return {"output": f"No such task {task_id!r}. Known: {known}", "error": True}
    if not task.running():
        return {"output": f"Task {task.id} already finished (exit {task.exit_code})."}
    task.stopped = True
    _kill_proc_tree(task.proc, task.pgid)
    try:
        task.proc.wait(timeout=5)
    except Exception:
        pass
    code = task.proc.poll()
    task.exit_code = code if code is not None else -9
    task.finished_at = time.time()
    task.done.set()
    return {
        "output": f"Stopped task {task.id} ({_redact_command(task.command).splitlines()[0][:60]}).",
        "metadata": {"task_id": task.id, "stopped": True},
    }


def stop_all() -> int:
    with _REG_LOCK:
        try:
            ids = [tid for tid, t in list(_TASKS.items()) if t.running()]
        except Exception:
            return 0
    stopped = 0
    for tid in ids:
        try:
            res = stop(tid)
            if not res.get("error"):
                stopped += 1
        except Exception:
            continue
    return stopped


def list_tasks() -> dict[str, Any]:
    with _REG_LOCK:
        tasks = sorted(_TASKS.values(), key=lambda t: t.id)
    if not tasks:
        return {"output": "No background tasks."}
    lines = [f"{len(tasks)} background task(s):"]
    for t in tasks:
        state = "RUNNING" if t.running() else f"done(exit {t.exit_code})"
        cmd = t.command.replace("\n", " ")[:60]
        lines.append(f"  {t.id}  {state:<12} {t.runtime_s():7.1f}s  {cmd}")
    return {"output": "\n".join(lines), "metadata": {"count": len(tasks)}}


@atexit.register
def _kill_all_on_exit() -> None:  # pragma: no cover - process teardown
    with _REG_LOCK:
        snapshot = list(_TASKS.values())
    for task in snapshot:
        if task.running():
            task.stopped = True
            try:
                _kill_proc_tree(task.proc, task.pgid)
            except Exception:
                pass


_ACTIONS = ("start", "status", "read", "wait", "stop", "list")


def tool(registry: Registry | None = None) -> Tool:
    description = """Starts and manages LONG-RUNNING background jobs that keep running across turns.

Unlike bash (which kills everything when the call ends, hard 120s cap), these
tasks persist: pip installs, builds, test runs, and dev servers keep going
while the conversation continues. Output accumulates in a bounded buffer you
poll incrementally. Tasks die when the app itself exits.

Actions (via `action`):
- start (default): launch `command`, return a task_id immediately.
- read: NEW output since your last read (incremental, 50KB); mode='tail' returns last 8KB only (cheap poll — errors live at the tail).
- status: running/exited, exit code, runtime, output totals.
- wait: block up to `timeout` ms for task_id to finish, then report exit code + output tail.
  NEVER start+wait back-to-back: start the job, do other work (reads, edits,
  searches), THEN poll with read/status. A wait blocks the whole turn.
- stop: kill task_id's whole process group.
- list: all tasks with ids and states.

Rules of thumb:
- Anything expected to exceed ~90s belongs here, not in bash.
- Start servers here; verify them later with bash (curl ...) or read.
- A single wait is capped at 30s so the turn stays responsive; the job keeps
  running — poll again with read or wait if you need more.
- Always stop() servers/jobs you no longer need."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "start").strip().lower()
        if action not in _ACTIONS:
            return {
                "output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True,
            }
        checker = getattr(registry, "interrupt_check", None) if registry else None
        if action == "start":
            return start(str(input.get("command") or ""), input.get("workdir"))
        if action == "list":
            return list_tasks()
        task_id = str(input.get("task_id") or "")
        if action == "status":
            return status(task_id)
        if action == "read":
            mode = str(input.get("mode") or "new").strip().lower()
            if mode.startswith("tail"):
                try:
                    limit = int(input.get("limit") or TAIL_DEFAULT)
                except (TypeError, ValueError):
                    limit = TAIL_DEFAULT
                return tail(task_id, max(1024, min(limit, MAX_BUFFER)))
            try:
                limit = int(input.get("limit") or MAX_READ_RETURN)
            except (TypeError, ValueError):
                limit = MAX_READ_RETURN
            return read(task_id, max(2048, min(limit, MAX_BUFFER)))
        if action == "stop":
            return stop(task_id)
        try:
            timeout = int(input.get("timeout") or DEFAULT_WAIT_MS)
        except (TypeError, ValueError):
            timeout = DEFAULT_WAIT_MS
        return wait(task_id, timeout_ms=max(1000, min(timeout, 600_000)),
                    is_interrupted=checker)

    return Tool(
        name="background_task",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "start (default), read, status, wait, stop, list",
                    "optional": True,
                },
                "command": {
                    "type": "string",
                    "description": "Shell command (for start)",
                    "optional": True,
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory",
                    "optional": True,
                },
                "task_id": {
                    "type": "string",
                    "description": "Task id, e.g. bg3",
                    "optional": True,
                },
                "timeout": {
                    "type": "number",
                    "description": "Wait ms (default 120000, capped 30000)",
                    "optional": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max bytes per read (default 51200, tail mode default 8192)",
                    "optional": True,
                },
                "mode": {
                    "type": "string",
                    "description": "new (default, incremental) or tail (last N bytes, cheap poll)",
                    "enum": ["new", "tail"],
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="background_task",
    )