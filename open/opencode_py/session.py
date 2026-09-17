"""Session persistence: JSON files under the data dir.

A session stores: id, title, created/completed times, directory, provider,
model, agent, and the OpenAI-style message history. Auto-save after each turn.

Sub-agent sessions (spawned via the `task` tool) are regular sessions with a
`parent_id` pointing at the session that spawned them.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .globals import Path as GPath

# Serializes all session-body and index writes. save_session can be called
# from the UI thread (exit / save-all) and from the in-flight autosave worker
# thread (TUI streaming) — the lock keeps those writers from interleaving and
# lets a `should_write` guard run atomically with the write it protects.
_WRITE_LOCK = threading.Lock()

# Save fingerprints: session-id -> (msg count, last-msg hash, top-field hash).
# save_session skips the 4MB json.dumps when nothing changed since the last
# save — the in-turn autosave otherwise re-serializes the same body every
# tick. ponytail: O(1) fingerprint vs O(n) dump; a hash collision would skip
# a real change, so fingerprints use (count, last-hash, top-hash) triple —
# upgrade path is a full content hash if collisions ever matter.
_SAVE_PRINTS: dict[str, tuple] = {}


def _fingerprint(session: Session) -> tuple:
    """O(1) change fingerprint: (count, last-msg hash, top-field hash)."""
    try:
        msgs = session.messages or []
        n = len(msgs)
        last = msgs[-1] if msgs else {}
        try:
            last_s = json.dumps(last, sort_keys=True, default=str)
        except (TypeError, ValueError):
            last_s = str(last)
        top = (session.title, session.provider, session.model, session.agent,
               session.completed, session.directory)
        try:
            top_s = json.dumps(top, sort_keys=True, default=str)
        except (TypeError, ValueError):
            top_s = str(top)
        import hashlib as _hl

        return (n, _hl.md5(last_s.encode("utf-8", "replace")).hexdigest(),
                _hl.md5(top_s.encode("utf-8", "replace")).hexdigest())
    except Exception:
        return (-1, "", "")


def _as_number(value: Any, fallback: float = 0.0) -> float:
    """Coerce a persisted timestamp to a finite float, tolerant of garbage.

    A corrupt session file (crash mid-write, partial rename, hand-edit) can
    carry a string or object in `created`/`completed`; letting that leak into
    list sorting would crash the whole picker. NaN/±Inf pass a naive
    isinstance check but poison every comparison and explode
    ``datetime.fromtimestamp`` — they get the fallback too. Return the
    fallback instead.
    """
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else fallback
    try:
        f = float(str(value))
    except (TypeError, ValueError):
        return fallback
    return f if math.isfinite(f) else fallback


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_str(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    try:
        return str(value)
    except Exception:
        return fallback


class Session:
    def __init__(self, data: dict[str, Any] | Any, directory: str | None = None):
        # `data` comes from a JSON file the user could corrupt at any point;
        # never assume it's a dict — bad JSON yields None/list/str and would
        # crash every listener with AttributeError otherwise.
        data = _as_dict(data)
        self.id = _as_str(data.get("id"), uuid.uuid4().hex) or uuid.uuid4().hex
        self.title = _as_str(data.get("title"))
        self.created = _as_number(data.get("created"), time.time())
        self.completed = None if data.get("completed") is None else _as_number(data.get("completed"))
        self.directory = _as_str(data.get("directory")) or directory or ""
        self.provider = _as_str(data.get("provider"))
        self.model = _as_str(data.get("model"))
        self.agent = _as_str(data.get("agent"), "build") or "build"
        self.parent_id = data.get("parent_id")
        # Coerce every message to a dict: a hand-edited/corrupt body can carry
        # bare strings in `messages`, and every consumer (_first_user_text,
        # export, history_search) does msg.get(...) — one bad item used to
        # crash the whole picker permanently (the index was never written).
        self.messages: list[dict[str, Any]] = [
            m for m in (_as_dict(m) for m in _as_list(data.get("messages"))) if m
        ]
        self.metadata: dict[str, Any] = _as_dict(data.get("metadata"))
        # Lightweight listing extras (populated by the session index, unused by
        # resume/export which always load the full file).
        self.has_messages = bool(data.get("has_messages")) if "has_messages" in data else bool(self.messages)
        self.first_user_text = _as_str(data.get("first_user_text"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "completed": self.completed,
            "directory": self.directory,
            "provider": self.provider,
            "model": self.model,
            "agent": self.agent,
            "parent_id": self.parent_id,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @property
    def path(self) -> Path:
        return session_path(self.id)


def session_path(session_id: str) -> Path:
    # Percent-encode anything outside [A-Za-z0-9_-] so the mapping stays
    # injective: the old "strip bad chars" scheme folded "a/b", "a.b" and
    # "ab" onto the same file, letting one save silently overwrite another
    # and load_session hand back the wrong body. uuid4().hex ids (the normal
    # case) contain only hex chars, so existing files keep their names.
    safe = "".join(
        c if (c.isascii() and c.isalnum()) or c in "-_" else f"%{ord(c):02X}"
        for c in str(session_id)
    )
    if not safe:
        safe = "invalid"
    return GPath.sessions_dir() / f"{safe}.json"


def _fsync_dir(directory: Path) -> None:
    """Durably persist a rename/delete in ``directory``.

    A rename followed by an fsync of the *file* only is not enough: the
    directory entry swap can still be lost on a sudden power loss unless the
    directory itself is flushed. Best effort — some filesystems can't open a
    directory for fsync, and that's fine (the write itself is the critical
    part).
    """
    dfd = -1
    try:
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        except OSError:
            pass
    except OSError:
        pass
    finally:
        if dfd >= 0:
            try:
                os.close(dfd)
            except OSError:
                pass


def _write_durable(path: Path, text: str, *, file_sync: bool = True, dir_sync: bool = True) -> bool:
    """Write ``text`` to ``path`` atomically (tmp + rename). Returns True on success.

    A sudden phone reboot (process kill, power loss) loses everything still
    living in the OS page cache — ``Path.write_text`` was buffering the whole
    session body there, so an autosave right before a crash could leave the
    session file empty (0 bytes) or missing entirely. With ``file_sync`` the
    bytes are fsync'd to flash before the rename, and with ``dir_sync`` the
    rename itself is persisted too: after this returns True, the data survives
    any power cut. If the write/flush itself fails, the previous good body is
    left untouched (never replaced by a partial temp file).

    Replicas (``.bak``) and the tiny index pass ``file_sync=False`` /
    ``dir_sync=False``: they stay atomic via rename, and the caller flushes
    the directory once for the whole save — one fsync storm per save instead
    of one per file.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = None
    wrote = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        fh = os.fdopen(fd, "w", encoding="utf-8")
        fd = None  # ownership transferred to the file object
        with fh:
            fh.write(text)
            if file_sync:
                fh.flush()
                os.fsync(fh.fileno())
        wrote = True
    except OSError:
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    if not wrote:
        return False
    try:
        os.replace(tmp, path)
    except OSError:
        # the temp file holds the complete text but the rename is impossible
        # (odd filesystem) — a direct write is a safe last resort here.
        try:
            path.write_text(text, encoding="utf-8")
        except OSError:
            return False
    if dir_sync:
        _fsync_dir(directory)
    return True


def _read_optional(path: Path) -> str | None:
    """Read a file's text, or None if it's missing/empty/undecodable."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return text if text else None


def _read_session_data(path: Path) -> dict[str, Any] | None:
    """Parse a session body with crash recovery.

    Tries the primary file first, then a ``.json.gz`` squeeze of an old
    session (see squeeze_sessions), then the previous-good ``.json.bak``
    replica kept by save_session, then a leftover ``.json.tmp`` (a crash
    between writing the temp file and renaming it over the body). A 0-byte
    body — the signature of a non-durable write lost to a sudden reboot —
    transparently falls back to the last good copy instead of vanishing.
    """
    for candidate in (path, path.with_suffix(".json.gz"), path.with_suffix(".json.bak"), path.with_suffix(".json.tmp")):
        if str(candidate).endswith(".gz"):
            try:
                raw = candidate.read_bytes()
            except OSError:
                continue
            try:
                data = json.loads(gzip.decompress(raw).decode("utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                return data
            continue
        text = _read_optional(candidate)
        if text is None:
            continue
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _is_orphan_child(session: Session) -> bool:
    """True for a child transcript whose parent link was lost.

    Orphans arise when a placeholder (id only, made at subagent_start)
    reaches the writer before the real identity lands: exit save-all racing
    the first engine save, autosave ticking on the placeholder, or an old
    file from before identity-stamping existed. Without a parent_id such a
    file shows in /sessions like a real parent — the exact stray-row bug.
    Detection is content-based (NOT identity-based) so real sessions can
    never match: a sub-agent transcript always opens with the isolated task
    prompt as its first user message AND carries tool traffic, while a real
    TUI conversation never starts that way. Checks, cheapest first:
      1. already has parent_id → not an orphan;
      2. empty / untitled / provider-less → blank placeholder, orphan;
      3. first user message looks like a task prompt AND the transcript has
         tool messages → launched child, orphan.
    Never raises.
    """
    try:
        if getattr(session, "parent_id", None):
            return False
        messages = getattr(session, "messages", None) or []
        if not messages:
            # blank placeholder: no parent, no title, no provider/model
            if getattr(session, "title", ""):
                return False
            if getattr(session, "provider", "") or getattr(session, "model", ""):
                return False
            return True
        first_user = ""
        try:
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content", "")
                    first_user = c if isinstance(c, str) else str(c or "")
                    break
        except Exception:
            first_user = ""
        if not first_user or len(first_user) < 40:
            return False
        has_tool = False
        try:
            for m in messages:
                if isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_calls")):
                    has_tool = True
                    break
        except Exception:
            pass
        if not has_tool:
            return False
        # task prompts are isolated directives ("In <dir>, hunt...", "Do a
        # very small...", "Research ..."), never a TUI chat opener.
        head = first_user.strip()[:120].lower()
        task_markers = ("in /data", "in /", "do a very small", "do a ",
                        "research ", "hunt ", "find the ", "cover 3-4",
                        "with 1-2 sentence")
        return head.startswith(task_markers)
    except Exception:
        return False


def save_session(
    session: Session,
    *,
    should_write: Callable[[], bool] | None = None,
) -> Path:
    GPath.sessions_dir().mkdir(parents=True, exist_ok=True)
    path = session_path(session.id)
    if _is_orphan_child(session):
        # REFUSE to persist orphan child transcripts as top-level sessions.
        # This is the write-side seal: even if a parent link is lost in
        # memory, the file can never be minted, so /sessions can never show
        # a subagent as if it were a parent — now and in the future.
        try:
            clear_session_cache()
        except Exception:
            pass
        return path
    # Persist the message list EXACTLY as-is. Do NOT inject synthetic
    # "[interrupted — tool result missing]" tool messages here: the saved
    # transcript must be the real conversation. The request path already
    # repairs orphaned tool pairs for the provider (agent/loop.py), so a
    # resumed session that ends mid-tool-run stays provider-safe without
    # corrupting what the user sees on resume.
    try:
        fp = _fingerprint(session)
        if _SAVE_PRINTS.get(session.id) == fp:
            return path  # unchanged since last save: skip the 4MB dump
    except Exception:
        fp = None
    text = json.dumps(session.to_dict(), indent=2)
    # Run the write-set (body + .bak + index + cache invalidation) atomically
    # with respect to other writers. Callers that must never overwrite a newer
    # save (the streaming autosave, which can race the exit save-all) pass a
    # `should_write` guard evaluated inside the lock, immediately before the
    # body hits disk.
    with _WRITE_LOCK:
        if should_write is not None and not should_write():
            return path
        # Keep the previous good body as a `.bak` replica so a crash that corrupts
        # the primary write (0-byte body) can still recover the last complete copy.
        # Only do real IO when the serialized body actually changed — the in-turn
        # autosave otherwise re-serializes the same body every tick.
        current = _read_optional(path)
        if text != current:
            # The replica is best-effort (no fsync): it only matters when the
            # primary is corrupt, and the directory is flushed once below for
            # the whole save — one fsync storm per save instead of per file.
            _write_durable(path.with_suffix(".json.bak"), current or text,
                           file_sync=False, dir_sync=False)
            if not _write_durable(path, text, dir_sync=False):
                # atomic write failed — never lose the new body (best effort)
                try:
                    path.write_text(text, encoding="utf-8")
                except OSError:
                    pass
        try:
            if fp is not None:
                _SAVE_PRINTS[session.id] = fp
        except Exception:
            pass
        _index_insert(session.path, session, dir_sync=False)
        clear_session_cache()
        try:
            _fsync_dir(path.parent)
        except OSError:
            pass
        # Fix 4: opportunistic vacuum — at most once an hour, on a daemon
        # thread, only when a cap is configured. Both TUI and headless saves
        # flow through here, so one hook covers every path.
        try:
            _maybe_vacuum_after_save()
        except Exception:
            pass
    return path


_vacuum_last: float = 0.0
_vacuum_lock = None


def _maybe_vacuum_after_save() -> None:
    """Throttle + dispatch the hourly vacuum. Never raises."""
    global _vacuum_last, _vacuum_lock
    import time as _t
    try:
        if _t.monotonic() - _vacuum_last < 3600:
            return
        _vacuum_last = _t.monotonic()
    except Exception:
        return
    try:
        import threading as _th
        if _vacuum_lock is None:
            _vacuum_lock = _th.Lock()
        if not _vacuum_lock.acquire(blocking=False):
            return  # a vacuum is already running
    except Exception:
        return

    def _run() -> None:
        try:
            import json as _j
            # read the cap from config without importing the config module
            # (session.py must stay import-light for the TUI boot path)
            cap = 0
            try:
                from .globals import Path as _GP
                cfg_path = _GP.config / "opencode.json"
                data = _j.loads(cfg_path.read_text(encoding="utf-8"))
                cap = int(data.get("session_file_cap", 0) or 0)
            except Exception:
                cap = 0
            if cap > 0:
                vacuum_sessions(cap)
            # hourly squeeze runs alongside the vacuum (same throttle/lock):
            # old big bodies become `.json.gz`, same open path, ~0.1s to open.
            try:
                squeeze_sessions()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            try:
                _vacuum_lock.release()
            except Exception:
                pass

    try:
        import threading as _th2
        _th2.Thread(target=_run, name="opencode_py-vacuum", daemon=True).start()
    except Exception:
        try:
            _vacuum_lock.release()
        except Exception:
            pass


def _index_insert(path: Path, session: Session, *, dir_sync: bool = True) -> None:
    """Add or update one entry in the persistent index."""
    index = _read_index()
    files: dict[str, Any] = (index or {}).get("files") or {}
    files[path.name] = {
        "id": session.id,
        "title": session.title,
        "created": session.created,
        "completed": session.completed,
        "directory": session.directory,
        "provider": session.provider,
        "model": session.model,
        "agent": session.agent,
        "parent_id": session.parent_id,
        "has_messages": bool(session.messages),
        "first_user_text": _first_user_text(session.messages),
        "mtime_ns": _stat_mtime_ns(path),
        "size": _stat_size(path),
    }
    _write_index(files)


def load_session(session_id: str) -> Session | None:
    path = session_path(session_id)
    data = _read_session_data(path)
    if data is None:
        return None
    sess = Session(data)
    # A crash left the primary body corrupt (0-byte write) and we recovered
    # from the `.bak`/`.tmp` replica — immediately restore the primary so a
    # later crash on the session doesn't fall back to a staler copy.
    text = _read_optional(path)
    if text is None:
        try:
            _write_durable(path, json.dumps(sess.to_dict(), indent=2))
        except OSError:
            pass
    return sess


# Session listing cache: keyed by path -> (mtime_ns, size, header). The header
# is a LIGHTWEIGHT Session built from index metadata only (no transcript
# bodies), so hundreds of sessions no longer pin megabytes of messages in
# RAM. Full bodies still load on demand via load_session (resume/export/
# history_search) — the picker path never touches them.
_session_cache: dict[str, tuple[int, int, Session]] = {}
_session_cache_loaded = False
_SESSION_LIST_TTL = 2.0
_session_cache_time = 0.0
_session_cache_lock = threading.Lock()


def _index_path() -> Path:
    return GPath.sessions_dir() / ".sessions-index.json"


def _is_session_file(name: str) -> bool:
    """True for session body files (any `*.json` except the listing index and
    an in-progress tmp write)."""
    return (
        name.endswith(".json")
        and name != ".sessions-index.json"
        and not name.endswith(".json.tmp")
    )


def _read_index() -> dict[str, Any] | None:
    try:
        data = json.loads(_index_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != 1:
        return None
    return data


def _write_index(files: dict[str, Any], *, dir_sync: bool = True) -> None:
    data = {"version": 1, "files": files}
    _write_durable(_index_path(), json.dumps(data), dir_sync=dir_sync)


def _index_sessions(sessions: list[Session]) -> list[Session]:
    """Persist a metadata-only index of all sessions for fast future lists."""
    files: dict[str, Any] = {}
    for s in sessions:
        # headers carry no transcript: prefer their backed-up extras so the
        # index stays exact (has_messages/first_user_text) after incremental
        # enrich passes that only ever see headers
        if s.messages:
            has_m = True
            first_t = _first_user_text(s.messages)
        else:
            has_m = bool(getattr(s, "has_messages", False))
            first_t = getattr(s, "first_user_text", "") or ""
        files[s.path.name] = {
            "id": s.id,
            "title": s.title,
            "created": s.created,
            "completed": s.completed,
            "directory": s.directory,
            "provider": s.provider,
            "model": s.model,
            "agent": s.agent,
            "parent_id": s.parent_id,
            "has_messages": has_m,
            "first_user_text": first_t,
            "mtime_ns": _stat_mtime_ns(s.path),
            "size": _stat_size(s.path),
        }
    _write_index(files)
    GPath.sessions_dir().mkdir(parents=True, exist_ok=True)
    return sessions


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in messages or []:
        # corrupt bodies can carry non-dict items — skip them, never crash
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        text = content if isinstance(content, str) else str(content or "")
        text = text.strip()
        if text:
            return text[:60]
    return ""


def _stat_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _suggested(first_user: str) -> str:
    text = first_user.strip()
    return text[:60] if text else ""


def _session_from_index(name: str, meta: dict[str, Any]) -> Session | None:
    """Rebuild a listing Session from index metadata (no body read)."""
    if meta.get("id") is None:
        return None
    return Session(
        {
            "id": meta.get("id"),
            "title": meta.get("title", ""),
            "created": meta.get("created", 0.0),
            "completed": meta.get("completed"),
            "directory": meta.get("directory", ""),
            "provider": meta.get("provider", ""),
            "model": meta.get("model", ""),
            "agent": meta.get("agent", "build"),
            "parent_id": meta.get("parent_id"),
            "metadata": meta.get("metadata", {}),
            "has_messages": bool(meta.get("has_messages", True)),
            "first_user_text": meta.get("first_user_text", ""),
        }
    )


def _cache_fresh() -> bool:
    with _session_cache_lock:
        if not _session_cache_loaded or not _session_cache:
            return False
        if (time.monotonic() - _session_cache_time) < _SESSION_LIST_TTL:
            return True
    dir_path = GPath.sessions_dir()
    try:
        entries = [e for e in os.scandir(dir_path) if _is_session_file(e.name)]
    except OSError:
        return False
    with _session_cache_lock:
        if len(entries) != len(_session_cache):
            return False
        for e in entries:
            try:
                st = e.stat()
            except OSError:
                return False
            cached = _session_cache.get(e.name)
            if cached is None or (st.st_mtime_ns, st.st_size) != (cached[0], cached[1]):
                return False
    return True


def clear_session_cache() -> None:
    global _session_cache_loaded, _session_cache_time
    with _session_cache_lock:
        _session_cache.clear()
        _session_cache_loaded = False
        _session_cache_time = 0.0


def _header_of(s: Session) -> Session:
    # Lightweight listing copy: identity + display fields only, transcript
    # dropped. Cache/picker paths use headers; resume/export/search load the
    # full body from disk on demand.
    # Idempotent: headers carry no transcript, so re-heading a header must
    # reuse its backed-up extras — otherwise the cache poisons itself with
    # has_messages=False and the picker filters everything out next call.
    try:
        msgs = s.messages or []
        if msgs:
            has_m = True
            first_t = _first_user_text(msgs)
        else:
            has_m = bool(getattr(s, "has_messages", False))
            first_t = getattr(s, "first_user_text", "") or ""
        h = Session(
            {
                "id": s.id,
                "title": s.title,
                "created": s.created,
                "completed": s.completed,
                "directory": s.directory,
                "provider": s.provider,
                "model": s.model,
                "agent": s.agent,
                "parent_id": s.parent_id,
                "metadata": {},
                "has_messages": has_m,
                "first_user_text": first_t,
            }
        )
        h.messages = []
        return h
    except Exception:
        return s


def _cache_sessions(sessions: list[Session]) -> list[Session]:
    global _session_cache_loaded, _session_cache_time
    with _session_cache_lock:
        _session_cache.clear()
        for s in sessions:
            try:
                st = s.path.stat()
            except OSError:
                continue
            _session_cache[s.path.name] = (st.st_mtime_ns, st.st_size, _header_of(s))
        _session_cache_loaded = True
        _session_cache_time = time.monotonic()
    return sessions


def list_sessions(directory: str | None = None) -> list[Session]:
    """List saved sessions, newest first.

    ``directory`` scopes the result to one project (normalized path
    comparison) — upstream opencode's picker only shows the current
    project's sessions. ``None`` (the default) keeps the old behaviour of
    returning every session on the device.
    """
    sessions = _list_sessions_all()
    if directory:
        want = os.path.normpath(directory)
        sessions = [
            s
            for s in sessions
            if s.directory and os.path.normpath(s.directory) == want
        ]
    return sessions


def _list_sessions_all() -> list[Session]:
    if _cache_fresh():
        return sorted((e[2] for e in _session_cache.values()), key=lambda s: s.created, reverse=True)
    index = _read_index()
    if index is not None:
        sessions = _enrich_index(index)
        if sessions:
            sessions.sort(key=lambda s: s.created, reverse=True)
            return _cache_sessions(sessions)

    # Full scan fallback (also rebuilds the index for next time). Raw bodies
    # may be corrupt (crash mid-write, partial rename, hand-edit) — recover
    # from the `.bak`/`.tmp` replica when possible, and otherwise skip anything
    # unparseable instead of letting one bad file break the picker.
    sessions: list[Session] = []
    dir_path = GPath.sessions_dir()
    dir_path.mkdir(parents=True, exist_ok=True)
    for path in dir_path.glob("*.json"):
        if not _is_session_file(path.name):
            continue
        # Squeezed bodies live as `.json.gz` (the `.json` is gone): resolve
        # the body through _read_session_data on the CANONICAL name so the
        # same file lists + opens whether squeezed or not.
        if not path.exists():
            gz = path.with_suffix(".json.gz")
            if gz.exists():
                path = gz
            else:
                continue
        data = _read_session_data(path)
        if data is None:
            continue
        try:
            sessions.append(Session(data))
        except (TypeError, ValueError, AttributeError):
            continue
    sessions.sort(key=lambda s: s.created, reverse=True)
    _index_sessions(sessions)
    return _cache_sessions(sessions)


def _enrich_index(index: dict[str, Any]) -> list[Session]:
    """Build listing Sessions from the index, refreshing only what changed.

    Per-file incremental: unchanged entries reuse their cached header, added
    or modified files are read individually, deleted files are dropped. One
    changed body no longer invalidates the whole index into a full rescan.
    Headers (no transcript bodies) are returned, so the picker stays light.
    """
    files: dict[str, Any] = index.get("files") or {}
    dir_path = GPath.sessions_dir()
    try:
        entries = {
            e.name: e
            for e in os.scandir(dir_path)
            if _is_session_file(e.name) or e.name.endswith(".json.gz")
        }
    except OSError:
        return []
    # Squeezed sessions: the `.json` is gone, only `.json.gz` remains. Map
    # them back to their canonical `.json` name so the picker shows the
    # same row and load_session rehydrates transparently.
    for gz_name in [n for n in list(entries) if n.endswith(".json.gz")]:
        canon = gz_name[: -len(".gz")]
        if canon not in entries:
            entries[canon] = entries[gz_name]
    # drop cached headers for files that vanished or changed on disk
    with _session_cache_lock:
        for name in list(_session_cache):
            e = entries.get(name)
            if e is None:
                _session_cache.pop(name, None)
                continue
            try:
                st = e.stat()
            except OSError:
                _session_cache.pop(name, None)
                continue
            cached = _session_cache.get(name)
            if cached is None or (st.st_mtime_ns, st.st_size) != (cached[0], cached[1]):
                _session_cache.pop(name, None)
    # reuse surviving headers from the cache where possible
    with _session_cache_lock:
        reuse = dict(_session_cache)
    out: list[Session] = []
    changed_files: dict[str, Any] = {}
    for name, meta in files.items():
        e = entries.get(name)
        if e is None:
            continue  # deleted after indexing — drop
        try:
            st = e.stat()
        except OSError:
            continue
        if st.st_mtime_ns != meta.get("mtime_ns") or st.st_size != meta.get("size"):
            changed_files[name] = meta
            continue  # modified after indexing — re-read below
        hit = reuse.get(name)
        if hit is not None and (st.st_mtime_ns, st.st_size) == (hit[0], hit[1]):
            out.append(hit[2])
            continue
        sess = _session_from_index(name, meta)
        if sess is None:
            changed_files[name] = meta
            continue
        out.append(sess)
    # new files the index never saw
    for name in entries:
        if name not in files:
            changed_files[name] = {}
    # read only the new/changed bodies, then merge into headers
    if changed_files:
        for s in _read_session_headers(sorted(changed_files)):
            out.append(s)
    # rebuild the on-disk index when something changed (_index_sessions
    # reads header-carried has_messages/first_user_text, so headers index
    # exactly like full bodies)
    if changed_files:
        try:
            _index_sessions(out)
        except Exception:
            pass
    return out


def _read_sessions_by_name(names: list[str]) -> list[Session]:
    """Read full session bodies for `names`, skipping the unparseable."""
    dir_path = GPath.sessions_dir()
    out: list[Session] = []
    for name in names:
        path = dir_path / name
        # Squeezed bodies: only `.json.gz` exists — read through it.
        if not path.exists():
            gz = path.with_suffix(".json.gz")
            if gz.exists():
                path = gz
            else:
                continue
        data = _read_session_data(path)
        if data is None:
            continue
        try:
            out.append(Session(data))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _read_session_headers(names: list[str]) -> list[Session]:
    """Read session bodies for `names`, returning lightweight headers."""
    return [_header_of(s) for s in _read_sessions_by_name(names)]


DEFAULT_SESSION_FILE_CAP = 150
SESSION_WARN_BYTES = 4 * 1024 * 1024
# Squeeze rules: bodies older than this with at least this many bytes become
# `.json.gz` (same open path, ~0.1s to rehydrate). Current + pinned + recent
# sessions are never squeezed. ponytail: fixed 500KB/7d — upgrade path is
# wiring these two into opencode.json when someone asks.
SQUEEZE_MIN_BYTES = 500 * 1024
SQUEEZE_MIN_AGE_S = 7 * 24 * 3600


def session_disk_stats() -> dict:
    """(files, bytes, bak_bytes) for the sessions dir. Never raises."""
    try:
        d = GPath.sessions_dir()
        files = [p for p in d.glob("*.json") if _is_session_file(p.name)]
        total = 0
        for p in files:
            try:
                total += p.stat().st_size
            except OSError:
                continue
        baks = [p for p in d.glob("*.json.bak")]
        btotal = 0
        for p in baks:
            try:
                btotal += p.stat().st_size
            except OSError:
                continue
        gz = [p for p in d.glob("*.json.gz")]
        gtotal = 0
        for p in gz:
            try:
                gtotal += p.stat().st_size
            except OSError:
                continue
        return {"files": len(files), "bytes": total, "bak_files": len(baks), "bak_bytes": btotal, "gz_files": len(gz), "gz_bytes": gtotal}
    except Exception:
        return {"files": 0, "bytes": 0, "bak_files": 0, "bak_bytes": 0, "gz_files": 0, "gz_bytes": 0}


def _is_pinned(session_id: str) -> bool:
    """True when the session is pinned (never vacuumed/squeezed)."""
    try:
        d = GPath.sessions_dir() / ".pinned"
        if not d.exists():
            return False
        return (d / str(session_id)).exists()
    except Exception:
        return False


def set_pinned(session_id: str, pinned: bool = True) -> bool:
    """Pin/unpin a session (pinned sessions survive vacuum + squeeze)."""
    try:
        d = GPath.sessions_dir()
        d.mkdir(parents=True, exist_ok=True)
        marker = d / ".pinned" / str(session_id)
        if pinned:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("1", encoding="utf-8")
        else:
            try:
                marker.unlink()
            except OSError:
                pass
        return True
    except Exception:
        return False


def squeeze_sessions(min_bytes: int = SQUEEZE_MIN_BYTES, min_age_s: float = SQUEEZE_MIN_AGE_S, live_ids: set[str] | None = None) -> dict:
    """gzip old big session bodies to `.json.gz` (same open path).

    Only bodies >= min_bytes and older than min_age_s. Never touches: live
    sessions (live_ids), pinned sessions, sub-agent children, or sessions
    already squeezed. The original is replaced atomically only after the
    .gz verifies; the picker lists squeezed sessions normally and
    load_session rehydrates transparently (~0.1s for 4MB).
    Returns {squeezed, bytes_saved}. Never raises.
    """
    report: dict = {"squeezed": 0, "bytes_saved": 0, "errors": 0}
    try:
        now = time.time()
        live = set(live_ids or ())
        d = GPath.sessions_dir()
        try:
            kids = {s.id for s in _list_sessions_all() if getattr(s, "parent_id", None)}
        except Exception:
            kids = set()
        for s in _list_sessions_all():
            try:
                sid = getattr(s, "id", "")
                if not sid or sid in live or sid in kids:
                    continue
                if _is_pinned(sid):
                    continue
                created = float(getattr(s, "created", 0) or 0)
                if created <= 0 or now - created < min_age_s:
                    continue
                path = session_path(sid)
                gz = path.with_suffix(".json.gz")
                if gz.exists():
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size < min_bytes:
                    continue
                try:
                    raw = path.read_bytes()
                except OSError:
                    continue
                try:
                    blob = gzip.compress(raw, compresslevel=6)
                except Exception:
                    report["errors"] += 1
                    continue
                try:
                    tmp = gz.with_suffix(".gz.tmp")
                    tmp.write_bytes(blob)
                    os.replace(tmp, gz)
                except OSError:
                    report["errors"] += 1
                    continue
                try:
                    back = gzip.decompress(gz.read_bytes())
                    if back != raw:
                        raise ValueError("roundtrip mismatch")
                except Exception:
                    try:
                        gz.unlink()
                    except OSError:
                        pass
                    report["errors"] += 1
                    continue
                try:
                    path.unlink()
                except OSError:
                    pass
                try:
                    _fsync_dir(d)
                except OSError:
                    pass
                try:
                    clear_session_cache()
                except Exception:
                    pass
                report["squeezed"] += 1
                report["bytes_saved"] += max(0, size - len(blob))
            except Exception:
                report["errors"] += 1
                continue
    except Exception:
        pass
    return report


def cleanup_report() -> str:
    """One-line summary of session disk state (for startup notice)."""
    try:
        stats = session_disk_stats()
        files = int(stats.get("files", 0) or 0)
        total = int(stats.get("bytes", 0) or 0)
        mb = total / (1024 * 1024)
        gz = int(stats.get("gz_files", 0) or 0)
        extra = f" + {gz} squeezed" if gz else ""
        return f"{files} saved chats{extra}, {mb:.1f}MB on disk"
    except Exception:
        return ""


def vacuum_sessions(max_files: int = DEFAULT_SESSION_FILE_CAP) -> dict:
    """Prune oldest INACTIVE sessions past `max_files` (0/negative = dry-run
    report only). Never touches: the live/current session, running sessions,
    sub-agent children (deleted with their parent), or sessions with
    messages from the last 7 days. Deletes via delete_session() so the
    index + .bak stay consistent. Returns {kept, pruned, bytes_freed}.
    Never raises."""
    import time as _t
    report: dict = {"kept": 0, "pruned": 0, "bytes_freed": 0, "errors": 0}
    try:
        cap = int(max_files or 0)
    except (TypeError, ValueError):
        return report
    try:
        all_sessions = _list_sessions_all()
    except Exception:
        return report
    # protect: children (by parent), recent activity, and anything running
    try:
        children = {getattr(s, "id", "") for s in all_sessions if getattr(s, "parent_id", None)}
    except Exception:
        children = set()
    week_ago = _t.time() - 7 * 24 * 3600
    cands: list = []
    kept_ids: set[str] = set()
    for s in all_sessions:
        try:
            sid = getattr(s, "id", "")
            if not sid:
                continue
            if getattr(s, "parent_id", None) or sid in children:
                kept_ids.add(sid)
                continue
            if _is_pinned(sid):
                kept_ids.add(sid)
                continue
            created = float(getattr(s, "created", 0) or 0)
            msgs = getattr(s, "messages", []) or []
            # empty junk (no messages, no title): deletable even when young
            if not msgs and not getattr(s, "title", ""):
                try:
                    sz = s.path.stat().st_size
                except (OSError, AttributeError):
                    sz = 0
                cands.append((created, sid, sz))
                continue
            if created >= week_ago or (msgs and len(msgs) > 0 and created == 0):
                kept_ids.add(sid)
                continue
            try:
                sz = s.path.stat().st_size
            except (OSError, AttributeError):
                sz = 0
            cands.append((created, sid, sz))
        except Exception:
            continue
    report["kept"] = len(all_sessions)
    if cap <= 0:
        report["pruned"] = 0
        return report
    # oldest first, prune past cap (counting kept + candidates)
    cands.sort()
    total_files = len(all_sessions)
    # empty junk always dies first (even under the cap), oldest first.
    # ponytail: O(n^2) id scan, fine for hundreds of session files.
    _junk_ids: set[str] = set()
    for _c, _sid, _sz in cands:
        try:
            _s = next(s for s in all_sessions if getattr(s, "id", "") == _sid)
            if not (getattr(_s, "messages", None) or []) and not getattr(_s, "title", ""):
                _junk_ids.add(_sid)
        except StopIteration:
            pass
    junk = [c for c in cands if c[1] in _junk_ids]
    junk.sort()
    over = total_files - cap
    if over <= 0 and not junk:
        return report
    # junk ALWAYS goes; aged fills the remainder of the over-count.
    victims = list(junk) + [c for c in cands if c[1] not in _junk_ids][: max(0, over)]
    freed = 0
    pruned = 0
    for _created, sid, sz in victims:
        try:
            if delete_session(sid):
                pruned += 1
                freed += sz
        except Exception:
            report["errors"] += 1
            continue
    report["pruned"] = pruned
    report["bytes_freed"] = freed
    report["kept"] = total_files - pruned
    return report


def delete_session(session_id: str, _seen: set[str] | None = None) -> bool:
    with _WRITE_LOCK:
        path = session_path(session_id)
        existed = path.exists() or path.with_suffix(".json.gz").exists()
        seen = _seen if _seen is not None else {session_id}
        def _collect_children(pid: str, acc: list[str]) -> None:
            try:
                kids = [s.id for s in _list_sessions_all() if s.parent_id == pid]
            except Exception:
                return
            for cid in kids:
                if cid in seen:
                    continue
                seen.add(cid)
                acc.append(cid)
                _collect_children(cid, acc)
        doomed: list[str] = []
        _collect_children(session_id, doomed)
        for cid in doomed:
            _delete_one_locked(cid, seen)
        _delete_one_locked(session_id, seen, _already_collected=False)
        return existed


def _delete_one_locked(session_id: str, seen: set[str], _already_collected: bool = True) -> None:
    path = session_path(session_id)
    for p in (path, path.with_suffix(".json.gz"), path.with_suffix(".json.bak"), path.with_suffix(".json.tmp")):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    # Single-store spill dies with its session (no orphan bodies on disk).
    try:
        from .agent.trim import clear_spill as _clear_spill

        _clear_spill(session_id)
    except Exception:
        pass
    try:
        _fsync_dir(path.parent)
    except OSError:
        pass
    index = _read_index()
    if index is not None:
        files = index.get("files") or {}
        if path.name in files:
            del files[path.name]
            _write_index(files)
    clear_session_cache()


def group_sessions(sessions: list[Any]) -> list[tuple[str, list[Any]]]:
    """Group sessions into opencode-style sections: ``Today``, ``Yesterday``,
    then one label per older day (``Monday, August 17``), newest-first.

    Accepts ``Session`` objects or dicts (must carry ``created``). Returns an
    ordered list of ``(label, [sessions])``; each group's sessions are
    newest-first.
    """
    import datetime

    def _created(obj: Any) -> float:
        if isinstance(obj, dict):
            return _as_number(obj.get("created"))
        return _as_number(getattr(obj, "created", None))

    now = datetime.datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - datetime.timedelta(days=1)

    buckets: dict[str, list[Any]] = {}
    order: list[str] = []
    for s in sorted(sessions, key=_created, reverse=True):
        created = _created(s)
        if created >= today_start.timestamp():
            label = "Today"
        elif created >= yesterday_start.timestamp():
            label = "Yesterday"
        else:
            # Include the YEAR: the same month/day recurs every year, and the
            # weekday repeat cycle (5/6/11 years, e.g. Aug 17 2020 = Aug 17
            # 2026 = a Monday) lets sessions years apart collide onto one label
            # and silently merge into the SAME group. Day + year is unique.
            try:
                label = datetime.datetime.fromtimestamp(created).strftime(
                    "%A, %B %d, %Y"
                )
            except (OSError, OverflowError, ValueError):
                # finite but out of platform range (year 300000, deep negative)
                label = "Unknown date"
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        buckets[label].append(s)
    return [(label, buckets[label]) for label in order]


def suggested_title(session: Any) -> str:
    """Title for display when a session has none saved: derive it from the
    first user message (opencode's behaviour). Accepts a Session or dict."""
    if isinstance(session, dict):
        title = session.get("title", "")
        first = session.get("first_user_text", "") if not session.get("messages") else ""
        messages = session.get("messages") or []
    else:
        title = session.title or ""
        first = getattr(session, "first_user_text", "") if not getattr(session, "messages", None) else ""
        messages = session.messages or []
    if title:
        return title
    if first:
        return first.strip()[:60]
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        text = content if isinstance(content, str) else str(content or "")
        if text.strip():
            return text.strip()[:60]
    return ""


def session_to_markdown(session: Session) -> str:
    """Render a session transcript to Markdown, tool calls included, for
    sharing/review files (the ``/export`` command writes this)."""
    import datetime

    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(p.get("text", ""))
                if isinstance(p, dict)
                else (
                    f"{p.get('name', '')}({p.get('arguments', '')})"
                    if isinstance(p, dict) and "name" in p
                    else str(p)
                )
                for p in content
            )
        return str(content)

    title = session.title or "Session"
    lines: list[str] = [f"# {title}", ""]
    lines += [f"- **ID**: `{session.id}`"]
    if session.directory:
        lines.append(f"- **Directory**: `{session.directory}`")
    if session.provider:
        lines.append(f"- **Provider**: {session.provider}")
    if session.model:
        lines.append(f"- **Model**: {session.model}")
    if session.agent:
        lines.append(f"- **Agent**: {session.agent}")
    if session.created:
        try:
            lines.append(f"- **Created**: {datetime.datetime.fromtimestamp(session.created).isoformat()}")
        except (OSError, ValueError, OverflowError):
            pass
    lines += ["", "---", ""]

    for msg in session.messages:
        role = msg.get("role")
        content = _text(msg.get("content", ""))
        if role == "system":
            continue
        if role == "compaction" or msg.get("compaction"):
            lines += ["## Compaction", "", content, ""]
            continue
        if role == "user":
            lines += ["## User", "", content, ""]
            continue
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            lines += ["## Assistant", ""]
            if content:
                lines.append(content)
            for call in tool_calls:
                fn = call.get("function") or {}
                try:
                    args = json.dumps(json.loads(fn.get("arguments", "{}")), indent=2)
                except json.JSONDecodeError:
                    args = str(fn.get("arguments", "{}"))
                lines += ["", f"- 🛠️ **{fn.get('name', '?')}**", "```json", args, "```"]
            reasoning = msg.get("reasoning_content")
            if reasoning:
                lines += ["", "<details><summary>Thought</summary>", "", str(reasoning), "", "</details>"]
            lines.append("")
            continue
        if role == "tool":
            lines += [
                "",
                f"**Tool result** (`{msg.get('name', '?')}`, id `{msg.get('tool_call_id', '')}`):",
                "",
                "```",
                str(content or "(no output)"),
                "```",
                "",
            ]
            continue
        lines += ["", f"**{role}**:", "", str(content), ""]

    return "\n".join(lines).rstrip() + "\n"


def new_session(
    *,
    directory: str | None = None,
    provider: str = "",
    model: str = "",
    agent: str = "build",
    title: str = "",
    parent_id: str | None = None,
) -> Session:
    return Session(
        {
            "id": uuid.uuid4().hex,
            "title": title,
            "created": time.time(),
            "directory": directory,
            "provider": provider,
            "model": model,
            "agent": agent,
            "parent_id": parent_id,
            "messages": [],
        }
    )
