"""checkpoint tool: repo-wide snapshots before risky work.

RULES (full guidance for maintainers; the sent schema is short):
- take FIRST on risky/wide changes; rollback restores exactly (deletes new
  files, restores deleted ones); a pre-rollback copy makes mistakes reversible.
- diff shows what changed; list/drop manage. Skips binaries/>512KB.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any

from ..globals import Path as GPath
from ..globals import resolve_worktree
from ..tools.read import _is_binary_sample
from .registry import Tool, schema_with
from .write import _atomic_write

_LOCK = threading.Lock()

VERSION = 1
MAX_CHECKPOINTS = 15      # retained payloads (oldest pruned)
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 12 * 1024 * 1024
MAX_FILES = 3000

# same ignore rules as the symbol engine — loaded lazily (see _ignores()):
# importing index.engine here cost ~80ms on every registry build.
_IGNORE_NAMES: frozenset | None = None


def _ignores() -> frozenset:
    """Symbol-engine ignore set, loaded on first checkpoint scan."""
    global _IGNORE_NAMES
    if _IGNORE_NAMES is None:
        try:
            from ..index.engine import DEFAULT_IGNORE as _names
        except Exception:
            _names = frozenset({".git", "node_modules", "__pycache__"})
        _IGNORE_NAMES = frozenset(_names)
    return _IGNORE_NAMES


def _store_dir() -> Path:
    return GPath.data / "checkpoints"


def _manifest_path() -> Path:
    return _store_dir() / "manifest.json"


def _load_manifest() -> list[dict[str, Any]]:
    try:
        data = json.loads(_manifest_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_manifest(entries: list[dict[str, Any]]) -> None:
    _store_dir().mkdir(parents=True, exist_ok=True)
    from ..session import _write_durable
    _write_durable(_manifest_path(), json.dumps(entries), file_sync=True, dir_sync=True)


def _payload_path(cid: str) -> Path:
    return _store_dir() / f"{cid}.json.z"


def _write_payload(cid: str, payload: dict[str, Any]) -> bool:
    _store_dir().mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload).encode("utf-8")
    blob = zlib.compress(raw, 6)
    path = _payload_path(cid)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(blob)
        try:
            with open(tmp, "rb") as fh:
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
        except OSError:
            pass
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            except OSError:
                pass
            finally:
                try:
                    os.close(dfd)
                except OSError:
                    pass
        except OSError:
            pass
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _read_payload(cid: str) -> dict[str, Any] | None:
    try:
        blob = _payload_path(cid).read_bytes()
        data = json.loads(zlib.decompress(blob).decode("utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, zlib.error):
        return None


def _delete_payload(cid: str) -> None:
    try:
        _payload_path(cid).unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# tree walking shared by take/diff
# ---------------------------------------------------------------------------

def _scan_tree(root: Path) -> dict[str, bytes]:
    """Relative-path -> raw bytes for every snapshot-able file under root."""
    files: dict[str, bytes] = {}
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _ignores() and not d.startswith(".")
        ]
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, root)
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if st.st_size > MAX_FILE_BYTES:
                continue
            try:
                with open(path, "rb") as fh:
                    data = fh.read(MAX_FILE_BYTES + 1024)
                    if _is_binary_sample(data[:1024]):
                        continue
                    if len(data) > MAX_FILE_BYTES:
                        continue
            except OSError:
                continue
            total += len(data)
            if total > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                return files, True
            files[rel.replace(os.sep, "/")] = data
    return files, False


def _new_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _action_take(label: str) -> dict:
    root = resolve_worktree(Path.cwd())
    files, truncated = _scan_tree(root)
    if not files:
        return {"output": f"No snapshot-able files found under {root}.", "error": True}
    cid = _new_id()
    payload = {
        "version": VERSION,
        "id": cid,
        "label": label or "",
        "root": str(root),
        "created": time.time(),
        "files": {
            rel: {
                "b64": base64.b64encode(data).decode("ascii"),
                "existed": True,
            }
            for rel, data in files.items()
        },
    }
    with _LOCK:
        if not _write_payload(cid, payload):
            return {"output": "Failed writing checkpoint payload.", "error": True}
        entries = _load_manifest()
        entries.append({
            "id": cid,
            "label": label or "",
            "created": payload["created"],
            "root": str(root),
            "files": len(files),
            "bytes": sum(len(d) for d in files.values()),
            "truncated": truncated,
        })
        while len(entries) > MAX_CHECKPOINTS:
            old = entries.pop(0)
            _delete_payload(old["id"])
        _save_manifest(entries)
    size_kb = sum(len(d) for d in files.values()) // 1024
    note = f" WARNING: caps hit — snapshot is PARTIAL ({len(files)} files)." if truncated else ""
    return {
        "output": (
            f"Checkpoint {cid} saved: {len(files)} files "
            f"(~{size_kb} KB) from {root}.{note}\n"
            "Restore with action=rollback."
        ),
        "metadata": {"id": cid, "files": len(files), "truncated": truncated},
    }


def _action_rollback(cid: str, force: bool) -> dict:
    with _LOCK:
        entries = _load_manifest()
        target = None
        if cid.strip():
            for e in entries:
                if e["id"] == cid.strip() or e["id"].startswith(cid.strip()):
                    target = e
                    break
        elif entries:
            target = entries[-1]
        if target is None:
            known = ", ".join(e["id"][:14] for e in entries[-5:]) or "(none)"
            return {"output": f"No checkpoint matches {cid!r}. Known: {known}",
                    "error": True}
        payload = _read_payload(target["id"])
        if payload is None:
            return {"output": f"Checkpoint {target['id']} payload is unreadable.",
                    "error": True}

        current_root = str(resolve_worktree(Path.cwd()))
        if payload.get("root") != current_root and not force:
            return {
                "output": (
                    f"Refusing: checkpoint was taken in {payload['root']} but you "
                    f"are in {current_root}. Pass force=true to roll back anyway."
                ),
                "error": True,
            }

        if target.get("truncated"):
            return {"output": "Refusing rollback: checkpoint is PARTIAL (caps hit at take time). "
                    "Restoring it would delete uncaptured files with no recovery. "
                    "Pick a non-partial checkpoint.", "error": True}
        root_path = Path(payload["root"])
        safe_ok = True
        for rel in payload["files"]:
            p = Path(rel)
            if p.is_absolute() or ".." in p.parts:
                safe_ok = False
            try:
                (root_path / p).resolve().relative_to(root_path.resolve())
            except ValueError:
                safe_ok = False
        if not safe_ok:
            return {"output": "Refusing rollback: payload contains unsafe paths (traversal).", "error": True}

        # safety net FIRST: preserve today's state so this rollback is undoable
        safety_files, trunc = _scan_tree(Path(payload["root"]))
        safety_id = _new_id()
        safety_payload = {
            "version": VERSION,
            "id": safety_id,
            "label": f"pre-rollback (auto, before {target['id']})",
            "root": payload["root"],
            "created": time.time(),
            "files": {
                rel: {"b64": base64.b64encode(d).decode("ascii"), "existed": True}
                for rel, d in safety_files.items()
            },
        }
        if not _write_payload(safety_id, safety_payload) or _read_payload(safety_id) is None:
            return {"output": "Refusing rollback: could not save safety snapshot, nothing changed.",
                    "error": True}
        entries.append({
            "id": safety_id,
            "label": f"pre-rollback (auto, before {target['id']})",
            "created": time.time(),
            "root": payload["root"],
            "files": len(safety_files),
            "bytes": sum(len(d) for d in safety_files.values()),
            "truncated": trunc,
        })
        # prune to capacity, never touching the checkpoint being restored
        # or its fresh safety snapshot
        protected = {target["id"], safety_id}
        excess = len(entries) - MAX_CHECKPOINTS
        if excess > 0:
            removable = [e for e in entries if e["id"] not in protected]
            for old in removable[:excess]:
                entries.remove(old)
                _delete_payload(old["id"])
        _save_manifest(entries)

        restored, removed, errors = 0, 0, []
        for rel, rec in payload["files"].items():
            path = (root_path / rel).resolve()
            try:
                path.relative_to(root_path.resolve())
            except ValueError:
                errors.append(f"{rel}: blocked traversal")
                continue
            try:
                if rec.get("existed"):
                    data = base64.b64decode(rec["b64"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(path.suffix + ".ckpt-tmp")
                    tmp.write_bytes(data)
                    try:
                        with open(tmp, "rb") as fh:
                            try: os.fsync(fh.fileno())
                            except OSError: pass
                    except OSError:
                        pass
                    os.replace(tmp, path)
                    restored += 1
            except (OSError, ValueError) as e:
                errors.append(f"{rel}: {e}")
        if errors:
            errors.append("Aborted before deleting newer files: restores had errors, nothing deleted.")
        else:
            current_files, _ = _scan_tree(root_path)
            for rel in current_files:
                if rel not in payload["files"]:
                    cpath = (root_path / rel).resolve()
                    try:
                        cpath.relative_to(root_path.resolve())
                    except ValueError:
                        continue
                    try:
                        (root_path / rel).unlink()
                        removed += 1
                    except OSError as e:
                        errors.append(f"{rel}: {e}")
        try:
            dfd = os.open(root_path, os.O_RDONLY)
            try: os.fsync(dfd)
            except OSError: pass
            finally:
                try: os.close(dfd)
                except OSError: pass
        except OSError:
            pass

    lines = [
        f"Rolled back to {target['id']}"
        + (f" ({target['label']})" if target.get("label") else "")
        + f": restored {restored} file(s), deleted {removed} newer file(s)."
    ]
    if errors:
        lines.append("ERRORS:")
        lines.extend(f"  {e}" for e in errors[:10])
    lines.append(f"Safety snapshot of pre-rollback state: {safety_id} "
                 "(undo this rollback by rolling back to it).")
    if trunc:
        lines.append("Warning: safety snapshot is PARTIAL (caps hit) — undo may refuse.")
    return {"output": "\n".join(lines), "error": bool(errors),
            "metadata": {"restored": restored, "removed": removed,
                         "safety_id": safety_id}}


def _action_diff(cid: str) -> dict:
    entries = _load_manifest()
    target = None
    if cid.strip():
        for e in entries:
            if e["id"] == cid.strip() or e["id"].startswith(cid.strip()):
                target = e
                break
    elif entries:
        target = entries[-1]
    if target is None:
        return {"output": f"No checkpoint matches {cid!r}.", "error": True}
    payload = _read_payload(target["id"])
    if payload is None:
        return {"output": "Payload unreadable.", "error": True}
    root = Path(payload["root"])
    if not root.exists():
        return {"output": f"Checkpoint root no longer exists: {root}", "error": True}
    current, _ = _scan_tree(root)
    stored = payload["files"]

    modified = [rel for rel in current if rel in stored
                and current[rel] != base64.b64decode(stored[rel]["b64"])]
    added = sorted(rel for rel in current if rel not in stored)
    deleted = sorted(rel for rel in stored if rel not in current)

    def _kb(n: int) -> str:
        return f"{n // 1024} KB" if n >= 1024 else f"{n} B"

    lines = [f"Changes since {target['id']}"
             + (f" ({target['label']})" if target.get('label') else "") + ":"]
    sections = (
        (f"modified ({len(modified)})", modified),
        (f"added ({len(added)})", added),
        (f"deleted ({len(deleted)})", deleted),
    )
    empty = True
    for title, items in sections:
        if not items:
            continue
        empty = False
        lines.append(f"• {title}")
        for rel in items[:80]:
            extra = ""
            if title.startswith("modified"):
                extra = f"  ({_kb(len(current[rel]))} now)"
            lines.append(f"  {rel}{extra}")
        if len(items) > 80:
            lines.append(f"  … and {len(items) - 80} more")
    if empty:
        lines.append("• working tree matches the checkpoint exactly.")
    return {"output": "\n".join(lines), "metadata": {
        "modified": len(modified), "added": len(added), "deleted": len(deleted)}}


def _action_list() -> dict:
    entries = _load_manifest()
    if not entries:
        return {"output": "No checkpoints yet. Take one with action=take."}
    lines = [f"{len(entries)} checkpoint(s), oldest first:"]
    for e in entries:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("created", 0)))
        label = f" — {e['label']}" if e.get("label") else ""
        part = " PARTIAL" if e.get("truncated") else ""
        lines.append(
            f"  {e['id']}  {when}  {e['files']:>4} files{part}{label}"
        )
    lines.append("Rollback defaults to the LAST one; diff shows changes since it.")
    return {"output": "\n".join(lines), "metadata": {"count": len(entries)}}


def _action_drop(cid: str) -> dict:
    with _LOCK:
        entries = _load_manifest()
        target = None
        for e in entries:
            if e["id"] == cid.strip() or e["id"].startswith(cid.strip()):
                target = e
                break
        if target is None:
            return {"output": f"No checkpoint matches {cid!r}.", "error": True}
        entries.remove(target)
        _delete_payload(target["id"])
        _save_manifest(entries)
    return {"output": f"Dropped checkpoint {target['id']}."}


_ACTIONS = ("take", "rollback", "diff", "list", "drop")


def tool() -> Tool:
    description = """Snapshot the repo before risky work; rollback restores exactly. take/rollback/diff/list/drop."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "take").strip().lower()
        if action not in _ACTIONS:
            return {
                "output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True,
            }
        cid = str(input.get("id") or "")
        if action == "take":
            return _action_take(str(input.get("label") or ""))
        if action == "rollback":
            return _action_rollback(cid, force=bool(input.get("force", False)))
        if action == "diff":
            return _action_diff(cid)
        if action == "drop":
            return _action_drop(cid)
        return _action_list()

    return Tool(
        name="checkpoint",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "take (default), rollback, diff, list, drop",
                    "optional": True,
                },
                "label": {
                    "type": "string",
                    "description": "Label, e.g. before refactor",
                    "optional": True,
                },
                "id": {
                    "type": "string",
                    "description": "Checkpoint id (default latest)",
                    "optional": True,
                },
                "force": {
                    "type": "boolean",
                    "description": "Cross-project rollback",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="checkpoint",
    )