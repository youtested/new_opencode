"""apply_patch tool: safe multi-file changes from one unified diff.

RULES (full guidance for maintainers; the sent schema is short):
- One diff may touch MANY files; every hunk verified BEFORE anything is
  written — default all-or-nothing aborts everything with all problems
  reported. partial=true applies clean files and reports only bad hunks
  (best-way: 1 bad hunk costs 1 hunk resend, not N files).
- Position-tolerant (exact context searched nearby), never fuzzy content.
  Miss reports the 3 closest real lines + read offset= anchor.
- `--- /dev/null` creates, `+++ /dev/null` deletes. dry_run previews.
- undo reverts the whole patch; history lists past patches.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..globals import Path as GPath
from .registry import Tool, schema_with
from .write import _atomic_write

_LOCK = threading.Lock()

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

JOURNAL_PATH_NAME = "patch_journal.json"
MAX_JOURNAL_ENTRIES = 20
MAX_SHIFT = 200  # how far a hunk may drift from its declared position


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _clean_path(p: str) -> str:
    p = p.strip()
    if p.startswith('"') and p.endswith('"') and len(p) > 1:
        p = p[1:-1]
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    return p


class _FilePatch:
    __slots__ = ("old_path", "new_path", "hunks")

    def __init__(self, old_path: str, new_path: str) -> None:
        self.old_path = old_path  # cleaned, or "/dev/null"
        self.new_path = new_path
        self.hunks: list[tuple[int, int, list[str]]] = []


def _parse_patches(diff_text: str) -> list[_FilePatch]:
    """Parse a multi-file unified diff.

    Unlike util/diff.parse_diff this accepts git-style ``--- /dev/null``
    headers (new files) and tolerates header lines without the a//b/ prefix.
    """
    patches: list[_FilePatch] = []
    cur: _FilePatch | None = None
    in_hunk = False

    for raw in diff_text.splitlines():
        stripped = raw.rstrip("\n")
        is_old_header = stripped.startswith("--- ")
        is_new_header = stripped.startswith("+++ ")
        m = _HUNK_RE.match(stripped)

        if is_old_header and not in_hunk:
            target = _clean_path(stripped[4:])
            cur = _FilePatch(target, "")
            patches.append(cur)
            continue
        if is_new_header and cur is not None and not cur.new_path and not cur.hunks:
            cur.new_path = _clean_path(stripped[4:])
            continue
        if m and cur is not None:
            old_start = int(m.group(1))
            old_count = int(m.group(2) if m.group(2) is not None else 1)
            cur.hunks.append((old_start, old_count, []))
            in_hunk = True
            continue
        if in_hunk and cur is not None and cur.hunks:
            if stripped.startswith(("--- ", "+++ ")) or (
                stripped.startswith("@@") and not m
            ):
                # next file's header or malformed hunk: stop this body
                if stripped.startswith("--- "):
                    in_hunk = False
                    cur = _FilePatch(_clean_path(stripped[4:]), "")
                    patches.append(cur)
                elif stripped.startswith("+++ ") and patches and cur.hunks[-1][2]:
                    pass  # stray header inside a hunk body: keep as data
                continue
            if stripped.startswith("diff ") or stripped.startswith("Index: "):
                in_hunk = False
                cur = None
                continue
            cur.hunks[-1][2].append(stripped)
    return [p for p in patches if p.hunks]


# ---------------------------------------------------------------------------
# tolerant hunk application
# ---------------------------------------------------------------------------

def _split_hunk(body: list[str]) -> tuple[list[str], list[str], int, int]:
    """Split a hunk body into (expected_old_lines, new_block, n_add, n_del)."""
    expected: list[str] = []
    new_block: list[str] = []
    n_add = 0
    n_del = 0
    for line in body:
        if line.startswith("\\"):  # "\ No newline at end of file"
            continue
        tag, rest = (line[0], line[1:]) if line else (" ", "")
        if tag == " " or tag == "":
            expected.append(rest)
            new_block.append(rest)
        elif tag == "-":
            expected.append(rest)
            n_del += 1
        elif tag == "+":
            new_block.append(rest)
            n_add += 1
    return expected, new_block, n_add, n_del


def _find_offset(
    lines: list[str], expected: list[str], declared_idx: int, last_end: int
) -> int | None:
    """Locate ``expected`` exactly at/before/after its declared position."""
    n = len(expected)
    span = len(lines) - n
    if span < 0:
        return None
    lo = max(0, declared_idx - MAX_SHIFT, last_end)
    hi = min(span, declared_idx + MAX_SHIFT)
    for delta in range(0, MAX_SHIFT + 1):
        for cand in (declared_idx + delta, declared_idx - delta):
            if cand < lo or cand > hi:
                continue
            if lines[cand : cand + n] == expected:
                return cand
        if declared_idx + delta > hi and declared_idx - delta < lo:
            break
    return None


def _closest_hint(lines: list[str], expected: list[str], n: int = 3) -> str:
    """Best-way fail help: 3 closest real lines to the expected block (~0.3KB)."""
    import difflib as _dl
    try:
        first = ""
        for ln in expected:
            if ln.strip():
                first = ln.strip()[:100]
                break
        if not first:
            return ""
        cands = []
        seen = set()
        for i, ln in enumerate(lines, 1):
            s = ln.strip()[:100]
            if not s or s in seen:
                continue
            seen.add(s)
            cands.append((s, i))
            if len(seen) > 2000:
                break
        pool = [s for s, _ in cands]
        close = _dl.get_close_matches(first, pool, n=n, cutoff=0.4)
        if not close:
            return ""
        parts = []
        for c in close:
            ln_no = next((i for s, i in cands if s == c), 0)
            parts.append(f"line {ln_no}: {c[:80]!r}")
        s0 = next((i for s, i in cands if s == close[0]), 1)
        return " Closest: " + " | ".join(parts) + f" — read offset={max(1, s0 - 2)} limit=7."
    except Exception:
        return ""


def _apply_hunks(
    lines: list[str], hunks: list[tuple[int, int, list[str]]]
) -> tuple[list[str], list[int]]:
    """Apply ordered hunks to a no-newline-suffix line list.

    Returns (new_lines, shifts) where shifts[i] is how far hunk i moved from
    its declared position. Raises ValueError on any conflict (nothing applied).
    """
    out: list[str] = []
    pos = 0
    last_end = 0
    shifts: list[int] = []
    for idx, (start, count, body) in enumerate(hunks):
        expected, new_block, _, _ = _split_hunk(body)
        if not expected and not new_block:
            raise ValueError(f"hunk {idx + 1} is empty")
        declared = start - 1 if start > 0 else 0
        found = _find_offset(lines, expected, declared, last_end)
        if found is None:
            first = expected[0] if expected else "(no context)"
            hint = _closest_hint(lines, expected)
            raise ValueError(
                f"hunk {idx + 1}: no exact match near line {start} "
                f"(expected block starting {first!r}).{hint}"
            )
        out.extend(lines[pos:found])
        out.extend(new_block)
        pos = found + len(expected)
        last_end = pos
        shifts.append(found - declared)
    out.extend(lines[pos:])
    return out, shifts


# ---------------------------------------------------------------------------
# per-file planning / application
# ---------------------------------------------------------------------------

def _resolve(base: Path, relpath: str) -> Path:
    p = Path(relpath)
    if p.is_absolute():
        base_res = base.resolve()
        try:
            pr = p.resolve()
            pr.relative_to(base_res)
        except ValueError:
            raise ValueError(f"refusing absolute path outside worktree: {relpath!r}")
        return p
    joined = (base / p).resolve()
    base_res = base.resolve()
    try:
        joined.relative_to(base_res)
    except ValueError:
        raise ValueError(f"refusing path traversal outside worktree: {relpath!r}")
    return joined


def _read_text(path: Path) -> tuple[str | None, str]:
    """Read preserving exact bytes-as-text; returns (text|None, error)."""
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return fh.read(), ""
    except FileNotFoundError:
        return None, ""
    except (OSError, UnicodeError) as e:
        return None, str(e)


def _plan_one(patch: _FilePatch, base: Path) -> dict[str, Any]:
    creates = patch.old_path == "/dev/null"
    deletes = patch.new_path == "/dev/null"
    target_rel = patch.new_path if creates else patch.old_path
    if deletes and patch.old_path:
        target_rel = patch.old_path
    try:
        path = _resolve(base, target_rel)
    except ValueError as e:
        return {"file": str(base / target_rel), "conflict": str(e), "creates": creates, "deletes": deletes}

    plan: dict[str, Any] = {
        "file": str(path),
        "creates": creates,
        "deletes": deletes,
        "conflict": "",
        "shifts": [],
        "added": 0,
        "removed": 0,
    }

    if creates:
        text, err = _read_text(path)
        if err:
            plan["conflict"] = f"cannot check existing file: {err}"
            return plan
        if text is not None:
            added = "".join(l[1:] + "\n" for h in patch.hunks for l in h[2] if l.startswith("+"))
            if text != added:
                plan["conflict"] = "file already exists with different content"
                return plan
        content = "".join(l[1:] + "\n" for h in patch.hunks for l in h[2] if l.startswith("+"))
        plan["new_content"] = content
        plan["added"] = sum(1 for h in patch.hunks for l in h[2] if l.startswith("+"))
        return plan

    text, err = _read_text(path)
    if err:
        plan["conflict"] = f"cannot read file: {err}"
        return plan
    if text is None:
        plan["conflict"] = "file does not exist"
        return plan

    had_nl = text.endswith("\n")
    lines = text.split("\n")
    if had_nl and lines and lines[-1] == "":
        lines.pop()

    try:
        new_lines, shifts = _apply_hunks(lines, patch.hunks)
    except ValueError as e:
        plan["conflict"] = str(e)
        return plan

    new_text = "\n".join(new_lines)
    if had_nl:
        new_text += "\n"

    if deletes:
        removed_ok = not any(l for l in new_lines)
        if not removed_ok:
            plan["conflict"] = "deletion leaves non-empty remainder; diff does not match file"
            return plan
        plan["delete_target"] = True
    else:
        plan["new_content"] = new_text

    plan["shifts"] = shifts
    for _, _, body in patch.hunks:
        plan["removed"] += sum(1 for l in body if l.startswith("-"))
        plan["added"] += sum(1 for l in body if l.startswith("+"))
    return plan


# ---------------------------------------------------------------------------
# journal (undo support)
# ---------------------------------------------------------------------------

def _journal_file() -> Path:
    return GPath.data / JOURNAL_PATH_NAME


def _journal_load() -> list[dict[str, Any]]:
    try:
        data = json.loads(_journal_file().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _journal_save(entries: list[dict[str, Any]]) -> None:
    path = _journal_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(entries)
    from ..session import _write_durable
    _write_durable(path, data, file_sync=True, dir_sync=True)


def _journal_append(entry: dict[str, Any]) -> None:
    with _LOCK:
        entries = _journal_load()
        entries.append(entry)
        while len(entries) > MAX_JOURNAL_ENTRIES:
            entries.pop(0)
        _journal_save(entries)


def _journal_pop() -> dict[str, Any] | None:
    with _LOCK:
        entries = _journal_load()
        if not entries:
            return None
        entry = entries.pop()
        _journal_save(entries)
        return entry


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _action_apply(diff_text: str, dry_run: bool, message: str, partial: bool = False) -> dict:
    diff_text = (diff_text or "").replace("\r\n", "\n")
    if not diff_text.strip():
        return {"output": "Empty diff: nothing to apply.", "error": True}
    patches = _parse_patches(diff_text)
    if not patches:
        return {
            "output": (
                "Could not parse any file sections from the diff. Expected "
                "unified format:\n--- a/file\n+++ b/file\n@@ ... @@\n..."
            ),
            "error": True,
        }

    plans: list[dict[str, Any]] = []
    base = Path.cwd()
    for patch in patches:
        plans.append(_plan_one(patch, base))

    problems = [p for p in plans if p.get("conflict")]
    ok_files = [p for p in plans if not p.get("conflict")]
    label = "Dry run" if dry_run else "Apply"
    per_file = [{"file": p["file"], "ok": not bool(p.get("conflict")),
                 "added": p.get("added", 0), "removed": p.get("removed", 0),
                 "conflict": p.get("conflict", "")} for p in plans]

    if problems and not (partial and ok_files):
        lines = [f"{label} FAILED — {len(problems)} of {len(plans)} files conflict. "
                 "Nothing was written."]
        for p in problems:
            lines.append(f"  {p['file']}: {p['conflict']}")
        if ok_files:
            lines.append(f"(remaining {len(ok_files)} file(s) would apply cleanly — retry with partial=true to apply them)")
        return {"output": "\n".join(lines), "error": True,
                "metadata": {"files": per_file, "applied": 0, "failed": len(problems)}}
    failed = problems if (partial and ok_files) else []
    if failed:
        plans = ok_files

    summary_lines = [f"{label}: {len(plans)} file(s) OK."]
    for p in plans:
        kind = "create" if p.get("creates") else ("delete" if p.get("delete_target") else "modify")
        shift_note = ""
        real_shifts = [abs(s) for s in p.get("shifts", [])]
        if real_shifts:
            shift_note = f" (hunk offset {max(real_shifts):+d})" if max(real_shifts) else ""
        summary_lines.append(f"  {kind} {p['file']}  +{p['added']} -{p['removed']}{shift_note}")

    if dry_run:
        if failed:
            summary_lines.append("")
            summary_lines.append(f"SKIPPED ({len(failed)} file(s) conflict — fix and resend only these):")
            for p in failed:
                summary_lines.append(f"  SKIP {p['file']}: {p['conflict']}")
        meta_files = ([{"file": p["file"], "ok": True, "added": p.get("added", 0),
                        "removed": p.get("removed", 0)} for p in plans]
                      + [{"file": p["file"], "ok": False, "conflict": p.get("conflict", "")} for p in failed])
        return {"output": "\n".join(summary_lines),
                "metadata": {"dry_run": True, "files": meta_files,
                             "applied": 0, "failed": len(failed),
                             "partial": bool(failed)}}

    # ---- commit: backup originals FIRST, then write/delete everything ----
    backup: list[dict[str, Any]] = []
    errors: list[str] = []
    for p in plans:
        path = Path(p["file"])
        existed = path.exists()
        original, err = (_read_text(path) if existed else (None, ""))
        if existed and err:
            errors.append(f"{path}: could not back up ({err}); aborting before any write")
            break
        backup.append({"path": str(path), "existed": existed,
                       "content": original, "creates": bool(p.get("creates"))})

    if errors:
        return {"output": "Apply aborted (backup failed):\n" + "\n".join(errors),
                "error": True}

    for item in backup:
        path = Path(item["path"])
        p = next(x for x in plans if x["file"] == str(path))
        if p.get("creates") or p.get("delete_target"):
            continue
        try:
            _atomic_write(path, p["new_content"])
        except (OSError, UnicodeError) as e:
            errors.append(f"write failed mid-patch: {path}: {e}")
    if errors:
        # roll back what we already wrote using the fresh backup
        for item in backup:
            path = Path(item["path"])
            try:
                if item["existed"]:
                    _atomic_write(path, item["content"])
                elif path.exists() and item["creates"]:
                    path.unlink()
            except OSError:
                pass
        return {"output": "Apply FAILED mid-write and was rolled back:\n" +
                "\n".join(errors), "error": True}

    for item in backup:
        path = Path(item["path"])
        p = next(x for x in plans if x["file"] == str(path))
        try:
            if p.get("creates"):
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(path, p["new_content"])
            elif p.get("delete_target"):
                path.unlink()
        except (OSError, UnicodeError) as e:
            errors.append(f"{path}: {e}")

    # feed the verify tool's homework checker: track writes, forget deletes
    from .verify import track, untrack

    for item in backup:
        path = Path(item["path"])
        p = next(x for x in plans if x["file"] == str(path))
        if p.get("delete_target"):
            untrack(path)
        else:
            track(path, "apply_patch")

    _journal_append({
        "time": time.time(),
        "message": message or "",
        "files": backup,
    })

    diag: dict = {}
    try:
        from .lsp import _diagnostics_py
        all_issues: dict[str, list[str]] = {}
        for p in plans:
            fp = Path(p["file"])
            if fp.suffix == ".py" and fp.exists():
                issues = _diagnostics_py(fp)
                if issues:
                    all_issues[str(fp)] = issues[:20]
        if all_issues:
            summary_lines.append("")
            summary_lines.append("LSP errors detected in patched files, please fix:")
            for f, issues in list(all_issues.items())[:5]:
                summary_lines.append(f"  {f}:")
                for x in issues[:5]:
                    summary_lines.append(f"    - {x}")
            diag = {"diagnostics": all_issues}
        else:
            diag = {"diagnostics": {}}
    except Exception:
        pass

    if errors:
        summary_lines.append("")
        summary_lines.append("PARTIAL FAILURES (see above files):")
        summary_lines.extend(f"  {e}" for e in errors)
    if failed:
        summary_lines.insert(1, f"Partial: {len(plans)} file(s) applied, {len(failed)} skipped.")
        summary_lines.append("")
        summary_lines.append(f"SKIPPED ({len(failed)} — resend only these hunks):")
        for p in failed:
            summary_lines.append(f"  SKIP {p['file']}: {p['conflict']}")
    summary_lines.append("Undo available: run apply_patch with action=\"undo\".")
    meta_files = ([{"file": p["file"], "ok": True, "added": p["added"], "removed": p["removed"]}
                   for p in plans]
                  + [{"file": p["file"], "ok": False, "conflict": p.get("conflict", "")} for p in failed])
    return {"output": "\n".join(summary_lines),
            "metadata": {"applied": True, "partial": bool(failed), "files": list(meta_files),
                         "failed": [p["file"] for p in failed], **diag}}


def _action_undo() -> dict:
    entry = _journal_pop()
    if entry is None:
        return {"output": "Nothing to undo: the journal is empty.", "error": True}
    restored: list[str] = []
    errors: list[str] = []
    for item in reversed(entry.get("files", [])):
        path = Path(item["path"])
        try:
            if item.get("existed") and item.get("content") is not None:
                _atomic_write(path, item["content"])
                restored.append(f"restored {path}")
            else:
                if path.exists():
                    path.unlink()
                    restored.append(f"deleted created file {path}")
        except (OSError, UnicodeError) as e:
            errors.append(f"{path}: {e}")
    msg = entry.get("message") or "(no message)"
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("time", 0)))
    lines = [f"Undid patch \"{msg}\" ({when}):"]
    lines.extend(f"  {r}" for r in restored)
    if errors:
        lines.append("ERRORS during undo:")
        lines.extend(f"  {e}" for e in errors)
    remaining = len(_journal_load())
    lines.append(f"Patches left in journal: {remaining}")
    return {"output": "\n".join(lines), "error": bool(errors),
            "metadata": {"undone": True, "restored": restored}}


def _action_history() -> dict:
    entries = _journal_load()
    if not entries:
        return {"output": "Patch journal is empty."}
    lines = [f"Last {len(entries)} applied patch(es):"]
    for i, e in enumerate(entries, 1):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("time", 0)))
        files = ", ".join(Path(f["path"]).name for f in e.get("files", [])[:5])
        more = "" if len(e.get("files", [])) <= 5 else ", ..."
        lines.append(f"{i}. [{when}] {e.get('message') or '(no message)'} — {files}{more}")
    lines.append("Undo reverts the LAST one.")
    return {"output": "\n".join(lines)}


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------

_ACTIONS = ("apply", "undo", "history")


def tool() -> Tool:
    description = """Apply one unified diff across many files, atomically (all-or-nothing). partial=true applies clean files, lists bad hunks (~0.3KB resend). dry_run previews; undo reverts."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "apply").strip().lower()
        if action not in _ACTIONS:
            return {"output": f"Unknown action {action!r} (want: {', '.join(_ACTIONS)}).",
                    "error": True}
        if action == "undo":
            return _action_undo()
        if action == "history":
            return _action_history()
        return _action_apply(
            str(input.get("diff") or ""),
            dry_run=bool(input.get("dry_run", False)),
            message=str(input.get("message") or ""),
            partial=bool(input.get("partial", False)),
        )

    return Tool(
        name="apply_patch",
        description=description,
        parameters=schema_with(
            {
                "diff": {
                    "type": "string",
                    "description": "Unified diff (multi-file ok)",
                    "optional": True,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview without writing",
                    "optional": True,
                },
                "message": {
                    "type": "string",
                    "description": "Journal label",
                    "optional": True,
                },
                "partial": {
                    "type": "boolean",
                    "description": "Apply clean files, skip bad hunks (1 hunk resend, not N files)",
                    "optional": True,
                },
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "apply, undo, or history",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="apply_patch",
    )