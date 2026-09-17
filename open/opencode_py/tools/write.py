"""write tool: create a NEW file (edits belong to edit/apply_patch).

RULES (full guidance for maintainers; the sent schema is short):
- Atomic (temp + rename), creates parent dirs.
- Returns stats only (bytes/lines) — never echoes content back (saves 2x data).
- Identical content -> no-op, 0 bytes written.
- Existing file with different content -> error guiding to edit/apply_patch
  (or force=true to blast, dry_run=true to preview size first).
- Never proactively create docs (*.md/README) unless asked.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .registry import Tool, schema_with


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            try:
                fh.flush()
                os.fsync(fh.fileno())
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
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write(filePath: str, content: str, dry_run: bool = False, force: bool = False) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    content = content if isinstance(content, str) else str(content)
    nbytes = len(content.encode("utf-8", errors="replace"))
    nlines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    existed = path.exists()
    if existed:
        try:
            current = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError) as e:
            return {"output": f"Error reading file {path}: {e}", "error": True}
        if current == content:
            return {
                "output": f"Already up to date ({nbytes} bytes, {nlines} lines) — wrote nothing.",
                "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                             "identical": True, "created": False},
            }
        if dry_run:
            return {
                "output": f"Dry run — no changes written. Would overwrite {path} "
                           f"({len(current.encode('utf-8', errors='replace'))}B -> {nbytes}B).",
                "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                             "dry_run": True, "created": False},
            }
        if not force:
            return {
                "output": (
                    f"File exists with different content ({path}). "
                    f"Use edit (small change) or apply_patch (multi-file), "
                    f"or retry write with force=true to overwrite "
                    f"({len(current.encode('utf-8', errors='replace'))}B -> {nbytes}B). "
                    f"Dry run first: dry_run=true previews size without writing."
                ),
                "error": True,
                "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines},
            }
    if dry_run:
        return {
            "output": f"Dry run — no changes written. Would create {path} ({nbytes}B, {nlines} lines).",
            "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                         "dry_run": True, "created": not existed},
        }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if existed and force:
            # Re-check the body right before overwriting: another writer may
            # have changed it since the first read — refuse instead of
            # silently losing their update (TOCTOU guard).
            try:
                again = path.read_text(encoding="utf-8", errors="replace")
                if again != current:
                    return {
                        "output": (
                            f"File changed since read ({path}). Re-read it and "
                            f"retry — refusing to silently overwrite another writer."
                        ),
                        "error": True,
                    }
            except (OSError, UnicodeError) as e:
                return {"output": f"Error re-reading file {path}: {e}", "error": True}
        _atomic_write(path, content)
    except (OSError, UnicodeError) as e:
        return {"output": f"Error writing file {path}: {e}", "error": True}
    # feed the verify tool's homework checker
    from .verify import track

    track(path, "write")
    verb = "Created" if not existed else "Overwrote (force)"
    note = ""
    diag: dict = {}
    try:
        if path.suffix == ".py":
            from .lsp import _diagnostics_py
            issues = _diagnostics_py(path)
            if issues:
                note = "\n\nLSP errors detected in this file, please fix:\n" + "\n".join(f"- {x}" for x in issues[:10])
                diag = {"diagnostics": {str(path): issues[:20]}}
            else:
                diag = {"diagnostics": {}}
    except Exception:
        pass
    return {
        "output": f"{verb} {path} ({nbytes} bytes, {nlines} lines)." + note,
        "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                     "created": not existed, **diag},
    }


def tool() -> Tool:
    description = """Write a NEW file, creating parent dirs. Stats back, no content echo. Existing file? Use edit/apply_patch (or force=true). dry_run previews size."""

    def run(input: dict) -> dict:
        return _write(input["filePath"], input["content"],
                      dry_run=bool(input.get("dry_run", False)),
                      force=bool(input.get("force", False)))

    return Tool(
        name="write",
        description=description,
        parameters=schema_with(
            {
                "content": {"type": "string", "description": "File content"},
                "filePath": {"type": "string", "description": "Absolute path"},
                "dry_run": {"type": "boolean", "description": "Preview size without writing", "optional": True},
                "force": {"type": "boolean", "description": "Overwrite an existing different file", "optional": True},
            },
            ["content", "filePath"],
        ),
        run=run,
        permission="edit",
    )
