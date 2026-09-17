"""Unified diff create/apply (pure Python).

Used by the edit tool and permission dialogs. Produces/consumes standard
`--- a/<path>` / `+++ b/<path>` unified diffs compatible with `patch`.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_HEADER_RE = re.compile(r"^--- a/(.*)$")
_HEADER2_RE = re.compile(r"^\+\+\+ b/(.*)$")


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    hunks: list[list[str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        lines = [f"--- a/{self.old_path}", f"+++ b/{self.new_path}"]
        for hunk in self.hunks:
            lines.extend(hunk)
        return "\n".join(lines) + "\n"


def create_diff(old: str, new: str, old_path: str = "a", new_path: str = "b", context: int = 3) -> str:
    """Create a unified diff string between two file contents.

    difflib is unreliable when a file lacks a trailing newline, so both inputs
    are normalized to end with one before diffing (line content is unchanged).
    """
    nold = old if old.endswith("\n") else old + "\n"
    nnew = new if new.endswith("\n") else new + "\n"
    old_lines = nold.splitlines(keepends=True)
    new_lines = nnew.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{old_path}",
        tofile=f"b/{new_path}",
        n=context,
    )
    result = "".join(diff)
    if not result:
        return ""
    if not result.endswith("\n"):
        result += "\n"
    return result


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff into FileDiff objects."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk: list[str] = []

    def flush_hunk() -> None:
        nonlocal current_hunk
        if current is not None and current_hunk:
            current.hunks.append(current_hunk)
        current_hunk = []

    for line in diff_text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("--- a/") and current is None:
            flush_hunk()
            current = FileDiff(old_path=stripped[6:], new_path="")
            continue
        if stripped.startswith("+++ b/") and current is not None and current.new_path == "":
            current.new_path = stripped[6:]
            continue
        if stripped.startswith("@@") and current is not None:
            flush_hunk()
        if current is not None:
            current_hunk.append(line)
    flush_hunk()
    if current is not None and current.new_path == "":
        current.new_path = current.old_path
    if current is not None:
        files.append(current)
    return files


def apply_diff(diff_text: str, old_text: str) -> str:
    """Apply a unified diff to old_text, returning the new text.

    Raises ValueError if the diff does not apply cleanly.
    """
    lines = old_text.splitlines(keepends=True)
    # Normalize: ensure old lines have no \r (we operate on splitlines content).
    # difflib normalized the file to end with a newline; do the same so hunks match.
    had_newline = old_text.endswith("\n")
    normalized_text = old_text if had_newline else old_text + "\n"
    normalized_lines = normalized_text.splitlines(keepends=True)
    last_old_line = 0
    result: list[str] = []

    files = parse_diff(diff_text)
    if not files:
        raise ValueError("no hunks in diff")

    # Parse hunks from the single-file diff
    hunks: list[tuple[int, int, list[str]]] = []
    for file in files:
        for hunk in file.hunks:
            text = "".join(hunk)
            m = _HUNK_RE.match(text)
            if not m:
                continue
            old_start = int(m.group(1))
            old_count = int(m.group(2) or 1)
            if old_start == 0 and old_count == 0:
                old_start = 1
            body = text.splitlines(keepends=True)[1:]
            hunks.append((old_start, old_count, body))

    for old_start, old_count, body in hunks:
        # copy context lines before this hunk (1-based old_start -> index old_start-1)
        idx = old_start - 1
        if idx < last_old_line:
            raise ValueError("overlapping hunks")
        result.extend(normalized_lines[last_old_line:idx])
        last_old_line = idx
        for line in body:
            if line.startswith(" ") or line.startswith("\n") or line == "\n":
                # context line - must match
                if line.startswith(" "):
                    content = line[1:]
                else:
                    content = line
                if last_old_line >= len(normalized_lines) or normalized_lines[last_old_line] != content:
                    raise ValueError(
                        f"context mismatch at line {last_old_line + 1}: expected "
                        f"{normalized_lines[last_old_line]!r} got {content!r}"
                    )
                result.append(content)
                last_old_line += 1
            elif line.startswith("-"):
                if last_old_line >= len(normalized_lines) or normalized_lines[last_old_line] != line[1:]:
                    raise ValueError(
                        f"removal mismatch at line {last_old_line + 1}: expected "
                        f"{normalized_lines[last_old_line]!r} got {line[1:]!r}"
                    )
                last_old_line += 1
            elif line.startswith("+"):
                result.append(line[1:])
            elif line.startswith("\\"):
                pass  # "\ No newline at end of file"
            else:
                raise ValueError(f"malformed diff line: {line!r}")

    result.extend(normalized_lines[last_old_line:])
    out = "".join(result)
    # restore the missing trailing newline (difflib normalization added one)
    if not had_newline and out.endswith("\n"):
        out = out[:-1]
    return out


def diff_stat(diff_text: str) -> tuple[int, int]:
    """Return (added, removed) counts from a diff string."""
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed
