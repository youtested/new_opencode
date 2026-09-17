"""edit tool: exact string replacement + diff verification.

RULES (full guidance for maintainers; the sent schema is short):
- oldString must match EXACTLY incl. indentation (never the `N:` line
  prefix from read output). Not found -> error WITH 3 closest line hints
  so the next try hits; found many times -> add context or use replaceAll.
- dry_run=true previews the diff without writing (~0.3KB).
- Have grep offset/limit? Use offset=+limit=+newString (line_range mode):
  no verbatim copy needed, no mismatch fail.
- Prefer editing over write. Cascade tries fuzzy fallbacks for near-miss.
"""

from __future__ import annotations

import bisect
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

from .registry import Tool, schema_with

from .write import _atomic_write


def _py_diag_note(path: Path) -> str:
    try:
        if path.suffix != ".py":
            return ""
        from .lsp import _diagnostics_py
        issues = _diagnostics_py(path)
        if not issues:
            return ""
        lines = ["", "LSP errors detected in this file, please fix:"]
        lines.extend(f"- {x}" for x in issues[:10])
        return "\n" + "\n".join(lines)
    except Exception:
        return ""


def _py_diag_meta(path: Path) -> dict:
    try:
        if path.suffix != ".py":
            return {}
        from .lsp import _diagnostics_py
        issues = _diagnostics_py(path)
        if not issues:
            return {"diagnostics": {}}
        return {"diagnostics": {str(path): issues[:20]}}
    except Exception:
        return {}

_HSPACE_RE = re.compile(r"[ \t]+")


def find_matches(content: str, old: str) -> list[tuple[int, int]]:
    """Return all (start, end) exact-match spans of `old` in `content`."""
    matches = []
    start = 0
    while True:
        idx = content.find(old, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(old)))
        start = idx + 1
    return matches


def _line_offsets(content: str) -> list[int]:
    """Character offset of the start of each line (plus one past the end)."""
    offsets = [0]
    for line in content.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _line_starts(text: str) -> list[int]:
    """Character offset where each line begins (no trailing sentinel).

    ``text.find`` skips forward, so this is O(n) total even on huge files;
    feeding the result to ``bisect`` gives O(log n) line-index lookups.
    """
    starts = [0]
    i = text.find("\n")
    while i != -1:
        starts.append(i + 1)
        i = text.find("\n", i + 1)
    return starts


def _line_window_spans(
    content: str,
    content_lines: list[str],
    offsets: list[int],
    old: str,
    key,
    allow_shorter: bool,
    keyed_content: list[str],
) -> list[tuple[int, int]]:
    """Return char spans of line windows whose keyed lines equal keyed old lines.

    ``key`` normalizes a single line (e.g. ``str.strip``, whitespace collapse).
    A single trailing empty old line (from a trailing ``\\n``) is dropped so we
    don't require the file itself to end with a newline.

    The caller precomputes ``content``/``content_lines``/``offsets``/
    ``keyed_content`` once (this helper is used by several successive fallback
    passes; re-splitting for each would be wasteful). The span covers the
    matched lines' text. When ``old`` does NOT end with a newline, the span
    stops before the newline that follows the last matched line (the natural
    exact-match semantics); when it does end with ``\\n`` the newline is part of
    the match and stays inside the span.
    """
    old_lines = old.split("\n")
    ends_with_nl = old.endswith("\n")
    if old_lines and old_lines[-1] == "":
        old_lines.pop()
    if not allow_shorter and any(line.strip() == "" for line in old_lines):
        return []
    keyed_old = [key(line) for line in old_lines]
    if not keyed_old:
        return []

    width = len(keyed_old)
    spans = []
    for i in range(len(content_lines) - width + 1):
        if keyed_content[i : i + width] == keyed_old:
            start = offsets[i]
            end = offsets[i + width]
            # offsets[i+width] points at the START of the next line, i.e. right
            # after the line terminator ending the last matched line. If the
            # caller's old text does not itself end with a newline, that line
            # terminator must not be swallowed by the replacement (it would
            # merge the next line into the edit). Walk back over the full
            # terminator so CRLF files don't leave a stray \r behind.
            if not ends_with_nl:
                while end > start and content[end - 1] in "\r\n":
                    end -= 1
            spans.append((start, end))
    return spans


def _collapse_hspace(line: str) -> str:
    return _HSPACE_RE.sub(" ", line).strip()


def _candidate_spans(content: str, old: str) -> list[tuple[int, int]]:
    if len(old) > 200_000 or old.count("\n") > 5000:
        return find_matches(content, old)
    """Return best-match spans of `old` in `content` under the replacer cascade.

    Successively fuzzier normalizers are tried; the first that finds any match
    wins. Exact matches are returned as plain substring spans (fast path).
    The lines/offsets/keyed-lines are computed ONCE and shared across the
    fallback passes (they used to be rebuilt for each pass).
    """
    exact = find_matches(content, old)
    if exact:
        return exact
    content_lines = content.split("\n")
    offsets = _line_offsets(content)
    keyed_cache: dict[Callable[[str], str], list[str]] = {}
    for key, allow_shorter in (
        (str.strip, False),     # leading/trailing whitespace per line
        (str.rstrip, True),     # trailing-only (file may not end with newline)
        (_collapse_hspace, False),
        (_collapse_hspace, True),
    ):
        keyed_content = keyed_cache.get(key)
        if keyed_content is None:
            keyed_content = [key(line) for line in content_lines]
            keyed_cache[key] = keyed_content
        spans = _line_window_spans(
            content, content_lines, offsets, old, key, allow_shorter, keyed_content
        )
        if spans:
            return spans
    return []


def _span_diff(
    content: str,
    new_content: str,
    spans: list[tuple[int, int]],
    new_texts: list[str],
    old_path: str,
    new_path: str,
    context: int = 3,
) -> str:
    """Build a unified diff for edits whose changed regions are KNOWN upfront.

    The edit tool always knows exactly which char spans it replaced, so it does
    not need difflib — diffing two whole files is quadratic on long runs of
    unique lines (1.7 s on a 50k-line file). Instead we emit hunks snapped to
    LINE boundaries:

    - the removed block is the *whole* old lines each edit touches, and
    - the added block is the *whole* new lines at the same (offset-adjusted)
      position, including the residual blank line a mid-line deletion leaves.

    Edits within ``2*context`` lines merge into one hunk (as difflib/GNU do);
    farther apart stay separate so a small edit never drags in a huge file as
    context. Context lines are strictly bounded by the neighboring edits, so
    every hunk is patch-applicable. ``spans``/``new_texts`` are zipped: for
    span index i, char range ``content[spans[i]]`` was replaced by
    ``new_texts[i]`` when building ``new_content``.

    Two cosmetic differences from GNU/difflib are accepted: hunk headers always
    carry an explicit ``,count`` (no compact ``-1 +1`` form), and when the
    replacement text contains a copy of a removed line (e.g. wrapping a line
    with itself), difflib's LCS alignment keeps the original line as context
    while this span-based writer emits the extra removal/addition instead. Both
    produce a valid, patch-applicable hunk; only the display differs.

    Line numbers are computed by bisecting precomputed line-start offsets
    (``str.find`` skips forward, so building them is O(n); each span's lookup
    is O(log n)) — never a per-line Python loop over the whole file, which is
    the dominant cost on slow (32-bit ARM) machines.
    """
    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    # Line-start offsets for O(log n) line-index lookups via bisect — a handful
    # of ``str.count``/``find`` scans per span was O(span offsets), quadratic on
    # replace_all with thousands of spans.
    old_starts = _line_starts(content)
    new_starts = _line_starts(new_content)
    out: list[str] = []
    delta = 0  # cumulative new_len - old_len from prior (left) spans
    spans = [(s, e, t) for (s, e), t in zip(spans, new_texts) if not (s == e and not t)]
    if not spans:
        return ""
    # Per-span line extents, tracked SEPARATELY in old and new coordinates.
    # For old, `lo`/`hi_old` derive from the original char offsets. For new,
    # the region lands at new_s = s + delta (leftward edits shift it), so its
    # line index in the final new content is a fresh lookup — it only equals
    # `lo` for the first/multi-span-free case.
    recs = []
    for s, e, t in spans:
        new_s = s + delta
        new_e = new_s + len(t)
        lo = bisect.bisect_right(old_starts, s) - 1
        hi_old = lo if e == s else bisect.bisect_right(old_starts, e - 1)
        new_lo = bisect.bisect_right(new_starts, new_s) - 1
        # The unchanged suffix resumes at the first line boundary on or after
        # new_e (the byte where content[e:] lands); hi_new is that line's index.
        hi_new = bisect.bisect_left(new_starts, new_e + 1)
        delta += len(t) - (e - s)
        if not old_lines[lo:hi_old] and not new_lines[new_lo:hi_new]:
            continue
        recs.append((lo, hi_old, new_lo, hi_new))

    # Collapse spans whose OLD line ranges touch or overlap into one change
    # region — several spans can sit inside a single line (e.g. replace_all of
    # one character), and treating them separately would duplicate the line in
    # the removed/added blocks. The region's added block spans the union of the
    # new ranges, so it naturally dedups too.
    regions: list[tuple[int, int, int, int]] = []
    for lo, hi_old, new_lo, hi_new in recs:
        if regions and lo <= regions[-1][1]:
            plo, phi, pnl, pnh = regions[-1]
            regions[-1] = (plo, max(phi, hi_old), min(pnl, new_lo), max(pnh, hi_new))
        else:
            regions.append((lo, hi_old, new_lo, hi_new))

    # Group consecutive regions into hunks. Regions within 2*context lines merge
    # into one hunk (as difflib/GNU do); farther apart stay separate so a small
    # edit never drags in half a huge file as context. Context is also clamped
    # to the neighboring hunks, so it never crosses a changed line and every
    # hunk stays patch-applicable.
    groups: list[list[tuple[int, int, int, int]]] = []
    for reg in regions:
        if groups:
            _, prev_hi_old, _, prev_hi_new = groups[-1][-1]
            lo, hi_old, new_lo, hi_new = reg
            if lo - prev_hi_old < 2 * context or new_lo - prev_hi_new < 2 * context:
                groups[-1].append(reg)
                continue
        groups.append([reg])

    out.extend([f"--- a/{old_path}\n", f"+++ b/{new_path}\n"])
    for gi, group in enumerate(groups):
        lo0, _, new_lo0, _ = group[0]
        _, _, _, hi_newN = group[-1]
        # Context stops at the neighboring hunks so it only ever shows lines
        # that exist (unchanged) in both files.
        prev_old = groups[gi - 1][-1][1] if gi else 0
        prev_new = groups[gi - 1][-1][3] if gi else 0
        old_ctx = max(prev_old, lo0 - context)
        new_ctx = max(prev_new, new_lo0 - context)
        next_new = groups[gi + 1][0][2] if gi + 1 < len(groups) else float("inf")
        ctx_before = old_lines[old_ctx:lo0]
        ctx_after = new_lines[hi_newN : min(next_new, hi_newN + context)]

        blocks: list[tuple[str, list[str]]] = []
        for idx, (lo, hi_old, new_lo, hi_new) in enumerate(group):
            if idx:
                prev_hi_old, prev_hi_new = group[idx - 1][1], group[idx - 1][3]
                old_gap = old_lines[prev_hi_old:lo]
                new_gap = new_lines[prev_hi_new:new_lo]
                if old_gap == new_gap:
                    # A shared unchanged run between the regions is context.
                    blocks.append((" ", list(old_gap)))
                else:
                    # The gaps diverge (a nearby edit changed line counts);
                    # show them as real removals/additions instead of false
                    # shared context.
                    blocks.append(("-", list(old_gap)))
                    blocks.append(("+", list(new_gap)))
            blocks.append(("-", list(old_lines[lo:hi_old])))
            blocks.append(("+", list(new_lines[new_lo:hi_new])))

        # Standard hunk header: 1-based start of the FIRST shown line (not the
        # first changed line) and the TOTAL line count for that side — context
        # lines included. The old/new starts can differ once earlier edits
        # changed line counts.
        old_start = old_ctx + 1
        new_start = new_ctx + 1
        old_count = len(ctx_before) + sum(len(lines) for kind, lines in blocks if kind != "+") + len(ctx_after)
        new_count = len(ctx_before) + sum(len(lines) for kind, lines in blocks if kind != "-") + len(ctx_after)
        out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@\n")

        def _emit(prefix: str, lines: list[str]) -> None:
            for ln in lines:
                if ln.endswith("\n"):
                    out.append(prefix + ln)
                else:
                    # Only the file's final line lacks a newline; GNU diff and
                    # opencode (jsdiff) flag it so a patch tool doesn't silently
                    # normalize the line ending.
                    out.append(prefix + ln + "\n")
                    out.append("\\ No newline at end of file\n")

        _emit(" ", ctx_before)
        for kind, lines in blocks:
            _emit(kind, lines)
        _emit(" ", ctx_after)
    return "".join(out)


def _closest_hints(content: str, old: str, n: int = 3) -> list[str]:
    """Best-way fail help: 3 closest file lines to old's first line (~0.3KB).
    Kills blind retries: the next try copies exact text with a line number."""
    import difflib as _dl
    try:
        first = (old.split("\n", 1)[0] if old else "").strip()[:120]
        lines = content.split("\n")
        cands = sorted({ln.strip()[:120] for ln in lines if ln.strip()})[:500]
        if not first or not cands:
            return []
        close = _dl.get_close_matches(first, cands, n=n, cutoff=0.4)
        out = []
        for c in close:
            try:
                idx = next(i for i, ln in enumerate(lines, 1) if ln.strip()[:120] == c)
            except StopIteration:
                idx = 0
            out.append(f"line {idx}: {c[:100]}")
        return out
    except Exception:
        return []


def _do_edit(path: Path, old: str, new: str, replace_all: bool = False, dry_run: bool = False) -> dict:
    # Read/write with newline="" to preserve the file's exact line endings
    # (universal newlines mode would silently convert CRLF files to LF on any
    # edit, rewriting the whole file's bytes for a one-line change).
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            content = fh.read()
    except (OSError, UnicodeError) as e:
        return {"output": f"Could not read file {path}: {e}", "error": True}

    if old == new:
        return {"output": "oldString and newString are identical. No changes made.", "error": True}
    if not old:
        return {"output": "oldString is empty. Use the write tool to replace the whole file.", "error": True}
    if old[0] == "\n" or old[-1] == "\n":
        return {
            "output": (
                "oldString must not start or end with a newline. "
                "Include only the text to replace; the surrounding line breaks "
                "are preserved automatically."
            ),
            "error": True,
        }

    matches = _candidate_spans(content, old)
    if not matches:
        hints = _closest_hints(content, old)
        msg = "oldString not found in content."
        if hints:
            msg += " Closest " + str(len(hints)) + ": " + " | ".join(hints)
            msg += " — copy exact text, or use offset=/limit= from grep."
        else:
            msg += " Use offset=/limit=+newString from your grep anchor (no verbatim copy needed)."
        return {"output": msg, "error": True, "metadata": {"hints": hints}}

    if len(matches) > 1 and not replace_all:
        starts = _line_starts(content)
        locs = []
        for s, e in matches[:5]:
            try:
                import bisect as _bi
                locs.append(str(_bi.bisect_right(starts, s)))
            except Exception:
                pass
        msg = "Found multiple matches for oldString. Provide more surrounding lines in oldString to identify the correct match."
        if locs:
            msg += f" Matches at lines: {', '.join(locs)} — or use offset=<one line>+limit= to target exactly."
        return {"output": msg, "error": True, "metadata": {"match_lines": locs}}

    if replace_all:
        # replace non-overlapping spans right-to-left so offsets stay valid
        spans = sorted(matches, key=lambda s: (s[0], s[1]))
        chosen = []
        last_end = -1
        for s, e in spans:
            if s >= last_end:
                chosen.append((s, e))
                last_end = e
        new_texts = [new] * len(chosen)
        # Repetitive whole-string splicing is O(n²) for thousands of spans
        # (75 s on a 50k-line replace_all); assemble in one pass instead.
        parts: list[str] = []
        last = 0
        for s_off, e_off in chosen:
            parts.append(content[last:s_off])
            parts.append(new)
            last = e_off
        parts.append(content[last:])
        new_content = "".join(parts)
    else:
        start, end = matches[0]
        new_text = new
        # A fuzzy (non-exact) match replaces the whole matched line region, so
        # the file's original leading indentation would be lost when the model's
        # new string doesn't carry it. Preserve the matched line's indentation
        # when the model omitted it (the common copy-without-indent case).
        if not find_matches(content, old):
            region = content[start:end]
            first_line = region.split("\n", 1)[0]
            leading = first_line[: len(first_line) - len(first_line.lstrip())]
            if leading and not new.startswith((" ", "\t")):
                new_text = leading + new
        chosen = [(start, end)]
        new_texts = [new_text]
        new_content = content[:start] + new_text + content[end:]

    diff = _span_diff(content, new_content, chosen, new_texts, str(path), str(path))
    if dry_run:
        return {
            "output": "Dry run — no changes written.\n" + diff,
            "metadata": {"diff": diff, "replaceAll": replace_all, "dry_run": True},
        }
    # verify by writing then reading back
    try:
        _atomic_write(path, new_content)
    except (OSError, UnicodeError) as e:
        return {"output": f"Error writing file {path}: {e}", "error": True}

    return {
        "output": "Edit applied successfully." + _py_diag_note(path),
        "metadata": {"diff": diff, "replaceAll": replace_all, **_py_diag_meta(path)},
    }


def _edit_lines(path: Path, offset: int, limit: int, new: str, dry_run: bool = False) -> dict:
    """Line-range mode: offset/limit from a grep anchor + newString.
    No verbatim copy, so no mismatch fail. Best with dry_run first."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError) as e:
        return {"output": f"Could not read file {path}: {e}", "error": True}
    lines = text.split("\n")
    total = len(lines)
    s = max(1, int(offset or 1))
    lim = max(1, int(limit or 1))
    if s > total:
        return {"output": f"offset {s} past end of file ({total} lines).", "error": True}
    e = min(total, s + lim - 1)
    old_block = "\n".join(lines[s - 1:e])
    new_lines = new.split("\n")
    out_lines = lines[:s - 1] + new_lines + lines[e:]
    new_content = "\n".join(out_lines)
    from .write import _atomic_write as _aw
    diff = _span_diff(text, new_content, [(0, 0)], [""], str(path), str(path)) if False else ""
    # cheap preview diff: old block vs new block
    preview = f"--- lines {s}-{e} ({e - s + 1} lines) -> {len(new_lines)} new line(s)\n- " + "\n- ".join(old_block.split("\n")[:10]) + "\n+ " + "\n+ ".join(new_lines[:10])
    if dry_run:
        return {"output": "Dry run — no changes written.\n" + preview, "metadata": {"dry_run": True, "offset": s, "limit": e - s + 1}}
    try:
        _aw(path, new_content)
    except (OSError, UnicodeError) as ex:
        return {"output": f"Error writing file {path}: {ex}", "error": True}
    return {"output": f"Replaced lines {s}-{e} ({e - s + 1} lines).\n" + preview, "metadata": {"offset": s, "limit": e - s + 1}}


def _edit(filePath: str, oldString: str, newString: str, replaceAll: bool = False, dry_run: bool = False, offset: int = 0, limit: int = 0) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        return {"output": f"File does not exist: {path}", "error": True}
    if offset and limit and not (oldString or "").strip():
        return _edit_lines(path, int(offset), int(limit), newString, dry_run)
    result = _do_edit(path, oldString, newString, replaceAll, dry_run=dry_run)
    if result.get("error") is not True:
        # feed the verify tool's homework checker
        from .verify import track

        track(path, "edit")
    return result


def tool() -> Tool:
    description = """Exact string replacement in a file. oldString must match verbatim (no line-number prefixes); miss returns 3 closest lines, multi-match returns line numbers. dry_run previews. Have grep offset/limit? Use offset+limit+newString (no verbatim copy)."""

    def run(input: dict) -> dict:
        try:
            off = int(input.get("offset") or 0)
        except (TypeError, ValueError):
            off = 0
        try:
            lim = int(input.get("limit") or 0)
        except (TypeError, ValueError):
            lim = 0
        return _edit(
            input["filePath"],
            input.get("oldString") or "",
            input.get("newString") or "",
            replaceAll=bool(input.get("replaceAll", False)),
            dry_run=bool(input.get("dry_run", False)),
            offset=off,
            limit=lim,
        )

    return Tool(
        name="edit",
        description=description,
        parameters=schema_with(
            {
                "filePath": {"type": "string", "description": "Absolute file path"},
                "oldString": {"type": "string", "description": "Text to replace, verbatim (or omit with offset+limit)", "optional": True},
                "newString": {"type": "string", "description": "Replacement text"},
                "replaceAll": {"type": "boolean", "description": "Replace every match", "optional": True},
                "dry_run": {"type": "boolean", "description": "Preview diff without writing", "optional": True},
                "offset": {"type": "integer", "description": "Start line from grep anchor (use with limit, no oldString needed)", "optional": True},
                "limit": {"type": "integer", "description": "Lines to replace (use with offset)", "optional": True},
            },
            ["filePath", "newString"],
        ),
        run=run,
        permission="edit",
    )
