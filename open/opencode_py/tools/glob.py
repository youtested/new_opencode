"""glob tool: find files by name pattern (max 100 results).

RULES (full guidance for maintainers; the sent schema is short):
- Patterns like "**/*.js", "src/**/*.ts" (braces work).
- Open-ended multi-round searches go through the task tool instead.
- Batch independent searches in one message.
"""
from __future__ import annotations

import fnmatch
import os
import threading
from pathlib import Path

from .registry import Tool, schema_with

MAX_RESULTS = 100

# Thread-local session worktree installed by the agent loop before every tool
# call, so `glob` with no `path` resolves against the SESSION directory (git
# worktree root), not Process CWD() — launching from a subdirectory would
# otherwise search the wrong tree. Each worker thread keeps its own.
_worktree = threading.local()


def set_worktree(path) -> None:
    """Set/clear the session worktree glob should fall back to."""
    if path is None:
        try:
            del _worktree.path
        except AttributeError:
            pass
        return
    _worktree.path = str(path)


def _default_base() -> Path:
    raw = getattr(_worktree, "path", None)
    if raw:
        return Path(raw).resolve()
    return Path.cwd()


def _expand_braces(pattern: str) -> list[str]:
    """Expand `{a,b,c}` into a list of patterns (recursively). fnmatch/pathlib
    have no brace support, so `*.{ts,tsx}` would otherwise match nothing."""
    if "{" not in pattern:
        return [pattern]
    open_idx = pattern.find("{")
    close_idx = pattern.find("}", open_idx + 1)
    if close_idx == -1:
        return [pattern]
    head = pattern[:open_idx]
    tail = pattern[close_idx + 1 :]
    out = []
    for alt in pattern[open_idx + 1 : close_idx].split(","):
        out.extend(_expand_braces(head + alt + tail))
    # de-dupe: "*.{js,js}" collapses to the same pattern twice
    return list(dict.fromkeys(out))


def _normalize_pattern(pattern: str) -> str:
    """Reject parent-traversal and pin absolute/trailing patterns to `base`.

    `Pattern.glob` happily followed `../*` up out of the search root, leaking
    results from outside the requested tree; and a leading `/` is an invalid
    (or cwd-anchored) absolute glob, not "under base". Stripping the leading
    `/` keeps `/**/*.py` meaning "everything under base" like opencode.
    """
    if not pattern:
        return "*"
    normalized = pattern.lstrip("/").replace("\\", "/")
    for comp in normalized.split("/"):
        if comp == "..":
            raise ValueError("glob pattern may not traverse outside the search path (no '..')")
    return normalized


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _walk(base: Path, pattern: str, cap: int, ignore=None) -> tuple[list[Path], bool]:
    """Backtracking glob over `base` matching pathlib/ripgrep semantics.

    - `**` recurses any number of directories but NEVER descends into hidden
      ones (Python 3.1x `Path.glob("**/...")` descends hidden dirs — diverging
      from rg and flooding results from .git/.venv), and never follows
      symlinked directories (loop-safe).
    - `*`/`?`/`[...]` components are fnmatch'd per directory name, so `*` does
      not match leading dots, matching rg.
    - permission errors on one subtree skip it instead of aborting the whole
      search.
    Returns (matched paths, truncated-with-cap flag).
    """
    if pattern.endswith("/"):
        pattern = pattern[:-1]
    parts = [p for p in pattern.split("/") if p and p != "."]
    results: list[Path] = []
    truncated = False
    chain: set[str] = set()  # resolved dirs in the current recursion chain

    def add(m: Path) -> None:
        nonlocal truncated
        if len(results) >= cap:
            truncated = True
            return
        results.append(m)

    def rec(i: int, cur: Path) -> None:
        if truncated:
            return
        if i == len(parts):
            add(cur)
            return
        part = parts[i]
        # prune ignored paths wholesale (project .gitignore): skipping a dir
        # here skips its entire subtree, like rg
        if ignore is not None:
            try:
                if ignore.match(cur.resolve(), is_dir=True):
                    return
            except OSError:
                pass
        if part == "**":
            # zero directories, then one-or-more (skipping hidden dirs)
            rec(i + 1, cur)
            if i == len(parts) - 1:
                # trailing `**`: matches everything under cur (already added
                # cur at i+1), and any deeper entry, hidden dirs still skipped.
                # Cycle-guarded like the non-trailing branch: symlink loops
                # otherwise recurse forever (hang/OOM).
                try:
                    rcur = cur.resolve()
                except OSError:
                    return
                if str(rcur) in chain:
                    return
                chain.add(str(rcur))
                try:
                    for child in sorted(_scandir(cur), key=lambda c: c[0]):
                        name, entry = child
                        if truncated:
                            break
                        if _is_hidden(name):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            rec(i, cur / name)
                        elif entry.is_file(follow_symlinks=False):
                            rec(i + 1, cur / name)
                        else:
                            try:
                                if entry.is_dir():
                                    rec(i, cur / name)
                            except OSError:
                                pass
                finally:
                    chain.discard(str(rcur))
                return
            try:
                rcur = cur.resolve()
            except OSError:
                return
            if str(rcur) in chain:
                return
            chain.add(str(rcur))
            for name, entry in sorted(_scandir(cur), key=lambda c: c[0]):
                if _is_hidden(name):
                    continue
                if entry.is_dir():
                    rec(i, cur / name)
            chain.discard(str(rcur))
        else:
            # `*`/`?`/`[...]` are fnmatch'd per directory name. Python's
            # fnmatch (`Path.glob` too) lets `*` match leading dots, diverging
            # from rg/globset — enforce the dotfile rule here so `*` never
            # matches `.git`/`.env` unless the pattern itself starts with a dot.
            for name, entry in sorted(_scandir(cur), key=lambda c: c[0]):
                if name.startswith(".") and not part.startswith("."):
                    continue
                if fnmatch.fnmatch(name, part):
                    rec(i + 1, cur / name)

    rec(0, base)
    if truncated:
        results = results[:cap]
    return results, truncated


def _scandir(path: Path):
    """Yield deterministic (name, DirEntry) pairs; missing dirs / permission
    errors yield nothing rather than aborting the outer walk."""
    try:
        with os.scandir(path) as it:
            for entry in sorted(it, key=lambda e: e.name):
                yield entry.name, entry
    except OSError:
        return


def _glob(pattern: str, path: str | None = None) -> dict:
    base = Path(path).resolve() if path else _default_base()
    if not base.is_dir():
        return {"output": f"Path is not a directory: {base}", "error": True}

    results: list[Path] = []
    truncated = False
    try:
        normalized = _normalize_pattern(pattern)
    except ValueError as e:
        return {"output": f"Glob error: {e}", "error": True}

    from ..util.gitignore import load as _load_gitignore

    ignore = _load_gitignore(base)
    try:
        for pat in _expand_braces(normalized):
            batch, batched_truncated = _walk(base, pat, MAX_RESULTS, ignore=ignore)
            results.extend(batch)
            truncated = truncated or batched_truncated
            # de-dupe across brace branches (e.g. "*.{js,js}"): collapse paths
            # that resolve to the SAME FILE. The key is the file identity
            # (device+inode) rather than a per-entry `resolve()` call — an
            # os.stat is a single syscall while resolve() walks every path
            # component, so this preserves exact symlink-alias dedup without
            # punishing large trees. Falls back to the string path if the stat
            # fails (e.g. a raced deletion).
            seen: set = set()
            deduped = []
            for m in results:
                try:
                    st = m.stat()
                    key = (st.st_dev, st.st_ino)
                except OSError:
                    key = str(m)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(m)
            results = deduped
            if len(results) >= MAX_RESULTS:
                truncated = True
                results = results[:MAX_RESULTS]
                break
    except (OSError, ValueError, RecursionError) as e:
        return {"output": f"Glob error: {e}", "error": True}

    truncated = truncated or len(results) >= MAX_RESULTS
    if not results:
        return {"output": "No files found"}
    out = "\n".join(str(p) for p in results)
    if truncated:
        out += "\n\n(Results are truncated: showing first 100 results. Consider using a more specific path or pattern.)"
    return {"output": out, "metadata": {"count": len(results), "truncated": truncated}}


def tool() -> Tool:
    description = """Find files by name pattern ("**/*.js"). Multi-round searches go through task."""

    def run(input: dict) -> dict:
        return _glob(input["pattern"], input.get("path"))

    return Tool(
        name="glob",
        description=description,
        parameters=schema_with(
            {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "path": {"type": "string", "description": "Directory", "optional": True},
            },
            ["pattern"],
        ),
        run=run,
        permission="glob",
    )
