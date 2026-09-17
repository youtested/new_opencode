"""Minimal .gitignore matcher for the pure-python search fallbacks.

Without `rg` on PATH, grep/glob scan the tree with pathlib - and used to walk
straight into venv/, node_modules/, dist/ ... because they ignored .gitignore.
This module gives both fallbacks practical gitignore semantics:

- blank lines / '#' comments
- trailing '/'  -> directory-only rule
- leading '/' or an embedded '/' -> anchored to the ignore file's directory
- otherwise -> matches any path component at any depth below that directory
- '!pattern' negation (LAST matching rule wins)
- '**', '*', '?' globs

Known simplification (documented, acceptable for search pruning): traversal
prunes an ignored DIRECTORY wholesale, so a '!keep' re-inclusion inside an
ignored dir is not honoured - same as rg's default behaviour in practice.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Optional


def _translate(pat: str) -> str:
    """git-glob -> regex fragment ('*' stays within a path segment)."""
    i, n = 0, len(pat)
    out: list[str] = []
    while i < n:
        c = pat[i]
        if c == "*":
            if pat[i : i + 2] == "**":
                if pat[i + 2 : i + 3] == "/":
                    out.append("(?:.*/)?")
                    i += 3
                else:
                    out.append(".*")
                    i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


class _Rule:
    __slots__ = ("regex", "negate", "dir_only")

    def __init__(self, raw: str):
        self.negate = False
        self.dir_only = False
        line = raw.rstrip()
        if line.startswith("!"):
            self.negate = True
            line = line[1:]
        if line.endswith("/"):
            self.dir_only = True
            line = line[:-1]
        anchored = line.startswith("/") or "/" in line
        line = line.lstrip("/")
        body = _translate(line)
        if anchored:
            core = "^" + body
            if self.dir_only:
                # a directory candidate carries no trailing slash; files under
                # it are pruned via ancestor matching
                core += "(?:$|/)"
            else:
                core += "(?:$|/.*)"
        else:
            # component match anywhere below base; children of an ignored dir
            # inherit the ignore
            core = "(?:^|/)" + body + "(?:$|/)"
        self.regex = re.compile(core)

    def hit(self, rel_posix: str) -> bool:
        return self.regex.search(rel_posix) is not None


class GitIgnore:
    def __init__(self, rules: list[tuple[Path, _Rule]]):
        # [(base_dir, rule)] in evaluation order: shallow first, deeper later,
        # so a deeper .gitignore overrides a shallower one (last match wins).
        self._rules = rules

    def match(self, path: Path, is_dir: bool = False) -> bool:
        """True when this absolute path should be pruned.

        A FILE also inherits the ignore status of its ancestor directories:
        a dir-only rule like ``venv/`` must prune ``venv/lib/junk.py`` even
        though the rule never names the file itself.
        """
        try:
            full = path.resolve()
        except OSError:
            full = path
        result = False
        for base, r in self._rules:
            try:
                rel = str(full.relative_to(base))
            except ValueError:
                continue  # rule's scope doesn't cover this path
            rel_posix = rel if rel.startswith("/") else "/" + rel
            hit = False
            if (is_dir or not r.dir_only) and r.hit(rel_posix):
                hit = True
            elif r.dir_only and not is_dir:
                parts = Path(rel).parts
                for i in range(1, len(parts)):
                    if r.hit("/" + "/".join(parts[:i])):
                        hit = True
                        break
            if hit:
                result = not r.negate
        return result


def load(base: Path) -> Optional[GitIgnore]:
    """Collect .gitignore rules from base up to the filesystem root.

    Shallow files are evaluated first and deeper ones last, so per-directory
    overrides win. Returns None when no .gitignore exists at all - callers
    keep their old fast path.

    Results are cached per base directory (keyed by every .gitignore's
    mtime+size): glob/grep used to re-walk to the filesystem root and
    recompile every regex on EVERY call.
    """
    return _cached_load(base)


_CACHE_LOCK = threading.Lock()
# base_dir -> (fingerprint, GitIgnore|None, cached_at)
_CACHE: dict[str, tuple[tuple, Optional[GitIgnore], float]] = {}
_CACHE_TTL = 5.0


def _cached_load(base: Path) -> Optional[GitIgnore]:
    try:
        start = base.resolve() if base.exists() else base
    except OSError:
        start = base
    key = str(start)
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[2] < _CACHE_TTL:
            return hit[1]
    files: list[Path] = []
    d = start
    while True:
        gi = d / ".gitignore"
        try:
            if gi.is_file():
                files.append(gi)
        except OSError:
            pass
        parent = d.parent
        if parent == d:
            break
        d = parent
    fingerprint: list[tuple] = []
    for gi in files:
        try:
            st = gi.stat()
            fingerprint.append((str(gi), st.st_mtime_ns, st.st_size))
        except OSError:
            fingerprint.append((str(gi), -1, -1))
    fingerprint_t = tuple(fingerprint)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == fingerprint_t:
            _CACHE[key] = (fingerprint_t, hit[1], now)
            return hit[1]
    result = _load_uncached(files)
    with _CACHE_LOCK:
        _CACHE[key] = (fingerprint_t, result, now)
    return result


def _load_uncached(files: list[Path]) -> Optional[GitIgnore]:
    rules: list[tuple[Path, _Rule]] = []
    for gi in reversed(files):  # shallowest first ... deepest last
        b = gi.parent
        try:
            text = gi.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rules.append((b, _Rule(line)))
            except re.error:
                continue
    return GitIgnore(rules) if rules else None
