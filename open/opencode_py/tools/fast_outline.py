"""Fast Python outline scan: regex header pass + (mtime,size) cache.

Sticky notes for thick books. First read of a file scans `^class|def`
headers once (~0.04s on 244KB vs ~0.5s ast.parse); repeats hit the cache
(~0.0001s). Output is byte-identical to the AST walk in tools/read.py.

Only class/def frames live on the scope stack — if/try/with/for lines
never push, so a method after a nested try-block still sees its class
frame. Triple-quote strings, `#` comments, NUL bytes or mixed tabs/spaces
mean uncertain (None) and the caller falls back to ast.parse unchanged.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

_CACHE: dict[str, tuple[int, int, list[tuple[int, str]]]] = {}
_LOCK = threading.Lock()
_MAX = 32

_HDR_RE = re.compile(
    r"^(?P<ind>[ \t]*)(?P<kw>class|def|async\s+def)\b\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)

_TQ = chr(34) * 3
_TS = chr(39) * 3


def scan_headers(lines: list[str]) -> tuple[list[tuple[int, str]], bool] | None:
    """One pass over lines. None = uncertain, use ast.parse."""
    out: list[tuple[int, str]] = []
    stack: list[tuple[int, str]] = []
    in_str: str | None = None
    seen_tab = False
    seen_space = False
    for i, raw in enumerate(lines, 1):
        if chr(0) in raw:
            return None
        if in_str is not None:
            if in_str in raw and raw.count(in_str) % 2 == 1:
                in_str = None
            continue
        s = raw.lstrip()
        if not s or s.startswith("#"):
            continue
        m = _HDR_RE.match(raw)
        if m is not None:
            ind_raw = m.group("ind")
            if "\t" in ind_raw:
                seen_tab = True
            if " " in ind_raw:
                seen_space = True
            try:
                ind = len(ind_raw.expandtabs(8))
            except Exception:
                return None
            kind = "def" if "def" in m.group("kw") else "class"
            while stack and stack[-1][0] >= ind:
                stack.pop()
            if not stack:
                out.append((i, m.group("name")))
            elif len(stack) == 1 and stack[0][1] == "class":
                out.append((i, "    " + m.group("name")))
            stack.append((ind, kind))
            code = raw.split("#", 1)[0]
            for q in (_TQ, _TS):
                if q in code and code.count(q) % 2 == 1:
                    in_str = q
                    break
            continue
        code = raw.split("#", 1)[0]
        for q in (_TQ, _TS):
            if q in code and code.count(q) % 2 == 1:
                in_str = q
                break
    if in_str is not None:
        return None
    if seen_tab and seen_space:
        return None
    return out, False


def cached_entries(path: Path, lines: list[str], st=None) -> list[tuple[int, str]] | None:
    """(lineno, indented-name) for .py outline. None = use AST."""
    try:
        if st is None:
            st = path.stat()
        key = (int(getattr(st, "st_mtime_ns", 0)), int(getattr(st, "st_size", 0)))
    except OSError:
        return None
    pkey = str(path)
    try:
        with _LOCK:
            hit = _CACHE.get(pkey)
            if hit is not None and hit[0] == key[0] and hit[1] == key[1]:
                try:
                    _CACHE[pkey] = _CACHE.pop(pkey)
                except KeyError:
                    pass
                return list(hit[2])
    except Exception:
        pass
    try:
        scanned = scan_headers(lines)
    except Exception:
        return None
    if scanned is None:
        return None
    entries, _uncertain = scanned
    try:
        with _LOCK:
            _CACHE[pkey] = (key[0], key[1], list(entries))
            while len(_CACHE) > _MAX:
                _CACHE.pop(next(iter(_CACHE)))
    except Exception:
        pass
    return entries


def cache_size() -> int:
    try:
        return len(_CACHE)
    except Exception:
        return 0


def cache_clear() -> None:
    try:
        _CACHE.clear()
    except Exception:
        pass
