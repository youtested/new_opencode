"""grep tool: search file contents with regex (ripgrep, else fallback).

RULES (full guidance for maintainers; the sent schema is short):
- Full regex; include filters files ("*.js", ["*.py", "*.md"]). Returns paths
  + line numbers WITH code (context=N surroundings) — location + body together,
  so the caller rarely needs a follow-up read.
- COUNT-FIRST (saves data): mode="count" returns only per-file counts
  (~0.4KB) so the caller picks ONE file, then searches that file with
  context=0 (~0.3KB) for the anchor, then ONE read symbol=/offset=. This is
  the AGENT.md anchor-then-ONE-read loop: 1 tiny grep + 1 read, not N big greps.
- Counting matches -> bash with rg directly, not this tool.
- Open-ended multi-round searches go through the task tool instead.
"""
from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

from .read import MAX_OUTPUT, _is_binary_sample, _is_low
from .registry import Tool, schema_with

MAX_RESULTS = 100

# Low-data preset (Settings > save data): active ONLY when cfg.low_data is
# true; defaults above are used otherwise, byte-identical.
LOW_MAX_RESULTS = 30
LOW_MAX_LINE = 400
LOW_MAX_OUTPUT = 10 * 1024  # 10 KB


def _expand_braces(pattern: str) -> list[str]:
    """Expand `{a,b,c}` in a fnmatch-style pattern into a list of patterns."""
    if "{" not in pattern:
        return [pattern]
    out: list[str] = []
    open_idx = pattern.find("{")
    close_idx = pattern.find("}", open_idx + 1)
    if close_idx == -1:
        return [pattern]
    for alt in pattern[open_idx + 1 : close_idx].split(","):
        expanded = pattern[:open_idx] + alt + pattern[close_idx + 1 :]
        out.append(expanded)
    return out


def _fnmatch_any(name: str, pattern: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in _expand_braces(pattern))


def _glob_matches(p: Path, base: Path, pattern: str) -> bool:
    """Match an include glob the way rg does.

    rg (globset) matches a glob CONTAINING a path separator against the path
    relative to the search root, and a separator-less glob against the file
    basename. The old code only tested basenames, so ``include="src/*.ts"``
    matched nothing. ``**`` is its own path segment in globset; fnmatch treats
    it as two stars (still permissive), so fall back to the separator rule.
    """
    for pat in _expand_braces(pattern):
        if "/" in pat:
            root = base if base.is_dir() else base.parent
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = None
            if rel is not None and fnmatch.fnmatch(str(rel), pat):
                return True
        elif fnmatch.fnmatch(p.name, pat):
            return True
    return False


def _as_include_list(include) -> list[str] | None:
    if include is None:
        return None
    if isinstance(include, str):
        s = include.strip()
        return [s] if s else None
    if isinstance(include, (list, tuple)):
        out = [str(x).strip() for x in include if str(x or "").strip()]
        return out or None
    s = str(include).strip()
    return [s] if s else None


def _matches_include(p: Path, base: Path, include) -> bool:
    incs = _as_include_list(include)
    if not incs:
        return True
    return any(_glob_matches(p, base, pat) for pat in incs)


def _include_flags(include) -> list:
    incs = _as_include_list(include)
    return [x for pat in (incs or []) for x in ("-g", pat)]


def _parse_rg_line(line: str) -> tuple[str, int, str] | None:
    """Parse an `rg --no-heading --line-number` line: path:lineno:content.

    File names may themselves contain colons, so we scan left-to-right for the
    first `:<digits>:` marker (the line number) and treat everything before it
    as the path. rg always emits `path:line:content`, so the first ALL-DIGITS
    colon segment after the last path separator is the line number.
    """
    path = ""
    for i, char in enumerate(line):
        if char == ":":
            rest = line[i + 1 :]
            idx = rest.find(":")
            if idx <= 0:
                continue
            try:
                lineno = int(rest[:idx])
            except ValueError:
                continue
            return line[:i], lineno, rest[idx + 1 :]
    return None


def _grep_rg(pattern: str, base: Path, include=None, *, max_results: int = MAX_RESULTS) -> list[tuple[str, int, str]] | None:
    # "--" guards the pattern/glob: a search for "-foo" or "--bar" must not be
    # consumed as an rg flag (rg would otherwise read stdin / error out).
    #
    # rg's cwd-set invocation: ripgrep anchors slash-containing -g globs (and
    # its matched paths) to the *current working directory*, not the search
    # root. When the caller passes an absolute path outside the agent's cwd, a
    # glob like "src/*.ts" silently matched nothing. Running rg with cwd=base
    # makes the globs behave exactly like the pure-python fallback (anchored
    # relative to the search root), and results are normalized back to
    # absolute paths so both paths agree.
    cwd = base if base.is_dir() else base.parent
    search_arg = "." if base.is_dir() else base.name
    # --with-filename: rg omits the path when exactly one file is searched,
    # but _parse_rg_line (and the caller's grouping) require a path prefix.
    cmd = ["rg", "--no-heading", "--with-filename", "--line-number", "--color", "never"]
    flags = _include_flags(include)
    if flags:
        cmd += flags
    else:
        for _g in _rg_excludes(base):
            cmd += ["-g", _g]
    cmd += ["--", pattern, search_arg]
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode not in (0, 1):
        return None
    base_resolved = base.resolve()
    results = []
    for line in proc.stdout.splitlines():
        parsed = _parse_rg_line(line)
        if parsed is None:
            continue
        filepath, lineno, text = parsed
        pp = Path(filepath)
        if not pp.is_absolute():
            pp = (cwd / pp).resolve()
        results.append((str(pp), lineno, text))
        if len(results) >= max_results:
            break
    return results


def _grep_py(pattern: str, base: Path, include=None, *, max_results: int = MAX_RESULTS) -> list[tuple[str, int, str]]:
    from ..util.gitignore import load as _load_gitignore

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return [("", 0, f"invalid regex: {e}")]
    results = []
    # mirror rg: respect the project's .gitignore so venv/node_modules/dist
    # don't turn a search into a minutes-long scan on armv7
    ignore = _load_gitignore(base if base.is_dir() else base.parent)
    base_parts_len = len(base.resolve().parts)
    if base.is_dir():
        iterator = base.rglob("*")
    else:
        iterator = iter([base])
    import time as _time

    deadline = _time.monotonic() + 15.0
    files_scanned = 0
    timed_out = False

    def _is_hidden(p: Path) -> bool:
        return any(seg.startswith(".") for seg in p.parts[base_parts_len:])

    try:
        for p in iterator:
            if _time.monotonic() > deadline or files_scanned >= 5000:
                timed_out = True
                break
            if p.is_dir():
                continue
            if ignore is not None and ignore.match(p.resolve()):
                continue
            if _walk_ignored(base, p):
                continue
            if not _matches_include(p, base, include):
                continue
            # skip hidden paths (mirrors rg's default) but not the search base itself
            if _is_hidden(p):
                continue
            # mirror rg: don't match inside binary files (errors=replace would
            # otherwise surface garbage "matches" of random high bytes)
            try:
                with p.open("rb") as bf:
                    head = bf.read(1024)
            except OSError:
                continue
            if _is_binary_sample(head):
                continue
            files_scanned += 1
            try:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if _time.monotonic() > deadline:
                            timed_out = True
                            break
                        if regex.search(line.rstrip("\n")):
                            results.append((str(p), i + 1, line.rstrip("\n")))
                            if len(results) >= max_results:
                                return results
                    if timed_out:
                        break
            except OSError:
                continue
        if timed_out:
            results.append(("", 0, f"search stopped: 15s / 5000-file budget hit after {files_scanned} files"))
        return results
    except (OSError, RecursionError, ValueError):
        return results


def _grep(pattern: str, path=None, include=None, cfg=None) -> dict:
    base = Path(path).resolve() if path else Path.cwd()
    if not base.exists():
        return {"output": f"Path does not exist: {base}", "error": True}

    low = _is_low(cfg)
    cap = LOW_MAX_RESULTS if low else MAX_RESULTS
    if shutil.which("rg"):
        results = _grep_rg(pattern, base, include, max_results=cap)
    else:
        results = None
    if results is None:
        results = _grep_py(pattern, base, include, max_results=cap)

    if results and results[0][0] == "" and results[0][1] == 0:
        return {"output": results[0][2], "error": True}

    truncated = len(results) >= cap
    if not results:
        return {"output": "No files found"}

    # group per file
    grouped: dict[str, list[tuple[int, str]]] = {}
    for filepath, lineno, text in results:
        grouped.setdefault(filepath, []).append((lineno, text))

    # These match lines just reached the model — record them in the context
    # ledger so a later read of the same regions doesn't send them twice.
    from pathlib import Path as _P

    from .context_ledger import mark_delivered

    for filepath, hits in grouped.items():
        try:
            gst = _P(filepath).stat()
        except OSError:
            continue
        runs: list[list[int]] = []
        for lineno, _text in sorted(hits):
            if runs and lineno == runs[-1][1] + 1:
                runs[-1][1] = lineno
            else:
                runs.append([lineno, lineno])
        for s, e in runs:
            mark_delivered(filepath, gst.st_mtime_ns, gst.st_size, s, e)

    total = len(results)
    header = f"Found {total} match" + ("es" if total != 1 else "")
    if truncated:
        header += " (more matches available)"
    # best match first (same tiers as context mode): files ordered by their
    # best hit, hits within a file by tier then line — defs before mentions.
    try:
        term = _plain_term(pattern)
    except Exception:
        term = ""
    def _rank_hit(_l, _t):
        try:
            tier, name = _hit_kind(_t, term)
        except Exception:
            return (9, 1, 99, _l)
        # exact-name hit (def `session_path` for term `session`) beats a
        # longer prefix sibling (`session_cache`) in the same tier
        exact = 0 if (term and name == term) else 1
        prefix = 0 if (term and name.startswith(term)) else 1
        return (tier, exact, prefix, len(name or _t), _l)

    def _file_key(item) -> tuple:
        _fp, _hits = item
        try:
            best = min(_rank_hit(_l, _t) for _l, _t in _hits)
        except Exception:
            best = (9, 1, 1, 99, 0)
        return (best, _fp)
    lines = [header]
    try:
        ordered_files = sorted(grouped.items(), key=_file_key)
    except Exception:
        ordered_files = list(grouped.items())
    for filepath, hits in ordered_files:
        lines.append(f"{filepath}:")
        try:
            ranked = sorted(hits, key=lambda h: (_rank_hit(h[0], h[1])))
        except Exception:
            ranked = hits
        for lineno, text in ranked:
            if low and len(text) > LOW_MAX_LINE:
                text = text[:LOW_MAX_LINE] + "..."
            lines.append(f"  Line {lineno}: {text}")
    if truncated:
        lines.append("(Results truncated. Consider using a more specific path or pattern.)")
    out = "\n".join(lines)
    if low and len(out) > LOW_MAX_OUTPUT:
        out = out[:LOW_MAX_OUTPUT] + "\n… (save-data: output capped at 10 KB — refine pattern/path.)"
        truncated = True
    return {"output": out, "metadata": {"matches": total, "truncated": truncated}}


# Directories never walked by default (build mirrors, caches, vendored envs):
# they duplicate every match and stall armv7 scans. Searching INSIDE one
# explicitly (path points there) still works — the filter only skips them
# during a wider walk.
_DEFAULT_IGNORE_DIRS = frozenset({"build", "dist", "node_modules", ".venv", "venv", "__pycache__"})

_RG_EXCLUDES = ["!build/**", "!dist/**", "!node_modules/**", "!.venv/**", "!venv/**", "!__pycache__/**", "!*.egg-info/**"]

# `def name` / `class name` hit rows get a read symbol= link (the search→read loop).
_DEF_HIT_RE = re.compile(r"^\s*(?:async\s+def\s+|def\s+|class\s+)(\w+)")

# -- best-match ranking --------------------------------------------------
# Groups sort by score, not walk order: def headers first (exact name beats
# prefix beats substring), then real code uses, then comments/docstrings/strings
# last — for ANY pattern, not just defs. Ties break by shorter name, fewer
# matches in the group, then path+line (stable run to run).
_COMMENT_RES = (
    re.compile(r"^\s*#"),
    re.compile(r"^\s*//"),
    re.compile(r"^\s*\*"),
    re.compile(r"^\s*/\*"),
    re.compile(r"^\s*<!--"),
    re.compile(r"^\s*['\"]{3}"),
)


def _plain_term(pattern: str) -> str:
    """A regex that is really just an identifier -> the identifier, else ''."""
    s = (pattern or "").strip()
    if s and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        return s
    return ""


def _is_comment_line(body: str) -> bool:
    s = body.strip()
    if not s:
        return True
    for rx in _COMMENT_RES:
        if rx.match(body):
            return True
    if (s.startswith('"') and s.rstrip('",').endswith('"')) or \
       (s.startswith("'") and s.rstrip("',").endswith("'")):
        return len(s) < 160  # short quoted line ≈ string mention, not code
    return False


def _hit_kind(body: str, term: str) -> tuple[int, str]:
    """(tier, matched_name): 0 def-exact, 1 def-prefix, 2 def-contains,
    3 code-exact, 4 code-substring, 5 comment/string. Never raises."""
    try:
        m = _DEF_HIT_RE.match(body)
        if m:
            name = m.group(1)
            if term and name == term:
                return 0, name
            if term and name.startswith(term):
                return 1, name
            return 2, name
        if term:
            for mm in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", body):
                if mm.group(0) == term:
                    if _is_comment_line(body):
                        return 5, term
                    return 3, term
            if term in body:
                return (5, term) if _is_comment_line(body) else (4, term)
        return (5, "") if _is_comment_line(body) else (4, "")
    except Exception:
        return 4, ""


def _group_score(items: list, term: str) -> tuple:
    """Sort key for one rendered group: best hit tier, exactness, prefix,
    name size, group size, span start. Stable run to run."""
    try:
        best = None
        for ln, body, is_hit in items:
            if not is_hit:
                continue
            tier, name = _hit_kind(body, term)
            exact = 0 if (term and name == term) else 1
            prefix = 0 if (term and name.startswith(term)) else 1
            key = (tier, exact, prefix, len(name or body))
            if best is None or key < best:
                best = key
        if best is None:
            return (9, 1, 1, 99, 0, 0, "")
        hits = sum(1 for _ln, _b, h in items if h)
        first = min(ln for ln, _b, h in items if h)
        return (best[0], best[1], best[2], best[3], hits, first)
    except Exception:
        return (9, 1, 1, 99, 0, 0, "")


def _walk_ignored(base: Path, p: Path) -> bool:
    """True when p sits under a default-ignored dir relative to the walk root."""
    root = base if base.is_dir() else base.parent
    try:
        rel = p.relative_to(root)
    except ValueError:
        try:
            rel = p.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            return False
    for seg in rel.parts[:-1]:
        if seg in _DEFAULT_IGNORE_DIRS or seg.endswith(".egg-info"):
            return True
    return False


def _rg_excludes(base: Path) -> list:
    """-g excludes for rg, dropped when the root is inside one (explicit search)."""
    try:
        segs = base.resolve().parts
    except OSError:
        return []
    if any(s in _DEFAULT_IGNORE_DIRS or s.endswith(".egg-info") for s in segs):
        return []
    return list(_RG_EXCLUDES)


def _grep_rg_json(pattern: str, base: Path, include: str | None, context: int):
    """rg --json -C: (hits, order, total), or None when rg is missing/fails.

    hits maps file -> [(lineno, body, is_hit)] in file order. Invalid regex
    (rg exit 2) returns None so the Python fallback reports the clean error.
    """
    import json as _json

    cwd = base if base.is_dir() else base.parent
    search_arg = "." if base.is_dir() else base.name
    cmd = ["rg", "--json", "-C", str(max(0, int(context)))]
    flags = _include_flags(include)
    if flags:
        cmd += flags
    else:
        for _g in _rg_excludes(base):
            cmd += ["-g", _g]
    cmd += ["--", pattern, search_arg]
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode not in (0, 1):
        return None
    # per-file resolve cache: 11k hits share ~100 files — resolving once
    # per file (not per hit) is what keeps floods fast on Termux.
    resolved: dict = {}
    skipped: set = {}

    def _abs(raw: str):
        hit = resolved.get(raw)
        if hit is not None or raw in skipped:
            return hit
        pp = Path(raw)
        if not pp.is_absolute():
            pp = (cwd / pp).resolve()
        if _walk_ignored(base, pp):
            skipped.add(raw)
            return None
        key = str(pp)
        resolved[raw] = key
        return key

    hits: dict = {}
    order: list = []
    total = 0
    for line in proc.stdout.splitlines():
        try:
            obj = _json.loads(line)
        except ValueError:
            continue
        kind = obj.get("type")
        if kind not in ("match", "context"):
            continue
        data = obj.get("data") or {}
        key = _abs((data.get("path") or {}).get("text") or "")
        if key is None:
            continue
        try:
            lineno = int(data.get("line_number") or 0)
        except (TypeError, ValueError):
            continue
        if lineno <= 0:
            continue
        body = ((data.get("lines") or {}).get("text") or "").rstrip("\n").rstrip("\r")
        is_hit = kind == "match"
        if key not in hits:
            hits[key] = []
            order.append(key)
        hits[key].append((lineno, body, is_hit))
        if is_hit:
            total += 1
    return hits, order, total


def _collect_py_context(pattern: str, base: Path, include: str | None, context: int):
    """Pure-Python context collector (rg missing/failed): same shape as rg-json.

    Returns an error dict for invalid regex instead of raising.
    """
    from ..util.gitignore import load as _load_gitignore

    try:
        rx = re.compile(pattern)
    except re.error as e:
        return {"output": f"invalid regex: {e}", "error": True}
    ignore = _load_gitignore(base if base.is_dir() else base.parent)
    try:
        base_parts_len = len(base.resolve().parts)
    except OSError:
        return {"output": f"Path does not exist: {base}", "error": True}
    hits: dict = {}
    order: list = []
    total = 0
    pad = max(0, int(context))
    iterator = base.rglob("*") if base.is_dir() else iter([base])

    def _hidden(p: Path) -> bool:
        try:
            return any(seg.startswith(".") for seg in p.parts[base_parts_len:])
        except Exception:
            return False

    for f in iterator:
        try:
            if f.is_dir() or _hidden(f):
                continue
            if ignore is not None and ignore.match(f.resolve()):
                continue
            if _walk_ignored(base, f):
                continue
            if not _matches_include(f, base, include):
                continue
            try:
                with f.open("rb") as bf:
                    head = bf.read(1024)
            except OSError:
                continue
            if _is_binary_sample(head):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.split("\n")
            found = [i for i, ln in enumerate(lines, 1) if rx.search(ln)]
            if not found:
                continue
            key = str(f)
            if key not in hits:
                hits[key] = []
                order.append(key)
            hit_set = set(found)
            spans: list = []
            for h in found:
                s, e = max(1, h - pad), min(len(lines), h + pad)
                if spans and s <= spans[-1][1] + 1:
                    spans[-1][1] = max(spans[-1][1], e)
                else:
                    spans.append([s, e])
            for s, e in spans:
                for ln in range(s, e + 1):
                    try:
                        body = lines[ln - 1]
                    except IndexError:
                        body = ""
                    hits[key].append((ln, body, ln in hit_set))
            total += len(found)
        except (OSError, RecursionError, ValueError):
            continue
    return hits, order, total


_MAX_GROUP_LINES = 60


def _count_only(pattern: str, path=None, include=None, cfg=None) -> dict:
    """Tiny count-first probe (~0.4KB): per-file counts + total, no code lines.
    Feeds the AGENT.md anchor loop: pick ONE file, then grep that file ctx=0."""
    import shutil as _sh
    import subprocess as _sp
    base = Path(path).resolve() if path else Path.cwd()
    if not base.exists():
        return {"output": f"Path does not exist: {base}", "error": True}
    flags = _include_flags(include)
    if _sh.which("rg"):
        cwd = base if base.is_dir() else base.parent
        search_arg = "." if base.is_dir() else base.name
        cmd = ["rg", "--count-matches", "--color", "never"]
        if flags:
            cmd += flags
        else:
            for _g in _rg_excludes(base):
                cmd += ["-g", _g]
        cmd += ["--", pattern, search_arg]
        try:
            proc = _sp.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=30)
        except Exception:
            proc = None
        if proc is not None and proc.returncode in (0, 1):
            counts: dict = {}
            for line in proc.stdout.splitlines():
                if ":" not in line:
                    continue
                fp, _, num = line.rpartition(":")
                try:
                    n = int(num.strip())
                except ValueError:
                    continue
                if not n:
                    continue
                pp = Path(fp)
                if not pp.is_absolute():
                    try:
                        pp = (cwd / pp).resolve()
                    except Exception:
                        pass
                if _walk_ignored(base, pp):
                    continue
                counts[str(pp)] = counts.get(str(pp), 0) + n
            if counts or proc.returncode == 1:
                total = sum(counts.values())
                try:
                    rel = {str(Path(fp).relative_to(base) if Path(fp).is_absolute() and str(Path(fp)).startswith(str(base)) else fp): n for fp, n in counts.items()}
                except Exception:
                    rel = dict(counts)
                top = sorted(rel.items(), key=lambda kv: -kv[1])[:20]
                lines = [f"Found {total} matches in {len(counts)} file(s) (count mode)"]
                for fp, n in top:
                    lines.append(f"  {n:4d}  {fp}")
                if len(counts) > 20:
                    lines.append(f"  ... +{len(counts) - 20} more files")
                if top:
                    lines.append(f"Next: grep ONE file context=0, then read. e.g. path={top[0][0]}")
                lines.append("Tip: count first, then one-file grep, then ONE read.")
                return {"output": "\n".join(lines), "metadata": {"matches": total, "files": len(counts), "mode": "count", "top": [f for f, _ in top]}}
    res = _grep(pattern, path if isinstance(path, str) else (str(base) if path else None), include, cfg=cfg)
    if res.get("error"):
        return res
    return {"output": res["output"][:2000], "metadata": {**res.get("metadata", {}), "mode": "count-fallback"}}


def _format_context_groups(hits, order, total_hits, pad, low, *, max_groups=50, max_groups_per_file=5, pattern: str = ""):
    """Path-once rendering + ledger + caps. Only emitted groups are marked.

    Mega-spans (adjacent hits merging into hundreds of lines) are chunked to
    _MAX_GROUP_LINES so one file can't blow past the byte cap in a single
    group — the cap only breaks *between* groups.

    Groups sort by best-match score (def-exact > def > code-exact >
    code-substring > comment), NOT walk order — the wanted def lands first
    instead of at result 20 under comment mentions.
    """
    from pathlib import Path as _P

    from .context_ledger import mark_delivered

    max_out = LOW_MAX_OUTPUT if low else MAX_OUTPUT
    header = f"Found {total_hits} match" + ("es" if total_hits != 1 else "") + f" (context={pad})"
    try:
        term = _plain_term(pattern)
    except Exception:
        term = ""
    wanted: list = []
    dropped = 0
    for fp in order:
        rows = sorted(hits.get(fp) or [], key=lambda r: r[0])
        spans: list = []
        for ln, body, is_hit in rows:
            if spans and ln <= spans[-1][1] + 1:
                spans[-1][1] = max(spans[-1][1], ln)
                spans[-1][2].append((ln, body, is_hit))
            else:
                spans.append([ln, ln, [(ln, body, is_hit)]])
        if len(spans) > max_groups_per_file:
            dropped += len(spans) - max_groups_per_file
            spans = spans[:max_groups_per_file]
        for s, e, items in spans:
            while items and not items[0][2] and not items[0][1].strip():
                items.pop(0)
                s += 1
            while items and not items[-1][2] and not items[-1][1].strip():
                items.pop()
                e -= 1
            if not items:
                dropped += 1
                continue
            for ci in range(0, len(items), _MAX_GROUP_LINES):
                chunk = items[ci:ci + _MAX_GROUP_LINES]
                wanted.append((fp, chunk[0][0], chunk[-1][0], chunk))
    # best match first: score every group, sort, THEN cap (walk order only
    # breaks exact ties via the span start inside the score key)
    try:
        wanted.sort(key=lambda g: (_group_score(g[3], term), g[0], g[1]))
    except Exception:
        pass
    if len(wanted) > max_groups:
        dropped += len(wanted) - max_groups
        wanted = wanted[:max_groups]
    out_lines = [header]
    out_chars = len(header) + 1
    emitted = 0
    shown_files: list = []
    truncated = False
    for idx, (fp, s, e, items) in enumerate(wanted):
        block = [f"--- {fp}:{s}-{e} · read offset={s} limit={e - s + 1} ---"]
        emit_rows: list = []
        for ln, body, is_hit in items:
            if low and len(body) > LOW_MAX_LINE:
                body = body[:LOW_MAX_LINE] + "..."
            if is_hit:
                row = f"> Line {ln}: {body}"
                _m = _DEF_HIT_RE.match(body)
                if _m:
                    row += f"  ← read symbol='{_m.group(1)}'"
                emit_rows.append((ln, True, row))
            else:
                emit_rows.append((ln, False, f"  {ln}: {body}"))
        block.extend(r for _, _, r in emit_rows)
        cost = sum(len(b) + 1 for b in block)
        if out_chars + cost > max_out and emitted:
            truncated = True
            dropped += len(wanted) - idx
            break
        out_lines.extend(block)
        out_chars += cost
        emitted += 1
        if fp not in shown_files:
            shown_files.append(fp)
        try:
            gst = _P(fp).stat()
        except OSError:
            continue
        run: list = []
        for ln, is_hit, _r in sorted(emit_rows):
            if not is_hit:
                continue
            if run and ln == run[-1] + 1:
                run.append(ln)
            else:
                if run:
                    mark_delivered(fp, gst.st_mtime_ns, gst.st_size, run[0], run[-1])
                run = [ln]
        if run:
            mark_delivered(fp, gst.st_mtime_ns, gst.st_size, run[0], run[-1])
    if truncated or dropped:
        out_lines.append(f"(… {dropped} more group(s) hidden — narrow pattern/path, or read one group via its offset/limit.)")
    out_lines.append("Tip: read offset=<lo> limit=<n> continues a group; read symbol='<Name>' jumps to a def.")
    out = "\n".join(out_lines)
    if low and len(out) > LOW_MAX_OUTPUT:
        out = out[:LOW_MAX_OUTPUT] + "\n… (save-data: output capped at 10 KB.)"
        truncated = True
    return {"output": out, "metadata": {"matches": total_hits, "context": pad, "truncated": truncated or dropped > 0, "groups": emitted, "files": shown_files}}


def _grep_with_context(pattern: str, path=None, include=None, context: int = 2, cfg=None, *, max_groups: int = 50, max_groups_per_file: int = 5) -> dict:
    """Context grep, rg-first: location + code in one shot.

    rg --json -C when rg exists (~15x faster repo-wide); pure-Python walk
    otherwise. Groups render path-once with read-ready offset/limit anchors,
    def/class hits link read symbol=, output is group- + byte-capped, and
    only emitted match lines enter the context ledger.
    """
    base = Path(path).resolve() if path else Path.cwd()
    if not base.exists():
        return {"output": f"Path does not exist: {base}", "error": True}
    pad = max(0, int(context or 0))
    low = _is_low(cfg)
    res = None
    if shutil.which("rg"):
        res = _grep_rg_json(pattern, base, include, pad)
    if res is None:
        res = _collect_py_context(pattern, base, include, pad)
        if isinstance(res, dict):
            return res
    hits, order, total = res
    if not order:
        return {"output": "No files found"}
    return _format_context_groups(hits, order, total, pad, low, max_groups=max_groups, max_groups_per_file=max_groups_per_file, pattern=pattern)


_BATCH_MAX_QUERIES = 8
_BATCH_GROUPS = 6
_BATCH_GROUPS_PER_FILE = 2


def _grep_many(queries, path=None, include=None, context: int = 2, cfg=None) -> dict:
    """Run several patterns concurrently (wall time ~= slowest, not the sum).

    Accepts ["re1", "re2"] or [{"pattern": "re", "path": ..., "include": ...,
    "context": N}]. Deduped in order, capped; each query renders with tight
    batch caps so one call stays usable. Results keep submission order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not isinstance(queries, list) or not queries:
        return {"output": "grep queries requires a non-empty 'queries' array.", "error": True}
    norm: list = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            norm.append({"pattern": q.strip()})
        elif isinstance(q, dict) and str(q.get("pattern") or "").strip():
            norm.append({
                "pattern": str(q["pattern"]).strip(),
                "path": q.get("path", path),
                "include": q.get("include", include),
                "context": q.get("context", context),
            })
    seen: list = []
    for q in norm:
        if q not in seen:
            seen.append(q)
    dropped = seen[_BATCH_MAX_QUERIES:]
    work = seen[:_BATCH_MAX_QUERIES]
    if not work:
        return {"output": "grep queries requires at least one non-empty pattern.", "error": True}

    def _one(q: dict) -> dict:
        try:
            ctx = int(q.get("context", context))
        except (TypeError, ValueError):
            ctx = context
        if ctx > 0:
            return _grep_with_context(q["pattern"], q.get("path"), q.get("include"), ctx, cfg=cfg,
                                      max_groups=_BATCH_GROUPS, max_groups_per_file=_BATCH_GROUPS_PER_FILE)
        return _grep(q["pattern"], q.get("path"), q.get("include"), cfg=cfg)

    workers = max(1, min(5, len(work)))
    results: list = [None] * len(work)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, q): i for i, q in enumerate(work)}
        for fut in as_completed(futs):
            try:
                results[futs[fut]] = fut.result()
            except Exception as e:
                results[futs[fut]] = {"output": f"query failed: {e}", "error": True}
    lines = [f"# Batch grep results ({len(work)} patterns, {workers} workers)"]
    ok = 0
    for i, (q, r) in enumerate(zip(work, results), 1):
        where = f" in {q.get('path')}" if q.get("path") else ""
        body = (r or {}).get("output", "")
        if not (r or {}).get("error"):
            ok += 1
            lines.append(f"\n## {i}. `{q['pattern']}`{where} — {(r or {}).get('metadata', {}).get('matches', '?')} matches\n{body}")
        else:
            lines.append(f"\n## {i}. `{q['pattern']}`{where} — FAILED\n{body}")
    if dropped:
        lines.append(f"\n[note: {len(dropped)} quer{'y' if len(dropped) == 1 else 'ies'} beyond the {_BATCH_MAX_QUERIES} cap dropped]")
    return {"output": "\n".join(lines), "metadata": {"count": len(work), "succeeded": ok,
            "failed": len(work) - ok, "dropped": len(dropped), "concurrency": workers}}


def tool(cfg=None) -> Tool:
    description = (
        "Plain-text search (NOT for code jumps — use lsp first for definitions/refs). "
        "Best match FIRST. COUNT-FIRST: "
        "mode='count' (~0.4KB) → pick ONE file → grep that "
        "file context=0 (~0.3KB) → ONE read symbol=/offset=. "
        "Default returns paths + line numbers WITH code (context=N, default 2). "
        "include accepts '*.py' or ['*.py','*.md']. "
        "PARALLEL: queries=[...] (up to 8) in ONE call."
    )

    def run(input: dict) -> dict:
        if input.get("queries"):
            try:
                ctx = int(input.get("context", 2))
            except (TypeError, ValueError):
                ctx = 2
            return _grep_many(input.get("queries"), input.get("path"), input.get("include"), ctx, cfg=cfg)
        mode = str(input.get("mode") or "lines").strip().lower()
        if mode.startswith("count"):
            if not str(input.get("pattern") or "").strip():
                return {"output": "grep requires 'pattern' (or parallel 'queries').", "error": True}
            return _count_only(input["pattern"], input.get("path"), input.get("include"), cfg=cfg)
        if not str(input.get("pattern") or "").strip():
            return {"output": "grep requires 'pattern' (or parallel 'queries').", "error": True}
        try:
            ctx = int(input.get("context", 2))
        except (TypeError, ValueError):
            ctx = 2
        if ctx > 0:
            return _grep_with_context(input["pattern"], input.get("path"), input.get("include"), ctx, cfg=cfg)
        return _grep(input["pattern"], input.get("path"), input.get("include"), cfg=cfg)

    return Tool(
        name="grep",
        description=description,
        parameters=schema_with(
            {
                "pattern": {"type": "string", "description": "Regex (or omit when using queries)", "optional": True},
                "path": {"type": "string", "description": "Dir or file", "optional": True},
                "include": {"type": "string", "description": "File filter '*.py' or list ['*.py','*.md'] (list also accepted)", "optional": True},
                "mode": {"type": "string", "description": "lines (default) or count (~0.4KB: per-file counts first)", "enum": ["lines", "count"], "optional": True},
                "context": {"type": "integer", "description": "Surrounding lines per match (default 2, 0 = lines only)", "optional": True},
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Regex to search"},
                            "path": {"type": "string", "description": "Dir or file"},
                            "include": {"type": "string", "description": "File filter"},
                            "context": {"type": "integer", "description": "Surrounding lines"},
                        },
                    },
                    "description": "PARALLEL: up to 8 {pattern,path?,include?,context?} objects in one call (bare strings also accepted)",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="grep",
    )
