"""Symbol index engine: build/update a per-worktree index and answer queries.

Query verbs (first token decides what is asked; everything else is a
case-insensitive substring match against names):

    def/define/definition/where     -> the definition(s) of a name
    callers/calls/called/call       -> usage sites that *invoke* a name
    refs/references/uses/usages     -> every usage site
    deps/dependencies/depends       -> imports of a file (by path or module)
    imports/who-imports             -> files that import a module/file
    symbols/list                    -> every definition in a file

Bare names (no verb) return definitions first, then references.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .cache import SymbolCache
from .heuristic_indexer import index_file as heuristic_index_file
from .model import FileIndex, ImportRecord, Ref, Symbol
from .python_indexer import index_file as python_index_file

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_IGNORE = {
    ".git", ".hg", ".svn", ".bzr",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    "dist", "build", "target", "out", "_build", "vendor", "third_party",
    ".venv", "venv", "env", ".tox", ".gradle", ".idea", ".vscode", "Pods",
    ".cache", ".next", ".nuxt", ".coverage", "coverage", "elm-stuff", "deps",
}

# languages given a real/rule-based indexer
CODE_LANGS = {
    "python", "javascript", "c", "csharp", "java", "kotlin", "scala", "groovy",
    "go", "rust", "swift", "php", "lua", "ruby", "perl", "r", "sql", "bash",
    "fish", "generic",
}

# explicit extensions routed to the generic heuristic indexer
_GENERIC_EXTS = {
    ".dart", ".m", ".mm", ".tcl", ".pas", ".v", ".vh", ".vhd", ".asm", ".s",
    ".nix", ".lisp", ".clj", ".cljs", ".elm", ".erl", ".hrl", ".ex", ".exs",
    ".gleam", ".zig", ".odin", ".nim", ".hs", ".ml", ".fs", ".fsx", ".vb",
    ".awk", ".sed", ".ps1", ".bat", ".cmd", ".d", ".hcl", ".proto", ".thrift",
    ".jl", ".pony", ".smt2", ".cr", ".e", ".gnu", ".m4", ".sls",
}

# data/doc languages: parsed but produce no symbols (noise)
_DATA_LANGS = {"json", "markdown", "yaml", "toml", "xml", "html", "css", "text"}

MAX_FILE_BYTES = 1_500_000  # skip oversized (typically vendored) files
MAX_RESULTS = 60


def _is_binary(src_byte_prefix: bytes) -> bool:
    from ..tools.read import _is_binary_sample

    return _is_binary_sample(src_byte_prefix)


def _language_for(rel: str) -> str:
    from .model import language_for

    lang = language_for(rel)
    if lang:
        return lang
    lower = rel.lower()
    idx = lower.rfind(".")
    if idx >= 0 and lower[idx:] in _GENERIC_EXTS:
        return "generic"
    return ""


# ---------------------------------------------------------------------------
# index builder
# ---------------------------------------------------------------------------

class IndexEngine:
    """Holds one in-memory index per root, refreshed lazily per query."""

    def __init__(self) -> None:
        self._roots: dict[str, dict[str, FileIndex]] = {}
        self._lock = threading.Lock()
        # ponytail: per-root exact-name index (lower name -> refs). 46K-ref
        # scans become dict hits for exact/case-exact; prefix/substring
        # still scan. Rebuilt when the root map is swapped.
        self._ref_exact: dict[str, dict[str, list]] = {}

    def ensure(
        self,
        root: Path,
        additional_ignores: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, FileIndex]:
        """Return the current per-file index for `root`, rebuilding as needed."""
        root = root.resolve()
        with self._lock:
            cached = self._roots.get(str(root))
            ignores = set(DEFAULT_IGNORE) | set(additional_ignores or [])
            snapshot = dict(cached) if cached is not None else None
        # The os.walk + full parse runs OUTSIDE the lock: one big repo no
        # longer blocks every other root's query. The fresh map is swapped
        # back in atomically under the lock.
        if snapshot is not None and not force:
            # refresh incrementally against disk (per-file mtime/size
            # compare — only changed files are re-parsed; the walk itself
            # is the remaining cost and can't be skipped without risking
            # stale results right after an edit)
            fresh, changed = self._refresh(snapshot, root, ignores)
        else:
            fresh, _changed = self._walk_and_index(root, ignores, force)
            changed = _changed
        with self._lock:
            self._roots[str(root)] = fresh
            if not changed:
                # Unchanged refresh: keep the existing exact-name bucket
                # (rebuilding it walks all 46K refs + re-saves 4MB cache —
                # the 0.5s warm-query cost). Bucket rebuilds only when files
                # actually changed.
                return fresh
            try:
                bucket: dict[str, list] = {}
                for fi in fresh.values():
                    try:
                        for r in fi.refs:
                            try:
                                bucket.setdefault(r.name, []).append(r)
                                ln = r.name.lower()
                                if ln != r.name:
                                    bucket.setdefault(ln, []).append(r)
                            except Exception:
                                continue
                    except Exception:
                        continue
                self._ref_exact[str(root)] = bucket
            except Exception:
                pass
            return fresh

    def _refresh(
        self, cached: dict[str, FileIndex], root: Path, ignores: set[str]
    ) -> tuple[dict[str, FileIndex], bool]:
        """Compare on-disk stamps to the cached ones; re-index only changes.

        Returns (fresh map, changed). Callers skip bucket rebuilds + cache
        re-saves when nothing changed.
        """
        new: dict[str, FileIndex] = {}
        changed = False
        seen = set()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in ignores and not (d.startswith(".") and d not in (".", ".."))
            ]
            dirnames.sort()
            rel_dir = os.path.relpath(dirpath, root)
            is_root = rel_dir == "."
            for fn in sorted(filenames):
                if fn.startswith("."):
                    continue
                rel = fn if is_root else os.path.join(rel_dir, fn)
                seen.add(rel)
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size > MAX_FILE_BYTES:
                    continue
                rec = cached.get(rel)
                if rec is not None and rec.size == st.st_size and rec.mtime == st.st_mtime_ns:
                    # Same size+mtime usually means unchanged — but same-size
                    # edits, coarse filesystems, and git checkouts can keep
                    # both stamps while changing content. Records that carry a
                    # content hash get a cheap 1KB head check; records without
                    # one (older caches) are re-parsed once, earning a hash.
                    if rec.content_hash and self._head_hash(path) == rec.content_hash:
                        new[rel] = rec
                        continue
                    if not rec.content_hash:
                        pass  # fall through to re-parse below
                fi = self._index_one(root, rel, path, st)
                if fi is not None:
                    new[rel] = fi
                    changed |= True
        if set(cached) != seen:
            # Only INDEXABLE files count: _index_one skips data langs
            # (markdown/toml/json...), dotfiles, oversize and binary — the
            # walk sees them but the cache never holds them. Comparing raw
            # `seen` re-saved the 4MB cache + rebuilt the 46K bucket on
            # EVERY query. ponytail: mirrors _index_one's skip rules; a new
            # skip rule must be added here too.
            try:
                indexable = set()
                for r in seen:
                    try:
                        lang = _language_for(str(r))
                        if not lang or lang in _DATA_LANGS or lang == "text":
                            continue
                        if str(r).startswith("."):
                            continue
                        indexable.add(r)
                    except Exception:
                        continue
            except Exception:
                indexable = seen
            if set(cached) != indexable:
                changed |= True  # files deleted or renamed
        if changed:
            SymbolCache(root).save(new)
        return new, changed

    @staticmethod
    def _head_hash(path: str) -> str:
        try:
            import hashlib as _hl

            with open(path, "rb") as fh:
                return _hl.sha1(fh.read(4096)).hexdigest()[:16]
        except OSError:
            return ""

    def _walk_and_index(
        self, root: Path, ignores: set[str], force: bool
    ) -> tuple[dict[str, FileIndex], bool]:
        files: dict[str, FileIndex] = {}
        disk_file_map: dict[str, tuple[Path, os.stat_result]] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in ignores and not (d.startswith(".") and d not in (".", ".."))
            ]
            dirnames.sort()
            rel_dir = os.path.relpath(dirpath, root)
            is_root = rel_dir == "."
            for fn in sorted(filenames):
                if fn.startswith("."):
                    continue
                rel = fn if is_root else os.path.join(rel_dir, fn)
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size > MAX_FILE_BYTES:
                    continue
                disk_file_map[rel] = (path, st)

        # consult cache unless forcing a full rebuild
        cached: dict[str, FileIndex] = {}
        if not force:
            try:
                cached = SymbolCache(root).load() or {}
            except Exception:
                cached = {}
        changed = not force
        for rel, (path, st) in disk_file_map.items():
            rec = cached.get(rel) if not force else None
            if not force and rec is not None and rec.size == st.st_size and rec.mtime == st.st_mtime_ns:
                if rec.content_hash and self._head_hash(path) == rec.content_hash:
                    changed |= False
                    files[rel] = rec
                    continue
                if rec.content_hash:
                    pass  # hash mismatch — re-parse below
                # no hash yet (older cache): re-parse once below
            fi = self._index_one(root, rel, path, st)
            if fi is not None:
                files[rel] = fi
                changed |= True
        if cached and set(cached) != set(disk_file_map):
            changed |= True
        if changed:
            SymbolCache(root).save(files)
        return files, changed

    def _index_one(
        self, root: Path, rel: str, path: str, st: os.stat_result
    ) -> FileIndex | None:
        lang = _language_for(rel)
        if not lang or lang in _DATA_LANGS or lang == "text":
            return None
        try:
            with open(path, "rb") as fh:
                head = fh.read(1024)
                if _is_binary(head):
                    return None
                fh.seek(0)
                data = fh.read()
        except OSError:
            return None
        src = data.decode("utf-8", errors="replace")
        mtime = st.st_mtime_ns
        size = st.st_size
        try:
            import hashlib as _hl

            ch = _hl.sha1(data[:4096]).hexdigest()[:16]
        except Exception:
            ch = ""
        if lang == "python":
            fi = python_index_file(root, rel, src, size, mtime)
        else:
            fi = heuristic_index_file(root, rel, src, size, mtime, lang)
        if fi is not None:
            fi.content_hash = ch
        return fi


_ENGINE = IndexEngine()


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

_VERB_DEF = ("def", "define", "definition", "definitions", "where", "what is", "what's")
_VERB_CALL = ("callers", "calls", "called", "call", "usages", "usage", "who calls", "who invokes")
_VERB_REF = ("refs", "references", "uses")
_VERB_DEPS = ("deps", "dependencies", "depends", "depend", "imports of", "what does")
_VERB_IMPORT = ("imports", "who imports", "used by")
_VERB_FILE = ("symbols", "list", "contents", "in file")


def _strip(q: str) -> str:
    return q.strip().strip("?:!.").strip()


def _parse_query(q: str) -> tuple[str, str]:
    """Return (mode, term). term is the rest-of-query name/path."""
    q0 = _strip(q)
    low = " " + q0.lower() + " "
    for verb, mode in (
        (_VERB_FILE, "symbols"),
        (_VERB_DEPS, "deps"),
        (_VERB_IMPORT, "imports"),
        (_VERB_DEF, "def"),
        (_VERB_CALL, "callers"),
        (_VERB_REF, "refs"),
    ):
        for v in sorted(verb, key=len, reverse=True):
            start = low.find(" " + v)
            if start >= 0:
                end = start + 1 + len(v)
                term = q0[end:].strip().strip("?\"'`")
                if term:
                    return mode, term
    # "who imports X"
    term = q0
    for stop in (" in ", " of "):
        i = low.find(stop)
        if i > 0:
            pre = q0[:i]
            if pre in ("imports", "what", "who", "which files"):
                return "imports", q0[i + len(stop):].strip()
            if pre in ("deps", "dependencies", "depends"):
                return "deps", q0[i + len(stop):].strip()
    return "def", q0


def _match_rank(name: str, term: str) -> int:
    """0 = exact, 1 = case-insensitive exact, 2 = startswith, 3 = substring, -1 none."""
    if name == term:
        return 0
    ln, lt = name.lower(), term.lower()
    if ln == lt:
        return 1
    if ln.startswith(lt):
        return 2
    if lt in ln:
        return 3
    return -1


def _resolve_file(root: Path, term: str) -> str | None:
    """Resolve a user-supplied path/module/filename to an index 'path' key."""
    t = term.strip()
    if not t:
        return None
    p = Path(t)
    if p.is_absolute():
        try:
            return str(p.relative_to(root))
        except ValueError:
            return None
    candidate = root / p
    if candidate.exists():
        rel = p.as_posix()
        return rel
    # dotted module name -> possible files
    dotted = t.replace("/", ".").lstrip(".")
    for suffix in (".py", "/__init__.py", ""):
        mid = dotted.replace(".", "/")
        if suffix.startswith("/"):
            cand = mid + suffix
        else:
            cand = (mid or t) + suffix
        if (root / cand.lstrip("/")).exists():
            return cand.lstrip("/")
    # fuzzy basename match
    base = os.path.basename(t.rstrip("/"))
    return base or None


def _query_defs(files: dict[str, FileIndex], term: str, kind: str) -> list[Symbol]:
    out: list[Symbol] = []
    for fi in files.values():
        for s in fi.symbols:
            rank = _match_rank(s.name, term)
            if rank < 0:
                continue
            if kind and s.kind != kind:
                # allow kind=fn to also match methods for callers ergonomics
                if not (kind in ("fn", "function") and s.kind in ("function", "method")):
                    continue
            out.append(s)
    out.sort(key=lambda s: (s.file, s.line))
    return out


def _query_refs(files: dict[str, FileIndex], term: str, only_calls: bool) -> list[Ref]:
    # Fast path: exact/case-exact hits via the per-root name index when the
    # caller passes it (engine wires it below); otherwise fall back to scan.
    # ponytail: index lives on the engine, not the file map — no new class.
    out: list[Ref] = []
    for fi in files.values():
        for r in fi.refs:
            rank = _match_rank(r.name, term)
            if rank < 0:
                continue
            if only_calls and r.role != "call":
                continue
            out.append(r)
    out.sort(key=lambda r: (r.file, r.line))
    return out


def _query_refs_indexed(exact: dict[str, list] | None, files: dict[str, FileIndex], term: str, only_calls: bool) -> list[Ref]:
    """Same as _query_refs but exact terms hit the name index (no 46K scan)."""
    try:
        if exact:
            lt = term.lower()
            if term in exact or lt in exact:
                out: list[Ref] = []
                seen: set[int] = set()
                for key in (term, lt) if term != lt else (term,):
                    for r in exact.get(key, ()):  # type: ignore[union-attr]
                        try:
                            if id(r) in seen:
                                continue
                            seen.add(id(r))
                            if only_calls and r.role != "call":
                                continue
                            # exact dict may hold lowercase bucket: verify rank
                            if _match_rank(r.name, term) < 0:
                                continue
                            out.append(r)
                        except Exception:
                            continue
                # exact hit: still need substring/startswith cousins — scan
                # ONLY when the exact bucket was empty (rare term).
                if out:
                    out.sort(key=lambda r: (r.file, r.line))
                    return out
    except Exception:
        pass
    return _query_refs(files, term, only_calls)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _fmt_def(s: Symbol, root: Path) -> str:
    try:
        rel = Path(s.file).as_posix()
    except Exception:
        rel = str(s.file)
    where = s.container + "." if s.container else ""
    sig = s.signature or s.name
    end = getattr(s, "end_line", 0) or s.line
    span = f"{s.line}-{end}" if end != s.line else f"{s.line}"
    return f"{rel}:{span}\n  {sig}  ({s.kind} in {where or 'module'}) — read symbol='{s.name}'"


def _run_query(files: dict[str, FileIndex], mode: str, term: str,
               kind: str, limit: int, root: Path) -> dict:
    matches: list = []
    if mode == "def":
        matches = _query_defs(files, term, kind)
        title = f"Definition{'' if len(matches) == 1 else 's'} of `{term}`"
    elif mode in ("callers", "refs"):
        try:
            exact = _ENGINE._ref_exact.get(str(root.resolve()))
        except Exception:
            exact = None
        matches = _query_refs_indexed(exact, files, term, only_calls=(mode == "callers"))
        title = f"`{term}` is referenced {len(matches)} time"
        title += "" if len(matches) == 1 else "s"
    elif mode == "deps":
        return _query_deps(files, term, root, limit)
    elif mode == "imports":
        return _query_imports(files, term, root, limit)
    elif mode == "symbols":
        return _query_symbols(files, term, root, limit)
    else:
        return {"output": f"Unhandled query mode: {mode}", "error": True}

    if not matches:
        return {
            "output": f"No {mode} found for `{term}`.",
            "metadata": {"term": term, "mode": mode, "matches": 0, "items": []},
        }

    shown = matches[:limit]
    if mode == "def":
        lines = [title]
        last = None
        for s in shown:
            rel = s.file
            if rel != last:
                lines.append(f"{rel}:")
                last = rel
            lines.append(_fmt_def(s, root))
    else:
        lines = [title]
        last = None
        for r in shown:
            if r.file != last:
                lines.append(r.file + ":")
                last = r.file
            role = r.role if r.role != "use" else ""
            lines.append(f"  line {r.line}" + (f"  ({role})" if role else ""))
    if len(matches) > limit:
        lines.append(f"... and {len(matches) - limit} more (limit={limit})")

    structured = {"term": term, "mode": mode, "matches": len(matches), "items": [
        {"file": s.file, "line": s.line, "end_line": getattr(s, "end_line", 0) or s.line,
         "name": s.name, "kind": s.kind,
         "signature": s.signature, "container": s.container} for s in shown[:limit]
    ]} if mode == "def" else {
        "term": term, "mode": mode, "matches": len(matches), "items": [
            {"file": r.file, "line": r.line, "name": r.name, "role": r.role}
            for r in shown[:limit]
        ]
    }
    return {"output": "\n".join(lines), "metadata": structured}


def _query_deps(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    rel = _resolve_file(root, term)
    if rel is None:
        return {"output": f"Could not resolve `{term}` to a file.", "error": True}
    # exact or fuzzy file key
    key = rel
    fi = files.get(key)
    if fi is None:
        for k in files:
            if k == rel or k.endswith(rel) or os.path.basename(k) == rel:
                fi = files.get(k)
                rel = k
                break
    if fi is None or fi.language not in CODE_LANGS:
        return {"output": f"No indexed file matches `{term}`.", "error": True}
    local: list[ImportRecord] = []
    external: list[ImportRecord] = []
    for imp in fi.imports:
        (local if imp.local else external).append(imp)
    lines = [f"Dependencies of {rel}:", ""]
    lines.append(f"\u2022 Local ({len(local)}):")
    for imp in sorted(local, key=lambda x: x.module):
        target = _module_to_file(imp.module)
        if target:
            lines.append(f"  {imp.module}  ->  {target}")
        else:
            lines.append(f"  {imp.module}")
    lines.append(f"\u2022 External / stdlib ({len(external)}):")
    for imp in sorted(external, key=lambda x: x.module):
        lines.append(f"  {imp.module}")
    return {
        "output": "\n".join(lines),
        "metadata": {"file": rel, "local": [i.module for i in local],
                     "external": [i.module for i in external]},
    }


def _module_to_file(module: str) -> str | None:
    """Best-effort module name -> relative file path (for local deps)."""
    m = module.replace(".", "/").lstrip("/")
    if not m:
        return None
    return f"{m}.py"


def _query_imports(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    results: list[tuple[str, str, int]] = []  # (file, module, line)
    lowered = term.lower().rstrip("/")
    for fi in files.values():
        for imp in fi.imports:
            if imp.module.lower() == lowered or imp.module.lower().endswith("." + lowered):
                results.append((fi.path, imp.module, imp.line))
    results.sort()
    if not results:
        return {"output": f"No files import `{term}`.", "metadata": {"term": term, "items": []}}
    shown = results[:limit]
    lines = [f"Files importing `{term}` ({len(results)}):"]
    last = None
    for file, module, line in shown:
        if file != last:
            lines.append(f"{file}:")
            last = file
        lines.append(f"  line {line}: import {module}")
    if len(results) > limit:
        lines.append(f"... and {len(results) - limit} more (limit={limit})")
    return {
        "output": "\n".join(lines),
        "metadata": {"term": term, "items": [
            {"file": f, "line": ln, "module": m} for f, m, ln in shown]},
    }


def _query_symbols(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    rel = _resolve_file(root, term)
    if rel is None:
        return {"output": f"Could not resolve `{term}` to a file.", "error": True}
    fi = files.get(rel)
    if fi is None:
        for k in files:
            if k == rel or os.path.basename(k) == rel:
                fi = files.get(k)
                rel = k
                break
    if fi is None:
        return {"output": f"No indexed file matches `{term}`.", "error": True}
    syms = sorted(fi.symbols, key=lambda s: s.line)
    if not syms:
        return {"output": f"No definitions found in {rel}.", "metadata": {"file": rel, "items": []}}
    shown = syms[:limit]
    lines = [f"Definitions in {rel}:"]
    for s in shown:
        lines.append(_fmt_def(s, root))
    lines.append("Tip: read symbol='<name>' returns the block directly — no offset math.")
    if len(syms) > limit:
        lines.append(f"... and {len(syms) - limit} more (limit={limit})")
    return {
        "output": "\n".join(lines),
        "metadata": {"file": rel, "items": [
            {"name": s.name, "kind": s.kind, "line": s.line,
             "signature": s.signature, "container": s.container} for s in shown]},
    }


def query(
    q: str,
    root: Path | str | None = None,
    kind: str = "",
    limit: int = MAX_RESULTS,
    ignore_extra: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Run a query against the (lazily built) symbol index for `root`."""
    from ..globals import resolve_worktree

    start = time.monotonic()
    root_path = Path(root) if root else Path.cwd()
    root_path = resolve_worktree(root_path)
    mode, term = _parse_query(q)
    if not term:
        return {"output": f"Empty query. Try e.g. `def {term or 'main'}`, `callers {term or 'x'}`, `deps {term or 'file.py'}`, `symbols {term or 'file.py'}`.", "error": True}
    files = _ENGINE.ensure(root_path, ignore_extra, force)
    elapsed_ms = (time.monotonic() - start) * 1000
    kind = (kind or "").strip().lower()
    result = _run_query(files, mode, term, kind, int(limit or MAX_RESULTS), root_path)
    result.setdefault("metadata", {})["indexed_files"] = len(files)
    result.setdefault("metadata", {})["elapsed_ms"] = round(elapsed_ms, 1)
    return result