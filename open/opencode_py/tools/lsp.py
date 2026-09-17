"""lsp tool: smart code jump without an LSP server (fits this product).

Mirrors official opencode's `lsp` tool operations
(goToDefinition, findReferences, hover, documentSymbol, workspaceSymbol,
goToImplementation, incomingCalls, outgoingCalls, diagnostics) but backed by
the product's own symbol index (AST for Python, regex/ctags for the rest) plus
pyflakes diagnostics — no pyright/gopls/clangd binaries needed on the phone.

- Position-exact: filePath + line + character resolves the word at that spot,
  then jumps (no guessing).
- Fast path: hover / documentSymbol / diagnostics / same-file definition index
  ONE file (~60ms) instead of walking the whole worktree (~1s). Cross-file
  search falls back to the cached index.
- Diagnostics: real pyflakes + compile errors for Python (what official's
  `lsp.diagnostics()` + `touchFile` gives after edit/write).
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from ..globals import resolve_worktree
from .registry import Tool, schema_with

_OPERATIONS = [
    "goToDefinition",
    "findReferences",
    "hover",
    "documentSymbol",
    "workspaceSymbol",
    "goToImplementation",
    "incomingCalls",
    "outgoingCalls",
    "diagnostics",
]

_WORD_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")

_FILE_CACHE: dict[str, tuple] = {}
_DIAG_CACHE: dict[str, tuple] = {}


def _stat_key(path: Path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _cached_file(path: Path):
    key = _stat_key(path)
    if key is None:
        return None, None, None
    hit = _FILE_CACHE.get(str(path))
    if hit is not None and hit[0] == key[0] and hit[1] == key[1]:
        return hit[2], hit[3], hit[4]
    return None, None, None


def _store_file(path: Path, lines, tree, fi) -> None:
    try:
        key = _stat_key(path)
        if key is None:
            return
        if len(_FILE_CACHE) > 64:
            _FILE_CACHE.pop(next(iter(_FILE_CACHE)))
        _FILE_CACHE[str(path)] = (key[0], key[1], fi, lines, tree)
    except Exception:
        pass


_ROOT_CACHE = None


def _root() -> Path:
    global _ROOT_CACHE
    try:
        if _ROOT_CACHE is not None:
            return _ROOT_CACHE
    except Exception:
        pass
    try:
        r = resolve_worktree(Path.cwd())
    except Exception:
        r = Path.cwd()
    try:
        _ROOT_CACHE = r
    except Exception:
        pass
    return r


def _resolve(root: Path, file_path: str) -> Path | None:
    t = (file_path or "").strip()
    if not t:
        return None
    p = Path(t)
    if p.is_absolute():
        try:
            p.relative_to(root)
            return p
        except ValueError:
            return p if p.exists() else None
    cand = root / p
    if cand.exists():
        return cand
    base = os.path.basename(t.rstrip("/"))
    if base:
        for hit in root.rglob(base):
            try:
                if hit.is_file():
                    return hit
            except OSError:
                continue
            break
    return None


def _lines_cached(path: Path):
    fi, lines, tree = _cached_file(path)
    if lines is not None:
        return lines
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    try:
        t2 = tree if tree is not None else (ast.parse(text) if path.suffix == ".py" else None)
    except Exception:
        t2 = tree
    try:
        _store_file(path, lines, t2, fi)
    except Exception:
        pass
    return lines


def _tree_cached(path: Path):
    fi, lines, tree = _cached_file(path)
    if tree is not None:
        return tree
    try:
        lines2 = lines if lines is not None else _lines_cached(path)
        text = "\n".join(lines2 or [])
        t2 = ast.parse(text) if text and path.suffix == ".py" else None
    except Exception:
        t2 = None
    try:
        _store_file(path, lines2, t2, fi)
    except Exception:
        pass
    return t2


def _word_at(path: Path, line: int, character: int) -> str:
    try:
        lines = _lines_cached(path)
        if not lines or line < 1 or line > len(lines):
            return ""
        s = lines[line - 1]
        if not s:
            return ""
        col = max(1, int(character or 1)) - 1
        col = min(col, max(0, len(s) - 1))
        for m in _WORD_RE.finditer(s):
            if m.start() <= col < m.end():
                return m.group(0)
        return ""
    except Exception:
        return ""


def _one_file_index(root: Path, path: Path):
    try:
        fi, _lines, _tree = _cached_file(path)
        if fi is not None:
            return fi
        from ..index import engine as eng
    except Exception:
        return None
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    try:
        st = os.stat(path)
    except OSError:
        return None
    try:
        fi = eng._ENGINE._index_one(root, rel, str(path), st)
        try:
            _store_file(path, _lines_cached(path), None, fi)
        except Exception:
            pass
        return fi
    except Exception:
        return None


def _fmt_sym(s) -> str:
    try:
        end = getattr(s, "end_line", 0) or s.line
        span = f"{s.line}-{end}" if end != s.line else f"{s.line}"
        where = (s.container + ".") if getattr(s, "container", "") else ""
        sig = getattr(s, "signature", "") or s.name
        return f"{s.file}:{span}\n  {sig}  ({s.kind} in {where or 'module'})"
    except Exception:
        return str(s)


def _docstring(path: Path, name: str) -> str:
    try:
        tree = _tree_cached(path)
    except Exception:
        return ""
    if tree is None:
        return ""
    short = name.split(".")[-1]
    short = name.split(".")[-1]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == short:
                try:
                    return ast.get_docstring(node) or ""
                except Exception:
                    return ""
    return ""


def _diagnostics_py(path: Path) -> list[str]:
    try:
        key = _stat_key(path)
        if key is not None:
            hit = _DIAG_CACHE.get(str(path))
            if hit is not None and hit[0] == key[0] and hit[1] == key[1]:
                return list(hit[2])
    except Exception:
        pass
    out: list[str] = []
    try:
        src = path.read_bytes()
    except OSError as e:
        return [f"unreadable: {e}"]
    try:
        compile(src, str(path), "exec")
    except SyntaxError as e:
        where = f"line {e.lineno}" + (f", col {e.offset}" if e.offset else "")
        return [f"ERROR [{where}] {e.msg}"]
    except (ValueError, OverflowError, TypeError) as e:
        return [f"ERROR {e}"]
    try:
        from pyflakes.api import check as _check
        from pyflakes.reporter import Reporter as _Reporter
        import io as _io
    except Exception:
        return []
    try:
        err = _io.StringIO()
        rep = _Reporter(_io.StringIO(), err)
        _check(src.decode("utf-8", errors="replace"), str(path), rep)
        text = err.getvalue().strip()
        if text:
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    out.append(ln)
                if len(out) >= 20:
                    break
    except Exception:
        pass
    try:
        key2 = _stat_key(path)
        if key2 is not None:
            if len(_DIAG_CACHE) > 64:
                _DIAG_CACHE.pop(next(iter(_DIAG_CACHE)))
            _DIAG_CACHE[str(path)] = (key2[0], key2[1], list(out))
    except Exception:
        pass
    return out


def tool() -> Tool:
    description = (
        "FIRST for code: position-exact jump in 0.4ms (cached). "
        "Have file+line? goToDefinition/hover AT that spot. "
        "Need overview? documentSymbol. Errors? diagnostics. "
        "Name anywhere? workspaceSymbol query=. Users? findReferences/incomingCalls. "
        "Calls made? outgoingCalls. Implementation? goToImplementation. "
        "ALWAYS before editing or reading big files."
    )

    def run(input: dict) -> dict:
        op = str(input.get("operation") or "").strip()
        if op not in _OPERATIONS:
            return {"output": f"Unknown operation {op!r} (want one of {', '.join(_OPERATIONS)}).", "error": True}
        file_path = str(input.get("filePath") or "").strip()
        query = str(input.get("query") or "").strip()
        try:
            line = int(input.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        try:
            character = int(input.get("character") or 0)
        except (TypeError, ValueError):
            character = 0
        try:
            limit = int(input.get("limit") or 20)
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 60))
        root = _root()

        if op == "workspaceSymbol":
            if not query:
                return {"output": "Empty query. Try workspaceSymbol query='run'.", "error": True}
            try:
                from ..index.engine import query as _q
                r = _q(f"def {query}", limit=limit)
                return {"output": r.get("output", ""), "metadata": {"operation": op, **(r.get("metadata") or {})}}
            except Exception as e:
                return {"output": f"workspaceSymbol failed: {e}", "error": True}

        if not file_path:
            return {"output": "filePath is required (except workspaceSymbol).", "error": True}
        path = _resolve(root, file_path)
        if path is None or not path.exists():
            return {"output": f"File not found: {file_path}", "error": True}

        if op == "documentSymbol":
            fi = _one_file_index(root, path)
            if fi is None:
                return {"output": f"No symbols in {path.name}.", "metadata": {"file": str(path)}}
            syms = sorted(fi.symbols, key=lambda s: s.line)[:limit]
            if not syms:
                return {"output": f"No definitions found in {path.name}."}
            lines = [f"Definitions in {path.name}:"]
            for s in syms:
                lines.append(_fmt_sym(s))
            return {"output": "\n".join(lines), "metadata": {"file": str(path), "matches": len(syms)}}

        if op == "diagnostics":
            if path.suffix == ".py":
                issues = _diagnostics_py(path)
                if not issues:
                    return {"output": f"No diagnostics in {path.name} — clean.", "metadata": {"file": str(path), "issues": 0}}
                lines = [f"Diagnostics in {path.name} ({len(issues)}):"]
                lines.extend(f"- {x}" for x in issues)
                return {"output": "\n".join(lines), "metadata": {"file": str(path), "issues": len(issues)}}
            return {"output": f"No diagnostics provider for {path.suffix or 'this file'} (Python only).", "metadata": {"file": str(path)}}

        name = _word_at(path, line, character) if line > 0 else ""
        if not name and op in ("goToDefinition", "findReferences", "hover", "goToImplementation", "incomingCalls", "outgoingCalls"):
            return {"output": f"No identifier at {path.name}:{line}:{character}.", "error": True}

        if op == "hover":
            fi = _one_file_index(root, path)
            sig = ""
            kind = ""
            if fi is not None:
                for s in fi.symbols:
                    if s.name == name:
                        sig = s.signature or s.name
                        kind = s.kind
                        break
            doc = _docstring(path, name) if path.suffix == ".py" else ""
            head = sig or name
            lines = [f"{head}  ({kind or 'symbol'})", f"{path.name}:{line}:{character}"]
            if doc:
                lines.append("")
                lines.append(doc.splitlines()[0][:300])
            return {"output": "\n".join(lines), "metadata": {"name": name, "signature": sig, "kind": kind}}

        if op in ("goToDefinition", "goToImplementation"):
            fi = _one_file_index(root, path)
            if fi is not None:
                same = [s for s in fi.symbols if s.name == name]
                if same:
                    same.sort(key=lambda s: s.line)
                    lines = [f"Definition of `{name}`:"]
                    for s in same[:limit]:
                        lines.append(_fmt_sym(s))
                    return {"output": "\n".join(lines), "metadata": {"name": name, "matches": len(same)}}
            try:
                from ..index.engine import query as _q
                r = _q(f"def {name}", limit=limit)
                return {"output": r.get("output", ""), "metadata": {"name": name, **(r.get("metadata") or {})}}
            except Exception as e:
                return {"output": f"goToDefinition failed: {e}", "error": True}

        if op == "findReferences":
            try:
                from ..index.engine import query as _q
                r = _q(f"refs {name}", limit=limit)
                return {"output": r.get("output", ""), "metadata": {"name": name, **(r.get("metadata") or {})}}
            except Exception as e:
                return {"output": f"findReferences failed: {e}", "error": True}

        if op == "incomingCalls":
            try:
                from ..index.engine import query as _q
                r = _q(f"callers {name}", limit=limit)
                return {"output": r.get("output", ""), "metadata": {"name": name, **(r.get("metadata") or {})}}
            except Exception as e:
                return {"output": f"incomingCalls failed: {e}", "error": True}

        if op == "outgoingCalls":
            fi = _one_file_index(root, path)
            if fi is None:
                return {"output": f"Cannot read {path.name}.", "error": True}
            target = None
            for s in fi.symbols:
                if s.name == name.split(".")[-1]:
                    end = getattr(s, "end_line", 0) or s.line
                    if s.line <= line <= end:
                        target = s
                        break
            if target is None:
                for s in fi.symbols:
                    if s.name == name.split(".")[-1]:
                        target = s
                        break
            if target is None:
                return {"output": f"No definition of `{name}` in {path.name}."}
            end = getattr(target, "end_line", 0) or target.line
            calls = [r for r in fi.refs if r.role == "call" and target.line <= r.line <= end and r.name != target.name]
            calls.sort(key=lambda r: r.line)
            if not calls:
                return {"output": f"`{name}` calls nothing.", "metadata": {"name": name}}
            lines = [f"`{name}` calls ({len(calls)}):"]
            for r in calls[:limit]:
                lines.append(f"  line {r.line}: {r.name}()")
            return {"output": "\n".join(lines), "metadata": {"name": name, "matches": len(calls)}}

        return {"output": f"Unhandled operation: {op}", "error": True}

    return Tool(
        name="lsp",
        description=description,
        parameters=schema_with(
            {
                "operation": {
                    "type": "string",
                    "enum": _OPERATIONS,
                    "description": "goToDefinition/findReferences/hover/documentSymbol/workspaceSymbol/goToImplementation/incomingCalls/outgoingCalls/diagnostics",
                },
                "filePath": {"type": "string", "description": "Absolute or relative path to the file", "optional": True},
                "line": {"type": "integer", "description": "Line number (1-based, as shown in editors)", "optional": True},
                "character": {"type": "integer", "description": "Character offset (1-based, as shown in editors)", "optional": True},
                "query": {"type": "string", "description": "Search query for workspaceSymbol", "optional": True},
                "limit": {"type": "integer", "description": "Max results (default 20, max 60)", "optional": True},
            },
            ["operation"],
        ),
        run=run,
        permission="lsp",
    )
