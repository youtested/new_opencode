"""summarize_file tool: one file's shape in one shot (outline + preview).

RULES (full guidance for maintainers; the sent schema is short):
- For large files read truncates: get classes/functions/headings/keys +
  line numbers here, then read exact windows. Single file only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .registry import Tool, schema_with

MAX_PREVIEW_LINES = 20
MAX_LINE_CHARS = 200
MAX_OUTPUT = 8 * 1024  # 8 KB cap on the whole summary

BINARY_BLOCKERS = {".png", ".jpeg", ".jpg", ".gif", ".webp", ".pdf", ".doc", ".docx", ".xls", ".xlsx"}
TEXT_PREVIEW_EXTENSIONS = {".py", ".pyi", ".md", ".txt", ".rst", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".js", ".ts", ".css", ".html", ".xml", ".csv", ".diff", ".log"}


def _suggest(path: Path) -> str | None:
    try:
        candidates = [p for p in path.parent.iterdir() if p.is_file()]
    except OSError:
        return None
    name = path.name
    matches = [p.name for p in candidates if p.stem == name or name in p.stem or p.stem in name]
    return ", ".join(matches[:3]) or None


def _error(path: Path, msg: str) -> dict:
    out = f"Could not summarize {path}: {msg}"
    suggestion = _suggest(path)
    if suggestion:
        out += f"\n\nDid you mean one of these?\n{suggestion}"
    return {"output": out, "error": True}


def _is_binary(sample: bytes) -> bool:
    if not sample:
        return False
    sample = sample[:1024]
    # Same window-truncation guard as tools/read.py: a multibyte char cut in
    # half by the 1024-byte slice is "unexpected end of data", not binary.
    for trim in range(4):
        try:
            sample[: len(sample) - trim].decode("utf-8")
            break
        except UnicodeDecodeError:
            continue
    else:
        return True
    nonprintable = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return nonprintable / len(sample) > 0.30


def _first_docstring(node: Any) -> str:
    """One-line summary from a node's docstring (rstrip-multiline, trimmed)."""
    import ast

    try:
        doc = ast.get_docstring(node, clean=True) or ""
    except (TypeError, ValueError):
        doc = ""
    if not doc:
        return ""
    first = doc.strip().splitlines()[0].strip()
    first = first.replace("`", "").strip()
    if len(first) > 100:
        first = first[:100] + "…"
    return first


def _py_outline(source: str, path: Path) -> list[str]:
    try:
        import ast

        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"(could not parse as Python: {e})"]
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            doc = _first_docstring(node)
            with_doc = f" - {doc}" if doc else ""
            lines.append(f"class {node.name} (line {node.lineno}, {len(methods)} methods){with_doc}")
            for m in methods[:8]:
                mdoc = _first_docstring(m)
                with_mdoc = f" - {mdoc}" if mdoc else ""
                lines.append(f"  def {m.name} (line {m.lineno}){with_mdoc}")
            if len(methods) > 8:
                lines.append(f"  ... and {len(methods) - 8} more methods")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = _first_docstring(node)
            with_doc = f" - {doc}" if doc else ""
            lines.append(f"def {node.name} (line {node.lineno}){with_doc}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ", ".join(getattr(a, "name", "") or "" for a in node.names[:6])
            lines.append(f"import {names}")
    if not lines:
        lines.append("(no top-level classes or functions found)")
    return lines


def _markdown_outline(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        stripped = ln.rstrip()
        if re.match(r"^\s*#{1,6}\s", stripped):
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append("  " * (level - 1) + stripped.strip())
        elif re.match(r"^\s*[-*]\s", stripped):
            out.append("  * " + stripped.lstrip(" -*")[:80])
    return out[:80] or ["(no headings found)"]


def _json_outline(source: str) -> list[str]:
    try:
        data = json.loads(source)
    except ValueError:
        # Multi-record JSON: count objects
        objs = len(re.findall(r"{", source))
        return [f"(not a single JSON value; roughly {objs} open braces)"]
    if isinstance(data, dict):
        keys = list(data.keys())
        prefix = " ".join(f"{k}: {type(data[k]).__name__}" for k in keys[:12])
        more = f" (+{len(keys) - 12} more keys)" if len(keys) > 12 else ""
        return [f"object with {len(keys)} keys: {prefix}{more}"]
    if isinstance(data, list):
        kinds: dict[str, int] = {}
        for item in data[:500]:
            kinds[type(item).__name__] = kinds.get(type(item).__name__, 0) + 1
        preview = "".join(f"{k}×{v} " for k, v in list(kinds.items())[:6])
        return [f"array with {len(data)} items ({preview.strip()})"]
    return [f"{type(data).__name__}: {str(data)[:120]}"]


def _xml_outline(source: str) -> list[str]:
    tags: dict[str, int] = {}
    for m in re.finditer(r"<\s*([/!?]?)([A-Za-z][\w.:-]*)", source):
        mark = m.group(1)
        name = m.group(2)
        if mark == "":
            tags[name] = tags.get(name, 0) + 1
    if not tags:
        return ["(no tags found)"]
    top = sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    return [f"{name} ×{count}" for name, count in top]


def _text_outline(lines: list[str]) -> list[str]:
    census: dict[str, int] = {}
    for pat in (r"\b(def|function)\b", r"\bclass\b", r"\bimport\b", r"\breturn\b", r"if\b"):
        census[pat.strip("\\b()| ")] = len(re.findall(pat, " ".join(lines)))
    nonempty = [l for l in lines if l.strip()]
    finger = "; ".join(f"{k}: {v}" for k, v in census.items())
    if nonempty:
        first = [l.strip()[:80] for l in nonempty[:3]]
        head = f" | starts with: {' / '.join(first)}"
    else:
        head = ""
    return [f"keyword census ({finger}){head}"]


def _summarize(filePath: str) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        return _error(path, "path does not exist.")
    if path.is_dir():
        return _error(path, "it is a directory — summarize_file reads one file at a time.")

    suffix = path.suffix.lower()
    if suffix in BINARY_BLOCKERS or suffix == "":
        try:
            sample = path.read_bytes()[:1024]
        except OSError as e:
            return _error(path, str(e))
        if _is_binary(sample) or suffix in BINARY_BLOCKERS:
            return _error(
                path,
                "it looks like a binary/image/PDF file — use the read tool "
                "for files that are shown as attachments.",
            )
    else:
        try:
            sample = path.read_bytes()[:1024]
        except OSError as e:
            return _error(path, str(e))
        if _is_binary(sample):
            return _error(
                path,
                "it looks like a binary file — summarize_file is for text files only.",
            )

    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return _error(path, f"cannot read as text: {e}")

    lines = source.splitlines()
    total = len(lines)
    size = path.stat().st_size
    size_s = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"

    if suffix in {".py", ".pyi"}:
        kind = "python"
        outline = _py_outline(source, path)
    elif suffix in {".md", ".txt", ".rst", ".log", ".diff"}:
        kind = "markdown" if suffix in {".md", ".rst"} else "text"
        outline = _markdown_outline(lines) if kind == "markdown" else _text_outline(lines)
    elif suffix in {".json", ".jsonl"}:
        kind = "json"
        outline = _json_outline(source)
    elif suffix in {".html", ".xml"}:
        kind = "xml"
        outline = _xml_outline(source)
    else:
        kind = "text"
        outline = _text_outline(lines)

    previewed = [l for l in lines if l.strip()][:MAX_PREVIEW_LINES]
    preview = "\n".join(
        l if len(l) <= MAX_LINE_CHARS else l[: MAX_LINE_CHARS - 1] + "…"
        for l in previewed
    )

    body = f"<{path}>\n<type>{kind}</type>\n<size>{size_s} | {total} lines</size>"
    body += f"\n\nSTRUCTURE\n" + "\n".join("  " + o for o in outline)
    if preview:
        body += f"\n\nPREVIEW (first {len(previewed)} non-empty lines)\n" + "\n".join(
            f"  {l[:MAX_LINE_CHARS]}" for l in previewed
        )
    if len(body) > MAX_OUTPUT:
        body = body[:MAX_OUTPUT] + "\n… (summary capped at 8 KB)"

    return {
        "output": body,
        "metadata": {"loaded": [str(path)]},
    }


def tool() -> Tool:
    description = """One file's outline + preview (classes/functions/headings + line numbers). MANDATORY first step for files over ~150 lines: pick the anchor, then read symbol='<name>' or the exact offset/limit — never page blindly."""

    def run(input: dict) -> dict:
        return _summarize(str(input["filePath"]))

    return Tool(
        name="summarize_file",
        description=description,
        parameters=schema_with(
            {
                "filePath": {
                    "type": "string",
                    "description": "Absolute file path",
                }
            },
            ["filePath"],
        ),
        run=run,
        permission="read",
    )