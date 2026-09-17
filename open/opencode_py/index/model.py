"""Data model for the symbol index.

IndexEntry records are intentionally small and JSON-serializable so the whole
index for a repo can be loaded/saved in one shot and random-access lookups stay
in-memory dict walks instead of disk queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _coerce_mtime(value: Any) -> int:
    """Cache mtime -> int nanoseconds, accepting old + new formats.

    New caches store int ns (exact). Old caches stored float ns (precision
    loss from JSON round-trip: 1789483227335323545 -> 1.7894832273353236e+18).
    A float is accepted when it round-trips within 1ms of an int; anything
    else (float seconds, garbage) becomes 0 = always re-parse that file.
    Never raises.
    """
    try:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value if value >= 0 else 0
        f = float(value or 0)
        if f <= 0:
            return 0
        n = int(round(f))
        # float ns loses ~100s of precision: accept within 1ms, else stale
        if abs(n - f) <= 1e6:
            return n if n >= 0 else 0
        # maybe float seconds (small): convert
        if f < 1e12:
            return int(round(f * 1e9))
        return 0
    except (TypeError, ValueError):
        return 0


@dataclass
class Symbol:
    """A *definition* of a name in the codebase.

    ``kind`` is a stable token (function/method/class/variable/constant/
    module/interface/struct/enum/trait/type/macro/enum_member/import/package/
    arg/unknown). ``container`` is the dotted scope the symbol lives in
    (e.g. ``"SessionStore"`` for a method inside a Python class), ``signature``
    is the human-readable declaration line (``def run()`` / ``fn main()``).
    """

    name: str
    kind: str
    file: str
    line: int
    end_line: int
    signature: str = ""
    container: str = ""
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "signature": self.signature,
            "container": self.container,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Symbol":
        return cls(
            name=str(d.get("name", "")),
            kind=str(d.get("kind", "unknown")),
            file=str(d.get("file", "")),
            line=int(d.get("line", 0)),
            end_line=int(d.get("end_line", 0)),
            signature=str(d.get("signature", "")),
            container=str(d.get("container", "")),
            language=str(d.get("language", "")),
        )


@dataclass
class Ref:
    """A *usage* of a name (a call, a plain read, an attribute access…).

    ``role`` distinguishes why the name was recorded: ``"call"`` (invoked as a
    function/method), ``"use"`` (identifier read), ``"import"`` or
    ``"attribute"``. Container is the enclosing function/class at that site.

    Compact form: to_tuple/from_tuple store [name, file, line, role, container]
    positionally — ~40% smaller JSON than dicts for 46K refs. from_dict still
    reads old caches.
    """

    name: str
    file: str
    line: int
    role: str = "use"
    container: str = ""

    def to_tuple(self) -> list:
        # omit defaults: role "use" and empty container collapse to short rows
        if self.role == "use" and not self.container:
            return [self.name, self.file, self.line]
        if not self.container:
            return [self.name, self.file, self.line, self.role]
        return [self.name, self.file, self.line, self.role, self.container]

    @classmethod
    def from_tuple(cls, t: list | tuple) -> "Ref":
        try:
            name = str(t[0])
            file = str(t[1])
            line = int(t[2])
            role = str(t[3]) if len(t) > 3 else "use"
            container = str(t[4]) if len(t) > 4 else ""
        except (IndexError, TypeError, ValueError):
            return cls(name="", file="", line=0)
        return cls(name=name, file=file, line=line, role=role, container=container)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "role": self.role,
            "container": self.container,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Ref":
        return cls(
            name=str(d.get("name", "")),
            file=str(d.get("file", "")),
            line=int(d.get("line", 0)),
            role=str(d.get("role", "use")),
            container=str(d.get("container", "")),
        )


@dataclass
class ImportRecord:
    """One top-level import edge of a file.

    ``module`` is the dependency as written (dotted name, file stem, or quoted
    relative path); ``local`` marks whether it resolved to a file in the same
    root (True), the stdlib (False), or is unknown (None). ``aliases`` lists
    names bound to it (``import x as y`` -> aliases=["y"]).
    """

    module: str
    file: str
    line: int
    local: bool | None = None
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "file": self.file,
            "line": self.line,
            "local": self.local,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImportRecord":
        aliases = d.get("aliases") or []
        return cls(
            module=str(d.get("module", "")),
            file=str(d.get("file", "")),
            line=int(d.get("line", 0)),
            local=d.get("local"),
            aliases=[str(a) for a in aliases],
        )


@dataclass
class FileIndex:
    """Per-file extraction. Everything is relative to the index root."""

    path: str  # path relative to the index root
    # ponytail: int nanoseconds (st_mtime_ns), NOT float seconds — a float
    # can't hold 19-digit ns exactly, so every cache load mismatched and the
    # whole repo re-parsed on every fresh process. int round-trips exactly.
    mtime: int
    size: int
    language: str = ""
    symbols: list[Symbol] = field(default_factory=list)
    # Raw ref rows (compact tuples from cache, or Ref objects fresh from the
    # indexer). The .refs property builds objects on first touch — def /
    # symbols / imports queries never pay the 46K-object cost. ponytail: one
    # property + one private slot, no new class.
    _refs_raw: list = field(default_factory=list, repr=False)
    imports: list[ImportRecord] = field(default_factory=list)
    content_hash: str = ""

    def __init__(self, path: str = "", mtime: int = 0, size: int = 0,
                 language: str = "", symbols: list | None = None,
                 refs: list | None = None, imports: list | None = None,
                 content_hash: str = "", **_kw: object) -> None:
        # Custom init (replaces the dataclass one): accepts the legacy
        # refs= kwarg AND the _refs_raw slot, so fresh indexer code
        # (refs=[Ref...]) and cache loads both work. Extra keys ignored
        # so old/new payloads never crash each other.
        self.path = path
        self.mtime = mtime
        self.size = size
        self.language = language
        self.symbols = list(symbols) if symbols is not None else []
        self.__dict__["_refs_raw"] = list(refs) if refs is not None else []
        self.imports = list(imports) if imports is not None else []
        self.content_hash = content_hash
        for _k, _v in _kw.items():
            if _k == "_refs_raw" and isinstance(_v, list):
                self.__dict__["_refs_raw"] = list(_v)

    @property
    def refs(self) -> list[Ref]:
        raw = self.__dict__.get("_refs_raw")
        if raw is None:
            return []
        if raw and not isinstance(raw[0], Ref):
            try:
                conv = []
                for r in raw:
                    try:
                        if isinstance(r, (list, tuple)):
                            conv.append(Ref.from_tuple(r))
                        elif isinstance(r, dict):
                            conv.append(Ref.from_dict(r))
                        elif isinstance(r, Ref):
                            conv.append(r)
                    except (TypeError, ValueError):
                        continue
                self.__dict__["_refs_raw"] = conv
                return conv
            except Exception:
                return []
        return raw

    @refs.setter
    def refs(self, value: list) -> None:
        self.__dict__["_refs_raw"] = list(value) if value is not None else []

    def to_dict(self) -> dict[str, Any]:
        try:
            raw = self.__dict__.get("_refs_raw") or []
            refs_out = []
            for r in raw:
                try:
                    if isinstance(r, Ref):
                        refs_out.append(r.to_tuple())
                    elif isinstance(r, (list, tuple)):
                        refs_out.append(list(r))
                    elif isinstance(r, dict):
                        refs_out.append(r)
                except (TypeError, ValueError):
                    continue
        except Exception:
            refs_out = []
        return {
            "path": self.path,
            "mtime": self.mtime,
            "size": self.size,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            # compact tuples: 46K refs were 89% of the 6.3MB cache
            "refs": refs_out,
            "imports": [i.to_dict() for i in self.imports],
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileIndex":
        h = d.get("content_hash", "")
        # Keep RAW rows (no 46K-object build): the .refs property converts
        # on first touch. Accepts tuples (v3), dicts (v1/v2), Ref objects.
        raw_refs = list(d.get("refs", []) or [])
        inst = cls(
            path=str(d.get("path", "")),
            mtime=_coerce_mtime(d.get("mtime", 0)),
            size=int(d.get("size", 0) or 0),
            language=str(d.get("language", "")),
            symbols=[Symbol.from_dict(s) for s in d.get("symbols", [])],
            imports=[ImportRecord.from_dict(i) for i in d.get("imports", [])],
            content_hash=str(h or ""),
        )
        inst.__dict__["_refs_raw"] = raw_refs
        return inst


# ---------------------------------------------------------------------------
# Language registry helpers used by both the heuristic indexer and the query
# engine (for picking the right backend for a given file path).
# ---------------------------------------------------------------------------

# canonical identifier patterns, shared by the heuristic indexers
NAME_RE = "([A-Za-z_][A-Za-z0-9_$]*)"
DOTTED_RE = r"([A-Za-z_][A-Za-z0-9_$-]*(?:\.[A-Za-z0-9_-]+)*)"

# extensions -> language id (an entry in LANGUAGES)
EXTENSION_LANGS: dict[str, str] = {
    # Python (handled by the ast indexer; listed so language lookups agree)
    ".py": "python",
    ".pyi": "python",
    # JavaScript / TypeScript family
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "typescript",
    # C / C++
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cu": "cpp",
    ".cuh": "cpp",
    # JVM family
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".groovy": "groovy",
    # Go, Rust, C#, Swift, PHP
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".swift": "swift",
    ".php": "php",
    # Shell family
    ".sh": "bash",
    ".bash": "bash",
    ".bats": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    # Lua / Ruby / Perl / R / SQL
    ".lua": "lua",
    ".rb": "ruby",
    ".rake": "ruby",
    ".pl": "perl",
    ".pm": "perl",
    ".r": "r",
    ".sql": "sql",
    # Web/doc-ish
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".less": "css",
    ".vue": "javascript",
    ".svelte": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".markdown": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
}


def language_for(path: str) -> str:
    """Language id for a path, lowercased, or "" if unknown."""
    key = path.lower()
    idx = key.rfind(".")
    if idx < 0:
        # dotfiles like `.bashrc` still index as shell
        base = key.rsplit("/", 1)[-1]
        if base in (".bashrc", ".zshrc", ".profile", ".bash_profile"):
            return "bash"
        return ""
    ext = key[idx:]
    return EXTENSION_LANGS.get(ext, "")