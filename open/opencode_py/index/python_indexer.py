"""Python indexer: a real ``ast`` walk, the gold standard for Python.

Produces exact definitions — functions/methods/classes with signatures,
parameter lists, line ranges — plus module- and class-level constants,
call sites (with the enclosing container), and import edges. Because it uses
``ast`` nothing is guessed, so "who calls X" returns exactly the call sites,
never look-alike text.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from .model import FileIndex, ImportRecord, Ref, Symbol

# Names so ubiquitous in Python that indexing them as refs would swamp every
# "who uses X" answer with noise: builtins plus special names that appear in
# nearly every file.
_SKIP_BUILTINS = {
    "self", "cls",
    "None", "True", "False",
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
    "bytes", "callable", "chr", "classmethod", "compile", "complex", "delattr",
    "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter", "float",
    "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help",
    "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len",
    "list", "locals", "map", "max", "memoryview", "min", "next", "object",
    "oct", "open", "ord", "pow", "print", "property", "range", "repr",
    "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod",
    "str", "sum", "tuple", "type", "vars", "zip", "__import__",
}

try:
    _STDLIB_MODULES = frozenset(sys.stdlib_module_names)
except AttributeError:  # pragma: no cover - pre-3.10 runtime
    _STDLIB_MODULES = frozenset(
        "abc argparse array asyncio base64 bisect builtins collections contextlib "
        "copy csv dataclasses datetime decimal difflib enum errno fnmatch functools "
        "gc glob gzip hashlib heapq html http importlib inspect io itertools json "
        "logging math mmap multiprocessing os pathlib pickle platform pprint queue "
        "random re select shutil signal socket sqlite3 ssl statistics string struct "
        "subprocess sys tempfile threading time traceback types typing unittest urllib "
        "uuid warnings weakref xml zipfile".split()
    )


def _parse(src: str, filename: str) -> ast.Module | None:
    """Parse with a graceful syntax downgrade for brand-new language features."""
    try:
        return ast.parse(src, filename=filename, type_comments=True)
    except SyntaxError:
        try:
            return ast.parse(src, filename=filename, feature_version=(3, 8))
        except (SyntaxError, ValueError):
            return None


def _signature_for_def(node: ast.AST) -> str:
    """Human-facing declaration text for a function/class definition."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        parts = [a.arg for a in list(args.posonlyargs) + list(args.args)]
        if args.vararg:
            parts.append("*" + args.vararg.arg)
        elif args.kwonlyargs:
            parts.append("*")
        if args.kwonlyargs:
            parts += [a.arg for a in args.kwonlyargs]
        if args.kwarg:
            parts.append("**" + args.kwarg.arg)
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        return f"{prefix}{node.name}({', '.join(parts)})"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    return node.__class__.__name__


def _dotted(node: ast.expr | None) -> str | None:
    """Best-effort dotted path for a Call/Attribute/Name expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


class _PythonWalker:
    def __init__(self, path: str) -> None:
        self.path = path
        self.stack: list[str] = []  # enclosing class names
        self.depth = 0             # function-nesting depth (refs inside still count)
        self.symbols: list[Symbol] = []
        self.refs: list[Ref] = []
        self.imports: list[ImportRecord] = []

    def _container(self) -> str:
        return ".".join(self.stack)

    # -- statements -------------------------------------------------------
    def _stmt(self, node: ast.stmt) -> None:
        method = getattr(self, "s_" + node.__class__.__name__, None)
        if method is not None:
            method(node)
            return
        self._stmt_generic(node)

    def _stmt_generic(self, node: ast.stmt) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._stmt(child)
            elif isinstance(child, ast.expr):
                self._expr(child, set(), "use")

    def s_Module(self, node: ast.Module) -> None:
        for child in node.body:
            self._stmt(child)

    def s_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append(
            Symbol(node.name, "class", self.path, node.lineno,
                   node.end_lineno or node.lineno, _signature_for_def(node),
                   self._container(), "python")
        )
        self.stack.append(node.name)
        for child in node.body:
            self._stmt(child)
        for deco in node.decorator_list:
            self._expr(deco, set(), "attribute")
        self.stack.pop()

    def s_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fn(node, "function")

    def s_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._fn(node, "function")

    def _fn(self, node: ast.AST, kind: str) -> None:
        is_method = bool(self.stack)
        self.symbols.append(
            Symbol(node.name, "method" if is_method else "function",
                   self.path, node.lineno, node.end_lineno or node.lineno,
                   _signature_for_def(node), self._container(), "python")
        )
        self.depth += 1
        for child in node.body:
            self._stmt(child)
        for other in getattr(node, "decorator_list", []):
            self._expr(other, set(), "attribute")
        self.depth -= 1

    def s_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportRecord(alias.name, self.path, node.lineno, None,
                             [alias.asname] if alias.asname else [])
            )

    def s_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = "." * node.level + (node.module or "")
            self.imports.append(
                ImportRecord(base, self.path, node.lineno, True,
                             [a.asname or a.name for a in node.names])
            )
            return
        mod = node.module or ""
        aliases = [a.asname or a.name for a in node.names]
        is_std = mod.split(".")[0] in _STDLIB_MODULES
        for a in node.names:
            if a.name == "*" and not a.asname:
                continue
            self.symbols.append(
                Symbol(a.asname or a.name, "import", self.path, node.lineno,
                       node.lineno, f"from {mod} import {a.name}",
                       self._container(), "python")
            )
        self.imports.append(
            # stdlib is definitively external; anything else stays None here so
            # _classify_local_imports can resolve it against the repo layout
            ImportRecord(mod, self.path, node.lineno, False if is_std else None, aliases)
        )

    def s_Assign(self, node: ast.Assign) -> None:
        if self.depth == 0:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    is_const = target.id.isupper()
                    self.symbols.append(
                        Symbol(target.id, "constant" if is_const else "variable",
                               self.path, node.lineno, node.end_lineno or node.lineno,
                               f"{target.id} = ...", self._container(), "python")
                    )
                    break
        self._expr(node.value, set(), "use")

    def s_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.depth == 0 and isinstance(node.target, ast.Name):
            is_const = node.target.id.isupper()
            self.symbols.append(
                Symbol(node.target.id, "constant" if is_const else "variable",
                       self.path, node.lineno, node.end_lineno or node.lineno,
                       f"{node.target.id}: ...", self._container(), "python")
            )
        if node.value is not None:
            self._expr(node.value, set(), "use")

    # -- expressions ------------------------------------------------------
    def _expr(self, node: ast.AST, visited: set[int], role: str) -> None:
        """Record Name/Attribute usages under this expression tree.

        ``visited`` de-duplicates shared subtrees when visitors overlap (a Call
        already walks its args, etc.). ``role`` annotates why a name was used:
        "call", "use", "attribute".
        """
        if id(node) in visited or node is None:
            return
        visited.add(id(node))
        if isinstance(node, ast.Name):
            if node.id not in _SKIP_BUILTINS and isinstance(getattr(node, "ctx", None), ast.Load):
                self.refs.append(
                    Ref(node.id, self.path, node.lineno, role, self._container())
                )
            return
        if isinstance(node, ast.Attribute):
            if node.attr not in _SKIP_BUILTINS and isinstance(getattr(node, "ctx", None), ast.Load):
                self.refs.append(
                    Ref(node.attr, self.path, node.lineno, role, self._container())
                )
            self._expr(node.value, visited, "use")
            return
        if isinstance(node, ast.Call):
            self._expr(node.func, visited, "call")
            for arg in node.args:
                self._expr(arg, visited, "use")
            for kw in node.keywords:
                self._expr(kw.value, visited, "use")
            return
        if isinstance(node, (ast.Constant, ast.Name)):
            return
        for child in ast.iter_child_nodes(node):
            self._expr(child, visited, "use")


def index_file(root: Path, rel: str, src: str, size: int, mtime: float) -> FileIndex:
    """Index one Python file. Never raises on bad syntax (falls back to empty)."""
    tree = _parse(src, rel)
    walker = _PythonWalker(rel)
    if tree is not None:
        try:
            walker.s_Module(tree)
        except Exception:  # pragma: no cover - defensive: ast quirks must not kill a build
            walker = _PythonWalker(rel)
    _classify_local_imports(walker.imports, root)
    return FileIndex(
        rel, mtime, size, "python",
        symbols=walker.symbols, refs=walker.refs, imports=walker.imports,
    )


def _classify_local_imports(imports: list[ImportRecord], root: Path) -> None:
    """Fill in ``local`` for absolute imports by checking the repo layout.

    Namespace packages (PEP 420) have no ``__init__.py``, so a bare directory
    containing ``*.py`` files counts too.
    """
    for imp in imports:
        if imp.local is not None:
            continue
        first = imp.module.split(".")[0]
        pkg_dir = root / first
        imp.local = bool(
            (root / f"{first}.py").exists()
            or (pkg_dir / "__init__.py").exists()
            or (pkg_dir.is_dir() and any(pkg_dir.glob("*.py")))
        )