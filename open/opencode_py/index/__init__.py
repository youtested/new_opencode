"""Code symbol index for the find_symbols tool.

Two-tier extraction, best of what a pure-Python / slow-hardware budget allows:

- Python files get a real ``ast`` walk: true definitions (functions, methods,
  classes, module constants), call sites, and import edges. This is exact, not
  guessed.
- Everything else gets language-aware line heuristics (JS/TS, C/C++, Java, Go,
  Rust, shell, ...): declaration patterns, import patterns and identifier
  usage. When ``universal-ctags`` is installed it is preferred for definitions
  across all of those languages (one binary, ~100 languages).

The extracted records are cached on disk per worktree and refreshed
incrementally by (mtime, size), so repeat queries are fast and a stale index
the moment a file changes is impossible.
"""

from __future__ import annotations

from .engine import IndexEngine, query as run_query

__all__ = ["IndexEngine", "run_query"]