"""Global paths, version, data dir, and config discovery helpers.

Mirrors opencode's Global.Path / config discovery behavior:
  - config:  ~/.config/opencode_py  (or $XDG_CONFIG_HOME/opencode_py)
  - data:    ~/.local/share/opencode_py  (or $XDG_DATA_HOME/opencode_py)  auth.json + sessions
  - cache:   ~/.cache/opencode_py  (or $XDG_CACHE_HOME/opencode_py)  models.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "VERSION",
    "Path",
    "GLOBAL_DIRS",
    "HOME",
    "Platform",
]

VERSION = "0.1.0"


def _xdg_dir(env: str, default: str) -> Path:
    """Resolve an XDG base dir without importing platformdirs (~0.25s cold)."""
    override = os.environ.get(env)
    if override:
        return Path(override)
    return Path.home() / default


HOME = Path.home()


def _user_config_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return _xdg_dir("XDG_CONFIG_HOME", ".config")


def _user_data_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return _xdg_dir("XDG_DATA_HOME", ".local/share")


def _user_cache_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return _xdg_dir("XDG_CACHE_HOME", ".cache")


class Path:
    config = _user_config_dir() / "opencode_py"
    data = _user_data_dir() / "opencode_py"
    cache = _user_cache_dir() / "opencode_py"
    tmp = Path("/tmp")

    @classmethod
    def init(cls) -> None:
        for d in (cls.config, cls.data, cls.cache):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def auth_file(cls) -> Path:
        return cls.data / "auth.json"

    @classmethod
    def sessions_dir(cls) -> Path:
        return cls.data / "sessions"

    @classmethod
    def models_file(cls) -> Path:
        return cls.cache / "models.json"

    @classmethod
    def catalog_file(cls) -> Path:
        return cls.cache / "models-catalog.json"

    @classmethod
    def truncation_dir(cls) -> Path:
        return cls.data / "truncation"


class Platform:
    @staticmethod
    def name() -> str:
        return sys.platform

    @staticmethod
    def is_windows() -> bool:
        return sys.platform == "win32"


GLOBAL_DIRS = [
    Path.config,
]


def freeze() -> None:
    """Stop the cyclic GC from re-scanning already-imported modules.

    Long-lived sessions (the TUI especially) on low-RAM armv7 phones pay a GC
    tax on every collection over the tens of thousands of already-imported
    objects. Collecting then freezing them after startup keeps peak memory
    and pause time steady; anything imported later (httpx, provider backends)
    is allocated after the freeze and stays fully collectable.
    """
    import gc

    try:
        gc.collect()
        gc.freeze()
    except Exception:  # pragma: no cover - exotic runtimes
        pass


def resolve_worktree(directory: Path) -> Path:
    """Walk up from directory to find the git worktree root (else the dir itself)."""
    d = directory.resolve()
    if (d / ".git").exists():
        return d
    for parent in d.parents:
        if (parent / ".git").exists():
            return parent
    return d
