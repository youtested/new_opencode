"""API key storage for opencode_py.

Precedence: environment variable > config providers.<id>.api_key > auth.json (0600).

Never logs or prints keys.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from .globals import Path as GPath

# env var names per provider id (first one set wins), matching models.dev `env`
_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "opencode": ("OPENCODE_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
    "togetherai": ("TOGETHER_API_KEY",),
    "github": ("GITHUB_TOKEN",),
    "sambanova": ("SAMBANOVA_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "deepinfra": ("DEEPINFRA_API_KEY",),
    "elevenlabs": ("ELEVENLABS_API_KEY",),
}


class Auth:
    def __init__(self, auth_file: Path | None = None):
        self._file = auth_file or GPath.auth_file()
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        env_override = os.environ.get("OPENCODE_AUTH_CONTENT")
        if env_override:
            try:
                self._data = json.loads(env_override)
                return
            except json.JSONDecodeError:
                pass
        try:
            self._data = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def _save(self) -> None:
        import tempfile
        self._file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=self._file.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(self._data, indent=2))
            os.replace(tmp_path, self._file)
        except OSError:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        try:
            os.chmod(self._file, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:
            pass

    def get(self, provider_id: str, config: dict | None = None) -> str | None:
        """Resolve an API key for a provider id (env > config > auth.json)."""
        for env_name in _ENV_KEYS.get(provider_id, ()):
            val = os.environ.get(env_name)
            if val:
                return val
        if config:
            entry = config.get(provider_id)
            if isinstance(entry, dict):
                for key_name in ("api_key", "apiKey", "key"):
                    key = entry.get(key_name)
                    if key:
                        return str(key)
        entry = self._data.get(provider_id)
        if isinstance(entry, dict):
            key = entry.get("key")
            if key:
                return str(key)
        return None

    def set(self, provider_id: str, key: str) -> None:
        self._data[provider_id] = {"type": "api", "key": key}
        self._save()

    def remove(self, provider_id: str) -> None:
        self._data.pop(provider_id, None)
        self._save()

    def has(self, provider_id: str) -> bool:
        return self.get(provider_id) is not None

    def list(self) -> list[str]:
        return list(self._data.keys())

    def load_json(self) -> dict:
        return dict(self._data)


def load_auth() -> Auth:
    return Auth()
