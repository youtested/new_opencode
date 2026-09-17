"""Local Ollama provider (OpenAI-compatible /v1/chat/completions)."""

from __future__ import annotations

from typing import Any

from .openai_compat import OpenAICompatProvider


class OllamaProvider(OpenAICompatProvider):
    def __init__(self, *, model: str = "qwen2.5-coder:7b", base_url: str = "http://localhost:11434/v1", **kwargs: Any):
        super().__init__(
            id="ollama",
            name="Ollama",
            base_url=base_url,
            api_key="ollama",
            model=model,
            is_free=True,
            **kwargs,
        )
