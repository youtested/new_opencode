from __future__ import annotations

from typing import Any

FREE_PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "env": ("GROQ_API_KEY",)},
    "cerebras": {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "env": ("CEREBRAS_API_KEY",)},
    "google": {
        "name": "Google AI Studio",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "env": ("OPENROUTER_API_KEY",),
        "headers": {"HTTP-Referer": "https://opencode.ai/", "X-Title": "opencode"},
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env": ("NVIDIA_API_KEY",),
        "headers": {"HTTP-Referer": "https://opencode.ai/", "X-Title": "opencode"},
    },
    "mistral": {"name": "Mistral", "base_url": "https://api.mistral.ai/v1", "env": ("MISTRAL_API_KEY",)},
    "github": {"name": "GitHub Models", "base_url": "https://models.github.ai/inference", "env": ("GITHUB_TOKEN",)},
    "sambanova": {"name": "SambaNova", "base_url": "https://api.sambanova.ai/v1", "env": ("SAMBANOVA_API_KEY",)},
    "togetherai": {"name": "Together", "base_url": "https://api.together.xyz/v1", "env": ("TOGETHER_API_KEY",)},
}

FREE_DEFAULT_MODELS: dict[str, str] = {
    "zen": "muse-spark-1.3-contributor-free",
    "groq": "llama-3.3-70b-versatile",
    "cerebras": "llama-3.3-70b",
    "google": "gemini-2.5-flash",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia": "nemotron-3-ultra-free",
    "mistral": "codestral-latest",
    "github": "gpt-4o-mini",
    "sambanova": "Meta-Llama-3.3-70B-Instruct",
    "togetherai": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

PAID_PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "env": ("ANTHROPIC_API_KEY",),
        "api_kind": "anthropic",
    },
    "openai": {"name": "OpenAI", "base_url": "https://api.openai.com/v1", "env": ("OPENAI_API_KEY",)},
    "deepseek": {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "env": ("DEEPSEEK_API_KEY",)},
    "xai": {"name": "xAI", "base_url": "https://api.x.ai/v1", "env": ("XAI_API_KEY",)},
    "deepinfra": {"name": "DeepInfra", "base_url": "https://api.deepinfra.com/v1/openai", "env": ("DEEPINFRA_API_KEY",)},
}
