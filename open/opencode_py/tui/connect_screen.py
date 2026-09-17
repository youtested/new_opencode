"""Connect screen (/connect): pick a provider, paste an API key, save it.

Keys are stored by Auth (auth.json, chmod 0600) or used via env vars. The
screen lists the providers opencode_py supports plus where to get each key.
"""

from __future__ import annotations

from typing import Any, Callable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from ..auth import Auth

# (provider id, display name, where to get a key)
PROVIDERS: list[tuple[str, str, str]] = [
    ("opencode", "OpenCode Zen", "https://opencode.ai/auth"),
    ("groq", "Groq", "https://console.groq.com/keys"),
    ("cerebras", "Cerebras", "https://cloud.cerebras.ai/"),
    ("google", "Google AI Studio (Gemini)", "https://aistudio.google.com/apikey"),
    ("openrouter", "OpenRouter", "https://openrouter.ai/keys"),
    ("nvidia", "NVIDIA NIM", "https://build.nvidia.com/"),
    ("mistral", "Mistral", "https://console.mistral.ai/"),
    ("github", "GitHub Models", "https://github.com/settings/tokens"),
    ("sambanova", "SambaNova", "https://cloud.sambanova.ai/"),
    ("togetherai", "Together", "https://api.together.ai/"),
    ("anthropic", "Anthropic", "https://console.anthropic.com/"),
    ("openai", "OpenAI", "https://platform.openai.com/api-keys"),
    ("deepseek", "DeepSeek", "https://platform.deepseek.com/api_keys"),
    ("elevenlabs", "ElevenLabs TTS (voice cloning)", "https://elevenlabs.io/app/settings/api-keys"),
    ("ollama", "Ollama (local)", "no key required — runs on localhost:11434"),
]


class ConnectScreen(ModalScreen[str]):
    """Modal to add/update a provider API key.

    Dismisses with the provider id on success, or None on cancel.
    """

    CSS = """
    ConnectScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }
    #connect-box {
        width: 64;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $panel;
        border: round $border;
    }
    #connect-box .dialog-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 0;
    }
    #connect-box .dialog-desc {
        color: $text-muted;
        margin-bottom: 1;
    }
    #connect-providers {
        height: 10;
        border: solid $panel-lighten-2;
        margin-bottom: 1;
    }
    #connect-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    #connect-key {
        margin-bottom: 1;
    }
    #connect-buttons {
        height: 3;
        align-horizontal: right;
    }
    """

    def __init__(
        self,
        auth: Auth,
        on_connected: Callable[[str], None] | None = None,
        initial: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.auth = auth
        self.on_connected = on_connected
        self._provider_id = ""
        self._provider_url = ""
        self._initial = initial

    def compose(self) -> ComposeResult:
        with Vertical(id="connect-box"):
            yield Label("Connect a provider", classes="dialog-title")
            yield Static(
                "Pick a provider, paste your API key, then press Enter.",
                classes="dialog-desc",
            )
            yield ListView(
                *[ListItem(Label(f"{name}  ({pid})")) for pid, name, _ in PROVIDERS],
                id="connect-providers",
            )
            yield Static("", id="connect-hint")
            yield Label("API key", classes="dialog-title")
            yield Input(placeholder="sk-...", id="connect-key", password=True)
            with Horizontal(id="connect-buttons"):
                yield Button("Save", id="connect-save", variant="primary")
                yield Button("Cancel", id="connect-cancel", variant="default")

    def on_mount(self) -> None:
        if not self._initial:
            return
        ids = [pid for pid, _name, _url in PROVIDERS]
        if self._initial in ids:
            self.query_one("#connect-providers", ListView).index = ids.index(self._initial)
            self._preselect(ids.index(self._initial))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.index is None or not (0 <= event.index < len(PROVIDERS)):
            return
        self._preselect(event.index)

    def _preselect(self, index: int) -> None:
        self._provider_id, _name, self._provider_url = PROVIDERS[index]
        self.query_one("#connect-hint", Static).update(f"Get a key: {self._provider_url}")
        self.query_one("#connect-key", Input).focus()

    def _save(self) -> None:
        key = self.query_one("#connect-key", Input).value.strip()
        if not self._provider_id:
            self.notify("Select a provider first.")
            return
        if not key:
            self.notify("Paste your API key.")
            return
        self.auth.set(self._provider_id, key)
        if self.on_connected:
            self.on_connected(self._provider_id)
        self.dismiss(self._provider_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect-save":
            self._save()
        elif event.button.id == "connect-cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "connect-key":
            self._save()

    def on_key(self, event: Any) -> None:
        from textual.events import Key

        if isinstance(event, Key) and event.key == "escape":
            self.dismiss(None)
            event.stop()
