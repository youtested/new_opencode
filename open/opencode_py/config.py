"""Config loading / deep-merge for opencode_py.

Mirrors opencode's config model (ConfigV1.Info) without pydantic:
  - Files: opencode.json / opencode.jsonc (JSONC = JSON with comments/trailing commas)
    discovered from the project dir (walking up to worktree) then the user-level
    config dir. Later sources override earlier (deep merge).
  - Env overrides: OPENCODE_CONFIG (file), OPENCODE_CONFIG_CONTENT (raw JSON),
    OPENCODE_PERMISSION (JSON merged into permission).
  - Config variable substitution: {env:VAR}, {file:path}.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .globals import Path as GPath
from .util.ram import is_low_ram

_VAR_RE = re.compile(r"\{env:([A-Za-z0-9_]+)\}|\{file:([^}]+)\}")
_ALLOWED_ENV_VARS = {"OPENCODE_API_KEY", "OPENCODE_BASE_URL", "OPENCODE_MODEL", "OPENCODE_PROVIDER", "OPENCODE_SMALL_MODEL"}
_FILE_VAR_MAX_BYTES = 4096


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(_strip_jsonc(text))


def load_json_with_fallback(path: Path) -> tuple[dict | None, str]:
    """Load a config file with crash recovery: primary → .bak → .tmp.

    Returns (data, source) where source is 'primary'/'bak'/'tmp'/'missing'.
    None data means no usable copy exists (caller uses defaults, but must
    NOT overwrite the broken files with those defaults — see save_config).
    """
    for suffix, source in (("", "primary"), (".bak", "bak"), (".tmp", "tmp")):
        p = path if not suffix else Path(str(path) + suffix)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if not text.strip():
            continue
        try:
            data = json.loads(_strip_jsonc(text))
        except ValueError:
            continue
        if isinstance(data, dict):
            return data, source
    return None, "missing"


def _strip_jsonc(text: str) -> str:
    """JSONC -> JSON: strip // and /* */ comments (keeps strings intact).

    Handles nested block comments and treats string escapes (including \\uXXXX
    unicode escapes) as opaque so a comment marker can never appear from inside
    an escape sequence.
    """
    out = []
    i = 0
    in_str = False
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                nxt = text[i + 1]
                out.append(nxt)
                i += 2
                # skip the rest of a \uXXXX unicode escape atomically so its
                # hex digits can never be confused with comment markers
                if nxt == "u" and i + 4 <= n:
                    for _ in range(4):
                        out.append(text[i])
                        i += 1
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                depth = 1
                i += 2
                while depth > 0 and i < n:
                    if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                        depth -= 1
                        i += 2
                    elif text[i] == "/" and i + 1 < n and text[i + 1] == "*":
                        depth += 1
                        i += 2
                    else:
                        i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (returns a new dict). Lists are replaced."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def substitute_vars(text: str, base_dir: Path) -> str:
    def repl(m: re.Match) -> str:
        env = m.group(1)
        if env:
            if env not in _ALLOWED_ENV_VARS:
                return ""
            return os.environ.get(env, "")[:_FILE_VAR_MAX_BYTES]
        fpath = m.group(2)
        p = Path(fpath).expanduser()
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        else:
            p = p.resolve()
        try:
            base_res = base_dir.resolve()
            p.relative_to(base_res)
        except ValueError:
            return ""
        try:
            data = p.read_text(encoding="utf-8")
        except OSError:
            return ""
        if len(data) > _FILE_VAR_MAX_BYTES:
            data = data[:_FILE_VAR_MAX_BYTES]
        return data
    return _VAR_RE.sub(repl, text)


def _apply_vars(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        return substitute_vars(value, base_dir)
    if isinstance(value, list):
        return [_apply_vars(v, base_dir) for v in value]
    if isinstance(value, dict):
        return {k: _apply_vars(v, base_dir) for k, v in value.items()}
    return value


DEFAULT_MODEL = "muse-spark-1.3-contributor-free"
DEFAULT_SMALL_MODEL = "mimo-v2.5-free"


def _usage_path() -> Path:
    return GPath.data / "model-usage.json"


_USAGE_LOCK = threading.Lock()


def _safe_count(v: object) -> int:
    try:
        n = int(v or 0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            n = int(float(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
    return n if n > 0 else 0


def record_model_use(provider: str, model: str) -> None:
    """Count one answered turn for (provider, model).

    Best-effort, never raises: powers the most-used default (see
    most_used_model). Corrupt cache degrades to a fresh count.
    Thread-safe: a process-wide lock serializes the read-modify-write so
    parallel turns never lose counts or corrupt the JSON.
    """
    try:
        provider = str(provider or "").strip() or "opencode"
        model = str(model or "").strip()
        if not model:
            return
        with _USAGE_LOCK:
            try:
                data = json.loads(_usage_path().read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except (OSError, ValueError):
                data = {}
            counts = data.get("counts")
            if not isinstance(counts, dict):
                counts = {}
            key = f"{provider}/{model.split('/', 1)[-1]}"
            counts[key] = _safe_count(counts.get(key, 0)) + 1
            data["counts"] = counts
            _usage_path().parent.mkdir(parents=True, exist_ok=True)
            tmp = _usage_path().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, _usage_path())
    except Exception:
        pass


def most_used_offline() -> str | None:
    """Most-counted model from the usage file with zero network.

    Used at startup (from_dict) so config load is instant and offline-safe.
    Never raises; None on any failure or empty counts.
    """
    try:
        data = json.loads(_usage_path().read_text(encoding="utf-8"))
        counts = data.get("counts")
        if not isinstance(counts, dict) or not counts:
            return None
    except (OSError, ValueError):
        return None
    try:
        best = max(counts.items(), key=lambda kv: _safe_count(kv[1]))
    except Exception:
        return None
    if _safe_count(best[1]) <= 0:
        return None
    return str(best[0])


def most_used_model() -> str | None:
    """The user's most-answered (provider/model), or None when unknown.

    Counts only models that still exist upstream (a removed model can never
    win, even if it was once #1). Never raises; None on any failure.
    """
    try:
        data = json.loads(_usage_path().read_text(encoding="utf-8"))
        counts = data.get("counts")
        if not isinstance(counts, dict) or not counts:
            return None
    except (OSError, ValueError):
        return None
    try:
        from .providers.rotation import fetch_catalog
        catalog = (fetch_catalog().get("opencode") or {}).get("models", {})
        live = {mid for mid, m in catalog.items()
                if m.get("status", "active") == "active"}
    except Exception:
        live = set()
    if not live or not catalog:
        # catalog unreachable (offline / first run): can't prove removal —
        # count everything rather than wrongly dropping the real winner.
        live = set()
    try:
        from .providers.rotation import check_zen_model_health
        health = check_zen_model_health(
            [str(k).split("/", 1)[-1] if "/" in str(k) else str(k) for k in counts])
    except Exception:
        health = {}
    best: str | None = None
    best_n = 0
    for key, n in counts.items():
        n = _safe_count(n)
        if n <= best_n:
            continue
        bare = str(key).split("/", 1)[-1] if "/" in str(key) else str(key)
        if live and bare not in live and str(key) not in live:
            continue  # removed upstream — can never win
        if health.get(bare) is False:
            continue  # not answering right now — skip, don't crown it
        best, best_n = str(key), n
    if best is None:
        # every counted model is dead/removed: fall back to the
        # most-answering LIVE model rather than None (which would start on
        # the hardcoded default while a healthy winner exists).
        for key, n in sorted(counts.items(), key=lambda kv: _safe_count(kv[1]), reverse=True):
            bare = str(key).split("/", 1)[-1] if "/" in str(key) else str(key)
            if live and bare not in live and str(key) not in live:
                continue
            if health.get(bare) is False:
                continue
            return str(key)
    return best


@dataclass
class Config:
    provider: str = "opencode"
    model: str = DEFAULT_MODEL
    small_model: str = DEFAULT_SMALL_MODEL
    default_agent: str = "build"
    subagent_depth: int = 1
    username: str = field(default_factory=lambda: os.environ.get("USER", "user"))
    instructions: list[str] = field(default_factory=list)
    permission: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    commands: dict[str, Any] = field(default_factory=dict)
    rotation: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    theme: str = "opencode"
    diff_style: str = "split"  # "split" (auto side-by-side >120 cols) | "stacked" (unified always)
    diff_wrap_mode: str = "word"  # "word" | "none"
    suppress_backgrounds: bool = False
    # the `Build (2 of 4) 12,345 (2%) Prev/Next` bar under a sub-agent chat.
    # OFF by default: arrow-key session navigation works without it, and the
    # bar ate a screen line on small phone displays.
    subagent_footer: bool = False
    tool_output_max_lines: int = 2000
    tool_output_max_bytes: int = 51200
    compaction_enabled: bool = True
    compaction_tail_turns: int = 2
    context_budget: int = 120000
    # Fix 1 (request trim): last N user turns go verbatim; older tool
    # outputs shrink to one-line receipts for sending (disk stays full).
    trim_keep_turns: int = 2
    trim_max_chars: int = 500
    # Fix 2 (early summary): kick a background summary at this fraction of
    # the usable window so overflows adopt it with zero extra model calls.
    early_summary_at: float = 0.7
    # Fix 4 (disk): cap session files (0 = keep default 200-row popup cap
    # only). Positive N prunes oldest inactive sessions past N files.
    session_file_cap: int = 0
    bash_default_timeout: int = 120
    model_read_timeout: float = 300.0
    auto_retry: bool = True
    auto_retry_count: int = 5
    auto_continue: bool = False
    auto_approve: bool = False
    rotation_lock: bool = False
    # "auto" = never ask (allow-all; explicit deny rules still apply),
    # "ask"  = popup allow/deny for every action not covered by allow rules,
    # "deny" = block everything not explicitly allowed,
    # "fully_auto" = auto + zero popups of ANY kind: suppresses the question
    #   tool dialog too, so the model runs to the goal uninterrupted.
    permission_mode: str = "auto"
    # Reasoning effort (official variant parity): e.g. "high" for muse-spark.
    # "" = gateway default. Only sent when valid for the CURRENT model per the
    # live catalog — future models with new levels work with zero code changes.
    reasoning_effort: str = ""
    # Whether thought bubbles (`+ Thought` rows) render at all. Off hides
    # them without dropping the underlying reasoning (re-enabling shows them
    # again, /thinking last still prints in headless).
    show_thoughts: bool = True
    # Save-data mode for mobile packages: shrinks the budgets/caps that drive
    # per-turn bytes (history window, tool-result caps, compaction tail) when
    # the user hasn't set them explicitly. Never disables tools. OFF by
    # default; set "low_data": true (alias "save_data") to enable.
    low_data: bool = False
    # Auto low-RAM mode: "auto" (default) enables the mobile save-data budgets
    # AND the small live chat window on devices reporting <=2GB total RAM
    # (util/ram.py, /proc/meminfo); "on" forces them everywhere, "off" never.
    # Explicit user values always win over the preset. Never disables tools.
    low_ram: str = "auto"
    # Resolved at load: True when the low-RAM budgets actually applied
    # (low_data flag OR low_ram auto/on matched). Read-only for callers.
    low_ram_active: bool = False
    # Live chat window: at most this many message bubbles stay mounted on the
    # chat screen; older ones are evicted to the "older messages" pill and
    # reload on scroll-up (same pill the resume path uses). 0 = unlimited.
    chat_live_window: int = 120
    # Text-to-speech ("speak" tool). OFF by default: the agent never talks
    # unless the user opts in (Settings > speak). Offline engines need no
    # key; online ElevenLabs voices need ELEVENLABS_API_KEY.
    tts_enabled: bool = False
    tts_auto: bool = False
    tts_engine: str = "auto"  # auto | offline | elevenlabs
    tts_voice: str = ""
    tts_language: str = ""
    tts_rate: float = 1.0
    tts_pitch: float = 1.0
    tts_voice_id: str = ""
    tts_model: str = "eleven_multilingual_v2"
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "Config":
        raw = data
        cfg = cls()
        cfg.raw = raw
        base = base_dir or Path.cwd()
        if "provider" in data:
            cfg.provider = data["provider"]
        if "model" in data:
            cfg.model = _parse_model(data["model"], cfg.provider)
        elif "model" not in data:
            # no saved pick: start on the user's most-used model WITHOUT
            # network (offline-first). most_used_model() can hit the catalog
            # + health endpoints; here we count only, and the full
            # live/health filtering happens lazily on warm_startup. Startup
            # never hangs or fails offline.
            try:
                fav = most_used_offline()
                if fav:
                    cfg.model = _parse_model(fav, cfg.provider)
            except Exception:
                pass
        if "small_model" in data:
            cfg.small_model = _parse_model(data["small_model"], cfg.provider)
        if "default_agent" in data:
            cfg.default_agent = data["default_agent"]
        if "subagent_depth" in data:
            try:
                cfg.subagent_depth = int(data["subagent_depth"])
            except (ValueError, TypeError):
                pass
        if "username" in data:
            cfg.username = data["username"]
        if "instructions" in data:
            cfg.instructions = [str(x) for x in data["instructions"]]
        if "permission" in data:
            perm = _apply_vars(data["permission"], base)
            # opencode allows `permission: "ask"` as a global default action;
            # normalize it to `{"*": <action>}` so consumer code (permission
            # engine, /config validate) always sees a dict and a malformed
            # value can never crash the engine at startup.
            if isinstance(perm, str):
                perm = {"*": perm}
            cfg.permission = perm if isinstance(perm, dict) else {}
        if "agents" in data:
            # plural is the canonical key; non-dict values (e.g. a stray
            # string from a hand-edited file) are rejected, never coerced
            # into phantom agents. Values are kept VERBATIM (no
            # {env:}/{file:} substitution — agent text is user content).
            agents_val = data["agents"]
            cfg.agents = agents_val if isinstance(agents_val, dict) else {}
        elif "agent" in data:
            # LEGACY singular key: only honored when it holds a dict of
            # agent specs (old files). Anything else is ignored. The next
            # save canonicalizes to plural only.
            legacy = data["agent"]
            cfg.agents = legacy if isinstance(legacy, dict) else {}
        if "provider" in data and isinstance(data.get("provider"), dict):
            # 'provider' key is the default provider id; custom providers live under
            # 'providers' to avoid ambiguity with the top-level model selection.
            pass
        if "providers" in data:
            cfg.providers = _apply_vars(data["providers"], base)
        if "commands" in data:
            cfg.commands = data["commands"]
        if "rotation" in data:
            # Sanitize so a malformed `rotation` (string, list of non-dicts /
            # entries without a provider/model) can never crash build_rotation
            # or Rotation.stream at startup. Non-dict lanes are dropped; a
            # wholly-invalid value just falls back to the default lane.
            rot = data["rotation"]
            if isinstance(rot, list):
                cfg.rotation = [
                    {k: v for k, v in lane.items() if k in ("provider", "model")}
                    for lane in rot
                    if isinstance(lane, dict) and lane.get("provider") and lane.get("model")
                ]
            else:
                cfg.rotation = []
        if "system_prompt" in data:
            cfg.system_prompt = data["system_prompt"]
        if "theme" in data:
            cfg.theme = data["theme"]
        if "diff" in data:
            d = data["diff"]
            if isinstance(d, dict):
                cfg.diff_style = str(d.get("style", cfg.diff_style))
                cfg.diff_wrap_mode = str(d.get("wrap", cfg.diff_wrap_mode))
                cfg.suppress_backgrounds = bool(d.get("suppress_backgrounds", cfg.suppress_backgrounds))
        if "subagent_footer" in data:
            cfg.subagent_footer = bool(data["subagent_footer"])
        if "tool_output" in data:
            to = data["tool_output"]
            try:
                cfg.tool_output_max_lines = int(to.get("max_lines", 2000))
            except (ValueError, TypeError):
                pass
            try:
                cfg.tool_output_max_bytes = int(to.get("max_bytes", 51200))
            except (ValueError, TypeError):
                pass
        if "context_budget" in data:
            try:
                cfg.context_budget = int(data["context_budget"])
            except (ValueError, TypeError):
                pass
        if "bash_default_timeout" in data:
            try:
                cfg.bash_default_timeout = int(data["bash_default_timeout"])
            except (ValueError, TypeError):
                pass
        if "model_read_timeout" in data:
            try:
                cfg.model_read_timeout = float(data["model_read_timeout"])
            except (ValueError, TypeError):
                pass
        if "auto_retry" in data:
            cfg.auto_retry = bool(data["auto_retry"])
        if "auto_continue" in data:
            cfg.auto_continue = bool(data["auto_continue"])
        if "auto_retry_count" in data:
            try:
                cfg.auto_retry_count = int(data["auto_retry_count"])
            except (ValueError, TypeError):
                pass
        if "permission_mode" in data:
            pm = str(data["permission_mode"]).lower()
            if pm in ("auto", "ask", "deny", "fully_auto"):
                cfg.permission_mode = pm
        if "reasoning_effort" in data:
            cfg.reasoning_effort = str(data["reasoning_effort"] or "").strip().lower()
        if "show_thoughts" in data:
            cfg.show_thoughts = bool(data["show_thoughts"])
        if "rotation_lock" in data:
            cfg.rotation_lock = bool(data["rotation_lock"])
        if "low_data" in data:
            cfg.low_data = bool(data["low_data"])
        elif "save_data" in data:
            cfg.low_data = bool(data["save_data"])
        if "low_ram" in data:
            lr = str(data["low_ram"] or "").strip().lower()
            if lr in ("auto", "on", "off", "yes", "no", "true", "false", "1", "0"):
                cfg.low_ram = {"yes": "on", "true": "on", "1": "on",
                               "no": "off", "false": "off", "0": "off"}.get(lr, lr)
        if "chat_live_window" in data:
            try:
                cfg.chat_live_window = max(0, int(data["chat_live_window"]))
            except (ValueError, TypeError):
                pass
        # Auto low-RAM: same save-data budgets, enabled by device memory
        # (<=2GB) instead of an explicit flag. Explicit values always win.
        low_ram_on = (
            cfg.low_ram == "on"
            or (cfg.low_ram == "auto" and is_low_ram())
        )
        cfg.low_ram_active = bool(low_ram_on or cfg.low_data)
        if cfg.low_ram_active:
            # Preset only fills values the user didn't set explicitly.
            to = data.get("tool_output") if isinstance(data.get("tool_output"), dict) else {}
            if "max_lines" not in to:
                cfg.tool_output_max_lines = 200
            if "max_bytes" not in to:
                cfg.tool_output_max_bytes = 10240
            if "context_budget" not in data:
                cfg.context_budget = 30000
            c = data.get("compaction") if isinstance(data.get("compaction"), dict) else {}
            if "tail_turns" not in c:
                cfg.compaction_tail_turns = 1
            if "session_file_cap" not in data:
                cfg.session_file_cap = 50
            tr = data.get("trim") if isinstance(data.get("trim"), dict) else {}
            if "keep_turns" not in tr:
                cfg.trim_keep_turns = 1
            if "max_chars" not in tr:
                cfg.trim_max_chars = 200
        if "compaction" in data:
            c = data["compaction"]
            cfg.compaction_enabled = bool(c.get("auto", True))
            try:
                cfg.compaction_tail_turns = int(c.get("tail_turns", 2))
            except (ValueError, TypeError):
                pass
        if "trim" in data and isinstance(data["trim"], dict):
            tr = data["trim"]
            try:
                if "keep_turns" in tr:
                    cfg.trim_keep_turns = max(1, int(tr["keep_turns"]))
            except (ValueError, TypeError):
                pass
            try:
                if "max_chars" in tr:
                    cfg.trim_max_chars = max(100, int(tr["max_chars"]))
            except (ValueError, TypeError):
                pass
        if "early_summary_at" in data:
            try:
                cfg.early_summary_at = min(0.95, max(0.5, float(data["early_summary_at"])))
            except (ValueError, TypeError):
                pass
        if "session_file_cap" in data:
            try:
                cfg.session_file_cap = max(0, int(data["session_file_cap"]))
            except (ValueError, TypeError):
                pass
        if "tts" in data and isinstance(data["tts"], dict):
            t = data["tts"]
            if "enabled" in t:
                cfg.tts_enabled = bool(t["enabled"])
            if "auto" in t:
                cfg.tts_auto = bool(t["auto"])
            if "engine" in t:
                eng = str(t["engine"] or "").strip().lower()
                if eng in ("auto", "offline", "elevenlabs"):
                    cfg.tts_engine = eng
            if "voice" in t:
                cfg.tts_voice = str(t["voice"] or "")
            if "language" in t:
                cfg.tts_language = str(t["language"] or "")
            try:
                if "rate" in t:
                    cfg.tts_rate = min(4.0, max(0.25, float(t["rate"])))
            except (ValueError, TypeError):
                pass
            try:
                if "pitch" in t:
                    cfg.tts_pitch = min(2.0, max(0.5, float(t["pitch"])))
            except (ValueError, TypeError):
                pass
            if "voice_id" in t:
                cfg.tts_voice_id = str(t["voice_id"] or "")
            if "model" in t:
                cfg.tts_model = str(t["model"] or "")
        # OPENCODE_PERMISSION env overrides permission
        if os.environ.get("OPENCODE_PERMISSION"):
            try:
                override = json.loads(os.environ["OPENCODE_PERMISSION"])
                cfg.permission = deep_merge(cfg.permission, override)
            except json.JSONDecodeError:
                pass
        return cfg

    def as_dict(self, *, show_secrets: bool = False) -> dict:
        providers = self.providers
        if not show_secrets and isinstance(providers, dict):
            redacted: dict[str, Any] = {}
            for pid, pconf in providers.items():
                if isinstance(pconf, dict) and "api_key" in pconf:
                    p = dict(pconf)
                    p["api_key"] = "*** (hidden, use --show-secrets to display)"
                    redacted[pid] = p
                else:
                    redacted[pid] = pconf
            providers = redacted
        return {
            "provider": self.provider,
            "model": self.model,
            "small_model": self.small_model,
            "default_agent": self.default_agent,
            "subagent_depth": self.subagent_depth,
            "username": self.username,
            "instructions": self.instructions,
            "permission": self.permission,
            "agents": self.agents,
            "providers": providers,
            "commands": self.commands,
            "rotation": self.rotation,
            "system_prompt": self.system_prompt,
            "theme": self.theme,
            "diff": {
                "style": self.diff_style,
                "wrap": self.diff_wrap_mode,
                "suppress_backgrounds": self.suppress_backgrounds,
            },
            "subagent_footer": self.subagent_footer,
            "tool_output": {
                "max_lines": self.tool_output_max_lines,
                "max_bytes": self.tool_output_max_bytes,
            },
            "context_budget": self.context_budget,
            "bash_default_timeout": self.bash_default_timeout,
            "model_read_timeout": self.model_read_timeout,
            "auto_retry": self.auto_retry,
            "auto_retry_count": self.auto_retry_count,
            "auto_continue": self.auto_continue,
            "rotation_lock": self.rotation_lock,
            "low_data": self.low_data,
            "low_ram": self.low_ram,
            "low_ram_active": self.low_ram_active,
            "chat_live_window": self.chat_live_window,
            "tts": {
                "enabled": self.tts_enabled,
                "auto": self.tts_auto,
                "engine": self.tts_engine,
                "voice": self.tts_voice,
                "language": self.tts_language,
                "rate": self.tts_rate,
                "pitch": self.tts_pitch,
                "voice_id": self.tts_voice_id,
                "model": self.tts_model,
            },
            "permission_mode": self.permission_mode,
            "reasoning_effort": self.reasoning_effort,
            "show_thoughts": self.show_thoughts,
            "compaction": {
                "auto": self.compaction_enabled,
                "tail_turns": self.compaction_tail_turns,
            },
        }


def _parse_model(model: str, default_provider: str) -> str:
    if "/" in model:
        parts = model.split("/")
        if len(parts) == 2 and parts[0] and parts[1]:
            return model
        return f"{default_provider}/{model.replace('/', '-')}"
    return f"{default_provider}/{model}"


def _project_config_files(directory: Path) -> list[Path]:
    """Find opencode.json/opencode.jsonc walking up from directory to worktree root."""
    from .globals import resolve_worktree

    worktree = resolve_worktree(directory)
    files: list[Path] = []
    d = directory.resolve()
    while True:
        for name in ("opencode.jsonc", "opencode.json"):
            p = d / name
            if p.exists():
                files.append(p)
        if d == worktree:
            break
        if d.parent == d:
            break
        d = d.parent
    return files


def _user_config_file() -> Path | None:
    for name in ("opencode.jsonc", "opencode.json"):
        p = GPath.config / name
        if p.exists():
            return p
    return None


def load_config(directory: Path | None = None) -> Config:
    """Load config following opencode's merge order (later wins)."""
    directory = directory or Path.cwd()
    merged: dict[str, Any] = {}

    sources: list[tuple[Path, dict]] = []
    # files whose primary copy was unreadable (corrupt/empty): recovered
    # from .bak/.tmp, or skipped entirely. save_config must never
    # overwrite these with defaults — see _config_recovered below.
    recovered: dict[str, str] = {}

    def add(path: Path) -> None:
        try:
            sources.append((path, _load_json(path)))
            return
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        data, source = load_json_with_fallback(path)
        if data is not None:
            sources.append((path, data))
            if source != "primary":
                recovered[str(path)] = source
                return
        # no usable copy at all: remember so a later save won't cement
        # defaults over a (possibly transiently unreadable) real file
        recovered.setdefault(str(path), "missing")

    # user-level global config first (lowest priority), then project configs
    user = _user_config_file()
    if user:
        add(user)
    for p in _project_config_files(directory):
        add(p)

    # track every layer that contributed, in merge order (for warnings)
    layered = [str(p) for p, _d in sources]

    # OPENCODE_CONFIG env points to an explicit file (overrides discovered)
    if os.environ.get("OPENCODE_CONFIG"):
        add(Path(os.environ["OPENCODE_CONFIG"]).expanduser())
        layered.append(f"env:OPENCODE_CONFIG={os.environ['OPENCODE_CONFIG']}")

    # OPENCODE_CONFIG_CONTENT inline JSON (highest)
    if os.environ.get("OPENCODE_CONFIG_CONTENT"):
        try:
            sources.append((Path.cwd(), json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])))
            layered.append("env:OPENCODE_CONFIG_CONTENT")
        except json.JSONDecodeError:
            pass

    for path, data in sources:
        # agents carry user-authored text (descriptions, rules, md notes):
        # {env:..}/{file:..} substitution must NEVER rewrite it (a rule
        # mentioning "{env:FOO}" would be blanked on load, then re-saved
        # blank — silent data loss). Strip agent keys before substitution;
        # from_dict merges them back verbatim via deep_merge.
        agents_raw: dict[str, Any] = {}
        if isinstance(data, dict):
            for key in ("agents", "agent"):
                if key in data:
                    agents_raw[key] = data.pop(key)
        data = _apply_vars(data, path.parent)
        if agents_raw:
            data = deep_merge(data, agents_raw) if isinstance(data, dict) else agents_raw
        merged = deep_merge(merged, data)

    cfg = Config.from_dict(merged, directory)
    # carry the recovery map so save_config can refuse to pave over files
    # that failed to load (corrupt/missing) with fresh defaults
    try:
        cfg.recovered_sources = recovered
    except Exception:
        pass
    try:
        cfg.config_layers = layered
    except Exception:
        pass
    return cfg


def config_shadow_warnings(cfg: Config) -> list[str]:
    """Human-readable warnings when env/project layers shadow the user file.

    Displayed once at startup so a "vanished" agent is explained instead
    of mysterious (e.g. OPENCODE_CONFIG_CONTENT set in one shell only).
    """
    out: list[str] = []
    try:
        layers = getattr(cfg, "recovered_sources", None) or {}
        for path, source in layers.items():
            if source in ("bak", "tmp"):
                out.append(f"Config {path} was corrupt — recovered from .{source}.")
            elif source == "missing":
                out.append(f"Config {path} unreadable — using defaults (not saved over it).")
    except Exception:
        pass
    try:
        if os.environ.get("OPENCODE_CONFIG"):
            out.append(f"OPENCODE_CONFIG overrides: {os.environ['OPENCODE_CONFIG']}")
        if os.environ.get("OPENCODE_CONFIG_CONTENT"):
            out.append("OPENCODE_CONFIG_CONTENT overrides all files (agents here are session-only).")
    except Exception:
        pass
    return out


def save_config(cfg: Config, path: Path | None = None, merge_disk_agents: bool = True) -> None:
    """Persist cfg. Raises OSError on failure (callers decide how loud).

    Safety rules (agent permanence):
    - keep a `.bak` replica of the previous good file before overwriting.
    - NEVER pave over a file that failed to load at startup
      (`cfg.recovered_sources`): writing fresh defaults there would
      permanently erase agents saved before a crash.
    - the live cfg is authoritative for `agents`: exactly one plural key.
    - merge_disk_agents (default True): union agents that appeared on
      disk since load (another process added one). Pass False ONLY from
      the agent manager itself (_persist_agents), where memory owns the
      dict and a union would resurrect just-deleted agents.
    """
    from .session import _write_durable

    path = path or (GPath.config / "opencode.json")
    blocked = ""
    try:
        blocked = (getattr(cfg, "recovered_sources", None) or {}).get(str(path), "")
    except Exception:
        blocked = ""
    if blocked in ("missing",):
        # no usable copy existed at load AND we still have no evidence the
        # file is writable-safe: write only when agents actually exist to
        # persist (a defaults-only write would create a decoy empty file).
        try:
            if not getattr(cfg, "agents", None):
                return
        except Exception:
            return
    if blocked not in ("", "missing"):
        # primary was corrupt/empty but .bak/.tmp rescued us: preserve the
        # broken primary aside instead of silently replacing it, so a human
        # can inspect what the crash left behind.
        try:
            bad = path.read_text(encoding="utf-8", errors="replace")[:20000]
            path.parent.mkdir(parents=True, exist_ok=True)
            Path(str(path) + ".corrupt").write_text(bad, encoding="utf-8")
        except OSError:
            pass
    out = cfg.as_dict()
    for key, value in (cfg.raw or {}).items():
        if key not in out:
            out[key] = value
    # canonicalize the agents key: older files carry singular "agent"
    # alongside plural "agents", and on load the plural shadows the
    # singular — a stale plural from an earlier save could resurrect
    # over fresh data (created agents "vanishing" after any later save).
    # The live cfg is authoritative: keep exactly one plural key.
    out.pop("agent", None)
    # last-writer-wins guard (see docstring): union disk agents unknown to
    # memory. Skipped when the agent manager saves (it owns the dict —
    # unioning there would resurrect just-deleted agents).
    if merge_disk_agents:
        try:
            disk_data, _src = load_json_with_fallback(path)
            if isinstance(disk_data, dict):
                disk_agents = disk_data.get("agents")
                if isinstance(disk_agents, dict) and isinstance(out.get("agents"), dict):
                    for aname, aspec in disk_agents.items():
                        if aname not in out["agents"]:
                            out["agents"][aname] = aspec
        except Exception:
            pass
    out["agents"] = cfg.agents if isinstance(out.get("agents"), dict) else out.get("agents")
    text = json.dumps(out, indent=2, ensure_ascii=False)
    # .bak replica of the previous good body (best effort, never fatal)
    try:
        if path.exists():
            prev = path.read_text(encoding="utf-8", errors="replace")
            if prev.strip():
                Path(str(path) + ".bak").write_text(prev, encoding="utf-8")
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = _write_durable(path, text, file_sync=True, dir_sync=True)
    if not ok:
        raise OSError(f"could not write config to {path}")
