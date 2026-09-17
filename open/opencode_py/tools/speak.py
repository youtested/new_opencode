"""speak tool: text-to-speech for the agent (offline first, online clone optional).

RULES (full guidance for maintainers; the sent schema is short):
- speak reads text aloud on THIS phone (Settings > speak must be ON).
- Offline, no key: termux-tts-speak, espeak-ng/espeak, spd-say.
- Online (ELEVENLABS_API_KEY): any voice incl. clones; clone needs a
  ~20s recording + key (/connect -> ElevenLabs TTS).
- speak/stop/voices/clone/status. Max 2000 chars. Honest errors telling
  what to install or which key to add.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .registry import Tool, schema_with

_ACTIONS = ("speak", "stop", "voices", "clone", "status")

_ENGINES = ("auto", "offline", "elevenlabs")

_ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
_ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_ELEVENLABS_ADD_URL = "https://api.elevenlabs.io/v1/voices/add"

_MAX_CHARS = 2000
_CLONE_MAX_BYTES = 10 * 1024 * 1024


_MISSING = object()


def _cfg_tts(cfg: Any) -> dict[str, Any]:
    try:
        raw = (getattr(cfg, "raw", None) or {}) if cfg is not None else {}
        tts = raw.get("tts")
        if isinstance(tts, dict):
            return tts
    except Exception:
        pass
    return {}


def _opt(cfg: Any, name: str, default: Any) -> Any:
    # Live Settings edits mutate cfg.tts_* fields; the raw file dict goes
    # stale until save+reload — so fields win, raw is only a fallback for
    # cfg-like objects without the typed fields.
    try:
        attr = getattr(cfg, f"tts_{name}", _MISSING) if cfg is not None else _MISSING
        if attr is not _MISSING:
            return attr
    except Exception:
        pass
    return _cfg_tts(cfg).get(name, default)


def _enabled(cfg: Any) -> bool:
    try:
        return bool(_opt(cfg, "enabled", False))
    except Exception:
        return False


def auto_enabled(cfg: Any) -> bool:
    """Auto-voice switch: every model reply speaks on its own."""
    try:
        return _enabled(cfg) and bool(_opt(cfg, "auto", False))
    except Exception:
        return False


def speak_text(cfg: Any, text: str) -> dict:
    """Speak one reply, honoring the on/off + auto switches. Never raises."""
    try:
        if not auto_enabled(cfg):
            return {"output": "Auto voice is off.", "metadata": {"auto": False}}
        return _action_speak({"text": text}, cfg)
    except Exception as e:
        return {"output": f"speak failed: {e}", "error": True}


def stop_speech() -> dict:
    """Silence current speech. Never raises."""
    try:
        return _action_stop()
    except Exception as e:
        return {"output": f"stop failed: {e}", "error": True}


def _run(args: list[str], timeout: float = 60.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, f"{args[0]} not found"
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or f"{args[0]} exited {proc.returncode}"
    return True, (proc.stdout or "").strip()


def _clean_text(text: str) -> str:
    """Make model output speakable: drop code fences/URLs, collapse space."""
    t = str(text or "")
    t = re.sub(r"```.*?```", " code omitted ", t, flags=re.DOTALL)
    t = t.replace("```", " ")
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"[*_`>|#\-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…])\s+|\n+")


def split_stream_chunks(buffer: str) -> tuple[list[str], str]:
    """Split streaming text into speak-now sentences + unsaid remainder.

    Plain-English: as the model writes, each finished sentence is returned
    for immediate speech while the unfinished tail is held back. A very long
    sentence without punctuation is cut at a word boundary so speech still
    starts fast. Text inside an unclosed code fence is held, not spoken.
    Never raises.
    """
    try:
        buf = str(buffer or "")
        if not buf.strip():
            return [], ""
        # hold an unclosed code fence: only split what came before it
        hold = ""
        head = buf
        try:
            if buf.count("```") % 2 == 1:
                idx = buf.rfind("```")
                head, hold = buf[:idx], buf[idx:]
        except Exception:
            head, hold = buf, ""
        parts = _SENTENCE_SPLIT_RE.split(head)
        if len(parts) <= 1:
            stripped = head.strip()
            # fast start on a clause boundary (comma/semicolon/colon): a
            # greeting like "Hello, what can I do for you today?" talks its
            # first clause while the rest is still streaming.
            if len(stripped) >= 30:
                for sep in (",", ";", ":"):
                    idx = stripped.rfind(sep, 0, 140)
                    if idx and idx > 3:
                        chunk = stripped[: idx + 1].strip()
                        rest = (stripped[idx + 1 :].strip() + (" " + hold.strip() if hold.strip() else "")).strip()
                        return ([chunk] if chunk else [], rest or buf)
            # no finished sentence yet: cut a long run-on at a word boundary
            if len(stripped) >= 120:
                cut = head.rfind(" ", 0, 140)
                if cut < 60:
                    cut = head.rfind(" ")
                if cut and cut > 40:
                    chunk = head[:cut].strip()
                    rest = (head[cut:].strip() + (" " + hold.strip() if hold.strip() else "")).strip()
                    return ([chunk] if chunk else [], rest or buf)
            return [], buf
        *done, rest = parts
        chunks: list[str] = []
        for d in done:
            d = (d or "").strip()
            if not d:
                continue
            # a single over-long sentence: cut into ~250-char word pieces
            while len(d) > 400:
                cut = d.rfind(" ", 0, 260)
                if cut < 120:
                    break
                chunks.append(d[:cut].strip())
                d = d[cut:].strip()
            if d:
                chunks.append(d)
        remainder = (rest or "").strip()
        if hold.strip():
            remainder = ((remainder + " " + hold.strip()).strip() if remainder
                         else hold.strip())
        return chunks, remainder
    except Exception:
        return [], str(buffer or "")


def _elevenlabs_key() -> str | None:
    val = os.environ.get("ELEVENLABS_API_KEY")
    if val:
        return val
    try:
        from ..auth import Auth

        return Auth().get("elevenlabs")
    except Exception:
        return None


def _offline_players() -> list[str]:
    found = []
    for binary in ("termux-tts-speak", "espeak-ng", "espeak", "spd-say"):
        if shutil.which(binary):
            found.append(binary)
    return found


def _speak_offline(text: str, voice: str, language: str,
                   rate: float, pitch: float) -> dict | None:
    """Try each offline engine in order. Returns result dict or None if none."""
    if shutil.which("termux-tts-speak"):
        cmd = ["termux-tts-speak"]
        flagged = False
        if language:
            cmd += ["-l", language]
            flagged = True
        try:
            if rate and abs(rate - 1.0) > 1e-6:
                cmd += ["-r", str(rate)]
                flagged = True
        except Exception:
            pass
        try:
            if pitch and abs(pitch - 1.0) > 1e-6:
                cmd += ["-p", str(pitch)]
                flagged = True
        except Exception:
            pass
        cmd.append(text)
        budget = min(120.0, max(30.0, len(text) / 15.0))
        ok, msg = _run(cmd, timeout=budget)
        if ok:
            return {"output": f"Spoke ({len(text)} chars) via Termux TTS.",
                    "metadata": {"engine": "termux-tts-speak", "chars": len(text)}}
        if flagged:
            # Older Termux:API builds may not know -r/-p: retry bare text
            # before giving up so a flag mismatch never silences speech.
            ok2, msg2 = _run(["termux-tts-speak", text], timeout=budget)
            if ok2:
                return {"output": f"Spoke ({len(text)} chars) via Termux TTS.",
                        "metadata": {"engine": "termux-tts-speak", "chars": len(text),
                                     "note": "voice options unsupported by this Termux:API, used defaults"}}
            msg = msg2
        return {"output": f"Termux TTS failed: {msg}", "error": True}
    for binary in ("espeak-ng", "espeak"):
        if shutil.which(binary):
            cmd = [binary]
            try:
                wpm = int(175 * float(rate or 1.0))
                cmd += ["-s", str(max(80, min(450, wpm)))]
            except Exception:
                pass
            if voice:
                cmd += ["-v", voice]
            cmd.append(text)
            ok, msg = _run(cmd, timeout=min(120.0, max(30.0, len(text) / 15.0)))
            if ok:
                return {"output": f"Spoke ({len(text)} chars) via {binary}.",
                        "metadata": {"engine": binary, "chars": len(text)}}
            return {"output": f"{binary} failed: {msg}", "error": True}
    if shutil.which("spd-say"):
        ok, msg = _run(["spd-say", text], timeout=min(120.0, max(30.0, len(text) / 15.0)))
        if ok:
            return {"output": f"Spoke ({len(text)} chars) via spd-say.",
                    "metadata": {"engine": "spd-say", "chars": len(text)}}
        return {"output": f"spd-say failed: {msg}", "error": True}
    return None


def _play_file(path: str) -> str:
    """Play a saved audio file with whatever player exists. Returns player used or ''."""
    if shutil.which("termux-media-player"):
        ok, _ = _run(["termux-media-player", "play", path], timeout=15.0)
        if ok:
            return "termux-media-player"
    for player, args in (("mpv", ["--no-video", "--really-quiet"]),
                         ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
                         ("play", [])):
        if shutil.which(player):
            ok, _ = _run([player] + list(args) + [path], timeout=120.0)
            if ok:
                return player
    return ""


def _speak_elevenlabs(text: str, voice_id: str, model: str,
                      stability: float, similarity: float, speed: float) -> dict:
    key = _elevenlabs_key()
    if not key:
        return {"output": ("ElevenLabs key missing. Fix: /connect -> ElevenLabs TTS "
                           "(https://elevenlabs.io/app/settings/api-keys), or set "
                           "ELEVENLABS_API_KEY."), "error": True}
    try:
        import httpx  # deferred: heavy import, only for online voices
    except ImportError:
        return {"output": "ElevenLabs needs httpx: `pip install httpx`.", "error": True}
    vid = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Rachel (public default)
    payload = {"text": text, "model_id": model or "eleven_multilingual_v2",
               "voice_settings": {"stability": stability, "similarity_boost": similarity,
                                  "speed": speed}}
    try:
        resp = httpx.post(_ELEVENLABS_TTS_URL.format(voice_id=vid),
                          headers={"xi-api-key": key, "Content-Type": "application/json"},
                          json=payload, timeout=60.0)
    except Exception as e:
        return {"output": f"ElevenLabs request failed: {e}", "error": True}
    if resp.status_code != 200:
        detail = (resp.text or "")[:300]
        return {"output": f"ElevenLabs error {resp.status_code}: {detail}", "error": True}
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(resp.content)
        tmp.close()
        path = tmp.name
    except OSError as e:
        return {"output": f"could not save audio: {e}", "error": True}
    player = _play_file(path)
    meta: dict[str, Any] = {"engine": "elevenlabs", "voice_id": vid,
                            "chars": len(text), "bytes": len(resp.content)}
    if player:
        meta["player"] = player
        try:
            import os as _os

            _os.unlink(path)
        except OSError:
            meta["file"] = path
        return {"output": f"Spoke ({len(text)} chars) as voice {vid} via ElevenLabs.", "metadata": meta}
    meta["file"] = path
    return {"output": (f"Rendered ({len(text)} chars) as voice {vid} via ElevenLabs; "
                       f"no audio player found, saved to {path}."), "metadata": meta}


def _action_speak(args: dict, cfg: Any) -> dict:
    raw_text = str(args.get("text") or "").strip()
    if not raw_text:
        return {"output": "speak: no text given (pass 'text').", "error": True}
    text = _clean_text(raw_text)
    if not text:
        return {"output": "speak: text was empty after cleanup.", "error": True}
    truncated = False
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS].rstrip() + "…"
        truncated = True
    voice = str(args.get("voice") or _opt(cfg, "voice", "") or "").strip()
    language = str(args.get("language") or _opt(cfg, "language", "") or "").strip()
    try:
        rate = float(args.get("rate") if args.get("rate") not in (None, "") else _opt(cfg, "rate", 1.0))
    except (TypeError, ValueError):
        rate = 1.0
    rate = min(4.0, max(0.25, rate))
    try:
        pitch = float(args.get("pitch") if args.get("pitch") not in (None, "") else _opt(cfg, "pitch", 1.0))
    except (TypeError, ValueError):
        pitch = 1.0
    pitch = min(2.0, max(0.5, pitch))
    want = str(args.get("engine") or _opt(cfg, "engine", "auto") or "auto").strip().lower()
    if want not in _ENGINES:
        return {"output": f"Unknown engine {want!r} (want one of {', '.join(_ENGINES)}).", "error": True}
    voice_id = str(args.get("voice_id") or _opt(cfg, "voice_id", "") or "").strip()
    model = str(args.get("model") or _opt(cfg, "model", "eleven_multilingual_v2") or "").strip()
    try:
        stability = float(_opt(cfg, "stability", 0.5))
    except (TypeError, ValueError):
        stability = 0.5
    try:
        similarity = float(_opt(cfg, "similarity", 0.75))
    except (TypeError, ValueError):
        similarity = 0.75
    stability = min(1.0, max(0.0, stability))
    similarity = min(1.0, max(0.0, similarity))

    if want in ("auto", "offline"):
        res = _speak_offline(text, voice, language, rate, pitch)
        if res is not None and not res.get("error"):
            if truncated:
                try:
                    res.setdefault("metadata", {})["truncated"] = True
                except Exception:
                    pass
            return res
        # res is None (no offline engine) or an offline error dict.
        has_key = bool(_elevenlabs_key())
        if want == "offline" or not has_key:
            if res is not None:
                return res  # honest offline failure (termux/espeak said why)
            return {"output": ("No offline speech engine found. Fix: `pkg install termux-api` "
                               "(Termux:API app) for termux-tts-speak, or `pkg install espeak` "
                               "for the espeak fallback."), "error": True}
        # auto WITH a cloud key: offline missing/failed — fall through to cloud.
    # auto fallback or explicit elevenlabs
    res = _speak_elevenlabs(text, voice_id, model, stability, similarity, rate)
    if truncated and not res.get("error"):
        try:
            res.setdefault("metadata", {})["truncated"] = True
        except Exception:
            pass
    return res


def _action_stop() -> dict:
    stopped: list[str] = []
    if shutil.which("termux-media-player"):
        ok, _ = _run(["termux-media-player", "stop"], timeout=10.0)
        if ok:
            stopped.append("termux-media-player")
    if shutil.which("pkill"):
        ok, _ = _run(["pkill", "-f", "termux-tts-speak"], timeout=10.0)
        if ok:
            stopped.append("termux-tts-speak")
        ok2, _ = _run(["pkill", "-f", "espeak"], timeout=10.0)
        if ok2:
            stopped.append("espeak")
    if stopped:
        return {"output": f"Speech stopped ({', '.join(stopped)}).", "metadata": {"stopped": stopped}}
    return {"output": "Nothing to stop (no player/kill tool found).", "metadata": {"stopped": []}}


def _action_voices() -> dict:
    offline = _offline_players()
    key = _elevenlabs_key()
    lines = [f"Offline engines: {', '.join(offline) if offline else '(none — pkg install termux-api / espeak)'}",
             f"ElevenLabs: {'key set' if key else 'no key (/connect -> ElevenLabs TTS)'}"]
    items: list[dict[str, Any]] = []
    if not key:
        return {"output": "\n".join(lines), "metadata": {"offline": offline, "elevenlabs": []}}
    try:
        import httpx  # deferred: heavy import
    except ImportError:
        return {"output": "\n".join(lines + ["ElevenLabs list needs httpx: `pip install httpx`."]),
                "metadata": {"offline": offline, "elevenlabs": []}, "error": True}
    try:
        resp = httpx.get(_ELEVENLABS_VOICES_URL, headers={"xi-api-key": key}, timeout=30.0)
    except Exception as e:
        return {"output": "\n".join(lines + [f"ElevenLabs list failed: {e}"]),
                "metadata": {"offline": offline, "elevenlabs": []}, "error": True}
    if resp.status_code != 200:
        return {"output": "\n".join(lines + [f"ElevenLabs error {resp.status_code}: {(resp.text or '')[:200]}"]),
                "metadata": {"offline": offline, "elevenlabs": []}, "error": True}
    try:
        data = resp.json()
    except Exception:
        data = {}
    raw_list = data.get("voices") if isinstance(data, dict) else None
    if not isinstance(raw_list, list):
        raw_list = []
    for v in raw_list:
        if isinstance(v, dict) and v.get("voice_id"):
            items.append({"voice_id": str(v.get("voice_id")),
                          "name": str(v.get("name") or ""),
                          "category": str(v.get("category") or "")})
    lines.append(f"ElevenLabs voices ({len(items)}): " +
                 (", ".join(f"{v['name']} [{v['voice_id']}]" for v in items[:20])
                  if items else "(none)"))
    if len(items) > 20:
        lines.append(f"… +{len(items) - 20} more")
    return {"output": "\n".join(lines), "metadata": {"offline": offline, "elevenlabs": items}}


def _action_clone(args: dict) -> dict:
    key = _elevenlabs_key()
    if not key:
        return {"output": ("ElevenLabs key missing. Fix: /connect -> ElevenLabs TTS "
                           "(https://elevenlabs.io/app/settings/api-keys), or set "
                           "ELEVENLABS_API_KEY."), "error": True}
    name = str(args.get("name") or "").strip()
    if not name:
        return {"output": "clone: pass 'name' for the new voice.", "error": True}
    fpath = str(args.get("file") or args.get("audio") or "").strip()
    if not fpath:
        return {"output": "clone: pass 'file' (a ~20s recording: mp3/wav/m4a).", "error": True}
    p = Path(fpath).expanduser()
    if not p.is_file():
        return {"output": f"clone: file not found: {fpath}", "error": True}
    try:
        if p.stat().st_size > _CLONE_MAX_BYTES:
            return {"output": f"clone: file too large ({p.stat().st_size} bytes, max 10 MB).", "error": True}
    except OSError as e:
        return {"output": f"clone: cannot read file: {e}", "error": True}
    try:
        import httpx  # deferred: heavy import
    except ImportError:
        return {"output": "Voice clone needs httpx: `pip install httpx`.", "error": True}
    try:
        with p.open("rb") as fh:
            blob = fh.read()
        files = {"files": (p.name, blob, "application/octet-stream")}
        resp = httpx.post(_ELEVENLABS_ADD_URL, headers={"xi-api-key": key},
                          data={"name": name}, files=files, timeout=120.0)
    except Exception as e:
        return {"output": f"Voice clone failed: {e}", "error": True}
    if resp.status_code not in (200, 201):
        return {"output": f"ElevenLabs clone error {resp.status_code}: {(resp.text or '')[:300]}",
                "error": True}
    vid = ""
    try:
        vid = str((resp.json() or {}).get("voice_id") or "")
    except Exception:
        vid = ""
    hint = (f" Clone {name!r}: voice_id={vid}. Save it: Settings > online voice id, "
            f"or opencode.json tts.voice_id." if vid else f" Cloned {name!r}.")
    return {"output": f"Voice cloned.{hint}", "metadata": {"voice_id": vid, "name": name}}


def _action_status(cfg: Any) -> dict:
    offline = _offline_players()
    key = _elevenlabs_key()
    lines = [
        f"Speak: {'on' if _enabled(cfg) else 'off (Settings > speak)'}",
        f"Auto voice: {'on' if auto_enabled(cfg) else 'off (Settings > auto voice)'}",
        f"Engine: {_opt(cfg, 'engine', 'auto')}",
        f"Offline: {', '.join(offline) if offline else '(none)'}",
        f"ElevenLabs: {'key set' if key else 'no key'}",
        f"Voice: {_opt(cfg, 'voice', '') or '(default)'}  "
        f"Language: {_opt(cfg, 'language', '') or '(default)'}  "
        f"Rate: {_opt(cfg, 'rate', 1.0)}  Pitch: {_opt(cfg, 'pitch', 1.0)}",
        f"Online voice id: {_opt(cfg, 'voice_id', '') or '(default Rachel)'}",
    ]
    return {"output": "\n".join(lines),
            "metadata": {"enabled": _enabled(cfg), "auto": auto_enabled(cfg),
                         "engine": _opt(cfg, "engine", "auto"),
                         "offline": offline, "elevenlabs_key": bool(key)}}


def tool(cfg: Any = None) -> Tool:
    description = """Speak text aloud on this phone (Settings > speak ON). Offline free; ElevenLabs voices need a key. Max 2000 chars."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "speak").strip().lower()
        if action not in _ACTIONS:
            return {"output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                    "error": True}
        if not _enabled(cfg):
            if action == "status":
                return _action_status(cfg)
            return {"output": "Speak is off. Turn it on: Settings > speak.",
                    "metadata": {"enabled": False}, "error": True}
        if action == "speak":
            return _action_speak(input, cfg)
        if action == "stop":
            return _action_stop()
        if action == "voices":
            return _action_voices()
        if action == "clone":
            return _action_clone(input)
        return _action_status(cfg)

    return Tool(
        name="speak",
        description=description,
        parameters=schema_with(
            {
                "action": {"type": "string", "enum": list(_ACTIONS),
                           "description": "speak (default), stop, voices, clone, status", "optional": True},
                "text": {"type": "string", "description": "Text to speak (max 2000)",
                         "optional": True},
                "voice": {"type": "string", "description": "Offline voice (espeak -v)",
                          "optional": True},
                "language": {"type": "string", "description": "Language (Termux TTS -l)",
                             "optional": True},
                "rate": {"type": "number", "description": "Speed 0.25-4.0",
                         "optional": True},
                "pitch": {"type": "number", "description": "Pitch 0.5-2.0",
                          "optional": True},
                "engine": {"type": "string", "enum": list(_ENGINES),
                           "description": "auto, offline, or elevenlabs", "optional": True},
                "voice_id": {"type": "string", "description": "ElevenLabs voice id",
                             "optional": True},
                "model": {"type": "string", "description": "ElevenLabs model", "optional": True},
                "name": {"type": "string", "description": "Clone voice name", "optional": True},
                "file": {"type": "string", "description": "Clone recording path (~20s)", "optional": True},
            },
            [],
        ),
        run=run,
        permission="speak",
    )
