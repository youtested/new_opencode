"""Request trimming: shrink old tool results before sending, keep history intact.

The whole slowdown of turns 2..20 is re-sending every old 8KB file read on
every request. After 2-3 turns the model no longer needs the full bodies —
it needs to know WHAT was read and WHAT was found. So for the REQUEST copy
only (saved history and disk transcripts stay 100% full):

- the last `keep_turns` user turns go verbatim (full bodies, exact ids/args);
- older `role == "tool"` messages become one-line receipts:
  `[trimmed for sending: <name> result, <N> chars — <first 200 chars>]`;
- user + assistant text (questions, answers, decisions) is NEVER touched;
- assistant `tool_calls` declarations are NEVER touched (ids/args must replay
  exactly or strict backends reject the request);
- compaction summary messages are NEVER touched;
- multimodal (list) contents are NEVER touched.

Pairing stays valid (role/tool_call_id/name untouched), so
`repair_tool_pairs` still passes. Shrinking is idempotent: receipts are
short, so a second pass leaves them alone. If the model ever needs a
shrunk body back, it re-reads the file — one fast call, no rework.
"""

from __future__ import annotations

from typing import Any

RECEIPT_PREFIX = "[trimmed for sending:"
RECEIPT_KEEP_CHARS = 200
# Spill receipts share the SAME prefix family so one parser serves trim,
# spill, and free-compact alike. A spilled body keeps a `spill:` key in
# spill_meta; a free-compacted (dropped, re-readable) body keeps a
# `recall:` key instead — the model re-fetches either with the same recall
# tool. ponytail: two key kinds, one tool; upgrade path is merging spill
# files into the session body if spill dirs ever annoy anyone.
RECALL_PREFIX = "[cut for context:"
RECALL_KEEP_CHARS = 200
# Single-store spill: tool bodies past this size are ALSO kept on disk
# (spill dir, keyed by session+index) so the engine can drop its copy and
# re-read on demand (~1.5ms for 34KB). ponytail: fixed 8KB — upgrade path
# is wiring it into opencode.json when someone asks.
SPILL_MIN_CHARS = 8000


def _spill_dir(session_id: str = "") -> Any:
    """Spill directory for big tool bodies (per-session files)."""
    from pathlib import Path as _P

    try:
        from ..globals import Path as _GP

        d = _GP.data / "spill" / str(session_id or "default")
    except Exception:
        import tempfile as _t

        d = _P(_t.gettempdir()) / "opencode_py_spill" / str(session_id or "default")
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def spill_body(session_id: str, index: int, body: str) -> str:
    """Write a big tool body to disk, returning its spill key. Never raises."""
    try:
        d = _spill_dir(session_id)
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        p = d / f"{int(index)}.txt"
        p.write_text(body, encoding="utf-8")
        return f"spill:{session_id}:{int(index)}"
    except Exception:
        return ""


def read_spill(key: str) -> str | None:
    """Read a spilled body back (~1.5ms for 34KB). None on any failure."""
    try:
        parts = str(key or "").split(":")
        if len(parts) != 3 or parts[0] != "spill":
            return None
        _, session_id, index = parts
        p = _spill_dir(session_id) / f"{int(index)}.txt"
        text = p.read_text(encoding="utf-8")
        return text if text else None
    except Exception:
        return None


def clear_spill(session_id: str) -> int:
    """Delete all spill files for a session (on session delete)."""
    try:
        import shutil as _sh

        d = _spill_dir(session_id)
        if not d.exists():
            return 0
        n = sum(1 for _ in d.glob("*.txt"))
        _sh.rmtree(d, ignore_errors=True)
        # rmtree with ignore_errors can leave an empty dir behind (busy
        # file / race): drop it so no empty folders pile up on disk.
        try:
            d.rmdir()
        except OSError:
            pass
        return n
    except Exception:
        return 0


def spill_stats() -> dict:
    """(files, bytes) across all spill dirs. Never raises."""
    try:
        from ..globals import Path as _GP

        root = _GP.data / "spill"
    except Exception:
        return {"files": 0, "bytes": 0}
    files = 0
    total = 0
    try:
        for p in root.rglob("*.txt"):
            try:
                files += 1
                total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return {"files": files, "bytes": total}


def _receipt(name: str, content: str) -> str:
    first = (content or "").strip().split("\n", 1)[0][:RECEIPT_KEEP_CHARS]
    return (
        f"{RECEIPT_PREFIX} {name or 'tool'} result, "
        f"{len(content or '')} chars — first line: {first!r}]"
    )


def shrink_tool_history(
    messages: list[dict[str, Any]],
    keep_turns: int = 2,
    max_chars: int = 500,
) -> list[dict[str, Any]]:
    """Return a send-ready copy with old tool outputs shrunk. Never raises."""
    try:
        keep_turns = max(1, int(keep_turns))
        max_chars = max(100, int(max_chars))
    except (TypeError, ValueError):
        keep_turns, max_chars = 2, 500
    try:
        # newest user-turn starts, walking back; compaction summaries count
        # as keep-worthy (they ARE the compressed past).
        starts: list[int] = []
        for i, m in enumerate(messages):
            try:
                if isinstance(m, dict) and m.get("role") == "user" and not m.get("compaction"):
                    starts.append(i)
            except Exception:
                continue
        cutoff = starts[-keep_turns] if len(starts) >= keep_turns else 0
        out: list[dict[str, Any]] = []
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or i >= cutoff:
                out.append(m)
                continue
            try:
                if m.get("role") != "tool":
                    out.append(m)
                    continue
                content = m.get("content")
                if not isinstance(content, str):
                    out.append(m)  # multimodal: leave alone
                    continue
                if len(content) <= max_chars or content.startswith(RECEIPT_PREFIX):
                    out.append(m)
                    continue
                slim = dict(m)
                slim["content"] = _receipt(str(m.get("name", "tool")), content)
                out.append(slim)
            except Exception:
                out.append(m)
        return out
    except Exception:
        return messages


def spill_old_tools(
    messages: list[dict[str, Any]],
    session_id: str = "",
    keep_turns: int = 2,
    min_chars: int = SPILL_MIN_CHARS,
) -> tuple[list[dict[str, Any]], int]:
    """Single-store pass: spill big OLD tool bodies to disk, keep receipts.

    Same keep-turn rule as shrink_tool_history (last keep_turns go verbatim),
    but PERSISTENT: the body moves to a spill file and the in-memory message
    keeps a one-line receipt with a `spill:` key. rehydrate_spills() restores
    any body on demand (~1.5ms). Disk session saves stay FULL (see
    rehydrate before save); only live RAM drops. Returns (messages, spilled).
    Never raises.
    """
    try:
        keep_turns = max(1, int(keep_turns))
        min_chars = max(1000, int(min_chars))
    except (TypeError, ValueError):
        keep_turns, min_chars = 2, SPILL_MIN_CHARS
    try:
        starts: list[int] = []
        for i, m in enumerate(messages):
            try:
                if isinstance(m, dict) and m.get("role") == "user" and not m.get("compaction"):
                    starts.append(i)
            except Exception:
                continue
        cutoff = starts[-keep_turns] if len(starts) >= keep_turns else 0
        out: list[dict[str, Any]] = []
        spilled = 0
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or i >= cutoff:
                out.append(m)
                continue
            try:
                if m.get("role") != "tool":
                    out.append(m)
                    continue
                content = m.get("content")
                if not isinstance(content, str):
                    out.append(m)
                    continue
                if len(content) < min_chars or content.startswith(RECEIPT_PREFIX):
                    out.append(m)
                    continue
                key = spill_body(session_id, i, content)
                if not key:
                    out.append(m)
                    continue
                slim = dict(m)
                slim["content"] = _receipt(str(m.get("name", "tool")), content)
                try:
                    meta = dict(slim.get("spill_meta") or {})
                except Exception:
                    meta = {}
                meta["key"] = key
                meta["chars"] = len(content)
                slim["spill_meta"] = meta
                spilled += 1
                out.append(slim)
            except Exception:
                out.append(m)
        return out, spilled
    except Exception:
        return messages, 0


def rehydrate_spills(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore spilled bodies in place (before save/export/send-full)."""
    try:
        out: list[dict[str, Any]] = []
        for m in messages:
            try:
                if not isinstance(m, dict):
                    out.append(m)
                    continue
                meta = m.get("spill_meta")
                if not isinstance(meta, dict):
                    out.append(m)
                    continue
                key = str(meta.get("key") or "")
                body = read_spill(key) if key else None
                if body is None:
                    out.append(m)
                    continue
                full = dict(m)
                full["content"] = body
                try:
                    del full["spill_meta"]
                except KeyError:
                    pass
                out.append(full)
            except Exception:
                out.append(m)
        return out
    except Exception:
        return messages


def _recall_note(name: str, content: str, key: str) -> str:
    """One-line recall note: what was cut + how to fetch it back (~200 chars)."""
    first = (content or "").strip().split("\n", 1)[0][:RECALL_KEEP_CHARS]
    return (
        f"{RECALL_PREFIX} {name or 'tool'} result, "
        f"{len(content or '')} chars — first line: {first!r}. "
        f"To read it back call recall with recall_key={key!r}.]"
    )


def free_compact(
    messages: list[dict[str, Any]],
    session_id: str = "",
    keep_turns: int = 2,
    min_chars: int = 2000,
) -> tuple[list[dict[str, Any]], int]:
    """Free rule-cut: drop big OLD tool bodies WITHOUT a model call.

    Same keep-turn rule as the request trim (last keep_turns go verbatim),
    but the cut body is NOT summarized — it stays in the session file on
    disk (get_history rehydrates before save) and the in-memory message
    keeps a recall note with a `recall:` key. The model re-fetches any cut
    body with the recall tool (~1ms). User + assistant text is NEVER
    touched; tool_calls declarations NEVER touched; compaction summaries
    NEVER touched. Returns (messages, cut). Never raises.
    """
    try:
        keep_turns = max(1, int(keep_turns))
        min_chars = max(500, int(min_chars))
    except (TypeError, ValueError):
        keep_turns, min_chars = 2, 2000
    try:
        starts: list[int] = []
        for i, m in enumerate(messages):
            try:
                if isinstance(m, dict) and m.get("role") == "user" and not m.get("compaction"):
                    starts.append(i)
            except Exception:
                continue
        cutoff = starts[-keep_turns] if len(starts) >= keep_turns else 0
        out: list[dict[str, Any]] = []
        cut = 0
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or i >= cutoff:
                out.append(m)
                continue
            try:
                if m.get("role") != "tool":
                    out.append(m)
                    continue
                content = m.get("content")
                if not isinstance(content, str):
                    out.append(m)
                    continue
                if len(content) < min_chars:
                    out.append(m)
                    continue
                if content.startswith(RECEIPT_PREFIX) or content.startswith(RECALL_PREFIX):
                    out.append(m)
                    continue
                key = f"recall:{session_id}:{int(i)}"
                slim = dict(m)
                slim["content"] = _recall_note(str(m.get("name", "tool")), content, key)
                try:
                    meta = dict(slim.get("recall_meta") or {})
                except Exception:
                    meta = {}
                meta["key"] = key
                meta["chars"] = len(content)
                meta["index"] = int(i)
                slim["recall_meta"] = meta
                cut += 1
                out.append(slim)
            except Exception:
                out.append(m)
        return out, cut
    except Exception:
        return messages, 0


def rehydrate_recalls(
    messages: list[dict[str, Any]],
    full: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Restore free-cut bodies from the FULL history (disk save / export)."""
    try:
        if full is None:
            return messages
        by_index: dict[int, str] = {}
        for idx, m in enumerate(full):
            try:
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    by_index[idx] = str(m.get("content"))
            except Exception:
                continue
        out: list[dict[str, Any]] = []
        for m in messages:
            try:
                if not isinstance(m, dict):
                    out.append(m)
                    continue
                meta = m.get("recall_meta")
                if not isinstance(meta, dict):
                    out.append(m)
                    continue
                idx = meta.get("index")
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    out.append(m)
                    continue
                body = by_index.get(idx)
                if body is None:
                    out.append(m)
                    continue
                full_m = dict(m)
                full_m["content"] = body
                try:
                    del full_m["recall_meta"]
                except KeyError:
                    pass
                out.append(full_m)
            except Exception:
                out.append(m)
        return out
    except Exception:
        return messages


def request_prefix_key(system_prompt: str, tools: list[dict]) -> str:
    """Stable hash of the cacheable request prefix (system + tool schemas).

    Providers with prefix caching reuse the prefix when it is byte-identical
    across turns; this key lets tests/logs assert that stability. Never raises.
    """
    try:
        import hashlib
        import json

        names = sorted(
            str((t.get("function") or {}).get("name", "")) for t in (tools or []) if isinstance(t, dict)
        )
        blob = system_prompt + "\x00" + "\x00".join(names)
        return hashlib.sha1(blob.encode("utf-8", errors="replace")).hexdigest()[:16]
    except Exception:
        return ""
