"""browser tool: drive the phone's own Chrome app via CDP.

RULES (full guidance for maintainers; the sent schema is short):
- One tool, many actions: status/connect/open/tabs/goto/snapshot/click/
  type/press/scroll/wait/screenshot/js/close. Keeps the model schema small.
- Transport: httpx for http://127.0.0.1:9222/json (tab list, open, close),
  stdlib socket for the CDP WebSocket (no new deps: websockets/ is NOT in
  requirements.txt, so a raw client keeps armv7 purity).
- One WS connection per CDP call (open, use, close): stateless, survives
  Chrome restarts and doze without stale-socket bugs. Localhost cost is ms.
- Element refs: snapshot stores descriptors per target; click/type resolve
  ref -> selector -> visible text. Never trust a ref from another tab.
- goto allows http/https only. js is capped and timed out.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
from typing import Any

from ..globals import Path as GPath
from .registry import Tool, schema_with

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 9222

_WS_LOCK = threading.Lock()
_CDP_RID = [0]
_LAST_SNAPSHOT: dict[str, list[dict]] = {}
_SNAPSHOT_LOCK = threading.Lock()

_ACTIONS = (
    "status",
    "connect",
    "open",
    "tabs",
    "goto",
    "snapshot",
    "click",
    "type",
    "press",
    "scroll",
    "wait",
    "screenshot",
    "js",
    "close",
)

_READONLY_ACTIONS = ("status", "tabs", "snapshot", "screenshot")
READONLY_ACTIONS = _READONLY_ACTIONS

_KEY_CODES = {
    "enter": 13,
    "tab": 9,
    "escape": 27,
    "esc": 27,
    "backspace": 8,
    "delete": 46,
    "arrowleft": 37,
    "arrowup": 38,
    "arrowright": 39,
    "arrowdown": 40,
    "left": 37,
    "up": 38,
    "right": 39,
    "down": 40,
    "space": 32,
    "home": 36,
    "end": 35,
}

_SETUP_HINT = (
    "Phone Chrome is not reachable on 127.0.0.1:9222. One-time setup:\n"
    "1. `pkg install android-tools` (Termux).\n"
    "2. Android Settings > Developer options > Wireless debugging ON > Pair "
    "with pairing code; then `adb pair <host>:<port> <code>`.\n"
    "3. `adb forward tcp:9222 localabstract:chrome_devtools_remote`.\n"
    "4. Open Chrome once, then retry. Re-run `browser connect` after reboot."
)


def _endpoint(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _validate_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        raise ValueError("url is required")
    if not u.startswith(("http://", "https://")):
        raise ValueError(f"only http/https allowed, got {u[:40]!r}")
    if len(u) > 4000:
        raise ValueError("url too long (max 4000 chars)")
    return u


def _ws_url_parts(ws_url: str) -> tuple[str, int, str]:
    if not ws_url.startswith("ws://"):
        raise ValueError(f"unsupported debugger url {ws_url[:40]!r}")
    rest = ws_url[len("ws://"):]
    host_port, _, path = rest.partition("/")
    host, _, port_s = host_port.partition(":")
    return host or "127.0.0.1", int(port_s or "80"), "/" + path


class _RawWS:
    """Minimal stdlib WebSocket client (text frames, localhost CDP)."""

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def connect(self, ws_url: str) -> None:
        host, port, path = _ws_url_parts(ws_url)
        key = base64.b64encode(os.urandom(16)).decode()
        sock = socket.create_connection((host, port), timeout=self.timeout)
        req = (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(req.encode())
        sock.settimeout(self.timeout)
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                sock.close()
                raise ConnectionError("cdp handshake: connection closed")
            buf += chunk
            if len(buf) > 65536:
                sock.close()
                raise ConnectionError("cdp handshake: header too large")
        head = buf.split(b"\r\n", 1)[0].decode("latin1")
        if "101" not in head:
            sock.close()
            raise ConnectionError(f"cdp handshake failed: {head[:80]}")
        self.sock = sock

    def send_text(self, text: str) -> None:
        assert self.sock is not None
        data = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytes([0x81])
        n = len(data)
        if n < 126:
            header += struct.pack("!B", 0x80 | n)
        elif n < 65536:
            header += struct.pack("!BH", 0x80 | 126, n)
        else:
            header += struct.pack("!BQ", 0x80 | 127, n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(header + mask + masked)

    def _recv_exact(self, n: int) -> bytes:
        assert self.sock is not None
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("cdp socket closed mid-frame")
            buf += chunk
        return buf

    def recv_text(self) -> str:
        hdr = self._recv_exact(2)
        opcode = hdr[0] & 0x0F
        length = hdr[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        if hdr[1] & 0x80:
            self._recv_exact(4)
        payload = self._recv_exact(length) if length else b""
        if opcode == 0x8:
            raise ConnectionError("cdp socket closed by peer")
        if opcode == 0x9:
            assert self.sock is not None
            self.sock.sendall(bytes([0x8A, 0x00]))
            return self.recv_text()
        parts = [payload]
        while hdr[0] & 0x80 == 0:
            hdr = self._recv_exact(2)
            length = hdr[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            if hdr[1] & 0x80:
                self._recv_exact(4)
            parts.append(self._recv_exact(length) if length else b"")
        return b"".join(parts).decode("utf-8", errors="replace")

    def close(self) -> None:
        sock, self.sock = self.sock, None
        if sock is not None:
            try:
                sock.sendall(bytes([0x88, 0x00]))
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass


def _cdp_call(ws_url: str, method: str, params: dict | None = None,
              timeout: float = 20.0) -> dict:
    ws = _RawWS(timeout=timeout)
    ws.connect(ws_url)
    try:
        with _WS_LOCK:
            _CDP_RID[0] += 1
            rid = _CDP_RID[0]
        ws.send_text(json.dumps({"id": rid, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"cdp {method} timed out after {timeout}s")
            msg = json.loads(ws.recv_text())
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                err = msg["error"]
                raise RuntimeError(f"cdp {method}: {err.get('message', err)}")
            return msg.get("result") or {}
    finally:
        ws.close()


def _http_targets(host: str, port: int, timeout: float = 10.0) -> list[dict]:
    import httpx

    resp = httpx.get(f"{_endpoint(host, port)}/json/list", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _pick_target(targets: list[dict], target_id: str = "") -> dict:
    pages = [t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
    if not pages:
        raise RuntimeError("no open Chrome tabs found (open Chrome first)")
    if target_id:
        for t in pages:
            if t.get("id") == target_id or (t.get("id") or "").startswith(target_id):
                return t
        raise ValueError(f"unknown tab {target_id!r}")
    for t in pages:
        url = str(t.get("url") or "")
        if not url.startswith("chrome"):
            return t
    return pages[0]


_SNAPSHOT_JS = """(() => {
  const els = [...document.querySelectorAll(
    'a,button,input,select,textarea,[role=button],[role=link],[role=textbox]')];
  const vis = els.filter(e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }).slice(0, 100);
  return {
    url: location.href, title: document.title,
    nodes: vis.map(e => ({
      tag: e.tagName.toLowerCase(),
      type: e.getAttribute('type') || '',
      text: (e.innerText || e.value || e.getAttribute('aria-label')
             || e.getAttribute('placeholder') || '').trim().slice(0, 80),
      sel: e.id ? '#' + e.id : (e.name ? `[name="${e.name}"]` : ''),
    })),
  };
})()"""


def _build_snapshot_js() -> str:
    return _SNAPSHOT_JS


def _format_nodes(url: str, title: str, nodes: list[dict]) -> str:
    lines = [f"{title or '(no title)'} — {url}", f"{len(nodes)} interactive element(s):"]
    for i, n in enumerate(nodes):
        label = str(n.get("text") or n.get("type") or n.get("tag") or "").strip()
        lines.append(f"[{i}] <{n.get('tag', '?')}> {label}"[:140])
    lines.append("Use click/type with ref=N.")
    return "\n".join(lines)


def _remember_snapshot(target_id: str, nodes: list[dict]) -> None:
    with _SNAPSHOT_LOCK:
        _LAST_SNAPSHOT[target_id] = list(nodes)


def _lookup_ref(target_id: str, ref: int) -> dict | None:
    with _SNAPSHOT_LOCK:
        nodes = list(_LAST_SNAPSHOT.get(target_id) or [])
    if 0 <= ref < len(nodes):
        return nodes[ref]
    return None


_CLICK_JS = """(sel, idx, text) => {
  let el = null;
  if (sel) { try { el = document.querySelector(sel); } catch (e) {} }
  if (!el && idx >= 0) {
    const els = [...document.querySelectorAll(
      'a,button,input,select,textarea,[role=button],[role=link]')];
    const vis = els.filter(e => {
      const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0;
    });
    el = vis[idx] || null;
  }
  if (!el && text) {
    const els = [...document.querySelectorAll('a,button,[role=button]')];
    const t = text.toLowerCase();
    el = els.find(e => (e.innerText || '').toLowerCase().includes(t)) || null;
  }
  if (!el) return {ok: false, error: 'element not found'};
  el.scrollIntoView({block: 'center'});
  el.click();
  return {ok: true, tag: el.tagName.toLowerCase(),
          text: (el.innerText || '').trim().slice(0, 80)};
}"""

_TYPE_JS = """(sel, idx, text) => {
  let el = null;
  if (sel) { try { el = document.querySelector(sel); } catch (e) {} }
  if (!el && idx >= 0) {
    const els = [...document.querySelectorAll('input,textarea,select,[role=textbox]')];
    const vis = els.filter(e => {
      const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0;
    });
    el = vis[idx] || null;
  }
  if (!el) {
    el = document.activeElement;
    if (!el || !/INPUT|TEXTAREA|SELECT/.test(el.tagName)) return {ok: false, error: 'field not found'};
  }
  el.focus();
  return {ok: true, key: true,
          tag: el.tagName.toLowerCase(), sel: sel || '',
          needText: text};
}"""


def _eval(ws_url: str, expression: str, timeout: float = 20.0) -> Any:
    res = _cdp_call(ws_url, "Runtime.evaluate",
                    {"expression": expression, "returnByValue": True,
                     "awaitPromise": True}, timeout=timeout)
    remote = res.get("result") or {}
    if remote.get("subtype") == "error" or res.get("exceptionDetails"):
        detail = res.get("exceptionDetails") or {}
        raise RuntimeError(f"page error: {detail.get('text', remote.get('description', 'js failed'))}")
    return remote.get("value")


def _eval_obj(ws_url: str, expression: str, timeout: float = 20.0) -> Any:
    """_eval that JSON-parses a string payload (mock/edge tolerance)."""
    value = _eval(ws_url, expression, timeout=timeout)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _a_status(host: str, port: int) -> dict:
    import shutil
    import subprocess

    adb = shutil.which("adb")
    try:
        targets = _http_targets(host, port)
        pages = [t for t in targets if t.get("type") == "page"]
        lines = [f"Chrome reachable at {host}:{port}: {len(pages)} tab(s)."]
        for t in pages[:10]:
            lines.append(f"- {t.get('title', '')[:60]} :: {str(t.get('url', ''))[:80]}")
        return {"output": "\n".join(lines),
                "metadata": {"reachable": True, "tabs": len(pages),
                             "host": host, "port": port}}
    except Exception:
        out = [f"Chrome NOT reachable at {host}:{port}.", _SETUP_HINT]
        if adb is None:
            out.append("Note: `adb` not installed — `pkg install android-tools` first.")
        else:
            try:
                proc = subprocess.run(["adb", "forward", "--list"],
                                      capture_output=True, text=True, timeout=10)
                forwards = (proc.stdout or "").strip()
                out.append("adb forwards now:\n" + (forwards or "(none)"))
            except Exception:
                pass
        return {"output": "\n".join(out),
                "metadata": {"reachable": False, "host": host, "port": port}, "error": True}


def _a_connect(host: str, port: int) -> dict:
    import shutil
    import subprocess

    adb = shutil.which("adb")
    if adb is None:
        return {"output": "adb not found. `pkg install android-tools`, then pair "
                "Wireless debugging and retry.\n" + _SETUP_HINT, "error": True}
    try:
        proc = subprocess.run(
            ["adb", "forward", f"tcp:{port}", "localabstract:chrome_devtools_remote"],
            capture_output=True, text=True, timeout=30)
    except Exception as e:
        return {"output": f"adb forward failed: {e}", "error": True}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "no devices" in err.lower() or "device" in err.lower():
            return {"output": f"adb sees no device paired. Pair Wireless debugging "
                    f"first (`adb pair <host>:<port> <code>`), then retry.\nDetail: {err}",
                    "error": True}
        return {"output": f"adb forward failed: {err or proc.returncode}", "error": True}
    try:
        targets = _http_targets(host, port)
        n = len([t for t in targets if t.get("type") == "page"])
        return {"output": f"Connected: Chrome reachable at {host}:{port} ({n} tab(s)).",
                "metadata": {"reachable": True, "tabs": n}}
    except Exception:
        return {"output": "Port forwarded but Chrome did not answer. Open Chrome "
                "once on the phone, then retry.", "error": True}


def _a_open(url: str, host: str, port: int) -> dict:
    import httpx

    target_url = _validate_url(url)
    try:
        resp = httpx.put(f"{_endpoint(host, port)}/json/new",
                         params={"url": target_url}, timeout=15.0)
        resp.raise_for_status()
        t = resp.json()
    except Exception as e:
        return {"output": f"open failed: {e}\n{_SETUP_HINT}", "error": True}
    return {"output": f"Opened: {t.get('title', '') or target_url}\n{tab_line(t)}",
            "metadata": {"targetId": t.get("id", ""), "url": t.get("url", target_url)}}


def tab_line(t: dict) -> str:
    return f"tab {t.get('id', '')[:8]} :: {str(t.get('url', ''))[:100]}"


def _resolve_target(host: str, port: int, target_id: str = "") -> dict:
    try:
        targets = _http_targets(host, port)
    except Exception as e:
        raise RuntimeError(f"chrome not reachable: {e}\n{_SETUP_HINT}") from e
    return _pick_target(targets, target_id)


def run(input: dict) -> dict:
    action = str(input.get("action") or "").strip().lower()
    if action not in _ACTIONS:
        return {"output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True}
    host = str(input.get("host") or _DEFAULT_HOST).strip() or _DEFAULT_HOST
    try:
        port = int(input.get("port") or _DEFAULT_PORT)
    except (TypeError, ValueError):
        return {"output": "port must be a number", "error": True}

    if action == "status":
        return _a_status(host, port)
    if action == "connect":
        return _a_connect(host, port)
    if action == "open":
        try:
            return _a_open(str(input.get("url") or ""), host, port)
        except ValueError as e:
            return {"output": str(e), "error": True}
    if action == "tabs":
        try:
            targets = _http_targets(host, port)
        except Exception as e:
            return {"output": f"tabs failed: {e}\n{_SETUP_HINT}", "error": True}
        pages = [t for t in targets if t.get("type") == "page"]
        if not pages:
            return {"output": "No open tabs. Use action=open first."}
        lines = [f"{len(pages)} tab(s):"]
        for t in pages:
            lines.append(f"- {tab_line(t)} :: {(t.get('title') or '')[:60]}")
        return {"output": "\n".join(lines), "metadata": {"tabs": len(pages)}}
    if action == "wait":
        try:
            ms = int(input.get("ms") or 1000)
        except (TypeError, ValueError):
            return {"output": "ms must be a number", "error": True}
        ms = max(100, min(ms, 30000))
        time.sleep(ms / 1000.0)
        return {"output": f"Waited {ms} ms."}
    if action == "close":
        import httpx

        try:
            target = _resolve_target(host, port, str(input.get("targetId") or ""))
        except Exception as e:
            return {"output": str(e), "error": True}
        try:
            resp = httpx.get(f"{_endpoint(host, port)}/json/close/{target['id']}",
                             timeout=10.0)
            ok = resp.status_code == 200
        except Exception as e:
            return {"output": f"close failed: {e}", "error": True}
        return {"output": f"Closed tab {target['id'][:8]}." if ok else "Close not confirmed."}

    try:
        target = _resolve_target(host, port, str(input.get("targetId") or ""))
    except Exception as e:
        return {"output": str(e), "error": True}
    ws_url = target.get("webSocketDebuggerUrl") or ""
    tid = target.get("id") or ""
    timeout = 25.0

    if action == "goto":
        try:
            url = _validate_url(str(input.get("url") or ""))
        except ValueError as e:
            return {"output": str(e), "error": True}
        try:
            _cdp_call(ws_url, "Page.navigate", {"url": url}, timeout=timeout)
        except Exception as e:
            return {"output": f"goto failed: {e}", "error": True}
        time.sleep(1.0)
        return {"output": f"Navigated to {url}", "metadata": {"targetId": tid, "url": url}}

    if action == "snapshot":
        try:
            value = _eval(ws_url, _build_snapshot_js(), timeout=timeout)
        except Exception as e:
            return {"output": f"snapshot failed: {e}", "error": True}
        if not isinstance(value, dict):
            return {"output": "snapshot failed: unexpected page reply", "error": True}
        nodes = value.get("nodes") or []
        _remember_snapshot(tid, nodes if isinstance(nodes, list) else [])
        return {"output": _format_nodes(str(value.get("url") or target.get("url") or ""),
                                       str(value.get("title") or target.get("title") or ""),
                                       nodes if isinstance(nodes, list) else []),
                "metadata": {"targetId": tid, "elements": len(nodes)}}

    if action == "click":
        ref = input.get("ref")
        sel = str(input.get("selector") or "")
        text = str(input.get("text") or "")
        idx = int(ref) if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()) else -1
        node = _lookup_ref(tid, idx) if idx >= 0 else None
        if node and node.get("sel"):
            sel = sel or str(node["sel"])
        try:
            value = _eval_obj(ws_url, f"({_CLICK_JS})({json.dumps(sel)}, {idx}, {json.dumps(text)})",
                              timeout=timeout)
        except Exception as e:
            return {"output": f"click failed: {e}", "error": True}
        if not isinstance(value, dict) or not value.get("ok"):
            err = value.get("error", "element not found") if isinstance(value, dict) else "bad reply"
            return {"output": f"click failed: {err}. Run snapshot first for ref numbers.",
                    "error": True}
        return {"output": f"Clicked <{value.get('tag', '?')}> {str(value.get('text', ''))[:80]}",
                "metadata": {"targetId": tid}}

    if action == "type":
        text = str(input.get("text") or "")
        if not text:
            return {"output": "type needs text", "error": True}
        if len(text) > 2000:
            return {"output": "text too long (max 2000 chars)", "error": True}
        ref = input.get("ref")
        sel = str(input.get("selector") or "")
        idx = int(ref) if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()) else -1
        node = _lookup_ref(tid, idx) if idx >= 0 else None
        if node and node.get("sel"):
            sel = sel or str(node["sel"])
        try:
            value = _eval_obj(ws_url, f"({_TYPE_JS})({json.dumps(sel)}, {idx}, {json.dumps(text)})",
                              timeout=timeout)
        except Exception as e:
            return {"output": f"type failed: {e}", "error": True}
        if not isinstance(value, dict) or not value.get("ok"):
            err = value.get("error", "field not found") if isinstance(value, dict) else "bad reply"
            return {"output": f"type failed: {err}. Run snapshot first.", "error": True}
        try:
            _cdp_call(ws_url, "Input.insertText", {"text": text}, timeout=timeout)
        except Exception as e:
            return {"output": f"type focus ok, insert failed: {e}", "error": True}
        submit = bool(input.get("submit"))
        if submit:
            try:
                _cdp_call(ws_url, "Input.dispatchKeyEvent",
                           {"type": "rawKeyDown", "windowsVirtualKeyCode": 13,
                            "key": "Enter", "code": "Enter", "text": "\r"}, timeout=timeout)
            except Exception:
                pass
        return {"output": f"Typed {len(text)} char(s){' + Enter' if submit else ''}.",
                "metadata": {"targetId": tid}}

    if action == "press":
        key = str(input.get("key") or "Enter").strip().lower()
        if len(key) == 1:
            params: dict[str, Any] = {"type": "char", "text": key, "key": key}
        elif key in _KEY_CODES:
            code = _KEY_CODES[key]
            name = "Enter" if key == "enter" else key.capitalize()
            params = {"type": "rawKeyDown", "windowsVirtualKeyCode": code,
                      "key": name, "code": name}
            if key == "enter":
                params["text"] = "\r"
        else:
            return {"output": f"unknown key {key!r} (use Enter/Tab/Escape/arrows or one char)",
                    "error": True}
        try:
            _cdp_call(ws_url, "Input.dispatchKeyEvent", params, timeout=timeout)
            if params["type"] == "rawKeyDown":
                _cdp_call(ws_url, "Input.dispatchKeyEvent",
                           {"type": "keyUp", "key": params["key"], "code": params.get("code", ""),
                            "windowsVirtualKeyCode": params.get("windowsVirtualKeyCode", 0)},
                           timeout=timeout)
        except Exception as e:
            return {"output": f"press failed: {e}", "error": True}
        return {"output": f"Pressed {key}."}

    if action == "scroll":
        direction = str(input.get("direction") or "").strip().lower()
        try:
            dx = int(input.get("dx") or 0)
            dy = int(input.get("dy") or 0)
        except (TypeError, ValueError):
            return {"output": "dx/dy must be numbers", "error": True}
        if direction in ("down", "up", "left", "right"):
            dist = 600
            dx, dy = {"down": (0, dist), "up": (0, -dist),
                      "right": (dist, 0), "left": (-dist, 0)}[direction]
        expr = f"window.scrollBy({dx}, {dy}); JSON.stringify({{x: window.scrollX, y: window.scrollY}})"
        try:
            value = _eval(ws_url, expr, timeout=timeout)
        except Exception as e:
            return {"output": f"scroll failed: {e}", "error": True}
        return {"output": f"Scrolled to {value}." if value else "Scrolled."}

    if action == "screenshot":
        try:
            res = _cdp_call(ws_url, "Page.captureScreenshot", {"format": "png"}, timeout=timeout)
        except Exception as e:
            return {"output": f"screenshot failed: {e}", "error": True}
        b64 = res.get("data") or ""
        if not b64:
            return {"output": "screenshot failed: empty reply", "error": True}
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            return {"output": f"screenshot decode failed: {e}", "error": True}
        out_dir = GPath.data / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"browser_{int(time.time())}.png"
        try:
            out_path.write_bytes(raw)
        except OSError as e:
            return {"output": f"screenshot save failed: {e}", "error": True}
        kb = len(raw) // 1024
        return {"output": f"Screenshot saved: {out_path} ({kb} KB). "
                "Open it with the read tool.",
                "metadata": {"path": str(out_path), "bytes": len(raw), "targetId": tid}}

    if action == "js":
        expr = str(input.get("expression") or "")
        if not expr.strip():
            return {"output": "js needs expression", "error": True}
        if len(expr) > 4000:
            return {"output": "expression too long (max 4000 chars)", "error": True}
        lowered = expr.lower()
        if "while(true)" in lowered.replace(" ", "") or "for(;;)" in lowered.replace(" ", ""):
            return {"output": "refused: endless loop pattern", "error": True}
        try:
            value = _eval(ws_url, expr, timeout=timeout)
        except Exception as e:
            return {"output": f"js failed: {e}", "error": True}
        text = json.dumps(value, ensure_ascii=False)[:4000] if not isinstance(value, str) else value[:4000]
        return {"output": text if text else "(no return value)", "metadata": {"targetId": tid}}

    return {"output": f"Unhandled action {action!r}", "error": True}


def tool() -> Tool:
    description = (
        "Drive the phone's Chrome: status/connect, open/tabs/goto, snapshot "
        "(list buttons+fields), click/type/press/scroll, screenshot, js, close. "
        "Run snapshot first to get ref numbers."
    )

    return Tool(
        name="browser",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "status, connect, open, tabs, goto, snapshot, click, "
                                   "type, press, scroll, wait, screenshot, js, close",
                },
                "url": {"type": "string", "description": "URL for open/goto", "optional": True},
                "ref": {"type": "integer", "description": "Element ref from snapshot", "optional": True},
                "selector": {"type": "string", "description": "CSS selector (alt to ref)",
                             "optional": True},
                "text": {"type": "string", "description": "Text for type/click-search",
                         "optional": True},
                "key": {"type": "string", "description": "Key for press (Enter/Tab/char)",
                        "optional": True},
                "direction": {"type": "string", "description": "down/up/left/right",
                              "optional": True},
                "dx": {"type": "integer", "description": "Scroll X px", "optional": True},
                "dy": {"type": "integer", "description": "Scroll Y px", "optional": True},
                "ms": {"type": "integer", "description": "Wait ms", "optional": True},
                "expression": {"type": "string", "description": "JS for js action", "optional": True},
                "targetId": {"type": "string", "description": "Tab id (default current)",
                             "optional": True},
                "submit": {"type": "boolean", "description": "Press Enter after type",
                           "optional": True},
                "host": {"type": "string", "description": "CDP host (default 127.0.0.1)",
                         "optional": True},
                "port": {"type": "integer", "description": "CDP port (default 9222)",
                         "optional": True},
            },
            ["action"],
        ),
        run=run,
        permission="browser",
    )


__all__ = ["tool", "run", "READONLY_ACTIONS"]
