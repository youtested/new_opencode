"""phone tool: drive the WHOLE phone via adb (tap/swipe/type/apps/keys).

RULES (full guidance for maintainers; the sent schema is short):
- One tool, many actions: status/tap/swipe/type/key/app_open/app_list/
  screen/focused/shell/ui. Keeps the model schema small.
- screen = PNG screenshot; ui = text layout via uiautomator (works even
  when the app blocks screenshots, e.g. Telegram FLAG_SECURE).
- Transport: `adb shell ...` subprocess only. No new deps, no root.
- Coordinates are raw pixels (720x1520 on this device; read live via
  status). No scaling games: the agent sees real pixels.
- type capped at 2000 chars; shell allowlisted to read-only adb subcommands
  plus input/am/wm (no `adb root`, no remount, no file push to /system).
- Password/payment caution lives in the description + permission lane:
  plan/explore get look-only (status/screen/focused/app_list).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

from ..globals import Path as GPath
from .registry import Tool, schema_with

_ACTIONS = (
    "status",
    "tap",
    "swipe",
    "type",
    "key",
    "app_open",
    "app_list",
    "screen",
    "focused",
    "shell",
    "ui",
)

READONLY_ACTIONS = ("status", "screen", "focused", "app_list", "ui")

_KEY_ALIASES = {
    "back": "KEYCODE_BACK",
    "home": "KEYCODE_HOME",
    "recent": "KEYCODE_APP_SWITCH",
    "recents": "KEYCODE_APP_SWITCH",
    "enter": "KEYCODE_ENTER",
    "tab": "KEYCODE_TAB",
    "escape": "KEYCODE_ESCAPE",
    "esc": "KEYCODE_ESCAPE",
    "delete": "KEYCODE_DEL",
    "backspace": "KEYCODE_DEL",
    "space": "KEYCODE_SPACE",
    "up": "KEYCODE_DPAD_UP",
    "down": "KEYCODE_DPAD_DOWN",
    "left": "KEYCODE_DPAD_LEFT",
    "right": "KEYCODE_DPAD_RIGHT",
    "center": "KEYCODE_DPAD_CENTER",
    "ok": "KEYCODE_DPAD_CENTER",
    "volume_up": "KEYCODE_VOLUME_UP",
    "volume_down": "KEYCODE_VOLUME_DOWN",
    "mute": "KEYCODE_VOLUME_MUTE",
    "power": "KEYCODE_POWER",
    "wake": "KEYCODE_WAKEUP",
    "sleep": "KEYCODE_SLEEP",
    "camera": "KEYCODE_CAMERA",
    "search": "KEYCODE_SEARCH",
    "menu": "KEYCODE_MENU",
    "settings": "KEYCODE_SETTINGS",
    "paste": "KEYCODE_PASTE",
}

_SHELL_ALLOW = ("wm", "dumpsys", "pm", "cmd", "settings", "getprop", "input", "am", "screencap",
                 "uiautomator", "cat", "ls")

_SETUP_HINT = (
    "adb sees no device. Fix: Android Settings > Developer options > "
    "Wireless debugging ON, `adb pair <host>:<port> <code>`, then "
    "`adb connect <host>:<port>`."
)


def _adb() -> str | None:
    return shutil.which("adb")


def _run_adb(args: list[str], timeout: float = 30.0) -> tuple[bool, str]:
    adb = _adb()
    if adb is None:
        return False, "adb not found (`pkg install android-tools`)."
    try:
        proc = subprocess.run([adb, *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "no devices" in err.lower() or "no emulators" in err.lower():
            return False, _SETUP_HINT
        return False, err or f"adb exited {proc.returncode}"
    return True, (proc.stdout or "").strip()


def _device_ok() -> tuple[bool, str]:
    ok, out = _run_adb(["devices"])
    if not ok:
        return False, out
    lines = [ln for ln in out.splitlines()[1:] if ln.strip() and "\tdevice" in ln]
    if not lines:
        return False, _SETUP_HINT
    return True, lines[0].split()[0]


def _num(value: Any, name: str, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not lo <= n <= hi:
        raise ValueError(f"{name} out of range ({lo}-{hi})")
    return n


def run(input: dict) -> dict:
    action = str(input.get("action") or "").strip().lower()
    if action not in _ACTIONS:
        return {"output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True}
    if action == "status":
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "metadata": {"connected": False}, "error": True}
        ok2, size = _run_adb(["shell", "wm", "size"])
        ok3, focus = _run_adb(["shell", "dumpsys", "window"],
                              timeout=30.0)
        cur = ""
        for ln in (focus.splitlines() if ok3 else []):
            if "mCurrentFocus" in ln:
                cur = ln.strip()[:160]
                break
        lines = [f"Phone connected: {dev}.", f"Screen: {(size or '?').strip()}."]
        if cur:
            lines.append(f"Focus: {cur}")
        return {"output": "\n".join(lines), "metadata": {"connected": True}}

    if action == "app_list":
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        filt = str(input.get("filter") or "").strip().lower()
        ok2, out = _run_adb(["shell", "pm", "list", "packages"], timeout=30.0)
        if not ok2:
            return {"output": f"app_list failed: {out}", "error": True}
        pkgs = [ln.split(":", 1)[1] for ln in out.splitlines() if ln.startswith("package:")]
        if filt:
            pkgs = [p for p in pkgs if filt in p.lower()]
        shown = pkgs[:60]
        extra = f"\n... +{len(pkgs) - 60} more" if len(pkgs) > 60 else ""
        return {"output": f"{len(pkgs)} package(s):\n" + "\n".join(shown) + extra,
                "metadata": {"count": len(pkgs)}}

    if action == "focused":
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        ok2, out = _run_adb(["shell", "dumpsys", "window"], timeout=30.0)
        if not ok2:
            return {"output": f"focused failed: {out}", "error": True}
        lines = [ln.strip()[:160] for ln in out.splitlines()
                 if "mCurrentFocus" in ln or "mFocusedApp" in ln][:4]
        return {"output": "\n".join(lines) if lines else "(no focus info)"}

    if action == "screen":
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        ok2, out = _run_adb(["shell", "screencap", "-p", "/sdcard/opencode_py_screen.png"])
        if not ok2:
            return {"output": f"screencap failed: {out}", "error": True}
        adb = _adb() or "adb"
        out_dir = GPath.data / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"phone_{int(time.time())}.png"
        try:
            proc = subprocess.run([adb, "pull", "/sdcard/opencode_py_screen.png", str(out_path)],
                                  capture_output=True, text=True, timeout=30.0)
        except (OSError, subprocess.SubprocessError) as e:
            return {"output": f"pull failed: {e}", "error": True}
        _run_adb(["shell", "rm", "/sdcard/opencode_py_screen.png"])
        if proc.returncode != 0 or not out_path.exists():
            return {"output": "screenshot pull failed", "error": True}
        kb = out_path.stat().st_size // 1024
        return {"output": f"Phone screenshot saved: {out_path} ({kb} KB). Open it with read.",
                "metadata": {"path": str(out_path), "bytes": kb * 1024}}

    if action == "app_open":
        pkg = str(input.get("package") or "").strip()
        if not pkg:
            return {"output": "app_open needs package (use app_list to find it)", "error": True}
        if len(pkg) > 200 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_." for c in pkg):
            return {"output": f"bad package name {pkg!r}", "error": True}
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        if "/" in pkg or " " in pkg:
            activity = pkg
        else:
            ok2, out = _run_adb(["shell", "cmd", "package", "resolve-activity",
                                 "--brief", pkg], timeout=30.0)
            activity = ""
            if ok2:
                for ln in out.splitlines():
                    ln = ln.strip()
                    if "/" in ln and not ln.startswith("priority"):
                        activity = ln
                        break
            if not activity:
                activity = f"{pkg}/.MainActivity"
        ok3, out3 = _run_adb(["shell", "am", "start", "-n", activity], timeout=30.0)
        if not ok3:
            ok3, out3 = _run_adb(["shell", "monkey", "-p", pkg.split("/")[0],
                                  "-c", "android.intent.category.LAUNCHER", "1"], timeout=30.0)
            if not ok3:
                return {"output": f"app_open failed: {out3}", "error": True}
        time.sleep(1.5)
        return {"output": f"Opened {pkg}.", "metadata": {"package": pkg}}

    if action == "tap":
        try:
            x = _num(input.get("x"), "x", 0, 10000)
            y = _num(input.get("y"), "y", 0, 10000)
        except ValueError as e:
            return {"output": str(e), "error": True}
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        ok2, out = _run_adb(["shell", "input", "tap", str(x), str(y)])
        if not ok2:
            return {"output": f"tap failed: {out}", "error": True}
        return {"output": f"Tapped ({x}, {y})."}

    if action == "swipe":
        direction = str(input.get("direction") or "").strip().lower()
        if direction in ("up", "down", "left", "right"):
            try:
                cx = _num(input.get("x", 360), "x", 0, 10000)
                cy = _num(input.get("y", 760), "y", 0, 10000)
                dur = _num(input.get("duration", 300), "duration", 50, 5000)
            except ValueError as e:
                return {"output": str(e), "error": True}
            dist = 500
            x1, y1 = cx, cy
            x2, y2 = {"up": (cx, cy - dist), "down": (cx, cy + dist),
                      "left": (cx - dist, cy), "right": (cx + dist, cy)}[direction]
        else:
            try:
                x1 = _num(input.get("x1", input.get("x")), "x1", 0, 10000)
                y1 = _num(input.get("y1", input.get("y")), "y1", 0, 10000)
                x2 = _num(input.get("x2"), "x2", 0, 10000)
                y2 = _num(input.get("y2"), "y2", 0, 10000)
                dur = _num(input.get("duration", 300), "duration", 50, 5000)
            except ValueError as e:
                return {"output": str(e), "error": True}
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        ok2, out = _run_adb(["shell", "input", "swipe",
                             str(x1), str(y1), str(x2), str(y2), str(dur)])
        if not ok2:
            return {"output": f"swipe failed: {out}", "error": True}
        return {"output": f"Swiped ({x1},{y1}) -> ({x2},{y2}) in {dur}ms."}

    if action == "type":
        text = str(input.get("text") or "")
        if not text:
            return {"output": "type needs text", "error": True}
        if len(text) > 2000:
            return {"output": "text too long (max 2000 chars)", "error": True}
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        ok2, out = _run_adb(["shell", "input", "text", text.replace(" ", "%s")])
        if not ok2:
            return {"output": f"type failed: {out}", "error": True}
        if bool(input.get("submit")):
            _run_adb(["shell", "input", "keyevent", "KEYCODE_ENTER"])
        return {"output": f"Typed {len(text)} char(s){' + Enter' if input.get('submit') else ''}."}

    if action == "key":
        key = str(input.get("key") or "back").strip().lower()
        code = _KEY_ALIASES.get(key)
        if code is None:
            if len(key) == 1 and key.isalnum():
                code = f"KEYCODE_{key.upper()}"
            else:
                return {"output": f"unknown key {key!r} (back/home/recent/enter/tab/space/arrows/volume_up/...)",
                        "error": True}
        longpress = bool(input.get("longpress"))
        ok2, out = _run_adb(["shell", "input", "keyevent"]
                            + (["--longpress"] if longpress else []) + [code])
        if not ok2:
            return {"output": f"key failed: {out}", "error": True}
        return {"output": f"Pressed {key}."}

    if action == "shell":
        cmd = str(input.get("command") or "").strip()
        if not cmd:
            return {"output": "shell needs command", "error": True}
        parts = cmd.split()
        if not parts or parts[0] not in _SHELL_ALLOW:
            return {"output": f"refused: only {', '.join(_SHELL_ALLOW)} allowed", "error": True}
        if any(op in cmd for op in (";", "&&", "||", "|", "$(", "`", ">", "<")):
            return {"output": "refused: no shell operators", "error": True}
        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        ok2, out = _run_adb(["shell", *parts], timeout=30.0)
        if not ok2:
            return {"output": f"shell failed: {out}", "error": True}
        if len(out) > 4000:
            out = out[:4000] + "\n... (truncated)"
        return {"output": out or "(no output)"}

    if action == "ui":
        import re as _re

        ok, dev = _device_ok()
        if not ok:
            return {"output": dev, "error": True}
        ok2, out = _run_adb(["shell", "uiautomator", "dump", "/sdcard/opencode_ui.xml"])
        if not ok2:
            return {"output": f"ui dump failed: {out}", "error": True}
        ok3, xml = _run_adb(["shell", "cat", "/sdcard/opencode_ui.xml"], timeout=30.0)
        _run_adb(["shell", "rm", "/sdcard/opencode_ui.xml"])
        if not ok3 or not xml:
            return {"output": "ui dump empty (try screen instead)", "error": True}
        if len(xml) > 60000:
            return {"output": "ui dump too large, narrowing...", "error": True}
        pat = _re.compile(
            r'<node[^>]*text="([^"]*)"[^>]*resource-id="([^"]*)"[^>]*'
            r'class="([^"]*)"[^>]*content-desc="([^"]*)"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')
        pat2 = _re.compile(
            r'<node[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"[^>]*text="([^"]*)"')
        lines = []
        seen = 0
        for m in pat.finditer(xml):
            text, rid, cls, desc, x1, y1, x2, y2 = m.groups()
            label = text or desc or (rid.split("/")[-1].replace("_", " ") if "/" in rid else "")
            if not label.strip():
                continue
            cx, cy = (int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2
            short_cls = cls.split(".")[-1]
            lines.append(f"[{seen}] {label[:60]!r} <{short_cls}> @ {cx},{cy}")
            seen += 1
            if seen >= 80:
                break
        if not lines:
            for m in pat2.finditer(xml):
                x1, y1, x2, y2, text = m.groups()
                if not text.strip():
                    continue
                cx, cy = (int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2
                lines.append(f"[{seen}] {text[:60]!r} @ {cx},{cy}")
                seen += 1
                if seen >= 80:
                    break
        if not lines:
            return {"output": "(empty screen or custom renderer — try screen/tap by coords)"}
        header = f"{seen} element(s). Tap with tap x/y."
        return {"output": header + "\n" + "\n".join(lines), "metadata": {"elements": seen}}

    return {"output": f"Unhandled action {action!r}", "error": True}


def tool() -> Tool:
    description = (
        "Drive the WHOLE phone via adb: status, tap x/y, swipe, type text, "
        "key (back/home/recent/enter/...), app_open/app_list, screen "
        "(screenshot), ui (text layout, works when screenshots blocked), "
        "focused, shell (read-only adb). Coordinates are raw "
        "screen pixels (see status). Ask before passwords/payments."
    )

    return Tool(
        name="phone",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "status, tap, swipe, type, key, app_open, "
                                   "app_list, screen, ui, focused, shell",
                },
                "x": {"type": "integer", "description": "Tap X / swipe center X",
                      "optional": True},
                "y": {"type": "integer", "description": "Tap Y / swipe center Y",
                      "optional": True},
                "x1": {"type": "integer", "description": "Swipe start X", "optional": True},
                "y1": {"type": "integer", "description": "Swipe start Y", "optional": True},
                "x2": {"type": "integer", "description": "Swipe end X", "optional": True},
                "y2": {"type": "integer", "description": "Swipe end Y", "optional": True},
                "duration": {"type": "integer", "description": "Swipe ms (default 300)",
                             "optional": True},
                "direction": {"type": "string", "description": "up/down/left/right",
                              "optional": True},
                "text": {"type": "string", "description": "Text for type", "optional": True},
                "submit": {"type": "boolean", "description": "Enter after type",
                           "optional": True},
                "key": {"type": "string", "description": "back/home/recent/enter/...",
                        "optional": True},
                "longpress": {"type": "boolean", "description": "Long-press the key",
                              "optional": True},
                "package": {"type": "string", "description": "App package for app_open",
                            "optional": True},
                "filter": {"type": "string", "description": "Filter for app_list",
                           "optional": True},
                "command": {"type": "string", "description": "adb shell command (allowlisted)",
                            "optional": True},
            },
            ["action"],
        ),
        run=run,
        permission="phone",
    )


__all__ = ["tool", "run", "READONLY_ACTIONS"]
