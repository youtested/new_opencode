from __future__ import annotations

from rich.text import Text

from .theme import active_theme

_LOGO_OPEN = [
    "                   ",
    "█▀▀█ █▀▀█ █▀▀█ █▀▀█",
    "█__█ █__█ █^^^ █  █",
    "▀▀▀▀ █▀▀▀ ▀▀▀▀ ▀  ▀",
]
_LOGO_CODE = [
    "             ▄     ",
    "█▀▀▀ █▀▀█ █▀▀█ █▀▀█",
    "█___ █__█ █__█ █^^^",
    "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀",
]
_LOGO_MARKS = {"_": " ", "^": "▀", "~": "▀", ",": "▄"}


def _logo_expand(line: str) -> str:
    return "".join(_LOGO_MARKS.get(ch, ch) for ch in line)


def _tint(hex_color: str, overlay: str, alpha: float) -> str:
    def _rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    base, ov = _rgb(hex_color), _rgb(overlay)
    mixed = tuple(round(b + (o - b) * alpha) for b, o in zip(base, ov))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _logo_char(char: str, fg: str, bold: bool, shadow: str) -> tuple[str, str]:
    b = "bold " if bold else ""
    if char == "_":
        return " ", f"{b}on {shadow}"
    if char == "^":
        return "▀", f"{b}{fg} on {shadow}"
    if char == "~":
        return "▀", f"{b}{shadow}"
    if char == ",":
        return "▄", f"{b}{shadow}"
    return char, f"{b}{fg}"


OPENCODE_LOGO = "\n".join(
    _logo_expand(lo) + " " + _logo_expand(rc) for lo, rc in zip(_LOGO_OPEN, _LOGO_CODE)
)


def opencode_logo_text() -> Text:
    theme = active_theme()
    gray, white = theme.c("text_muted"), theme.c("text")
    shadow_open = _tint(theme.c("background"), gray, 0.25)
    shadow_code = _tint(theme.c("background"), white, 0.25)
    out = Text()
    for i, (lo, rc) in enumerate(zip(_LOGO_OPEN, _LOGO_CODE)):
        if i:
            out.append("\n")
        for ch in lo:
            txt, style = _logo_char(ch, gray, False, shadow_open)
            out.append(txt, style=style)
        out.append(" ")
        for ch in rc:
            txt, style = _logo_char(ch, white, True, shadow_code)
            out.append(txt, style=style)
    return out
