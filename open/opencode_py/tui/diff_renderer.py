"""Unified diff rendering matching opencode's opentui `<diff>` component.

When the model edits a file, opencode shows the patch as a color-coded block:

- a fixed-width gutter: left pad, right-aligned line number (fg ``diffLineNumber``),
  a space, the ``+``/``-``/`` `` sign (fg ``diffHighlightAdded``/``diffHighlightRemoved``)
  and a trailing space,
- the content with the *content* background (``diffAddedBg``/``diffRemovedBg``/
  ``diffContextBg``) while the gutter uses the *gutter* background
  (``diffAddedLineNumberBg``/``diffRemovedLineNumberBg``),
- the content is syntax-highlighted with the opencode palette (pygments).

Hunk headers (`@@ ... @@`) and `---/+++` file headers are folded out of the body,
exactly like the official renderer. The sign column is only reserved when the
patch actually contains ``+``/``-`` lines (mirroring opentui's ``maxAfterWidth``),
long lines can be word-wrapped (``diff_wrap_mode``) and every background can be
suppressed (``suppress_backgrounds``).
"""

from __future__ import annotations

import functools
import re
import unicodedata

from rich.console import Group
from rich.text import Text
from rich.text import TextType

from opencode_py.tui.theme import active_theme

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

_TYPE_TO_SCOPE_CACHE: list[tuple[tuple[object, ...], str]] | None = None


def _get_type_to_scope() -> list[tuple[tuple[object, ...], str]]:
    """Lazily build the pygments base token -> theme scope table."""
    global _TYPE_TO_SCOPE_CACHE
    if _TYPE_TO_SCOPE_CACHE is None:
        try:
            from pygments.token import (
                Comment,
                Error as PymError,
                Generic,
                Keyword,
                Name,
                Number,
                Operator,
                Punctuation,
                String,
                Whitespace,
            )
        except Exception:  # pragma: no cover - pygments absent
            return []
        _TYPE_TO_SCOPE_CACHE = [
            ((Comment,), "comment"),
            ((Keyword,), "keyword"),
            ((Name.Function,), "function"),
            ((Name.Class, Name.Namespace, Name.Decorator), "type"),
            ((Name.Variable, Name.Attribute), "variable"),
            ((Name.Builtin, Name.Constant), "variable"),
            ((String,), "string"),
            ((Number,), "number"),
            ((Operator,), "operator"),
            ((Punctuation,), "punctuation"),
            ((Generic,), "text"),
            ((PymError,), "error"),
        ]
    return _TYPE_TO_SCOPE_CACHE


class _DiffHighlighter:
    """Highlight diff content with pygments using the opencode theme scopes."""

    def __init__(self) -> None:
        self._lexers: dict[str, object] = {}
        self._style_styles: dict[str, str] | None = None
        self._style_cls: object | None = None

    def _get_style(self) -> object:
        if self._style_cls is not None:
            return self._style_cls
        theme = active_theme()
        try:
            from pygments.style import Style

            scopes = {
                "text": theme.c("text"),
                "comment": theme.c("text_muted"),
                "keyword": theme.c("syntax_keyword"),
                "function": theme.c("syntax_function"),
                "type": theme.c("syntax_type"),
                "variable": theme.c("syntax_variable"),
                "string": theme.c("syntax_string"),
                "number": theme.c("syntax_number"),
                "operator": theme.c("syntax_operator"),
                "punctuation": theme.c("syntax_punctuation"),
                "error": theme.c("error"),
            }
            styles: dict[str, str] = {}
            for toks, scope in _get_type_to_scope():
                for tok in toks:
                    if tok is not None and scope in scopes:
                        styles[tok] = scopes[scope]
            self._style_styles = styles
            self._style_cls = type("_OpenCodeStyle", (Style,), {"styles": styles})
        except Exception:
            self._style_cls = None
            self._style_styles = {}
        return self._style_cls

    def _lexer(self, filename: str):
        if not filename:
            return None
        if filename in self._lexers:
            return self._lexers[filename]
        lexer = None
        try:
            from pygments.lexers import get_lexer_by_name

            name = filename.rsplit("/", 1)[-1]
            if name.startswith("."):
                name = name[1:]
            # a bare extension like "py" -> python
            try:
                lexer = get_lexer_by_name(name)
            except Exception:
                base = name.split(".")[-1]
                lexer = get_lexer_by_name(base) if base != name else None
        except Exception:
            lexer = None
        self._lexers[filename] = lexer
        return lexer

    def highlight(self, content: str, filename: str, bg: str | None = None) -> Text:
        return _highlight_cached(content, filename, bg)


@functools.lru_cache(maxsize=2048)
def _highlight_cached(content: str, filename: str, bg: str | None = None) -> Text:
    # bg is applied as the base style; later fg syntax spans overlay it, so
    # both background and syntax colors survive (rich stack order).
    text = Text(content, style=f"on {bg}" if bg else None)
    lexer = _HIGHLIGHTER._lexer(filename)
    style = _HIGHLIGHTER._get_style()
    if lexer is None or style is None:
        return text
    try:
        spans: list[tuple[int, int, str]] = []
        pos = 0
        for tok_type, value in lexer.get_tokens(content):
            st = style.style_for_token(tok_type)
            color = st.get("color")  # type: ignore[union-attr, index]
            if color and not color.startswith("#"):
                color = "#" + color
            if color:
                spans.append((pos, pos + len(value), color))
            pos += len(value)
        for start, end, color in spans:
            text.stylize(color, start, end)
    except Exception:
        return Text(content, style=f"on {bg}" if bg else None)
    return text


def _lex_spans(content: str, filename: str) -> list[tuple[int, int, str]]:
    """Tokenize `content` once and return (start, end, color) spans.

    Used by the batch path so a whole diff is pygments-lexed in a SINGLE pass
    instead of one lexer invocation per line — per-line lexing dominates the
    render cost for large patches. Returns [] when there is nothing to
    highlight with (no filename, no lexer, no style).
    """
    if not filename:
        return []
    lexer = _HIGHLIGHTER._lexer(filename)
    style = _HIGHLIGHTER._get_style()
    if lexer is None or style is None:
        return []
    spans: list[tuple[int, int, str]] = []
    pos = 0
    try:
        for tok_type, value in lexer.get_tokens(content):
            st = style.style_for_token(tok_type)
            color = st.get("color")  # type: ignore[union-attr, index]
            if color and not color.startswith("#"):
                color = "#" + color
            if color:
                spans.append((pos, pos + len(value), color))
            pos += len(value)
    except Exception:
        return []
    return spans


def _apply_spans(text: Text, spans: list[tuple[int, int, str]], start: int, stop: int) -> None:
    """Apply the spans overlapping ``[start, stop)`` of the lexed text to ``text``."""
    for s, e, color in spans:
        if e <= start:
            continue
        if s >= stop:
            break
        lo, hi = max(s, start) - start, min(e, stop) - start
        if hi > lo:
            text.stylize(color, lo, hi)


_HIGHLIGHTER = _DiffHighlighter()


def _diff_lines(diff_text: str) -> list[str]:
    """Normalize diff body lines; keep @@ hunk headers (rest of body is (+| |-) lines)."""
    return [ln for ln in diff_text.splitlines() if ln and not ln.startswith(("---", "+++", "\\"))]


def _display_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _word_wrap(content: str, width: int) -> list[str]:
    """Wrap content to ``width`` display columns, preserving leading whitespace.

    Mirrors opentui's ``wrapMode="word"``: breaks on spaces (a token wider than the
    available space is hard-split), and continuation rows keep no line number.
    """
    if width <= 0 or _display_width(content) <= width:
        return [content]
    # split into words/tokens keeping whitespace runs so indentation survives
    tokens = re.findall(r"\S+|\s+", content)
    lines: list[str] = []
    current = ""
    current_w = 0
    for token in tokens:
        token_w = _display_width(token)
        if token_w >= width and not token.strip():
            # whitespace wider than the line: just drop its overflow
            if current_w + width <= width and token.startswith((" ", "\t")):
                pass
            continue
        if token_w >= width and token.strip():
            # a single oversized word: flush and hard-split it
            if current.strip():
                lines.append(current.rstrip())
                current = ""
                current_w = 0
            while token:
                cut, token = _hard_cut(token, width)
                if not cut:
                    break
                lines.append(cut)
            continue
        if current_w + token_w > width and current.strip():
            lines.append(current.rstrip())
            current = token
            current_w = token_w
            continue
        current += token
        current_w += token_w
    if current.strip():
        lines.append(current.rstrip())
    elif not lines:
        lines.append("")
    return lines


def _hard_cut(text: str, width: int) -> tuple[str, str]:
    """Split ``text`` after ``width`` display columns (wide chars count as 2)."""
    acc = ""
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if _display_width(acc) + w > width:
            break
        acc += ch
    return acc, text[len(acc) :]


def _diff_lines_wrap(content: str, width: int) -> list[str]:
    return _word_wrap(content, width)


class _Row:
    __slots__ = ("sign", "content", "old_num", "new_num")

    def __init__(self, sign: str, content: str, old_num: int | None, new_num: int | None) -> None:
        self.sign = sign
        self.content = content
        self.old_num = old_num
        self.new_num = new_num


def _parse_rows(lines: list[str]) -> list[_Row]:
    """Parse diff body into rows with old/new line numbers for each side."""
    rows: list[_Row] = []
    old_num: int | None = None
    new_num: int | None = None
    in_hunk = False
    for raw in lines:
        if raw.startswith("@@") and not in_hunk:
            in_hunk = True
            m = _HUNK_RE.match(raw)
            old_num = int(m.group(1)) if m else None
            new_num = int(m.group(2)) if m else None
            if old_num == 0:
                old_num = None
            continue
        if not in_hunk:
            continue
        sign = raw[0]
        if sign not in ("+", "-", " "):
            continue
        content = raw[1:]
        r = _Row(sign, content, old_num, new_num)
        if sign == "+":
            if new_num is not None:
                new_num += 1
        elif sign == "-":
            if old_num is not None:
                old_num += 1
        else:
            if old_num is not None:
                old_num += 1
            if new_num is not None:
                new_num += 1
        rows.append(r)
    return rows


def _gutter(
    sign: str,
    lineno: int | None,
    gutter_w: int,
    has_signs: bool,
    bg: str | None,
    fg: str,
    sign_fg: str,
) -> Text:
    """Build the gutter cell for one physical row."""
    if lineno is None:
        text = Text(" " * (gutter_w + (2 if has_signs else 0) + 1), style=f"on {bg}" if bg else None)
        return text
    num = str(lineno).rjust(gutter_w)
    # official layout: [pad][num][sign slot][pad] when signs exist, else [pad][num][pad]
    slot = f" {sign} " if has_signs else " "
    text = Text(f" {num}{slot}", style=f"on {bg}" if bg else None)
    text.stylize(f"{fg} on {bg}" if bg else fg, 1, 1 + gutter_w)
    if has_signs and sign in ("+", "-"):
        start = 2 + gutter_w
        text.stylize(f"{sign_fg} on {bg}" if bg else sign_fg, start, start + 1)
    return text


def _build_unified(
    rows: list[_Row],
    *,
    filename: str,
    width: int,
    wrap: str,
    suppress: bool,
    line_fg: str,
    context_bg: str,
    added_bg: str,
    removed_bg: str,
    added_gutter: str,
    removed_gutter: str,
    added_sign: str,
    removed_sign: str,
) -> Group:
    has_signs = any(r.sign in ("+", "-") for r in rows)
    max_lineno = max([n for r in rows for n in (r.new_num, r.old_num) if n is not None] or [0])
    gutter_w = len(str(max_lineno)) if max_lineno > 0 else 1
    content_w = width - gutter_w - (3 if has_signs else 2)

    # Lex the whole patch body ONCE (lines joined with "\n") and slice spans
    # back per physical row. Per-line lexing was the dominant render cost.
    joined: list[str] = []
    for r in rows:
        pieces = _diff_lines_wrap(r.content, content_w) if wrap == "word" else [r.content]
        joined.extend(pieces)
    body = "\n".join(joined)
    spans = _lex_spans(body, filename)
    del joined

    out: list[TextType] = []
    char = 0
    for r in rows:
        pieces = _diff_lines_wrap(r.content, content_w) if wrap == "word" else [r.content]
        for i, piece in enumerate(pieces):
            start = char
            char += len(piece) + 1  # +1 for the "\n" joiner
            if r.sign == "+":
                bg, gutter_bg, sc = added_bg, added_gutter, added_sign
                ln = r.new_num
            elif r.sign == "-":
                bg, gutter_bg, sc = removed_bg, removed_gutter, removed_sign
                ln = r.old_num
            else:
                bg, gutter_bg, sc = context_bg, context_bg, line_fg
                ln = r.new_num if r.new_num is not None else r.old_num
            if suppress:
                bg = gutter_bg = None
            body = Text(piece, style=f"on {bg}" if bg else None)
            _apply_spans(body, spans, start, start + len(piece))
            gut = _gutter(r.sign, ln if i == 0 else None, gutter_w, has_signs, gutter_bg, line_fg, sc)
            out.append(Text().append(gut).append(body))
    return Group(*out)


def _build_split(
    rows: list[_Row],
    *,
    filename: str,
    width: int,
    wrap: str,
    suppress: bool,
    line_fg: str,
    context_bg: str,
    added_bg: str,
    removed_bg: str,
    added_gutter: str,
    removed_gutter: str,
    added_sign: str,
    removed_sign: str,
) -> Group:
    # Official split view groups remove/add runs so they line up row-by-row:
    # a `-` line goes to the left pane, the matching `+` line to the right pane,
    # and unmatched lines leave an empty padded slot on the opposite side.
    left: list[_Row] = []
    right: list[_Row] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        if r.sign == " ":
            left.append(r)
            right.append(r)
            i += 1
        else:
            removes: list[_Row] = []
            adds: list[_Row] = []
            while i < len(rows) and rows[i].sign in ("+", "-"):
                if rows[i].sign == "-":
                    removes.append(rows[i])
                else:
                    adds.append(rows[i])
                i += 1
            n = max(len(removes), len(adds))
            for j in range(n):
                if j < len(removes):
                    left.append(removes[j])
                else:
                    left.append(_Row(" ", "", None, None))
                if j < len(adds):
                    right.append(adds[j])
                else:
                    right.append(_Row(" ", "", None, None))

    def pane(rows_pane: list[_Row]) -> list[TextType]:
        has_signs = any(r.sign in ("+", "-") for r in rows_pane)
        max_lineno = max([n for r in rows_pane for n in (r.new_num, r.old_num) if n is not None] or [0])
        gutter_w = len(str(max_lineno)) if max_lineno > 0 else 1
        avail = width if width > 0 else 120
        content_w = max(1, (avail - 3) // 2 - (gutter_w + (3 if has_signs else 2)))
        if width <= 0:
            content_w = 0  # no wrapping when the terminal width is unknown
        pane_rows: list[TextType] = []
        for r in rows_pane:
            if r.sign == "-":
                bg, gutter_bg, sc = removed_bg, removed_gutter, removed_sign
                ln = r.old_num
            elif r.sign == "+":
                bg, gutter_bg, sc = added_bg, added_gutter, added_sign
                ln = r.new_num
            else:
                bg, gutter_bg, sc = context_bg, context_bg, line_fg
                ln = r.new_num if r.new_num is not None else r.old_num
            if suppress:
                bg = gutter_bg = None
            pieces = _diff_lines_wrap(r.content, content_w) if wrap == "word" else [r.content]
            for j, piece in enumerate(pieces):
                body = _HIGHLIGHTER.highlight(piece, filename, bg=bg)
                # for the new-side pane the difference sign shows as +, in the
                # old-side pane the - sign; context rows show no sign
                sign = r.sign if (r.sign in ("+", "-")) else " "
                gut = _gutter(sign, ln if j == 0 else None, gutter_w, has_signs, gutter_bg, line_fg, sc)
                pane_rows.append(Text().append(gut).append(body))
        return pane_rows

    left_rows = pane(left)
    right_rows = pane(right)
    n = max(len(left_rows), len(right_rows))
    gap = Text("  ", style=None)
    combined: list[TextType] = []
    for j in range(n):
        lr = left_rows[j] if j < len(left_rows) else Text("")
        rr = right_rows[j] if j < len(right_rows) else Text("")
        combined.append(Text().append(lr).append(gap).append(rr))
    return Group(*combined)


def render_diff(
    diff_text: str,
    filename: str = "",
    view: str = "unified",
    width: int | None = None,
    wrap: str = "word",
    suppress_backgrounds: bool = False,
) -> Group:
    """Render a patch the way opencode's edit block does.

    - ``view``: ``"unified"`` (stacked, default) or ``"split"`` (side-by-side).
      opencode picks split when the terminal is wider than 120 columns
      (``diff_style`` config can force ``"stacked"``); pass the terminal width so
      ``view="auto"`` resolves like the official ``ctx.width > 120`` rule.
    - ``wrap``: ``"word"`` (default, matches ``diff_wrap_mode``) or ``"none"``.
    - ``suppress_backgrounds``: use transparent diff backgrounds.
    - ``filename`` enables syntax highlighting.

    The result is memoized on the exact arguments: a tool row re-renders the
    SAME patch on every spinner/status refresh, and pygments lexing dominates
    the render cost. Identical re-renders return the cached object.
    """
    return _render_diff_cached(diff_text, filename, view, width or 0, wrap, suppress_backgrounds, _theme_key())


def _theme_key() -> str:
    try:
        from .theme import _active_name as _name  # type: ignore

        return str(_name or "opencode")
    except Exception:
        return "opencode"


@functools.lru_cache(maxsize=64)
def _render_diff_cached(
    diff_text: str,
    filename: str,
    view: str,
    width: int,
    wrap: str,
    suppress_backgrounds: bool,
    theme_key: str = "opencode",
) -> Group:
    theme = active_theme()
    line_fg = theme.c("diff_gutter")
    context_bg = theme.c("diff_context_bg")
    added_bg = theme.c("diff_added_bg")
    removed_bg = theme.c("diff_removed_bg")
    added_gutter = theme.c("diff_added_line_number_bg")
    removed_gutter = theme.c("diff_removed_line_number_bg")
    added_sign = theme.c("diff_highlight_added")
    removed_sign = theme.c("diff_highlight_removed")

    rows = _parse_rows(_diff_lines(diff_text))
    if not rows:
        return Group(Text(""))

    width = width or 0
    if view == "auto":
        view = "split" if width > 120 else "unified"

    kwargs = dict(
        filename=filename,
        width=width,
        wrap=wrap if wrap in ("word",) else "none",
        suppress=suppress_backgrounds,
        line_fg=line_fg,
        context_bg=context_bg,
        added_bg=added_bg,
        removed_bg=removed_bg,
        added_gutter=added_gutter,
        removed_gutter=removed_gutter,
        added_sign=added_sign,
        removed_sign=removed_sign,
    )
    if view == "split":
        return _build_split(rows, **kwargs)
    return _build_unified(rows, **kwargs)