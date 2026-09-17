"""Markdown renderer with opencode theme colors and syntax highlighting.

Heavy third-party imports (markdown_it, rich.syntax) are deferred to first
render so importing the module (and thus the whole TUI) stays cheap.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Optional, TYPE_CHECKING

from rich.console import Group
from rich.text import Text
from rich.text import TextType

from opencode_py.tui.theme import active_theme

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from markdown_it.token import Token

# URL pattern for auto-linking bare URLs
URL_PATTERN = re.compile(r'https?://[^\s<>"]+')

# MarkdownIt construction is ~3ms, so reuse ONE immutable parser across every
# render instead of building a fresh one per call.
_parser: Any | None = None


def _get_markdown_parser() -> Any:
    """Return the shared, lazily-initialized MarkdownIt parser."""
    global _parser
    if _parser is None:
        from markdown_it import MarkdownIt  # deferred: heavy import

        _parser = MarkdownIt().enable("strikethrough").enable("table")
    return _parser


class MarkdownRenderer:
    """Render markdown to rich renderables with opencode theme colors."""

    def __init__(self, width: int | None = None):
        self.theme = active_theme()
        self.width = width
        self.parser = _get_markdown_parser()

    def render(self, markdown_text: str) -> Group:
        """Render markdown string to a Group of rich renderables."""
        tokens = self.parser.parse(markdown_text)
        parts: list[TextType] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == "heading_open":
                level = int(token.tag[1])
                i += 1
                # heading content is in the next inline token
                heading_text = ""
                while i < len(tokens) and tokens[i].type != "heading_close":
                    if tokens[i].type == "inline":
                        heading_text = tokens[i].content
                    i += 1
                style = f"bold {self.theme.c('markdown_heading')}"
                prefix = "#" * level + " "
                parts.append(Text(prefix + heading_text, style=style))
                if level <= 2:
                    w = self.width or 80
                    parts.append(Text("─" * w, style=self.theme.c("markdown_hr")))
            elif token.type == "paragraph_open":
                i += 1
                # paragraph content is in inline tokens until paragraph_close
                para_parts: list[Text] = []
                while i < len(tokens) and tokens[i].type != "paragraph_close":
                    if tokens[i].type == "inline":
                        self._render_inline(tokens[i], para_parts)
                    i += 1
                if para_parts:
                    para_text = Text.assemble(*para_parts)
                    parts.append(para_text)
            elif token.type == "fence" or token.type == "code_block":
                info = token.info.strip() if token.info else ""
                parts.append(self._render_code_block(token.content, info))
            elif token.type == "blockquote_open":
                # Find the matching blockquote_close and render content recursively
                blockquote_tokens = []
                depth = 1
                i += 1
                while i < len(tokens) and depth > 0:
                    if tokens[i].type == "blockquote_open":
                        depth += 1
                    elif tokens[i].type == "blockquote_close":
                        depth -= 1
                    if depth > 0:
                        blockquote_tokens.append(tokens[i])
                    i += 1
                # Recursively render blockquote content
                blockquote_parts = self._render_tokens(blockquote_tokens)
                for bp in blockquote_parts:
                    if isinstance(bp, Text):
                        parts.append(Text("│ ", style=self.theme.c("markdown_quote")))
                        parts.append(bp)
                    else:
                        parts.append(bp)
                continue  # i already advanced
            elif token.type == "hr":
                w = self.width or 80
                parts.append(Text("─" * w, style=self.theme.c("markdown_hr")))
            elif token.type == "table_open":
                table_tokens = []
                depth = 1
                i += 1
                while i < len(tokens) and depth > 0:
                    if tokens[i].type == "table_open":
                        depth += 1
                    elif tokens[i].type == "table_close":
                        depth -= 1
                    if depth > 0:
                        table_tokens.append(tokens[i])
                    i += 1
                table_parts = self._render_table(table_tokens)
                parts.extend(table_parts)
                continue
            elif token.type == "bullet_list_open":
                # Find matching close and render list items
                list_tokens = []
                depth = 1
                i += 1
                while i < len(tokens) and depth > 0:
                    if tokens[i].type in ("bullet_list_open", "ordered_list_open"):
                        depth += 1
                    elif tokens[i].type in ("bullet_list_close", "ordered_list_close"):
                        depth -= 1
                    if depth > 0:
                        list_tokens.append(tokens[i])
                    i += 1
                list_parts = self._render_list(list_tokens, ordered=False)
                parts.extend(list_parts)
                continue
            elif token.type == "ordered_list_open":
                list_tokens = []
                depth = 1
                i += 1
                while i < len(tokens) and depth > 0:
                    if tokens[i].type in ("bullet_list_open", "ordered_list_open"):
                        depth += 1
                    elif tokens[i].type in ("bullet_list_close", "ordered_list_close"):
                        depth -= 1
                    if depth > 0:
                        list_tokens.append(tokens[i])
                    i += 1
                list_parts = self._render_list(list_tokens, ordered=True)
                parts.extend(list_parts)
                continue
            i += 1

        return Group(*parts) if parts else Group(Text(""))

    def _render_tokens(self, tokens: list[Token]) -> list[TextType]:
        """Render a list of tokens recursively."""
        parts: list[TextType] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == "paragraph_open":
                i += 1
                para_parts: list[Text] = []
                while i < len(tokens) and tokens[i].type != "paragraph_close":
                    if tokens[i].type == "inline":
                        self._render_inline(tokens[i], para_parts)
                    i += 1
                if para_parts:
                    parts.append(Text.assemble(*para_parts))
            elif token.type == "fence" or token.type == "code_block":
                info = token.info.strip() if token.info else ""
                parts.append(self._render_code_block(token.content, info))
            elif token.type == "hr":
                w = self.width or 80
                parts.append(Text("─" * w, style=self.theme.c("markdown_hr")))
            elif token.type == "blockquote_open":
                blockquote_tokens = []
                depth = 1
                i += 1
                while i < len(tokens) and depth > 0:
                    if tokens[i].type == "blockquote_open":
                        depth += 1
                    elif tokens[i].type == "blockquote_close":
                        depth -= 1
                    if depth > 0:
                        blockquote_tokens.append(tokens[i])
                    i += 1
                blockquote_parts = self._render_tokens(blockquote_tokens)
                for bp in blockquote_parts:
                    if isinstance(bp, Text):
                        parts.append(Text("│ ", style=self.theme.c("markdown_quote")))
                        parts.append(bp)
                    else:
                        parts.append(bp)
                continue
            elif token.type in ("bullet_list_open", "ordered_list_open"):
                list_tokens = []
                depth = 1
                is_ordered = token.type == "ordered_list_open"
                i += 1
                while i < len(tokens) and depth > 0:
                    if tokens[i].type in ("bullet_list_open", "ordered_list_open"):
                        depth += 1
                    elif tokens[i].type in ("bullet_list_close", "ordered_list_close"):
                        depth -= 1
                    if depth > 0:
                        list_tokens.append(tokens[i])
                    i += 1
                list_parts = self._render_list(list_tokens, ordered=is_ordered)
                parts.extend(list_parts)
                continue
            i += 1
        return parts

    def _render_list(self, tokens: list[Token], ordered: bool) -> list[TextType]:
        """Render list items."""
        parts: list[TextType] = []
        i = 0
        counter = 1
        while i < len(tokens):
            token = tokens[i]
            if token.type == "list_item_open":
                # Get the list number from token.info for ordered lists
                num = token.info if ordered else None
                i += 1
                item_parts: list[Text] = []
                # Render the content of this list item
                while i < len(tokens) and tokens[i].type != "list_item_close":
                    if tokens[i].type == "paragraph_open":
                        i += 1
                        para_parts: list[Text] = []
                        while i < len(tokens) and tokens[i].type != "paragraph_close":
                            if tokens[i].type == "inline":
                                self._render_inline(tokens[i], para_parts)
                            i += 1
                        if para_parts:
                            item_parts.append(Text.assemble(*para_parts))
                    elif tokens[i].type in ("bullet_list_open", "ordered_list_open"):
                        # Nested list
                        nested_tokens = []
                        depth = 1
                        nested_is_ordered = tokens[i].type == "ordered_list_open"
                        i += 1
                        while i < len(tokens) and depth > 0:
                            if tokens[i].type in ("bullet_list_open", "ordered_list_open"):
                                depth += 1
                            elif tokens[i].type in ("bullet_list_close", "ordered_list_close"):
                                depth -= 1
                            if depth > 0:
                                nested_tokens.append(tokens[i])
                            i += 1
                        nested_parts = self._render_list(nested_tokens, ordered=nested_is_ordered)
                        for np in nested_parts:
                            if isinstance(np, Text):
                                # Indent nested list items while preserving style
                                indented = Text("  ")
                                indented.append(np.plain, style=np.style)
                                item_parts.append(indented)
                            else:
                                item_parts.append(np)
                        continue
                    else:
                        i += 1
                task_box = self._task_marker(item_parts)
                if task_box is not None:
                    item_parts[0] = task_box
                    parts.extend(item_parts)
                    i += 1
                    continue
                # Build the list item with bullet/number
                if ordered and num is not None:
                    prefix = f"{num}. "
                    style = self.theme.c("markdown_list_enumeration")
                else:
                    prefix = "• "
                    style = self.theme.c("markdown_list_item")
                bullet_text = Text(prefix, style=style)
                if item_parts:
                    # Prepend bullet to first part
                    first = item_parts[0]
                    if isinstance(first, Text):
                        bullet_text.append(first.plain, style=first.style or self.theme.c("text"))
                        item_parts[0] = bullet_text
                    else:
                        parts.append(bullet_text)
                        parts.extend(item_parts)
                        continue
                else:
                    parts.append(bullet_text)
                parts.extend(item_parts)
            i += 1
        return parts

    def _task_marker(self, item_parts: list[TextType]) -> Text | None:
        """- [x] done / - [ ] todo -> styled checkbox marker (else None)."""
        if not item_parts or not isinstance(item_parts[0], Text):
            return None
        plain = item_parts[0].plain
        for mark, done in (("[x] ", True), ("[X] ", True), ("[ ] ", False)):
            if plain.startswith(mark):
                box = Text(
                    "✔ " if done else "○ ",
                    style=self.theme.c("markdown_todo_done") if done else self.theme.c("markdown_todo_open"),
                )
                box.append(plain[len(mark):], style=item_parts[0].style or self.theme.c("text"))
                return box
        return None

    def _render_inline(self, token: Token, out: list[Text]) -> None:
        """Render inline tokens with proper styling."""
        if not token.children:
            return
        for child in token.children:
            # Check link text BEFORE general text
            if child.type == "text" and getattr(self, "_link_href", None) is not None:
                self._link_text_parts.append(Text(child.content))
            elif child.type == "text":
                # Auto-detect and style bare URLs in plain text
                self._render_text_with_urls(child.content, out)
            elif child.type == "code_inline":
                out.append(Text(child.content, style=self.theme.c("markdown_code")))
            elif child.type == "strong_open":
                self._strong = True
            elif child.type == "strong_close":
                self._strong = False
            elif child.type == "em_open":
                self._em = True
            elif child.type == "em_close":
                self._em = False
            elif child.type == "s_open":
                self._strike = True
            elif child.type == "s_close":
                self._strike = False
            elif child.type == "link_open":
                self._link_href = child.attrs.get("href", "") if child.attrs else ""
                self._link_text_parts: list[Text] = []
            elif child.type == "link_close":
                href = getattr(self, "_link_href", "")
                link_text = Text.assemble(*getattr(self, "_link_text_parts", []))
                # Apply link style (underline + cyan) to the entire link text
                link_text.stylize(f"underline {self.theme.c('markdown_link_text')}")
                if href:
                    link_text.stylize(f"link {href}")
                out.append(link_text)
                self._link_href = None
                self._link_text_parts = None
            elif child.type == "image":
                alt = child.content or ""
                src = child.attrs.get("src", "") if child.attrs else ""
                img = Text(f"[image: {alt}]", style=self.theme.c("markdown_link"))
                if src:
                    img.stylize(f"link {src}")
                out.append(img)
            else:
                style = self.theme.c("text")
                if getattr(self, "_strong", False):
                    style = self.theme.c("markdown_strong")
                elif getattr(self, "_em", False):
                    style = self.theme.c("markdown_quote")
                if getattr(self, "_strike", False):
                    style = self.theme.c("markdown_strike") + " strikethrough"
                if child.type == "text":
                    self._render_text_with_urls(child.content, out, base_style=style)

    def _render_code_block(self, code: str, info: str) -> Any:
        """Render a fenced code block with syntax highlighting.

        Same pixels as before: full Syntax highlight. The win is upstream —
        finished bubbles freeze (chat_view.frozen) so this runs once per
        message, and render_markdown's lru_cache returns the SAME object on
        every redraw. Streaming bodies stay plain Text until end_stream.
        """
        from rich.syntax import Syntax  # deferred: heavy import

        language = info.split()[0] if info else "text"
        return Syntax(
            code.rstrip(),
            language,
            theme="monokai",
            line_numbers=False,
            word_wrap=False,
            indent_guides=False,
            background_color="default",
        )

    def _render_table(self, tokens: list[Token]) -> list[TextType]:
        """Render a markdown table as aligned text."""
        parts: list[TextType] = []
        rows: list[list[str]] = []
        current_row: list[str] = []
        in_header = False
        col_count = 0

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == "thead_open":
                in_header = True
            elif token.type == "tbody_open":
                in_header = False
            elif token.type == "tr_open":
                current_row = []
            elif token.type == "tr_close":
                if current_row:
                    rows.append(current_row)
                    col_count = max(col_count, len(current_row))
            elif token.type in ("th_open", "td_open"):
                # Get cell content from next inline token
                i += 1
                cell_text = ""
                while i < len(tokens) and tokens[i].type != ("th_close" if token.type == "th_open" else "td_close"):
                    if tokens[i].type == "inline":
                        cell_text = tokens[i].content
                        break
                    i += 1
                current_row.append(cell_text)
            i += 1

        if not rows:
            return parts

        # Calculate column widths
        col_widths = [0] * col_count
        for row in rows:
            for j, cell in enumerate(row):
                col_widths[j] = max(col_widths[j], len(cell))

        # Render rows
        for row_idx, row in enumerate(rows):
            line_parts: list[Text] = []
            for j, cell in enumerate(row):
                padded = cell.ljust(col_widths[j])
                if row_idx == 0:  # Header row
                    line_parts.append(Text(" " + padded + " ", style=f"bold {self.theme.c('markdown_heading')}"))
                else:
                    line_parts.append(Text(" " + padded + " ", style=self.theme.c("text")))
            parts.append(Text.assemble(*line_parts))
            # Add separator after header
            if row_idx == 0:
                sep_parts: list[Text] = []
                for w in col_widths:
                    sep_parts.append(Text(" " + "─" * w + " ", style=self.theme.c("markdown_hr")))
                parts.append(Text.assemble(*sep_parts))

        return parts

    def _render_text_with_urls(self, text: str, out: list[Text], base_style: str | None = None) -> None:
        """Split text by URLs and style them with link color."""
        style = base_style or self.theme.c("text")
        if base_style is None:
            if getattr(self, "_strong", False):
                style = self.theme.c("markdown_strong")
            elif getattr(self, "_em", False):
                style = self.theme.c("markdown_quote")
            if getattr(self, "_strike", False):
                style = self.theme.c("markdown_strike") + " strikethrough"

        last_end = 0
        for match in URL_PATTERN.finditer(text):
            start, end = match.span()
            url = match.group(0)
            # Text before URL
            if start > last_end:
                self._append_with_mentions(text[last_end:start], style, out)
            # The URL itself - blue with underline
            url_text = Text(url, style=f"underline {self.theme.c('markdown_link_text')}")
            url_text.stylize(f"link {url}")
            out.append(url_text)
            last_end = end
        # Remaining text after last URL
        if last_end < len(text):
            self._append_with_mentions(text[last_end:], style, out)

    def _append_with_mentions(self, text: str, style: str, out: list[Text]) -> None:
        """Style @file mentions with their own color."""
        import re as _re
        last = 0
        for m in _re.finditer(r"@[A-Za-z0-9_.\-/]+", text):
            s, e = m.span()
            if s > last:
                out.append(Text(text[last:s], style=style))
            out.append(Text(text[s:e], style=f"underline {self.theme.c('mention_file')}"))
            last = e
        if last < len(text):
            out.append(Text(text[last:], style=style))


@functools.lru_cache(maxsize=512)
def render_markdown(markdown_text: str, width: Optional[int] = None) -> Group:
    """Render markdown to a Group with opencode theme colors and syntax highlighting.

    Results are cached keyed by ``(text, width)`` — the same message re-rendered
    at the same width (spinner ticks, redraws, resize settle) costs ~0ms. ``width``
    is part of the key because hr/divider widths depend on it. Callers must never
    mutate the returned ``Group`` in place (they only wrap it); the renderer itself
    is immutable, so sharing one cached object across bubbles is safe.
    """
    return MarkdownRenderer(width=width).render(markdown_text)