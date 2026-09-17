"""opencode theme colors + Textual theme registration.

Mirrors opencode's theme/assets/opencode.json dark palette so the Python TUI
picks up the exact same look: background #0a0a0a, backgroundPanel #141414,
backgroundElement #1e1e1e, primary #fab283, accent #9d7cd8, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

# Official opencode "opencode" dark theme.
OPENSE_DARK: dict[str, str] = {
    "background": "#0a0a0a",
    "background_panel": "#141414",
    "background_element": "#1e1e1e",
    "background_menu": "#141414",
    "border": "#484848",
    "border_active": "#606060",
    "border_subtle": "#3c3c3c",
    "primary": "#fab283",
    "secondary": "#5c9cf5",
    "accent": "#9d7cd8",
    "error": "#e06c75",
    "warning": "#f5a742",
    "success": "#7fd88f",
    "info": "#56b6c2",
    "text": "#eeeeee",
    "text_muted": "#808080",
    # diff (official diffAdded/diffRemoved/diffContext + backgrounds)
    "diff_added": "#4fd6be",
    "diff_removed": "#c53b53",
    "diff_context": "#828bb8",
    "diff_hunk_header": "#828bb8",
    "diff_highlight_added": "#b8db87",
    "diff_highlight_removed": "#e26a75",
    "diff_added_bg": "#20303b",
    "diff_removed_bg": "#37222c",
    "diff_context_bg": "#141414",
    "diff_line_number": "#8f8f8f",
    "diff_added_line_number_bg": "#1b2b34",
    "diff_removed_line_number_bg": "#2d1f26",
    # syntax (official syntax* scopes)
    "syntax_comment": "#808080",
    "syntax_keyword": "#9d7cd8",
    "syntax_function": "#fab283",
    "syntax_variable": "#e06c75",
    "syntax_string": "#7fd88f",
    "syntax_number": "#f5a742",
    "syntax_type": "#e5c07b",
    "syntax_operator": "#56b6c2",
    "syntax_punctuation": "#eeeeee",
# markdown (official markdown* colors)
    "markdown_heading": "#9d7cd8",
    "markdown_link": "#fab283",
    "markdown_link_text": "#5c9cf5",
    "markdown_code": "#7fd88f",
    "markdown_quote": "#e5c07b",
    "markdown_strong": "#f5a742",
    "markdown_hr": "#808080",
    "markdown_list_item": "#fab283",
    "markdown_code_block": "#eeeeee",
    # custom UX colors (revert safe: every .c() falls back to OPENSE_DARK)
    "user_bubble": "#a8c7fa",
    "streaming_cursor": "#ffd166",
    "tool_running": "#f5a742",
    "tool_success": "#5f9e6e",
    "tool_error": "#ff5555",
    "tool_denied": "#ff8a5c",
    "markdown_list_enumeration": "#c9a37e",
    "markdown_strike": "#6e6e6e",
    "markdown_todo_done": "#7fd88f",
    "markdown_todo_open": "#808080",
    "mention_file": "#6cb6ff",
    "search_hit": "#ffd166",
    "diff_gutter": "#8f8f8f",
    "thinking_time": "#808080",
    "queue_badge": "#9d7cd8",
    "footer_active": "#f5a742",
}

# Solarized stays as a lightweight alternative.
SOLARIZED: dict[str, str] = {
    "background": "#002b36",
    "background_panel": "#073642",
    "background_element": "#073642",
    "background_menu": "#073642",
    "border": "#586e75",
    "border_active": "#839496",
    "border_subtle": "#073642",
    "primary": "#cb4b16",
    "secondary": "#2aa198",
    "accent": "#6c71c4",
    "error": "#dc322f",
    "warning": "#b58900",
    "success": "#859900",
    "info": "#2aa198",
    "text": "#839496",
    "text_muted": "#586e75",
    "diff_added_bg": "#12332a",
    "diff_removed_bg": "#3d1f26",
    "diff_context_bg": "#073642",
}


def _palette(**overrides: str) -> dict[str, str]:
    """A theme = the opencode-dark key layout with a palette's overrides.

    Every key this TUI ever reads exists in OPENSE_DARK, so a merge-based
    definition stays complete while each theme only spells out what makes it
    distinct (backgrounds, core colors, diff/syntax/markdown hues)."""
    d = dict(OPENSE_DARK)
    d.update(overrides)
    return d


# -- dark themes (listed first) ----------------------------------------------

TOKYO_NIGHT = _palette(
    background="#1a1b26", background_panel="#16161e", background_element="#1f2233",
    background_menu="#16161e",
    border="#3b4261", border_active="#565f89", border_subtle="#2f334d",
    primary="#ff9e64", secondary="#7aa2f7", accent="#bb9af7",
    error="#f7768e", warning="#e0af68", success="#9ece6a", info="#7dcfff",
    text="#c0caf5", text_muted="#565f89",
    diff_added="#9ece6a", diff_removed="#f7768e", diff_context="#a9b1d6",
    diff_hunk_header="#7aa2f7", diff_highlight_added="#9ece6a", diff_highlight_removed="#f7768e",
    diff_added_bg="#20302a", diff_removed_bg="#33212c", diff_context_bg="#16161e",
    diff_line_number="#565f89", diff_added_line_number_bg="#1d2b25", diff_removed_line_number_bg="#2d1f28",
    syntax_comment="#565f89", syntax_keyword="#bb9af7", syntax_function="#ff9e64",
    syntax_variable="#f7768e", syntax_string="#9ece6a", syntax_number="#e0af68",
    syntax_type="#7dcfff", syntax_operator="#7dcfff", syntax_punctuation="#c0caf5",
    markdown_heading="#bb9af7", markdown_link="#ff9e64", markdown_link_text="#7aa2f7",
    markdown_code="#9ece6a", markdown_quote="#e0af68", markdown_strong="#e0af68",
    markdown_hr="#565f89", markdown_list_item="#ff9e64", markdown_code_block="#c0caf5",
)

CATPPUCCIN_MOCHA = _palette(
    background="#1e1e2e", background_panel="#181825", background_element="#313244",
    background_menu="#181825",
    border="#45475a", border_active="#585b70", border_subtle="#313244",
    primary="#fab387", secondary="#89b4fa", accent="#cba6f7",
    error="#f38ba8", warning="#f9e2af", success="#a6e3a1", info="#94e2d5",
    text="#cdd6f4", text_muted="#7f849c",
    diff_added="#a6e3a1", diff_removed="#f38ba8", diff_context="#bac2de",
    diff_hunk_header="#89b4fa", diff_highlight_added="#a6e3a1", diff_highlight_removed="#f38ba8",
    diff_added_bg="#263243", diff_removed_bg="#3b2536", diff_context_bg="#181825",
    diff_line_number="#7f849c", diff_added_line_number_bg="#24273a", diff_removed_line_number_bg="#24273a",
    syntax_comment="#6c7086", syntax_keyword="#cba6f7", syntax_function="#fab387",
    syntax_variable="#f38ba8", syntax_string="#a6e3a1", syntax_number="#f9e2af",
    syntax_type="#94e2d5", syntax_operator="#94e2d5", syntax_punctuation="#cdd6f4",
    markdown_heading="#cba6f7", markdown_link="#fab387", markdown_link_text="#89b4fa",
    markdown_code="#a6e3a1", markdown_quote="#f9e2af", markdown_strong="#f9e2af",
    markdown_hr="#6c7086", markdown_list_item="#fab387", markdown_code_block="#cdd6f4",
)

DRACULA = _palette(
    background="#282a36", background_panel="#21222c", background_element="#44475a",
    background_menu="#21222c",
    border="#44475a", border_active="#6272a4", border_subtle="#3b3d4f",
    primary="#ffb86c", secondary="#8be9fd", accent="#bd93f9",
    error="#ff5555", warning="#f1fa8c", success="#50fa7b", info="#8be9fd",
    text="#f8f8f2", text_muted="#6272a4",
    diff_added="#50fa7b", diff_removed="#ff5555", diff_context="#a2a6c5",
    diff_hunk_header="#bd93f9", diff_highlight_added="#50fa7b", diff_highlight_removed="#ff5555",
    diff_added_bg="#243127", diff_removed_bg="#3b2233", diff_context_bg="#21222c",
    diff_line_number="#6272a4", diff_added_line_number_bg="#243127", diff_removed_line_number_bg="#3b2233",
    syntax_comment="#6272a4", syntax_keyword="#ff79c6", syntax_function="#50fa7b",
    syntax_variable="#f8f8f2", syntax_string="#f1fa8c", syntax_number="#bd93f9",
    syntax_type="#8be9fd", syntax_operator="#ff79c6", syntax_punctuation="#f8f8f2",
    markdown_heading="#bd93f9", markdown_link="#ffb86c", markdown_link_text="#8be9fd",
    markdown_code="#50fa7b", markdown_quote="#f1fa8c", markdown_strong="#f1fa8c",
    markdown_hr="#6272a4", markdown_list_item="#ff79c6", markdown_code_block="#f8f8f2",
)

NORD = _palette(
    background="#2e3440", background_panel="#3b4252", background_element="#434c5e",
    background_menu="#3b4252",
    border="#4c566a", border_active="#81a1c1", border_subtle="#3b4252",
    primary="#88c0d0", secondary="#81a1c1", accent="#b48ead",
    error="#bf616a", warning="#ebcb8b", success="#a3be8c", info="#8fbcbb",
    text="#eceff4", text_muted="#7b88a1",
    diff_added="#a3be8c", diff_removed="#bf616a", diff_context="#d8dee9",
    diff_hunk_header="#81a1c1", diff_highlight_added="#a3be8c", diff_highlight_removed="#bf616a",
    diff_added_bg="#39473f", diff_removed_bg="#47333c", diff_context_bg="#3b4252",
    diff_line_number="#7b88a1", diff_added_line_number_bg="#3b4a40", diff_removed_line_number_bg="#453239",
    syntax_comment="#616e88", syntax_keyword="#81a1c1", syntax_function="#88c0d0",
    syntax_variable="#d8dee9", syntax_string="#a3be8c", syntax_number="#b48ead",
    syntax_type="#8fbcbb", syntax_operator="#81a1c1", syntax_punctuation="#eceff4",
    markdown_heading="#88c0d0", markdown_link="#8fbcbb", markdown_link_text="#81a1c1",
    markdown_code="#a3be8c", markdown_quote="#ebcb8b", markdown_strong="#ebcb8b",
    markdown_hr="#4c566a", markdown_list_item="#88c0d0", markdown_code_block="#eceff4",
)

GRUVBOX_DARK = _palette(
    background="#282828", background_panel="#3c3836", background_element="#504945",
    background_menu="#3c3836",
    border="#504945", border_active="#a89984", border_subtle="#3c3836",
    primary="#fe8019", secondary="#83a598", accent="#d3869b",
    error="#fb4934", warning="#fabd2f", success="#b8bb26", info="#8ec07c",
    text="#ebdbb2", text_muted="#928374",
    diff_added="#b8bb26", diff_removed="#fb4934", diff_context="#ebdbb2",
    diff_hunk_header="#83a598", diff_highlight_added="#b8bb26", diff_highlight_removed="#fb4934",
    diff_added_bg="#3a432b", diff_removed_bg="#46332c", diff_context_bg="#3c3836",
    diff_line_number="#928374", diff_added_line_number_bg="#3a432b", diff_removed_line_number_bg="#46332c",
    syntax_comment="#928374", syntax_keyword="#fb4934", syntax_function="#fabd2f",
    syntax_variable="#ebdbb2", syntax_string="#b8bb26", syntax_number="#d3869b",
    syntax_type="#8ec07c", syntax_operator="#fe8019", syntax_punctuation="#ebdbb2",
    markdown_heading="#fabd2f", markdown_link="#8ec07c", markdown_link_text="#83a598",
    markdown_code="#b8bb26", markdown_quote="#928374", markdown_strong="#fabd2f",
    markdown_hr="#504945", markdown_list_item="#fe8019", markdown_code_block="#ebdbb2",
)

ONE_DARK = _palette(
    background="#282c34", background_panel="#21252b", background_element="#2c313a",
    background_menu="#21252b",
    border="#3e4451", border_active="#6699cc", border_subtle="#31353f",
    primary="#61afef", secondary="#56b6c2", accent="#c678dd",
    error="#e06c75", warning="#e5c07b", success="#98c379", info="#56b6c2",
    text="#abb2bf", text_muted="#5c6370",
    diff_added="#98c379", diff_removed="#e06c75", diff_context="#abb2bf",
    diff_hunk_header="#61afef", diff_highlight_added="#98c379", diff_highlight_removed="#e06c75",
    diff_added_bg="#2b3531", diff_removed_bg="#3a2d31", diff_context_bg="#21252b",
    diff_line_number="#5c6370", diff_added_line_number_bg="#2b3531", diff_removed_line_number_bg="#3a2d31",
    syntax_comment="#5c6370", syntax_keyword="#c678dd", syntax_function="#61afef",
    syntax_variable="#e06c75", syntax_string="#98c379", syntax_number="#d19a66",
    syntax_type="#e5c07b", syntax_operator="#56b6c2", syntax_punctuation="#abb2bf",
    markdown_heading="#61afef", markdown_link="#56b6c2", markdown_link_text="#61afef",
    markdown_code="#98c379", markdown_quote="#d19a66", markdown_strong="#d19a66",
    markdown_hr="#3e4451", markdown_list_item="#61afef", markdown_code_block="#abb2bf",
)

# True-black OLED theme: zero-luminance background, high-chroma accents tuned
# to stay crisp on pure black.
BLACKOUT = _palette(
    background="#000000", background_panel="#0d0d0d", background_element="#161616",
    background_menu="#0d0d0d",
    border="#2a2a2a", border_active="#4d4d4d", border_subtle="#1f1f1f",
    primary="#ffc66d", secondary="#6cb6ff", accent="#d2a8ff",
    error="#ff7b72", warning="#ffa657", success="#7ee787", info="#79c0ff",
    text="#eaeaea", text_muted="#7a7a7a",
    diff_added="#7ee787", diff_removed="#ff7b72", diff_context="#8b949e",
    diff_hunk_header="#79c0ff", diff_highlight_added="#7ee787", diff_highlight_removed="#ff7b72",
    diff_added_bg="#0f2417", diff_removed_bg="#2d1214", diff_context_bg="#0d0d0d",
    diff_line_number="#6e7681", diff_added_line_number_bg="#11251a", diff_removed_line_number_bg="#291315",
    syntax_comment="#6e7681", syntax_keyword="#ff7b72", syntax_function="#d2a8ff",
    syntax_variable="#ff9da7", syntax_string="#a5d6ff", syntax_number="#79c0ff",
    syntax_type="#ffa657", syntax_operator="#79c0ff", syntax_punctuation="#eaeaea",
    markdown_heading="#d2a8ff", markdown_link="#ffa657", markdown_link_text="#79c0ff",
    markdown_code="#7ee787", markdown_quote="#8b949e", markdown_strong="#ffa657",
    markdown_hr="#6e7681", markdown_list_item="#ffc66d", markdown_code_block="#eaeaea",
)

# Classic black / orange / gray terminal look: pure dark neutrals, one loud
# orange for primaries and highlights, everything else quiet grays.
DARK = _palette(
    background="#0b0b0b", background_panel="#141414", background_element="#1c1c1c",
    background_menu="#141414",
    border="#333333", border_active="#565656", border_subtle="#242424",
    primary="#ffaa33", secondary="#b0b0b0", accent="#8a8a8a",
    error="#e5534b", warning="#ffd166", success="#6fbf73", info="#a0a0a0",
    text="#d6d6d6", text_muted="#8a8a8a",
    diff_added="#85bf8a", diff_removed="#e5534b", diff_context="#a8a8a8",
    diff_hunk_header="#ffab40", diff_highlight_added="#85bf8a", diff_highlight_removed="#e5534b",
    diff_added_bg="#13251a", diff_removed_bg="#2b1613", diff_context_bg="#141414",
    diff_line_number="#7a7a7a", diff_added_line_number_bg="#152218", diff_removed_line_number_bg="#27140f",
    syntax_comment="#6e6e6e", syntax_keyword="#ffaa33", syntax_function="#ffc06b",
    syntax_variable="#d6d6d6", syntax_string="#bdbdbd", syntax_number="#ffab40",
    syntax_type="#cfcfcf", syntax_operator="#9e9e9e", syntax_punctuation="#a8a8a8",
    markdown_heading="#ffaa33", markdown_link="#ffc06b", markdown_link_text="#b0b0b0",
    markdown_code="#e0e0e0", markdown_quote="#8a8a8a", markdown_strong="#f0f0f0",
    markdown_hr="#4a4a4a", markdown_list_item="#ffaa33", markdown_code_block="#d6d6d6",
)

# -- light themes -------------------------------------------------------------

SOLARIZED_LIGHT = _palette(
    background="#fdf6e3", background_panel="#eee8d5", background_element="#eee8d5",
    background_menu="#eee8d5",
    border="#93a1a1", border_active="#657b83", border_subtle="#ddd6c1",
    primary="#cb4b16", secondary="#268bd2", accent="#6c71c4",
    error="#dc322f", warning="#b58900", success="#859900", info="#2aa198",
    text="#073642", text_muted="#93a1a1",
    diff_added="#859900", diff_removed="#dc322f", diff_context="#586e75",
    diff_hunk_header="#268bd2", diff_highlight_added="#859900", diff_highlight_removed="#dc322f",
    diff_added_bg="#e6efdc", diff_removed_bg="#f2ded7", diff_context_bg="#eee8d5",
    diff_line_number="#93a1a1", diff_added_line_number_bg="#e6efdc", diff_removed_line_number_bg="#f2ded7",
    syntax_comment="#93a1a1", syntax_keyword="#859900", syntax_function="#268bd2",
    syntax_variable="#b58900", syntax_string="#2aa198", syntax_number="#d33682",
    syntax_type="#cb4b16", syntax_operator="#6c71c4", syntax_punctuation="#073642",
    markdown_heading="#268bd2", markdown_link="#cb4b16", markdown_link_text="#2aa198",
    markdown_code="#859900", markdown_quote="#93a1a1", markdown_strong="#b58900",
    markdown_hr="#93a1a1", markdown_list_item="#cb4b16", markdown_code_block="#073642",
)

CATPPUCCIN_LATTE = _palette(
    background="#eff1f5", background_panel="#e6e9ef", background_element="#ccd0da",
    background_menu="#e6e9ef",
    border="#bcc0cc", border_active="#8c8fa1", border_subtle="#ccd0da",
    primary="#fe640b", secondary="#1e66f5", accent="#8839ef",
    error="#d20f39", warning="#df8e1d", success="#40a02b", info="#179299",
    text="#4c4f69", text_muted="#9ca0b0",
    diff_added="#40a02b", diff_removed="#d20f39", diff_context="#6c6f85",
    diff_hunk_header="#1e66f5", diff_highlight_added="#40a02b", diff_highlight_removed="#d20f39",
    diff_added_bg="#dcebdd", diff_removed_bg="#ecdcd7", diff_context_bg="#e6e9ef",
    diff_line_number="#9ca0b0", diff_added_line_number_bg="#dcebdd", diff_removed_line_number_bg="#ecdcd7",
    syntax_comment="#9ca0b0", syntax_keyword="#8839ef", syntax_function="#fe640b",
    syntax_variable="#d20f39", syntax_string="#40a02b", syntax_number="#df8e1d",
    syntax_type="#179299", syntax_operator="#04a5e5", syntax_punctuation="#4c4f69",
    markdown_heading="#8839ef", markdown_link="#fe640b", markdown_link_text="#1e66f5",
    markdown_code="#40a02b", markdown_quote="#df8e1d", markdown_strong="#df8e1d",
    markdown_hr="#9ca0b0", markdown_list_item="#fe640b", markdown_code_block="#4c4f69",
)

GITHUB_LIGHT = _palette(
    background="#ffffff", background_panel="#f6f8fa", background_element="#eaeef2",
    background_menu="#f6f8fa",
    border="#d0d7de", border_active="#0969da", border_subtle="#d8dee4",
    primary="#0969da", secondary="#0550ae", accent="#8250df",
    error="#cf222e", warning="#9a6700", success="#1a7f37", info="#0969da",
    text="#1f2328", text_muted="#57606a",
    diff_added="#1a7f37", diff_removed="#cf222e", diff_context="#57606a",
    diff_hunk_header="#8250df", diff_highlight_added="#1a7f37", diff_highlight_removed="#cf222e",
    diff_added_bg="#dafbe1", diff_removed_bg="#ffebe9", diff_context_bg="#f6f8fa",
    diff_line_number="#6e7781", diff_added_line_number_bg="#dafbe1", diff_removed_line_number_bg="#ffebe9",
    syntax_comment="#6e7781", syntax_keyword="#cf222e", syntax_function="#8250df",
    syntax_variable="#953800", syntax_string="#0a3069", syntax_number="#0550ae",
    syntax_type="#0550ae", syntax_operator="#0550ae", syntax_punctuation="#1f2328",
    markdown_heading="#1f2328", markdown_link="#0969da", markdown_link_text="#0969da",
    markdown_code="#cf222e", markdown_quote="#57606a", markdown_strong="#1f2328",
    markdown_hr="#d0d7de", markdown_list_item="#57606a", markdown_code_block="#1f2328",
)


# Insertion order IS the listing order: dark themes first, light last.
THEMES: dict[str, dict[str, str]] = {
    # dark
    "opencode": OPENSE_DARK,
    "opencode-dark": OPENSE_DARK,  # alias used by older configs / README
    "dark": DARK,
    "tokyo-night": TOKYO_NIGHT,
    "blackout": BLACKOUT,
    "catppuccin-mocha": CATPPUCCIN_MOCHA,
    "dracula": DRACULA,
    "nord": NORD,
    "gruvbox-dark": GRUVBOX_DARK,
    "one-dark": ONE_DARK,
    "solarized": SOLARIZED,
    # light
    "solarized-light": SOLARIZED_LIGHT,
    "catppuccin-latte": CATPPUCCIN_LATTE,
    "github-light": GITHUB_LIGHT,
}

DARK_THEMES = [
    "opencode", "opencode-dark", "dark", "tokyo-night", "blackout", "catppuccin-mocha",
    "dracula", "nord", "gruvbox-dark", "one-dark", "solarized",
]
LIGHT_THEMES = ["solarized-light", "catppuccin-latte", "github-light"]


def theme_names() -> list[str]:
    """All selectable theme names, dark ones first (alias `opencode-dark`
    hidden from listings — `opencode` is the canonical dark entry)."""
    return [k for k in THEMES if k != "opencode-dark"]


# -- active theme -------------------------------------------------------------
# Widgets resolve colors at render time via active_theme(); set_active_theme()
# is called at startup (cfg.theme) and by /theme + Settings, so switching
# re-styles the running app without restart.

_active_name = "opencode"


def set_theme_applier(fn) -> None:
    """Register a callback invoked whenever the active theme changes.

    The TUI registers one that rebuilds + reapplies the Textual design-token
    theme, so CSS using $variables restyles live (chrome, panels, list rows,
    buttons) instead of staying frozen on the startup palette."""
    global _theme_applier
    _theme_applier = fn


_theme_applier = None


def build_textual_theme():
    """A Textual design-token theme mirroring the ACTIVE opencode palette.

    Drives every $variable used in widget CSS ($background/$surface/$panel/
    $primary/...) so structural chrome follows theme switches — previously
    those stayed hardcoded dark while Python-rendered text switched, leaving
    light themes unreadable (dark ink on dark chrome)."""
    from textual.theme import Theme as TTheme

    t = active_theme()

    def lum(h: str) -> float:
        h = h.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

    def mix(a: str, b: str, t: float) -> str:
        """Blend hex a toward b by t (0..1) — used for the selection tint."""
        pa = tuple(int(a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        pb = tuple(int(b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        m = tuple(round(pa[i] + (pb[i] - pa[i]) * t) for i in range(3))
        return "#{:02x}{:02x}{:02x}".format(*m)

    # List-row highlights follow official opencode: the selected row gets a
    # full primary background (not a faint tint), so the highlight is
    # unmistakable on every palette, dark or light.
    highlight = t.c("primary")
    _ = mix  # keep the blender for future tints (avoids an unused-def lint)
    return TTheme(
        name=f"opencode-py-{t.name}",
        primary=t.c("primary"),
        secondary=t.c("secondary"),
        accent=t.c("accent"),
        foreground=t.c("text"),
        background=t.c("background"),
        surface=t.c("background_element"),
        panel=t.c("background_panel"),
        success=t.c("success"),
        warning=t.c("warning"),
        error=t.c("error"),
        dark=lum(t.c("background")) < 0.5,
        variables={
            # Only BUILT-IN token names may be referenced from CSS in this
            # Textual version — custom $names crash the stylesheet parser.
            "block-cursor-background": highlight,
            "block-cursor-text-style": "bold",
            "input-cursor-background": t.c("primary"),
            "input-cursor-foreground": t.c("background"),
        },
    )


def set_active_theme(name: str) -> str:
    """Select the live theme; unknown names fall back to opencode dark.
    Returns the name actually applied."""
    global _active_name
    _active_name = name if name in THEMES else "opencode"
    if _theme_applier is not None:
        try:
            _theme_applier()
        except Exception:
            pass
    return _active_name


def active_theme() -> Theme:
    return get_theme(_active_name)

TEXTUAL_THEME_NAME = "opencode_py"


# Default agent colors, matched to opencode's local agent palette (ordered by
# the built-in agents: build -> secondary blue, plan -> accent purple, ...).
AGENT_COLORS: dict[str, str] = {
    "build": "#fab283",  # build's accent is the opencode primary orange
    "plan": "#5c9cf5",   # secondary blue
    "general": "#9d7cd8",
    "test": "#7fd88f",
}

_AGENT_PALETTE = [
    "#5c9cf5",  # secondary
    "#9d7cd8",  # accent
    "#7fd88f",  # success
    "#f5a742",  # warning
    "#fab283",  # primary
    "#e06c75",  # error
    "#56b6c2",  # info
]


@dataclass
class Theme:
    name: str
    colors: dict[str, str]

    def c(self, key: str) -> str:
        return self.colors.get(key, OPENSE_DARK.get(key, "#ffffff"))

    def agent_color(self, agent: str) -> str:
        """Color accent for an agent name — THEME-AWARE so tags stay readable
        on both dark and light palettes (the old hardcoded hexes were tuned
        for the dark theme and vanished on light backgrounds)."""
        roles = {
            "build": "primary",
            "plan": "secondary",
            "general": "accent",
            "explore": "info",
            "test": "success",
        }
        key = roles.get(agent)
        if key:
            return self.colors.get(key) or OPENSE_DARK[key]
        palette_keys = ("primary", "secondary", "accent", "success", "warning", "error", "info")
        return self.c(palette_keys[hash(agent) % len(palette_keys)])


def get_theme(name: str) -> Theme:
    return Theme(name=name, colors=THEMES.get(name, OPENSE_DARK))
