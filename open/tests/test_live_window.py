"""Live chat window + auto low-RAM profile (pure logic, no TUI pump needed)."""

from opencode_py.config import Config
from opencode_py.util.ram import is_low_ram, mem_total_mb


def test_low_ram_defaults_to_auto_with_window_120():
    cfg = Config.from_dict({})
    assert cfg.low_ram == "auto"
    assert cfg.chat_live_window == 120


def test_low_ram_on_applies_budgets_and_window_stays():
    cfg = Config.from_dict({"low_ram": "on"})
    assert cfg.low_ram_active is True
    assert cfg.context_budget == 30000
    assert cfg.compaction_tail_turns == 1
    assert cfg.trim_keep_turns == 1
    assert cfg.trim_max_chars == 200
    assert cfg.session_file_cap == 50
    assert cfg.tool_output_max_lines == 200
    assert cfg.tool_output_max_bytes == 10240


def test_low_ram_off_keeps_desktop_budgets():
    cfg = Config.from_dict({"low_ram": "off"})
    assert cfg.low_ram_active is False
    assert cfg.context_budget == 120000
    assert cfg.compaction_tail_turns == 2
    assert cfg.trim_keep_turns == 2
    assert cfg.session_file_cap == 0


def test_low_ram_explicit_values_win_over_preset():
    cfg = Config.from_dict({
        "low_ram": "on",
        "context_budget": 60000,
        "chat_live_window": 40,
        "tool_output": {"max_lines": 500, "max_bytes": 20480},
        "compaction": {"tail_turns": 3},
        "trim": {"keep_turns": 3, "max_chars": 900},
        "session_file_cap": 9,
    })
    assert cfg.context_budget == 60000
    assert cfg.chat_live_window == 40
    assert cfg.tool_output_max_lines == 500
    assert cfg.tool_output_max_bytes == 20480
    assert cfg.compaction_tail_turns == 3
    assert cfg.trim_keep_turns == 3
    assert cfg.trim_max_chars == 900
    assert cfg.session_file_cap == 9


def test_low_ram_roundtrips_through_as_dict():
    cfg = Config.from_dict({"low_ram": "on", "chat_live_window": 60})
    out = cfg.as_dict()
    assert out["low_ram"] == "on"
    assert out["low_ram_active"] is True
    assert out["chat_live_window"] == 60


def test_low_ram_aliases_normalize():
    assert Config.from_dict({"low_ram": "yes"}).low_ram == "on"
    assert Config.from_dict({"low_ram": "no"}).low_ram == "off"
    assert Config.from_dict({"low_ram": "bogus"}).low_ram == "auto"


def test_chat_live_window_validation():
    assert Config.from_dict({"chat_live_window": 0}).chat_live_window == 0
    # negative clamps to 0 (unlimited), matching the validator's valid range
    assert Config.from_dict({"chat_live_window": -5}).chat_live_window == 0
    assert Config.from_dict({"chat_live_window": "nope"}).chat_live_window == 120


def test_mem_total_mb_parses_and_unknown_is_zero(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:        1813256 kB\nMemFree: 1 kB\n", encoding="utf-8")
    assert mem_total_mb(str(f)) == 1813256 // 1024
    assert mem_total_mb(str(tmp_path / "missing")) == 0
    bad = tmp_path / "bad"
    bad.write_text("nope\n", encoding="utf-8")
    assert mem_total_mb(str(bad)) == 0


def test_is_low_ram_threshold(tmp_path):
    small = tmp_path / "small"
    small.write_text("MemTotal:        1048576 kB\n", encoding="utf-8")
    assert is_low_ram(path=str(small)) is True
    big = tmp_path / "big"
    big.write_text("MemTotal:       16777216 kB\n", encoding="utf-8")
    assert is_low_ram(path=str(big)) is False
    assert is_low_ram(path=str(tmp_path / "missing")) is False


def test_config_validate_accepts_low_ram_and_window():
    from opencode_py.commands import _validate_config

    assert _validate_config(Config.from_dict({})) == []
    bad = Config.from_dict({"low_ram": "auto", "chat_live_window": 120})
    bad.low_ram = "sometimes"
    bad.chat_live_window = -1
    problems = _validate_config(bad)
    assert any("low_ram" in p for p in problems)
    assert any("chat_live_window" in p for p in problems)


def test_history_pill_counts_hidden_without_app():
    from opencode_py.tui.chat_view import ChatView

    # Headless: no app/pump exists, so build the view without __init__
    # (which would paint) and exercise only the pure pill logic.
    c = ChatView.__new__(ChatView)
    c._history_pending = []
    c._hidden_count = 0
    c._history_anchor = None
    c._history_session_id = ""
    c._history_chunk = 80
    assert c._history_count() == 0
    assert c._history_pill_text() == ""
    c._hidden_count = 3879
    assert c._history_count() == 3879
    assert c._history_pill_text() == "↑ 3879 older messages — scroll up to load"
    c._hidden_count = 0
    c._history_pending = [1, 2, 3]
    assert c._history_count() == 3
    assert c._history_pill_text() == "↑ 3 older messages — scroll up to load"
    # hidden wins over pending (same-session hides are cheaper to unhide)
    c._hidden_count = 5
    assert c._history_pill_text() == "↑ 5 older messages — scroll up to load"


def test_frozen_bubble_skips_rerender():
    # Freeze RULES (headless-safe): completed tool runs end frozen, running
    # ones stay live; watchers unfreeze on any state flip. The _refresh
    # no-op itself is exercised live (frozen bubbles skip spinner repaints).
    import inspect

    from opencode_py.tui.chat_view import MessageBubble

    src_update = inspect.getsource(MessageBubble.update_tool)
    assert "self.freeze()" in src_update
    assert 'not in ("running", "pending")' in src_update
    src_refresh = inspect.getsource(MessageBubble._refresh)
    assert "frozen" in src_refresh and "return" in src_refresh
    for watcher in ("watch_queued", "watch_streaming", "watch_expanded", "watch_selected"):
        assert "unfreeze" in inspect.getsource(getattr(MessageBubble, watcher))
    assert "unfreeze" in inspect.getsource(MessageBubble.set_tool_metadata)


def test_spinner_paces_real_arrivals():
    from opencode_py.tui.chat_view import MessageBubble

    b = MessageBubble.__new__(MessageBubble)
    b._spinner = 0
    b._tick_slow = False
    b._last_token_mono = 0.0
    b._token_chars = 0
    # no arrivals yet: no stamp
    assert float(getattr(b, "_last_token_mono", 0.0) or 0.0) == 0.0
    b.note_stream_activity(50)
    assert float(getattr(b, "_last_token_mono", 0.0) or 0.0) > 0.0
    assert int(getattr(b, "_token_chars", 0)) == 50
    b.note_stream_activity(25)
    assert int(getattr(b, "_token_chars", 0)) == 75


def test_input_bar_spinner_paces_arrivals():
    from opencode_py.tui.input_bar import InputBar

    bar = InputBar.__new__(InputBar)
    bar._last_token_mono = 0.0
    bar._tick_slow = False
    assert float(getattr(bar, "_last_token_mono", 0.0) or 0.0) == 0.0
    bar.note_stream_activity(10)
    assert float(getattr(bar, "_last_token_mono", 0.0) or 0.0) > 0.0
