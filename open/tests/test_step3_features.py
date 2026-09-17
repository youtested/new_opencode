"""Step 3 features (trimmed): fast unit tests for the session grouping,
Markdown export and the /export command. TUI picker/runtime coverage lives in
the git history of this file; these seven tests cover the pure logic."""

from __future__ import annotations

import datetime

from opencode_py.commands import CommandContext, build_registry
from opencode_py.config import Config
from opencode_py.session import (
    Session,
    group_sessions,
    save_session,
    session_to_markdown,
)


def _ts(offset_days: float) -> float:
    return (
        datetime.datetime.now() - datetime.timedelta(days=offset_days)
    ).timestamp()


# ---------------------------------------------------------------------------
# group_sessions: Today / Yesterday / older-by-day, newest-first
# ---------------------------------------------------------------------------

def test_group_sessions_buckets_today_yesterday_older():
    older = _ts(9)
    yesterday = _ts(1.2)
    today_early = _ts(0)
    today_late = _ts(-0.01)  # a few minutes in the future vs the first stamp
    older_label = datetime.datetime.fromtimestamp(older).strftime("%A, %B %d, %Y")
    sessions = [
        {"id": "old", "created": older, "title": "old"},
        {"id": "yday", "created": yesterday, "title": "yday"},
        {"id": "today-early", "created": today_early, "title": "early"},
        {"id": "today-late", "created": today_late, "title": "late"},
    ]
    groups = group_sessions(sessions)
    labels = [label for label, _ in groups]
    assert labels[0] == "Today"
    assert labels[1] == "Yesterday"
    assert labels[2] == older_label
    assert [s["id"] for s in groups[0][1]] == ["today-late", "today-early"]
    assert [s["id"] for s in groups[1][1]] == ["yday"]


def test_group_sessions_accepts_session_objects():
    older = _ts(5)
    now = _ts(0)
    sessions = [
        Session({"id": "a", "created": older, "title": "old"}),
        Session({"id": "b", "created": now, "title": "new"}),
    ]
    groups = group_sessions(sessions)
    assert groups[0][0] == "Today"
    assert [s.id for s in groups[0][1]] == ["b"]


def test_group_sessions_drops_missing_created():
    groups = group_sessions([{"id": "x", "title": "x"}])
    assert groups[0][1][0]["id"] == "x"


def test_group_sessions_never_mixes_years():
    """older-day labels must include the year so sessions from different years
    that share the same month+day+weekday can't silently merge into one group."""
    from opencode_py.session import group_sessions

    # Anchor to the REAL current date so the fixture never rots (a hardcoded
    # "today" stops being today and the labels legitimately change). "mid" sits
    # 45 days back this year; "old" is the same month/day five years earlier —
    # the collision case that used to merge into one group.
    now = datetime.datetime.now()
    mid_dt = (now - datetime.timedelta(days=45)).replace(hour=10, minute=0, second=0, microsecond=0)
    try:
        old_dt = mid_dt.replace(year=now.year - 5)
    except ValueError:  # Feb 29 -> non-leap target year
        old_dt = mid_dt.replace(month=2, day=28, year=now.year - 5)
    today = datetime.datetime(now.year, now.month, now.day, 10, 0).timestamp()

    def mk(ts, ident):
        return {"id": ident, "created": ts, "title": ident}

    groups = group_sessions([mk(today, "t"), mk(old_dt.timestamp(), "old"), mk(mid_dt.timestamp(), "mid")])
    labels = [label for label, _ in groups]
    assert labels[0] == "Today"
    mid_label = mid_dt.strftime("%A, %B %d, %Y")
    old_label = old_dt.strftime("%A, %B %d, %Y")
    assert mid_label in labels, labels
    assert old_label in labels, labels
    ids = {s["id"] for _, items in groups for s in items}
    assert ids == {"t", "old", "mid"}
    by_label = {label: [s["id"] for s in items] for label, items in groups}
    assert by_label[mid_label] == ["mid"]
    assert by_label[old_label] == ["old"]


# ---------------------------------------------------------------------------
# session_to_markdown export format
# ---------------------------------------------------------------------------

def test_session_to_markdown_includes_tools_and_skips_system():
    sess = Session(
        {
            "id": "testid",
            "title": "Build it",
            "provider": "opencode",
            "model": "deepseek-v4-flash-free",
            "agent": "build",
            "messages": [
                {"role": "system", "content": "you are an agent"},
                {"role": "user", "content": "make a file"},
                {
                    "role": "assistant",
                    "content": "on it",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "write", "arguments": '{"path": "a.txt"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "write", "content": "wrote a.txt"},
                {"role": "assistant", "content": "done"},
                {"role": "compaction", "content": "summarized earlier"},
            ],
        }
    )
    md = session_to_markdown(sess)
    assert md.startswith("# Build it")
    assert "deepseek-v4-flash-free" in md
    assert "make a file" in md
    assert "**write**" in md
    assert "a.txt" in md
    assert "wrote a.txt" in md
    assert "done" in md
    assert "Compaction" in md
    assert "you are an agent" not in md


def test_session_to_markdown_serializes_pretty_arguments():
    sess = Session(
        {
            "id": "x",
            "title": "t",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "a.txt"},
            ],
        }
    )
    md = session_to_markdown(sess)
    assert '"cmd": "ls"' in md
    assert "a.txt" in md


# ---------------------------------------------------------------------------
# /export command
# ---------------------------------------------------------------------------

def test_export_command_writes_markdown(tmp_path, monkeypatch):
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    monkeypatch.chdir(tmp_path)
    sess = Session(
        {
            "id": "abc123",
            "title": "Export me",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        }
    )
    save_session(sess)

    reg = build_registry()
    cmd = reg.get("export")
    assert cmd is not None
    out: list[str] = []

    class FakeEngine:
        session_id = "abc123"

    ctx = CommandContext(config=Config(), auth=None, engine=FakeEngine())
    ctx.reply = out.append
    cmd.handler(ctx, "")

    path = tmp_path / "opencode-session-abc123.md"
    assert path.exists()
    assert "Export me" in path.read_text(encoding="utf-8")
    assert any("opencode-session-abc123.md" in line for line in out)
