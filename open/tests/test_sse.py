"""Tests for the SSE decoder, especially tail/flush handling (regression for
the "last chunk without trailing newline silently drops content" bug)."""

from opencode_py.util.sse import SSEDecoder, parse_sse_block


def test_feed_basic():
    d = SSEDecoder()
    events = d.feed(b'data: hello\n\n')
    assert len(events) == 1
    assert events[0]["event"] == "message"
    assert events[0]["data"] == "hello"


def test_close_flushes_tail_without_newline():
    d = SSEDecoder()
    d.feed(b'data: hello')
    events = d.close()
    assert len(events) == 1
    assert events[0]["data"] == "hello"


def test_close_flushes_multiple_tail_events():
    d = SSEDecoder()
    flushed_by_feed = d.feed(b'data: a\n\ndata: b')
    events = d.close()
    assert [e["data"] for e in flushed_by_feed] == ["a"]
    assert [e["data"] for e in events] == ["b"]


def test_close_returns_empty_when_nothing_pending():
    d = SSEDecoder()
    d.feed(b'data: done\n\n')
    assert d.close() == []


def test_crlf_and_chunk_boundaries():
    d = SSEDecoder()
    events = d.feed(b'data: he')
    events += d.feed(b'llo\r\n\r\ndata: x\r\n\r\n')
    assert [e["data"] for e in events] == ["hello", "x"]


def test_multi_line_data():
    d = SSEDecoder()
    events = d.feed(b'data: a\ndata: b\n\n')
    assert events[0]["data"] == "a\nb"


def test_done_sentinel_is_data():
    d = SSEDecoder()
    events = d.feed(b'data: [DONE]\n\n')
    assert events[0]["data"] == "[DONE]"


def test_parse_sse_block_returns_dict():
    assert parse_sse_block("data: hi\n\n") == {"event": "message", "data": "hi"}
    assert parse_sse_block("data: hi") == {"event": "message", "data": "hi"}


def test_close_after_partial_event_and_full_event():
    d = SSEDecoder()
    events = d.feed(b'data: a\n\ndata: b')
    assert [e["data"] for e in events] == ["a"]
    tail = d.close()
    assert [e["data"] for e in tail] == ["b"]
