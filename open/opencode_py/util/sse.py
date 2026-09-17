"""SSE (Server-Sent Events) parsing from an httpx streaming response.

Robust to chunk boundaries, CRLF, comments, and the [DONE] sentinel used by
OpenAI-compatible endpoints.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator


class SSEDecoder:
    """Incremental SSE decoder; feed bytes, read events."""

    MAX_BUFFER_BYTES = 8 * 1024 * 1024
    MAX_DATA_BYTES = 8 * 1024 * 1024

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._event = ""
        self._data: list[str] = []
        self._data_bytes = 0

    def feed(self, chunk: bytes) -> list[dict]:
        self._buffer.extend(chunk)
        if len(self._buffer) > self.MAX_BUFFER_BYTES:
            del self._buffer[: len(self._buffer) - self.MAX_BUFFER_BYTES]
            self._data = []
            self._data_bytes = 0
        return self._process()

    def _process(self) -> list[dict]:
        events: list[dict] = []
        while True:
            idx = self._buffer.find(b"\n")
            if idx == -1:
                return events
            raw = bytes(self._buffer[:idx]).decode("utf-8", errors="replace")
            del self._buffer[: idx + 1]
            line = raw.rstrip("\r")
            if line == "":
                evt = self._emit()
                if evt is not None:
                    events.append(evt)
                continue
            if line.startswith(":"):
                continue  # comment
            if line.startswith("event:"):
                self._event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                piece = line[len("data:"):].lstrip(" ")
                self._data.append(piece)
                self._data_bytes += len(piece)
                while self._data and self._data_bytes > self.MAX_DATA_BYTES:
                    self._data_bytes -= len(self._data.pop(0))
            elif line.startswith("id:"):
                pass
            elif line.startswith("retry:"):
                pass
            else:
                # field without colon is ignored; unknown field name ignored
                pass
        return events

    def _emit(self) -> dict | None:
        if not self._data:
            self._event = ""
            return None
        data = "\n".join(self._data)
        self._data = []
        self._data_bytes = 0
        event = self._event or "message"
        self._event = ""
        return {"event": event, "data": data}

    def close(self) -> list[dict]:
        # flush any remaining (newline-less) buffer content and pending data
        events: list[dict] = []
        if self._buffer:
            self._buffer.extend(b"\n")
            events.extend(self._process())
        if self._data:
            evt = self._emit()
            if evt is not None:
                events.append(evt)
        return events


def iter_sse_lines(stream: Iterator[bytes] | AsyncIterator[bytes]) -> Iterator[dict]:
    """Iterate decoded SSE events from a sync or async byte iterator."""
    decoder = SSEDecoder()
    if hasattr(stream, "__aiter__"):
        # async: collect via running loop
        raise TypeError("iter_sse_lines requires a sync iterator; use async_iter_sse_lines")

    for chunk in stream:
        for evt in decoder.feed(chunk):
            yield evt
    yield from decoder.close()


async def async_iter_sse_lines(stream: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    decoder = SSEDecoder()
    async for chunk in stream:
        for evt in decoder.feed(chunk):
            yield evt
    for evt in decoder.close():
        yield evt


def parse_sse_block(text: str) -> dict | None:
    """Parse a single SSE block (used in tests / fixtures)."""
    decoder = SSEDecoder()
    events = decoder.feed(text.encode())
    if not events:
        events = decoder.close()
    return events[-1] if events else None
