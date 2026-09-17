"""Context ledger: remembers exactly which file content has already been sent
to the model, so nothing is ever delivered twice.

Why this exists: the model has no memory between turns — every tool result is
uploaded again inside each new request. When the agent re-reads a region it
already saw (or reads around lines a previous grep already printed), the old
code re-sent identical bytes: pure mobile-data waste that then sticks in the
context and re-uploads on EVERY following turn.

The ledger records delivered line-ranges per file version (path + mtime_ns +
size fingerprint). Consumers:
- read      : marks returned windows; stubs/trim re-delivery
- grep      : marks printed match lines (they reached the model too)
- compaction: reset_for_compaction() — after history summarization the old
              bodies are gone from the model's head, so it must be able to
              receive them fresh again.

Correctness rules baked in here:
- A changed file (different mtime_ns/size) starts with an empty slate.
- Ranges are 1-indexed INCLUSIVE line numbers, matching read's output.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

_LOCK = threading.Lock()

# generation bumps when the conversation context gets compacted: everything
# recorded before that may no longer be visible to the model.
_generation = 0

# path -> entry; entries evicted oldest-first past MAX_FILES so long sessions
# can't grow this forever.
_files: "OrderedDict[str, _FileLedger]" = OrderedDict()

MAX_FILES = 500


class _FileLedger:
    __slots__ = ("mtime_ns", "size", "ranges")

    def __init__(self, mtime_ns: int, size: int):
        self.mtime_ns = mtime_ns
        self.size = size
        self.ranges: list[list[int]] = []  # sorted merged [start, end] pairs

    def matches(self, mtime_ns: int, size: int) -> bool:
        return self.mtime_ns == mtime_ns and self.size == size

    def add(self, start: int, end: int) -> None:
        if end < start:
            start, end = end, start
        new_start, new_end = start, end
        merged: list[list[int]] = []
        for s, e in self.ranges:
            if e < new_start - 1:
                merged.append([s, e])
            elif s > new_end + 1:
                merged.append([s, e])
            else:
                # overlapping or adjacent -> absorb into the growing range
                new_start = min(new_start, s)
                new_end = max(new_end, e)
        merged.append([new_start, new_end])
        merged.sort()
        self.ranges = merged


def _entry_for(path: str, mtime_ns: int, size: int, *, create: bool) -> "_FileLedger | None":
    """Current-version ledger entry for path, or None. A stale entry (file
    edited since delivery) is dropped so the file starts with a clean slate."""
    entry = _files.get(path)
    if entry is not None and not entry.matches(mtime_ns, size):
        del _files[path]
        entry = None
    if entry is None:
        if not create:
            return None
        entry = _FileLedger(mtime_ns, size)
        _files[path] = entry
        while len(_files) > MAX_FILES:
            _files.popitem(last=False)
    else:
        _files.move_to_end(path)
    return entry


def mark_delivered(path: str, mtime_ns: int, size: int, start: int, end: int) -> None:
    """Record that lines start..end (inclusive) of this exact file version
    reached the model."""
    if end < start:
        start, end = end, start
    with _LOCK:
        entry = _entry_for(path, mtime_ns, size, create=True)
        entry.add(start, end)


def unseen_ranges(
    path: str, mtime_ns: int, size: int, start: int, end: int
) -> list[tuple[int, int]]:
    """Sub-ranges of [start, end] NOT yet delivered for this file version.

    Empty list => everything requested was already sent (caller should stub).
    Unknown/stale file => the whole window is unseen."""
    if end < start:
        start, end = end, start
    with _LOCK:
        entry = _entry_for(path, mtime_ns, size, create=False)
        if entry is None:
            return [(start, end)]
        covered = [
            (s, e) for s, e in entry.ranges if not (e < start or s > end)
        ]
    if not covered:
        return [(start, end)]
    unseen: list[tuple[int, int]] = []
    cursor = start
    for s, e in covered:
        if s > cursor:
            unseen.append((cursor, min(s - 1, end)))
        cursor = max(cursor, e + 1)
        if cursor > end:
            break
    if cursor <= end:
        unseen.append((cursor, end))
    return [(a, b) for a, b in unseen if b >= a]


def fully_delivered(path: str, mtime_ns: int, size: int, start: int, end: int) -> bool:
    return not unseen_ranges(path, mtime_ns, size, start, end)


def reset_for_compaction() -> None:
    """Conversation was compacted: previously delivered bodies may no longer
    be visible to the model, so every file must be deliverable fresh again."""
    global _generation
    with _LOCK:
        _files.clear()
        _generation += 1


def generation() -> int:
    with _LOCK:
        return _generation


def clear() -> None:
    """Test helper / manual wipe (does NOT bump the generation)."""
    global _generation
    with _LOCK:
        _files.clear()
