"""_is_binary_sample / _is_binary must not misflag UTF-8 text files whose
multibyte character is cut in half by the fixed 1024-byte sampling window
(the README.md box-drawing divider bug: a `─` straddling byte 1024 made the
whole file report as binary)."""

from __future__ import annotations

from opencode_py.tools.read import _is_binary_sample
from opencode_py.tools.summarize_file import _is_binary


def test_multibyte_char_straddling_window_is_not_binary():
    # '─' encodes to E2 94 80; 1020 'a' bytes put the window cut INSIDE it
    data = b"a" * 1020 + "─".encode() * 10
    sample = data[:1024]
    assert sample[-2:] == b"\x80\xe2"  # window ends mid-character
    assert not _is_binary_sample(data)
    assert not _is_binary(data)


def test_em_dash_straddling_window_is_not_binary():
    data = b"x" * 1023 + "—".encode() * 4
    assert data[1023:1024] == b"\xe2"
    assert not _is_binary_sample(data)


def test_genuinely_invalid_utf8_still_reports_binary():
    assert _is_binary_sample(b"\xff\xfe broken \x80bytes")
    assert _is_binary(b"\xc3\x28 more junk")


def test_nonprintable_ratio_still_reports_binary():
    assert _is_binary_sample(b"\x00" * 800)


def test_plain_ascii_and_short_samples_pass():
    assert not _is_binary_sample(b"plain ascii text\n" * 5)
    assert not _is_binary_sample(b"")  # empty never flags
