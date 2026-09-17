"""Edit tool replacer cascade tests: near-miss oldStrings get salvaged."""

from opencode_py.tools.edit import _edit


SAMPLE = '''def hello():
    print("Hello, World!")

def world():
    return 42
'''


def test_exact_match_still_works(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), '    print("Hello, World!")', "    print('hi')")
    assert r.get("error") is None
    assert "    print('hi')" in p.read_text()


def test_trailing_space_in_old_string_salvaged(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), '    print("Hello, World!") ', "    print('hi')")
    assert r.get("error") is None
    assert "    print('hi')" in p.read_text()


def test_wrong_indentation_salvaged(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), '      print("Hello, World!")', "      print('hi')")
    assert r.get("error") is None
    assert "      print('hi')" in p.read_text()


def test_crlf_file_salvaged(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE.replace("\n", "\r\n"))
    r = _edit(str(p), "def world():", "def world():  # v2")
    assert r.get("error") is None
    assert "def world():  # v2" in p.read_text()


def test_blank_line_context_removed(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), '    print("Hello, World!")\n\ndef world():', '    print("Hello, World!")\ndef world():')
    assert r.get("error") is None
    assert "\n\n" not in p.read_text()


def test_ambiguous_fallback_still_errors(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), "def ", "fn ")
    assert r.get("error") is True
    assert "multiple matches" in r["output"].lower()


def test_unmatched_still_errors(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), "totally not in the file", "x")
    assert r.get("error") is True
    assert "not found" in r["output"]


def test_fallback_replace_all(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("  a\n   a\n    a\n")
    r = _edit(str(p), " a", "b", True)
    assert r.get("error") is None
    assert p.read_text() == " b\n  b\n   b\n"


def test_fuzzy_midfile_does_not_swallow_next_line(tmp_path):
    """A fuzzy (whitespace-mismatched) match must not merge the line after the
    edit: the replacement span must exclude the trailing newline of the last
    matched line unless old itself ends in a newline."""
    p = tmp_path / "t.py"
    p.write_text("a\nb\nc\n")
    r = _edit(str(p), "b ", "B")  # trailing-space old -> fuzzy match
    assert r.get("error") is None
    assert p.read_text() == "a\nB\nc\n"


def test_fuzzy_midfile_multi_line_no_newline_merge(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("l1\nl2\nl3\nl4\n")
    r = _edit(str(p), "l2\nl3 ", "L")
    assert r.get("error") is None
    assert p.read_text() == "l1\nL\nl4\n"


def test_fuzzy_explicit_trailing_newline_still_swallowed(tmp_path):
    """An old string that starts or ends with a newline is rejected up front:
    replacing a boundary newline silently merges adjacent lines, so we refuse
    instead of corrupting the file."""
    p = tmp_path / "t.py"
    p.write_text("a\nb\nc\n")
    r = _edit(str(p), "b\n", "B")
    assert r.get("error") is True
    assert "newline" in r["output"].lower()
    assert p.read_text() == "a\nb\nc\n"  # untouched


def test_boundary_newline_rejected(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("a\nb\nc\n")
    r = _edit(str(p), "\nb", "B")
    assert r.get("error") is True
    assert p.read_text() == "a\nb\nc\n"


def test_fuzzy_last_line_preserves_trailing_newline(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("a\nb\nc\n")
    r = _edit(str(p), "c ", "C")
    assert r.get("error") is None
    assert p.read_text() == "a\nb\nC\n"


def test_crlf_line_endings_preserved(tmp_path):
    """Editing a CRLF file must not rewrite the whole file to LF."""
    p = tmp_path / "t.py"
    p.write_bytes(b"a\r\nb\r\nc\r\n")
    r = _edit(str(p), "b", "B")
    assert r.get("error") is None
    assert p.read_bytes() == b"a\r\nB\r\nc\r\n"


def test_crlf_preserved_after_fuzzy_edit(tmp_path):
    p = tmp_path / "t.py"
    p.write_bytes(b"def a():\r\n    x = 1\r\n    y = 2\r\n")
    r = _edit(str(p), "x = 1 ", "x = 3")  # trailing space -> fuzzy path
    assert r.get("error") is None
    assert p.read_bytes() == b"def a():\r\n    x = 3\r\n    y = 2\r\n"


# --------------------------------------------------------------------------
# write tool: CRLF preservation, exact-bytes round trip
# --------------------------------------------------------------------------

def test_write_preserves_crlf_content(tmp_path):
    from opencode_py.tools.write import _write

    p = tmp_path / "crlf.txt"
    r = _write(str(p), "a\r\nb\r\n")
    assert r.get("error") is None
    assert p.read_bytes() == b"a\r\nb\r\n"


def test_write_exact_bytes_no_extra_newline(tmp_path):
    from opencode_py.tools.write import _write

    p = tmp_path / "n.txt"
    r = _write(str(p), "x = 1")  # no trailing newline
    assert r.get("error") is None
    assert p.read_bytes() == b"x = 1"


def test_write_nested_dirs_created(tmp_path):
    from opencode_py.tools.write import _write

    p = tmp_path / "a" / "b" / "c.txt"
    r = _write(str(p), "deep")
    assert r.get("error") is None
    assert p.read_text() == "deep"