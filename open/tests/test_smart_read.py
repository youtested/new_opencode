"""Tests for smart-read + never-send-twice (context_ledger integration).

Covers:
- outline mode (python AST + non-python fallback), auto threshold
- dedup: full-overlap stub, force_full override, partial-overlap trimming
- cross-tool dedup: grep marks its printed lines
- invalidation: edited file delivers fresh; compaction reset delivers fresh
- legacy output format untouched for fresh small-file reads
"""

import time

import pytest

from opencode_py.tools import context_ledger as cl
from opencode_py.tools.grep import tool as grep_tool
from opencode_py.tools.read import AUTO_OUTLINE_LINES, tool as read_tool


@pytest.fixture(autouse=True)
def _clean_ledger():
    cl.clear()
    yield
    cl.clear()


def rread(filePath, **kw):
    return read_tool().run({"filePath": str(filePath), **kw})


def make_py(tmp_path, n_funcs=3, name="mod.py"):
    lines = ['"""module doc"""', "", "class Foo:", '    """cls"""', ""]
    for i in range(n_funcs):
        lines += [
            f"def func_{i}(x, y=0):",
            f'    """doc {i}"""',
            "    total = x + y",
            "    return total",
            "",
        ]
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# outline
# ---------------------------------------------------------------------------


def test_outline_python_shows_structure_not_bodies(tmp_path):
    p = make_py(tmp_path, n_funcs=2)
    out = rread(p, mode="outline")["output"]
    assert "<type>outline</type>" in out
    assert "class Foo:" in out
    assert "def func_0(x, y=0):" in out
    # bodies must NOT be in the outline
    assert "total = x + y" not in out
    # entries carry real line numbers (class at line 3 of the fixture)
    assert "\n3: class Foo:" in out


def test_outline_fallback_for_markdown(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nintro\n\n## Section A\n\ntext\n", encoding="utf-8")
    out = rread(p, mode="outline")["output"]
    assert "1: # Title" in out
    assert "5: ## Section A" in out


def test_auto_returns_outline_for_big_files_full_for_small(tmp_path):
    big = tmp_path / "big.py"
    big.write_text(
        "\n".join(f"def f_{i}():\n    return {i}\n" for i in range(AUTO_OUTLINE_LINES + 20)),
        encoding="utf-8",
    )
    out = rread(big)["output"]
    assert "<type>outline</type>" in out

    small = tmp_path / "small.py"
    small.write_text("x = 1\ny = 2\n", encoding="utf-8")
    out = rread(small)["output"]
    assert "<type>file</type>" in out
    assert "1: x = 1" in out

    # explicit narrow windows stay literal even on big files
    win = rread(big, offset=1, limit=10)["output"]
    assert "<type>file</type>" in win


# ---------------------------------------------------------------------------
# dedup: nothing gets sent twice
# ---------------------------------------------------------------------------


def test_identical_reread_returns_stub_force_full_overrides(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    first = rread(p)
    assert "1: alpha" in first["output"]

    again = rread(p)
    assert again["metadata"].get("dedup_stub") is True
    assert "1: alpha" not in again["output"]
    assert "already delivered" in again["output"]

    forced = rread(p, force_full=True)
    assert "1: alpha" in forced["output"]
    assert "beta" in forced["output"]


def test_partial_overlap_sends_only_new_lines(tmp_path):
    lines = [f"line{i}" for i in range(1, 81)]
    p = tmp_path / "g.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rread(p, offset=1, limit=50)  # delivers 1..50
    second = rread(p, offset=30, limit=50)  # asks 30..79

    out = second["output"]
    assert "51: line51" in out          # new content arrives
    assert "79: line79" in out
    assert "\n31: line31" not in out    # old content trimmed...
    assert "\n45: line45" not in out
    assert "shown earlier" in out       # ...and marked instead
    assert "Deduped:" in second["output"]


def test_edited_file_delivers_fresh_content(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("v1 line\nsecond\n", encoding="utf-8")
    rread(p)

    # rewrite with clearly different size/content (size change defeats any
    # mtime-granularity flakiness on coarse filesystems)
    time.sleep(0.01)
    p.write_text("v1 line\nsecond\nthird line added\nand more\n", encoding="utf-8")

    out = rread(p)["output"]
    assert "third line added" in out
    assert out.count("v1 line") == 1  # delivered once, not stubbed away twice


def test_compaction_reset_allows_fresh_delivery(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("data one\ndata two\n", encoding="utf-8")
    rread(p)
    stub = rread(p)
    assert stub["metadata"].get("dedup_stub") is True

    cl.reset_for_compaction()  # engine does this after summarizing history

    fresh = rread(p)
    assert fresh["metadata"].get("dedup_stub") is None
    assert "data one" in fresh["output"]


def test_different_windows_are_independent(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("\n".join(f"L{i}" for i in range(1, 21)) + "\n", encoding="utf-8")
    rread(p, offset=1, limit=10)   # 1..10
    other = rread(p, offset=11, limit=10)  # 11..20 -> fully new
    assert other["metadata"].get("dedup_stub") is None
    assert "11: L11" in other["output"] and "20: L20" in other["output"]


# ---------------------------------------------------------------------------
# cross-tool dedup: grep's printed lines count as delivered
# ---------------------------------------------------------------------------


def test_grep_marks_lines_so_later_read_trims_them(tmp_path):
    p = tmp_path / "s.py"
    body = ["import os", "pass", "TOKEN = 'foo'", "pass", "other = 'foo'", "done"]
    p.write_text("\n".join(body) + "\n", encoding="utf-8")

    g = grep_tool().run({"pattern": "foo", "path": str(p)})
    assert "Line 3" in g["output"] and "Line 5" in g["output"]

    rd = rread(p)["output"]
    # matched lines were already shown by grep -> trimmed from the read...
    assert "\n3: TOKEN" not in rd
    assert "\n5: other" not in rd
    # ...non-matching lines still arrive normally
    assert "1: import os" in rd
    assert "6: done" in rd
    assert "shown earlier" in rd


# ---------------------------------------------------------------------------
# legacy compatibility
# ---------------------------------------------------------------------------


def test_fresh_small_file_keeps_legacy_format(tmp_path):
    p = tmp_path / "leg.txt"
    p.write_text("hello\nworld\n", encoding="utf-8")
    res = rread(p)
    out = res["output"]
    assert out.startswith(f"<{p}>…</{p}>\n<type>file</type>\n<content>\n")
    assert "1: hello\n2: world" in out
    assert "(End of file - total 2 lines)" in out
    assert res["metadata"] == {"loaded": [str(p)]}


def test_binary_and_directory_untouched_by_modes(tmp_path):
    b = tmp_path / "bin.dat"
    b.write_bytes(bytes([0, 1, 2, 0xFF, 0xFE]) * 20 + b"tail")
    assert "binary file" in rread(b, mode="outline")["output"]

    d = tmp_path / "dir"
    d.mkdir()
    (d / "inner.txt").write_text("x", encoding="utf-8")
    out = rread(d, mode="outline")["output"]
    assert "<type>directory</type>" in out and "inner.txt" in out


def test_unseen_ranges_math():
    import os

    f = "/virtual/x.txt"
    cl.mark_delivered(f, 111, 7, 1, 10)
    assert cl.unseen_ranges(f, 111, 7, 1, 10) == []
    assert cl.unseen_ranges(f, 111, 7, 5, 15) == [(11, 15)]
    assert cl.unseen_ranges(f, 111, 7, 12, 14) == [(12, 14)]
    assert cl.unseen_ranges(f, 222, 7, 1, 5) == [(1, 5)]  # version changed
    assert os.path.exists("/virtual") is False  # never touched disk
