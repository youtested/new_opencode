"""The pure-python grep/glob fallbacks must respect the project's
.gitignore — otherwise a bare-pip install (no rg binary) scans venv/,
node_modules/ and dist/ for minutes on armv7."""

from pathlib import Path

from opencode_py.tools.glob import _glob
from opencode_py.tools.grep import _grep_py
from opencode_py.util.gitignore import load as load_gitignore


def make_tree(tmp_path: Path) -> Path:
    (tmp_path / ".gitignore").write_text("venv/\n*.log\n!keep.log\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("NEEDLE = 1\n")
    venv = tmp_path / "venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "junk.py").write_text("NEEDLE = 2\n")
    (tmp_path / "noise.log").write_text("NEEDLE = 3\n")
    (tmp_path / "keep.log").write_text("NEEDLE = 4\n")
    return tmp_path


def test_glob_skips_ignored_dirs_and_files(tmp_path):
    root = make_tree(tmp_path)
    out = _glob("**/*.py", str(root))
    paths = out["output"].splitlines()
    assert any(p.endswith("src/app.py") for p in paths)
    assert not any("/venv/" in p for p in paths)


def test_grep_fallback_skips_ignored_paths(tmp_path):
    root = make_tree(tmp_path)
    res = _grep_py("NEEDLE", root)
    files = {r[0] for r in res}
    assert any(r.endswith("src/app.py") for r in files)
    assert not any("/venv/" in f for f in files)
    # *.log ignored, but the negated !keep.log is searchable again? git says:
    # cannot re-include inside an ignored dir; here keep.log is at ROOT and
    # our last-match-wins model re-includes it.
    assert all(not f.endswith("noise.log") for f in files)


def test_no_gitfile_keeps_old_behavior(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    gi = load_gitignore(tmp_path)
    assert gi is None


def test_deeper_gitignore_overrides_shallower(tmp_path):
    (tmp_path / ".gitignore").write_text("build/\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".gitignore").write_text("!build/\n")
    (sub / "build").mkdir(parents=True)
    (sub / "build" / "gen.py").write_text("ok = True\n")
    gi = load_gitignore(sub)
    target = sub / "build" / "gen.py"
    assert not gi.match(target.resolve())
