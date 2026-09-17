import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencode_py import mcp_manager as mm
from opencode_py.commands import CommandContext, build_registry, handle_command
from opencode_py.config import Config


def test_parse_add_python_and_npx():
    assert mm.parse_add_args("add files -- python -m my_server") == ("files", "python", ["-m", "my_server"], False)
    name, cmd, args, glob = mm.parse_add_args("add fs --global -- npx -y @modelcontextprotocol/server-filesystem /tmp")
    assert (name, cmd, glob) == ("fs", "npx", True)
    assert args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


def test_check_command_warns_missing_binary():
    assert mm.check_command("definitely-not-here-xyz", []) != []


def test_save_and_remove_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = mm.save_server("files", "python", ["-m", "my_server"], str(tmp_path), False)
    assert path.exists()
    assert (tmp_path / "opencode.json").exists()
    assert mm.remove_server("files", str(tmp_path), False) is not None
    assert mm.remove_server("files", str(tmp_path), False) is None


def test_mcp_command_registered_and_preview_safe():
    out = []
    cfg = Config()
    cfg.raw = {}
    reg = build_registry()
    assert reg.get("mcp") is not None
    ctx = CommandContext(config=cfg, auth=None, worktree="/tmp", reply=out.append, preview_only=True)
    handle_command(reg, ctx, "/mcp add files -- python -m my_server")
    assert "Would add" in out[-1]
