"""opencode_py headless entry point.

Usage:
    opencode_py [dir] [--no-tui] [--auto] [--agent <name>] [--model <id>]
                [--provider <name>] [-m <message>]

Reads a prompt from stdin when no -m/--message is given (like opencode).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

from . import __version__
from .config import load_config
from .globals import Path as GPath
from .globals import freeze as gc_freeze
from .globals import resolve_worktree


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("directory", required=False, default=".")
@click.option("--no-tui", is_flag=True, help="Force headless (no Textual TUI)")
@click.option("--auto", is_flag=True, help="Auto-approve tool calls (like --auto)")
@click.option("--agent", "agent_name", default="build", help="Agent to use (build|plan)")
@click.option("--model", "model_id", default=None, help="Model id (e.g. big-pickle)")
@click.option("--provider", "provider_name", default=None, help="Provider (e.g. opencode)")
@click.option("-m", "--message", default=None, help="Prompt message (otherwise read stdin)")
@click.option("--print-config", is_flag=True, help="Print resolved config and exit")
@click.option("--show-secrets", is_flag=True, help="With --print-config: display API keys (default: redacted)")
@click.option("--check", is_flag=True, help="Ping configured providers and report status")
@click.option("--models", is_flag=True, help="List available Zen models")
@click.option("--version", is_flag=True, help="Print version and exit")
def main(
    directory: str,
    no_tui: bool,
    auto: bool,
    agent_name: str,
    model_id: str | None,
    provider_name: str | None,
    message: str | None,
    print_config: bool,
    show_secrets: bool,
    check: bool,
    models: bool,
    version: bool,
) -> None:
    if version:
        click.echo(f"opencode_py {__version__}")
        return

    try:
        cfg = load_config()
    except Exception as e:
        click.echo(f"error: failed to load config: {e}", err=True)
        sys.exit(1)

    if provider_name:
        cfg.provider = provider_name
    if model_id:
        cfg.model = model_id
        cfg.provider = provider_name or cfg.provider
    if agent_name:
        cfg.default_agent = agent_name

    # Modules imported so far are now frozen out of the cyclic-GC scan path
    # (steady memory on long TUI sessions); heavy late imports stay collectable.
    gc_freeze()

    if print_config:
        click.echo(json.dumps(cfg.as_dict(show_secrets=show_secrets), indent=2))
        return

    worktree = resolve_worktree(Path(directory))

    GPath.init()

    if check:
        from .auth import Auth

        auth = Auth(auth_file=GPath.auth_file())
        _run_check(cfg, auth)
        return

    if models:
        _run_models()
        return

    if not no_tui and sys.stdin.isatty() and message is None:
        # interactive terminal: launch the Textual TUI
        from .tui import run_tui

        run_tui(cfg=cfg, directory=Path(worktree))
        return

    from .auth import Auth
    from .agent.loop import AgentLoop
    from .session import new_session, save_session
    from .commands import CommandContext, build_registry as build_command_registry
    from .tools import build_registry as build_tool_registry

    auth = Auth(auth_file=GPath.auth_file())

    # headless has no mount moment: kick the same background warm the TUI
    # gets, so the first prompt finds hot lane/context caches, not network.
    try:
        from .providers.rotation import warm_startup as _warm
        _warm(cfg, auth)
    except Exception:
        pass

    def on_engine_event(event: dict) -> None:
        if event.get("kind") == "rotated":
            if getattr(engine, "rotation_locked", False):
                # locked: the selected model can't change — no announcement
                return
            reason = event.get("reason") or "provider error"
            click.echo(
                f"[model changed -> {event.get('provider')}/{event.get('model')} ({reason})]",
                err=True,
            )

    engine = AgentLoop(
        cfg=cfg,
        registry=build_tool_registry(cfg),
        directory=Path(worktree),
        auth=auth,
        agent=cfg.default_agent,
        on_event=on_engine_event,
    )

    session = new_session(
        directory=str(worktree),
        provider=cfg.provider,
        model=cfg.model,
        agent=cfg.default_agent,
        title=message[:60] if message else "new session",
    )
    # Persist the session immediately so a sudden phone reboot before the run
    # finishes can't destroy an untouched conversation (durable, fsync'd write).
    # Only when a real prompt was given: running with no prompt (usage) or a
    # piped stdin must not leave an empty "new session" file behind.
    try:
        if message:
            session.messages = [{"role": "user", "content": message}]
            save_session(session)
    except Exception:
        pass

    def reply(text: str) -> None:
        click.echo(text)

    def get_session() -> Any:
        return session

    def exit_app() -> None:
        if getattr(session, "messages", None):
            save_session(session)
        sys.exit(0)

    ctx = CommandContext(
        config=cfg,
        auth=auth,
        session=session,
        engine=engine,
        worktree=worktree,
        reply=reply,
        get_session=get_session,
        exit_app=exit_app,
        registry=build_command_registry(),
    )

    prompt = message
    if prompt is None:
        try:
            prompt = sys.stdin.read(256 * 1024)
            if prompt is not None and len(prompt) >= 256 * 1024:
                try:
                    extra = sys.stdin.read(1)
                except Exception:
                    extra = ""
                if extra:
                    try:
                        click.echo("warning: piped input truncated to 256 KiB", err=True)
                    except Exception:
                        pass
        except Exception:
            prompt = ""
        if prompt is None:
            prompt = ""

    run(cfg, engine, ctx, session, prompt, auto)


def run(
    cfg: Any,
    engine: AgentLoop,
    ctx: CommandContext,
    session: Any,
    prompt: str,
    auto: bool,
) -> None:
    from .commands import build_registry as build_command_registry, handle_command
    from .session import save_session

    if auto:
        engine.permission.mode = "auto"

    if not prompt.strip():
        click.echo("Usage: opencode_py [dir] --message '...'  (or pipe a prompt via stdin)")
        return

    if prompt.lstrip().startswith("/"):
        handled = handle_command(build_command_registry(), ctx, prompt)
        if handled:
            if getattr(session, "messages", None):
                save_session(session)
            return

    session.messages = engine.get_history()
    result = engine.run_turn(prompt)
    session.messages = engine.get_history()
    if getattr(session, "messages", None):
        save_session(session)
    # headless turns count toward the most-used default too
    try:
        if getattr(result, "provider_id", ""):
            from .config import record_model_use
            record_model_use(result.provider_id, getattr(result, "model_id", "") or engine.model_id)
    except Exception:
        pass
    if result.text:
        click.echo(result.text)
        # auto voice (headless): speak the reply without blocking the exit
        # on a missing engine — speak_text never raises.
        try:
            from .tools.speak import auto_enabled, speak_text
            if auto_enabled(cfg) and not result.error:
                speak_text(cfg, result.text)
        except Exception:
            pass
    if result.error:
        click.echo(f"error: {result.error}", err=True)
        sys.exit(1)


def _run_check(cfg: Any, auth: Auth) -> None:
    """--check: ping each configured provider lane and report OK/fail."""
    from .providers import check_provider

    results = check_provider(cfg, auth)
    if not results:
        click.echo("No providers configured.")
        return
    failed = 0
    for pid, info in results.items():
        if info.get("ok"):
            click.echo(f"[OK]   {pid:<12} {info.get('model', '')}")
        else:
            failed += 1
            detail = info.get("error") or info.get("status") or "?"
            click.echo(f"[FAIL] {pid:<12} {info.get('model', '')} - {detail}")
    click.echo(f"\n{len(results) - failed}/{len(results)} providers OK")
    if failed:
        sys.exit(1)


def _run_models() -> None:
    """--models: print the live Zen model list (free first)."""
    from .providers import fetch_zen_models

    models = fetch_zen_models()
    free = [m for m in models if m.get("free")]
    paid = [m for m in models if not m.get("free")]
    click.echo("Free models:")
    for m in free:
        ctx_size = f"{m['context']:,}" if m.get("context") else "?"
        name = m.get("name") or m["id"]
        click.echo(f"  opencode/{m['id']:<26} {name}  ctx={ctx_size}")
    click.echo(f"\nPaid models ({len(paid)}):")
    for m in paid[:20]:
        click.echo(f"  opencode/{m['id']:<26} {m.get('name') or m['id']}")
    if len(paid) > 20:
        click.echo(f"  ... and {len(paid) - 20} more")


if __name__ == "__main__":
    main()
