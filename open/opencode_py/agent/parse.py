"""Tool-call parsing: convert model output into tool calls, and results back to
messages. Handles OpenAI (tool_calls with function.arguments JSON) and Anthropic
(tool_use blocks) shapes.
"""

from __future__ import annotations

import json
from typing import Any


class ToolCallParseError(Exception):
    pass


def parse_openai_tool_calls(assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from an OpenAI-style assistant message."""
    calls = []
    for tc in assistant_message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append(
            {
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
            }
        )
    return calls


def parse_anthropic_tool_blocks(assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool_use blocks from an Anthropic-style assistant message."""
    calls = []
    content = assistant_message.get("content")
    if not isinstance(content, list):
        return calls
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            calls.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                }
            )
    return calls


def parse_arguments(arguments: str | dict) -> dict:
    """Parse tool-call arguments (JSON string or already-dict)."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments)
        if isinstance(parsed, dict):
            return parsed
        return {"arguments": parsed}
    except json.JSONDecodeError:
        # some models emit Python-ish or partial JSON; best effort
        return {"arguments": arguments}


def assistant_message_from_calls(calls: list[dict[str, Any]], reasoning: str = "", content: str = "") -> dict[str, Any]:
    """Build the assistant message that declares the tool calls (for history).

    Carries the model's own text and reasoning alongside the tool_calls so the
    follow-up request is a faithful replay of what the model actually said —
    reasoning models lose thread if their previous message is stored empty.
    `reasoning_content` is ALWAYS present ("") so thinking-mode gateways that
    demand it on every reassembled assistant message don't reject history.
    """
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": content or "",
        "reasoning_content": reasoning or "",
        "tool_calls": [
            {
                "id": c.get("id") or f"call_{i}",
                "type": "function",
                "function": {"name": c["name"], "arguments": c.get("arguments", "{}")},
            }
            for i, c in enumerate(calls)
        ],
    }
    return msg


def tool_result_message(call_id: str, name: str, output: str, error: bool = False) -> dict[str, Any]:
    """Build the OpenAI-style tool result message."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": output if not error else f"[tool error] {output}",
    }


def anthropic_assistant_from_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an Anthropic-style assistant message with tool_use blocks."""
    content = []
    for c in calls:
        try:
            parsed = json.loads(c.get("arguments", "{}"))
            if not isinstance(parsed, dict):
                parsed = {}
        except json.JSONDecodeError:
            parsed = {}
        content.append(
            {
                "type": "tool_use",
                "id": c.get("id") or f"toolu_{len(content)}",
                "name": c["name"],
                "input": parsed,
            }
        )
    return {"role": "assistant", "content": content}


def anthropic_tool_result_message(call_id: str, output: str, error: bool = False) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": output if not error else f"[tool error] {output}",
                "is_error": error,
            }
        ],
    }


def convert_openai_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style history to Anthropic messages (used if needed)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            out.append({"role": "user", "content": msg.get("content", "")})
            out.append({"role": "assistant", "content": "Understood."})
            continue
        if role == "tool":
            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if msg.get("is_error"):
                tool_result["is_error"] = True
            out.append(
                {
                    "role": "user",
                    "content": [tool_result],
                }
            )
            continue
        if role == "assistant" and msg.get("tool_calls"):
            out.append(anthropic_assistant_from_calls(parse_openai_tool_calls(msg)))
            continue
        if role == "assistant":
            content = msg.get("content")
            if not content:
                continue
            out.append({"role": "assistant", "content": content})
            continue
        if role == "user":
            out.append({"role": "user", "content": msg.get("content", "")})
            continue
    return out
