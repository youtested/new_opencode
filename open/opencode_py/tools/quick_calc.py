"""quick_calc tool: instant small computations without spawning a shell.

RULES (full guidance for maintainers; the sent schema is short):
- math: strict AST whitelist (numbers, + - * / // % **, comparisons,
  sqrt/floor/ceil/abs/min/max/round/sum/len). No names/imports/files.
- json/base64/hash/regex/time degrade honestly (exact errors).
- Use INSTEAD OF bash whenever the job fits (~100x cheaper on phones).
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math as _math
import re
import time as _time
from typing import Any

from .registry import Tool, schema_with

_MAX_INPUT = 20_000          # chars accepted for text-ish payloads
_MAX_MATH_LEN = 500
_MAX_MATH_POW = 10_000       # exponent cap so 9**9**9 can't eat the phone
_MAX_OUTPUT = 8 * 1024
_MAX_REGEX_MATCHES = 50

# --- math sandbox -----------------------------------------------------------

_SAFE_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: _math.pow,
}

_SAFE_UNARY = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

_SAFE_NAMES: dict[str, Any] = {
    "pi": _math.pi,
    "e": _math.e,
    "tau": _math.tau,
    "inf": _math.inf,
    "true": True,
    "false": False,
}

_SAFE_CALLS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "len": len,
    "sum": sum,
    "sqrt": _math.sqrt,
    "floor": _math.floor,
    "ceil": _math.ceil,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"only numbers allowed, got {type(node.value).__name__}")
    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key in _SAFE_NAMES:
            return _SAFE_NAMES[key]
        # allow calling whitelist functions written bare too (sqrt(2) form)
        raise ValueError(f"name {node.id!r} not allowed")
    if isinstance(node, ast.BinOp):
        op = _SAFE_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError("operator not allowed")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_MATH_POW or abs(left) > 1e154:
                raise ValueError("exponent too large")
            return left**right
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        fn = _SAFE_UNARY.get(type(node.op))
        if fn is None:
            raise ValueError("unary operator not allowed")
        return fn(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("calls must be plain function names")
        fname = node.func.id.lower()
        fn = _SAFE_CALLS.get(fname)
        if fn is None:
            raise ValueError(f"function {node.func.id!r} not allowed")
        args = [_eval_node(a) for a in node.args]
        if node.keywords:
            raise ValueError("keyword arguments not allowed")
        return fn(*args)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_node(e) for e in node.elts]
    if isinstance(node, ast.Compare):  # single comparison chains
        left = _eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator)
            if isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            else:
                raise ValueError("comparison operator not allowed")
            if not ok:
                return False
            left = right
        return True
    raise ValueError(f"{type(node).__name__} not allowed")


def _fmt_num(value: Any) -> str:
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        inner = ", ".join(_fmt_num(v) for v in value)
        return f"({inner})" if value else "()"
    return str(value)


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _action_math(expression: str) -> dict[str, Any]:
    expression = (expression or "").strip()
    if not expression:
        return {"output": "Empty expression.", "error": True}
    if len(expression) > _MAX_MATH_LEN:
        return {"output": f"Expression too long ({len(expression)} > {_MAX_MATH_LEN}).",
                "error": True}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return {"output": f"Not valid arithmetic: line {e.lineno}: {e.msg}", "error": True}
    try:
        result = _eval_node(tree)
    except ZeroDivisionError:
        return {"output": "Division by zero.", "error": True}
    except ValueError as e:
        return {
            "output": f"Not allowed: {e}. Only numbers, + - * / // % ** , "
            "comparisons, parentheses, and "
            + ", ".join(sorted(_SAFE_CALLS)) + " are permitted.",
            "error": True,
        }
    except (OverflowError, TypeError) as e:
        return {"output": f"Math error: {e}", "error": True}
    out = _fmt_num(result)
    return {
        "output": f"{expression} = {out}",
        "metadata": {"expression": expression, "result": out},
    }


def _action_json(text: str) -> dict[str, Any]:
    text = text or ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "output": f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}.",
            "error": True,
        }
    pretty = json.dumps(data, indent=2, ensure_ascii=False)[:_MAX_OUTPUT]
    def _shape(d: Any) -> str:
        if isinstance(d, dict):
            return f"object with {len(d)} keys"
        if isinstance(d, list):
            return f"array with {len(d)} items"
        return type(d).__name__
    return {
        "output": f"Valid JSON — top level: {_shape(data)}\n\n{pretty}",
        "metadata": {"valid": True, "shape": _shape(data)},
    }


def _action_base64(mode: str, text: str) -> dict[str, Any]:
    text = text or ""
    if not text:
        return {"output": "Nothing to convert.", "error": True}
    try:
        if mode == "decode":
            raw = base64.b64decode(text.strip(), validate=True)
            out = raw.decode("utf-8", errors="replace")[:_MAX_OUTPUT]
            return {"output": out, "metadata": {"bytes": len(raw)}}
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return {"output": encoded[:_MAX_OUTPUT], "metadata": {"chars": len(encoded)}}
    except Exception as e:
        return {"output": f"base64 {mode} failed: {e}", "error": True}


def _action_hash(algorithm: str, text: str) -> dict[str, Any]:
    algo = (algorithm or "sha256").lower()
    if algo not in ("md5", "sha1", "sha256"):
        return {"output": f"Unknown algorithm {algo!r} (md5/sha1/sha256).", "error": True}
    digest = hashlib.new(algo, (text or "").encode("utf-8")).hexdigest()
    return {"output": digest, "metadata": {"algorithm": algo}}


_MAX_REGEX_TIME = 1.0  # seconds per regex scan; evil patterns abort, never hang


def _looks_evil(pattern: str) -> bool:
    # Classic ReDoS shape: a quantified group containing a quantifier, e.g.
    # (a+)+, (a*)+, (a|b)+. Such patterns blow up exponentially on a
    # near-miss; reject them outright with guidance instead of hanging.
    depth = 0
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\":
            i += 2
            continue
        if c == "(":
            depth += 1
            i += 1
            continue
        if c == ")":
            inner_start = i
            i += 1
            if i < n and pattern[i] in "+*":
                # quantified group: look inside for any quantifier
                j = inner_start - 1
                inner_depth = 1
                while j >= 0 and inner_depth > 0:
                    cc = pattern[j]
                    if cc == ")":
                        inner_depth += 1
                    elif cc == "(":
                        inner_depth -= 1
                    j -= 1
                inner = pattern[j + 2 : inner_start]
                k = 0
                while k < len(inner):
                    if inner[k] == "\\":
                        k += 2
                        continue
                    if inner[k] in "+*":
                        return True
                    if inner[k] == "{" and "}" in inner[k:]:
                        return True
                    k += 1
            elif i < n and pattern[i] == "{" and "}" in pattern[i:]:
                j = inner_start - 1
                inner_depth = 1
                while j >= 0 and inner_depth > 0:
                    cc = pattern[j]
                    if cc == ")":
                        inner_depth += 1
                    elif cc == "(":
                        inner_depth -= 1
                    j -= 1
                inner = pattern[j + 2 : inner_start]
                if any(cc in "+*{" for cc in inner):
                    return True
            depth = max(0, depth - 1)
            continue
        i += 1
    return False


def _action_regex(pattern: str, sample: str, ignore_case: bool) -> dict[str, Any]:
    pattern = pattern or ""
    if len(pattern) > 2000 or len(sample or "") > _MAX_INPUT:
        return {"output": "Pattern or sample too long.", "error": True}
    if _looks_evil(pattern):
        return {
            "output": (
                f"Refused risky regex /{pattern}/: nested quantifiers "
                "(e.g. (a+)+) hang on near-misses. Rewrite without nesting, "
                "e.g. /a+/."
            ),
            "error": True,
        }
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return {"output": f"Invalid regex: {e}", "error": True}
    matches = []
    deadline = _time.monotonic() + _MAX_REGEX_TIME
    try:
        it = rx.finditer(sample or "")
        while True:
            if _time.monotonic() > deadline:
                return {
                    "output": (
                        f"Regex timed out after {_MAX_REGEX_TIME:g}s on "
                        f"{len(sample or '')} chars — pattern too costly."
                    ),
                    "error": True,
                }
            try:
                m = next(it)
            except StopIteration:
                break
            piece = {
                "match": m.group(0),
                "span": [m.start(), m.end()],
            }
            groups = m.groups()
            if groups:
                piece["groups"] = list(groups)
            matches.append(piece)
            if len(matches) >= _MAX_REGEX_MATCHES:
                break
    except re.error as e:
        return {"output": f"Invalid regex: {e}", "error": True}
    if not matches:
        return {"output": f"No match for /{pattern}/.", "metadata": {"matches": 0}}
    lines = [f"/{pattern}/ matched {len(matches)} time(s):"]
    for i, m in enumerate(matches[:20], 1):
        extra = f"  groups={m['groups']}" if "groups" in m else ""
        lines.append(f"  {i}. {m['match']!r} at {m['span'][0]}-{m['span'][1]}{extra}")
    if len(matches) > 20:
        lines.append(f"  … and {len(matches) - 20} more")
    return {"output": "\n".join(lines), "metadata": {"matches": len(matches)}}


def _action_time(mode: str, ts: float, date_text: str) -> dict[str, Any]:
    mode = (mode or "now").lower()
    try:
        import datetime
    except ImportError:  # pragma: no cover - stdlib always present
        return {"output": "datetime unavailable.", "error": True}
    if mode == "now":
        now = _time.time()
        local = datetime.datetime.fromtimestamp(now)
        return {
            "output": (
                f"Now: {local.strftime('%Y-%m-%d %H:%M:%S')} local | "
                f"unix {int(now)}"
            ),
            "metadata": {"unix": int(now)},
        }
    if mode == "from_unix":
        dt = datetime.datetime.fromtimestamp(float(ts))
        return {
            "output": f"{int(ts)} -> {dt.strftime('%Y-%m-%d %H:%M:%S')} local",
            "metadata": {"iso": dt.isoformat()},
        }
    # to_unix: accept a few common layouts
    candidates = (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M",
        "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%S",
    )
    text = (date_text or "").strip()
    for fmt in candidates:
        try:
            dt = datetime.datetime.strptime(text, fmt)
            return {
                "output": f"{text!r} -> unix {int(dt.timestamp())}",
                "metadata": {"unix": int(dt.timestamp())},
            }
        except ValueError:
            continue
    return {
        "output": f"Could not parse date {text!r}. Try formats like 2026-08-22, "
        "'2026-08-22 14:30' or '2026-08-22 14:30:00'.",
        "error": True,
    }


_ACTIONS = ("math", "json", "base64", "hash", "regex", "time")


def tool() -> Tool:
    description = """Desk calculator, no shell spawn: math, json, base64, hash, regex, time. Use instead of bash (~100x cheaper)."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "math").strip().lower()
        if action not in _ACTIONS:
            return {
                "output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True,
            }
        if action == "math":
            return _action_math(str(input.get("expression") or ""))
        if action == "json":
            return _action_json(str(input.get("text") or ""))
        if action == "base64":
            mode = str(input.get("mode") or "encode").lower()
            return _action_base64(
                "decode" if mode.startswith("dec") else "encode",
                str(input.get("text") or ""),
            )
        if action == "hash":
            return _action_hash(str(input.get("algorithm") or "sha256"),
                                str(input.get("text") or ""))
        if action == "regex":
            return _action_regex(
                str(input.get("pattern") or ""),
                str(input.get("sample") or input.get("text") or ""),
                bool(input.get("ignore_case", False)),
            )
        try:
            ts = float(input.get("ts") or 0)
        except (TypeError, ValueError):
            ts = 0.0
        return _action_time(
            str(input.get("mode") or "now"), ts, str(input.get("date") or "")
        )

    return Tool(
        name="quick_calc",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "math (default), json, base64, hash, regex, time",
                    "optional": True,
                },
                "expression": {
                    "type": "string",
                    "description": "Arithmetic, e.g. (450*7)/1024",
                    "optional": True,
                },
                "text": {
                    "type": "string",
                    "description": "Input for json/base64/hash/regex",
                    "optional": True,
                },
                "mode": {
                    "type": "string",
                    "description": "encode|decode; now|from_unix|to_unix",
                    "optional": True,
                },
                "algorithm": {
                    "type": "string",
                    "enum": ["sha256", "sha1", "md5"],
                    "description": "Hash (default sha256)",
                    "optional": True,
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex to test",
                    "optional": True,
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive match",
                    "optional": True,
                },
                "ts": {
                    "type": "number",
                    "description": "Unix timestamp",
                    "optional": True,
                },
                "date": {
                    "type": "string",
                    "description": "Human date",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="quick_calc",
    )