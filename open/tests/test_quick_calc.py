"""Tests for quick_calc: the six actions plus sandbox escape attempts."""

from __future__ import annotations

import base64
import hashlib
import json
import time

import pytest

from opencode_py.tools.quick_calc import tool


@pytest.fixture
def calc():
    return tool()


# ------------------------------------------------------------------- math


def test_math_arithmetic(calc):
    res = calc.run({"action": "math", "expression": "(450*7)/1024"})
    assert res.get("error") is not True
    assert res["output"].startswith("(450*7)/1024 = ")
    assert "3.07" in res["output"]


def test_math_functions_and_constants(calc):
    assert "1.41" in calc.run({"action": "math", "expression": "sqrt(2)"})["output"]
    r = calc.run({"action": "math", "expression": "round(pi*100)/100"})
    assert "3.14" in r["output"]
    r = calc.run({"action": "math", "expression": "min(3, 1, 2)"})
    assert "= 1" in r["output"]


def test_math_float_output_cleaned(calc):
    # 2.0 prints as 2, not 2.0
    out = calc.run({"action": "math", "expression": "4/2"})["output"]
    assert out.endswith("= 2")


def test_math_division_by_zero(calc):
    res = calc.run({"action": "math", "expression": "5/0"})
    assert res.get("error") is True
    assert "zero" in res["output"].lower()


def test_sandbox_blocks_every_escape(calc):
    escapes = [
        "__import__('os').system('id')",
        "open('/etc/passwd')",
        "().__class__",
        "lambda: 1",
        "[].append(1)",
        "'a' + 'b'",
        "(1).__add__(1)",
        "exec('1')",
        "9**9**9",           # memory bomb
        "x=1",               # statements
    ]
    for expr in escapes:
        res = calc.run({"action": "math", "expression": expr})
        assert res.get("error") is True, f"NOT BLOCKED: {expr}"
        low = res["output"].lower()
        assert (
            "not allowed" in low
            or "error" in low
            or "too large" in low
            or "valid arithmetic" in low
        )


# ------------------------------------------------------------- json / base64 / hash


def test_json_valid_and_invalid(calc):
    ok = calc.run({"action": "json", "text": '{"a": [1,2], "b": null}'})
    assert ok.get("error") is not True
    assert "object with 2 keys" in ok["output"]
    bad = calc.run({"action": "json", "text": "{a: 1}"})
    assert bad.get("error") is True
    assert "column" in bad["output"]


def test_base64_roundtrip(calc):
    enc = calc.run({"action": "base64", "mode": "encode", "text": "héllo"})
    token = enc["output"].strip()
    assert base64.b64encode("héllo".encode()).decode() in token
    dec = calc.run({"action": "base64", "mode": "decode", "text": token})
    assert dec["output"].strip() == "héllo"
    bad = calc.run({"action": "base64", "mode": "decode", "text": "!!!not-b64!!!"})
    assert bad.get("error") is True


def test_hash_algorithms(calc):
    text = "hello world"
    assert hashlib.sha256(text.encode()).hexdigest() == calc.run(
        {"action": "hash", "algorithm": "sha256", "text": text}
    )["output"].strip()
    assert hashlib.md5(text.encode()).hexdigest() == calc.run(
        {"action": "hash", "algorithm": "md5", "text": text}
    )["output"].strip()
    assert calc.run({"action": "hash", "algorithm": "crc32", "text": "x"}).get("error") is True


# ------------------------------------------------------------------- regex


def test_regex_finds_matches_with_positions(calc):
    res = calc.run({"action": "regex", "pattern": r"\d+", "sample": "a1 b22 c333"})
    assert res["metadata"]["matches"] == 3
    assert "at 1-2" in res["output"]
    assert "333" in res["output"]


def test_regex_groups_and_case_flag(calc):
    res = calc.run({
        "action": "regex", "pattern": r"(\w+)@(\w+)",
        "sample": "mail ME@HOME now", "ignore_case": True,
    })
    assert res["metadata"]["matches"] == 1
    assert "groups=['ME', 'HOME']" in res["output"]


def test_regex_no_match_and_bad_pattern(calc):
    none = calc.run({"action": "regex", "pattern": "zzz", "sample": "abc"})
    assert none["metadata"]["matches"] == 0
    bad = calc.run({"action": "regex", "pattern": "([unclosed", "sample": "x"})
    assert bad.get("error") is True
    assert "Invalid regex" in bad["output"]


# -------------------------------------------------------------------- time


def test_time_now_and_roundtrip(calc):
    now = calc.run({"action": "time", "mode": "now"})
    unix = json.loads(json.dumps(now))["metadata"]["unix"]
    back = calc.run({"action": "time", "mode": "from_unix", "ts": unix})
    assert back.get("error") is not True
    to_unix = calc.run({"action": "time", "mode": "to_unix", "date": "2026-08-22 12:00"})
    assert to_unix["metadata"]["unix"] == int(
        time.mktime(time.strptime("2026-08-22 12:00", "%Y-%m-%d %H:%M"))
    )


def test_time_bad_date_lists_formats(calc):
    res = calc.run({"action": "time", "mode": "to_unix", "date": "tomorrow"})
    assert res.get("error") is True
    assert "2026-08-22" in res["output"]  # shows an example format


def test_unknown_action(calc):
    assert calc.run({"action": "fly"}).get("error") is True