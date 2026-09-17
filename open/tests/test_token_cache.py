"""Token-count cache: same numbers, no recount (headless-safe)."""

from opencode_py.agent import messages as M
from opencode_py.util.truncate import cached_tokens, clear_token_cache, estimate_tokens


def test_cache_same_numbers_as_recount():
    msgs = [{"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
            {"role": "tool", "tool_call_id": "c1", "name": "read",
             "content": "X" * 8000}]
    clear_token_cache()
    cold = M.trim_history(list(msgs), 100000)
    warm = M.trim_history(list(msgs), 100000)
    assert [m.get("content") for m in cold] == [m.get("content") for m in warm]
    out = M.trim_history(list(msgs), 5)
    assert len(out) == 2  # budget still enforced


def test_changed_message_recounts():
    clear_token_cache()
    m = {"role": "user", "content": "short"}
    assert cached_tokens(m, lambda: estimate_tokens("short")) == 1
    m["content"] = "a much longer message here yes indeed"
    # length changed -> recompute, new value differs
    assert cached_tokens(m, lambda: estimate_tokens(m["content"])) == \
        estimate_tokens("a much longer message here yes indeed")


def test_new_messages_counted_once():
    clear_token_cache()
    msgs = [{"role": "user", "content": f"q{i}"} for i in range(50)]
    M.trim_history(msgs, 100000)
    from opencode_py.util.truncate import _TOKEN_CACHE

    n_cached = len(_TOKEN_CACHE)
    assert n_cached >= 50
    msgs2 = msgs + [{"role": "user", "content": "brand new"}]
    M.trim_history(msgs2, 100000)
    assert len(_TOKEN_CACHE) >= n_cached + 1
