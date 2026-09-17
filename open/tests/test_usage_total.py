"""usage_total: official-style context count (total, else parts + cache)."""

from opencode_py.providers.base import Usage, usage_total, usage_total_dict


def test_prefers_provider_total():
    u = Usage(input_tokens=100, output_tokens=50, total_tokens=1000, raw={})
    assert usage_total(u) == 1000


def test_falls_back_to_input_plus_output():
    u = Usage(input_tokens=100, output_tokens=50, raw={})
    assert usage_total(u) == 150


def test_includes_anthropic_cache_tokens():
    u = Usage(
        input_tokens=100,
        output_tokens=50,
        raw={"cache_read_input_tokens": 1000, "cache_creation_input_tokens": 200},
    )
    assert usage_total(u) == 1350


def test_dict_helper():
    assert usage_total_dict({"input_tokens": 10, "output_tokens": 5}) == 15
    assert usage_total_dict({"total_tokens": 99}) == 99
    assert usage_total_dict({}) == 0
