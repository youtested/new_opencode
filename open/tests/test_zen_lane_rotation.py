"""Tests for Zen lane rotation: when the gateway pins a session to a dead
upstream lane (streams end finish_reason='network_error' with zero output),
retrying with the SAME session id re-hits the same corpse. The fix rotates
the session identity so the next attempt draws a fresh lane."""

import threading

import pytest

from opencode_py.providers.base import ProviderError, ProviderEvent, RateLimitError
from opencode_py.providers.rotation import Rotation
from opencode_py.providers.zen import ZenProvider, _LANE_EPOCH


def zen(session_id):
    return ZenProvider(api_key=None, model="x-preview-f-free", session_id=session_id)


def test_rotate_session_changes_wire_headers():
    p = zen("sess-abc")
    assert p.extra_headers["x-opencode-session"] == "sess-abc"
    # request id is fresh per call, never a copy of the session
    assert p.extra_headers["x-opencode-request"]
    assert p.extra_headers["x-opencode-request"] != "sess-abc"

    p.rotate_session()

    assert p.extra_headers["x-opencode-session"] == "sess-abc::r1"
    # rotate touches ONLY the session (sticky lane); request stays fresh-per-call
    h1 = p._headers()["x-opencode-request"]
    h2 = p._headers()["x-opencode-request"]
    assert h1 and h2 and h1 != h2


def test_rotation_persists_across_provider_rebuilds():
    """rotation.build_provider constructs a NEW ZenProvider per attempt with
    the same engine sid — the rotated epoch must survive that."""
    zen("sess-persist")  # register base
    p1 = zen("sess-persist")
    p1.rotate_session()
    p2 = zen("sess-persist")  # next attempt's fresh instance
    assert p2.extra_headers["x-opencode-session"] == "sess-persist::r1"
    p2.rotate_session()
    p3 = zen("sess-persist")
    assert p3.extra_headers["x-opencode-session"] == "sess-persist::r2"


def test_sessions_rotate_independently():
    a = zen("sess-a")
    b = zen("sess-b")
    a.rotate_session()
    assert "::r1" in a.extra_headers["x-opencode-session"]
    assert b.extra_headers["x-opencode-session"] == "sess-b"


def test_no_session_id_still_rotates_request_header():
    p = ZenProvider(api_key=None, model="m", session_id=None)
    before_session = p.extra_headers.get("x-opencode-session")
    # no session yet: absent or cli epoch form both fine
    assert before_session is None or before_session.startswith("cli::r")
    p.rotate_session()
    after_session = p.extra_headers.get("x-opencode-session")
    p.rotate_session()
    after_session = p.extra_headers.get("x-opencode-session")
    assert after_session and after_session.startswith("cli::r")
    assert after_session != before_session
    # request id is always fresh per call
    assert p._headers()["x-opencode-request"]
    assert p._headers()["x-opencode-request"] != p._headers()["x-opencode-request"]


# ---------------------------------------------------------------------------
# rotation.stream hooks
# ---------------------------------------------------------------------------


class FakeProv:
    def __init__(self, behavior, log, name="fake"):
        self.behavior = behavior
        self.log = log
        self.name = name
        self.id = name
        self.is_free = True
        self.rotations = 0

    def abort_stream(self):
        pass

    def rotate_session(self):
        self.rotations += 1
        self.log.append(("rotate", self.name))

    def stream_chat(self, messages, tools, on_event, **kw):
        self.log.append(("stream", self.name))
        b = self.behavior
        if b == "empty":
            return  # no events at all -> empty response path
        if b == "error":
            on_event(ProviderEvent(kind="error", error="network_error"))
            return
        if b == "rate":
            raise RateLimitError("limited")
        on_event(ProviderEvent(kind="text_delta", text="hi"))
        on_event(ProviderEvent(kind="done", finish_reason="stop"))


def _rot(make):
    return Rotation(
        lanes=[{"provider": "prov", "model": "m"}], make_provider=lambda pid, m: make
    )


def test_rotate_called_on_empty_response_then_raises():
    log = []
    prov = FakeProv("empty", log)
    rot = _rot(prov)
    with pytest.raises(ProviderError) as ei:
        rot.stream([], [], lambda e: None)
    assert prov.rotations == 1
    assert "empty response" in str(ei.value)


def test_rotate_called_on_inband_error_before_text():
    log = []
    prov = FakeProv("error", log)
    rot = _rot(prov)
    with pytest.raises(ProviderError):
        rot.stream([], [], lambda e: None)
    assert prov.rotations == 1


def test_no_rotate_on_success_or_rate_limit():
    log = []
    ok_prov = FakeProv("ok", log)
    rot = _rot(ok_prov)
    rot.stream([], [], lambda e: None)
    assert ok_prov.rotations == 0

    rate_prov = FakeProv("rate", log)
    rot2 = _rot(rate_prov)
    with pytest.raises(RateLimitError):
        rot2.stream([], [], lambda e: None)
    # rate limit is quota, not lane death -> identity must stay stable
    assert rate_prov.rotations == 0


def test_thread_safety_of_epoch_map():
    p = zen("sess-threaded")
    def bump():
        for _ in range(50):
            p.rotate_session()
    ts = [threading.Thread(target=bump) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    key = "sess-threaded"
    assert _LANE_EPOCH[key] == 200
