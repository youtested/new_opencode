"""Low-level network helpers for interrupting stubborn streamed reads."""

from __future__ import annotations

import socket
import threading
from typing import Any

# Traversal into httpx/httpcore internals is brittle across versions; every
# step is guarded so a layout change just degrades to the old close() behaviour
# instead of raising. The socket is what recv() actually blocks on, and
# closing it from another thread does NOT wake that recv on Linux — only an
# explicit shutdown() does. Reaching the socket lets the interrupt path
# genuinely unblock a stream that's idle mid-"thinking".
_SOCKET_ATTR_CHAIN = (
    "stream._stream._httpcore_stream._stream._connection._network_stream._sock",
)


def socket_of(resp) -> socket.socket | None:
    """Best-effort extraction of the live socket for an in-flight httpx response."""
    target = resp
    for attr in _SOCKET_ATTR_CHAIN[0].split("."):
        try:
            target = getattr(target, attr)
        except Exception:
            return None
    if isinstance(target, socket.socket):
        return target
    return None


def force_close_response(resp) -> None:
    """Close an httpx response AND unblock a reader stuck in ``iter_bytes``.

    ``Response.close()`` alone leaves a blocked recv hanging until the read
    timeout on this platform (the close removes the fd from the table but the
    in-flight syscall keeps its reference). ``socket.shutdown(SHUT_RDWR)``
    resets the connection, which wakes the reader with an EOF/connection-reset
    that the provider's error handling turns into ``StreamInterrupted`` when an
    interrupt is pending — so ESC aborts instantly even mid-thought.

    The shutdown must run BEFORE ``Response.close()``: closing first makes the
    fd invalid so the later ``shutdown()`` is a no-op and the recv is never
    woken.
    """
    if resp is None:
        return
    sock = socket_of(resp)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    try:
        resp.close()
    except Exception:
        pass
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass


_CLIENTS_LOCK = threading.Lock()
_CLIENTS: dict[tuple, Any] = {}


def shared_client(timeout: Any) -> Any:
    """Process-wide pooled httpx.Client per timeout shape (connection reuse).

    Every provider stream used to mint its own ``httpx.Client`` — N parallel
    agents meant N TLS handshakes competing for phone radio. httpx Clients
    are thread-safe; one per timeout shape is shared forever (sockets pool
    inside, default max 100 connections). ``force_close_response`` only ever
    closes the *response*, never the client, so aborts stay safe. Never
    raises: falls back to a fresh client when anything is unexpected.

    The cache key is the (Client class object, timeout shape) pair — keyed
    by object identity, never id(), so garbage-collected test doubles can
    never collide with a later mock (a stale pooled client used to leak
    across tests when CPython reused the id).
    """
    import httpx

    try:
        shape = (float(timeout.connect), float(timeout.read),
                 float(timeout.write), float(timeout.pool))
    except Exception:
        shape = (10.0, 300.0, 30.0, 10.0)
    cls = httpx.Client
    with _CLIENTS_LOCK:
        # Key on the Client CLASS OBJECT itself (identity), never id(): a
        # garbage-collected mock's id can be reused by the next mock, which
        # made pooled lookups return a stale test double from an earlier test.
        for (cached_cls, cached_shape), client in list(_CLIENTS.items()):
            if cached_cls is cls and cached_shape == shape:
                return client
        try:
            import httpx

            client = httpx.Client(timeout=timeout, follow_redirects=True)
        except Exception:
            import httpx

            client = httpx.Client(follow_redirects=True)
        _CLIENTS[(cls, shape)] = client
        return client