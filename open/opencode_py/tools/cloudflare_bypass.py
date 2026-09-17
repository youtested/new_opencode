"""Vendored Cloudflare bypass (HTTP-only).

Adapted from https://github.com/youtested/Cloudflare-bypass (MIT).
Changes for this project:
- HTTP-only: keeps the requests / httpx / curl_cffi(if present) / curl / wget
  methods. The Chrome-based methods (selenium, scrapling, browser, browser-wait)
  are intentionally omitted so this stays pure-Python and armv7-safe for Termux.
- No CLI/progress bars / global logging config: this is consumed as a library
  by the webfetch tool.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional
from uuid import uuid4

# Try import requests (hard dep for our use)
try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False

# Optional modules (armv7-safe only). curl_cffi ships binary wheels and is
# optional; httpx is a hard dep of this project already.
try:
    import httpx  # type: ignore
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

try:
    from curl_cffi import requests as curl_requests  # type: ignore
    CURL_CFFI_AVAILABLE = True
except ImportError:
    curl_requests = None  # type: ignore
    CURL_CFFI_AVAILABLE = False

# Pure-Python Cloudflare challenge solver (solves the older JS "cf_chl"
# challenge without a browser). Installs on 32-bit Termux.
try:
    import cloudscraper  # type: ignore
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    cloudscraper = None  # type: ignore
    CLOUDSCRAPER_AVAILABLE = False

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/131.0.0.0",
]

# Proxy-quality bar for the harvester: a candidate must tunnel a real HTTPS
# request to this target and return a clean, unblocked body, within this many
# seconds, before it's admitted to the rotation pool.
# Default is a real bot wall (Cloudflare challenge test site) so only proxies
# that genuinely pass bot protection get in; if that target is down and nothing
# survives, the harvester falls back to the plain reachability target below.
# Override with OPENCODE_PROXY_VALIDATION_URL.
VALIDATION_TARGET = os.environ.get("OPENCODE_PROXY_VALIDATION_URL", "https://nowsecure.nl")
VALIDATION_FALLBACK_TARGET = "https://example.com"
VALIDATION_TIMEOUT = 6

# Disk cache for validated proxy lists. Harvesting + live validation takes ~30s
# on every process start; caching the result and re-using it for a few minutes
# (or until the pool is proven stale) makes startup nearly free and keeps good
# IPs alive across sessions. Override the TTL with OPENCODE_PROXY_CACHE_TTL
# (seconds; 0 disables the cache).
PROXY_CACHE_PATH = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "opencode" / "proxies.json"
PROXY_CACHE_TTL = int(os.environ.get("OPENCODE_PROXY_CACHE_TTL", "600") or "600")


class CookieManager:
    """Save/load cookies per domain (used by the requests method)."""

    def __init__(self, cookie_dir: Optional[str] = None):
        self.cookie_dir = Path(cookie_dir or (Path.home() / ".cache" / "opencode" / "cookies"))
        self.cookie_dir.mkdir(parents=True, exist_ok=True)

    def _get_cookie_path(self, domain: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", domain)
        return self.cookie_dir / f"{safe}.json"

    def load(self, domain: str) -> Optional[Dict]:
        path = self._get_cookie_path(domain)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, domain: str, cookies: Dict) -> None:
        path = self._get_cookie_path(domain)
        try:
            path.write_text(json.dumps(cookies))
        except OSError:  # pragma: no cover
            pass


def _check_block(content: str, status_code: Optional[int] = None) -> bool:
    """Return True if the response looks like a Cloudflare/JS challenge page.

    The keyword list used to include near-universal words ("cloudflare",
    "captcha", "access denied", "blocked"): a perfectly good 200 page that just
    MENTIONS Cloudflare (docs, articles, blocklist tools) was then rejected by
    every cascade method, so webfetch kept hammering a site that had already
    served the real content. Strong markers are challenge-specific; generic
    "access denied"/403 handling only fires on non-200 statuses or tiny pages.
    """
    if content is None:
        content = ""
    lower = content.lower()[:1000]
    strong = [
        "checking your browser",
        "just a moment",
        "attention required",
        "error 1020",
        "turnstile",
        "verifying you are human",
        "cf-challenge",
        "cf-browser-verification",
    ]
    weak = [
        "cloudflare",
        "ray id",
        "captcha",
        "access denied",
        "blocked",
    ]
    if any(s in lower for s in strong):
        return True
    if status_code == 403:
        return True
    if status_code in (401, 429, 503, 504):
        return True
    # only count generic words on a small page (challenge pages are small;
    # a full article that says "blocked" is not a bot wall)
    if len(content) <= 4096:
        return any(w in lower for w in weak)
    return False


class ProxyPool:
    """Rotating list of proxies used to fetch through a fresh IP per attempt.

    Sources (no Tor):
      1. OPENCODE_PROXY_POOL env var — comma/space separated proxy URLs
         (e.g. "http://user:pass@host:port, http://host:port").
      2. Open harvest (opt-in): when OPENCODE_HARVEST_PROXIES=1, free public
         HTTP/SOCKS proxies are pulled from public list APIs and filtered to
         reachable, HTTPS-capable ones, so every blocked fetch retries from a
         different IP.

    Session rotation (for rotating residential gateways): a pool entry that
    contains a ``-sess-<token>``-style userinfo segment is treated as a
    session-based gateway. Each *fetch* (not each attempt) is given a fresh
    random session token by :meth:`next`, so a single
    ``OPENCODE_PROXY_POOL`` entry like
    ``http://user-sess-XXX:pass@gw:port`` yields a different exit IP per fetch.
    The session is pinned for the duration of one fetch's cascade so redirects
    and cookies stay on one IP, then rotates for the next fetch.

    Rotation picks the next proxy each round. NOTE: free proxies are
    third-party relays — fine for reading public pages, not for private data.
    """

    # Markers that flag a proxy URL as session-based (rewritten per fetch).
    SESSION_MARKERS = ("-sess-", "_session_", "-session-")

    def __init__(self, proxies: Optional[list] = None, harvest: Optional[bool] = None):
        self._proxies = list(proxies or [])
        self._index = 0
        self._lock = threading.Lock()
        self._load_env()
        # Self-healing: per-proxy consecutive-failure counters; a proxy that
        # fails `self.max_fails` fetches in a row is demoted to `_dead` and
        # skipped until it either succeeds elsewhere or the pool refills.
        self._fails: Dict[str, int] = {}
        self._dead: set = set()
        self.max_fails = int(os.environ.get("OPENCODE_PROXY_MAX_FAILS", "3"))
        # The active session token for session-based gateways; None when the
        # pool holds no session-style entries (plain proxies rotate as-is).
        self._session_token: Optional[str] = None
        # Per-fetch session token: begin_fetch() sets it on the CALLING THREAD,
        # and next() applies only that thread's token. A pool-global attribute
        # was racy — two concurrent fetches (parallel webfetch_many) calling
        # begin_fetch() would clobber each other's token, so one fetch's request
        # carried another's session (leaking exit IPs across fetches).
        self._thread_local = threading.local()
        if harvest is None:
            harvest = os.environ.get("OPENCODE_HARVEST_PROXIES", "") not in ("", "0", "false")
        if not self._proxies and harvest:
            self._proxies = self._harvest_or_cache()

    def _session_proxy(self, proxy: str) -> Optional[str]:
        """Rewrite a session-based proxy URL to use the pool's current token.

        Returns the token-flavored URL when ``proxy`` is session-based and a
        token is active, else None (the proxy is used as-is).
        """
        if self._session_token is None:
            return None
        for marker in self.SESSION_MARKERS:
            if marker in proxy:
                # replace the token after the marker (any chars up to :@ or end)
                return proxy
        return None

    def mark_success(self, proxy: Optional[str]) -> None:
        """A fetch through ``proxy`` succeeded: reset its failure streak."""
        if not proxy:
            return
        with self._lock:
            self._fails.pop(proxy, None)
            self._dead.discard(proxy)

    def mark_failure(self, proxy: Optional[str]) -> None:
        """A full cascade through ``proxy`` failed: count it, demote on streak.

        Free proxies die constantly; a proxy that fails N fetches in a row is
        almost certainly dead and would otherwise waste seconds on every fetch
        until the process restarts. Demote it so `next()` skips it.
        """
        if not proxy:
            return
        with self._lock:
            count = self._fails.get(proxy, 0) + 1
            self._fails[proxy] = count
            if count >= self.max_fails:
                self._dead.add(proxy)

    def refill(self) -> bool:
        """Refill the pool when pruning has drained it.

        Pulls the cached/refreshed list (disk cache first, live harvest on a
        stale cache) so a self-healed pool doesn't stay empty for the rest of
        the process. Returns True when the pool now has a usable proxy.
        """
        with self._lock:
            if self._proxies and not all(p in self._dead for p in self._proxies):
                return True
        try:
            fresh = self._harvest_or_cache()
        except Exception:
            fresh = []
        if fresh:
            with self._lock:
                self._proxies = list(fresh)
                self._dead.clear()
                self._fails.clear()
                self._index = 0
            return True
        return False

    def _harvest_or_cache(self) -> list:
        """Harvest a fresh proxy list, preferring a recent on-disk copy.

        Harvesting + live HTTPS validation is expensive (~30s); a cached list
        from the last few minutes is re-used as-is, and a fresh harvest is saved
        back to disk so the next process starts fast. Set
        OPENCODE_PROXY_CACHE_TTL=0 to always harvest fresh.
        """
        cached = self._load_cache()
        if cached is not None:
            return cached
        try:
            fresh = self._harvest()
            self._save_cache(fresh)
            return fresh
        except Exception:
            # harvest failed: fall back to a stale cache rather than nothing
            return self._load_cache(ignore_ttl=True) or []

    def _cache_path(self) -> Path:
        return PROXY_CACHE_PATH

    def _load_cache(self, ignore_ttl: bool = False) -> Optional[list]:
        """Return the cached proxy list when it exists and is fresh enough."""
        if PROXY_CACHE_TTL <= 0 and not ignore_ttl:
            return None
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            ts = float(data.get("ts", 0))
            if not ignore_ttl and time.time() - ts > PROXY_CACHE_TTL:
                return None
            proxies = data.get("proxies", [])
            return list(proxies) if isinstance(proxies, list) else None
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _save_cache(self, proxies: list) -> None:
        try:
            self._cache_path().parent.mkdir(parents=True, exist_ok=True)
            self._cache_path().write_text(json.dumps({"ts": time.time(), "proxies": list(proxies)}))
        except OSError:  # pragma: no cover
            pass

    def _load_env(self) -> None:
        raw = os.environ.get("OPENCODE_PROXY_POOL", "")
        for item in re.split(r"[,\s]+", raw.strip()):
            if item:
                self._proxies.append(item)

    @property
    def available(self) -> bool:
        return bool(self._proxies)

    def all_dead(self) -> bool:
        """True when the pool has proxies but every one has been demoted."""
        with self._lock:
            return bool(self._proxies) and all(p in self._dead for p in self._proxies)

    def next(self) -> Optional[str]:
        if not self._proxies:
            return None
        # Locked so concurrent fetches (parallel webfetch_many) rotate through
        # the same pool without two workers landing on the same proxy. Dead
        # (self-healed) proxies are skipped; if everything is dead the whole
        # demotion list is cleared so the pool isn't starved on a bad streak.
        with self._lock:
            for _ in range(len(self._proxies)):
                proxy = self._proxies[self._index % len(self._proxies)]
                self._index += 1
                if proxy not in self._dead:
                    return self._apply_session(proxy)
            self._dead.clear()
            return self._apply_session(self._proxies[self._index % len(self._proxies)])

    def _apply_session(self, proxy: str) -> str:
        """Rewrite a session-based gateway URL to carry the current thread's
        session token.

        ``http://user-sess-XXX:pass@gw:port`` -> ``http://user-sess-<token>:pass@gw:port``.
        When no token is active (no begin_fetch on this thread, or the pool has
        no session-style entries) plain proxies pass through unchanged. The
        token comes from the thread-local set by ``begin_fetch`` so parallel
        fetches never share (and clobber) each other's session.
        """
        token = getattr(self._thread_local, "session_token", None)
        if token is None:
            return proxy
        for marker in self.SESSION_MARKERS:
            if marker in proxy:
                pre, _, post = proxy.partition(marker)
                # token runs from after the marker up to the next ':' (userinfo)
                head, _, tail = post.partition(":")
                return f"{pre}{marker}{token}:{tail}"
        return proxy

    def _detect_session(self) -> bool:
        """True when any pool entry looks like a session-based gateway."""
        return any(
            any(m in p for m in self.SESSION_MARKERS)
            for p in self._proxies
        )

    def begin_fetch(self) -> str:
        """Start a new fetch: rotate to a fresh session token.

        Called once per fetch (not per attempt) so a session-based gateway exits
        from a NEW IP for this fetch. The token is stored on the CALLING thread
        (``_thread_local.session_token``), so parallel fetches each keep their
        own session instead of racing over a pool-global value. For plain
        proxies this is a no-op and returns the pool's stable label for the
        caller.
        """
        with self._lock:
            if self._detect_session():
                token = f"{uuid4().hex[:8]}"
            else:
                token = None
            self._session_token = token
        # deliberately OUTSIDE the lock: thread-local write is per-thread
        self._thread_local.session_token = token
        return ""

    def end_fetch(self) -> None:
        """End the current fetch, releasing its session affinity.

        For plain (non-session) pools this is a no-op. Kept so callers have a
        symmetric begin/end pair even though a fresh token is minted on the
        next :meth:`begin_fetch`.
        """
        return

    def new_identity(self) -> None:
        """Rotate the exit IP for a retry round.

        Session-based gateways already rotate per fetch; within one fetch's
        cascade each :meth:`next` returns the next pool entry, so there is
        nothing extra to do here.
        """

    @classmethod
    def shared(cls) -> "ProxyPool":
        """Return the process-wide pool (rebuilt automatically when the
        OPENCODE_PROXY_POOL / OPENCODE_HARVEST_PROXIES env changes)."""
        return _shared_pool_get()

    def _harvest(self, max_keep: int = 12, max_candidates: int = 60) -> list:
        """Pull free proxies from public list APIs and keep the BEST ones.

        A proxy list entry is worthless until proven: candidates are fetched,
        deduped, then each is validated with a REAL HTTPS request through it
        (not a TCP connect — that keeps tons of dead/non-HTTP proxies) and only
        the fast, unblocked ones are kept, ranked by latency. Proxies are
        probed against a real bot wall; if that wall is unreachable and nothing
        survives, validation falls back to a plain reachability target so a
        transient bot-wall outage can't empty the pool.
        """
        candidates = self._gather_candidates()[:max_candidates]
        good = self._validate_candidates(candidates, max_keep)
        if not good:
            good = self._validate_candidates(candidates, max_keep, fallback=True)
        return good

    def _gather_candidates(self) -> list:
        """Fetch proxy URLs from public list endpoints, deduped in order.

        Several sources are GitHub-hosted proxy lists (the best-maintained free
        lists). Entries are kept only when they match the strict ``host:port``
        form, so a misformatted line is silently ignored.
        """
        import urllib.request

        candidates: list[str] = []
        endpoints = [
            "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=all&ssl=all&anonymity=all",
            "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
            "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
            "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
            "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
            "https://raw.githubusercontent.com/shiftytr/proxy-list/master/proxy.txt",
            "https://proxylist.geonode.com/api/proxy-list?limit=50&page=1&sort_by=lastChecked&sort_type=desc&protocols=http%2Chttps",
        ]
        for ep in endpoints:
            try:
                req = urllib.request.Request(ep, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as r:
                    for line in (r.read().decode("utf-8", "replace") or "").splitlines():
                        line = line.strip()
                        if re.match(r"^[\w.\-]+:\d{2,5}$", line):
                            candidates.append("http://" + line)
                        elif re.match(r"^https?://[\w.\-]+:\d{2,5}$", line):
                            candidates.append(line if line.startswith("http://") else "http://" + line[8:])
            except Exception:
                continue
        return list(dict.fromkeys(candidates))

    def _proxy_check(self, proxy: str, fallback: bool = False):
        """One real HTTPS request through ``proxy``.

        Returns ``(proxy, latency)`` when the proxy actually tunnels HTTPS and
        returns a clean, unblocked body; ``None`` otherwise. This is the real
        filter: a TCP-connectable proxy can still be a dead port, a non-HTTP
        service, or a broken tunnel. With ``fallback`` the probe hits the plain
        reachability target (no bot wall) so a transient wall outage doesn't
        empty the pool.
        """
        import urllib.request

        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
        opener.addheaders = [("User-Agent", random.choice(USER_AGENTS))]
        target = VALIDATION_FALLBACK_TARGET if fallback else VALIDATION_TARGET
        t0 = time.time()
        try:
            with opener.open(target, timeout=VALIDATION_TIMEOUT) as r:
                body = r.read(2000)
            latency = time.time() - t0
            text = body.decode("utf-8", "replace") or ""
            if r.status == 200 and len(text) > 0 and not _check_block(text, r.status):
                return (proxy, latency)
        except Exception:
            pass
        return None

    def _validate_candidates(self, candidates: list, max_keep: int = 12, checker=None, fallback: bool = False) -> list:
        """Validate proxies in parallel and return the fastest ``max_keep``.

        ``checker`` is injectable for tests; default is a live HTTPS probe.
        """
        import concurrent.futures

        checker = checker or (lambda p: self._proxy_check(p, fallback=fallback))
        if not candidates:
            return []
        good: list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            for proxy, result in zip(candidates, ex.map(checker, candidates)):
                if result:
                    good.append(result)
        good.sort(key=lambda p: p[1])
        return [proxy for proxy, _ in good[:max_keep]]


# A process-wide shared pool so parallel fetches (webfetch_many) all rotate
# through the same proxy list (and only harvest from the list APIs once) instead
# of each fetch building — and repeatedly re-harvesting — its own pool.
_shared_pool: Optional["ProxyPool"] = None
_shared_pool_sig: tuple = ("", [])
_shared_pool_lock = threading.Lock()


def _pool_signature() -> tuple:
    """Return (harvest_enabled, env_proxy_list) — rebuild the shared pool when
    this changes (env is read at build time, and tests patch it)."""
    harvest = os.environ.get("OPENCODE_HARVEST_PROXIES", "") not in ("", "0", "false")
    raw = os.environ.get("OPENCODE_PROXY_POOL", "")
    proxies = [i for i in re.split(r"[,\s]+", raw.strip()) if i]
    return (harvest, proxies)


def _shared_pool_get() -> "ProxyPool":
    global _shared_pool, _shared_pool_sig
    sig = _pool_signature()
    with _shared_pool_lock:
        if _shared_pool is None or _shared_pool_sig != sig:
            _shared_pool = ProxyPool(harvest=sig[0])
            _shared_pool_sig = sig
        return _shared_pool


def _reset_shared_pool() -> None:
    """Test hook: drop the cached shared pool so the next call rebuilds it."""
    global _shared_pool, _shared_pool_sig
    with _shared_pool_lock:
        _shared_pool = None
        _shared_pool_sig = ("", [])


class UltimateBypass:
    """Fetch a URL through a cascade of HTTP methods to slip past Cloudflare."""

    def __init__(
        self,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        use_rotation: bool = True,
        timeout: int = 20,
        proxy_pool: Optional[ProxyPool] = None,
        ip_retries: int = 2,
    ):
        self.proxy = proxy
        self.user_agent = user_agent or random.choice(USER_AGENTS)
        self.use_rotation = use_rotation
        self.timeout = timeout
        self.cookie_manager = CookieManager()
        self.stats: Dict = {"methods_tried": 0, "success_method": None}
        self.proxy_pool = proxy_pool or ProxyPool()
        self.ip_retries = ip_retries
        self._last_proxy: Optional[str] = None

    def _get_headers(self) -> Dict:
        ua = random.choice(USER_AGENTS) if self.use_rotation else self.user_agent
        # Only advertise brotli if the installed HTTP stack can decode it;
        # otherwise a br response comes back as undecoded bytes on Termux.
        accept_encoding = "gzip, deflate"
        try:
            import brotli  # noqa: F401

            accept_encoding += ", br"
        except ImportError:
            pass
        # Modern browser header set: the missing sec-ch-ua / sec-fetch-* /
        # accept-language trio is exactly what bot detectors (DataDome, Akamai,
        # Cloudflare) score against, and it applies to the Python-based fetchers
        # where we control headers directly.
        return {
            "User-Agent": ua,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": accept_encoding,
            "Sec-CH-UA": (
                '"Chromium";v="131", "Not_A Brand";v="24", '
                '"Google Chrome";v="131"'
            ),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

    def _proxy_dict(self) -> Optional[Dict]:
        if not self.proxy:
            return None
        return {"http": self.proxy, "https": self.proxy}

    def _run_cmd(self, cmd: list, timeout: int = 25, env: Optional[Dict] = None) -> tuple:
        """Run an external fetch tool as an argument list (no shell), so flags
        and URLs are never interpreted by a shell and BusyBox/toybox quirks
        can't mangle quoting."""
        try:
            result = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return (result.returncode == 0, result.stdout or "", result.stderr or "")
        except subprocess.TimeoutExpired:
            return (False, "", "Timeout")
        except Exception as e:  # pragma: no cover
            return (False, "", str(e))

    def try_basic_request(self, url: str) -> Dict:
        if not REQUESTS_AVAILABLE:
            return {"success": False, "method": "requests", "error": "requests not installed"}
        from urllib.parse import urlparse

        try:
            domain = urlparse(url).netloc
            r = requests.get(
                url,
                headers=self._get_headers(),
                proxies=self._proxy_dict(),
                cookies=self.cookie_manager.load(domain),
                timeout=self.timeout,
            )
            if r.status_code == 200 and len(r.text) > 100 and not _check_block(r.text, r.status_code):
                if r.cookies:
                    self.cookie_manager.save(domain, dict(r.cookies))
                return {"success": True, "method": "requests", "content": r.text}
            return {"success": False, "method": "requests", "error": f"Status: {r.status_code}"}
        except Exception as e:
            return {"success": False, "method": "requests", "error": str(e)}

    def try_httpx(self, url: str) -> Dict:
        if not HTTPX_AVAILABLE:
            return {"success": False, "method": "httpx", "error": "httpx not installed"}
        from urllib.parse import urlparse

        domain = urlparse(url).netloc
        try:
            kwargs: Dict = {"timeout": self.timeout, "follow_redirects": True}
            proxy = self._proxy_for_httpx()
            if proxy:
                # httpx >= 0.28 dropped `proxies=` in favour of the single
                # `proxy=` (older versions accept a plain proxy string too).
                kwargs["proxy"] = proxy
            with httpx.Client(**kwargs) as client:
                # Reuse cookies another method earned for this domain (e.g. a
                # cf_clearance from requests) so httpx isn't scored fresh.
                r = client.get(url, headers=self._get_headers(), cookies=self.cookie_manager.load(domain))
            if r.status_code == 200 and len(r.text) > 100 and not _check_block(r.text, r.status_code):
                if r.cookies:
                    self.cookie_manager.save(domain, dict(r.cookies))
                return {"success": True, "method": "httpx", "content": r.text}
            detail = (r.text[:200] or "").replace("\n", " ").strip()
            return {
                "success": False,
                "method": "httpx",
                "error": f"Status: {r.status_code}" + (f": {detail}" if detail else ""),
            }
        except Exception as e:
            return {"success": False, "method": "httpx", "error": str(e)}

    def _proxy_for_httpx(self) -> Optional[str]:
        """Return the current proxy as an httpx-usable URL, or None.

        httpx requires a scheme on a proxy URL (``http://host:port``), while
        the env pool (OPENCODE_PROXY_POOL) and requests' ProxyHandler accept a
        bare ``host:port``. Normalize so the same proxy works in both.
        """
        if not self.proxy:
            return None
        p = self.proxy
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", p):
            return "http://" + p
        return p

    def try_curl_cffi(self, url: str) -> Dict:
        if not CURL_CFFI_AVAILABLE:
            return {"success": False, "method": "curl-cffi", "error": "curl_cffi not installed"}
        try:
            session = curl_requests.Session(proxies=self._proxy_dict())
            try:
                r = session.get(url, impersonate="chrome", timeout=self.timeout)
            finally:
                # Session holds its own connection pool; leaking it across a
                # long cascade (curl, httpx, wget...) would pin sockets/fds.
                session.close()
            if r.status_code == 200 and len(r.text) > 100 and not _check_block(r.text, r.status_code):
                return {"success": True, "method": "curl-cffi", "content": r.text}
            return {"success": False, "method": "curl-cffi", "error": f"Status: {r.status_code}"}
        except Exception as e:
            return {"success": False, "method": "curl-cffi", "error": str(e)}

    def try_cloudscraper(self, url: str) -> Dict:
        if not CLOUDSCRAPER_AVAILABLE:
            return {"success": False, "method": "cloudscraper", "error": "cloudscraper not installed"}
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True}
            )
            try:
                r = scraper.get(url, timeout=self.timeout)
            finally:
                scraper.close()
            if r.status_code == 200 and len(r.text) > 100 and not _check_block(r.text, r.status_code):
                return {"success": True, "method": "cloudscraper", "content": r.text}
            return {"success": False, "method": "cloudscraper", "error": f"Status: {r.status_code}"}
        except Exception as e:
            return {"success": False, "method": "cloudscraper", "error": str(e)}

    def try_curl(self, url: str) -> Dict:
        ua = random.choice(USER_AGENTS) if self.use_rotation else self.user_agent
        cmd = ["curl", "-s", "-L", "-A", ua, "--compressed"]
        if self.proxy:
            cmd += ["--proxy", self.proxy]
        # A browser-like header set is what makes curl pass bot walls that a
        # bare UA-only request trips; the host header ordering mirrors Chrome.
        for h in (
            "accept: text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8",
            "accept-language: en-US,en;q=0.9",
            "sec-ch-ua: \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\", \"Google Chrome\";v=\"131\"",
            "sec-ch-ua-mobile: ?0",
            "sec-ch-ua-platform: \"Windows\"",
            "sec-fetch-dest: document",
            "sec-fetch-mode: navigate",
            "sec-fetch-site: none",
            "sec-fetch-user: ?1",
            "upgrade-insecure-requests: 1",
        ):
            cmd += ["-H", h]
        cmd.append(url)
        success, stdout, stderr = self._run_cmd(cmd, self.timeout)
        if success and stdout and len(stdout) > 100 and not _check_block(stdout):
            return {"success": True, "method": "curl", "content": stdout}
        return {"success": False, "method": "curl", "error": stderr[:100] if stderr else "Empty"}

    def try_wget(self, url: str) -> Dict:
        ua = random.choice(USER_AGENTS) if self.use_rotation else self.user_agent
        # BusyBox/toybox wget (Termux/Android) only understand a subset of GNU
        # wget's flags, so walk a ladder of increasingly minimal invocations and
        # fall through when a flag is rejected. The final variant has no flags
        # beyond the mandatory ones, which every wget supports.
        base = [
            ["wget", "-q", "-O", "-", "--user-agent", ua],
            ["wget", "-q", "-O", "-", "-U", ua],
            ["wget", "-O", "-", "-U", ua],
            ["wget", "-O", "-"],
        ]
        last_err = ""
        for head in base:
            cmd = list(head)
            if self.proxy:
                cmd += ["-e", f"http_proxy={self.proxy}", "-e", f"https_proxy={self.proxy}"]
            cmd.append(url)
            success, stdout, stderr = self._run_cmd(cmd, self.timeout)
            if success and stdout and len(stdout) > 100 and not _check_block(stdout):
                return {"success": True, "method": "wget", "content": stdout}
            last_err = stderr[:100] if stderr else ""
            # Only flag-rejection warrants trying the next variant; a real
            # network failure would recur identically and waste time. GNU wget,
            # BusyBox and toybox each word the error differently (Unsupported
            # option / bad option / unrecognized option / invalid option).
            err = (stderr + " " + stdout).lower()
            if any(
                token in err
                for token in (
                    "no such option",
                    "unrecognized option",
                    "invalid option",
                    "bad option",
                    "unsupported option",
                    "unknown option",
                    "option requires an argument",
                )
            ):
                continue
            break
        return {"success": False, "method": "wget", "error": last_err or "Empty"}

    def fetch(self, url: str, force_method: Optional[str] = None,
              is_interrupted: Optional[Callable[[], bool]] = None) -> Dict:
        """Try each method in order until one returns clean content.

        When every method is blocked, rotates to a fresh proxy (a different
        exit IP) and repeats the cascade up to ``ip_retries`` times — this is
        what defeats IP-rate-limited bot walls: the request leaves from a new
        address each round instead of being throttled on one IP.

        A wall-clock budget caps the whole cascade (default ~2× the per-attempt
        timeout): without it a dead host burns up to 18 sequential full-timeout
        attempts and holds the turn for many minutes.
        """
        import time as _time

        methods = [
            "requests",
            "cloudscraper",
            "curl-cffi",
            "curl",
            "httpx",
            "wget",
        ]
        if force_method and force_method in methods:
            methods = [force_method] + [m for m in methods if m != force_method]

        budget = min(max(float(self.timeout) * 2, 45.0), 150.0)
        deadline = _time.monotonic() + budget

        def _cancelled() -> bool:
            if _time.monotonic() >= deadline:
                return True
            if is_interrupted is not None:
                try:
                    return bool(is_interrupted())
                except Exception:
                    return False
            return False

        last_result: Dict = {"success": False, "error": "Unknown"}
        # New fetch -> new session token (rotating gateways exit from a fresh
        # IP). end_fetch is a no-op but keeps begin/end symmetric; the token
        # is minted again on the next begin_fetch.
        self.proxy_pool.begin_fetch()
        for attempt in range(max(1, self.ip_retries + 1)):
            # pick a fresh exit IP each round (rotate within the pool / Tor)
            if self.proxy_pool.available:
                self._last_proxy = self.proxy_pool.next()
                if attempt > 0:
                    self.proxy_pool.new_identity()
            self.proxy = self._last_proxy

            result: Dict = {"success": False, "error": "Unknown"}
            for method in methods:
                if _cancelled():
                    result = {"success": False, "error": f"bypass budget exceeded ({budget:.0f}s)"}
                    break
                self.stats["methods_tried"] += 1
                if method == "requests":
                    result = self.try_basic_request(url)
                elif method == "cloudscraper":
                    result = self.try_cloudscraper(url)
                elif method == "curl-cffi":
                    result = self.try_curl_cffi(url)
                elif method == "curl":
                    result = self.try_curl(url)
                elif method == "httpx":
                    result = self.try_httpx(url)
                elif method == "wget":
                    result = self.try_wget(url)

                if result.get("success"):
                    self.stats["success_method"] = method
                    result["method"] = method
                    result["proxy"] = self._last_proxy
                    result["stats"] = self.stats
                    self.proxy_pool.mark_success(self._last_proxy)
                    self.proxy_pool.end_fetch()
                    return result

            last_result = result
            if _cancelled():
                break
            # the whole cascade failed through this proxy: let the pool learn
            self.proxy_pool.mark_failure(self._last_proxy)
            # pool fully demoted or drained? refill from cache/refresh rather
            # than quit; if nothing can be refilled there's no point retrying
            # the same dead IPs
            if not self.proxy_pool.available or self.proxy_pool.all_dead():
                if not self.proxy_pool.refill():
                    break

        self.proxy_pool.end_fetch()
        last_result["stats"] = self.stats
        last_result["rotations"] = attempt + 1 if self.proxy_pool.available else 1
        return last_result


def fetch_url(url: str, timeout: int = 20) -> Dict:
    """Convenience wrapper returning the raw result dict."""
    return UltimateBypass(timeout=timeout).fetch(url)
