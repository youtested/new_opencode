"""webfetch tool: fetch URL -> markdown/text with size caps.

RULES (full guidance for maintainers; the sent schema is short):
- GET-only, read-only: no forms, no JS, no clicks. HTTP upgrades to HTTPS.
- markdown default keeps headings/links/lists/tables/code/images; page
  <title> becomes top heading; nav/aside/footer chrome dropped.
- Blocked (4xx/5xx/JS-challenge) -> Cloudflare-bypass cascade
  (requests, cloudscraper, curl-cffi, curl, httpx, wget), shared proxy pool.
- Cookies persist per-domain. Results may be summarized when very large.
- webfetch_many: same stack, concurrent (default 5, max 10), wall time =
  slowest fetch, not the sum. URLs deduped, capped at 50, each capped to
  content_limit chars (default 8000).
"""

from __future__ import annotations

import json as _json
import re
from html.parser import HTMLParser as _HTMLParser
from typing import Any, Callable

from .registry import Tool, schema_with

# The Cloudflare bypass module pulls `requests` (and, when present on the
# phone, optional curl_cffi/cloudscraper), so it must NOT load at startup —
# opencode_py boots in headless/TUI mode without it until a fetch is actually
# blocked. `_ensure_bypass()` imports it lazily on first blocked fetch. The
# attributes stay module-level so tests that patch `webfetch.UltimateBypass`
# keep working (a patch makes it truthy before any lazy import runs).
UltimateBypass = None  # type: ignore[assignment]
CLOUDFLARE_BYPASS_AVAILABLE = False


def _ensure_bypass() -> bool:
    """Import the Cloudflare-bypass module on first use; True when usable."""
    global UltimateBypass, CLOUDFLARE_BYPASS_AVAILABLE
    if CLOUDFLARE_BYPASS_AVAILABLE or UltimateBypass is not None:
        # already imported, or the tests patched it — either way it's usable
        return True
    try:
        from .cloudflare_bypass import UltimateBypass as _UB  # type: ignore

        UltimateBypass = _UB
        CLOUDFLARE_BYPASS_AVAILABLE = True
        return True
    except Exception:  # pragma: no cover - library must not hard-fail
        return False

MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5 MB
DEFAULT_TIMEOUT = 30

MAX_BATCH_URLS = 50
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_BATCH_LIMIT = 8000  # chars of content returned per URL in a batch
DEFAULT_SINGLE_LIMIT = 8000  # chars for a single fetch: head + headings + refetch anchor (was: full 5MB dump)

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


FORMATS = ("markdown", "text", "html")

# content-types that are never worth decoding to text (binary mojibake)
NON_TEXT_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    "font/",
    "application/octet-stream",
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/x-gzip",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/vnd.rar",
    "application/x-bzip2",
    "application/x-lzma",
    "application/xz",
)


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    """Coerce a caller-supplied integer while clamping to a sane range.

    Calls reach tools with anything the model types (strings, None, "abc"),
    so trusting ``int(value)``/``min(max)`` directly can crash a fetch. Returns
    ``default`` for non-numeric input instead of raising.
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(v, hi))


def _is_non_text(content_type: str) -> bool:
    ct = (content_type or "").lower()
    for prefix in NON_TEXT_PREFIXES:
        if ct.startswith(prefix):
            return True
    return False


def _extract_main_html(html: str) -> str:
    """Readability-lite: prefer <main>/<article> body when present (stdlib only)."""
    try:
        blocks = []
        for tag in ("main", "article"):
            for m in re.finditer(r"(?is)<%s(?:\s[^>]*)?>(.*?)</%s\s*>" % (tag, tag), html):
                blocks.append(m.group(1))
        if blocks:
            return max(blocks, key=len)
    except Exception:
        pass
    return html


def _strip_boilerplate(html: str) -> str:
    """Drop nav/aside/footer chrome so the model reads content, not menus."""
    try:
        html = re.sub(r"(?is)<nav(?:\s[^>]*)?>.*?</nav\s*>", "\n", html)
        html = re.sub(r"(?is)<aside(?:\s[^>]*)?>.*?</aside\s*>", "\n", html)
        html = re.sub(r"(?is)<footer(?:\s[^>]*)?>.*?</footer\s*>", "\n", html)
    except Exception:
        pass
    return html


def _html_to_text(html: str) -> str:
    """HTML -> plain text (readability-lite, stdlib only)."""
    try:
        html = _strip_boilerplate(_extract_main_html(html))
    except Exception:
        pass
    html = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>", "", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|li|h[1-6]|tr|pre|blockquote)>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", "", html)
    import html as h

    text = h.unescape(html)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class _MDParser(_HTMLParser):
    """Stdlib HTML -> markdown: headings, links, lists, code, images."""

    def __init__(self, base_url: str = ""):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url or ""
        self.out: list[str] = []
        self.title = ""
        self.description = ""
        self._in_title = False
        self._skip = 0
        self._link: str | None = None
        self._link_text: list[str] = []
        self._in_pre = False
        self._in_cell = False
        self.forms: list[dict] = []
        self._form: dict | None = None

    def _emit(self, s: str) -> None:
        if self._link is not None:
            self._link_text.append(s)
        else:
            self.out.append(s)

    def _newline(self, n: int = 2) -> None:
        tail = "".join(self.out[-8:])
        if tail.strip() == "":
            return
        self.out.append("\n" * n)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        t = tag.lower()
        d = {k.lower(): (v or "") for k, v in attrs}
        if t in ("script", "style", "noscript", "template", "svg"):
            self._skip += 1
            return
        if self._skip:
            return
        if t == "title":
            self._in_title = True
            return
        if t == "meta":
            name = (d.get("name") or d.get("property") or "").lower()
            if name in ("description", "og:description", "twitter:description"):
                if d.get("content") and not self.description:
                    self.description = d["content"].strip()[:500]
            return
        if t == "base" and d.get("href") and not self.base_url:
            self.base_url = d["href"]
            return
        if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._newline(2)
            self.out.append("#" * int(t[1]) + " ")
        elif t in ("p", "div", "section", "article", "main", "header"):
            self._newline(2)
        elif t == "br":
            self.out.append("\n")
        elif t == "hr":
            self._newline(2)
            self.out.append("---")
            self._newline(2)
        elif t == "li":
            self._newline(1)
            self.out.append("- ")
        elif t == "a":
            href = d.get("href", "").strip()
            if href and not href.lower().startswith(("javascript:", "mailto:")):
                try:
                    from urllib.parse import urljoin

                    href = urljoin(self.base_url, href)
                except Exception:
                    pass
                self._link = href
                self._link_text = []
            else:
                self._link = None
        elif t in ("strong", "b"):
            self._emit("**")
        elif t in ("em", "i"):
            self._emit("*")
        elif t == "code" and not self._in_pre:
            self._emit("`")
        elif t == "pre":
            self._in_pre = True
            self._newline(2)
            self.out.append("```\n")
        elif t == "blockquote":
            self._newline(2)
            self.out.append("> ")
        elif t == "tr":
            self._newline(1)
        elif t in ("td", "th"):
            self.out.append(" | " if self._in_cell else "")
            self._in_cell = True
        elif t == "table":
            self._newline(2)
            self._in_cell = False
        elif t == "img":
            alt = d.get("alt", "").strip()
            src = d.get("src", "").strip()
            if src:
                try:
                    from urllib.parse import urljoin

                    src = urljoin(self.base_url, src)
                except Exception:
                    pass
                self._newline(1)
                self.out.append("![%s](%s)" % (alt, src))
                self._newline(1)
        elif t == "form":
            self._form = {
                "action": d.get("action", "").strip(),
                "method": (d.get("method") or "get").strip().lower() or "get",
                "inputs": [],
            }
        elif t in ("input", "select", "textarea", "button") and self._form is not None:
            name = (d.get("name") or d.get("id") or d.get("type") or t).strip()
            if name:
                self._form["inputs"].append(name[:80])

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript", "template", "svg"):
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if t == "title":
            self._in_title = False
            return
        if t in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "tr", "table"):
            self._newline(2)
            self._in_cell = False
        elif t == "li":
            self._newline(1)
        elif t == "a" and self._link is not None:
            text = "".join(self._link_text).strip()
            href = self._link
            self._link = None
            self._link_text = []
            if text:
                if text == href:
                    self.out.append(text)
                else:
                    self.out.append("[%s](%s)" % (text, href))
            else:
                self.out.append(href)
        elif t in ("strong", "b"):
            self._emit("**")
        elif t in ("em", "i"):
            self._emit("*")
        elif t == "code" and not self._in_pre:
            self._emit("`")
        elif t == "pre":
            self._in_pre = False
            self.out.append("\n```")
            self._newline(2)
        elif t == "form" and self._form is not None:
            try:
                from urllib.parse import urljoin

                if self._form.get("action"):
                    self._form["action"] = urljoin(self.base_url, self._form["action"])
            except Exception:
                pass
            if self._form.get("inputs"):
                self._form["inputs"] = self._form["inputs"][:20]
            self.forms.append(self._form)
            self._form = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title += data.strip()
            return
        if not data or (not self._in_pre and data.strip() == ""):
            return
        if self._in_pre:
            self._emit(data)
        else:
            self._emit(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        raw = "".join(self.out)
        lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in raw.splitlines()]
        cleaned: list[str] = []
        blank = 0
        for ln in lines:
            s = ln.strip()
            if not s:
                blank += 1
                if blank <= 1 and cleaned:
                    cleaned.append("")
                continue
            blank = 0
            cleaned.append(s)
        while cleaned and cleaned[0] == "":
            cleaned.pop(0)
        while cleaned and cleaned[-1] == "":
            cleaned.pop()
        return "\n".join(cleaned)


def _extract_metadata(html: str) -> dict:
    """Title + meta description via stdlib parser (never raises)."""
    try:
        p = _MDParser()
        p.feed(html[:500000])
        meta: dict = {}
        if p.title:
            meta["title"] = p.title[:300]
        if p.description:
            meta["description"] = p.description
        return meta
    except Exception:
        return {}


def _extract_forms(html: str, base_url: str = "") -> list[dict]:
    """Read-only form discovery (no submission): action/method/inputs."""
    try:
        p = _MDParser(base_url=base_url)
        p.feed(html[:1000000])
        return list(p.forms)[:10]
    except Exception:
        return []


def _html_to_markdown(html: str, base_url: str = "") -> str:
    """HTML -> markdown with headings/links/lists/tables (stdlib only).

    The page <title> becomes the top heading and discovered forms are listed
    (read-only discovery, no submission) so the model sees structure that
    metadata alone cannot carry — tool metadata is TUI-only, only `output`
    reaches the model.
    """
    try:
        main = _strip_boilerplate(_extract_main_html(html))
    except Exception:
        main = html
    try:
        meta = _extract_metadata(html)
        title = (meta.get("title") or "").strip()[:200]
    except Exception:
        title = ""
    try:
        forms = _extract_forms(html, base_url)
    except Exception:
        forms = []
    try:
        p = _MDParser(base_url=base_url)
        p.feed(main[:2000000])
        out = p.text()
        if not forms:
            try:
                forms = list(p.forms)[:10]
            except Exception:
                forms = []
        head = ""
        if title and not out.lstrip().startswith("# "):
            head = "# %s\n\n" % title
        if forms:
            rows = []
            for f in forms[:10]:
                action = (f.get("action") or "(same page)").strip() or "(same page)"
                method = (f.get("method") or "get").upper()
                inputs = ", ".join(f.get("inputs", [])[:20])
                rows.append("- form [%s] %s (%s)" % (method, action, inputs))
            block = "Forms (read-only):\n" + "\n".join(rows)
            out = (out + "\n\n---\n" + block) if out.strip() else block
        if out.strip():
            return head + out
        if head.strip():
            return head.rstrip()
    except Exception:
        pass
    return _html_to_text(html)


def _looks_like_block(content: str, status: int) -> bool:
    """Heuristic detection of a Cloudflare / challenge page body.

    A status code alone isn't enough and neither is a keyword: the whole point
    is to NOT route a perfectly good 200 page (e.g. an article about Cloudflare
    or a demo captcha) through the slow bypass cascade. Strong markers are
    challenge-specific and almost never appear in legit content; weak markers
    (the brand/tech words) only count on a non-200 status or when the page is
    small — challenge pages are small, full docs are not.
    """
    if not content:
        return False
    lower = content.lower()
    strong = [
        "checking your browser",
        "just a moment",
        "verifying you are human",
        "attention required",
        "cf-challenge",
        "cf-chl",
    ]
    weak = ["cloudflare", "captcha", "enable javascript", "cf-mitigated"]
    small = len(content) < 256 * 1024
    hit_strong = any(m in lower for m in strong)
    hit_weak = any(m in lower for m in weak)
    if hit_strong:
        return True
    if status != 200 and small and hit_weak:
        return True
    if status in (401, 403, 429, 503, 504) and hit_weak:
        return True
    return False


def _pretty_json(body: str) -> str:
    """Pretty-print a JSON body (never raises, falls back to raw)."""
    try:
        data = _json.loads(body)
        out = _json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
        return out
    except Exception:
        return body


def _xml_feed_to_markdown(body: str, base_url: str = "") -> str:
    """RSS/Atom/XML -> markdown list of items (stdlib regex, never raises)."""
    try:
        items: list[str] = []
        chan_title = ""
        m_chan = re.search(r"(?is)<channel[^>]*>.*?<title[^>]*>(.*?)</title>", body)
        if m_chan:
            chan_title = re.sub(r"(?is)<[^>]+>", "", m_chan.group(1)).strip()
            import html as h

            chan_title = h.unescape(chan_title)[:200]
        blocks: list[str] = []
        blocks += re.findall(r"(?is)<item[^>]*>(.*?)</item\s*>", body)
        blocks += re.findall(r"(?is)<entry[^>]*>(.*?)</entry\s*>", body)
        import html as h

        for b in blocks[:50]:
            def _tag(name: str) -> str:
                m = re.search(r"(?is)<%s[^>]*>(.*?)</%s\s*>" % (name, name), b)
                if not m:
                    return ""
                t = re.sub(r"(?is)<[^>]+>", "", m.group(1)).strip()
                return h.unescape(t)[:300]

            title = _tag("title")
            link = ""
            m_link = re.search(r'(?is)<link[^>]*href=["\']([^"\']+)["\']', b)
            if m_link:
                link = m_link.group(1).strip()
            else:
                link = _tag("link")
            desc = _tag("description") or _tag("summary") or _tag("content")
            if base_url and link:
                try:
                    from urllib.parse import urljoin

                    link = urljoin(base_url, link)
                except Exception:
                    pass
            if title and link:
                items.append("- [%s](%s)%s" % (title, link, (" — " + desc[:200]) if desc else ""))
            elif title:
                items.append("- %s%s" % (title, (" — " + desc[:200]) if desc else ""))
            elif link:
                items.append("- %s" % link)
        head = ("# %s\n\n" % chan_title) if chan_title else ""
        if items:
            return head + "\n".join(items)
    except Exception:
        pass
    return _html_to_text(body)


def _convert_body(body: str, format: str, content_type: str, base_url: str = "") -> str:
    """Apply the requested format conversion to a body string."""
    ct = (content_type or "").lower()
    if "json" in ct:
        pretty = _pretty_json(body)
        if format == "html":
            return body
        return pretty
    if "rss" in ct or "atom" in ct or "xml" in ct:
        if format == "html":
            return body
        if format == "text":
            return _html_to_text(body)
        return _xml_feed_to_markdown(body, base_url)
    if format == "text":
        if "text/html" in ct:
            return _html_to_text(body)
        return body
    if format == "html":
        return body
    if "text/html" in ct:
        return _html_to_markdown(body, base_url)
    return body


def _cookie_file(domain: str):
    """Cookie file path shared with the bypass CookieManager (no heavy import)."""
    from pathlib import Path as _P

    import os as _os

    base = _P(_os.environ.get("XDG_CACHE_HOME", str(_P.home() / ".cache"))) / "opencode" / "cookies"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", domain)
    return base / f"{safe}.json"


def _load_cookies(domain: str) -> dict | None:
    try:
        p = _cookie_file(domain)
        if not p.exists():
            return None
        data = _json.loads(p.read_text())
        if isinstance(data, dict) and data:
            return data
    except Exception:
        pass
    return None


def _save_cookies(domain: str, jar) -> None:
    try:
        d: dict = {}
        if jar is None:
            return
        if isinstance(jar, dict):
            d = dict(jar)
        else:
            try:
                d = dict(jar)
            except Exception:
                try:
                    d = {c.name: c.value for c in jar}  # type: ignore
                except Exception:
                    return
        if not d:
            return
        lock = getattr(_save_cookies, "_lock", None)
        if lock is None:
            try:
                import threading as _threading

                lock = _threading.Lock()
                _save_cookies._lock = lock  # type: ignore[attr-defined]
            except Exception:
                lock = None
        if lock is not None:
            lock.acquire()
        try:
            prev = _load_cookies(domain) or {}
            prev.update({str(k): str(v) for k, v in d.items()})
            _cookie_file(domain).parent.mkdir(parents=True, exist_ok=True)
            tmp = _cookie_file(domain).with_suffix(".json.tmp")
            tmp.write_text(_json.dumps(prev))
            import os as _os

            _os.replace(tmp, _cookie_file(domain))
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass
    except Exception:
        pass


# Hosts where a forced HTTPS upgrade breaks the fetch: RFC1918 / link-local /
# loopback / mDNS names serve plain HTTP on LAN devices (routers, IoT, local
# dev servers). Upgrading those to https:// produced instant SSL failures.
_LAN_HOST_RE = re.compile(
    r"^(localhost$|127\.|10\.|192\.168\.|169\.254\.|"
    r"172\.(1[6-9]|2\d|3[01])\.|\[::1\]$|.*\.local$)",
    re.IGNORECASE,
)


def _is_lan_host(url: str) -> bool:
    try:
        from urllib.parse import urlsplit

        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        host = ""
    if not host:
        return False
    return bool(_LAN_HOST_RE.match(host))


_BLOCKED_FETCH_HOSTS = re.compile(r"^(localhost$|127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.|0\.0\.0\.0|::1|\[::1\]$)", re.IGNORECASE)


_LAN_ONLY_UNBLOCKED_RE = re.compile(r"^(127\.0\.0\.1$|localhost$|\[::1\]$|::1$)", re.IGNORECASE)


def _is_blocked_host(url: str) -> bool:
    try:
        from urllib.parse import urlsplit
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return True
    if not host:
        return True
    if _LAN_ONLY_UNBLOCKED_RE.match(host):
        return False
    return bool(_BLOCKED_FETCH_HOSTS.match(host))


def _webfetch(
    url: str,
    format: str = "markdown",
    timeout: int = DEFAULT_TIMEOUT,
    is_interrupted: Callable[[], bool] | None = None,
    registry: Any | None = None,
) -> dict:
    if not re.match(r"^https?://", url):
        return {"output": "URL must start with http:// or https://", "error": True}
    _wb = _is_blocked_host(url)
    if _wb:
        return {"output": "Fetch blocked: private/loopback/metadata hosts are not allowed.", "error": True}
    was_interrupted_at_entry = False
    if is_interrupted is not None:
        try:
            was_interrupted_at_entry = bool(is_interrupted())
        except Exception:
            pass
    if format not in FORMATS:
        format = "markdown"
    timeout = _clamp_int(timeout, DEFAULT_TIMEOUT, 1, 120)
    # Local / LAN hosts may serve plain HTTP only; don't force-upgrade them.
    upgraded = url.startswith("http://") and not _is_lan_host(url)
    if upgraded:
        url = "https://" + url[len("http://"):]
    headers = {
        "User-Agent": BROWSER_UA,
        "Accept": {
            "markdown": "text/markdown, text/plain;q=0.9, text/html;q=0.5, */*;q=0.1",
            "text": "text/plain, text/markdown;q=0.9, text/html;q=0.5, */*;q=0.1",
            "html": "text/html, */*;q=0.8",
        }.get(format, "*/*"),
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    }
    truncated_note = ""

    def interrupted() -> dict:
        return {
            "output": "(interrupted)",
            "error": True,
            "interrupted": True,
            "stopped": True,
            "metadata": {"upgraded_to_https": upgraded},
        }

    def _wants_stop() -> bool:
        return is_interrupted is not None and bool(is_interrupted())

    try:
        import httpx

        from urllib.parse import urlsplit as _urlsplit

        try:
            _domain = (_urlsplit(url).hostname or "").lower()
        except Exception:
            _domain = ""
        _jar = _load_cookies(_domain) if _domain else None
        with httpx.Client(timeout=timeout, follow_redirects=False, cookies=_jar) as client:
            redirects_followed = 0
            parts: list[bytes] = []
            size = 0
            hit_cap = False
            status = 0
            content_type = ""
            while True:
                if _is_blocked_host(url):
                    return {"output": "Fetch blocked: private/loopback/metadata hosts are not allowed.", "error": True}
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code in (301, 302, 303, 307, 308):
                        loc = resp.headers.get("location") or ""
                        resp.close()
                        if not loc:
                            status = resp.status_code
                            break
                        from urllib.parse import urljoin as _urljoin

                        url = _urljoin(url, loc)
                        if not re.match(r"^https?://", url):
                            return {"output": "Redirect to non-http URL blocked.", "error": True}
                        redirects_followed += 1
                        if redirects_followed > 5:
                            return {"output": "Too many redirects (over 5).", "error": True}
                        try:
                            _domain = (_urlsplit(url).hostname or "").lower()
                        except Exception:
                            _domain = ""
                        continue
                    content_type = resp.headers.get("content-type", "")
                    # Don't decode images/binaries into 5 MB of mojibake: a quick
                    # content-type check (before streaming the body) short-circuits
                    # media, PDFs and archives with a short notice.
                    if _is_non_text(content_type):
                        resp.close()
                        return {
                            "output": (
                                f"Remote content is non-text ({content_type or 'unknown type'}). "
                                "The body was not fetched."
                            ),
                            "metadata": {"content_type": content_type, "upgraded_to_https": upgraded},
                        }
                    if _wants_stop():
                        resp.close()
                        return interrupted()
                    parts = []
                    size = 0
                    hit_cap = False
                    # Register the response so an interrupt (2nd ESC / Ctrl+C) can
                    # force-close the socket and wake a blocked read instantly;
                    # the tool then polls the shared interrupt flag and aborts.
                    registered = registry is not None
                    if registered:
                        try:
                            registry.register_fetch(resp)  # type: ignore[attr-defined]
                        except Exception:
                            registered = False
                    try:
                        for chunk in resp.iter_bytes():
                            if _wants_stop():
                                resp.close()
                                return interrupted()
                            room = MAX_RESPONSE_SIZE - size
                            if room <= 0:
                                hit_cap = True
                                truncated_note = f"\n\n[Response truncated at {MAX_RESPONSE_SIZE} bytes]"
                                # Stop reading and release the connection immediately.
                                # Draining the remainder (old behavior) lets an endless
                                # streaming response hang the tool forever and defeats
                                # the byte cap. The httpx context manager also closes on
                                # exit; resp.close() frees the socket right now.
                                resp.close()
                                break
                            parts.append(chunk[:room])
                            size += len(chunk[:room])
                    finally:
                        if registered:
                            try:
                                registry.unregister_fetch(resp)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                    status = resp.status_code
                    try:
                        _save_cookies(_domain, getattr(resp, "cookies", None))
                    except Exception:
                        pass
                    break
        body = b"".join(parts).decode("utf-8", errors="replace")
    except httpx.HTTPError as e:
        # A forced abort (registry.abort_fetches) closed the response while we
        # were blocked reading — that's an interrupt, not a network failure.
        if _wants_stop():
            return interrupted()
        msg = f"Fetch failed: {e}"
        if upgraded:
            msg += " (the http:// URL was upgraded to https://)"
        return {"output": msg, "error": True}

    if status == 200 and not _looks_like_block(body, status):
        if was_interrupted_at_entry:
            return {"output": "(interrupted)", "error": True, "interrupted": True, "stopped": True}
        out = _convert_body(body, format, content_type, url) + truncated_note
        meta: dict = {"upgraded_to_https": upgraded, "content_type": content_type,
                      "final_url": url, "status": status}
        if "text/html" in (content_type or "").lower():
            meta.update(_extract_metadata(body))
        return {"output": out, "metadata": meta}
    # Try the cascade for ANY failed status (not just challenge-looking bodies):
    # a bare 403 page is often only a bot wall that a different method or UA
    # (requests vs httpx vs curl) slips past (e.g. wikipedia/reddit 403 on
    # httpx but 200 via requests/curl).
    if _ensure_bypass():
        return _bypass_fetch(url, format, timeout, upgraded, hint=status, is_interrupted=is_interrupted)
    return {"output": f"Fetch failed: HTTP {status}", "error": True}


def _bypass_fetch(
    url: str,
    format: str,
    timeout: int,
    upgraded: bool,
    hint: int = 0,
    is_interrupted: Callable[[], bool] | None = None,
) -> dict:
    """Try the Cloudflare bypass cascade with optional per-attempt IP rotation.

    Rotation sources (no Tor required):
      * OPENCODE_PROXY_POOL  — a comma/space list of proxy URLs.
      * OPENCODE_HARVEST_PROXIES=1 — auto-pull free public proxies.
    Rotating means a blocked attempt retries via a fresh exit IP, which dodges
    per-IP rate limiting; it does not change the blocked-JS-challenge outcome.
    """
    if is_interrupted is not None and is_interrupted():
        return {"output": "(interrupted)", "error": True, "interrupted": True, "stopped": True}
    try:
        from .cloudflare_bypass import ProxyPool

        # Shared process-wide pool: parallel fetches (webfetch_many) all rotate
        # through the same proxy list and the free-proxy harvest runs once.
        # Falls back to an empty pool when no rotation is configured.
        ub = UltimateBypass(timeout=timeout, proxy_pool=ProxyPool.shared())
        result = ub.fetch(url, is_interrupted=is_interrupted)
    except Exception as e:  # pragma: no cover
        return {"output": f"Fetch failed (bypass): {e}", "error": True}
    if is_interrupted is not None and is_interrupted():
        return {"output": "(interrupted)", "error": True, "interrupted": True, "stopped": True}

    if not result.get("success"):
        err = result.get("error") or "Unknown"
        msg = f"Fetch blocked (HTTP {hint}) and bypass failed: {err}"
        if upgraded:
            msg += " (the http:// URL was upgraded to https://)"
        return {"output": msg, "error": True}

    content = result.get("content", "")
    content_type = "text/html; charset=utf-8"
    truncated_note = ""
    if len(content) > MAX_RESPONSE_SIZE:
        content = content[:MAX_RESPONSE_SIZE]
        truncated_note = f"\n\n[Response truncated at {MAX_RESPONSE_SIZE} bytes]"
    out = _convert_body(content, format, content_type, url) + truncated_note
    meta = {
        "bypassed": True,
        "bypass_method": result.get("method"),
        "upgraded_to_https": upgraded,
        "content_type": content_type,
        "final_url": url,
    }
    meta.update(_extract_metadata(content))
    return {"output": out, "metadata": meta}


def _headings_of(md: str, n: int = 20) -> list[str]:
    """First N markdown headings — the map of what was cut."""
    out = []
    try:
        for ln in md.splitlines():
            s = ln.strip()
            if s.startswith("#") and len(s) > 1:
                out.append(s[:120])
                if len(out) >= n:
                    break
    except Exception:
        pass
    return out


def _cap_single(out: str, limit: int) -> tuple[str, bool, list[str]]:
    """Head + refetch anchor: first `limit` chars, headings map, totals. Nothing deleted upstream."""
    if len(out) <= limit:
        return out, False, []
    heads = _headings_of(out)
    head = out[:limit].rstrip()
    note = [f"\n\n[Single fetch capped at {limit} of {len(out)} chars — batch already caps at 8000/URL.]"]
    if heads:
        note.append("Headings on this page:")
        note.extend(f"  {h}" for h in heads[:20])
    note.append("Need more? Re-fetch with a bigger content_limit (e.g. 20000/50000), or format='text' for tables/docs past the cut.")
    return head + "\n".join(note), True, heads


def tool(registry: Any | None = None) -> Tool:
    description = """Fetch URLs you ALREADY have (NOT for search — use websearch to find URLs first). Fetch one URL (pass urls=[...] for up to 50 in parallel, wall time = slowest). Markdown/text/html. Blocked pages retry via bypass cascade."""

    def run(input: dict) -> dict:
        fmt = input.get("format", "markdown")
        if fmt not in FORMATS:
            fmt = "markdown"
        # Read the engine's interrupt callback at call time (the registry hook
        # is installed by AgentLoop.__init__) so ESC/Ctrl+C aborts an in-flight
        # fetch instead of letting it run to its timeout. Each engine (main +
        # sub-agents) owns its own registry, so workers always see their own
        # turn's interrupt state.
        is_interrupted: Callable[[], bool] | None = None
        if registry is not None:
            checker = getattr(registry, "interrupt_check", None)
            if callable(checker):
                is_interrupted = checker
        if not input.get("urls") and not input.get("url"):
            return {"output": "webfetch requires 'url' or parallel 'urls=[...]'.", "error": True}
        if input.get("urls"):
            return webfetch_many(
                input.get("urls", []),
                format=fmt,
                timeout=input.get("timeout") or DEFAULT_TIMEOUT,
                max_concurrent=input.get("max_concurrent") or DEFAULT_MAX_CONCURRENT,
                content_limit=input.get("content_limit") or DEFAULT_BATCH_LIMIT,
                is_interrupted=is_interrupted,
                registry=registry,
            )
        single = _webfetch(
            input["url"],
            fmt,
            input.get("timeout", DEFAULT_TIMEOUT),
            is_interrupted=is_interrupted,
            registry=registry,
        )
        if single.get("error"):
            return single
        try:
            slim = int(input.get("content_limit") or DEFAULT_SINGLE_LIMIT)
        except (TypeError, ValueError):
            slim = DEFAULT_SINGLE_LIMIT
        slim = max(1000, min(slim, MAX_RESPONSE_SIZE))
        body = str(single.get("output", "") or "")
        capped, was_cut, heads = _cap_single(body, slim)
        if was_cut:
            single = dict(single)
            single["output"] = capped
            md = dict(single.get("metadata") or {})
            md.update({"truncated": True, "total_chars": len(body),
                       "shown_chars": len(capped), "headings": heads,
                       "content_limit": slim})
            single["metadata"] = md
        else:
            md = dict(single.get("metadata") or {})
            md.update({"truncated": False, "total_chars": len(body),
                       "content_limit": slim})
            single["metadata"] = md
        return single

    return Tool(
        name="webfetch",
        description=description,
        parameters=schema_with(
            {
                "url": {"type": "string", "description": "Full URL to fetch", "optional": True},
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Parallel: up to 50 URLs (wall time = slowest)",
                    "optional": True,
                },
                "format": {
                    "type": "string",
                    "description": "markdown (default), text, or html",
                    "enum": ["markdown", "text", "html"],
                    "optional": True,
                },
                "timeout": {"type": "integer", "description": "Seconds (max 120)", "optional": True},
                "max_concurrent": {
                    "type": "integer",
                    "description": "Parallel workers 1-10 (default 5)",
                    "optional": True,
                },
                "content_limit": {
                    "type": "integer",
                    "description": "Chars per fetch, single or parallel (default 8000)",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="webfetch",
    )


def _fetch_one(
    url: str,
    format: str,
    timeout: int,
    content_limit: int,
    is_interrupted: Callable[[], bool] | None = None,
    registry: Any | None = None,
) -> tuple:
    """Fetch a single URL via the shared _webfetch cascade; returns
    (url, ok, content, truncated, elapsed_s, chars). Runs in a worker thread."""
    import time as _t
    _s = _t.monotonic()
    r = _webfetch(url, format, timeout, is_interrupted=is_interrupted, registry=registry)
    elapsed = _t.monotonic() - _s
    if r.get("error"):
        return (url, False, str(r.get("output", "Unknown error")), False, elapsed, 0)
    out = str(r.get("output", "") or "")
    truncated = False
    if len(out) > content_limit:
        out = out[:content_limit].rstrip()
        truncated = True
    return (url, True, out, truncated, elapsed, len(out))


def webfetch_many(
    urls: list[str],
    format: str = "markdown",
    timeout: int = DEFAULT_TIMEOUT,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    content_limit: int = DEFAULT_BATCH_LIMIT,
    is_interrupted: Callable[[], bool] | None = None,
    registry: Any | None = None,
) -> dict:
    """Fetch many URLs concurrently and return each result keyed by URL.

    Wall time is bounded by the slowest fetch (not the sum), up to
    ``max_concurrent`` workers. Every URL goes through the same primary fetch
    and Cloudflare-bypass cascade as ``_webfetch``, and all parallel fetches
    share the process-wide proxy pool so IP rotation still works under load.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not isinstance(urls, list) or not urls:
        return {"output": "webfetch_many requires a non-empty 'urls' array.", "error": True}
    if not all(isinstance(u, str) for u in urls):
        return {"output": "webfetch_many 'urls' must be an array of strings.", "error": True}

    # Dedupe keeping order, cap the count so the result stays usable.
    seen: list[str] = []
    for u in urls:
        if u not in seen:
            seen.append(u)
    dropped = seen[MAX_BATCH_URLS:]
    urls = seen[:MAX_BATCH_URLS]

    workers = _clamp_int(max_concurrent, DEFAULT_MAX_CONCURRENT, 1, min(10, len(urls)))
    limit = _clamp_int(content_limit, DEFAULT_BATCH_LIMIT, 1, MAX_RESPONSE_SIZE)
    timeout = _clamp_int(timeout, DEFAULT_TIMEOUT, 1, 120)

    _progress = None
    try:
        _pctx = getattr(registry, "_progress_ctx", None)
        if _pctx is not None:
            _em = getattr(_pctx, "emitter", None)
            if callable(_em):
                _progress = _em
    except Exception:
        _progress = None

    results: list = [None] * len(urls)  # deterministic (submission) order
    interrupted = False
    import time as _tt
    _wall0 = _tt.monotonic()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wf") as ex:
        futures = {
            ex.submit(_fetch_one, u, format, timeout, limit, is_interrupted, registry): i
            for i, u in enumerate(urls)
        }
        # Each fetch polls the shared interrupt flag (2nd ESC / Ctrl+C) and
        # aborts itself; an interrupt stops the turn without waiting for every
        # slow URL. Remaining in-flight workers check the flag on their own and
        # return immediately.
        for fut in as_completed(futures):
            try:
                results[futures[fut]] = fut.result()
            except Exception as e:
                i = futures[fut]
                results[i] = (urls[i], False, f"Fetch failed: {e}", False, 0.0, 0)
            try:
                if _progress is not None:
                    _done = sum(1 for r in results if r is not None)
                    _progress(_done, len(urls))
            except Exception:
                pass
            if is_interrupted is not None and is_interrupted():
                interrupted = True
                break

    if interrupted:
        return {"output": "(interrupted)", "error": True, "interrupted": True, "stopped": True}

    _wall = _tt.monotonic() - _wall0
    ok = sum(1 for r in results if len(r) > 1 and r[1])
    per_url = []
    for r in results:
        try:
            per_url.append({"url": r[0], "ok": bool(r[1]), "seconds": round(float(r[4]), 2), "chars": int(r[5])})
        except Exception:
            per_url.append({"url": str(r[0]) if r else "?", "ok": False, "seconds": 0.0, "chars": 0})
    slowest = max((p["seconds"] for p in per_url), default=0.0)
    lines = [f"# Batch fetch results ({len(urls)} urls, {workers} workers, wall {_wall:.2f}s ~ slowest {slowest:.2f}s)"]
    for i, r in enumerate(results, 1):
        url, success, content, truncated = r[0], r[1], r[2], r[3]
        secs = per_url[i - 1]["seconds"]
        if success:
            lines.append(f"\n## {i}. {url}  (OK{' [truncated]' if truncated else ''} {secs:.2f}s)\n{content}")
        else:
            lines.append(f"\n## {i}. {url}  (FAILED {secs:.2f}s)\n{content}")
    if dropped:
        lines.append(f"\n[note: {len(dropped)} urls beyond the {MAX_BATCH_URLS} cap were dropped]")

    metadata = {
        "count": len(urls),
        "succeeded": ok,
        "failed": len(urls) - ok,
        "dropped": len(dropped),
        "concurrency": workers,
        "wall_seconds": round(_wall, 2),
        "slowest_seconds": slowest,
        "per_url": per_url,
    }
    if dropped:
        metadata["dropped_urls"] = dropped
    return {"output": "\n".join(lines), "metadata": metadata}


def batch_tool(registry: Any | None = None) -> Tool:
    description = """Fetch up to 50 URLs in parallel (wall time = slowest fetch). Same stack as webfetch; each capped to content_limit chars."""

    def run(input: dict) -> dict:
        is_interrupted: Callable[[], bool] | None = None
        if registry is not None:
            checker = getattr(registry, "interrupt_check", None)
            if callable(checker):
                is_interrupted = checker
        return webfetch_many(
            input.get("urls", []),
            format=input.get("format", "markdown"),
            timeout=input.get("timeout") or DEFAULT_TIMEOUT,
            max_concurrent=input.get("max_concurrent") or DEFAULT_MAX_CONCURRENT,
            content_limit=input.get("content_limit") or DEFAULT_BATCH_LIMIT,
            is_interrupted=is_interrupted,
            registry=registry,
        )

    return Tool(
        name="webfetch_many",
        description=description,
        parameters=schema_with(
            {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs (deduped, max 50)",
                },
                "format": {
                    "type": "string",
                    "description": "markdown (default), text, or html",
                    "enum": ["markdown", "text", "html"],
                    "optional": True,
                },
                "timeout": {"type": "integer", "description": "Seconds, max 120", "optional": True},
                "max_concurrent": {
                    "type": "integer",
                    "description": "Workers 1-10 (default 5)",
                    "optional": True,
                },
                "content_limit": {
                    "type": "integer",
                    "description": "Chars per URL (default 8000)",
                    "optional": True,
                },
            },
            ["urls"],
        ),
        run=run,
        permission="webfetch",
    )
