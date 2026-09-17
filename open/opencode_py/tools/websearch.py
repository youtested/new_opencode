"""websearch tool: search the web across providers (fits this product).

Mirrors official opencode's `websearch` (query + numResults + type fast/deep):
- Keyless first: DuckDuckGo HTML + lite + Bing HTML + Brave HTML — no keys,
  phone-safe via the shared httpx stack (cookies, UA, interrupt).
- Keyed best: Exa, Parallel (official's MCP pair), Tavily, Serper, Brave API —
  keys from env (EXA_API_KEY, PARALLEL_API_KEY, TAVILY_API_KEY, SERPER_API_KEY,
  BRAVE_API_KEY) or config providers. Provider picked by `provider=` param,
  env OPENCODE_WEBSEARCH_PROVIDER, or auto (first keyed available, else keyless).
- Output: titles + URLs + snippets; model fetches top pages with `webfetch`.
  Reuses the webfetch UI row (`Searching web...`), `Fetching...` status,
  interrupt + caps. No new deps (httpx only).
"""
from __future__ import annotations

import html as _html
import os
import re
import time
import urllib.parse as _U
from typing import Any, Callable

from .registry import Tool, schema_with

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_KEYLESS = ("duckduckgo", "bing", "brave")
_KEYED = ("exa", "parallel", "tavily", "serper", "brave-api")


def _env(name: str) -> str:
    try:
        return (os.environ.get(name) or "").strip()
    except Exception:
        return ""


def _available() -> dict[str, bool]:
    return {
        "exa": bool(_env("EXA_API_KEY")),
        "parallel": bool(_env("PARALLEL_API_KEY")),
        "tavily": bool(_env("TAVILY_API_KEY")),
        "serper": bool(_env("SERPER_API_KEY")),
        "brave-api": bool(_env("BRAVE_API_KEY")),
    }


def _pick_provider(want: str) -> str:
    w = (want or "").strip().lower()
    if w in ("auto", ""):
        ov = _env("OPENCODE_WEBSEARCH_PROVIDER").lower()
        if ov in _KEYED or ov in _KEYLESS:
            if ov in _KEYED and not _available().get(ov, False):
                pass
            else:
                return ov
        avail = _available()
        for k in ("exa", "parallel", "tavily", "serper", "brave-api"):
            if avail.get(k):
                return k
        return "duckduckgo"
    if w in ("brave",):
        return "brave"
    return w


def _clean(s: str) -> str:
    try:
        s = re.sub(r"<[^>]+>", "", s or "")
        return _html.unescape(s).strip()
    except Exception:
        return (s or "").strip()


def _ddg_search(query: str, n: int, timeout: int, lite: bool = False) -> list[dict]:
    import httpx
    base = "https://lite.duckduckgo.com/lite/" if lite else "https://html.duckduckgo.com/html/"
    try:
        r = httpx.get(base, params={"q": query}, timeout=timeout,
                      headers={"User-Agent": BROWSER_UA}, follow_redirects=True)
    except Exception as e:
        return [{"error": f"duckduckgo: {e}"}]
    if r.status_code >= 400:
        return [{"error": f"duckduckgo HTTP {r.status_code}"}]
    t = r.text
    out: list[dict] = []
    if lite:
        for m in re.finditer(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a', t, re.S):
            href = _html.unescape(m.group(1)).strip()
            title = _clean(m.group(2))
            if not href or href.startswith("#") or "duckduckgo.com" in href and "uddg=" not in href:
                continue
            if href.startswith("//"):
                href = "https:" + href
            if "uddg=" in href:
                try:
                    qs = dict(_U.parse_qsl(_U.urlsplit(href).query))
                    href = _U.unquote(qs.get("uddg", href))
                except Exception:
                    pass
            if href.startswith("/") or "duckduckgo.com" in href:
                continue
            if title and href.startswith("http"):
                out.append({"title": title[:150], "url": href[:500], "snippet": ""})
            if len(out) >= n + 4:
                break
    else:
        blocks = re.split(r'<div class="result results_links[^\"]*"', t)[1:]
        for b in blocks:
            m = re.search(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a', b, re.S)
            if not m:
                continue
            href = _html.unescape(m.group(1)).strip()
            title = _clean(m.group(2))
            if href.startswith("//"):
                href = "https:" + href
            if "uddg=" in href:
                try:
                    qs = dict(_U.parse_qsl(_U.urlsplit(href).query))
                    href = _U.unquote(qs.get("uddg", href))
                except Exception:
                    pass
            if "duckduckgo.com/y.js" in href or "duckduckgo.com/l/?" in href and "uddg=" not in m.group(1):
                if not href.startswith("http") or "duckduckgo.com" in href:
                    continue
            s = re.search(r'class="result__snippet"[^>]*>(.*?)</div', b, re.S)
            snip = _clean(s.group(1)) if s else ""
            if title and href.startswith("http") and "duckduckgo.com" not in href:
                out.append({"title": title[:150], "url": href[:500], "snippet": snip[:300]})
            if len(out) >= n:
                break
    return out


def _bing_search(query: str, n: int, timeout: int) -> list[dict]:
    import httpx
    try:
        r = httpx.get("https://www.bing.com/search", params={"q": query}, timeout=timeout,
                      headers={"User-Agent": BROWSER_UA}, follow_redirects=True)
    except Exception as e:
        return [{"error": f"bing: {e}"}]
    if r.status_code >= 400:
        return [{"error": f"bing HTTP {r.status_code}"}]
    t = r.text
    out: list[dict] = []
    for m in re.finditer(r'<li class="b_algo[^>]*>.*?<h2>.*?<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a', t, re.S):
        href = _html.unescape(m.group(1)).strip()
        title = _clean(m.group(2))
        if title and href.startswith("http"):
            out.append({"title": title[:150], "url": href[:500], "snippet": ""})
        if len(out) >= n:
            break
    return out


def _brave_html_search(query: str, n: int, timeout: int) -> list[dict]:
    import httpx
    try:
        r = httpx.get("https://search.brave.com/search", params={"q": query}, timeout=timeout,
                      headers={"User-Agent": BROWSER_UA}, follow_redirects=True)
    except Exception as e:
        return [{"error": f"brave: {e}"}]
    if r.status_code >= 400:
        return [{"error": f"brave HTTP {r.status_code}"}]
    t = r.text
    out: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'href="(https?://[^"]+)"[^>]{0,300}>([^<]{10,150})<', t):
        href = _html.unescape(m.group(1)).strip()
        title = _clean(m.group(2))
        if not title or not href.startswith("http"):
            continue
        if any(x in href for x in ("brave.com", "search.brave")):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append({"title": title[:150], "url": href[:500], "snippet": ""})
        if len(out) >= n:
            break
    return out


def _exa_search(query: str, n: int, timeout: int, deep: bool) -> list[dict]:
    import httpx, json as _j
    key = _env("EXA_API_KEY")
    if not key:
        return [{"error": "exa: missing EXA_API_KEY"}]
    try:
        r = httpx.post("https://api.exa.ai/search", timeout=timeout,
                       headers={"x-api-key": key, "Content-Type": "application/json"},
                       json={"query": query, "numResults": n,
                             "type": "deep" if deep else "auto",
                             "contents": {"text": {"maxCharacters": 800}}})
    except Exception as e:
        return [{"error": f"exa: {e}"}]
    if r.status_code >= 400:
        return [{"error": f"exa HTTP {r.status_code}: {r.text[:200]}"}]
    try:
        data = r.json()
    except Exception:
        return [{"error": "exa: bad JSON"}]
    out: list[dict] = []
    for it in (data.get("results") or [])[:n]:
        out.append({"title": str(it.get("title") or "")[:150],
                    "url": str(it.get("url") or "")[:500],
                    "snippet": str(it.get("text") or it.get("snippet") or "")[:300]})
    return out


def _parallel_search(query: str, n: int, timeout: int) -> list[dict]:
    import httpx, json as _j
    key = _env("PARALLEL_API_KEY")
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "web_search",
                       "arguments": {"objective": query, "search_queries": [query]}}}
    try:
        r = httpx.post("https://search.parallel.ai/mcp", timeout=timeout, headers=headers, json=body)
    except Exception as e:
        return [{"error": f"parallel: {e}"}]
    if r.status_code >= 400:
        return [{"error": f"parallel HTTP {r.status_code}: {r.text[:200]}"}]
    text = r.text
    payload = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("data: "):
            s = s[6:]
        if s.startswith("{"):
            try:
                d = _j.loads(s)
                content = ((d.get("result") or {}).get("content") or [])
                for c in content:
                    if isinstance(c, dict) and c.get("text"):
                        payload += str(c["text"]) + "\n"
            except Exception:
                continue
    if not payload:
        payload = text
    out: list[dict] = []
    for m in re.finditer(r'https?://[^\s"\'<>]+', payload):
        url = m.group(0).rstrip(".,);]")
        if url not in [x.get("url") for x in out]:
            out.append({"title": url[:80], "url": url[:500], "snippet": ""})
        if len(out) >= n:
            break
    if not out and payload.strip():
        out.append({"title": query[:80], "url": "", "snippet": payload.strip()[:1500]})
    return out


def _tavily_search(query: str, n: int, timeout: int, deep: bool) -> list[dict]:
    import httpx
    key = _env("TAVILY_API_KEY")
    if not key:
        return [{"error": "tavily: missing TAVILY_API_KEY"}]
    try:
        r = httpx.post("https://api.tavily.com/search", timeout=timeout,
                       headers={"Content-Type": "application/json"},
                       json={"api_key": key, "query": query, "max_results": n,
                             "search_depth": "advanced" if deep else "basic",
                             "include_answer": False})
    except Exception as e:
        return [{"error": f"tavily: {e}"}]
    if r.status_code >= 400:
        return [{"error": f"tavily HTTP {r.status_code}: {r.text[:200]}"}]
    try:
        data = r.json()
    except Exception:
        return [{"error": "tavily: bad JSON"}]
    out: list[dict] = []
    for it in (data.get("results") or [])[:n]:
        out.append({"title": str(it.get("title") or "")[:150],
                    "url": str(it.get("url") or "")[:500],
                    "snippet": str(it.get("content") or it.get("snippet") or "")[:300]})
    return out


def _serper_search(query: str, n: int, timeout: int) -> list[dict]:
    import httpx
    key = _env("SERPER_API_KEY")
    if not key:
        return [{"error": "serper: missing SERPER_API_KEY"}]
    try:
        r = httpx.post("https://google.serper.dev/search", timeout=timeout,
                       headers={"X-API-KEY": key, "Content-Type": "application/json"},
                       json={"q": query, "num": n})
    except Exception as e:
        return [{"error": f"serper: {e}"}]
    if r.status_code >= 400:
        return [{"error": f"serper HTTP {r.status_code}: {r.text[:200]}"}]
    try:
        data = r.json()
    except Exception:
        return [{"error": "serper: bad JSON"}]
    out: list[dict] = []
    for it in (data.get("organic") or [])[:n]:
        out.append({"title": str(it.get("title") or "")[:150],
                    "url": str(it.get("link") or "")[:500],
                    "snippet": str(it.get("snippet") or "")[:300]})
    return out


def _brave_api_search(query: str, n: int, timeout: int) -> list[dict]:
    import httpx
    key = _env("BRAVE_API_KEY")
    if not key:
        return [{"error": "brave-api: missing BRAVE_API_KEY"}]
    try:
        r = httpx.get("https://api.search.brave.com/res/v1/web/search",
                      params={"q": query, "count": n}, timeout=timeout,
                      headers={"X-Subscription-Token": key, "Accept": "application/json"})
    except Exception as e:
        return [{"error": f"brave-api: {e}"}]
    if r.status_code >= 400:
        return [{"error": f"brave-api HTTP {r.status_code}: {r.text[:200]}"}]
    try:
        data = r.json()
    except Exception:
        return [{"error": "brave-api: bad JSON"}]
    web = data.get("web") or {}
    out: list[dict] = []
    for it in (web.get("results") or [])[:n]:
        out.append({"title": str(it.get("title") or "")[:150],
                    "url": str(it.get("url") or "")[:500],
                    "snippet": str(it.get("description") or "")[:300]})
    return out


def tool(registry: Any | None = None) -> Tool:
    import datetime as _dt
    year = _dt.datetime.now().year
    description = (
        "FIRST for any web info: search the web, then fetch. "
        "Search the web across providers — real-time results beyond knowledge cutoff. "
        f"The current year is {year}: use it for recent queries. "
        "Keyless: duckduckgo/bing/brave work with no keys. "
        "Keyed best: exa/parallel/tavily/serper/brave-api via env keys. "
        "Returns titles+URLs+snippets — fetch top pages with webfetch. "
        "type fast=quick, deep=thorough."
    )

    def run(input: dict) -> dict:
        query = str(input.get("query") or "").strip()
        if not query:
            return {"output": "websearch requires 'query'.", "error": True}
        try:
            n = int(input.get("numResults") or 8)
        except (TypeError, ValueError):
            n = 8
        n = max(1, min(n, 20))
        typ = str(input.get("type") or "auto").strip().lower()
        deep = typ == "deep"
        if typ == "fast":
            n = min(n, 5)
        provider = _pick_provider(str(input.get("provider") or ""))
        try:
            timeout = int(input.get("timeout") or 25)
        except (TypeError, ValueError):
            timeout = 25
        timeout = max(5, min(timeout, 60))
        t0 = time.monotonic()
        if registry is not None:
            try:
                checker = getattr(registry, "interrupt_check", None)
                if callable(checker) and checker():
                    return {"output": "(interrupted)", "error": True, "interrupted": True}
            except Exception:
                pass
        if provider == "exa":
            items = _exa_search(query, n, timeout, deep)
        elif provider == "parallel":
            items = _parallel_search(query, n, timeout)
        elif provider == "tavily":
            items = _tavily_search(query, n, timeout, deep)
        elif provider == "serper":
            items = _serper_search(query, n, timeout)
        elif provider == "brave-api":
            items = _brave_api_search(query, n, timeout)
        elif provider == "bing":
            items = _bing_search(query, n, timeout)
            if not items or (len(items) == 1 and items[0].get("error")):
                items = _ddg_search(query, n, timeout)
                provider = "duckduckgo"
        elif provider == "brave":
            items = _brave_html_search(query, n, timeout)
            if not items or (len(items) == 1 and items[0].get("error")):
                items = _ddg_search(query, n, timeout)
                provider = "duckduckgo"
        else:
            provider = "duckduckgo"
            items = _ddg_search(query, n, timeout)
            if not items or (len(items) == 1 and items[0].get("error")):
                items = _ddg_search(query, n, timeout, lite=True)
        errs = [x.get("error", "") for x in items if x.get("error")]
        items = [x for x in items if not x.get("error")][:n]
        elapsed = round(time.monotonic() - t0, 1)
        if not items:
            msg = "; ".join(errs[:2]) if errs else "no results"
            return {"output": f"No search results for '{query}' ({provider}, {elapsed}s). {msg} Try a different query.",
                    "error": True, "metadata": {"provider": provider, "elapsed_s": elapsed}}
        lines = [f"Web results for '{query}' ({provider}, {len(items)}, {elapsed}s):", ""]
        for i, it in enumerate(items, 1):
            lines.append(f"{i}. {it.get('title') or it.get('url')}")
            if it.get("url"):
                lines.append(f"   {it['url']}")
            if it.get("snippet"):
                lines.append(f"   {it['snippet'][:280]}")
        lines.append("")
        lines.append("Fetch top pages with webfetch for full content.")
        return {"output": "\n".join(lines),
                "metadata": {"provider": provider, "numResults": len(items),
                             "elapsed_s": elapsed,
                             "items": items}}

    return Tool(
        name="websearch",
        description=description,
        parameters=schema_with(
            {
                "query": {"type": "string", "description": "Websearch query (include the year for recent topics)"},
                "numResults": {"type": "integer", "description": "Number of results (default 8, max 20)", "optional": True},
                "type": {"type": "string", "enum": ["auto", "fast", "deep"], "description": "fast=quick, deep=thorough", "optional": True},
                "provider": {"type": "string",
                             "description": "duckduckgo/bing/brave (keyless) or exa/parallel/tavily/serper/brave-api (keyed). auto picks best available.",
                             "optional": True},
                "timeout": {"type": "integer", "description": "Seconds (max 60, default 25)", "optional": True},
            },
            ["query"],
        ),
        run=run,
        permission="websearch",
    )
