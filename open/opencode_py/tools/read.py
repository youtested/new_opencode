"""read tool: read a file or directory with line numbers + offset/limit.

RULES (full guidance for maintainers; the sent schema is short):
- mode="auto" (default): files over ~150 lines return a compact OUTLINE
  first (definitions/headings + line numbers). NEVER guess offset/limit
  blindly: target with symbol="Name" (exact def/class block), pattern="re"
  + context=N (matches with surroundings), or the exact lines an outline /
  grep / find_symbols anchor gave you. mode="full" reads raw.
- Search loop: grep queries=[...] fans out (parallel) -> each group header
  carries read offset=/limit=, each def hit carries read symbol= -> ONE read
  lands the body. Max 2 window reads per file.
- Dedup: already-delivered content returns a stub, overlaps trimmed to NEW
  lines; force_full=true resends. Never pay twice for the same lines.
- Line numbers come as `N: content` prefix — never include it in edits.
- Lines over 2000 chars truncated. Batch parallel reads in one message.
- Reads images/PDFs as attachments. Directories list entries (`/` = dir).
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from .context_ledger import mark_delivered, unseen_ranges
from .registry import Tool, schema_with

MAX_LINES = 2000
MAX_CHARS = 2000
MAX_OUTPUT = 50 * 1024  # 50 KB

# Files longer than this many lines return an outline under mode="auto".
AUTO_OUTLINE_LINES = 150

# Low-data preset (Settings > save data): every cap shrinks so a mobile
# package survives. Active ONLY when cfg.low_data is true; the defaults
# above stay byte-identical otherwise.
LOW_MAX_LINES = 50
LOW_AUTO_OUTLINE = 40
LOW_MAX_CHARS = 400
LOW_MAX_OUTPUT = 10 * 1024  # 10 KB
LOW_MAX_FILE_BYTES = 50 * 1024  # default reads of bigger files: outline only
LOW_DIR_MAX = 30


def _is_low(cfg) -> bool:
    """save-data on? Never raises; missing cfg means normal mode."""
    try:
        return bool(getattr(cfg, "low_data", False))
    except Exception:
        return False

# Files up to this size are read whole in one syscall and sliced in memory;
# anything larger streams the window so memory stays bounded for huge files.
# Kept small-ish: on a large file a top-of-file read stops after the window,
# which is cheaper than slurping the whole payload just to slice a few lines.
FAST_READ_BYTES = 256 * 1024

IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg", ".gif", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def _is_binary_sample(sample: bytes) -> bool:
    if not sample:
        return False
    sample = sample[:1024]
    # The fixed window can cut a multibyte UTF-8 char in half (e.g. the box-
    # drawing divider in README.md straddling byte 1024); that truncated tail
    # is "unexpected end of data", not invalid UTF-8. Retry after dropping up
    # to 3 trailing bytes (longest UTF-8 sequence) before declaring binary.
    for trim in range(4):
        try:
            sample[: len(sample) - trim].decode("utf-8")
            break
        except UnicodeDecodeError:
            continue
    else:
        return True
    nonprintable = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return nonprintable / len(sample) > 0.30


def _fuzzy_suggestion(path: Path) -> str | None:
    try:
        candidates = [p for p in path.parent.iterdir() if p.is_file()]
    except OSError:
        return None
    name = path.name
    matches = []
    for p in candidates:
        if p.stem == name or name in p.stem or p.stem in name:
            matches.append(p.name)
    return ", ".join(matches[:3]) or None


def _read_error(path: Path, error: BaseException) -> dict:
    suggestion = _fuzzy_suggestion(path)
    msg = f"Could not read file {path}: {error}"
    if suggestion:
        msg += f"\n\nDid you mean one of these?\n{suggestion}"
    return {"output": msg, "error": True}


def _read_window_lines(
    lines: list[str],
    offset: int,
    limit: int,
    *,
    max_chars: int = MAX_CHARS,
    max_output: int = MAX_OUTPUT,
) -> tuple[list[str], int | None, bool, bool]:
    """Slice a line list into the requested window, capping the output size.

    Shared by the fast (whole-file) and streaming (large-file) read paths so
    both produce byte-identical results. Returns ``(numbered, total,
    reached_eof, truncated_out)`` where ``total`` is the last line index read
    (None if the window started past EOF) and ``numbered`` holds
    ``f"{lineno}: {content}"`` rows.
    """
    numbered: list[str] = []
    total: int | None = None
    truncated_out = False
    out_chars = 0
    start = max(0, offset - 1)
    limit = max(1, int(limit))
    end = start + limit
    last_existing = min(len(lines), end)
    if last_existing < start + 1:
        # window starts past EOF: mirror the streaming path, which reads every
        # line (total = line count) and reports end-of-file
        total = len(lines)
        return [], total, True, False
    total = last_existing
    for lineno in range(start + 1, last_existing + 1):
        content = lines[lineno - 1].rstrip("\n").rstrip("\r")
        if len(content) > max_chars:
            content = content[:max_chars] + f"... (line truncated to {max_chars} chars)"
        numbered.append(f"{lineno}: {content}")
        capped_chars = len(content) + 16
        if out_chars + capped_chars > max_output:
            truncated_out = True
            break
        out_chars += capped_chars
    # reached EOF when the window consumed the file's last line without the
    # output cap cutting it short
    reached_eof = total >= len(lines) and not truncated_out
    return numbered, total, reached_eof, truncated_out


# Outline reads are capped: definition lines live at the top of files, so
# scanning the head is enough — a 50 MB minified bundle no longer gets
# slurped + ast-parsed whole just to list its (nonexistent) structure.
OUTLINE_MAX_BYTES = 512 * 1024


def _outline(path: Path, *, max_output: int = MAX_OUTPUT) -> dict:
    """Structural skeleton of a text file: definition/heading lines with their
    line numbers, NO bodies. A 500 KB file shrinks to a few KB so the model
    can target small offset windows instead of swallowing everything."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(OUTLINE_MAX_BYTES + 1)
    except OSError as e:
        return _read_error(path, e)
    if _is_binary_sample(data):
        return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
    head_truncated = len(data) > OUTLINE_MAX_BYTES
    if head_truncated:
        data = data[:OUTLINE_MAX_BYTES]
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    total = len(lines)
    total_suffix = "+" if head_truncated else ""
    entries: list[str] = []

    def _emit(lineno: int, indent: str) -> None:
        raw = lines[lineno - 1].strip()
        if raw:
            entries.append(f"{indent}{lineno}: {raw[:120]}")

    if path.suffix == ".py" and not head_truncated:
        try:
            from .fast_outline import cached_entries as _fast_entries
            try:
                st_outline = path.stat()
            except OSError as e:
                return _read_error(path, e)
            fast = _fast_entries(path, lines, st_outline)
        except Exception:
            fast = None
        if fast is not None:
            for _ln, _nm in fast:
                try:
                    raw = lines[_ln - 1].strip()
                except IndexError:
                    continue
                if raw:
                    _emit(_ln, "    " if _nm.startswith("    ") else "")
    if path.suffix == ".py" and not entries:
        import ast as _ast

        try:
            tree = _ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            def walk_body(body, indent: str) -> None:
                for node in body:
                    if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                        _emit(node.lineno, indent)
                        if isinstance(node, _ast.ClassDef):
                            walk_body(node.body, indent + "    ")
            walk_body(tree.body, "")
    if not entries:
        # fallback heuristics for non-Python text: headings + common defs
        import re as _re

        pat = _re.compile(r"^(#{1,6}\s+\S|\s{0,8}(?:def |class |func |fn |function |async def ))")
        for i, line in enumerate(lines, 1):
            if pat.match(line):
                entries.append(f"{i}: {line.strip()[:120]}")

    if not entries:
        return {
            "output": (
                f"<{path}>…</{path}>\n<type>outline</type>\n"
                f"({total}{total_suffix} lines; no recognizable structure — use pattern='<re>' + "
                "context=N for matches, or exact offset/limit from grep)"
            ),
            "metadata": {"loaded": [str(path)], "outline": True},
        }
    capped = False
    out_chars = 0
    kept: list[str] = []
    for entry in entries:
        cost = len(entry) + 1
        if out_chars + cost > max_output:
            capped = True
            break
        kept.append(entry)
        out_chars += cost
    body = "\n".join(kept)
    footer = f"(outline: {len(kept)}/{len(entries)} definitions, {total}{total_suffix} lines total"
    if capped:
        footer += ", outline truncated"
    footer += ' — read symbol="<name>" for one block, or exact offset/limit from an anchor line)' 
    return {
        "output": f"<{path}>…</{path}>\n<type>outline</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)], "outline": True},
    }


def _finalize_file_window(
    path: Path,
    st,
    numbered: list[str],
    start_line: int,
    force_full: bool,
    *,
    reached_eof: bool,
    total_all: int | None,
    truncated_out: bool,
    max_output: int = MAX_OUTPUT,
    low_strip: bool = False,
) -> dict:
    """Shared tail for both read paths: dedup-gate the built window against
    the context ledger, trim already-delivered lines, mark what actually
    goes out, and assemble the legacy output format. With low_strip (save-data
    ON only) blank + `#`/`//` comment lines are dropped after the dedup gate
    and counted in the footer; defaults are byte-identical."""
    end_line = start_line + len(numbered) - 1
    by_lineno: dict[int, str] = {}
    for ln in numbered:
        try:
            by_lineno[int(ln.split(":", 1)[0])] = ln
        except ValueError:
            continue

    omitted = 0
    parts: list[str] = []
    new_ranges: list[tuple[int, int]] = []
    if not numbered:
        unseen: list[tuple[int, int]] = []
    elif force_full:
        unseen = [(start_line, end_line)]
    else:
        unseen = unseen_ranges(str(path), st.st_mtime_ns, st.st_size, start_line, end_line)

    if numbered and not unseen and not force_full:
        return {
            "output": (
                f"<{path}>…</{path}>\n<type>file</type>\n"
                f"(lines {start_line}-{end_line}: identical content was already delivered "
                "to you earlier this session — nothing new here, the file is unchanged on disk)\n"
                '(Re-request with force_full=true only if you genuinely need it again.)'
            ),
            "metadata": {"loaded": [str(path)], "dedup_stub": True},
        }

    cursor = start_line
    last_real = start_line - 1
    for a, b in unseen:
        if a > cursor:
            skipped = a - cursor
            parts.append(f"… lines {cursor}-{a - 1}: unchanged, shown earlier ({skipped} lines omitted)")
            omitted += skipped
        for lineno in range(a, min(b, end_line) + 1):
            ln = by_lineno.get(lineno)
            if ln is None:
                continue
            parts.append(ln)
            last_real = lineno
        new_ranges.append((a, b))
        cursor = b + 1
    if cursor <= end_line:
        skipped = end_line - cursor + 1
        parts.append(f"… lines {cursor}-{end_line}: unchanged, shown earlier ({skipped} lines omitted)")
        omitted += skipped

    # honour the output cap AFTER dedup-trim; drop any new_range whose lines
    # were cut so the ledger never records content that did not go out.
    body_parts: list[str] = []
    out_chars = 0
    kept_ranges: list[tuple[int, int]] = []
    cur: list[int] = []
    stripped = 0

    def flush_cur():
        if cur:
            kept_ranges.append((cur[0], cur[-1]))
            cur.clear()

    for part in parts:
        is_real = not part.startswith("… lines ")
        cost = len(part) + 1
        if low_strip and is_real:
            try:
                _body = part.split(":", 1)[1].strip()
            except (IndexError, ValueError):
                _body = None
            if _body is not None and (not _body or _body.startswith("#") or _body.startswith("//")):
                # trivially re-derivable: count as delivered, don't send bytes.
                stripped += 1
                try:
                    _sl = int(part.split(":", 1)[0])
                except ValueError:
                    continue
                if cur and _sl == cur[-1] + 1:
                    cur.append(_sl)
                else:
                    flush_cur()
                    cur.append(_sl)
                continue
        if out_chars + cost > max_output:
            truncated_out = True
            break
        body_parts.append(part)
        out_chars += cost
        if is_real:
            try:
                lineno = int(part.split(":", 1)[0])
            except ValueError:
                continue
            if cur and lineno == cur[-1] + 1:
                cur.append(lineno)
            else:
                flush_cur()
                cur.append(lineno)
    flush_cur()

    for s, e in new_ranges:
        if (s, e) in kept_ranges or any(s >= ks and e <= ke for ks, ke in kept_ranges):
            mark_delivered(str(path), st.st_mtime_ns, st.st_size, s, e)
        else:
            # partial survival after the cap: mark only the kept prefix
            for ks, ke in kept_ranges:
                if ks <= s <= e <= ke:
                    continue
            overlap = max(0, min(e, max(ke for _, ke in kept_ranges)) - s + 1) if kept_ranges else 0
            if overlap:
                mark_delivered(str(path), st.st_mtime_ns, st.st_size, s, s + overlap - 1)

    body = "\n".join(body_parts)

    if reached_eof and total_all is not None:
        footer = f"(End of file - total {total_all} lines)"
    else:
        footer = (f"(Showing line {start_line}-{last_real}. Prefer read symbol='<name>' / "
                  f"pattern='<re>' over paging; else offset={last_real + 1} to continue.)")
    if truncated_out:
        footer += f" (Output capped at {max_output // 1024} KB.)"
    if omitted:
        footer += f" Deduped: {omitted} previously-sent lines omitted."
    if stripped:
        footer += f" Save-data: {stripped} blank/comment lines removed."

    return {
        "output": f"<{path}>…</{path}>\n<type>file</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _read_fast(
    path: Path,
    offset: int,
    limit: int,
    st=None,
    mode: str = "auto",
    force_full: bool = False,
    cfg=None,
    default_req: bool = False,
) -> dict:
    """Read a small file in one shot: fewer syscalls and no per-line IO loop.

    Used for files up to ~2 MB; the whole payload is decoded at once and the
    requested window is sliced from the line list. Universal-newline semantics
    (CRLF / lone-CR folding) are mirrored so line numbers match the streaming
    path exactly. Bigger files still use the streaming path so memory stays
    bounded.
    """
    try:
        with path.open("rb") as f:
            data = f.read()
    except OSError as e:
        return _read_error(path, e)
    if _is_binary_sample(data):
        return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
    # newline=None (universal) in the streaming path folds \r\n and lone \r to
    # \n before splitting into lines — mirror that here. splitlines(keepends)
    # yields exactly the same items the TextIOWrapper iterator would (each line
    # with its terminator, or a final unterminated line; nothing for an empty
    # file), so line counts match the streaming path for CRLF too.
    text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines(keepends=True)

    low = _is_low(cfg)
    auto_at = LOW_AUTO_OUTLINE if low else AUTO_OUTLINE_LINES
    # Big files return the skeleton on DEFAULT requests. In normal mode the
    # rule is exactly the legacy one (offset==1 and limit>=threshold); in
    # save-data mode an explicit small window (offset/limit) is always
    # honored literally so 50-line paging works (default_req=False).
    if low:
        _want_outline = (
            mode == "auto"
            and not force_full
            and int(offset) == 1
            and default_req
            and len(lines) > auto_at
        )
    else:
        _want_outline = (
            mode == "auto"
            and not force_full
            and int(offset) == 1
            and int(limit) >= auto_at
            and len(lines) > auto_at
        )
    if _want_outline:
        return _outline(path, max_output=LOW_MAX_OUTPUT if low else MAX_OUTPUT)

    numbered, total, reached_eof, truncated_out = _read_window_lines(
        lines, max(1, int(offset)), limit,
        max_chars=LOW_MAX_CHARS if low else MAX_CHARS,
        max_output=LOW_MAX_OUTPUT if low else MAX_OUTPUT,
    )

    return _finalize_file_window(
        path,
        st,
        numbered,
        max(1, int(offset)),
        force_full,
        reached_eof=reached_eof and not truncated_out,
        total_all=len(lines),
        truncated_out=truncated_out,
        max_output=LOW_MAX_OUTPUT if low else MAX_OUTPUT,
        low_strip=low,
    )


def _read_file(
    path: Path,
    offset: int = 1,
    limit: int = MAX_LINES,
    mode: str = "auto",
    force_full: bool = False,
    cfg=None,
    default_req: bool = False,
) -> dict:
    # image / pdf -> base64 file attachment
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS or suffix in PDF_EXTENSIONS:
        try:
            data = path.read_bytes()
        except OSError as e:
            return _read_error(path, e)
        mime, _ = mimetypes.guess_type(str(path))
        b64 = base64.b64encode(data).decode()
        return {
            "output": f"<{path.name}><type>{mime}</type><content>data:{mime};base64,{b64}</content>",
            "metadata": {"loaded": [str(path)], "mime": mime},
        }
    if mode == "outline":
        return _outline(path, max_output=LOW_MAX_OUTPUT if _is_low(cfg) else MAX_OUTPUT)

    # Whole-file fast path for ordinary-sized files; larger files stream so we
    # never hold a huge payload in memory just to read a window of it.
    try:
        st = path.stat()
    except OSError as e:
        return _read_error(path, e)
    low = _is_low(cfg)
    if (
        low
        and mode == "auto"
        and not force_full
        and int(offset) == 1
        and default_req
        and st.st_size > LOW_MAX_FILE_BYTES
    ):
        # save-data ON: a DEFAULT read of a big file returns the skeleton,
        # not the body — use symbol=/pattern= (or an anchored offset/limit).
        # Explicit windows (default_req=False) always stream literally.
        return _outline(path, max_output=LOW_MAX_OUTPUT)
    if st.st_size <= FAST_READ_BYTES:
        return _read_fast(path, offset, limit, st=st, mode=mode,
                          force_full=force_full, cfg=cfg, default_req=default_req)

    # binary detection + text window share ONE open: the old code opened the
    # file once for the binary sample and again for the text window, so a file
    # modified between the two opens could be torn (or the second open could
    # hit a different inode / fail on a deleted path). Seek back after the
    # sample and read the window from the same handle.
    start = max(0, offset - 1)
    limit = max(1, int(limit))
    end = start + limit

    # stream the window line-by-line, holding only the selected lines in memory
    numbered: list[str] = []
    total: int | None = None
    reached_eof = True
    out_chars = 0
    truncated_out = False
    _mc = LOW_MAX_CHARS if low else MAX_CHARS
    _mo = LOW_MAX_OUTPUT if low else MAX_OUTPUT
    try:
        with path.open("rb") as f:
            if _is_binary_sample(f.read(1024)):
                return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
            f.seek(0)
            # newline=None so universal-newline splitting matches the old
            # text-mode open exactly (a CRLF / lone-CR file is not treated as
            # one giant line).
            stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline=None)
            try:
                for lineno, line in enumerate(stream, 1):
                    if lineno > end:
                        reached_eof = False
                        break
                    total = lineno
                    if lineno <= start:
                        continue
                    content = line.rstrip("\n").rstrip("\r")
                    if len(content) > _mc:
                        content = content[:_mc] + f"... (line truncated to {_mc} chars)"
                    numbered.append(f"{lineno}: {content}")
                    capped_chars = (len(content) + 16)
                    if out_chars + capped_chars > _mo:
                        truncated_out = True
                        reached_eof = False
                        break
                    out_chars += capped_chars
            finally:
                # keep the wrapper from closing/stealing the raw buffer when it
                # goes out of scope (the `with` on `f` manages fd lifetime).
                stream.detach()
    except OSError as e:
        return _read_error(path, e)

    return _finalize_file_window(
        path,
        st,
        numbered,
        start + 1,
        force_full,
        reached_eof=reached_eof and not truncated_out,
        total_all=total if (reached_eof and total is not None) else None,
        truncated_out=truncated_out,
        max_output=_mo,
        low_strip=low,
    )


def _read_directory(path: Path, offset: int = 1, limit: int = MAX_LINES, cfg=None) -> dict:
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return {"output": f"Could not read directory {path}: {e}", "error": True}
    if _is_low(cfg):
        # save-data ON: directory dumps cap at 30 entries per call — page with offset.
        limit = max(1, min(int(limit), LOW_DIR_MAX))
    total = len(entries)
    start = max(0, offset - 1)
    selected = entries[start : start + limit]
    lines = [(f"{p.name}/" if p.is_dir() else p.name) for p in selected]
    body = "\n".join(lines)
    if start == 0 and total <= limit:
        footer = f"(Showing {total} entries)"
    else:
        end = min(start + limit, total)
        footer = f"(Showing {start + 1}-{end} of {total} entries. Use offset={end + 1} to continue.)"
    return {
        "output": f"<{path}>…</{path}>\n<type>directory</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _symbol_span(lines: list[str], path: Path, symbol: str, context: int = 0) -> tuple[int, int, str] | dict:
    """Exact (start, end) 1-indexed lines of `symbol`'s block, padded by context.

    Header found by name (Python: AST with dotted scope; others: declaration
    patterns); the block end comes from index/spans.py (brace scan for
    brace languages, indent scan otherwise, AST for Python). Returns an
    error dict with close-name suggestions when nothing matches.
    """
    from ..index.spans import block_span as _span
    from ..index.model import language_for as _lang_for
    name = (symbol or "").strip().strip(chr(34) + chr(39) + chr(96))
    if not name:
        return {"output": f"No symbol given for {path}.", "error": True}
    want = name.split(".")[-1]
    total = len(lines)
    pad = max(0, int(context or 0))
    lang = ""
    try:
        lang = _lang_for(str(path)) or ""
    except Exception:
        lang = ""
    if not lang:
        lang = "python" if path.suffix == ".py" else ""
    if lang == "python":
        import ast as _ast
        try:
            tree = _ast.parse("\n".join(lines))
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            best = None
            names: list[str] = []

            def walk(node, scope: str = "") -> None:
                nonlocal best
                for child in _ast.iter_child_nodes(node):
                    if isinstance(child, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                        dotted = f"{scope}.{child.name}" if scope else child.name
                        names.append(dotted)
                        if child.name == want or dotted == name or dotted.endswith("." + name):
                            span = (child.lineno, child.end_lineno or child.lineno)
                            if best is None or (span[0] >= best[0] and span[1] <= best[1]):
                                best = span
                        walk(child, dotted)
                    elif isinstance(child, (_ast.Assign, _ast.AnnAssign)):
                        targets = []
                        if isinstance(child, _ast.Assign):
                            targets = list(child.targets)
                        else:
                            targets = [child.target]
                        for tgt in targets:
                            if isinstance(tgt, _ast.Name):
                                names.append(tgt.id)
                                if tgt.id == want:
                                    span = (child.lineno, child.end_lineno or child.lineno)
                                    if best is None:
                                        best = span
            walk(tree)
            if best is not None:
                s, e = best
                return max(1, s - pad), min(total, e + pad), f"{path}:{s}-{e}"
            import difflib as _dl
            close = _dl.get_close_matches(want, sorted(set(names)), n=5, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else " Try mode=outline for exact names."
            return {"output": f"No symbol `{name}` in {path}.{hint}", "error": True}
    import re as _re
    try:
        from ..index.heuristic_indexer import rule_for as _rule_for, _clean_line as _clean
        rule = _rule_for(lang)
        hits = []
        for i, ln in enumerate(lines, 1):
            c = _clean(ln, rule)
            for _kind, _pat in rule.decls:
                try:
                    m = _pat.match(c)
                except Exception:
                    m = None
                if m:
                    try:
                        nm = m.group("name")
                    except Exception:
                        nm = None
                    if nm == want or (nm or "").endswith("::" + want) or (nm or "").endswith("." + want):
                        hits.append(i)
                        break
            else:
                continue
    except Exception:
        hits = []
    if not hits and _re.search(r"\.[ch](?:pp)?$", str(path).lower()) or (not hits and lang == "c"):
        # C anonymous struct: name trails at the close (`} Point;`)
        try:
            _m = _re.search(r"^\s*}\s*" + _re.escape(want) + r"\s*;", "\n".join(lines), _re.M)
            if _m:
                _ln = "\n".join(lines)[:_m.start()].count("\n") + 1
                _open = None
                for _i in range(_ln, 0, -1):
                    if _re.match(r"^\s*(?:typedef\s+)?struct\s*(\{)?\s*$", lines[_i - 1]) or "struct" in lines[_i - 1] and "{" in lines[_i - 1]:
                        _open = _i
                        break
                if _open is not None:
                    hits = [_open]
        except Exception:
            pass
    if not hits:
        # last resort: bare `name(` method shorthand (JS getName(), C++ Foo::bar)
        pat2 = _re.compile(r"(?:^|[\s{;])(?:" + _re.escape(want) + r"\s*\(|\w+::" + _re.escape(want) + r"\s*\()")
        hits = [i for i, ln in enumerate(lines, 1) if pat2.search(ln)]
    if not hits:
        import difflib as _dl
        try:
            cands = sorted({str(m.group("name")) for _k, _p in rule.decls for m in
                            (_p.match(_clean(ln, rule)) for ln in lines) if m} - {""})
        except Exception:
            cands = []
        close = _dl.get_close_matches(want, cands, n=5, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else " Try mode=outline for exact names."
        return {"output": f"No symbol `{name}` in {path}.{hint}", "error": True}
    s = hits[0]
    try:
        _, e = _span(lines, s, lang)
    except Exception:
        e = s
    return max(1, s - pad), min(total, e + pad), f"{path}:{s}-{e}"


def _read_symbol(path: Path, symbol: str, context: int, mode: str, force_full: bool, cfg, _limit: int = 0) -> dict:
    """One exact def/class block — no offset guessing. Never raises."""
    try:
        with path.open("rb") as f:
            head = f.read(1024)
        if _is_binary_sample(head):
            return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return _read_error(path, e)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    span = _symbol_span(lines, path, symbol, context)
    if isinstance(span, dict):
        return span
    s, e, _label = span
    span_len = e - s + 1
    out = _read_file(path, offset=s, limit=span_len, mode="full",
                     force_full=force_full, cfg=cfg, default_req=False)
    try:
        out.setdefault("metadata", {})["symbol"] = symbol
    except Exception:
        pass
    return out


def _read_pattern(path: Path, pattern: str, context: int, mode: str, force_full: bool, cfg) -> dict:
    """Regex matches with `context` surroundings, merged — no blind paging.

    Disjoint hit groups stay separate sections; each goes through the normal
    window pipeline (caps + dedup ledger) so nothing is ever sent twice.
    """
    import re as _re
    try:
        rx = _re.compile(pattern)
    except _re.error as e:
        return {"output": f"Invalid regex `{pattern}`: {e}", "error": True}
    try:
        with path.open("rb") as f:
            head = f.read(1024)
        if _is_binary_sample(head):
            return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
        st = path.stat()
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return _read_error(path, e)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    total = len(lines)
    pad = max(0, int(context if context else 3))
    hits = [i for i, ln in enumerate(lines, 1) if rx.search(ln)]
    if not hits:
        return {"output": f"No matches for `{pattern}` in {path}."}
    spans: list[list[int]] = []
    for h in hits:
        s, e = max(1, h - pad), min(total, h + pad)
        if spans and s <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    low = _is_low(cfg)
    max_out = LOW_MAX_OUTPUT if low else MAX_OUTPUT
    if len(spans) == 1:
        out = _read_file(path, offset=spans[0][0], limit=spans[0][1] - spans[0][0] + 1,
                         mode="full",
                         force_full=force_full, cfg=cfg, default_req=False)
        try:
            out.setdefault("metadata", {})["pattern"] = pattern
        except Exception:
            pass
        return out
    parts: list[str] = []
    out_chars = 0
    truncated = False
    shown = 0
    for s, e in spans:
        numbered, _t, _eof, _tr = _read_window_lines(lines, s, e - s + 1)
        block = "\n".join(numbered)
        head_ln = f"--- matches around line {s}-{e} ---"
        cost = len(head_ln) + 1 + len(block) + 1
        if out_chars + cost > max_out and shown:
            truncated = True
            break
        parts.append(head_ln + "\n" + block)
        out_chars += cost
        shown += 1
        for a, b in unseen_ranges(str(path), st.st_mtime_ns, st.st_size, s, e):
            mark_delivered(str(path), st.st_mtime_ns, st.st_size, a, b)
    body = "\n".join(parts)
    footer = f"(pattern `{pattern}`: {len(hits)} matches in {len(spans)} groups, showed {shown}"
    if truncated:
        footer += f", capped at {max_out // 1024} KB — raise context or narrow the pattern"
    footer += ")"
    return {"output": f"<{path}>…</{path}>\n<type>file</type>\n<content>\n{body}\n</content>\n{footer}",
            "metadata": {"loaded": [str(path)], "pattern": pattern, "matches": len(hits)}}


def _read(
    filePath: str,
    offset: int = 1,
    limit: int = MAX_LINES,
    mode: str = "auto",
    force_full: bool = False,
    cfg=None,
    default_req: bool = False,
    symbol: str | None = None,
    pattern: str | None = None,
    context: int = 0,
) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        suggestion = _fuzzy_suggestion(path)
        msg = f"Path {path} does not exist."
        if suggestion:
            msg += f"\n\nDid you mean one of these?\n{suggestion}"
        return {"output": msg, "error": True}
    if path.is_dir():
        return _read_directory(path, offset=offset, limit=limit, cfg=cfg)
    if symbol:
        return _read_symbol(path, symbol, context, mode or "auto", force_full, cfg)
    if pattern:
        return _read_pattern(path, pattern, context, mode or "auto", force_full, cfg)
    if _is_low(cfg):
        # save-data ON: one call returns at most 50 lines — page with offset.
        limit = max(1, min(int(limit), LOW_MAX_LINES))
    return _read_file(path, offset=offset, limit=limit, mode=mode,
                      force_full=force_full, cfg=cfg, default_req=default_req)


def tool(cfg=None) -> Tool:
    if _is_low(cfg):
        description = (
            "Read a file/dir with line numbers (SAVE-DATA: max 50 lines/call, "
            "outlines for big files, blanks/comments trimmed). "
            "NEVER guess offset/limit: use symbol='Name' for one def/class, "
            "pattern='re' + context=N for matches, or lsp/find_symbols/grep for an "
            "anchor first. Re-reads return only new lines."
        )
    else:
        description = (
            "Read a file/dir with line numbers. Big files give an outline first. "
            "NEVER guess offset/limit blindly: use symbol='FuncName' for one exact "
            "def/class block, pattern='regex' + context=N for matches with surroundings, "
            "or lsp (first for code) / find_symbols / grep / summarize_file for an anchor first. "
            "Max 2 window reads per file — re-search instead of paging. "
            "Re-reads return only new lines."
        )

    def run(input: dict) -> dict:
        return _read(
            input["filePath"],
            offset=int(input.get("offset") or 1),
            limit=int(input.get("limit") or MAX_LINES),
            mode=input.get("mode") or "auto",
            force_full=bool(input.get("force_full")),
            cfg=cfg,
            default_req=input.get("offset") is None and input.get("limit") is None
            and not input.get("symbol") and not input.get("pattern"),
            symbol=input.get("symbol"),
            pattern=input.get("pattern"),
            context=int(input.get("context") or 0),
        )

    return Tool(
        name="read",
        description=description,
        parameters=schema_with(
            {
                "filePath": {"type": "string", "description": "Absolute path to file or dir"},
                "offset": {"type": "integer", "description": "Start line (1-indexed). Use ONLY with an anchor line from outline/grep/find_symbols — never guess", "optional": True},
                "limit": {"type": "integer", "description": "Lines to read", "optional": True},
                "mode": {
                    "type": "string",
                    "enum": ["auto", "full", "outline"],
                    "description": "auto (default), full, or outline",
                    "optional": True,
                },
                "force_full": {
                    "type": "boolean",
                    "description": "Resend already-delivered content",
                    "optional": True,
                },
                "symbol": {
                    "type": "string",
                    "description": "Exact def/class/var name (dotted 'Class.method' ok) — returns that whole block, no offset math",
                    "optional": True,
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex — returns match groups with surroundings, no offset math",
                    "optional": True,
                },
                "context": {
                    "type": "integer",
                    "description": "Surrounding lines for symbol/pattern (default 0 / 3)",
                    "optional": True,
                },
            },
            ["filePath"],
        ),
        run=run,
        permission="read",
    )
