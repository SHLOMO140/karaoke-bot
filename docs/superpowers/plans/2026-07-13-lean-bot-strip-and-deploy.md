# Lean Karaoke Bot — Strip-Down, Redesign & Free Deploy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the bot to three features — chords (Tab4U), YouTube→MP3, YouTube→video — with a redesigned Telegram flow, auto-sync of found chords into the Lovable Supabase song library, running free 24/7 on Hugging Face Spaces.

**Architecture:** One always-on Python process on HF Spaces runs Telegram long-polling **plus** an aiohttp server on port 7860 (health page + temporary download links). The heavy ML pipeline is removed; the chord path is decoupled from ML by extracting its HTTP/search helpers into a new lean `web_search` module. Found chords are upserted into Supabase via REST using a service-role key held only as a Space secret.

**Tech Stack:** python-telegram-bot 21.6 (async), yt-dlp, aiohttp, ffmpeg + node (system), Supabase REST, Docker on HF Spaces.

**Spec:** `docs/superpowers/specs/2026-07-13-lean-bot-strip-and-deploy-design.md`

---

## File Structure

**Keep (chord music-theory core, no ML):**
- `karaoke/models.py` — dataclasses (no internal deps)
- `karaoke/exceptions.py` — error types
- `karaoke/harmony.py` — chord/key theory, transpose (deps: exceptions, models)
- `karaoke/chord_sources.py` — Tab4U lookup + parse (repointed to `web_search`)

**Create:**
- `karaoke/web_search.py` — HTTP fetch + Tab4U search helpers extracted from `lyrics_verifier`
- `karaoke/media.py` — `search_youtube`, `download_audio`, `download_video`, `transcode_to_mp3`
- `karaoke/library_sync.py` — inline-`[Chord]` converter + Supabase upsert
- `karaoke/file_server.py` — aiohttp health page + `/d/<token>` link server
- `app.py` — process entrypoint (polling + aiohttp on one loop)
- `Dockerfile` — HF Spaces image (python, ffmpeg, node)
- `README_DEPLOY.md` — HF Spaces + secrets + cron-ping setup

**Rewrite:**
- `bot.py` — lean handlers + UX flow (~350 lines)
- `karaoke/config.py` — trimmed to yt-dlp/cookies/ffmpeg/dirs + Supabase/base-url env
- `requirements.txt` — 4 deps

**Delete (ML + orphaned):**
`karaoke/aligner.py`, `audio_extractor.py` (after extracting transcode), `auto_repair.py`, `char_diff.py`, `consensus.py`, `error_formatter.py`, `google_search.py`, `job_manager.py`, `language_detector.py`, `legacy_media.py` (after extracting search/download), `lyrics_verifier.py`, `pipeline.py`, `providers.py`, `singer_analysis.py`, `styles.py`, `subtitle_generator.py`, `subtitle_guardian.py`, `transcriber.py`, `video_renderer.py`, `vocal_separator.py`, plus their `tests/test_*.py`, plus `tools/run_pipeline_smoke.py`.

---

## Phase 0 — Safety net & branch

### Task 0: Isolate work and capture chord-path baseline

**Files:** none (git + test run)

- [ ] **Step 1: Create a work branch**

```bash
git checkout -b lean-bot-strip
```

- [ ] **Step 2: Record which chord tests currently pass (baseline)**

Run: `python -m pytest tests/test_chord_sources.py tests/test_site_parsers.py tests/test_harmony.py -q`
Expected: note pass/fail counts. These guard the chord path during extraction.

- [ ] **Step 3: Commit a checkpoint marker (docs already committed)**

```bash
git commit --allow-empty -m "chore: start lean-bot-strip"
```

---

## Phase 1 — Decouple chords from ML (`web_search.py`)

Goal: `chord_sources` must import ONLY from a lean module, so `lyrics_verifier`/`consensus`/`google_search`/`aligner`/`char_diff` can be deleted.

### Task 1: Identify the exact helper set chord_sources needs

**Files:** read `karaoke/chord_sources.py:20-30`, `karaoke/lyrics_verifier.py`

Symbols imported by `chord_sources` from `lyrics_verifier`:
`SearchResult, _build_query_variants, _evaluate_candidate_text_against_draft, _extract_title_context, _fetch_text, _normalize_token, _sanitize_internal_site_query, _search_known_site_results, _search_tab4u_results`

- [ ] **Step 1: List transitive helpers**

Run: `grep -nE "^def |^class |^_[A-Z_]+ =" karaoke/lyrics_verifier.py`
Action: mark every function/constant referenced (directly or transitively) by the 9 symbols above. Anything they call inside `lyrics_verifier` must be copied too. Anything that touches Gemini/XAI/consensus/aligner is NOT reachable from these 9 (verify) and must be left behind.

- [ ] **Step 2: Confirm no ML reachability**

Run: `python -c "import ast,sys; src=open('karaoke/lyrics_verifier.py',encoding='utf-8').read(); print('aligner' in src, 'torch' in src)"`
Expected: the align/verify code exists in the file but is NOT called by the 9 helpers (they only fetch HTTP + parse/search text). If any of the 9 transitively reaches `realign_changed_words`/LLM, STOP and narrow the copy set.

### Task 2: Create `web_search.py` with the extracted helpers (test first)

**Files:**
- Create: `karaoke/web_search.py`
- Test: `tests/test_web_search.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web_search.py
from karaoke.web_search import SearchResult, _normalize_token, _build_query_variants


def test_normalize_token_strips_and_lowers_latin():
    assert _normalize_token("  Hello!  ") == "hello"


def test_search_result_is_constructible():
    r = SearchResult(title="t", url="u", snippet="s")
    assert r.url == "u"


def test_build_query_variants_includes_title():
    variants = _build_query_variants("שיר בדיקה", None)
    assert any("שיר בדיקה" in v for v in variants)
```

(Adjust `SearchResult` field names to the real dataclass in `lyrics_verifier`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_search.py -q`
Expected: FAIL — `ModuleNotFoundError: karaoke.web_search`.

- [ ] **Step 3: Create `web_search.py` by moving the reachable helpers**

Copy from `lyrics_verifier.py` into `web_search.py`: the `SearchResult` dataclass, HTTP cache/`_fetch_text`, Tab4U/known-site search (`_search_tab4u_results`, `_search_known_site_results`, `_sanitize_internal_site_query`), query building (`_build_query_variants`, `_extract_title_context`), text normalization (`_normalize_token`), and `_evaluate_candidate_text_against_draft` **plus every private helper/constant they reference**. Keep imports limited to: `re`, `html`, `urllib`, `logging`, stdlib, and `from .config import HTTP_CACHE_DIR, HTTP_CACHE_TTL_SECONDS`. Do NOT import consensus/google_search/models-LLM.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add karaoke/web_search.py tests/test_web_search.py
git commit -m "feat: extract lean web_search helpers from lyrics_verifier"
```

### Task 3: Repoint `chord_sources` to `web_search`

**Files:** Modify `karaoke/chord_sources.py:20-30`

- [ ] **Step 1: Change the import**

Replace `from .lyrics_verifier import (...)` with `from .web_search import (...)` (same symbol list).

- [ ] **Step 2: Run the chord tests (must still pass)**

Run: `python -m pytest tests/test_chord_sources.py tests/test_site_parsers.py -q`
Expected: same pass count as the Phase-0 baseline.

- [ ] **Step 3: Prove ML-free import**

Run: `python -c "import karaoke.chord_sources; import sys; print('torch' in sys.modules, 'whisperx' in sys.modules)"`
Expected: `False False`.

- [ ] **Step 4: Commit**

```bash
git add karaoke/chord_sources.py
git commit -m "refactor: chord_sources imports web_search, decoupled from ML"
```

---

## Phase 2 — Lean media module (`media.py`)

### Task 4: `transcode_to_mp3` + `search_youtube` (test first)

**Files:**
- Create: `karaoke/media.py`
- Test: `tests/test_media.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_media.py
from unittest.mock import patch, MagicMock
from karaoke import media


def test_search_youtube_maps_entries():
    fake_info = {"entries": [
        {"id": "abc123", "title": "Song A", "channel": "Chan", "duration": 65},
    ]}
    fake_ydl = MagicMock()
    fake_ydl.__enter__.return_value.extract_info.return_value = fake_info
    with patch.object(media.yt_dlp, "YoutubeDL", return_value=fake_ydl):
        results = media.search_youtube("q", max_results=5)
    assert results[0]["id"] == "abc123"
    assert results[0]["duration"] == "1:05"
    assert results[0]["url"] == "https://www.youtube.com/watch?v=abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_media.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `media.py` (search + transcode)**

Port `search_youtube` and the mp3 transcode from `legacy_media.py`/`audio_extractor.py` verbatim (they already use `config.ytdlp_base_opts()`), dropping all demucs/vocal code. Keep `transcode_to_mp3(src, dst, title, artist)` using ffmpeg.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_media.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add karaoke/media.py tests/test_media.py
git commit -m "feat: lean media module (search + mp3 transcode)"
```

### Task 5: `download_audio` and `download_video(quality)`

**Files:** Modify `karaoke/media.py`; Test: `tests/test_media.py`

- [ ] **Step 1: Write the failing test (format string per quality)**

```python
def test_video_format_selector_maps_quality():
    assert media._video_format("720") == "bestvideo[height<=720]+bestaudio/best[height<=720]"
    assert media._video_format("best") == "bestvideo+bestaudio/best"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_media.py::test_video_format_selector_maps_quality -q`
Expected: FAIL — `_video_format` missing.

- [ ] **Step 3: Implement download functions**

```python
def _video_format(quality: str) -> str:
    if quality == "best":
        return "bestvideo+bestaudio/best"
    return f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]"


def download_audio(url: str) -> tuple[str, str]:
    """Return (mp3_path, title). Downloads bestaudio then transcodes."""
    ...  # port from legacy_media.download_audio, minus caching-for-karaoke

def download_video(url: str, quality: str) -> tuple[str, str]:
    """Return (mp4_path, title). Uses _video_format(quality) and merges to mp4."""
    opts = {"format": _video_format(quality), "merge_output_format": "mp4",
            "outtmpl": str(DOWNLOAD_DIR / "%(title)s.%(ext)s"), **ytdlp_base_opts()}
    ...
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_media.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add karaoke/media.py tests/test_media.py
git commit -m "feat: media download_audio + download_video(quality)"
```

---

## Phase 3 — Trim `config.py`

### Task 6: Remove ML cache wiring, add deploy env

**Files:** Modify `karaoke/config.py`

- [ ] **Step 1: Delete torch/HF/demucs/transformers/pyannote cache-env block**

Remove the `for env_name, default_path in {...TORCH_HOME...DEMUCS_CACHE_DIR...}` loop and related dirs. Keep `RUNTIME_DIR`, `TMP_DIR`, `YTDLP_STAGING_DIR`, `HTTP_CACHE_DIR`, `HTTP_CACHE_TTL_SECONDS`, `ytdlp_base_opts()`, ffmpeg discovery, `DOWNLOAD_DIR`.

- [ ] **Step 2: Add deploy/config env**

```python
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")           # e.g. https://user-space.hf.space
FILE_SERVER_PORT = int(os.getenv("PORT", "7860"))
TELEGRAM_FILE_LIMIT_BYTES = 50 * 1024 * 1024
LINK_TTL_SECONDS = int(os.getenv("LINK_TTL_SECONDS", str(2 * 3600)))
```

- [ ] **Step 3: Verify config imports clean**

Run: `python -c "import karaoke.config as c; print(bool(c.ytdlp_base_opts()), c.FILE_SERVER_PORT)"`
Expected: prints `True 7860`.

- [ ] **Step 4: Commit**

```bash
git add karaoke/config.py
git commit -m "refactor: trim config to lean bot + add deploy env"
```

---

## Phase 4 — Chords wrapper + inline-[Chord] converter

### Task 7: `chords.lookup` + original/easy transpose

**Files:** Create `karaoke/chords.py`; Test: `tests/test_chords_wrapper.py`

- [ ] **Step 1: Write the failing test (with mocked lookup)**

```python
from unittest.mock import patch
from karaoke import chords
from karaoke.models import SongAnalysis


def test_lookup_returns_none_when_no_sheet():
    with patch("karaoke.chords.lookup_external_chord_sheet_by_title", return_value=None):
        assert chords.lookup("שיר לא קיים") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_chords_wrapper.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement wrapper**

```python
# karaoke/chords.py
from .chord_sources import lookup_external_chord_sheet_by_title
from .harmony import render_chord_sheet_text, EASY_KEY_TARGET
from .models import SongAnalysis

def lookup(title: str) -> SongAnalysis | None:
    return lookup_external_chord_sheet_by_title(title, provider="tab4u")

def render(analysis: SongAnalysis, mode: str = "original") -> str:
    """mode='original' -> as-fetched key; mode='easy' -> transposed to EASY_KEY_TARGET."""
    # reuse analysis.chord_sheet_text for original; call harmony transpose for easy
    ...
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_chords_wrapper.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add karaoke/chords.py tests/test_chords_wrapper.py
git commit -m "feat: chords wrapper with original/easy render"
```

### Task 8: inline-`[Chord]` converter for the library

**Files:** Modify `karaoke/library_sync.py` (create); Test: `tests/test_library_convert.py`

- [ ] **Step 1: Write the failing test using the parsed structure**

```python
# tests/test_library_convert.py
from karaoke.chord_sources import _ChordToken, _LyricWord, _ParsedTab4USheet
from karaoke.library_sync import to_inline_chords


def _sheet():
    # lyric line "אני והיא" with [Dmaj7] at col 0 and [F#m] at col 3
    tokens = [_ChordToken(label="Dmaj7", column=0), _ChordToken(label="F#m", column=3)]
    words = [_LyricWord(text="אני", column=0, global_index=0),
             _LyricWord(text="והיא", column=3, global_index=1)]
    return _ParsedTab4USheet(source_url="u", tables=[], lyric_lines=["אני והיא"],
                             line_word_pairs=[(tokens, words)], chord_labels=["Dmaj7", "F#m"])


def test_inline_places_chords_at_columns():
    out = to_inline_chords(_sheet())
    assert out == "[Dmaj7]אני [F#m]והיא"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_library_convert.py -q`
Expected: FAIL — `library_sync` / `to_inline_chords` missing.

- [ ] **Step 3: Implement the converter**

```python
# karaoke/library_sync.py
from .chord_sources import _ParsedTab4USheet


def to_inline_chords(sheet: _ParsedTab4USheet) -> str:
    """Render a parsed Tab4U sheet as inline [Chord] text for the Lovable library."""
    lines: list[str] = []
    for chord_tokens, lyric_words in sheet.line_word_pairs:
        if lyric_words:
            text = "".join(_reconstruct_line(lyric_words))
            # insert chords by descending column so earlier indices don't shift
            chars = list(text)
            for tok in sorted(chord_tokens, key=lambda t: t.column, reverse=True):
                col = min(tok.column, len(chars))
                chars.insert(col, f"[{tok.label}]")
            lines.append("".join(chars))
        elif chord_tokens:
            lines.append("  |  ".join(f"[{t.label}]" for t in chord_tokens))
        else:
            lines.append("")
    return "\n".join(lines).strip()


def _reconstruct_line(words) -> list[str]:
    """Rebuild the lyric line with words at their column positions."""
    buf: list[str] = []
    for w in sorted(words, key=lambda x: x.column):
        while len(buf) < w.column:
            buf.append(" ")
        buf.extend(list(w.text))
    return buf
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_library_convert.py -q`
Expected: PASS. (If spacing differs, adjust `_reconstruct_line` and re-run against the real "סלינה" sheet as a manual check.)

- [ ] **Step 5: Commit**

```bash
git add karaoke/library_sync.py tests/test_library_convert.py
git commit -m "feat: Tab4U -> inline [Chord] converter"
```

### Task 9: Supabase upsert with dedup

**Files:** Modify `karaoke/library_sync.py`; Test: `tests/test_library_sync.py`

- [ ] **Step 1: Write the failing test (mock aiohttp session)**

```python
# tests/test_library_sync.py
import asyncio
from unittest.mock import AsyncMock, patch
from karaoke import library_sync


def test_norm_collapses_whitespace():
    assert library_sync._norm("  שיר   בדיקה ") == "שיר בדיקה"


def test_upsert_skips_when_no_credentials(monkeypatch):
    monkeypatch.setattr(library_sync, "SUPABASE_URL", "")
    # returns None (no-op) without raising
    assert asyncio.run(library_sync.upsert_song("t", "a", "C", "content")) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_library_sync.py -q`
Expected: FAIL — `_norm`/`upsert_song` missing.

- [ ] **Step 3: Implement upsert (best-effort, never raises to caller)**

```python
import logging
import aiohttp
from .config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

logger = logging.getLogger(__name__)

def _norm(s: str) -> str:
    return " ".join((s or "").split())

async def upsert_song(title, artist, original_key, content) -> str | None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.info("Supabase not configured; skipping library sync")
        return None
    base = SUPABASE_URL.rstrip("/") + "/rest/v1/songs"
    headers = {"apikey": SUPABASE_SERVICE_ROLE_KEY,
               "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
               "Content-Type": "application/json"}
    tn, an = _norm(title), _norm(artist)
    payload = {"title": title, "artist": artist, "original_key": original_key,
               "content": content, "title_norm": tn, "artist_norm": an}
    try:
        async with aiohttp.ClientSession() as s:
            q = f"{base}?title_norm=eq.{tn}&artist_norm=eq.{an}&select=id"
            async with s.get(q, headers=headers) as r:
                rows = await r.json() if r.status == 200 else []
            if rows:
                sid = rows[0]["id"]
                async with s.patch(f"{base}?id=eq.{sid}", headers=headers,
                                   json={"content": content, "original_key": original_key}) as r:
                    r.raise_for_status()
                return sid
            async with s.post(base, headers={**headers, "Prefer": "return=representation"},
                              json=payload) as r:
                r.raise_for_status()
                return (await r.json())[0]["id"]
    except Exception as exc:                       # best-effort: log, never block user
        logger.warning("Library sync failed for %s: %s", title, exc)
        return None
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_library_sync.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add karaoke/library_sync.py tests/test_library_sync.py
git commit -m "feat: best-effort Supabase song upsert with dedup"
```

---

## Phase 5 — File server (`file_server.py`)

### Task 10: token registry + aiohttp routes

**Files:** Create `karaoke/file_server.py`; Test: `tests/test_file_server.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_file_server.py
from karaoke.file_server import LinkRegistry


def test_register_returns_token_and_resolves(tmp_path):
    f = tmp_path / "v.mp4"; f.write_bytes(b"x")
    reg = LinkRegistry(ttl_seconds=100, now=lambda: 0.0)
    token = reg.register(str(f))
    assert reg.resolve(token) == str(f)


def test_expired_token_resolves_none(tmp_path):
    f = tmp_path / "v.mp4"; f.write_bytes(b"x")
    clock = {"t": 0.0}
    reg = LinkRegistry(ttl_seconds=10, now=lambda: clock["t"])
    token = reg.register(str(f))
    clock["t"] = 999
    assert reg.resolve(token) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_file_server.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement registry + app**

```python
# karaoke/file_server.py
import os, secrets, time
from aiohttp import web

class LinkRegistry:
    def __init__(self, ttl_seconds: int, now=time.monotonic):
        self._ttl = ttl_seconds; self._now = now; self._map: dict[str, tuple[str, float]] = {}
    def register(self, path: str) -> str:
        token = secrets.token_urlsafe(16); self._map[token] = (path, self._now()); return token
    def resolve(self, token: str) -> str | None:
        item = self._map.get(token)
        if not item: return None
        path, ts = item
        if self._now() - ts > self._ttl:
            self._map.pop(token, None); return None
        return path
    def sweep(self):
        for tok in [t for t,(p,ts) in self._map.items() if self._now()-ts > self._ttl]:
            path,_ = self._map.pop(tok); 
            try: os.remove(path)
            except OSError: pass

def make_app(registry: LinkRegistry) -> web.Application:
    app = web.Application()
    async def health(_): return web.Response(text="ok")
    async def download(request):
        token = request.match_info["token"]
        path = registry.resolve(token)
        if not path or not os.path.exists(path):
            return web.Response(status=404, text="not found or expired")
        return web.FileResponse(path)
    app.add_routes([web.get("/", health), web.get("/d/{token}", download)])
    return app
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_file_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add karaoke/file_server.py tests/test_file_server.py
git commit -m "feat: aiohttp file server with expiring download links"
```

---

## Phase 6 — Bot rewrite (`bot.py`)

State model: `context.user_data["sel"]` = `{token: {"url","title"}}`; each result button carries a short token. Chords/download flows read the selected entry.

### Task 11: Search handler + result keyboard

**Files:** Rewrite `bot.py`; Test: `tests/test_bot_helpers.py` (replace old)

- [ ] **Step 1: Write failing tests for pure helpers**

```python
# tests/test_bot_helpers.py
import bot

def test_is_youtube_url():
    assert bot.is_youtube_url("https://youtu.be/abc")
    assert not bot.is_youtube_url("שיר של פאר טסי")

def test_result_line_formats():
    line = bot.format_result({"title":"A","channel":"C","duration":"1:05"})
    assert "A" in line and "C" in line and "1:05" in line
```

- [ ] **Step 2: Run to verify fails**

Run: `python -m pytest tests/test_bot_helpers.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement search + selection handlers**

Implement in `bot.py`: `is_youtube_url`, `format_result`, `build_results_keyboard(results, user_data)`, and async `on_message` that runs `media.search_youtube(text, 5)` off-thread (`asyncio.to_thread`) and replies with the keyboard. Selecting a result edits the message to show `[🎸 אקורדים] [⬇️ הורדת השיר]`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_bot_helpers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bot.py tests/test_bot_helpers.py
git commit -m "feat: lean bot search + selection flow"
```

### Task 12: Chords callback + library sync trigger

**Files:** Modify `bot.py`

- [ ] **Step 1: Implement chords handler**

On `chords:<token>`: `analysis = await asyncio.to_thread(chords.lookup, title)`. If None → edit "לא נמצאו אקורדים". Else send `chords.render(analysis, "original")` with `[מקורי][גרסה קלה]` buttons; fire-and-forget `library_sync.upsert_song(title, artist, key, to_inline_chords(sheet))` (only on success). Toggle buttons re-render transposed text via `edit_message_text`.

- [ ] **Step 2: Manual check**

Run the bot locally (Task 16) and verify a known song (e.g. "פאר טסי סלינה") returns chords and appears/updates in Supabase.

- [ ] **Step 3: Commit**

```bash
git add bot.py
git commit -m "feat: chords callback + easy/original toggle + library sync"
```

### Task 13: Download flow (MP3 / video / quality / link / cleanup)

**Files:** Modify `bot.py`

- [ ] **Step 1: Implement download callbacks**

- `dl:<token>` → `[🎬 וידאו][🎵 MP3]`.
- `mp3:<token>` → `path,title = await to_thread(media.download_audio, url)`; `await send_audio`; reply "✅ הועלה בהצלחה"; `os.remove(path)`.
- `vid:<token>` → quality keyboard `[best|1080|720|480|360]`.
- `q:<token>:<quality>` → `path = await to_thread(media.download_video, url, quality)`; if `size<=TELEGRAM_FILE_LIMIT_BYTES` → `send_video` then remove; else `token=registry.register(path)`; send `f"{PUBLIC_BASE_URL}/d/{token}"`; reply "✅ הועלה בהצלחה" (link file removed on download/expiry).

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest -q`
Expected: PASS (only lean tests remain after Phase 7).

- [ ] **Step 3: Commit**

```bash
git add bot.py
git commit -m "feat: download flow with 50MB link fallback + cleanup"
```

---

## Phase 7 — Delete ML modules, tests, and trim requirements

### Task 14: Remove heavy code and dead tests

**Files:** delete per File Structure list

- [ ] **Step 1: Delete modules**

```bash
git rm karaoke/aligner.py karaoke/audio_extractor.py karaoke/auto_repair.py \
  karaoke/char_diff.py karaoke/consensus.py karaoke/error_formatter.py \
  karaoke/google_search.py karaoke/job_manager.py karaoke/language_detector.py \
  karaoke/legacy_media.py karaoke/lyrics_verifier.py karaoke/pipeline.py \
  karaoke/providers.py karaoke/singer_analysis.py karaoke/styles.py \
  karaoke/subtitle_generator.py karaoke/subtitle_guardian.py karaoke/transcriber.py \
  karaoke/video_renderer.py karaoke/vocal_separator.py tools/run_pipeline_smoke.py
```

- [ ] **Step 2: Delete their tests**

```bash
git rm tests/test_aligner*.py tests/test_auto_repair.py tests/test_char_diff.py \
  tests/test_char_timing.py tests/test_consensus*.py tests/test_error_formatter.py \
  tests/test_google_search.py tests/test_grok_provider.py tests/test_harmony_e2e.py \
  tests/test_integration_verification.py tests/test_job_manager.py \
  tests/test_lyrics_*.py tests/test_multi_singer_render.py tests/test_partial_alignment.py \
  tests/test_pipeline_*.py tests/test_post_review_steps.py tests/test_style_rendering.py \
  tests/test_subtitle_generator.py tests/test_transcriber.py tests/test_whisperx_metadata.py \
  tests/test_bot_review_ui.py tests/test_models_new.py tests/test_reset_workspace.py
```

- [ ] **Step 3: Fix `karaoke/__init__.py`**

Remove any imports/exports referencing deleted modules.

- [ ] **Step 4: Verify no dangling imports**

Run: `python -c "import bot"` then `python -m pytest -q`
Expected: import succeeds; suite passes. Fix any `ImportError` by removing the reference.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove ML pipeline modules and dead tests"
```

### Task 15: New `requirements.txt`

**Files:** Rewrite `requirements.txt`

- [ ] **Step 1: Replace contents**

```
python-telegram-bot==21.6
yt-dlp==2026.3.17
aiohttp
regex==2026.2.28
```

- [ ] **Step 2: Fresh-env smoke**

```bash
python -m venv .venv_clean && . .venv_clean/Scripts/activate && pip install -r requirements.txt && python -c "import bot" && deactivate
```
Expected: installs small/fast; `import bot` works with no torch/whisper.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: slim requirements to lean bot deps"
```

---

## Phase 8 — Local run entrypoint

### Task 16: `app.py` — polling + aiohttp on one loop

**Files:** Create `app.py`

- [ ] **Step 1: Implement entrypoint**

```python
# app.py
import asyncio, logging
from aiohttp import web
from telegram.ext import ApplicationBuilder
from karaoke.config import FILE_SERVER_PORT, LINK_TTL_SECONDS, TELEGRAM_BOT_TOKEN
from karaoke.file_server import LinkRegistry, make_app
import bot

logging.basicConfig(level=logging.INFO)

async def main():
    registry = LinkRegistry(ttl_seconds=LINK_TTL_SECONDS)
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    bot.register_handlers(application, registry)          # wire handlers, pass registry
    runner = web.AppRunner(make_app(registry)); await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", FILE_SERVER_PORT).start()
    await application.initialize(); await application.start()
    await application.updater.start_polling()
    try:
        while True:
            await asyncio.sleep(300); registry.sweep()
    finally:
        await application.updater.stop(); await application.stop(); await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run locally against the real bot token**

Run: `python app.py` (with `bot_token.txt`/env + `cookies.txt` present)
Expected: bot answers a search in Telegram; `GET http://localhost:7860/` returns `ok`.

- [ ] **Step 3: Commit**

```bash
git add app.py bot.py
git commit -m "feat: single-process entrypoint (polling + file server)"
```

---

## Phase 9 — HF Spaces deployment

### Task 17: Dockerfile + deploy docs

**Files:** Create `Dockerfile`, `README_DEPLOY.md`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg nodejs && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
```

- [ ] **Step 2: Write `README_DEPLOY.md`**

Document: create a Docker Space; add Space secrets `BOT_TOKEN`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PUBLIC_BASE_URL` (the Space URL), and upload `cookies.txt` as a secret file (set `YTDLP_COOKIE_FILE=/app/cookies.txt`); confirm `TELEGRAM_BOT_TOKEN` is read from `BOT_TOKEN`. Set up a free cron ping (cron-job.org) hitting `PUBLIC_BASE_URL/` every 5 min.

- [ ] **Step 3: Verify token source**

Ensure `config.TELEGRAM_BOT_TOKEN` falls back to `os.getenv("BOT_TOKEN")` then `bot_token.txt`. Adjust if needed.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile README_DEPLOY.md karaoke/config.py
git commit -m "feat: HF Spaces Dockerfile + deploy guide"
```

- [ ] **Step 5: Push and deploy (user provides secrets)**

User action: paste secrets into the Space (never in chat/code), upload cookies, add cron ping. Then smoke-test end to end in Telegram.

---

## Self-Review (completed by author)

**Spec coverage:** chords (Tasks 7–8, 12) ✓; MP3 (Task 5, 13) ✓; video + quality + 50MB link (Task 5, 13, 10) ✓; UX flow (11–13) ✓; delete-after-delivery (13) ✓; library sync only-on-chords (12) ✓; HF Spaces + ping (16–17) ✓; cookies reuse (config/deploy) ✓; ML removal (14–15) ✓.
**Placeholders:** module bodies ported "verbatim from existing" are explicitly named with source; converter/sync/file-server/media selectors have full code. Bot Telegram wiring is described step-wise (hard to unit-test; validated by local run in Task 16/16.2).
**Type consistency:** `LinkRegistry.register/resolve/sweep`, `to_inline_chords(sheet)`, `upsert_song(title,artist,original_key,content)`, `_video_format(quality)`, `chords.lookup/render` used consistently across tasks.
```
