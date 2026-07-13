# Lean Karaoke Bot — Strip-Down, Redesign & Free Deploy

**Date:** 2026-07-13
**Status:** Design — pending user review
**Branch target:** new work off `accuracy-upgrade` (or a fresh branch)

## 1. Goal

Reduce the existing Hebrew karaoke Telegram bot to **three features only**, redesign
its UX flow, sync results into an existing Lovable song library, and run it **free,
24/7, independent of the user's PC**.

Keep exactly:
1. **Chords** from the internet (Tab4U lookup by title).
2. **YouTube → MP3** download.
3. **YouTube → video** download (with quality choice).

Remove everything else (transcription, vocal separation, karaoke video rendering,
lyrics verification, subtitle generation, harmony analysis pipeline, job review UI,
delivery approval flow, etc.).

## 2. Decisions (confirmed with user)

| # | Decision |
|---|----------|
| Large video (>50 MB Telegram cap) | Serve a **temporary download link** instead of uploading to Telegram. |
| YouTube datacenter-IP blocking | Use existing **cookies** mechanism (`cookies.txt` / `YTDLP_COOKIE_FILE`). |
| Search results count | **5** results. |
| Chords key | Offer **[Original] / [Easy version]** toggle. |
| Hosting | **Hugging Face Spaces (free CPU)** + external uptime **ping** to defeat idle-pause. No credit card. |
| Song library | Stays in **Lovable** (`all-4-music-guitar`, Supabase). |
| Library sync trigger | Upsert a song **only when chords are found**. |

## 3. Architecture

Two independently-owned pieces, connected by a thin data link:

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  Telegram bot (Python)      │        │  Lovable app (React+Supabase)│
│  runs on HF Spaces          │        │  "Shlomo's Guitar Academy"   │
│                             │        │                              │
│  • search / chords / dl     │        │  • song library UI           │
│  • aiohttp file server      │        │  • songs table (Postgres)    │
│    (port 7860, HTTPS)       │        │                              │
└──────────────┬──────────────┘        └───────────────▲──────────────┘
               │  upsert song (title, artist,           │
               │  original_key, content) via            │
               └────────  Supabase service_role  ───────┘
```

The bot host is the only stateful/always-on component we own. Lovable/Supabase is
untouched except for **row inserts into `public.songs`**.

### 3.1 Why HF Spaces, and the idle-pause mitigation

A YouTube-downloading bot needs a persistent Python process with `yt-dlp` + `ffmpeg`
and real egress. Genuinely-free always-on VMs (Oracle/GCP) require a credit card for
identity verification; the user declined. HF Spaces (free CPU) needs no card, runs
arbitrary Python, and exposes one public **HTTPS** port (7860).

Caveats and how we handle them:
- **Idle-pause (~48 h no HTTP):** an external free cron pinger (e.g. cron-job.org /
  UptimeRobot, no card) hits the Space URL every few minutes to keep it awake.
- **Ephemeral storage:** fine — downloads are temporary and deleted after delivery.
- **Occasional platform restart:** the bot process restarts automatically; brief blip only.
- **Single web port:** the aiohttp **file server binds 7860**, serving both a health
  page (satisfies Space health check) and download links at `/d/<token>`.

## 4. Components

### 4.1 `bot.py` (new, lean — clean rewrite, ~300–400 lines)
Rationale: the current `bot.py` is 2596 lines entangled with the karaoke pipeline.
A clean rewrite is far cleaner than surgical stripping.

Telegram handlers:
- **Message handler** — free text (song name) or a pasted YouTube URL.
- **Search** → `media.search_youtube(query, max_results=5)` → inline keyboard of 5
  results: `title • channel • duration`. Each button carries a short token.
- **Song selected** → message with `[🎸 אקורדים]  [⬇️ הורדת השיר]`.
- **🎸 אקורדים** → `chords.lookup(title)`; render sheet text; buttons `[מקורי][גרסה קלה]`
  to re-render transposed. On success → trigger library sync (§4.4).
  On miss → "לא נמצאו אקורדים".
- **⬇️ הורדת השיר** → `[🎬 וידאו]  [🎵 MP3]`.
  - **🎵 MP3** → `media.download_audio(url)` → send audio → "✅ הועלה בהצלחה" → delete file.
  - **🎬 וידאו** → `[הכי טוב | 1080p | 720p | 480p | 360p]` → `media.download_video(url, q)`:
    - result ≤ 50 MB → send video to Telegram.
    - result  > 50 MB → register with file server → send `http://…/d/<token>` link.
    - → "✅ הועלה בהצלחה" → delete (link files deleted on download or expiry).

State: selected song stored in `context.user_data` keyed by a short token embedded in
each callback (`callback_data`), so concurrent users never collide.

### 4.2 `media.py` (new, extracted from `legacy_media.py`)
Keep only: `search_youtube`, `download_audio` (→ mp3 via ffmpeg), `download_video(quality)`.
Drop everything demucs/vocal/karaoke. Reuse `config.ytdlp_base_opts()` (cookies, JS solver).

### 4.3 `chords.py` (thin wrapper over existing chord path)
Reuse **as-is** (no ML deps confirmed): `chord_sources.lookup_external_chord_sheet_by_title`
+ `harmony` (transpose/key) + `lyrics_verifier` text-search helpers + `models`.
Expose: `lookup(title) -> ChordSheet` and `transpose(sheet, mode)` for original/easy.

### 4.4 `library_sync.py` (new)
On a found chord sheet, **upsert** into Supabase `public.songs`:
- Map: `title`, `artist`, `original_key`, `content`.
- **Format conversion required:** the bot's native renderer emits a `כותרת: …` header
  layout; the library expects **inline `[Chord]`** format (e.g. `[Bm]צמד חמד`). Build a
  converter from the parsed Tab4U sheet (`chord_labels` + `lyric_lines` + positions) →
  inline `[Chord]` text.
- **Dedup:** check by normalized `title_norm` + `artist_norm`; insert if absent, else
  update `content`/`original_key`.
- **Auth:** Supabase **service_role** key (bypasses RLS), supplied as bot secret
  `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`. Writes are best-effort: a sync failure
  logs a warning and never blocks the user's chord/download response.

### 4.5 `file_server.py` (new)
aiohttp app on port 7860:
- `GET /` → tiny health page (keeps Space alive / uptime ping target).
- `GET /d/<token>` → stream the mapped file; delete after successful send; 404 on
  expired/unknown token.
- In-memory token→path map with TTL (~2 h) + periodic sweep of expired entries.

### 4.6 `config.py` (trimmed)
Keep: yt-dlp opts, cookies, ffmpeg discovery, runtime/tmp dirs.
Drop: torch/HF/demucs/transformers cache env wiring.

## 5. Dependencies

`requirements.txt` collapses from 12 heavy libs to:
```
python-telegram-bot
yt-dlp
regex            # chord parsing
aiohttp          # already a PTB dep; file server + Supabase REST calls
```
System binaries on the Space: `ffmpeg`, `node` (yt-dlp EJS challenge solver).

## 6. Cleanup / lifecycle
- Temp files live under a `downloads/` dir; deleted immediately after successful delivery.
- File-server link files deleted on download or TTL expiry.
- Periodic sweep removes orphaned downloads and expired tokens.

## 7. Deployment (HF Spaces)
- Dockerfile or `app.py` entrypoint that installs ffmpeg+node, runs bot polling +
  aiohttp server (port 7860) in one process.
- Secrets: `BOT_TOKEN`, `YTDLP_COOKIE_FILE` (or cookies as a Space secret file),
  `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PUBLIC_BASE_URL` (the Space HTTPS URL).
- External cron ping configured against `PUBLIC_BASE_URL/`.
- README section for the whole setup.

## 8. Out of scope (removed)
Transcription, vocal separation, karaoke/subtitle video rendering, lyrics
verification/consensus, harmony analysis pipeline, job review & delivery-approval UI,
group-delivery flow, and all torch/whisper/demucs/librosa dependencies.

## 9. Open implementation risks
- **YouTube blocking from HF IPs** — mitigated by cookies; may need periodic cookie refresh.
- **Tab4U → inline-`[Chord]` conversion fidelity** — needs its own small test set.
- **HF free-tier limits** (CPU/disk/egress) under heavy video use — acceptable for
  personal use; revisit if it becomes a shared/public bot.
```
