"""Sync found chord sheets into the Lovable Supabase song library.

Renders a parsed Tab4U sheet into the library's inline `[Chord]` format and
upserts it (dedup by normalized title+artist). All network work is best-effort:
failures are logged and never propagate to the user-facing chord/download flow.
"""

from __future__ import annotations

import logging
import os

import aiohttp

from .config import SUPABASE_URL as _CONFIG_SUPABASE_URL

logger = logging.getLogger(__name__)

# The Lovable-managed Supabase belongs to Lovable's org, so the service_role
# secret isn't obtainable from outside. Instead the bot calls a token-gated
# SECURITY DEFINER function (public.bot_upsert_song) using the PUBLIC publishable
# key; the shared SUPABASE_SYNC_TOKEN (only the bot knows it) is what authorizes
# the write. The URL and publishable key are public (already exposed in the
# Lovable web app's bundle), so they carry safe defaults; only the token is a
# secret and MUST come from the environment — never commit it to this repo.
SUPABASE_URL = _CONFIG_SUPABASE_URL or os.getenv(
    "SUPABASE_URL", "https://fsbbvesdhfeepburvssa.supabase.co"
)
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY", "sb_publishable_VUi2ZOW2aPaxkSUqNC8v-A_2y6-DASJ"
)
SUPABASE_SYNC_TOKEN = os.getenv("SUPABASE_SYNC_TOKEN", "")


# --------------------------------------------------------------------------- #
# Rendering: parsed sheet -> inline [Chord] text (library format)
# --------------------------------------------------------------------------- #
def _reconstruct_line(words) -> list[str]:
    """Rebuild a lyric line as a char list with words at their column positions."""
    buf: list[str] = []
    for w in sorted(words, key=lambda x: x.column):
        while len(buf) < w.column:
            buf.append(" ")
        buf.extend(list(w.text))
    return buf


def to_inline_chords(sheet) -> str:
    """Render a parsed Tab4U sheet as inline [Chord] text for the library."""
    lines: list[str] = []
    for chord_tokens, lyric_words in sheet.line_word_pairs:
        if lyric_words:
            chars = _reconstruct_line(lyric_words)
            for tok in sorted(chord_tokens, key=lambda t: t.column, reverse=True):
                col = min(tok.column, len(chars))
                chars.insert(col, f"[{tok.label}]")
            lines.append("".join(chars).rstrip())
        elif chord_tokens:
            lines.append("  |  ".join(f"[{t.label}]" for t in chord_tokens))
        else:
            lines.append("")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Supabase upsert via token-gated RPC (best-effort)
# --------------------------------------------------------------------------- #
async def upsert_song(title: str, artist: str, original_key: str, content: str) -> str | None:
    """Insert or update a song row via the bot_upsert_song RPC. Returns the row
    id, or None on skip/failure. Dedup (by lower/trim title+artist) and the
    insert-vs-update decision happen inside the SQL function."""
    if not (SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_SYNC_TOKEN):
        logger.info("Supabase sync not configured; skipping library sync for %s", title)
        return None
    endpoint = SUPABASE_URL.rstrip("/") + "/rest/v1/rpc/bot_upsert_song"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "p_token": SUPABASE_SYNC_TOKEN,
        "p_title": title,
        "p_artist": artist or "",
        "p_key": original_key or "",
        "p_content": content or "",
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(endpoint, headers=headers, json=payload) as r:
                r.raise_for_status()
                sid = await r.json()
        logger.info("Library sync: upserted %s (%s)", title, sid)
        return sid
    except Exception as exc:  # noqa: BLE001 - best-effort, never block the user
        logger.warning("Library sync failed for %s: %s", title, exc)
        return None
