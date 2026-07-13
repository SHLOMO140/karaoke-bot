"""Sync found chord sheets into the Lovable Supabase song library.

Renders a parsed Tab4U sheet into the library's inline `[Chord]` format and
upserts it (dedup by normalized title+artist). All network work is best-effort:
failures are logged and never propagate to the user-facing chord/download flow.
"""

from __future__ import annotations

import logging

import aiohttp

from .config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)


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
# Supabase upsert (best-effort)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return " ".join((s or "").split())


async def upsert_song(title: str, artist: str, original_key: str, content: str) -> str | None:
    """Insert or update a song row in Supabase. Returns row id, or None on skip/failure."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.info("Supabase not configured; skipping library sync for %s", title)
        return None
    base = SUPABASE_URL.rstrip("/") + "/rest/v1/songs"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    tn, an = _norm(title), _norm(artist)
    row = {
        "title": title,
        "artist": artist,
        "original_key": original_key,
        "content": content,
        "title_norm": tn,
        "artist_norm": an,
    }
    try:
        async with aiohttp.ClientSession() as s:
            params = {"title_norm": f"eq.{tn}", "artist_norm": f"eq.{an}", "select": "id"}
            async with s.get(base, headers=headers, params=params) as r:
                existing = await r.json() if r.status == 200 else []
            if existing:
                sid = existing[0]["id"]
                async with s.patch(
                    base, headers=headers, params={"id": f"eq.{sid}"},
                    json={"content": content, "original_key": original_key},
                ) as r:
                    r.raise_for_status()
                logger.info("Library sync: updated %s (%s)", title, sid)
                return sid
            async with s.post(
                base, headers={**headers, "Prefer": "return=representation"}, json=row,
            ) as r:
                r.raise_for_status()
                sid = (await r.json())[0]["id"]
            logger.info("Library sync: inserted %s (%s)", title, sid)
            return sid
    except Exception as exc:  # noqa: BLE001 - best-effort, never block the user
        logger.warning("Library sync failed for %s: %s", title, exc)
        return None
