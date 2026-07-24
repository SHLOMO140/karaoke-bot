"""Thin chord-lookup wrapper: fetch a Tab4U sheet and render original/easy."""

from __future__ import annotations

from .chord_sources import (
    _render_external_chord_sheet,
    lookup_external_chord_sheet_by_title,
)
from .models import SongAnalysis


def lookup(title: str) -> SongAnalysis | None:
    """Return a SongAnalysis (with .parsed_sheet attached) or None if not found."""
    return lookup_external_chord_sheet_by_title(title, provider="tab4u")


def render(
    analysis: SongAnalysis, title: str, mode: str = "original", *, for_telegram: bool = False
) -> str:
    """Render the chord sheet as text.

    mode='original' -> keep the scraped key; mode='easy' -> transpose to the easy key.
    for_telegram=True reverses which label sits in which slot on multi-chord
    rows, so they pair with the right word in the Hebrew lyric line Telegram
    renders right-to-left beneath them (see
    chord_sources._reverse_chord_order_in_place).
    """
    sheet = getattr(analysis, "parsed_sheet", None)
    if sheet is None:
        return analysis.chord_sheet_text
    semitones = analysis.transpose_semitones if mode == "easy" else 0
    return _render_external_chord_sheet(
        title,
        sheet,
        bpm=analysis.bpm,
        time_signature=analysis.time_signature,
        original_key=analysis.original_key,
        target_key=analysis.target_key,
        semitones=semitones,
        mirror_chords_for_rtl=for_telegram,
    )
