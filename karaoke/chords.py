"""Thin chord-lookup wrapper: fetch a Tab4U sheet and render original/easy."""

from __future__ import annotations

from .chord_image import render_chord_sheet_images
from .chord_sources import (
    _render_external_chord_sheet,
    lookup_external_chord_sheet_by_title,
)
from .models import SongAnalysis


def lookup(title: str) -> SongAnalysis | None:
    """Return a SongAnalysis (with .parsed_sheet attached) or None if not found."""
    return lookup_external_chord_sheet_by_title(title, provider="tab4u")


def _semitones_for_mode(analysis: SongAnalysis, mode: str) -> int:
    return analysis.transpose_semitones if mode == "easy" else 0


def render(analysis: SongAnalysis, title: str, mode: str = "original") -> str:
    """Render the chord sheet as plain text (file fallback for huge sheets).

    mode='original' -> keep the scraped key; mode='easy' -> transpose to the easy key.
    """
    sheet = getattr(analysis, "parsed_sheet", None)
    if sheet is None:
        return analysis.chord_sheet_text
    return _render_external_chord_sheet(
        title,
        sheet,
        bpm=analysis.bpm,
        time_signature=analysis.time_signature,
        original_key=analysis.original_key,
        target_key=analysis.target_key,
        semitones=_semitones_for_mode(analysis, mode),
    )


def render_images(analysis: SongAnalysis, title: str, mode: str = "original") -> list[bytes]:
    """Render the chord sheet as one or more PNG images (bytes).

    This is the Telegram delivery format: each chord label is drawn at the
    pixel x-position of the lyric word it belongs to, so alignment cannot be
    broken by Telegram's font metrics or bidi reordering (which defeated every
    text/whitespace-based approach).
    """
    sheet = getattr(analysis, "parsed_sheet", None)
    if sheet is None:
        return []
    return render_chord_sheet_images(
        sheet,
        title=title,
        original_key=analysis.original_key,
        target_key=analysis.target_key,
        semitones=_semitones_for_mode(analysis, mode),
    )
