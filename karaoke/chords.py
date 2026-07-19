"""Thin chord-lookup wrapper: fetch a Tab4U sheet and render original/easy."""

from __future__ import annotations

from . import library_sync
from .chord_sources import (
    _render_external_chord_sheet,
    lookup_external_chord_sheet_by_title,
)
from .models import SongAnalysis


def lookup(title: str) -> SongAnalysis | None:
    """Return a SongAnalysis (with .parsed_sheet attached) or None if not found."""
    return lookup_external_chord_sheet_by_title(title, provider="tab4u")


def render_inline(analysis: SongAnalysis, title: str, mode: str = "original") -> str:
    """Render the sheet as inline `[Chord]word` text for Telegram.

    Unlike the column layout (chords on a line above the lyrics), inline chords
    stay glued to their word, so Hebrew (RTL) + Latin chord labels (LTR) no longer
    scramble under Telegram's bidirectional text layout. mode='easy' transposes
    to the easy key.
    """
    sheet = getattr(analysis, "parsed_sheet", None)
    if sheet is None:
        return analysis.chord_sheet_text
    semitones = analysis.transpose_semitones if mode == "easy" else 0
    key = (analysis.target_key if mode == "easy" else analysis.original_key) or ""
    body = library_sync.to_inline_chords(sheet, semitones)
    header = f"🎸 {title}".strip()
    if key:
        header += f"  ·  סולם: {key}"
    return f"{header}\n\n{body}" if body else header


def render(analysis: SongAnalysis, title: str, mode: str = "original") -> str:
    """Render the chord sheet as text.

    mode='original' -> keep the scraped key; mode='easy' -> transpose to the easy key.
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
    )
