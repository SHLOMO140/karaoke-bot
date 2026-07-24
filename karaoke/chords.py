"""Thin chord-lookup wrapper: fetch a Tab4U sheet and render original/easy."""

from __future__ import annotations

import html

from .chord_image import render_chord_sheet_images
from .chord_sources import (
    _render_external_chord_sheet,
    lookup_external_chord_sheet_by_title,
)
from .harmony import transpose_chord_label
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


def render_inline_html(analysis: SongAnalysis, title: str, mode: str = "original") -> str:
    """Render as Telegram HTML with each chord bolded directly before the word
    it belongs to, e.g. "<b>[Cm]</b>שלום".

    Unlike a chords-above-lyrics layout (which requires a Latin chord row and
    a Hebrew lyric row to land in the same monospace columns — unreliable
    across Telegram clients, since different platforms use different <pre>
    fonts with different relative Hebrew/Latin glyph widths), a chord glued to
    its own word in the same text run needs no cross-line alignment at all, so
    it cannot drift regardless of font or device. Bold keeps the chord visually
    distinct from the lyrics despite the tighter spacing.
    """
    sheet = getattr(analysis, "parsed_sheet", None)
    if sheet is None:
        return html.escape(analysis.chord_sheet_text)
    semitones = _semitones_for_mode(analysis, mode)

    def label_html(raw: str) -> str:
        shown = transpose_chord_label(raw, semitones) if semitones else raw
        return f"<b>[{html.escape(shown)}]</b>"

    lines: list[str] = []
    for chord_tokens, lyric_words in sheet.line_word_pairs:
        if lyric_words:
            chars: list[str] = []
            for word in sorted(lyric_words, key=lambda w: w.column):
                while len(chars) < word.column:
                    chars.append(" ")
                # One list slot per SOURCE character, even though an escaped
                # character (e.g. "'" -> "&#x27;") is itself multiple chars —
                # otherwise inserting a chord marker at a later column can land
                # mid-entity and corrupt it (e.g. splitting "&#x27;" in two).
                chars.extend(html.escape(ch) for ch in word.text)
            for token in sorted(chord_tokens, key=lambda t: t.column, reverse=True):
                col = min(token.column, len(chars))
                chars.insert(col, label_html(token.label))
            lines.append("".join(chars).rstrip())
        elif chord_tokens:
            lines.append("  |  ".join(label_html(t.label) for t in chord_tokens))
        else:
            lines.append("")
    body = "\n".join(lines).strip()

    key = (analysis.target_key if mode == "easy" else analysis.original_key) or ""
    header = f"🎸 <b>{html.escape(title)}</b>"
    if key:
        header += f"  ·  סולם: {html.escape(key)}"
    return f"{header}\n\n{body}" if body else header


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
