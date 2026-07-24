"""Tests for the chords lookup/render wrapper."""

from unittest.mock import patch

from karaoke import chords
from karaoke.chord_sources import _ChordRow, _ChordToken, _LyricWord, _ParsedTab4USheet
from karaoke.models import SongAnalysis


def test_lookup_returns_none_when_no_sheet():
    with patch("karaoke.chords.lookup_external_chord_sheet_by_title", return_value=None):
        assert chords.lookup("שיר לא קיים") is None


def _analysis_with_sheet():
    a = SongAnalysis()
    a.original_key = "C"
    a.target_key = "Am"
    a.transpose_semitones = 0
    rows = [_ChordRow(kind="chords", text="C  G", tokens=[_ChordToken("C", 0), _ChordToken("G", 3)])]
    a.parsed_sheet = _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[],
        line_word_pairs=[], chord_labels=["C", "G"],
    )
    return a


def test_render_original_includes_chords_and_key_header():
    a = _analysis_with_sheet()
    out = chords.render(a, "שיר בדיקה", mode="original")
    assert "C" in out and "G" in out
    assert "סולם מקור: C" in out


def test_render_falls_back_to_sheet_text_without_parsed_sheet():
    a = SongAnalysis()
    a.chord_sheet_text = "fallback text"
    assert chords.render(a, "t", mode="easy") == "fallback text"


def test_render_images_empty_without_parsed_sheet():
    a = SongAnalysis()
    a.chord_sheet_text = "fallback text"
    assert chords.render_images(a, "t") == []


def test_render_images_produces_png_bytes():
    a = _analysis_with_sheet()
    images = chords.render_images(a, "שיר בדיקה", mode="original")
    assert images
    for data in images:
        assert data.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_text_matches_real_tab4u_song():
    # Real data captured from https://www.tab4u.com/tabs/songs/77303 (אושר כהן
    # - כולם גנבים) — layout_text exactly as scraped, including Tab4U's own
    # leading &nbsp; padding. Locks in that "Ab" (a chord for the phrase's
    # start) renders one column in, matching Tab4U's own row rather than at
    # column 0 (which is what the pre-fix, whitespace-stripped scraper produced
    # and which put every multi-chord line one word off).
    a = SongAnalysis()
    a.original_key = "Fm"
    a.transpose_semitones = 0
    rows = [
        _ChordRow(
            kind="chords", text="Ab              Fm",
            layout_text=" Ab              Fm             ",
            tokens=[_ChordToken("Ab", 1), _ChordToken("Fm", 17)],
        ),
        _ChordRow(
            kind="song", text="תסתכלי לי בעיניים בטח שוב תגלגלי",
            layout_text="תסתכלי לי בעיניים בטח שוב תגלגלי",
        ),
    ]
    a.parsed_sheet = _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[], line_word_pairs=[], chord_labels=["Ab", "Fm"],
    )
    out = chords.render(a, "כולם גנבים")
    lines = out.splitlines()
    chord_line = next(line for line in lines if "Ab" in line)
    lyric_line = next(line for line in lines if "תסתכלי" in line)
    assert chord_line == " Ab              Fm"
    assert lyric_line == "תסתכלי לי בעיניים בטח שוב תגלגלי"


def test_render_text_preserves_tab4u_leading_padding():
    # Regression test for the actual root cause of six failed alignment
    # attempts: Tab4U pads a chords row with LEADING &nbsp; whenever its first
    # token isn't meant to sit at column 0 (e.g. a lone "Cm" belongs under the
    # middle of its lyric line, not the start) — row.text (used for search
    # matching) has that padding stripped by _clean_visible_text; row.layout_text
    # preserves it verbatim, and _render_external_chord_sheet must use the
    # latter for the chords row or every chord's effective column shifts.
    a = SongAnalysis()
    a.original_key = "C"
    a.transpose_semitones = 0
    rows = [
        _ChordRow(
            kind="chords", text="Cm", layout_text="   Cm",
            tokens=[_ChordToken("Cm", 3)],
        ),
        _ChordRow(kind="song", text="את מלכה אבל עדיין", layout_text="את מלכה אבל עדיין"),
    ]
    a.parsed_sheet = _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[], line_word_pairs=[], chord_labels=["Cm"],
    )
    out = chords.render(a, "t")
    chord_line = next(line for line in out.splitlines() if "Cm" in line)
    assert chord_line == "   Cm"
