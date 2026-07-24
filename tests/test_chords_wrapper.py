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
