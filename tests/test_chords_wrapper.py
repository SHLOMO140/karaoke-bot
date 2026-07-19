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


def test_render_for_telegram_right_anchors_a_single_chord_to_its_lyric_row():
    a = SongAnalysis()
    a.original_key = "C"
    a.target_key = "Am"
    a.transpose_semitones = 0
    lyric = "את מלכה אבל עדיין"
    rows = [
        _ChordRow(kind="chords", text="Cm", tokens=[_ChordToken("Cm", 0)]),
        _ChordRow(kind="song", text=lyric),
    ]
    a.parsed_sheet = _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[], line_word_pairs=[], chord_labels=["Cm"],
    )
    out = chords.render(a, "t", for_telegram=True)
    chord_line = next(line for line in out.splitlines() if "Cm" in line)
    # Right-anchored to the lyric row's own width, not left at column 0 — a
    # Hebrew-reading reader expects the chord to line up with the RIGHT edge,
    # matching where Telegram displays that row's first character.
    assert chord_line == "Cm".rjust(len(lyric))
