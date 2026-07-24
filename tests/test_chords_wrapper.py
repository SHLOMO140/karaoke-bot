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


def test_render_for_telegram_leaves_a_single_chord_row_unchanged():
    a = SongAnalysis()
    a.original_key = "C"
    a.transpose_semitones = 0
    rows = [
        _ChordRow(kind="chords", text="Cm", tokens=[_ChordToken("Cm", 0)]),
        _ChordRow(kind="song", text="את מלכה אבל עדיין"),
    ]
    a.parsed_sheet = _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[], line_word_pairs=[], chord_labels=["Cm"],
    )
    out = chords.render(a, "t", for_telegram=True)
    # A single chord has no "order" to fix — its position was already correct
    # once the sheet renders as a monospace block, so it's left untouched.
    assert "\nCm\n" in out or out.rstrip().endswith("\nCm")


def test_render_for_telegram_reverses_multi_chord_order_in_place():
    a = SongAnalysis()
    a.original_key = "C"
    a.transpose_semitones = 0
    rows = [
        _ChordRow(
            kind="chords", text="Ab                Fm",
            tokens=[_ChordToken("Ab", 0), _ChordToken("Fm", 18)],
        ),
        _ChordRow(kind="song", text="תסתכלי לי בעינים בטח שוב תגלגלי"),
    ]
    a.parsed_sheet = _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[], line_word_pairs=[], chord_labels=["Ab", "Fm"],
    )
    out = chords.render(a, "t", for_telegram=True)
    chord_line = next(line for line in out.splitlines() if "Ab" in line or "Fm" in line)
    # Same slots/spacing as the source row — only which label sits in which
    # slot is swapped (Fm now first, Ab now last), matching the Hebrew lyric
    # line's right-to-left reading beneath it.
    assert chord_line == "Fm                Ab"
