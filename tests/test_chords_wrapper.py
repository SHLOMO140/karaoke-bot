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


def test_render_for_telegram_right_aligns_a_single_chord_without_reordering():
    # Real Tab4U row for אושר כהן - כולם גנבים: a lone "Cm" (2 chars) paired
    # with a 17-char Hebrew lyric row (confirmed via chord_sources._right_align_chord_row's
    # docstring: Tab4U's own .chords CSS is direction:ltr; text-align:right).
    a = SongAnalysis()
    a.original_key = "C"
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
    # Right-justified to the lyric row's width, label order untouched (there's
    # only one label anyway).
    assert chord_line == "Cm".rjust(len(lyric))


def test_render_for_telegram_right_aligns_multi_chord_row_without_reordering():
    # Real Tab4U row: "Ab              Fm" (18 chars) paired with a 32-char
    # Hebrew lyric row.
    a = SongAnalysis()
    a.original_key = "C"
    a.transpose_semitones = 0
    chord_row = "Ab              Fm"
    lyric = "תסתכלי לי בעיניים בטח שוב תגלגלי"
    rows = [
        _ChordRow(
            kind="chords", text=chord_row,
            tokens=[_ChordToken("Ab", 0), _ChordToken("Fm", 16)],
        ),
        _ChordRow(kind="song", text=lyric),
    ]
    a.parsed_sheet = _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[], line_word_pairs=[], chord_labels=["Ab", "Fm"],
    )
    out = chords.render(a, "t", for_telegram=True)
    chord_line = next(line for line in out.splitlines() if "Ab" in line or "Fm" in line)
    # Same left-to-right label order as the source (Ab still before Fm) — only
    # left-padded so the whole row shifts right to the lyric row's width.
    assert chord_line == chord_row.rjust(len(lyric))
    assert chord_line.index("Ab") < chord_line.index("Fm")
