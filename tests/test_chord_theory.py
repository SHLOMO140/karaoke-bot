"""Chord-theory and Tab4U-parser tests (salvaged from test_chords_upgrade)."""

import pytest

from karaoke import chord_sources
from karaoke.models import ChordEvent


def test_slash_label_applied_when_bass_dominates():
    np = pytest.importorskip("numpy")
    if not hasattr(np, "argmax"):
        pytest.skip("real numpy required")
    from karaoke.harmony import _apply_slash_bass_label

    candidate = {"root_index": 0, "quality": "major"}  # C major
    bass = np.zeros(12)
    bass[4] = 0.6  # E dominates the bass register
    bass[0] = 0.2
    row = {"bass_vec": bass}

    assert _apply_slash_bass_label("C", candidate, row, np) == "C/E"

    bass_root = np.zeros(12)
    bass_root[0] = 0.6
    assert _apply_slash_bass_label("C", candidate, {"bass_vec": bass_root}, np) == "C"

    bass_foreign = np.zeros(12)
    bass_foreign[1] = 0.9  # C# is not a C-major chord tone
    assert _apply_slash_bass_label("C", candidate, {"bass_vec": bass_foreign}, np) == "C"


def _viterbi_candidate(label, score, root_index, quality="major", family="major"):
    return {
        "label": label,
        "root": label[0] if label != "N" else "",
        "root_index": root_index,
        "quality": quality if label != "N" else "",
        "family": family if label != "N" else "",
        "score": score,
        "confidence": 0.5,
    }


def test_viterbi_does_not_freeze_on_stale_chord():
    from karaoke.harmony import _decode_chord_path

    first = [_viterbi_candidate("C", 0.6, 0)]
    quiet = [
        _viterbi_candidate("G", 0.28, 7),
        _viterbi_candidate("C", 0.24, 0),
        _viterbi_candidate("N", 0.22, -1),
    ]
    path = _decode_chord_path([first] + [list(quiet) for _ in range(5)])
    assert [c["label"] for c in path][1:] == ["G"] * 5


def test_viterbi_quiet_fade_resolves_to_no_chord():
    from karaoke.harmony import _decode_chord_path

    first = [_viterbi_candidate("C", 0.6, 0)]
    fade = [
        _viterbi_candidate("N", 0.30, -1),
        _viterbi_candidate("C", 0.20, 0),
    ]
    path = _decode_chord_path([first] + [list(fade) for _ in range(5)])
    assert [c["label"] for c in path][1:] == ["N"] * 5


def test_key_tie_break_prefers_played_tonic():
    from karaoke.harmony import infer_song_key

    events = [
        ChordEvent("Am", 0.0, 4.0, confidence=0.8, root="A", quality="minor"),
        ChordEvent("F", 4.0, 6.0, confidence=0.8, root="F", quality="major"),
        ChordEvent("G", 6.0, 8.0, confidence=0.8, root="G", quality="major"),
        ChordEvent("Am", 8.0, 12.0, confidence=0.8, root="A", quality="minor"),
    ]
    key, _tonic, mode = infer_song_key(events)
    assert key == "Am"
    assert mode == "minor"


_TAB4U_FIXTURE_TABLES = """
<table>
<tr><td class="chords_en">Am   F</td></tr>
<tr><td class="song">שלום עולם <b>טוב</b></td></tr>
<tr><td class="chords_en">G</td></tr>
<tr><td class="song">עוד שורה של מילים</td></tr>
</table>
"""


def test_tab4u_parser_with_content_container():
    page = f'<div id="songContentTPL">{_TAB4U_FIXTURE_TABLES}</div>'
    parsed = chord_sources._parse_tab4u_sheet(page, "https://tab4u.com/x")
    assert parsed is not None
    assert parsed.chord_labels == ["Am", "F", "G"]
    assert len(parsed.lyric_lines) == 2
    assert "טוב" in parsed.lyric_lines[0]


def test_tab4u_parser_falls_back_without_container():
    page = f"<html><body>{_TAB4U_FIXTURE_TABLES}</body></html>"
    parsed = chord_sources._parse_tab4u_sheet(page, "https://tab4u.com/x")
    assert parsed is not None, "layout change must not silently kill the parser"
    assert parsed.chord_labels == ["Am", "F", "G"]


def test_tab4u_parser_rejects_pages_without_cells():
    assert chord_sources._parse_tab4u_sheet("<html>שום דבר</html>", "u") is None
