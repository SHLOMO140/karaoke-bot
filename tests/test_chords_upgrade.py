"""Tests for the chords-track upgrades: C3 visible warning, C4 slash chords,
C5 key tie-break, C6 Tab4U parser fallback."""

import asyncio
from types import SimpleNamespace

import pytest

import bot
from karaoke import chord_sources, job_manager
from karaoke.models import ChordEvent, SongAnalysis, TranscriptSegment, WordTiming


def _analysis(confidence: float, *, external: bool = False) -> SongAnalysis:
    events = [
        ChordEvent("C", 0.0, 1.0, confidence=confidence, root="C", quality="major"),
        ChordEvent("F", 1.0, 2.0, confidence=confidence, root="F", quality="major"),
        ChordEvent("G", 2.0, 3.0, confidence=confidence, root="G", quality="major"),
    ]
    return SongAnalysis(
        bpm=120.0,
        provider="librosa_harmony_v5",
        chord_events=events,
        original_chord_events=list(events),
        chord_sheet_text="C F G\nשלום עולם",
        chord_source_name="Tab4U" if external else "",
    )


def _run_chords_delivery(job) -> list[str]:
    sent: list[str] = []

    class _FakeBot:
        async def send_document(self, **kwargs):
            return SimpleNamespace(message_id=7, link=None)

    class _FakeMessage:
        chat_id = 123

        async def reply_text(self, text, **kwargs):
            sent.append(text)
            return SimpleNamespace(message_id=1, link=None)

        def get_bot(self):
            return _FakeBot()

    asyncio.run(bot.send_chords_text_response(_FakeMessage(), job))
    return sent


def _make_job(tmp_path, monkeypatch, analysis: SongAnalysis):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="שיר", input_type="audio_file", chat_id=123)
    job.manifest.delivery_chat_id = 123
    job.lyrics_with_chords_path.write_text("C F G\nשלום עולם", encoding="utf-8")
    job_manager.save_song_analysis(job, analysis)
    return job


def test_low_confidence_chords_carry_visible_warning(tmp_path, monkeypatch):
    job = _make_job(tmp_path, monkeypatch, _analysis(0.2))

    sent = _run_chords_delivery(job)

    assert sent, "expected preview chunks"
    assert any("אמינות נמוכה" in text for text in sent)


def test_reliable_chords_have_no_warning(tmp_path, monkeypatch):
    job = _make_job(tmp_path, monkeypatch, _analysis(0.9, external=True))

    sent = _run_chords_delivery(job)

    assert sent
    assert not any("אמינות נמוכה" in text for text in sent)


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

    # Root-dominated bass keeps the plain label.
    bass_root = np.zeros(12)
    bass_root[0] = 0.6
    assert _apply_slash_bass_label("C", candidate, {"bass_vec": bass_root}, np) == "C"

    # Non-chord-tone bass never produces a slash.
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
    """Review finding: at weight 4.0 the continuity prior froze on a stale 'C'
    through a quiet stretch where 'G' consistently scored higher per frame."""
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
    # Truly quiet frames: NO_CHORD scores above the stale chord.
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
    # Nested tags inside lyric cells must not break word extraction.
    assert "טוב" in parsed.lyric_lines[0]


def test_tab4u_parser_falls_back_without_container():
    page = f"<html><body>{_TAB4U_FIXTURE_TABLES}</body></html>"
    parsed = chord_sources._parse_tab4u_sheet(page, "https://tab4u.com/x")

    assert parsed is not None, "layout change must not silently kill the parser"
    assert parsed.chord_labels == ["Am", "F", "G"]


def test_tab4u_parser_rejects_pages_without_cells():
    assert chord_sources._parse_tab4u_sheet("<html>שום דבר</html>", "u") is None
