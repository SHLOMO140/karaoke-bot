from karaoke.harmony import build_word_chord_map, render_chord_sheet_text
from karaoke.models import ChordEvent, SongAnalysis, TranscriptSegment, WordTiming


def _segments():
    return [
        TranscriptSegment(
            words=[
                WordTiming("כמו", 0.0, 0.5),
                WordTiming("השמש", 0.5, 1.1),
                WordTiming("מחממת", 1.1, 1.8),
            ],
            text="כמו השמש מחממת",
            start=0.0,
            end=1.8,
        )
    ]


def test_build_word_chord_map_anchors_changes_to_nearest_words():
    segments = _segments()
    chord_events = [
        ChordEvent("C", 0.05, 0.60),
        ChordEvent("Am", 0.82, 1.30),
        ChordEvent("Em", 1.45, 1.90),
    ]

    word_map = build_word_chord_map(segments, chord_events)

    assert [event.label for event in word_map[(0, 0)]] == ["C"]
    assert [event.label for event in word_map[(0, 1)]] == ["Am"]
    assert [event.label for event in word_map[(0, 2)]] == ["Em"]


def test_render_chord_sheet_text_includes_bpm_and_lyrics():
    analysis = SongAnalysis(
        bpm=96.4,
        chord_events=[
            ChordEvent("C", 0.05, 0.60),
            ChordEvent("G7", 0.82, 1.30),
            ChordEvent("Em7", 1.45, 1.90),
        ],
    )

    rendered = render_chord_sheet_text("demo", _segments(), analysis)

    assert "Title: demo" in rendered
    assert "BPM: 96.40" in rendered
    assert "C" in rendered
    assert "G7" in rendered
    assert "Em7" in rendered
    assert "כמו השמש מחממת" in rendered
