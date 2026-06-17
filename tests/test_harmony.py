import numpy as np

from karaoke.harmony import (
    _collect_chord_candidates,
    _simplify_same_root_variants,
    build_word_chord_map,
    infer_key_from_chroma_profile,
    prepare_song_analysis_for_display,
    render_chord_sheet_text,
    resolve_song_analysis_key_labels,
    summarize_song_analysis_quality,
    transpose_chord_label,
)
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


def test_build_word_chord_map_carries_active_chord_to_next_segment_start():
    segments = [
        TranscriptSegment(
            words=[WordTiming("first", 0.0, 0.5), WordTiming("line", 0.5, 0.95)],
            text="first line",
            start=0.0,
            end=0.95,
        ),
        TranscriptSegment(
            words=[WordTiming("second", 1.0, 1.3), WordTiming("line", 1.3, 1.7)],
            text="second line",
            start=1.0,
            end=1.7,
        ),
    ]
    chord_events = [
        ChordEvent("C", 0.0, 1.4),
        ChordEvent("G", 1.4, 1.8),
    ]

    word_map = build_word_chord_map(segments, chord_events)

    assert [event.label for event in word_map[(1, 0)]] == ["C"]
    assert [event.label for event in word_map[(1, 1)]] == ["G"]


def test_render_chord_sheet_text_includes_metadata_and_lyrics():
    analysis = SongAnalysis(
        bpm=96.4,
        original_key="Em",
        target_key="Am",
        chord_events=[
            ChordEvent("C", 0.05, 0.60),
            ChordEvent("G7", 0.82, 1.30),
            ChordEvent("Em7", 1.45, 1.90),
        ],
    )

    rendered = render_chord_sheet_text("demo", _segments(), analysis)

    assert "כותרת: demo" in rendered
    assert "קצב: 96" in rendered
    assert "סולם מקור: Em" in rendered
    assert "סולם קל: Am" in rendered
    assert "C" in rendered
    assert "G7" in rendered
    assert "Em7" in rendered
    assert "כמו השמש מחממת" in rendered


def test_transpose_chord_label_preserves_full_suffix_and_bass():
    assert transpose_chord_label("Em7b5/B", 5) == "Am7b5/E"


def test_prepare_song_analysis_for_display_transposes_to_am_by_default():
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("one", 0.0, 0.5),
                WordTiming("two", 0.5, 1.0),
                WordTiming("three", 1.0, 1.5),
                WordTiming("four", 1.5, 2.0),
            ],
            text="one two three four",
            start=0.0,
            end=2.0,
        )
    ]
    analysis = SongAnalysis(
        chord_events=[
            ChordEvent("Em", 0.0, 0.5, root="E", quality="minor"),
            ChordEvent("Am", 0.5, 1.0, root="A", quality="minor"),
            ChordEvent("B7", 1.0, 1.5, root="B", quality="dominant7"),
            ChordEvent("Em7b5", 1.5, 2.0, root="E", quality="half_diminished"),
        ]
    )

    prepared = prepare_song_analysis_for_display(analysis, segments)

    assert prepared.original_key == "Em"
    assert prepared.target_key == "Am"
    assert prepared.transpose_semitones == 5
    assert [event.label for event in prepared.original_chord_events] == ["Em", "Am", "B7", "Em7b5"]
    assert [event.label for event in prepared.chord_events] == ["Am", "Dm", "E7", "Am7b5"]


def test_prepare_song_analysis_for_display_transposes_when_target_key_requested():
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("one", 0.0, 0.5),
                WordTiming("two", 0.5, 1.0),
                WordTiming("three", 1.0, 1.5),
                WordTiming("four", 1.5, 2.0),
            ],
            text="one two three four",
            start=0.0,
            end=2.0,
        )
    ]
    analysis = SongAnalysis(
        chord_events=[
            ChordEvent("Em", 0.0, 0.5, root="E", quality="minor"),
            ChordEvent("Am", 0.5, 1.0, root="A", quality="minor"),
            ChordEvent("B7", 1.0, 1.5, root="B", quality="dominant7"),
            ChordEvent("Em7b5", 1.5, 2.0, root="E", quality="half_diminished"),
        ]
    )

    prepared = prepare_song_analysis_for_display(analysis, segments, target_key="Am")

    assert prepared.original_key == "Em"
    assert prepared.target_key == "Am"
    assert prepared.transpose_semitones == 5
    assert [event.label for event in prepared.chord_events] == ["Am", "Dm", "E7", "Am7b5"]


def test_prepare_song_analysis_for_display_can_keep_original_key_when_requested():
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("one", 0.0, 0.5),
                WordTiming("two", 0.5, 1.0),
                WordTiming("three", 1.0, 1.5),
                WordTiming("four", 1.5, 2.0),
            ],
            text="one two three four",
            start=0.0,
            end=2.0,
        )
    ]
    analysis = SongAnalysis(
        chord_events=[
            ChordEvent("Em", 0.0, 0.5, root="E", quality="minor"),
            ChordEvent("Am", 0.5, 1.0, root="A", quality="minor"),
            ChordEvent("B7", 1.0, 1.5, root="B", quality="dominant7"),
            ChordEvent("Em7b5", 1.5, 2.0, root="E", quality="half_diminished"),
        ]
    )

    prepared = prepare_song_analysis_for_display(analysis, segments, target_key="")

    assert prepared.original_key == "Em"
    assert prepared.target_key == ""
    assert prepared.transpose_semitones == 0
    assert [event.label for event in prepared.chord_events] == ["Em", "Am", "B7", "Em7b5"]


def test_resolve_song_analysis_key_labels_prefers_inferred_event_keys():
    analysis = SongAnalysis(
        original_key="F#",
        target_key="Am",
        transpose_semitones=6,
        original_chord_events=[
            ChordEvent("Eb", 0.0, 1.0, root="Eb"),
            ChordEvent("Fm", 1.0, 2.0, root="F"),
            ChordEvent("Eb", 2.0, 3.0, root="Eb"),
            ChordEvent("Fm", 3.0, 4.0, root="F"),
            ChordEvent("Eb", 4.0, 5.0, root="Eb"),
            ChordEvent("Ab", 5.0, 6.0, root="Ab"),
            ChordEvent("C", 6.0, 7.0, root="C"),
            ChordEvent("C#", 7.0, 8.0, root="C#"),
        ],
        chord_events=[
            ChordEvent("A", 0.0, 1.0, root="A"),
            ChordEvent("Bm", 1.0, 2.0, root="B"),
            ChordEvent("A", 2.0, 3.0, root="A"),
            ChordEvent("Bm", 3.0, 4.0, root="B"),
            ChordEvent("A", 4.0, 5.0, root="A"),
            ChordEvent("D", 5.0, 6.0, root="D"),
            ChordEvent("F#", 6.0, 7.0, root="F#"),
            ChordEvent("G", 7.0, 8.0, root="G"),
        ],
    )

    original_key, target_key = resolve_song_analysis_key_labels(analysis)

    assert original_key == "Eb"
    assert target_key == "A"


def test_prepare_song_analysis_for_display_recovers_original_chords_from_legacy_transposed_analysis():
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("one", 0.0, 0.5),
                WordTiming("two", 0.5, 1.0),
            ],
            text="one two",
            start=0.0,
            end=1.0,
        )
    ]
    analysis = SongAnalysis(
        original_key="Em",
        target_key="Am",
        transpose_semitones=5,
        chord_events=[
            ChordEvent("Am", 0.0, 0.5, root="A", quality="minor"),
            ChordEvent("Dm", 0.5, 1.0, root="D", quality="minor"),
        ],
    )

    prepared = prepare_song_analysis_for_display(analysis, segments, target_key="")

    assert prepared.original_key == "Em"
    assert prepared.target_key == ""
    assert prepared.transpose_semitones == 0
    assert [event.label for event in prepared.original_chord_events] == ["Em", "Am"]
    assert [event.label for event in prepared.chord_events] == ["Em", "Am"]


def test_infer_key_from_chroma_profile_detects_e_minor_center():
    chroma = np.zeros((12, 4), dtype=float)
    chroma[4, :] = [1.0, 0.9, 1.0, 0.95]   # E
    chroma[7, :] = [0.85, 0.8, 0.88, 0.82]  # G
    chroma[11, :] = [0.78, 0.72, 0.8, 0.74]  # B
    chroma[2, :] = [0.35, 0.3, 0.34, 0.32]  # D
    chroma[9, :] = [0.28, 0.24, 0.26, 0.25]  # A

    key_label, tonic_pitch, mode = infer_key_from_chroma_profile(chroma, np)

    assert key_label == "Em"
    assert tonic_pitch == 4
    assert mode == "minor"


def test_collect_chord_candidates_prefers_plain_major_when_major_seventh_is_not_stable():
    chroma = np.array([1.0, 0.02, 0.03, 0.02, 0.82, 0.03, 0.02, 0.95, 0.02, 0.02, 0.02, 0.09], dtype=float)
    stable = np.array([1.0, 0.0, 0.01, 0.0, 0.76, 0.01, 0.0, 0.86, 0.0, 0.0, 0.0, 0.04], dtype=float)
    bass = np.array([0.82, 0.01, 0.0, 0.0, 0.14, 0.0, 0.0, 0.18, 0.0, 0.0, 0.0, 0.0], dtype=float)

    candidates = _collect_chord_candidates(chroma, bass, np, stable_vector=stable)

    assert candidates[0]["label"] == "C"


def test_collect_chord_candidates_uses_bass_and_stable_notes_to_pick_a_minor():
    chroma = np.array([0.7, 0.02, 0.02, 0.01, 0.62, 0.02, 0.01, 0.12, 0.02, 0.92, 0.01, 0.55], dtype=float)
    stable = np.array([0.56, 0.0, 0.0, 0.0, 0.58, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.12], dtype=float)
    bass = np.array([0.12, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.92, 0.0, 0.05], dtype=float)

    candidates = _collect_chord_candidates(chroma, bass, np, stable_vector=stable)

    assert candidates[0]["label"] == "Am"


def test_simplify_same_root_variants_collapses_same_root_decorations():
    events = [
        ChordEvent("B", 0.0, 0.8, confidence=0.45, root="B", quality="major"),
        ChordEvent("Bsus2", 0.8, 1.5, confidence=0.31, root="B", quality="sus2"),
        ChordEvent("Bmaj7", 1.5, 2.2, confidence=0.28, root="B", quality="major7"),
    ]

    simplified = _simplify_same_root_variants(events, beat_length=0.95)

    assert len(simplified) == 1
    assert simplified[0].label == "B"
    assert simplified[0].start == 0.0
    assert simplified[0].end == 2.2


def test_simplify_same_root_variants_drops_short_major7_for_stronger_minor_neighbor():
    events = [
        ChordEvent("Ebmaj7", 0.0, 0.6, confidence=0.4, root="Eb", quality="major7"),
        ChordEvent("Ebm", 0.6, 1.8, confidence=0.55, root="Eb", quality="minor"),
    ]

    simplified = _simplify_same_root_variants(events, beat_length=0.95)

    assert len(simplified) == 1
    assert simplified[0].label == "Ebm"


def test_simplify_same_root_variants_drops_short_colored_variant_for_stronger_neighbor():
    events = [
        ChordEvent("Absus2", 0.0, 0.7, confidence=0.29, root="Ab", quality="sus2"),
        ChordEvent("Abm", 0.7, 1.9, confidence=0.51, root="Ab", quality="minor"),
    ]

    simplified = _simplify_same_root_variants(events, beat_length=0.95)

    assert len(simplified) == 1
    assert simplified[0].label == "Abm"


def test_summarize_song_analysis_quality_flags_low_confidence_audio_output():
    analysis = SongAnalysis(
        provider="librosa_harmony_v5",
        chord_events=[
            ChordEvent("Am", 0.0, 4.0, confidence=0.31, root="A", quality="minor"),
            ChordEvent("C", 4.0, 8.0, confidence=0.28, root="C", quality="major"),
            ChordEvent("Dm", 8.0, 12.0, confidence=0.34, root="D", quality="minor"),
        ],
    )

    summary = summarize_song_analysis_quality(analysis)

    assert summary.visible_chord_count == 3
    assert summary.average_confidence < 0.52
    assert summary.low_confidence_ratio == 1.0
    assert summary.reliable_for_delivery is False


def test_summarize_song_analysis_quality_keeps_external_sources_deliverable():
    analysis = SongAnalysis(
        provider="librosa_harmony_v5",
        chord_source_name="Tab4U",
        chord_source_url="https://www.tab4u.com/tabs/songs/1_demo.html",
        chord_events=[
            ChordEvent("Am", 0.0, 4.0, confidence=0.2, root="A", quality="minor"),
            ChordEvent("Dm", 4.0, 8.0, confidence=0.22, root="D", quality="minor"),
        ],
    )

    summary = summarize_song_analysis_quality(analysis)

    assert summary.has_external_source is True
    assert summary.reliable_for_delivery is True
