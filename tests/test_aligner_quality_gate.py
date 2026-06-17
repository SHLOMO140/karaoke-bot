import karaoke.aligner as aligner_module
from karaoke.aligner import SequenceHebrewAligner, analyze_alignment_quality
from karaoke.models import CharacterTiming, TranscriptSegment, WordTiming


def test_preserved_approved_timings_rebuild_reliable_detail_timings_without_shifting_words(monkeypatch):
    approved_word = WordTiming(
        "Michelle",
        7.7,
        8.6,
        confidence=0.0,
        source="review_hint",
        aligned=False,
        char_timings=[
            CharacterTiming("M", 7.7, 7.82),
            CharacterTiming("i", 7.82, 7.92),
            CharacterTiming("c", 7.92, 8.03),
            CharacterTiming("h", 8.03, 8.13),
            CharacterTiming("e", 8.13, 8.24),
            CharacterTiming("l", 8.24, 8.35),
            CharacterTiming("l", 8.35, 8.47),
            CharacterTiming("e", 8.47, 8.6),
        ],
    )
    approved_segments = [
        TranscriptSegment(
            words=[approved_word],
            text="Michelle",
            start=7.7,
            end=8.6,
        ),
    ]
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("Michelle", 7.7, 8.6, confidence=0.9, source="draft_whisper")],
            text="Michelle",
            start=7.7,
            end=8.6,
        ),
    ]
    regressed_segments = [
        TranscriptSegment(
            words=[WordTiming("Michelle", 1.0, 1.5, confidence=0.9, source="forced_aligner", aligned=True)],
            text="Michelle",
            start=1.0,
            end=1.5,
        ),
    ]

    monkeypatch.setattr(aligner_module, "_rebuild_segments", lambda template_segments, final_words: regressed_segments)
    monkeypatch.setattr(
        aligner_module,
        "_refine_segments",
        lambda audio_path, segments, video_frame_rate=None: segments,
    )
    monkeypatch.setattr(aligner_module, "_validate_final_segments", lambda segments: segments)

    aligned = SequenceHebrewAligner().align("dummy.wav", approved_segments, draft_segments)
    repaired_word = aligned.segments[0].words[0]
    report = analyze_alignment_quality(aligned.segments)

    assert repaired_word.start == approved_word.start
    assert repaired_word.end == approved_word.end
    assert repaired_word.aligned is True
    assert repaired_word.subwords
    assert min(subword.confidence for subword in repaired_word.subwords) >= 0.55
    assert float(report["detail_ratio"]) >= 1.0


def test_preserved_approved_timings_synthesize_missing_subwords_for_quality_gate(monkeypatch):
    approved_segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 0.6, confidence=0.0, source="review_hint", aligned=False)],
            text="hello",
            start=0.0,
            end=0.6,
        ),
        TranscriptSegment(
            words=[WordTiming("world", 0.8, 1.5, confidence=0.0, source="review_hint", aligned=False)],
            text="world",
            start=0.8,
            end=1.5,
        ),
    ]
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 0.6, confidence=0.9, source="draft_whisper")],
            text="hello",
            start=0.0,
            end=0.6,
        ),
        TranscriptSegment(
            words=[WordTiming("world", 0.8, 1.5, confidence=0.9, source="draft_whisper")],
            text="world",
            start=0.8,
            end=1.5,
        ),
    ]
    regressed_segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 0.5, confidence=0.9, source="forced_aligner", aligned=True)],
            text="hello",
            start=0.0,
            end=0.5,
        ),
        TranscriptSegment(
            words=[WordTiming("world", 0.6, 0.9, confidence=0.9, source="forced_aligner", aligned=True)],
            text="world",
            start=0.6,
            end=0.9,
        ),
    ]

    monkeypatch.setattr(aligner_module, "_rebuild_segments", lambda template_segments, final_words: regressed_segments)
    monkeypatch.setattr(
        aligner_module,
        "_refine_segments",
        lambda audio_path, segments, video_frame_rate=None: segments,
    )
    monkeypatch.setattr(aligner_module, "_validate_final_segments", lambda segments: segments)

    aligned = SequenceHebrewAligner().align("dummy.wav", approved_segments, draft_segments)
    report = analyze_alignment_quality(aligned.segments)

    assert all(word.subwords for segment in aligned.segments for word in segment.words)
    assert float(report["detail_ratio"]) >= 1.0
    assert report["critical"] is False
