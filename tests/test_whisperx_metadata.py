from types import SimpleNamespace

import whisperx

import karaoke.aligner as aligner_module
from karaoke.aligner import (
    WhisperXHebrewAligner,
    _refine_segments,
    _refine_word_timings,
    _sanitize_whisperx_metadata,
    _whisperx_segment_candidate_score,
    _whisperx_segment_needs_retry,
    _words_from_whisperx_segment,
)
from karaoke.models import CharacterTiming, SubWordTiming, TranscriptSegment, WordTiming


class _FakeLmHead:
    out_features = 31


class _FakeModel:
    lm_head = _FakeLmHead()


def test_sanitize_whisperx_metadata_drops_out_of_range_special_tokens():
    metadata = {
        "language": "he",
        "type": "huggingface",
        "dictionary": {
            "א": 19,
            "|": 27,
            "[pad]": 30,
            "<s>": 31,
            "</s>": 32,
            "<pad>": 33,
            "<unk>": 34,
        },
    }

    sanitized = _sanitize_whisperx_metadata(_FakeModel(), metadata)

    assert sanitized["dictionary"] == {
        "א": 19,
        "|": 27,
        "[pad]": 30,
    }


def test_sanitize_whisperx_metadata_keeps_valid_dictionary_unchanged():
    metadata = {
        "language": "he",
        "type": "huggingface",
        "dictionary": {
            "א": 19,
            "|": 27,
            "[pad]": 30,
        },
    }

    sanitized = _sanitize_whisperx_metadata(_FakeModel(), metadata)

    assert sanitized is metadata


def test_whisperx_aligner_sanitizes_dictionary_before_align(monkeypatch, tmp_path):
    aligner = WhisperXHebrewAligner()
    approved_segments = [
        TranscriptSegment(
            words=[WordTiming("אב", 0.0, 0.5, source="review_hint")],
            text="אב",
            start=0.0,
            end=0.5,
        )
    ]
    captured = {}

    monkeypatch.setattr(
        aligner,
        "_load",
        lambda: (
            _FakeModel(),
            {
                "language": "he",
                "type": "huggingface",
                "dictionary": {
                    "א": 19,
                    "ב": 20,
                    "[pad]": 30,
                    "<pad>": 33,
                },
            },
        ),
    )
    monkeypatch.setattr(whisperx, "load_audio", lambda path: [0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda payload, model, metadata, audio, device, return_char_alignments=True: (
            captured.setdefault("dictionary", metadata["dictionary"]),
            {
                "segments": [
                    {
                        "text": "אב",
                        "start": 0.0,
                        "end": 0.5,
                        "words": [
                            {
                                "word": "אב",
                                "start": 0.0,
                                "end": 0.5,
                                "score": 0.9,
                                "chars": [
                                    {"char": "א", "start": 0.0, "end": 0.2, "score": 0.9},
                                    {"char": "ב", "start": 0.2, "end": 0.5, "score": 0.9},
                                ],
                            }
                        ],
                        "chars": [
                            {"char": "א", "start": 0.0, "end": 0.2, "score": 0.9},
                            {"char": "ב", "start": 0.2, "end": 0.5, "score": 0.9},
                        ],
                    }
                ],
                "word_segments": [],
            },
        )[1],
    )
    monkeypatch.setattr(aligner_module, "_refine_segments", lambda audio_path, segments, video_frame_rate=None: segments)
    monkeypatch.setattr(aligner_module, "_validate_final_segments", lambda segments: segments)

    result = aligner.align(str(tmp_path / "dummy.wav"), approved_segments, approved_segments)

    assert "<pad>" not in captured["dictionary"]
    assert captured["dictionary"]["[pad]"] == 30
    assert result.unaligned_word_count == 0
    assert result.segments[0].words[0].char_timings


def test_refine_word_timings_preserves_existing_direct_letter_alignment(monkeypatch):
    word = WordTiming(
        word="שלום",
        start=0.0,
        end=0.8,
        confidence=0.9,
        source="forced_aligner",
        aligned=True,
        subwords=[
            SubWordTiming("ש", 0.0, 0.1, 0.95),
            SubWordTiming("ל", 0.1, 0.2, 0.95),
            SubWordTiming("ו", 0.2, 0.55, 0.95),
            SubWordTiming("ם", 0.55, 0.8, 0.95),
        ],
        char_timings=[
            CharacterTiming("ש", 0.0, 0.1),
            CharacterTiming("ל", 0.1, 0.2),
            CharacterTiming("ו", 0.2, 0.55),
            CharacterTiming("ם", 0.55, 0.8),
        ],
    )

    monkeypatch.setattr(
        aligner_module,
        "_load_audio_features",
        lambda audio_path: SimpleNamespace(energy=[0.0], hop_seconds=0.01),
    )
    monkeypatch.setattr(aligner_module, "_find_word_onset", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(aligner_module, "_find_word_offset", lambda *args, **kwargs: 1.8)
    monkeypatch.setattr(
        aligner_module,
        "_build_subword_timings",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should preserve direct timings")),
    )

    refined = _refine_word_timings("dummy.wav", [word])

    assert len(refined) == 1
    assert refined[0].start == 1.0
    assert refined[0].end == 1.8
    assert [subword.text for subword in refined[0].subwords] == ["ש", "ל", "ו", "ם"]
    assert refined[0].subwords[0].start == 1.0
    assert refined[0].subwords[-1].end == 1.8
    assert refined[0].char_timings[0].start == 1.0
    assert refined[0].char_timings[-1].end == 1.8


def test_whisperx_aligner_retries_weak_segment_with_wider_window(monkeypatch, tmp_path):
    aligner = WhisperXHebrewAligner()
    approved_segments = [
        TranscriptSegment(
            words=[
                WordTiming("שלום", 0.0, 0.3, source="review_hint"),
                WordTiming("עולם", 0.3, 0.6, source="review_hint"),
            ],
            text="שלום עולם",
            start=0.0,
            end=0.6,
        )
    ]
    calls = []

    monkeypatch.setattr(
        aligner,
        "_load",
        lambda: (
            _FakeModel(),
            {
                "language": "he",
                "type": "huggingface",
                "dictionary": {
                    "ש": 1,
                    "ל": 2,
                    "ו": 3,
                    "ם": 4,
                    "ע": 5,
                    "[pad]": 30,
                },
            },
        ),
    )
    monkeypatch.setattr(whisperx, "load_audio", lambda path: [0.0] * 1600)

    def _fake_align(payload, model, metadata, audio, device, return_char_alignments=True):
        del model, metadata, audio, device, return_char_alignments
        calls.append(payload[0])
        if len(calls) == 1:
            return {
                "segments": [
                    {
                        "text": "שלום עולם",
                        "start": payload[0]["start"],
                        "end": payload[0]["end"],
                        "words": [{"word": "שלום", "start": 0.02, "end": 0.28, "score": 0.5}],
                        "chars": [],
                    }
                ],
                "word_segments": [],
            }
        return {
            "segments": [
                {
                    "text": "שלום עולם",
                    "start": payload[0]["start"],
                    "end": payload[0]["end"],
                    "words": [
                        {
                            "word": "שלום",
                            "start": 0.02,
                            "end": 0.28,
                            "score": 0.9,
                            "chars": [
                                {"char": "ש", "start": 0.02, "end": 0.08, "score": 0.9},
                                {"char": "ל", "start": 0.08, "end": 0.14, "score": 0.9},
                                {"char": "ו", "start": 0.14, "end": 0.22, "score": 0.9},
                                {"char": "ם", "start": 0.22, "end": 0.28, "score": 0.9},
                            ],
                        },
                        {
                            "word": "עולם",
                            "start": 0.31,
                            "end": 0.58,
                            "score": 0.9,
                            "chars": [
                                {"char": "ע", "start": 0.31, "end": 0.38, "score": 0.9},
                                {"char": "ו", "start": 0.38, "end": 0.45, "score": 0.9},
                                {"char": "ל", "start": 0.45, "end": 0.51, "score": 0.9},
                                {"char": "ם", "start": 0.51, "end": 0.58, "score": 0.9},
                            ],
                        },
                    ],
                    "chars": [],
                }
            ],
            "word_segments": [],
        }

    monkeypatch.setattr(whisperx, "align", _fake_align)
    monkeypatch.setattr(aligner_module, "_refine_segments", lambda audio_path, segments, video_frame_rate=None: segments)
    monkeypatch.setattr(aligner_module, "_validate_final_segments", lambda segments: segments)

    result = aligner.align(str(tmp_path / "dummy.wav"), approved_segments, approved_segments)

    assert len(calls) == 2
    assert calls[1]["start"] < calls[0]["start"] or calls[1]["end"] > calls[0]["end"]
    assert result.unaligned_word_count == 0
    assert [word.word for word in result.segments[0].words] == ["שלום", "עולם"]


def test_refine_segments_clips_previous_overlap_instead_of_shifting_next(monkeypatch):
    first = TranscriptSegment(
        words=[WordTiming("alpha", 0.0, 0.9, confidence=0.9, source="forced_aligner", aligned=True)],
        text="alpha",
        start=0.0,
        end=0.9,
    )
    second = TranscriptSegment(
        words=[WordTiming("beta", 1.0, 1.8, confidence=0.9, source="forced_aligner", aligned=True)],
        text="beta",
        start=1.0,
        end=1.8,
    )

    def _fake_refine(_audio_path, words, video_frame_rate=None):
        del _audio_path, video_frame_rate
        token = words[0].word
        if token == "alpha":
            return [WordTiming("alpha", 0.0, 1.05, confidence=0.9, source="forced_aligner", aligned=True)]
        return [WordTiming("beta", 0.95, 1.8, confidence=0.9, source="forced_aligner", aligned=True)]

    monkeypatch.setattr(aligner_module, "_refine_word_timings", _fake_refine)

    refined = _refine_segments("dummy.wav", [first, second])

    assert refined[0].words[0].end == 0.95
    assert refined[1].words[0].start == 0.95
    assert refined[0].words[0].aligned is True
    assert refined[1].words[0].aligned is True


def test_words_from_whisperx_segment_maps_timings_onto_corrected_words():
    source_segment = TranscriptSegment(
        words=[WordTiming("colour", 0.0, 0.6, source="review_hint")],
        text="colour",
        start=0.0,
        end=0.6,
    )
    aligned_segment = {
        "text": "color",
        "start": 0.0,
        "end": 0.6,
        "words": [
            {
                "word": "color",
                "start": 0.12,
                "end": 0.54,
                "score": 0.93,
                "chars": [
                    {"char": "c", "start": 0.12, "end": 0.18, "score": 0.93},
                    {"char": "o", "start": 0.18, "end": 0.27, "score": 0.93},
                    {"char": "l", "start": 0.27, "end": 0.35, "score": 0.93},
                    {"char": "o", "start": 0.35, "end": 0.44, "score": 0.93},
                    {"char": "r", "start": 0.44, "end": 0.54, "score": 0.93},
                ],
            }
        ],
        "chars": [],
    }

    words, unaligned_count = _words_from_whisperx_segment(source_segment, aligned_segment)

    assert unaligned_count == 0
    assert len(words) == 1
    assert words[0].word == "colour"
    assert words[0].start == 0.12
    assert words[0].end == 0.54
    assert len(words[0].char_timings) == len("colour")


def test_whisperx_segment_score_penalizes_distorted_duration_profile():
    source_segment = TranscriptSegment(
        words=[
            WordTiming("one", 0.0, 1.0, source="review_hint"),
            WordTiming("two", 1.0, 2.0, source="review_hint"),
            WordTiming("three", 2.0, 3.0, source="review_hint"),
        ],
        text="one two three",
        start=0.0,
        end=3.0,
    )
    balanced = {
        "words": [
            {"word": "one", "start": 0.0, "end": 0.8, "score": 0.9},
            {"word": "two", "start": 0.9, "end": 1.8, "score": 0.9},
            {"word": "three", "start": 1.9, "end": 3.0, "score": 0.9},
        ]
    }
    distorted = {
        "words": [
            {"word": "one", "start": 0.0, "end": 0.2, "score": 0.4},
            {"word": "two", "start": 0.2, "end": 0.4, "score": 0.4},
            {"word": "three", "start": 0.4, "end": 3.0, "score": 0.4},
        ]
    }

    assert _whisperx_segment_candidate_score(source_segment, balanced) > _whisperx_segment_candidate_score(
        source_segment,
        distorted,
    )
    assert _whisperx_segment_needs_retry(source_segment, distorted) is True
