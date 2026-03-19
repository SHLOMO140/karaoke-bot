import wave

import numpy as np

from karaoke.aligner import AutoHebrewAligner, SequenceHebrewAligner, _build_subwords_from_char_entries
from karaoke.exceptions import AlignmentError
from karaoke.models import TranscriptSegment, WordTiming


def _write_test_audio(path):
    sample_rate = 16_000
    duration_seconds = 3.2
    samples = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)

    def burst(start, end, frequency=220.0):
        start_index = int(start * sample_rate)
        end_index = int(end * sample_rate)
        timeline = np.arange(end_index - start_index, dtype=np.float32) / sample_rate
        samples[start_index:end_index] += 0.65 * np.sin(2 * np.pi * frequency * timeline)

    burst(0.20, 0.88, 220.0)
    burst(2.18, 2.86, 330.0)

    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _draft_segments():
    return [
        TranscriptSegment(
            words=[
                WordTiming("שלום", 0.0, 1.0, confidence=0.95, source="draft_whisper"),
                WordTiming("עולם", 2.0, 3.0, confidence=0.96, source="draft_whisper"),
            ],
            text="שלום עולם",
            start=0.0,
            end=3.0,
        )
    ]


def _approved_segments():
    return [
        TranscriptSegment(
            words=[
                WordTiming("שלום", 0.0, 1.0, source="review_hint"),
                WordTiming("יפה", 1.0, 2.0, source="review_hint"),
                WordTiming("עולם", 2.0, 3.0, source="review_hint"),
            ],
            text="שלום יפה עולם",
            start=0.0,
            end=3.0,
        )
    ]


def _aligned_segments():
    return [
        TranscriptSegment(
            words=[
                WordTiming("שלום", 0.0, 1.0, confidence=0.95, source="review_hint"),
                WordTiming("עולם", 2.0, 3.0, confidence=0.96, source="review_hint"),
            ],
            text="שלום עולם",
            start=0.0,
            end=3.0,
        )
    ]


def test_sequence_aligner_preserves_known_word_timings_and_marks_interpolated_words(tmp_path):
    wav_path = tmp_path / "vocals_16k.wav"
    _write_test_audio(wav_path)

    aligned = SequenceHebrewAligner().align(str(wav_path), _approved_segments(), _draft_segments())
    words = aligned.segments[0].words

    assert words[0].word == "שלום"
    assert words[0].aligned is True
    assert 0.16 <= words[0].start <= 0.28
    assert words[0].end <= words[1].start

    assert words[1].word == "יפה"
    assert words[1].aligned is False
    assert words[1].start >= words[0].end
    assert words[1].end <= words[2].start

    assert words[2].word == "עולם"
    assert 2.12 <= words[2].start <= 2.24
    assert 2.84 <= words[2].end <= 2.96
    assert words[0].subwords
    assert len(words[0].subwords) >= 4
    assert words[0].subwords[0].start == words[0].start
    assert words[0].subwords[-1].end == words[0].end
    assert all(subword.end >= subword.start for subword in words[0].subwords)
    assert aligned.unaligned_word_count == 1


def test_sequence_aligner_snaps_boundaries_to_video_frames(tmp_path):
    wav_path = tmp_path / "vocals_16k.wav"
    _write_test_audio(wav_path)

    aligned = SequenceHebrewAligner().align(
        str(wav_path),
        _aligned_segments(),
        _aligned_segments(),
        video_frame_rate=25.0,
    )
    words = aligned.segments[0].words
    frame = 1 / 25.0

    for word in words:
        assert abs((word.start / frame) - round(word.start / frame)) < 1e-6
        assert abs((word.end / frame) - round(word.end / frame)) < 1e-6


def test_build_subwords_from_char_entries_maps_hebrew_chars_to_letter_timings():
    subwords = _build_subwords_from_char_entries(
        "שלום",
        [
            {"char": "ש", "start": 0.10, "end": 0.18, "score": 0.91},
            {"char": "ל", "start": 0.18, "end": 0.24, "score": 0.93},
            {"char": "ו", "start": 0.24, "end": 0.36, "score": 0.89},
            {"char": "ם", "start": 0.36, "end": 0.51, "score": 0.88},
        ],
        0.10,
        0.51,
        0.9,
    )

    assert [subword.text for subword in subwords] == ["ש", "ל", "ו", "ם"]
    assert subwords[0].start == 0.10
    assert subwords[-1].end == 0.51
    assert all(subword.end >= subword.start for subword in subwords)
