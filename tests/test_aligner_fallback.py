import wave

import numpy as np

from karaoke.aligner import AutoHebrewAligner, SequenceHebrewAligner
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


def test_auto_aligner_falls_back_when_whisperx_raises(tmp_path):
    wav_path = tmp_path / "vocals_16k.wav"
    _write_test_audio(wav_path)
    expected = SequenceHebrewAligner().align(str(wav_path), _approved_segments(), _draft_segments())

    auto = AutoHebrewAligner()

    class _BrokenPrimary:
        def align(self, *args, **kwargs):
            raise AlignmentError("broken", "broken")

    class _FallbackStub:
        def align(self, *args, **kwargs):
            return expected

    auto.primary = _BrokenPrimary()
    auto.fallback = _FallbackStub()

    aligned = auto.align(str(wav_path), _approved_segments(), _draft_segments())

    assert aligned is expected
    assert auto.last_warning_message
