"""Tests for the conservative boundary-refinement clamp (T1).

wav2vec2-measured word boundaries must survive the energy heuristics;
unaligned/draft words keep the full ±BOUNDARY_SEARCH_SEC freedom.
"""

import pytest

np = pytest.importorskip("numpy")
if not hasattr(np, "ndarray"):  # conftest stub when numpy is missing
    pytest.skip("real numpy required", allow_module_level=True)

from karaoke import aligner
from karaoke.models import SubWordTiming, WordTiming

HOP = 0.001  # 1ms frames -> frame index == milliseconds


def _make_features(total_seconds: float, word_spans, onset_times):
    frames = int(total_seconds / HOP)
    energy = np.full(frames, 0.01, dtype=np.float32)
    for start, end in word_spans:
        energy[int(start / HOP):int(end / HOP)] = 1.0
    onsets = np.array(sorted(int(t / HOP) for t in onset_times), dtype=np.int64)
    zeros = np.zeros(frames, dtype=np.float32)
    return aligner.AudioFeatures(
        energy=energy,
        zcr=zeros,
        spectral_flux=zeros,
        onsets=onsets,
        hop_seconds=HOP,
    )


def _protected_word(text="שלום", start=1.0, end=1.4, confidence=0.9):
    graphemes = aligner._split_graphemes(text)
    step = (end - start) / len(graphemes)
    subwords = [
        SubWordTiming(
            text=grapheme,
            start=start + index * step,
            end=start + (index + 1) * step,
            confidence=confidence,
        )
        for index, grapheme in enumerate(graphemes)
    ]
    return WordTiming(
        word=text,
        start=start,
        end=end,
        confidence=confidence,
        source="whisperx",
        aligned=True,
        subwords=subwords,
    )


def test_aligned_word_shift_is_clamped(monkeypatch):
    # Spurious onset 80ms before the true (wav2vec2) start: inside the search
    # window for a 0.9-confidence word, so legacy refinement would snap to it.
    features = _make_features(3.0, [(1.0, 1.4)], onset_times=[0.92])
    monkeypatch.setattr(aligner, "_load_audio_features", lambda path: features)

    word = _protected_word()
    refined = aligner._refine_word_timings("fake.wav", [word])

    assert abs(refined[0].start - 1.0) <= aligner.ALIGNED_BOUNDARY_CLAMP_SEC + 1e-6
    assert abs(refined[0].end - 1.4) <= aligner.ALIGNED_BOUNDARY_CLAMP_SEC + 1e-6


def test_legacy_mode_keeps_old_snapping(monkeypatch):
    features = _make_features(3.0, [(1.0, 1.4)], onset_times=[0.92])
    monkeypatch.setattr(aligner, "_load_audio_features", lambda path: features)
    monkeypatch.setattr(aligner, "TIMING_REFINE_MODE", "legacy")

    word = _protected_word()
    refined = aligner._refine_word_timings("fake.wav", [word])

    assert refined[0].start == pytest.approx(0.92, abs=0.01)


def test_unaligned_word_still_snaps_to_onset(monkeypatch):
    features = _make_features(4.0, [(2.0, 2.4)], onset_times=[1.9])
    monkeypatch.setattr(aligner, "_load_audio_features", lambda path: features)

    word = WordTiming("עולם", 2.0, 2.4, confidence=0.0, source="draft_whisper", aligned=False)
    refined = aligner._refine_word_timings("fake.wav", [word])

    assert refined[0].start == pytest.approx(1.9, abs=0.01)


def test_whisperx_char_timings_survive_refinement(monkeypatch):
    features = _make_features(3.0, [(1.0, 1.4)], onset_times=[0.92])
    monkeypatch.setattr(aligner, "_load_audio_features", lambda path: features)
    calls = []
    original_builder = aligner._build_subword_timings

    def spy(*args, **kwargs):
        calls.append(args)
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(aligner, "_build_subword_timings", spy)

    protected = _protected_word()
    plain = WordTiming("עולם", 2.0, 2.4, confidence=0.0, source="draft_whisper", aligned=False)
    refined = aligner._refine_word_timings("fake.wav", [protected, plain])

    graphemes = aligner._split_graphemes(protected.word)
    assert [sub.text for sub in refined[0].subwords] == graphemes
    # The protected word never falls back to the energy/ZCR rebuilder.
    assert all(args[0].word != protected.word for args in calls)
    assert any(args[0].word == plain.word for args in calls)


def test_refinement_keeps_monotonic_non_overlapping_words(monkeypatch):
    features = _make_features(
        4.0, [(1.0, 1.3), (1.35, 1.7), (2.0, 2.5)], onset_times=[0.95, 1.34, 1.98]
    )
    monkeypatch.setattr(aligner, "_load_audio_features", lambda path: features)

    words = [
        _protected_word("שיר", 1.0, 1.3),
        WordTiming("חדש", 1.35, 1.7, confidence=0.4, source="draft_whisper", aligned=False),
        _protected_word("לגמרי", 2.0, 2.5),
    ]
    refined = aligner._refine_word_timings("fake.wav", words)

    for index, word in enumerate(refined):
        assert word.end > word.start
        if index:
            assert word.start >= refined[index - 1].end - 1e-6


def test_preserve_approved_path_reports_provenance(monkeypatch):
    from karaoke.models import TranscriptSegment

    approved_segments = [
        TranscriptSegment(
            words=[WordTiming("שלום", 7.7, 8.6, confidence=0.0, source="review_hint", aligned=False)],
            text="שלום",
            start=7.7,
            end=8.6,
        ),
    ]
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("שלום", 7.7, 8.6, confidence=0.9, source="draft_whisper")],
            text="שלום",
            start=7.7,
            end=8.6,
        ),
    ]
    regressed = [
        TranscriptSegment(
            words=[WordTiming("שלום", 1.0, 1.5, confidence=0.9, source="forced_aligner", aligned=True)],
            text="שלום",
            start=1.0,
            end=1.5,
        ),
    ]
    monkeypatch.setattr(aligner, "_rebuild_segments", lambda template, final_words: regressed)
    monkeypatch.setattr(aligner, "_refine_segments", lambda path, segments, video_frame_rate=None: segments)
    monkeypatch.setattr(aligner, "_validate_final_segments", lambda segments: segments)

    sequence = aligner.SequenceHebrewAligner()
    sequence.align("dummy.wav", approved_segments, draft_segments)

    assert sequence.last_provider_used == "approved_preserved"
    assert sequence.last_warning_message


def test_auto_aligner_tracks_provider_on_fallback():
    from karaoke.exceptions import AlignmentError
    from karaoke.models import AlignedTranscript, TranscriptSegment

    word = WordTiming("שלום", 0.0, 0.6, confidence=0.95, source="forced_aligner", aligned=True)
    result = AlignedTranscript(
        segments=[TranscriptSegment(words=[word], text="שלום", start=0.0, end=0.6)],
        provider="test",
        fully_aligned=True,
        unaligned_word_count=0,
    )

    class _Broken:
        def align(self, *args, **kwargs):
            raise AlignmentError("broken", "broken")

    class _Fallback:
        def align(self, *args, **kwargs):
            return result

    auto = aligner.AutoHebrewAligner()
    auto.primary = _Broken()
    auto.fallback = _Fallback()
    auto.align("dummy.wav", result.segments, result.segments)

    assert auto.last_provider_used == "sequence"
    assert auto.last_warning_message


def test_padding_attempts_add_union_window_for_isolated_segments():
    from karaoke.models import TranscriptSegment

    def _segment(start, end):
        return TranscriptSegment(
            words=[WordTiming("מילה", start, end, confidence=0.9)], text="מילה", start=start, end=end
        )

    # Tightly packed neighbors -> no union attempt beyond the fixed retries.
    packed = [_segment(0.0, 1.0), _segment(1.1, 2.0), _segment(2.1, 3.0)]
    assert aligner._padding_attempts_for_segment(packed, 1) == [
        aligner.WHISPERX_SEGMENT_PADDING_SEC,
        aligner.WHISPERX_SEGMENT_RETRY_PADDING_SEC,
    ]

    # Big gaps around the segment -> a third, wider (but capped) attempt.
    isolated = [_segment(0.0, 1.0), _segment(5.0, 6.0), _segment(12.0, 13.0)]
    attempts = aligner._padding_attempts_for_segment(isolated, 1)
    assert len(attempts) == 3
    assert attempts[2] == aligner.WHISPERX_SEGMENT_UNION_PADDING_CAP_SEC


def test_snap_segment_start_to_silence_edge():
    from karaoke.models import TranscriptSegment

    # Energy silent until 0.95s, active afterwards; word believed to start at 1.05.
    features = _make_features(3.0, [(0.95, 1.6)], onset_times=[])
    word = WordTiming("שלום", 1.05, 1.5, confidence=0.9, source="whisperx", aligned=True)
    segment = TranscriptSegment(words=[word], text="שלום", start=1.05, end=1.5)

    snapped = aligner._snap_segment_start_to_silence_edge(segment, features)

    assert snapped.words[0].start == pytest.approx(0.95, abs=0.01)
    # Never moves the start forward:
    early = TranscriptSegment(
        words=[WordTiming("שלום", 0.9, 1.5, confidence=0.9)], text="שלום", start=0.9, end=1.5
    )
    assert aligner._snap_segment_start_to_silence_edge(early, features).words[0].start <= 0.9 + 1e-6


def test_real_audio_roundtrip_respects_clamp(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    del soundfile
    import synthetic_audio

    word_spans = [(0.5, 0.9), (1.2, 1.6), (2.0, 2.45)]
    signal, sr = synthetic_audio.make_vocal_like_wav(word_spans)
    wav_path = tmp_path / "vocals.wav"
    synthetic_audio.write_wav(wav_path, signal, sr)

    words = [
        _protected_word(text, start, end)
        for text, (start, end) in zip(["שיר", "חדש", "לגמרי"], word_spans)
    ]
    refined = aligner._refine_word_timings(str(wav_path), words)

    for original, word in zip(words, refined):
        assert abs(word.start - original.start) <= aligner.ALIGNED_BOUNDARY_CLAMP_SEC + 1e-6
        assert abs(word.end - original.end) <= aligner.ALIGNED_BOUNDARY_CLAMP_SEC + 1e-6
