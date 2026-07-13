"""End-to-end tests for LibrosaHarmonyAnalyzer.analyze on synthetic audio (C0).

These are the first tests that exercise the real audio->chords path. They are
the regression gate for the beat-sync/Viterbi/confidence work (C1/C2): label
accuracy, key, BPM and meter must not regress.
"""

import pytest

librosa = pytest.importorskip("librosa")
pytest.importorskip("soundfile")
if not hasattr(librosa, "load"):
    pytest.skip("real librosa required", allow_module_level=True)

import synthetic_audio
from karaoke.harmony import LibrosaHarmonyAnalyzer, summarize_song_analysis_quality


def _analyze(tmp_path_factory, name, progression, bpm, **kwargs):
    signal, sr = synthetic_audio.make_chord_wav(progression, bpm, **kwargs)
    wav_path = tmp_path_factory.mktemp("harmony") / f"{name}.wav"
    synthetic_audio.write_wav(wav_path, signal, sr)
    return LibrosaHarmonyAnalyzer().analyze(str(wav_path))


@pytest.fixture(scope="module")
def pop_major(tmp_path_factory):
    return _analyze(tmp_path_factory, "pop", ["C", "Am", "F", "G"], 120)


@pytest.fixture(scope="module")
def minor_key(tmp_path_factory):
    return _analyze(tmp_path_factory, "minor", ["Am", "Dm", "E7", "Am"], 96)


@pytest.fixture(scope="module")
def noisy(tmp_path_factory):
    return _analyze(tmp_path_factory, "noisy", ["C", "Am", "F", "G"], 120, noise_level=0.08)


@pytest.fixture(scope="module")
def waltz(tmp_path_factory):
    return _analyze(tmp_path_factory, "waltz", ["C", "G", "Am", "F"], 90, beats_per_chord=3)


def _label_accuracy(analysis, progression, bpm, *, beats_per_chord=2):
    """Duration-weighted fraction of the timeline labeled with the right chord."""
    chord_seconds = 60.0 / bpm * beats_per_chord

    def expected_at(t: float) -> str:
        return progression[int(t // chord_seconds) % len(progression)]

    correct = 0
    total = 0
    for event in analysis.chord_events:
        cursor = event.start
        while cursor < event.end:
            midpoint = cursor + 0.05
            # Skip slices near chord boundaries: those disagreements are
            # rounding, not detection errors.
            offset = midpoint % chord_seconds
            if min(offset, chord_seconds - offset) > 0.08:
                total += 1
                if event.label == expected_at(midpoint):
                    correct += 1
            cursor += 0.1
    return correct / max(1, total)


def test_pop_major_progression_labels(pop_major):
    assert _label_accuracy(pop_major, ["C", "Am", "F", "G"], 120) >= 0.85


def test_minor_progression_labels(minor_key):
    assert _label_accuracy(minor_key, ["Am", "Dm", "E7", "Am"], 96) >= 0.85


def test_noise_robustness(noisy):
    assert _label_accuracy(noisy, ["C", "Am", "F", "G"], 120) >= 0.75


def test_bpm_detection(pop_major, minor_key):
    def _bpm_ok(measured, expected):
        return any(abs(measured - candidate) <= 6 for candidate in (expected, expected / 2, expected * 2))

    assert _bpm_ok(pop_major.bpm, 120)
    assert _bpm_ok(minor_key.bpm, 96)


def test_key_detection_major_and_relative_minor(pop_major, minor_key):
    assert pop_major.original_key == "C"
    assert minor_key.original_key == "Am"


def test_waltz_time_signature(waltz):
    assert waltz.time_signature == 3


def test_clean_fixture_confidence_calibration(pop_major):
    quality = summarize_song_analysis_quality(pop_major)
    assert quality.average_confidence >= 0.6
    assert quality.low_confidence_ratio <= 0.1
    assert quality.reliable_for_delivery


def test_no_single_beat_flicker(tmp_path):
    import numpy as np

    signal, sr = synthetic_audio.make_chord_wav(["C"], 120, beats_per_chord=8, repeats=2)
    # Corrupt a quarter second in the middle with noise: the transition prior
    # must hold the chord instead of emitting a one-beat flicker.
    rng = np.random.default_rng(1)
    start = int(4.0 * sr)
    end = int(4.25 * sr)
    signal = signal.copy()
    signal[start:end] = rng.normal(0.0, 0.3, end - start).astype("float32")
    wav_path = tmp_path / "flicker.wav"
    synthetic_audio.write_wav(wav_path, signal, sr)

    analysis = LibrosaHarmonyAnalyzer().analyze(str(wav_path))

    foreign_duration = sum(
        event.end - event.start
        for event in analysis.chord_events
        if event.label not in ("C", "N")
    )
    assert foreign_duration <= 0.5
