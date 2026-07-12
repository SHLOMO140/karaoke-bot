"""Synthetic audio builders for end-to-end harmony and alignment tests.

Everything is pure numpy so tests are deterministic and need no network or
model downloads. Callers must guard with pytest.importorskip("numpy") /
("soundfile") because conftest only mocks heavy packages when they are
missing from the environment.
"""

from __future__ import annotations

import re

import numpy as np

_NOTE_OFFSETS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Interval sets for the chord qualities the synthetic fixtures need.
_QUALITY_INTERVALS = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "sus4": (0, 5, 7),
}

_CHORD_PATTERN = re.compile(r"^([A-G][#b]?)(maj7|m7|dim|sus4|m|7)?(?:/([A-G][#b]?))?$")


def note_frequency(pitch_class: str, octave: int = 3) -> float:
    """Frequency of a pitch class at the given octave (A4 = 440 Hz)."""
    semitones_from_a4 = _NOTE_OFFSETS[pitch_class] - _NOTE_OFFSETS["A"] + (octave - 4) * 12
    return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))


def parse_chord(label: str) -> tuple[list[float], float]:
    """Return ([chord tone frequencies], bass frequency) for a chord label."""
    match = _CHORD_PATTERN.match(label.strip())
    if match is None:
        raise ValueError(f"Unsupported chord label for synthesis: {label!r}")
    root, quality, bass = match.group(1), match.group(2) or "", match.group(3)
    intervals = _QUALITY_INTERVALS[quality]
    root_offset = _NOTE_OFFSETS[root]
    freqs = []
    for interval in intervals:
        offset = (root_offset + interval) % 12
        pitch_class = next(name for name, value in _NOTE_OFFSETS.items() if value == offset)
        freqs.append(note_frequency(pitch_class, octave=4))
    bass_name = bass or root
    return freqs, note_frequency(bass_name, octave=2)


def make_chord_wav(
    progression: list[str],
    bpm: float = 120.0,
    *,
    beats_per_chord: int = 2,
    repeats: int = 2,
    sr: int = 22050,
    bass_boost: float = 1.6,
    noise_level: float = 0.0,
    harmonics: tuple[float, ...] = (1.0, 0.5, 0.25),
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Render a chord progression to a mono float32 signal.

    Each chord tone gets `harmonics` partials with a gentle exponential decay
    per chord hit, plus a boosted bass note an octave or two below — enough
    structure for chroma/beat analysis without any real recording.
    """
    rng = np.random.default_rng(seed)
    beat_seconds = 60.0 / bpm
    chord_seconds = beat_seconds * beats_per_chord
    chord_samples = int(round(chord_seconds * sr))
    chunks: list[np.ndarray] = []
    for _ in range(repeats):
        for label in progression:
            tone_freqs, bass_freq = parse_chord(label)
            t = np.arange(chord_samples) / sr
            envelope = np.exp(-t * 1.2)
            # Re-articulate on every beat so beat tracking has onsets to find.
            for beat in range(beats_per_chord):
                beat_start = int(beat * beat_seconds * sr)
                beat_t = t[beat_start:] - t[beat_start]
                envelope[beat_start:] = np.maximum(envelope[beat_start:], np.exp(-beat_t * 1.2) * 0.9)
            chunk = np.zeros(chord_samples, dtype=np.float64)
            for freq in tone_freqs:
                for partial_index, weight in enumerate(harmonics, start=1):
                    chunk += weight * np.sin(2 * np.pi * freq * partial_index * t)
            for partial_index, weight in enumerate(harmonics, start=1):
                chunk += bass_boost * weight * np.sin(2 * np.pi * bass_freq * partial_index * t)
            chunk *= envelope
            chunks.append(chunk)
    signal = np.concatenate(chunks)
    if noise_level > 0:
        signal = signal + rng.normal(0.0, noise_level * np.max(np.abs(signal)), signal.shape)
    peak = np.max(np.abs(signal)) or 1.0
    return (signal / peak * 0.8).astype(np.float32), sr


def make_vocal_like_wav(
    word_times: list[tuple[float, float]],
    *,
    sr: int = 16000,
    base_freq: float = 220.0,
    total_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """Voiced bursts at known onsets with true silence between them.

    Used for onset/refinement tests: the ground-truth word boundaries are the
    provided (start, end) pairs.
    """
    end_time = total_seconds if total_seconds is not None else max(end for _start, end in word_times) + 0.5
    signal = np.zeros(int(round(end_time * sr)), dtype=np.float64)
    for start, end in word_times:
        start_idx = int(round(start * sr))
        end_idx = min(int(round(end * sr)), len(signal))
        if end_idx <= start_idx:
            continue
        t = np.arange(end_idx - start_idx) / sr
        burst = np.sin(2 * np.pi * base_freq * t) + 0.4 * np.sin(2 * np.pi * base_freq * 2 * t)
        # Fast attack, gentle release — vocal-ish without being one.
        attack = np.minimum(t / 0.012, 1.0)
        release = np.minimum((t[-1] - t) / 0.03, 1.0) if len(t) else attack
        signal[start_idx:end_idx] += burst * attack * np.clip(release, 0.0, 1.0)
    peak = np.max(np.abs(signal)) or 1.0
    return (signal / peak * 0.8).astype(np.float32), sr


def write_wav(path, signal: np.ndarray, sr: int):
    import soundfile

    # PCM_16 so stdlib `wave` readers (aligner._load_audio_features) can parse it.
    soundfile.write(str(path), signal, sr, subtype="PCM_16")
