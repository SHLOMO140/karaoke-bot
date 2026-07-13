"""Music analysis helpers for BPM, chords, time signature, chord sheets, and overlays."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from .exceptions import MusicAnalysisError
from .models import ChordEvent, SongAnalysis, TranscriptSegment

# Harmonic/percussive separation aggressiveness. The analysis input is usually
# the demucs instrumental stem (already vocal-free), so a second aggressive
# HPSS pass can strip real harmonic content; 0 disables the pass entirely.
_HPSS_MARGIN = float(os.getenv("KARAOKE_HARMONY_HPSS_MARGIN", "3.0"))

logger = logging.getLogger(__name__)

ANALYSIS_SAMPLE_RATE = 22_050
ANALYSIS_HOP_LENGTH = 512
NO_CHORD_LABEL = "N"
NOTE_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
NOTE_TO_PITCH = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}
CHORD_LABEL_PATTERN = re.compile(
    r"^\s*(?P<root>[A-Ga-g](?:#|b)?)(?P<suffix>.*?)(?:/(?P<bass>[A-Ga-g](?:#|b)?))?\s*$"
)
MAJOR_KEY_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_KEY_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
EASY_KEY_TARGET = "Am"
EASY_KEY_TARGET_PITCH = NOTE_TO_PITCH["A"]

# --- Chord templates: (suffix, quality_name, 12-bin pitch-class profile, bias) ---
# Each template encodes expected energy at each semitone interval from root.
# Bias penalizes extended chords slightly so triads win when evidence is equal.
CHORD_TEMPLATES = (
    ("",     "major",      [1.0,  0.0,  0.0,  0.0,  0.85, 0.0,  0.0,  0.95, 0.0,  0.0,  0.0,  0.0 ], 1.00),
    ("m",    "minor",      [1.0,  0.0,  0.0,  0.85, 0.0,  0.0,  0.0,  0.95, 0.0,  0.0,  0.0,  0.0 ], 1.00),
    ("7",    "dominant7",  [1.0,  0.0,  0.0,  0.0,  0.78, 0.0,  0.0,  0.90, 0.0,  0.0,  0.72, 0.0 ], 0.96),
    ("maj7", "major7",     [1.0,  0.0,  0.0,  0.0,  0.78, 0.0,  0.0,  0.90, 0.0,  0.0,  0.0,  0.72], 0.96),
    ("m7",   "minor7",     [1.0,  0.0,  0.0,  0.78, 0.0,  0.0,  0.0,  0.90, 0.0,  0.0,  0.72, 0.0 ], 0.96),
    ("sus2", "sus2",       [1.0,  0.0,  0.78, 0.0,  0.0,  0.0,  0.0,  0.92, 0.0,  0.0,  0.0,  0.0 ], 0.91),
    ("sus4", "sus4",       [1.0,  0.0,  0.0,  0.0,  0.0,  0.78, 0.0,  0.92, 0.0,  0.0,  0.0,  0.0 ], 0.91),
    ("dim",  "diminished", [1.0,  0.0,  0.0,  0.80, 0.0,  0.0,  0.80, 0.0,  0.0,  0.0,  0.0,  0.0 ], 0.89),
    ("aug",  "augmented",  [1.0,  0.0,  0.0,  0.0,  0.80, 0.0,  0.0,  0.0,  0.80, 0.0,  0.0,  0.0 ], 0.89),
    ("dim7", "diminished7",[1.0,  0.0,  0.0,  0.76, 0.0,  0.0,  0.76, 0.0,  0.0,  0.68, 0.0,  0.0 ], 0.87),
    ("m7b5", "half_diminished", [1.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 0.68, 0.0], 0.89),
    ("add9", "add9",       [1.0,  0.0,  0.62, 0.0,  0.80, 0.0,  0.0,  0.90, 0.0,  0.0,  0.0,  0.0 ], 0.88),
    ("m9",   "minor9",     [1.0,  0.0,  0.62, 0.76, 0.0,  0.0,  0.0,  0.88, 0.0,  0.0,  0.68, 0.0 ], 0.86),
)

# Detection thresholds — raised for fewer false positives
_MIN_MATCH_SCORE = 0.32
_MIN_CONTRAST = 0.020
_SMOOTHING_CONFIDENCE = 0.78
# Viterbi transition prior scale (see _decode_chord_path). 4.0 was strong
# enough to freeze on a stale chord through quiet passages where another
# candidate consistently scored higher; 3.0 keeps flicker suppression while
# letting a persistent per-frame winner take over.
_TRANSITION_WEIGHT = 3.0
# Confidence calibration weights (see _collect_chord_candidates).
_CONFIDENCE_SCORE_WEIGHT = 1.0
_CONFIDENCE_CONTRAST_WEIGHT = 2.2
_CONFIDENCE_CORE_WEIGHT = 0.22
_SMOOTHING_MAX_DURATION = 0.32
_GAP_FILL_MAX_DURATION = 0.20
_MAX_CANDIDATES_PER_SEGMENT = 8
_CHORD_CHANGE_SIGMA = 0.80
_LOW_CONFIDENCE_EVENT_THRESHOLD = 0.35
_MIN_DELIVERY_AVERAGE_CONFIDENCE = 0.52
_MAX_DELIVERY_LOW_CONFIDENCE_RATIO = 0.45
_STABLE_NOTE_QUANTILE = 0.72
_SHORT_SEGMENT_CONTEXT_RATIO = 0.90
_CONTEXT_BLEND_WEIGHT = 0.18
_TRIAD_SUPPORT_BONUS = 0.025
_EXTENSION_MIN_STABLE_SUPPORT = 0.16
_EXTENSION_WEAK_SUPPORT_PENALTY = 0.090
_SUSPENDED_THIRD_PENALTY = 0.11


@dataclass(frozen=True)
class SongAnalysisQualitySummary:
    visible_chord_count: int = 0
    average_confidence: float = 0.0
    low_confidence_ratio: float = 0.0
    chords_per_minute: float = 0.0
    has_external_source: bool = False
    has_text_only_cache: bool = False
    reliable_for_delivery: bool = False


def _lazy_import_analysis_libs() -> tuple[Any, Any]:
    try:
        import librosa
        import numpy as np
    except Exception as exc:
        raise MusicAnalysisError(
            f"Optional music-analysis dependency is missing: {exc}",
            "חסרה ספריית ניתוח מוזיקלי. הרץ install.bat מחדש כדי להתקין את רכיבי האקורדים.",
        ) from exc
    return np, librosa


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else 0.0
    return float(value)


def _normalize_boundaries(boundaries: list[int], frame_count: int) -> list[int]:
    normalized = sorted({max(0, min(frame_count, int(boundary))) for boundary in boundaries})
    if not normalized:
        normalized = [0]
    if normalized[0] != 0:
        normalized.insert(0, 0)
    if normalized[-1] != frame_count:
        normalized.append(frame_count)
    return normalized


def _normalize_vector(vector: Any, np: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return vector
    return vector / norm


def _normalize_note_name(note_name: str) -> str:
    stripped = (note_name or "").strip()
    if not stripped:
        return ""
    stripped = stripped.replace("♯", "#").replace("♭", "b")
    head = stripped[0].upper()
    accidental = stripped[1:2]
    if accidental not in {"#", "b"}:
        accidental = ""
    return f"{head}{accidental}"


def _note_to_pitch(note_name: str) -> int | None:
    normalized = _normalize_note_name(note_name)
    return NOTE_TO_PITCH.get(normalized)


def _pitch_to_note(pitch_class: int) -> str:
    return NOTE_NAMES[int(pitch_class) % 12]


def _parse_chord_label(label: str) -> tuple[str, str, str | None] | None:
    normalized = (label or "").strip()
    if not normalized or normalized == NO_CHORD_LABEL:
        return None
    match = CHORD_LABEL_PATTERN.match(normalized)
    if not match:
        return None

    root = _normalize_note_name(match.group("root") or "")
    if root not in NOTE_TO_PITCH:
        return None

    bass = _normalize_note_name(match.group("bass") or "")
    if bass and bass not in NOTE_TO_PITCH:
        bass = ""

    suffix = (match.group("suffix") or "").strip()
    return root, suffix, bass or None


def transpose_chord_label(label: str, semitones: int) -> str:
    parsed = _parse_chord_label(label)
    if not parsed:
        return label

    root, suffix, bass = parsed
    transposed_root = _pitch_to_note((NOTE_TO_PITCH[root] + semitones) % 12)
    transposed_bass = _pitch_to_note((NOTE_TO_PITCH[bass] + semitones) % 12) if bass else ""
    bass_suffix = f"/{transposed_bass}" if transposed_bass else ""
    return f"{transposed_root}{suffix}{bass_suffix}"


def _clone_chord_event(
    event: ChordEvent,
    *,
    label: str | None = None,
    start: float | None = None,
    end: float | None = None,
    root: str | None = None,
    quality: str | None = None,
) -> ChordEvent:
    return ChordEvent(
        label=event.label if label is None else label,
        start=event.start if start is None else start,
        end=event.end if end is None else end,
        confidence=event.confidence,
        root=event.root if root is None else root,
        quality=event.quality if quality is None else quality,
    )


def _event_root_name(event: ChordEvent) -> str:
    normalized = _normalize_note_name(event.root)
    if normalized in NOTE_TO_PITCH:
        return normalized
    parsed = _parse_chord_label(event.label)
    if not parsed:
        return ""
    return parsed[0]


def _event_pitch_class(event: ChordEvent) -> int | None:
    root_name = _event_root_name(event)
    if not root_name:
        return None
    return NOTE_TO_PITCH[root_name]


def summarize_song_analysis_quality(analysis: SongAnalysis) -> SongAnalysisQualitySummary:
    candidate_events = analysis.chord_events or analysis.original_chord_events
    visible_events = [
        event
        for event in candidate_events
        if event.label and event.label != NO_CHORD_LABEL and event.end > event.start + 1e-3
    ]
    has_external_source = bool(
        (analysis.chord_source_name or "").strip() or (analysis.chord_source_url or "").strip()
    )
    has_text_only_cache = bool(analysis.chord_sheet_text.strip()) and not visible_events

    if not visible_events:
        return SongAnalysisQualitySummary(
            has_external_source=has_external_source,
            has_text_only_cache=has_text_only_cache,
            reliable_for_delivery=has_external_source or has_text_only_cache,
        )

    confidences = [max(0.0, min(1.0, float(event.confidence))) for event in visible_events]
    average_confidence = sum(confidences) / len(confidences)
    low_confidence_ratio = sum(
        1 for confidence in confidences if confidence < _LOW_CONFIDENCE_EVENT_THRESHOLD
    ) / len(confidences)
    start_time = min(event.start for event in visible_events)
    end_time = max(event.end for event in visible_events)
    duration_seconds = max(0.0, end_time - start_time)
    chords_per_minute = (len(visible_events) * 60.0 / duration_seconds) if duration_seconds > 1e-3 else 0.0
    reliable_for_delivery = has_external_source or has_text_only_cache or (
        average_confidence >= _MIN_DELIVERY_AVERAGE_CONFIDENCE
        and low_confidence_ratio <= _MAX_DELIVERY_LOW_CONFIDENCE_RATIO
    )

    return SongAnalysisQualitySummary(
        visible_chord_count=len(visible_events),
        average_confidence=average_confidence,
        low_confidence_ratio=low_confidence_ratio,
        chords_per_minute=chords_per_minute,
        has_external_source=has_external_source,
        has_text_only_cache=has_text_only_cache,
        reliable_for_delivery=reliable_for_delivery,
    )


def _quality_bucket(event: ChordEvent) -> str:
    quality = (event.quality or "").strip().lower()
    label = (event.label or "").strip().lower()
    parsed = _parse_chord_label(event.label)
    suffix = (parsed[1].lower() if parsed else label).strip()
    if quality in {"half_diminished"} or "m7b5" in label:
        return "half_diminished"
    if quality in {"diminished", "diminished7"} or "dim" in label:
        return "diminished"
    if quality == "augmented" or "aug" in label:
        return "augmented"
    if quality in {"dominant7"} or suffix.startswith("7") or suffix.startswith("9"):
        return "dominant"
    if quality.startswith("minor") or (suffix.startswith("m") and not suffix.startswith("maj")):
        return "minor"
    return "major"


def infer_song_key(chord_events: list[ChordEvent]) -> tuple[str, int | None, str]:
    visible_events = [event for event in chord_events if _event_pitch_class(event) is not None]
    if not visible_events:
        return "", None, ""

    best_score = float("-inf")
    best_tonic: int | None = None
    best_mode = ""

    scores_by_key: dict[tuple[int, str], float] = {}
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_KEY_PROFILE), ("minor", MINOR_KEY_PROFILE)):
            total_weight = 0.0
            score = 0.0
            for event in visible_events:
                pitch_class = _event_pitch_class(event)
                if pitch_class is None:
                    continue
                weight = max(event.end - event.start, 0.12) * max(event.confidence, 0.45)
                interval = (pitch_class - tonic) % 12
                chord_kind = _quality_bucket(event)

                total_weight += weight
                score += profile[interval] * weight

                if interval == 0:
                    if mode == "major" and chord_kind in {"major", "dominant"}:
                        score += 2.6 * weight
                    if mode == "minor" and chord_kind in {"minor", "diminished", "half_diminished"}:
                        score += 2.6 * weight
                elif interval == 7 and chord_kind in {"major", "dominant"}:
                    score += 1.1 * weight
                elif interval == 5:
                    if mode == "major" and chord_kind == "major":
                        score += 0.7 * weight
                    if mode == "minor" and chord_kind == "minor":
                        score += 0.7 * weight
                elif interval == 3 and mode == "major" and chord_kind == "minor":
                    score += 0.35 * weight
                elif interval == 9 and mode == "minor" and chord_kind == "major":
                    score += 0.35 * weight

            if total_weight <= 0:
                continue

            score /= total_weight
            last_pitch = _event_pitch_class(visible_events[-1])
            if last_pitch is not None:
                last_interval = (last_pitch - tonic) % 12
                if last_interval == 0:
                    score += 0.85
                elif last_interval == 7:
                    score += 0.15

            scores_by_key[(tonic, mode)] = score
            if score > best_score:
                best_score = score
                best_tonic = tonic
                best_mode = mode

    if best_tonic is None:
        return "", None, ""

    # Relative major/minor near-ties: prefer the mode whose tonic triad is
    # actually played more (Krumhansl profiles alone often confuse Am vs C).
    counterpart = (
        ((best_tonic + 9) % 12, "minor") if best_mode == "major" else ((best_tonic + 3) % 12, "major")
    )
    counterpart_score = scores_by_key.get(counterpart)
    if counterpart_score is not None and abs(best_score - counterpart_score) <= 0.05 * max(abs(best_score), 1e-6):

        def _tonic_presence(tonic: int, mode: str) -> float:
            kinds = {"major", "dominant"} if mode == "major" else {"minor", "diminished", "half_diminished"}
            return sum(
                max(event.end - event.start, 0.0)
                for event in visible_events
                if _event_pitch_class(event) == tonic and _quality_bucket(event) in kinds
            )

        if _tonic_presence(*counterpart) > _tonic_presence(best_tonic, best_mode) * 1.15:
            best_tonic, best_mode = counterpart

    key_label = _pitch_to_note(best_tonic)
    if best_mode == "minor":
        key_label = f"{key_label}m"
    return key_label, best_tonic, best_mode


def infer_key_from_chroma_profile(chroma: Any, np: Any) -> tuple[str, int | None, str]:
    if getattr(chroma, "size", 0) == 0:
        return "", None, ""

    profile = np.sum(chroma.astype(float), axis=1)
    total = float(np.sum(profile))
    if total <= 1e-6:
        return "", None, ""

    normalized = profile / total
    major_profile = np.array(MAJOR_KEY_PROFILE, dtype=float)
    major_profile = major_profile / max(float(np.sum(major_profile)), 1e-6)
    minor_profile = np.array(MINOR_KEY_PROFILE, dtype=float)
    minor_profile = minor_profile / max(float(np.sum(minor_profile)), 1e-6)

    best_score = float("-inf")
    best_tonic: int | None = None
    best_mode = ""
    for tonic in range(12):
        major_score = float(np.dot(normalized, np.roll(major_profile, tonic)))
        if major_score > best_score:
            best_score = major_score
            best_tonic = tonic
            best_mode = "major"

        minor_score = float(np.dot(normalized, np.roll(minor_profile, tonic)))
        if minor_score > best_score:
            best_score = minor_score
            best_tonic = tonic
            best_mode = "minor"

    if best_tonic is None:
        return "", None, ""

    key_label = _pitch_to_note(best_tonic)
    if best_mode == "minor":
        key_label = f"{key_label}m"
    return key_label, best_tonic, best_mode


def _parse_key_label(key_label: str) -> tuple[int, str] | None:
    normalized = (key_label or "").strip()
    if not normalized:
        return None
    mode = "minor" if len(normalized) > 1 and normalized.endswith("m") else "major"
    root = normalized[:-1] if mode == "minor" else normalized
    pitch = _note_to_pitch(root)
    if pitch is None:
        return None
    return pitch, mode


def _transpose_semitones_to_target_key(tonic_pitch: int, mode: str, target_key: str) -> int:
    parsed_target = _parse_key_label(target_key)
    if not parsed_target:
        return 0

    target_pitch, target_mode = parsed_target
    reference_pitch = tonic_pitch
    if mode == "major" and target_mode == "minor":
        reference_pitch = (tonic_pitch + 9) % 12
    elif mode == "minor" and target_mode == "major":
        reference_pitch = (tonic_pitch + 3) % 12

    semitones = target_pitch - reference_pitch
    while semitones <= -7:
        semitones += 12
    while semitones > 6:
        semitones -= 12
    return semitones


def _transpose_semitones_to_easy_key(tonic_pitch: int, mode: str) -> int:
    return _transpose_semitones_to_target_key(tonic_pitch, mode, EASY_KEY_TARGET)


def _transpose_chord_event(event: ChordEvent, semitones: int) -> ChordEvent:
    transposed_label = transpose_chord_label(event.label, semitones)
    parsed = _parse_chord_label(transposed_label)
    transposed_root = parsed[0] if parsed else event.root
    return _clone_chord_event(event, label=transposed_label, root=transposed_root)


def _normalize_display_events_for_segments(
    chord_events: list[ChordEvent],
    segments: list[TranscriptSegment],
) -> list[ChordEvent]:
    visible_events = [
        _clone_chord_event(event)
        for event in sorted(chord_events, key=lambda item: (item.start, item.end))
        if event.label and event.label != NO_CHORD_LABEL and event.end > event.start + 1e-3
    ]
    if not visible_events:
        return []

    lyric_segments = [segment for segment in segments if segment.words]
    if not lyric_segments:
        return visible_events

    lyric_start = min(segment.start for segment in lyric_segments)
    lyric_end = max(segment.end for segment in lyric_segments)
    adjusted: list[ChordEvent] = []

    for event in visible_events:
        current = _clone_chord_event(event, start=max(0.0, event.start), end=max(event.end, event.start))
        if not adjusted:
            current.start = min(current.start, lyric_start)
            adjusted.append(current)
            continue

        previous = adjusted[-1]
        if current.start > previous.end:
            previous.end = current.start
        else:
            current.start = max(current.start, previous.end)

        if current.end <= current.start + 1e-3:
            current.end = max(current.start + 0.02, event.end)
        adjusted.append(current)

    adjusted[-1].end = max(adjusted[-1].end, lyric_end)
    return _merge_adjacent_events(adjusted)


def _recover_source_chord_events(analysis: SongAnalysis) -> list[ChordEvent]:
    if analysis.original_chord_events:
        return [_clone_chord_event(event) for event in analysis.original_chord_events]

    recovered = [_clone_chord_event(event) for event in analysis.chord_events]
    if recovered and analysis.transpose_semitones:
        return [_transpose_chord_event(event, -analysis.transpose_semitones) for event in recovered]
    return recovered


def resolve_song_analysis_key_labels(analysis: SongAnalysis) -> tuple[str, str]:
    original_key = (analysis.original_key or "").strip()
    target_key = (analysis.target_key or "").strip()

    source_events = _recover_source_chord_events(analysis)
    inferred_original_key, _, _ = infer_song_key(source_events)
    if inferred_original_key:
        original_key = inferred_original_key

    if target_key:
        inferred_target_key, _, _ = infer_song_key(analysis.chord_events)
        if inferred_target_key:
            target_key = inferred_target_key

    return original_key, target_key


def prepare_song_analysis_for_display(
    analysis: SongAnalysis,
    segments: list[TranscriptSegment],
    target_key: str = EASY_KEY_TARGET,
) -> SongAnalysis:
    requested_target_key = (target_key or "").strip()
    source_events = _recover_source_chord_events(analysis)
    base_events = [
        _clone_chord_event(event)
        for event in source_events
        if event.label and event.label != NO_CHORD_LABEL and event.end > event.start + 1e-3
    ]

    inferred_key, inferred_tonic_pitch, inferred_mode = infer_song_key(base_events)
    resolved_original_key = (analysis.original_key or inferred_key).strip()
    parsed_original_key = _parse_key_label(resolved_original_key)
    tonic_pitch = parsed_original_key[0] if parsed_original_key else inferred_tonic_pitch
    mode = parsed_original_key[1] if parsed_original_key else inferred_mode
    transpose_semitones = (
        _transpose_semitones_to_target_key(tonic_pitch, mode, requested_target_key)
        if tonic_pitch is not None and requested_target_key
        else 0
    )
    display_events = (
        [_transpose_chord_event(event, transpose_semitones) for event in base_events]
        if transpose_semitones
        else [_clone_chord_event(event) for event in base_events]
    )
    display_events = _normalize_display_events_for_segments(display_events, segments)

    return SongAnalysis(
        bpm=analysis.bpm,
        time_signature=analysis.time_signature,
        preview_window_seconds=analysis.preview_window_seconds,
        provider=analysis.provider,
        source_audio=analysis.source_audio,
        beat_times=list(analysis.beat_times),
        measure_times=list(analysis.measure_times),
        original_key=resolved_original_key,
        target_key=requested_target_key if requested_target_key and display_events else "",
        transpose_semitones=transpose_semitones,
        original_chord_events=base_events,
        chord_events=display_events,
        chord_sheet_text=analysis.chord_sheet_text,
        chord_source_name=analysis.chord_source_name,
        chord_source_url=analysis.chord_source_url,
    )


# ---------------------------------------------------------------------------
# Time signature detection via beat-strength autocorrelation
# ---------------------------------------------------------------------------

def _detect_time_signature(beat_times: list[float], np: Any) -> int:
    """Estimate time signature (3 or 4) from beat onset intervals.

    Uses autocorrelation of inter-beat intervals to detect periodicity.
    Returns 3 for waltz-like patterns, 4 otherwise (default).
    """
    if len(beat_times) < 8:
        return 4

    intervals = np.diff(beat_times)
    if len(intervals) < 6:
        return 4

    # Normalize intervals
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        return 4
    normalized = intervals / median_interval

    # Check for strong grouping every 3 or 4 beats
    # by looking at accent pattern in interval ratios
    score_3 = 0.0
    score_4 = 0.0

    # Autocorrelation approach: correlate interval series with itself
    # shifted by 3 and 4 positions
    max_lag = min(8, len(normalized) - 1)
    for lag in range(2, max_lag + 1):
        correlation = float(np.corrcoef(normalized[:-lag], normalized[lag:])[0, 1])
        if np.isnan(correlation):
            continue
        if lag % 3 == 0:
            score_3 += correlation * (1.0 / (lag / 3))
        if lag % 4 == 0:
            score_4 += correlation * (1.0 / (lag / 4))

    # Also check if beat intervals show a repeating 3-pattern:
    # In 3/4, every 3rd beat often has a slightly longer interval
    if len(normalized) >= 9:
        groups_3 = [normalized[i::3].std() for i in range(3) if len(normalized[i::3]) > 2]
        groups_4 = [normalized[i::4].std() for i in range(4) if len(normalized[i::4]) > 2]
        variance_3 = float(np.mean(groups_3)) if groups_3 else 1.0
        variance_4 = float(np.mean(groups_4)) if groups_4 else 1.0
        # Lower within-group variance suggests a better grouping
        if variance_3 < variance_4 * 0.8:
            score_3 += 0.5
        elif variance_4 < variance_3 * 0.8:
            score_4 += 0.5

    return 3 if score_3 > score_4 + 0.15 else 4


def _compute_measure_times(beat_times: list[float], time_signature: int) -> list[float]:
    """Group beat times into measure boundaries."""
    if not beat_times:
        return []
    measures = [beat_times[0]]
    for i in range(time_signature, len(beat_times), time_signature):
        measures.append(beat_times[i])
    return measures


# ---------------------------------------------------------------------------
# Beat-synchronous chroma extraction + chord classification
# ---------------------------------------------------------------------------

def _normalize_chroma_columns(chroma: Any, np: Any) -> Any:
    if getattr(chroma, "size", 0) == 0:
        return chroma
    totals = np.sum(chroma, axis=0, keepdims=True).astype(float)
    totals[totals <= 1e-6] = 1.0
    return chroma / totals


def _smooth_feature_frames(chroma: Any, np: Any) -> Any:
    try:
        from scipy.ndimage import median_filter

        return median_filter(chroma, size=(1, 5))
    except ImportError:
        kernel = 5
        if chroma.shape[1] <= kernel:
            return chroma
        padded = np.pad(chroma, ((0, 0), (kernel // 2, kernel // 2)), mode="edge")
        smoothed = np.zeros_like(chroma)
        for index in range(chroma.shape[1]):
            smoothed[:, index] = padded[:, index:index + kernel].mean(axis=1)
        return smoothed


def _extract_analysis_chroma(harmonic_audio: Any, sample_rate: int, np: Any, librosa: Any) -> tuple[Any, Any]:
    """Extract a blended chroma feature and a bass-focused chroma view."""
    try:
        tuning = float(librosa.estimate_tuning(y=harmonic_audio, sr=sample_rate))
    except Exception:
        tuning = 0.0

    chroma_sources: list[tuple[float, Any]] = []
    try:
        chroma_cqt = librosa.feature.chroma_cqt(
            y=harmonic_audio,
            sr=sample_rate,
            hop_length=ANALYSIS_HOP_LENGTH,
            bins_per_octave=36,
            n_chroma=12,
            tuning=tuning,
        )
        if getattr(chroma_cqt, "size", 0) != 0:
            chroma_sources.append((0.50, chroma_cqt))
    except Exception:
        pass

    try:
        chroma_cens = librosa.feature.chroma_cens(
            y=harmonic_audio,
            sr=sample_rate,
            hop_length=ANALYSIS_HOP_LENGTH,
            tuning=tuning,
        )
        if getattr(chroma_cens, "size", 0) != 0:
            chroma_sources.append((0.30, chroma_cens))
    except Exception:
        pass

    try:
        chroma_stft = librosa.feature.chroma_stft(
            y=harmonic_audio,
            sr=sample_rate,
            hop_length=ANALYSIS_HOP_LENGTH,
            tuning=tuning,
        )
        if getattr(chroma_stft, "size", 0) != 0:
            chroma_sources.append((0.20, chroma_stft))
    except Exception:
        pass

    if not chroma_sources:
        raise MusicAnalysisError("Chord analysis did not produce chroma features.")

    total_weight = sum(weight for weight, _source in chroma_sources)
    combined = sum(
        (_normalize_chroma_columns(source.astype(float), np) * (weight / total_weight))
        for weight, source in chroma_sources
    )
    combined = _smooth_feature_frames(combined, np)

    bass_chroma = np.zeros_like(combined)
    try:
        n_fft = 4096
        stft = np.abs(librosa.stft(harmonic_audio, n_fft=n_fft, hop_length=ANALYSIS_HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        bass_mask = (freqs >= 35.0) & (freqs <= 260.0)
        if bass_mask.any():
            bass_spec = np.zeros_like(stft)
            bass_spec[bass_mask, :] = stft[bass_mask, :]
            bass_chroma = librosa.feature.chroma_stft(
                S=bass_spec,
                sr=sample_rate,
                hop_length=ANALYSIS_HOP_LENGTH,
                tuning=tuning,
            )
            bass_chroma = _smooth_feature_frames(_normalize_chroma_columns(bass_chroma.astype(float), np), np)
    except Exception:
        bass_chroma = np.zeros_like(combined)

    return combined, bass_chroma


def _detect_chord_change_frames(raw_chroma: Any, beat_frames: list[int], np: Any) -> list[int]:
    if getattr(raw_chroma, "size", 0) == 0 or raw_chroma.shape[1] < 4:
        return []

    normalized = _normalize_chroma_columns(raw_chroma.astype(float), np)
    novelty = np.linalg.norm(np.diff(normalized, axis=1), axis=0)
    if novelty.size < 3:
        return []

    if novelty.size >= 5:
        novelty = np.convolve(novelty, np.ones(5, dtype=float) / 5.0, mode="same")

    threshold = float(np.median(novelty) + np.std(novelty) * _CHORD_CHANGE_SIGMA)
    if threshold <= 0:
        return []

    median_beat_gap = int(np.median(np.diff(beat_frames))) if len(beat_frames) >= 2 else 0
    min_spacing = max(4, median_beat_gap // 2) if median_beat_gap > 0 else 6
    selected: list[tuple[int, float]] = []
    for index in range(1, len(novelty) - 1):
        if novelty[index] < threshold:
            continue
        if novelty[index] < novelty[index - 1] or novelty[index] < novelty[index + 1]:
            continue
        frame = index + 1
        if any(abs(frame - beat_frame) <= 2 for beat_frame in beat_frames):
            continue
        if selected and frame - selected[-1][0] < min_spacing:
            if novelty[index] > selected[-1][1]:
                selected[-1] = (frame, float(novelty[index]))
            continue
        selected.append((frame, float(novelty[index])))

    return [frame for frame, _score in selected]


def _core_intervals_for_quality(quality_name: str) -> list[int]:
    quality = (quality_name or "").strip().lower()
    if quality in {"minor", "minor7", "minor9"}:
        return [0, 3, 7]
    if quality in {"dominant7", "major", "major7", "add9"}:
        return [0, 4, 7]
    if quality == "sus2":
        return [0, 2, 7]
    if quality == "sus4":
        return [0, 5, 7]
    if quality in {"diminished", "diminished7", "half_diminished"}:
        return [0, 3, 6]
    if quality == "augmented":
        return [0, 4, 8]
    return [0, 4, 7]


def _extension_intervals_for_quality(quality_name: str) -> list[int]:
    quality = (quality_name or "").strip().lower()
    if quality == "dominant7":
        return [10]
    if quality == "major7":
        return [11]
    if quality == "minor7":
        return [10]
    if quality == "minor9":
        return [2, 10]
    if quality == "add9":
        return [2]
    if quality == "diminished7":
        return [9]
    if quality == "half_diminished":
        return [10]
    return []


def _build_segment_feature_rows(
    raw_chroma: Any,
    bass_chroma: Any,
    boundaries: list[int],
    sample_rate: int,
    np: Any,
    librosa: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start_frame, end_frame in zip(boundaries, boundaries[1:]):
        if end_frame <= start_frame:
            continue

        chroma_window = raw_chroma[:, start_frame:end_frame]
        if getattr(chroma_window, "size", 0) == 0:
            continue

        start_time = float(librosa.frames_to_time(start_frame, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH))
        end_time = float(librosa.frames_to_time(end_frame, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH))
        if end_time <= start_time + 0.04:
            continue

        normalized_window = _normalize_chroma_columns(chroma_window.astype(float), np)
        chroma_vec = np.median(normalized_window, axis=1)
        stable_vec = np.quantile(normalized_window, _STABLE_NOTE_QUANTILE, axis=1)

        bass_window = bass_chroma[:, start_frame:end_frame] if getattr(bass_chroma, "size", 0) else None
        bass_vec = (
            np.median(_normalize_chroma_columns(bass_window.astype(float), np), axis=1)
            if getattr(bass_window, "size", 0)
            else np.zeros(12, dtype=float)
        )

        rows.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "chroma_vec": chroma_vec,
                "stable_vec": stable_vec,
                "bass_vec": bass_vec,
                "analysis_vec": chroma_vec,
            }
        )

    if not rows:
        return rows

    median_duration = float(np.median([float(row["duration"]) for row in rows]))
    for index, row in enumerate(rows):
        duration = float(row["duration"])
        if duration > max(0.45, median_duration * _SHORT_SEGMENT_CONTEXT_RATIO):
            continue

        blended = np.array(row["chroma_vec"], dtype=float)
        total_weight = 1.0
        if index > 0:
            blended += np.array(rows[index - 1]["stable_vec"], dtype=float) * _CONTEXT_BLEND_WEIGHT
            total_weight += _CONTEXT_BLEND_WEIGHT
        if index + 1 < len(rows):
            blended += np.array(rows[index + 1]["stable_vec"], dtype=float) * _CONTEXT_BLEND_WEIGHT
            total_weight += _CONTEXT_BLEND_WEIGHT
        row["analysis_vec"] = blended / max(total_weight, 1e-6)

    return rows


def _quality_family(quality_name: str) -> str:
    quality = (quality_name or "").strip().lower()
    if quality in {"major", "major7"}:
        return "major"
    if quality in {"minor", "minor7", "minor9"}:
        return "minor"
    if quality == "dominant7":
        return "dominant"
    if quality in {"diminished", "diminished7"}:
        return "diminished"
    if quality == "half_diminished":
        return "half_diminished"
    if quality == "augmented":
        return "augmented"
    return quality or "major"


def _candidate_key_bias(root_index: int, quality_name: str, tonic_pitch: int | None, mode: str) -> float:
    if tonic_pitch is None or not mode:
        return 0.0

    if mode == "major":
        diatonic = {
            0: {"major"},
            2: {"minor"},
            4: {"minor"},
            5: {"major"},
            7: {"major", "dominant"},
            9: {"minor"},
            11: {"diminished", "half_diminished"},
        }
    else:
        diatonic = {
            0: {"minor"},
            2: {"diminished", "half_diminished"},
            3: {"major"},
            5: {"minor"},
            7: {"minor", "major", "dominant"},
            8: {"major"},
            10: {"major"},
        }

    family = _quality_family(quality_name)
    interval = (root_index - tonic_pitch) % 12
    if interval in diatonic:
        return 0.022 if family in diatonic[interval] else 0.005
    return -0.020


def _collect_chord_candidates(
    chroma_vector: Any,
    bass_vector: Any,
    np: Any,
    *,
    tonic_pitch: int | None = None,
    mode: str = "",
    stable_vector: Any | None = None,
) -> list[dict[str, object]]:
    energy = float(np.sum(chroma_vector))
    if energy <= 1e-6:
        return [
            {
                "label": NO_CHORD_LABEL,
                "root": "",
                "root_index": -1,
                "quality": "",
                "family": "",
                "score": 0.55,
                "confidence": 0.0,
            }
        ]

    normalized = chroma_vector.astype(float) / max(energy, 1e-6)
    stable_energy = float(np.sum(stable_vector)) if stable_vector is not None else 0.0
    stable = (
        stable_vector.astype(float) / max(stable_energy, 1e-6)
        if stable_vector is not None and stable_energy > 1e-6
        else normalized
    )
    bass_energy = float(np.sum(bass_vector))
    bass = bass_vector.astype(float) / bass_energy if bass_energy > 1e-6 else np.zeros_like(normalized)
    candidates: list[dict[str, object]] = []

    for suffix, quality_name, base_template, template_bias in CHORD_TEMPLATES:
        template = np.array(base_template, dtype=float)
        template = template / max(float(np.sum(template)), 1e-6)
        chord_intervals = [index for index, value in enumerate(base_template) if value >= 0.58]
        core_intervals = _core_intervals_for_quality(quality_name)
        extension_intervals = _extension_intervals_for_quality(quality_name)

        for root_index in range(12):
            rolled = np.roll(template, root_index)
            chord_tones = [(root_index + interval) % 12 for interval in chord_intervals]
            core_tones = [(root_index + interval) % 12 for interval in core_intervals]
            extension_tones = [(root_index + interval) % 12 for interval in extension_intervals]
            support = float(np.dot(normalized, rolled))
            outside = float(np.sum(normalized[rolled < 0.01]))
            stable_support = (
                sum(float(stable[index]) for index in chord_tones) / len(chord_tones)
                if chord_tones
                else 0.0
            )
            core_support = (
                sum(float(stable[index]) for index in core_tones) / len(core_tones)
                if core_tones
                else 0.0
            )
            stable_outside = float(np.sum(stable[rolled < 0.01]))
            root_support = max(float(normalized[root_index]), float(stable[root_index]))
            bass_support = float(bass[root_index])
            missing_core = 0.0
            for tone_index in core_tones[1:]:
                missing_core += max(0.0, 0.10 - float(stable[tone_index]))
            extension_support = (
                sum(float(stable[index]) for index in extension_tones) / len(extension_tones)
                if extension_tones
                else 0.0
            )
            extension_gap = max(0.0, _EXTENSION_MIN_STABLE_SUPPORT - extension_support) if extension_tones else 0.0

            score = support * template_bias
            score += stable_support * 0.26
            score += core_support * 0.34
            score += root_support * 0.18
            score += bass_support * 0.26
            score -= outside * 0.15
            score -= stable_outside * 0.22
            score -= missing_core * 0.42
            score += _candidate_key_bias(root_index, quality_name, tonic_pitch, mode)
            if extension_tones:
                score += extension_support * 0.06
                score -= extension_gap * _EXTENSION_WEAK_SUPPORT_PENALTY
            else:
                score += _TRIAD_SUPPORT_BONUS
            if quality_name in {"sus2", "sus4"}:
                third_support = max(
                    float(stable[(root_index + 3) % 12]),
                    float(stable[(root_index + 4) % 12]),
                )
                score -= third_support * _SUSPENDED_THIRD_PENALTY

            candidates.append(
                {
                    "label": f"{NOTE_NAMES[root_index]}{suffix}",
                    "root": NOTE_NAMES[root_index],
                    "root_index": root_index,
                    "quality": quality_name,
                    "family": _quality_family(quality_name),
                    "score": score,
                    "core_support": core_support,
                    "extension_support": extension_support,
                }
            )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    best_score = float(candidates[0]["score"])
    second_score = float(candidates[1]["score"]) if len(candidates) > 1 else best_score - 0.02
    best_contrast = best_score - second_score

    retained = [
        candidate
        for candidate in candidates[:_MAX_CANDIDATES_PER_SEGMENT]
        if float(candidate["score"]) >= best_score - 0.10
    ]

    if best_score < _MIN_MATCH_SCORE or best_contrast < _MIN_CONTRAST:
        if best_score < max(0.18, _MIN_MATCH_SCORE * 0.65):
            no_chord_score = best_score + 0.02
        else:
            no_chord_score = max(0.08, best_score - 0.06)
        retained.append(
            {
                "label": NO_CHORD_LABEL,
                "root": "",
                "root_index": -1,
                "quality": "",
                "family": "",
                "score": no_chord_score,
                "confidence": max(0.0, min(1.0, best_score * 0.72)),
            }
        )

    for index, candidate in enumerate(retained):
        next_score = float(retained[index + 1]["score"]) if index + 1 < len(retained) else float(candidate["score"]) - 0.02
        contrast = float(candidate["score"]) - next_score
        # Calibrated so a cleanly detected triad lands >= 0.7 and the
        # 0.35/0.52 delivery thresholds regain meaning (a perfect synthetic
        # input used to score ~0.55, so every real detection read as "low").
        candidate["confidence"] = max(
            0.0,
            min(
                1.0,
                float(candidate["score"]) * _CONFIDENCE_SCORE_WEIGHT
                + max(contrast, 0.0) * _CONFIDENCE_CONTRAST_WEIGHT
                + float(candidate.get("core_support", 0.0)) * _CONFIDENCE_CORE_WEIGHT
                - max(0.0, _EXTENSION_MIN_STABLE_SUPPORT - float(candidate.get("extension_support", 0.0))) * 0.16,
            ),
        )

    return retained


def _template_intervals_for_quality(quality_name: str) -> list[int]:
    for _suffix, name, base_template, _bias in CHORD_TEMPLATES:
        if name == quality_name:
            return [index for index, value in enumerate(base_template) if value >= 0.58]
    return []


def _apply_slash_bass_label(label: str, candidate: dict[str, object], row: dict[str, object], np: Any) -> str:
    """Emit X/Y inversions when a chord tone other than the root dominates the bass.

    The dedicated 35-260 Hz bass chroma is already computed per segment but was
    only used as a root-support weight; here it finally names the inversion.
    """
    root_index = int(candidate.get("root_index", -1))
    if root_index < 0 or "/" in label:
        return label
    bass_vec = np.asarray(row.get("bass_vec"), dtype=float)
    if bass_vec.size != 12 or float(np.sum(bass_vec)) <= 1e-6:
        return label
    bass_pitch = int(np.argmax(bass_vec))
    if bass_pitch == root_index:
        return label
    intervals = _template_intervals_for_quality(str(candidate.get("quality", "")))
    chord_tones = {(root_index + interval) % 12 for interval in intervals}
    if bass_pitch not in chord_tones:
        return label
    bass_strength = float(bass_vec[bass_pitch])
    root_strength = float(bass_vec[root_index])
    if bass_strength < 0.25 or bass_strength < root_strength * 1.25:
        return label
    return f"{label}/{NOTE_NAMES[bass_pitch]}"


def _transition_score(left: dict[str, object], right: dict[str, object]) -> float:
    if left["label"] == right["label"]:
        return 0.065
    if left["label"] == NO_CHORD_LABEL or right["label"] == NO_CHORD_LABEL:
        return -0.012

    left_root = int(left["root_index"])
    right_root = int(right["root_index"])
    interval = (right_root - left_root) % 12
    score = 0.0
    if left_root == right_root:
        score += 0.020
    if interval in {5, 7}:
        score += 0.010
    if interval in {3, 9}:
        score += 0.006
    if interval in {1, 11}:
        score -= 0.006
    if left["family"] == right["family"]:
        score += 0.004
    if left["family"] == "dominant" and interval == 5:
        score += 0.026
    if left["family"] in {"diminished", "half_diminished"} and interval in {1, 11}:
        score += 0.014
    return score


def _decode_chord_path(candidate_segments: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """Viterbi decode over per-segment chord candidates.

    _TRANSITION_WEIGHT lifts the musical transition prior into the same scale
    as the emission scores (~0.3-0.6); at the raw 0.004-0.065 values the
    decoder was effectively emission-only and chord continuity was cosmetic.
    """
    if not candidate_segments:
        return []

    scores: list[list[float]] = [[float(candidate["score"]) for candidate in candidate_segments[0]]]
    back_pointers: list[list[int]] = [[-1 for _candidate in candidate_segments[0]]]
    for segment_index in range(1, len(candidate_segments)):
        current_scores: list[float] = []
        current_back: list[int] = []
        previous_candidates = candidate_segments[segment_index - 1]
        current_candidates = candidate_segments[segment_index]
        for current in current_candidates:
            best_total = float("-inf")
            best_previous_index = 0
            for previous_index, previous in enumerate(previous_candidates):
                total = (
                    scores[segment_index - 1][previous_index]
                    + float(current["score"])
                    + _transition_score(previous, current) * _TRANSITION_WEIGHT
                )
                if total > best_total:
                    best_total = total
                    best_previous_index = previous_index
            current_scores.append(best_total)
            current_back.append(best_previous_index)
        scores.append(current_scores)
        back_pointers.append(current_back)

    best_indices = [max(range(len(scores[-1])), key=lambda index: scores[-1][index])]
    for segment_index in range(len(candidate_segments) - 1, 0, -1):
        best_indices.append(back_pointers[segment_index][best_indices[-1]])
    best_indices.reverse()
    return [
        candidate_segments[segment_index][candidate_index]
        for segment_index, candidate_index in enumerate(best_indices)
    ]


def _classify_chord(chroma_vector: Any, np: Any) -> tuple[str, float, str, str]:
    """Classify a chroma vector into a chord label with confidence scoring.

    Uses cosine similarity against templates plus a penalty for non-chord tones.
    """
    energy = float(np.sum(chroma_vector))
    if energy <= 1e-6:
        return NO_CHORD_LABEL, 0.0, "", ""

    normalized = _normalize_vector(chroma_vector.astype(float), np)
    best_score = -1.0
    second_score = -1.0
    best_root = 0
    best_suffix = ""
    best_quality = ""

    for suffix, quality_name, base_template, template_bias in CHORD_TEMPLATES:
        template = _normalize_vector(np.array(base_template, dtype=float), np)
        for root_index in range(12):
            rolled = np.roll(template, root_index)
            # Cosine similarity
            similarity = float(np.dot(normalized, rolled))
            # Penalty: energy in non-template positions (reduces false positives)
            mask = rolled < 0.01
            noise_energy = float(np.sum(normalized[mask])) if mask.any() else 0.0
            penalty = noise_energy * 0.25
            score = (similarity - penalty) * template_bias

            if score > best_score:
                second_score = best_score
                best_score = score
                best_root = root_index
                best_suffix = suffix
                best_quality = quality_name
            elif score > second_score:
                second_score = score

    contrast = best_score - max(second_score, 0.0)
    confidence = max(0.0, min(1.0, best_score * 0.75 + contrast * 1.8))

    if best_score < _MIN_MATCH_SCORE or contrast < _MIN_CONTRAST:
        return NO_CHORD_LABEL, confidence, "", ""

    root = NOTE_NAMES[best_root]
    return f"{root}{best_suffix}", confidence, root, best_quality


# ---------------------------------------------------------------------------
# Post-processing: merge, smooth, fill gaps
# ---------------------------------------------------------------------------

def _merge_adjacent_events(events: list[ChordEvent]) -> list[ChordEvent]:
    merged: list[ChordEvent] = []
    for event in events:
        if event.end <= event.start + 1e-3:
            continue
        if merged and event.label == merged[-1].label and event.start <= merged[-1].end + 0.05:
            merged[-1].end = max(merged[-1].end, event.end)
            merged[-1].confidence = max(merged[-1].confidence, event.confidence)
            continue
        merged.append(event)
    return merged


def _smooth_events(events: list[ChordEvent], beat_length: float) -> list[ChordEvent]:
    """Remove spurious short chords and fill small gaps."""
    if len(events) < 3:
        return events

    # Adaptive minimum chord duration: at least half a beat
    min_chord_duration = max(0.24, beat_length * 0.65) if beat_length > 0 else _SMOOTHING_MAX_DURATION

    smoothed: list[ChordEvent] = []
    index = 0
    while index < len(events):
        current = events[index]
        previous = smoothed[-1] if smoothed else None
        upcoming = events[index + 1] if index + 1 < len(events) else None
        duration = current.end - current.start

        # Remove short weak chords between two identical chords
        if (
            previous
            and upcoming
            and duration < min_chord_duration
            and previous.label == upcoming.label
            and current.confidence < _SMOOTHING_CONFIDENCE
        ):
            previous.end = upcoming.end
            previous.confidence = max(previous.confidence, upcoming.confidence)
            index += 2
            continue

        # Fill small "N" gaps by extending previous chord
        if current.label == NO_CHORD_LABEL and duration < _GAP_FILL_MAX_DURATION and previous and upcoming:
            previous.end = upcoming.start
            index += 1
            continue

        # Remove very short chords entirely (below half-beat threshold)
        if (
            duration < min_chord_duration * 0.8
            and current.confidence < _SMOOTHING_CONFIDENCE
            and current.label != NO_CHORD_LABEL
            and previous
        ):
            previous.end = current.end
            index += 1
            continue

        smoothed.append(current)
        index += 1

    return _merge_adjacent_events(smoothed)


def _same_root(left: ChordEvent, right: ChordEvent) -> bool:
    left_root = _event_root_name(left)
    right_root = _event_root_name(right)
    return bool(left_root) and left_root == right_root


def _simplified_same_root_label(event: ChordEvent) -> tuple[str, str, str]:
    root = _event_root_name(event)
    if not root:
        return event.label, event.root, event.quality

    family = _quality_bucket(event)
    if family == "minor":
        return f"{root}m", root, "minor"
    if family == "dominant":
        return f"{root}7", root, "dominant7"
    if family in {"diminished", "half_diminished"}:
        return f"{root}dim", root, "diminished"
    if family in {"major", "augmented"}:
        return root, root, "major"
    return event.label, root, event.quality


def _is_colored_same_root_variant(event: ChordEvent) -> bool:
    quality = (event.quality or "").strip().lower()
    return quality in {
        "major7",
        "minor7",
        "minor9",
        "sus2",
        "sus4",
        "add9",
        "augmented",
        "diminished7",
        "half_diminished",
    }


def _simplify_same_root_variants(events: list[ChordEvent], beat_length: float) -> list[ChordEvent]:
    if len(events) < 2:
        return events

    simplified = [_clone_chord_event(event) for event in events]
    short_variant_limit = max(0.90, beat_length * 1.05) if beat_length > 0 else 0.90

    for index in range(1, len(simplified) - 1):
        previous = simplified[index - 1]
        current = simplified[index]
        upcoming = simplified[index + 1]
        duration = current.end - current.start

        if (
            _same_root(previous, current)
            and _same_root(current, upcoming)
            and previous.label == upcoming.label
            and current.label != previous.label
            and duration <= short_variant_limit
        ):
            simplified[index] = _clone_chord_event(
                current,
                label=previous.label,
                root=previous.root,
                quality=previous.quality,
            )

    for index, current in enumerate(simplified):
        if not _is_colored_same_root_variant(current):
            continue

        duration = current.end - current.start
        if duration > short_variant_limit:
            continue

        candidate_neighbors: list[ChordEvent] = []
        if index > 0 and _same_root(current, simplified[index - 1]):
            candidate_neighbors.append(simplified[index - 1])
        if index + 1 < len(simplified) and _same_root(current, simplified[index + 1]):
            candidate_neighbors.append(simplified[index + 1])
        if not candidate_neighbors:
            continue

        stronger_neighbor = max(candidate_neighbors, key=lambda item: item.confidence)
        if stronger_neighbor.confidence < current.confidence + 0.08:
            continue
        simplified[index] = _clone_chord_event(
            current,
            label=stronger_neighbor.label,
            root=stronger_neighbor.root,
            quality=stronger_neighbor.quality,
        )

    normalized: list[ChordEvent] = []
    start_index = 0
    while start_index < len(simplified):
        end_index = start_index + 1
        while end_index < len(simplified) and _same_root(simplified[end_index - 1], simplified[end_index]):
            end_index += 1

        run = simplified[start_index:end_index]
        if len(run) > 1:
            simplified_forms = [_simplified_same_root_label(event) for event in run]
            simplified_labels = {label for label, _root, _quality in simplified_forms if label}
            if len(simplified_labels) == 1:
                canonical_label, canonical_root, canonical_quality = simplified_forms[0]
                run = [
                    _clone_chord_event(
                        event,
                        label=canonical_label,
                        root=canonical_root,
                        quality=canonical_quality,
                    )
                    for event in run
                ]

        normalized.extend(run)
        start_index = end_index

    return _merge_adjacent_events(normalized)


def _snap_to_beats(events: list[ChordEvent], beat_times: list[float]) -> list[ChordEvent]:
    """Snap chord boundaries to nearest beat when close enough.

    This ensures chords align to the rhythmic grid, producing cleaner
    transitions for the visual countdown.
    """
    if not beat_times or not events:
        return events

    snap_threshold = 0.08  # seconds — snap if within 80ms of a beat

    for event in events:
        for beat_time in beat_times:
            if abs(event.start - beat_time) < snap_threshold:
                event.start = beat_time
                break
        for beat_time in beat_times:
            if abs(event.end - beat_time) < snap_threshold:
                event.end = beat_time
                break

    return events


def _resolve_preview_window(bpm: float) -> float:
    if bpm <= 0:
        return 0.65
    beat_length = 60.0 / bpm
    return max(0.35, min(1.1, beat_length))


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class LibrosaHarmonyAnalyzer:
    name = "librosa_harmony_v5"

    def analyze(self, audio_path: str) -> SongAnalysis:
        np, librosa = _lazy_import_analysis_libs()

        try:
            audio, sample_rate = librosa.load(audio_path, sr=ANALYSIS_SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise MusicAnalysisError(f"Failed to load analysis audio: {exc}") from exc

        if getattr(audio, "size", 0) == 0:
            raise MusicAnalysisError("Loaded analysis audio is empty.")

        # Harmonic/percussive separation — use only harmonic for chord detection
        if _HPSS_MARGIN > 0:
            harmonic_audio = librosa.effects.harmonic(audio, margin=_HPSS_MARGIN)
        else:
            harmonic_audio = audio

        # Beat tracking on the analysis audio (typically the demucs
        # instrumental stem, not the original full mix)
        try:
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH,
            )
        except Exception:
            tempo, beat_frames = 0.0, []

        bpm = _to_float(tempo) if tempo is not None else 0.0
        beat_frame_list = [int(f) for f in getattr(beat_frames, "tolist", lambda: beat_frames or [])()]
        beat_length = 60.0 / max(bpm, 80.0)

        raw_chroma, bass_chroma = _extract_analysis_chroma(harmonic_audio, sample_rate, np, librosa)
        global_key, global_tonic_pitch, global_mode = infer_key_from_chroma_profile(raw_chroma, np)

        # Build frame boundaries for chord segmentation
        frame_count = raw_chroma.shape[1]
        if beat_frame_list:
            change_frames = _detect_chord_change_frames(raw_chroma, beat_frame_list, np)
            boundaries = [0, *beat_frame_list, *change_frames, frame_count]
        else:
            fallback_frames = int(round((beat_length * sample_rate) / ANALYSIS_HOP_LENGTH))
            step = max(6, fallback_frames)
            boundaries = list(range(0, frame_count, step))
            boundaries.append(frame_count)
        boundaries = _normalize_boundaries(boundaries, frame_count)

        segment_rows = _build_segment_feature_rows(raw_chroma, bass_chroma, boundaries, sample_rate, np, librosa)
        segment_candidates = [
            _collect_chord_candidates(
                row["analysis_vec"],
                row["bass_vec"],
                np,
                tonic_pitch=global_tonic_pitch,
                mode=global_mode,
                stable_vector=row["stable_vec"],
            )
            for row in segment_rows
        ]

        rough_path = _decode_chord_path(segment_candidates)
        rough_events = [
            ChordEvent(
                label=str(candidate["label"]),
                start=float(row["start_time"]),
                end=float(row["end_time"]),
                confidence=float(candidate["confidence"]),
                root=str(candidate["root"]),
                quality=str(candidate["quality"]),
            )
            for candidate, row in zip(rough_path, segment_rows)
            if str(candidate["label"]) != NO_CHORD_LABEL
        ]
        inferred_key, tonic_pitch, mode = infer_song_key(rough_events)
        resolved_key = inferred_key or global_key
        resolved_tonic_pitch = tonic_pitch if tonic_pitch is not None else global_tonic_pitch
        resolved_mode = mode or global_mode

        keyed_candidates = [
            _collect_chord_candidates(
                row["analysis_vec"],
                row["bass_vec"],
                np,
                tonic_pitch=resolved_tonic_pitch,
                mode=resolved_mode,
                stable_vector=row["stable_vec"],
            )
            for row in segment_rows
        ]
        decoded_path = _decode_chord_path(keyed_candidates)

        events: list[ChordEvent] = []
        for candidate, row in zip(decoded_path, segment_rows):
            label = str(candidate["label"])
            if label != NO_CHORD_LABEL:
                label = _apply_slash_bass_label(label, candidate, row, np)
            events.append(
                ChordEvent(
                    label=label,
                    start=float(row["start_time"]),
                    end=float(row["end_time"]),
                    confidence=float(candidate["confidence"]),
                    root=str(candidate["root"]),
                    quality=str(candidate["quality"]),
                )
            )

        # Post-processing pipeline
        merged = _merge_adjacent_events(events)
        smoothed = _smooth_events(merged, beat_length)
        simplified = _simplify_same_root_variants(smoothed, beat_length)
        beat_times = [
            float(librosa.frames_to_time(f, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH))
            for f in _normalize_boundaries(beat_frame_list, frame_count)[:-1]
        ]
        chord_events = _snap_to_beats(simplified, beat_times)
        resolved_original_key, _resolved_tonic, _resolved_mode = infer_song_key(
            [event for event in chord_events if event.label != NO_CHORD_LABEL]
        )

        # Time signature detection
        time_signature = _detect_time_signature(beat_times, np)
        measure_times = _compute_measure_times(beat_times, time_signature)

        logger.info(
            "Analysis complete: BPM=%.1f, time_sig=%d/4, chords=%d, beats=%d, measures=%d",
            bpm, time_signature, len(chord_events), len(beat_times), len(measure_times),
        )

        return SongAnalysis(
            bpm=round(bpm, 2) if bpm else 0.0,
            time_signature=time_signature,
            preview_window_seconds=_resolve_preview_window(bpm),
            provider=self.name,
            source_audio=audio_path,
            beat_times=beat_times,
            measure_times=measure_times,
            original_key=resolved_original_key or resolved_key,
            original_chord_events=list(chord_events),
            chord_events=chord_events,
        )


# ---------------------------------------------------------------------------
# Chord-to-word mapping (unchanged API, improved proximity scoring)
# ---------------------------------------------------------------------------

def _flatten_words(segments: list[TranscriptSegment]) -> list[tuple[int, int, Any]]:
    flattened: list[tuple[int, int, Any]] = []
    for segment_index, segment in enumerate(segments):
        for word_index, word in enumerate(segment.words):
            flattened.append((segment_index, word_index, word))
    return flattened


def _same_chord_event(left: ChordEvent, right: ChordEvent) -> bool:
    return (
        left.label == right.label
        and abs(left.start - right.start) <= 1e-3
        and abs(left.end - right.end) <= 1e-3
    )


def _append_assignment(
    assignments: dict[tuple[int, int], list[ChordEvent]],
    key: tuple[int, int],
    event: ChordEvent,
):
    bucket = assignments.setdefault(key, [])
    if any(_same_chord_event(existing, event) for existing in bucket):
        return
    bucket.append(event)


def build_word_chord_map(
    segments: list[TranscriptSegment], chord_events: list[ChordEvent]
) -> dict[tuple[int, int], list[ChordEvent]]:
    flattened = _flatten_words(segments)
    if not flattened:
        return {}

    visible_events = [event for event in chord_events if event.label and event.label != NO_CHORD_LABEL]
    assignments: dict[tuple[int, int], list[ChordEvent]] = {}
    for event in visible_events:
        best_index = 0
        best_score = float("inf")
        for flattened_index, (_segment_index, _word_index, word) in enumerate(flattened):
            if word.start <= event.start <= word.end + 0.08:
                best_index = flattened_index
                break
            score = min(abs(event.start - word.start), abs(event.start - word.end))
            if score < best_score:
                best_score = score
                best_index = flattened_index

        segment_index, word_index, _word = flattened[best_index]
        _append_assignment(assignments, (segment_index, word_index), event)

    for segment_index, segment in enumerate(segments):
        if not segment.words:
            continue
        carry_candidates = [
            event
            for event in visible_events
            if event.start <= segment.start + 0.08 and event.end > segment.start + 0.02
        ]
        if carry_candidates:
            _append_assignment(
                assignments,
                (segment_index, 0),
                max(carry_candidates, key=lambda item: (item.start, item.end)),
            )

    for flattened_index in range(1, len(flattened)):
        segment_index, word_index, word = flattened[flattened_index]
        previous_segment_index, _previous_word_index, previous_word = flattened[flattened_index - 1]
        boundary_crossed = segment_index != previous_segment_index
        noticeable_pause = word.start - previous_word.end >= 0.32
        if not boundary_crossed and not noticeable_pause:
            continue

        carry_candidates = [
            event
            for event in visible_events
            if event.start <= word.start + 0.08 and event.end > word.start + 0.02
        ]
        if not carry_candidates:
            continue
        _append_assignment(
            assignments,
            (segment_index, word_index),
            max(carry_candidates, key=lambda item: (item.start, item.end)),
        )

    return assignments


def build_word_id_chord_map(
    segments: list[TranscriptSegment], chord_events: list[ChordEvent]
) -> dict[int, list[ChordEvent]]:
    indexed_words = {
        (si, wi): word
        for si, seg in enumerate(segments)
        for wi, word in enumerate(seg.words)
    }
    return {
        id(indexed_words[key]): value
        for key, value in build_word_chord_map(segments, chord_events).items()
        if key in indexed_words
    }


def render_chord_sheet_text(title: str, segments: list[TranscriptSegment], analysis: SongAnalysis) -> str:
    word_map = build_word_chord_map(segments, analysis.chord_events)
    header_parts = [f"כותרת: {title or 'ללא שם'}"]
    if analysis.bpm:
        header_parts.append(f"קצב: {analysis.bpm:.0f}")
    else:
        header_parts.append("קצב: לא ידוע")
    header_parts.append(f"משקל: {analysis.time_signature}/4")
    if analysis.original_key:
        header_parts.append(f"סולם מקור: {analysis.original_key}")
    if analysis.target_key:
        header_parts.append(f"סולם קל: {analysis.target_key}")
    lines = header_parts + [""]

    for segment_index, segment in enumerate(segments):
        if not segment.words:
            continue

        chord_cells: list[str] = []
        lyric_cells: list[str] = []
        has_visible_chords = False

        for word_index, word in enumerate(segment.words):
            chord_label = " / ".join(event.label for event in word_map.get((segment_index, word_index), []))
            has_visible_chords = has_visible_chords or bool(chord_label)
            width = max(len(word.word), len(chord_label), 1)
            chord_cells.append(chord_label.rjust(width))
            lyric_cells.append(word.word.rjust(width))

        if has_visible_chords:
            lines.append(" ".join(chord_cells))
        lines.append(" ".join(lyric_cells))
        lines.append("")

    return "\n".join(lines).strip() + "\n"
