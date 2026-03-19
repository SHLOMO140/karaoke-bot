"""Music analysis helpers for BPM, chords, time signature, chord sheets, and overlays."""

from __future__ import annotations

import logging
from typing import Any

from .exceptions import MusicAnalysisError
from .models import ChordEvent, SongAnalysis, TranscriptSegment

logger = logging.getLogger(__name__)

ANALYSIS_SAMPLE_RATE = 22_050
ANALYSIS_HOP_LENGTH = 512
NO_CHORD_LABEL = "N"
NOTE_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

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
    ("add9", "add9",       [1.0,  0.0,  0.62, 0.0,  0.80, 0.0,  0.0,  0.90, 0.0,  0.0,  0.0,  0.0 ], 0.88),
    ("m9",   "minor9",     [1.0,  0.0,  0.62, 0.76, 0.0,  0.0,  0.0,  0.88, 0.0,  0.0,  0.68, 0.0 ], 0.86),
)

# Detection thresholds — raised for fewer false positives
_MIN_MATCH_SCORE = 0.68
_MIN_CONTRAST = 0.045
_SMOOTHING_CONFIDENCE = 0.78
_SMOOTHING_MAX_DURATION = 0.32
_GAP_FILL_MAX_DURATION = 0.20


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

def _extract_beat_sync_chroma(
    harmonic_audio: Any, sample_rate: int, beat_frames: list[int], np: Any, librosa: Any
) -> Any:
    """Extract chroma features synchronized to beat boundaries.

    Beat-sync chroma averages the chroma energy within each beat,
    producing one chroma vector per beat — much cleaner than raw frame chroma.
    """
    chroma = librosa.feature.chroma_cqt(
        y=harmonic_audio,
        sr=sample_rate,
        hop_length=ANALYSIS_HOP_LENGTH,
        bins_per_octave=36,
        n_chroma=12,
    )
    if getattr(chroma, "size", 0) == 0:
        chroma = librosa.feature.chroma_stft(
            y=harmonic_audio, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH
        )
    if getattr(chroma, "size", 0) == 0:
        raise MusicAnalysisError("Chord analysis did not produce chroma features.")

    # Apply median filtering (kernel=5 frames) to smooth noise before sync
    try:
        from scipy.ndimage import median_filter
        chroma = median_filter(chroma, size=(1, 5))
    except ImportError:
        # Fallback: simple moving average
        kernel = 5
        if chroma.shape[1] > kernel:
            padded = np.pad(chroma, ((0, 0), (kernel // 2, kernel // 2)), mode="edge")
            smoothed = np.zeros_like(chroma)
            for i in range(chroma.shape[1]):
                smoothed[:, i] = padded[:, i:i + kernel].mean(axis=1)
            chroma = smoothed

    if beat_frames:
        try:
            beat_sync_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        except Exception:
            beat_sync_chroma = chroma
    else:
        beat_sync_chroma = chroma

    return chroma, beat_sync_chroma


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
    min_chord_duration = max(0.18, beat_length * 0.45) if beat_length > 0 else _SMOOTHING_MAX_DURATION

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
            duration < min_chord_duration * 0.6
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
    name = "librosa_harmony_v2"

    def analyze(self, audio_path: str) -> SongAnalysis:
        np, librosa = _lazy_import_analysis_libs()

        try:
            audio, sample_rate = librosa.load(audio_path, sr=ANALYSIS_SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise MusicAnalysisError(f"Failed to load analysis audio: {exc}") from exc

        if getattr(audio, "size", 0) == 0:
            raise MusicAnalysisError("Loaded analysis audio is empty.")

        # Harmonic/percussive separation — use only harmonic for chord detection
        harmonic_audio = librosa.effects.harmonic(audio, margin=3.0)

        # Beat tracking on the full mix for best tempo accuracy
        try:
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH,
            )
        except Exception:
            tempo, beat_frames = 0.0, []

        bpm = _to_float(tempo) if tempo is not None else 0.0
        beat_frame_list = [int(f) for f in getattr(beat_frames, "tolist", lambda: beat_frames or [])()]
        beat_length = 60.0 / max(bpm, 80.0)

        # Beat-synchronous chroma extraction
        raw_chroma, beat_sync_chroma = _extract_beat_sync_chroma(
            harmonic_audio, sample_rate, beat_frame_list, np, librosa,
        )

        # Build frame boundaries for chord segmentation
        frame_count = raw_chroma.shape[1]
        if beat_frame_list:
            boundaries = [0, *beat_frame_list, frame_count]
        else:
            fallback_frames = int(round((beat_length * sample_rate) / ANALYSIS_HOP_LENGTH))
            step = max(6, fallback_frames)
            boundaries = list(range(0, frame_count, step))
            boundaries.append(frame_count)
        boundaries = _normalize_boundaries(boundaries, frame_count)

        # Classify chords per beat segment
        events: list[ChordEvent] = []
        for seg_idx, (start_frame, end_frame) in enumerate(zip(boundaries, boundaries[1:])):
            if end_frame <= start_frame:
                continue

            # Use beat-sync chroma if available (one vector per beat)
            if beat_frame_list and seg_idx < beat_sync_chroma.shape[1]:
                chroma_vec = beat_sync_chroma[:, seg_idx]
            else:
                window = raw_chroma[:, start_frame:end_frame]
                if getattr(window, "size", 0) == 0:
                    continue
                chroma_vec = np.median(window, axis=1)

            chord_label, confidence, root, quality = _classify_chord(chroma_vec, np)
            start_time = float(librosa.frames_to_time(start_frame, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH))
            end_time = float(librosa.frames_to_time(end_frame, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH))
            if end_time <= start_time + 0.04:
                continue

            events.append(ChordEvent(
                label=chord_label, start=start_time, end=end_time,
                confidence=confidence, root=root, quality=quality,
            ))

        # Post-processing pipeline
        merged = _merge_adjacent_events(events)
        smoothed = _smooth_events(merged, beat_length)
        beat_times = [
            float(librosa.frames_to_time(f, sr=sample_rate, hop_length=ANALYSIS_HOP_LENGTH))
            for f in _normalize_boundaries(beat_frame_list, frame_count)[:-1]
        ]
        chord_events = _snap_to_beats(smoothed, beat_times)

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


def build_word_chord_map(
    segments: list[TranscriptSegment], chord_events: list[ChordEvent]
) -> dict[tuple[int, int], list[ChordEvent]]:
    flattened = _flatten_words(segments)
    if not flattened:
        return {}

    assignments: dict[tuple[int, int], list[ChordEvent]] = {}
    for event in chord_events:
        if not event.label or event.label == NO_CHORD_LABEL:
            continue

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
        assignments.setdefault((segment_index, word_index), []).append(event)

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
    header_parts = [f"Title: {title or 'Untitled'}"]
    if analysis.bpm:
        header_parts.append(f"BPM: {analysis.bpm:.0f}")
    else:
        header_parts.append("BPM: unknown")
    header_parts.append(f"Time Signature: {analysis.time_signature}/4")
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
