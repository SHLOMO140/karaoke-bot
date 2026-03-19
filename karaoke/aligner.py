"""Alignment providers for approved Hebrew karaoke text."""

from __future__ import annotations

import importlib.util
import logging
import math
import re
import wave
from difflib import SequenceMatcher

import numpy as np

try:
    import regex as regex_lib
except Exception:
    regex_lib = None

from .config import (
    ALIGNMENT_BOUNDARY_SEARCH_MS,
    ALIGNMENT_MIN_WORD_DURATION_MS,
    ALIGNMENT_MODEL_NAME,
    ALIGNMENT_PROVIDER,
)
from .exceptions import AlignmentError
from .models import AlignedTranscript, SubWordTiming, TranscriptSegment, WordTiming

logger = logging.getLogger(__name__)

MIN_WORD_DURATION_SEC = ALIGNMENT_MIN_WORD_DURATION_MS / 1000.0
BOUNDARY_SEARCH_SEC = ALIGNMENT_BOUNDARY_SEARCH_MS / 1000.0
MIN_SUBWORD_DURATION_SEC = 0.012
HEBREW_NIQQUD_RANGE = range(0x05B0, 0x05C8)


def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w\u0590-\u05FF]+", "", word.lower(), flags=re.UNICODE)


def _split_graphemes(word: str) -> list[str]:
    if regex_lib is not None:
        clusters = [cluster for cluster in regex_lib.findall(r"\X", word) if cluster and not cluster.isspace()]
        if clusters:
            return clusters
    return [char for char in word if not char.isspace()]


def _grapheme_weight(grapheme: str) -> float:
    normalized = _normalize_word(grapheme)
    if not normalized:
        return 0.25
    weight = 1.0
    if any(ord(char) in HEBREW_NIQQUD_RANGE for char in grapheme):
        weight += 0.18
    if normalized in {"ו", "י", "א", "ה", "ע"}:
        weight += 0.28
    elif normalized in {"ל", "מ", "נ", "ר"}:
        weight += 0.08
    if normalized in {"ך", "ם", "ן", "ף", "ץ"}:
        weight -= 0.1
    if len(grapheme) > 1:
        weight += 0.05 * (len(grapheme) - 1)
    return max(0.35, weight)


def _normalize_curve(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def _activity_curve(values: np.ndarray, hop_seconds: float) -> np.ndarray:
    if values.size == 0:
        return values
    smoothing_window = max(3, int(round(0.012 / max(hop_seconds, 1e-6))))
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
    smoothed = np.convolve(values.astype(np.float32), kernel, mode="same")
    novelty = np.abs(np.gradient(smoothed)).astype(np.float32)
    floor = float(np.percentile(smoothed, 12))
    voiced = np.maximum(smoothed - floor, 0.0)
    # Voiced energy dominates: when a singer sustains a vowel, the high
    # energy should allocate proportionally more time to that grapheme.
    # Novelty (gradient) catches transitions but should not override duration.
    activity = voiced * 0.82 + novelty * 0.18
    if float(activity.sum()) <= 1e-6:
        activity = smoothed + 1e-6
    return activity.astype(np.float32)


def _flatten_words(segments: list[TranscriptSegment]) -> tuple[list[WordTiming], list[int]]:
    words: list[WordTiming] = []
    segment_indices: list[int] = []
    for segment_index, segment in enumerate(segments):
        for word in segment.words:
            words.append(word)
            segment_indices.append(segment_index)
    return words, segment_indices


def _rebuild_segments(template_segments: list[TranscriptSegment], final_words: list[WordTiming]) -> list[TranscriptSegment]:
    rebuilt = []
    cursor = 0
    for segment in template_segments:
        count = len(segment.words)
        segment_words = final_words[cursor:cursor + count]
        cursor += count
        if not segment_words:
            rebuilt.append(TranscriptSegment(words=[], text=segment.text, start=segment.start, end=segment.end))
            continue
        rebuilt.append(
            TranscriptSegment(
                words=segment_words,
                text=" ".join(word.word for word in segment_words).strip(),
                start=segment_words[0].start,
                end=segment_words[-1].end,
            )
        )
    return rebuilt


def _interpolate_word(
    index: int,
    approved_word: WordTiming,
    final_words: list[WordTiming | None],
    hint_words: list[WordTiming],
) -> WordTiming:
    previous = next((item for item in reversed(final_words[:index]) if item is not None), None)
    next_word = next((item for item in final_words[index + 1:] if item is not None), None)

    # Determine the time span available for this word.
    # Use the surrounding matched anchors, not the hint (which may be wrong
    # after text edits).
    if previous is not None and next_word is not None:
        span_start = previous.end
        span_end = next_word.start
        # Count how many consecutive None words share this span
        none_run_start = index
        while none_run_start > 0 and final_words[none_run_start - 1] is None:
            none_run_start -= 1
        none_run_end = index
        while none_run_end < len(final_words) - 1 and final_words[none_run_end + 1] is None:
            none_run_end += 1
        run_length = none_run_end - none_run_start + 1
        position_in_run = index - none_run_start

        if span_end > span_start + MIN_WORD_DURATION_SEC * run_length:
            # Distribute by word length (weighted)
            run_words = [hint_words[i].word for i in range(none_run_start, none_run_end + 1)]
            weights = [max(1, len(_normalize_word(w))) for w in run_words]
            total_weight = sum(weights)
            cumulative = 0.0
            for k in range(position_in_run):
                cumulative += weights[k]
            start = span_start + (cumulative / total_weight) * (span_end - span_start)
            end = span_start + ((cumulative + weights[position_in_run]) / total_weight) * (span_end - span_start)
        else:
            step = max(MIN_WORD_DURATION_SEC, (span_end - span_start) / max(1, run_length))
            start = span_start + position_in_run * step
            end = start + step
    else:
        # Fallback: use hint timing bounded by neighbors
        hint = hint_words[index]
        start = hint.start
        end = hint.end
        if previous is not None:
            start = max(start, previous.end)
        if next_word is not None:
            end = min(end, next_word.start)

    if end <= start:
        end = start + MIN_WORD_DURATION_SEC

    return WordTiming(
        word=approved_word.word,
        start=round(start, 6),
        end=round(end, 6),
        confidence=0.25,
        source="forced_aligner",
        aligned=False,
    )


class AudioFeatures:
    """Pre-computed audio features for timing refinement."""
    __slots__ = ("energy", "zcr", "spectral_flux", "onsets", "hop_seconds")

    def __init__(
        self,
        energy: np.ndarray,
        zcr: np.ndarray,
        spectral_flux: np.ndarray,
        onsets: np.ndarray,
        hop_seconds: float,
    ):
        self.energy = energy
        self.zcr = zcr
        self.spectral_flux = spectral_flux
        self.onsets = onsets  # indices of detected onset frames
        self.hop_seconds = hop_seconds


def _load_audio_envelope(audio_path: str) -> tuple[np.ndarray, float]:
    """Load audio and return (energy, hop_seconds) for backward compat."""
    features = _load_audio_features(audio_path)
    return features.energy, features.hop_seconds


def _load_audio_features(audio_path: str) -> AudioFeatures:
    """Load audio and compute energy, ZCR, real spectral flux, and global onsets."""
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channel_count = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        raw_audio = wav_file.readframes(frame_count)

    dtype_map = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise AlignmentError(
            f"Unsupported wav sample width: {sample_width}",
            "פורמט האודיו לא נתמך עבור דיוק הטיימינג.",
        )

    samples = np.frombuffer(raw_audio, dtype=dtype)
    if samples.size == 0:
        raise AlignmentError("Audio signal is empty.", "קובץ האודיו ריק ולא ניתן ליישר אותו.")

    samples = samples.astype(np.float32)
    if channel_count > 1:
        samples = samples.reshape(-1, channel_count).mean(axis=1)

    if sample_width == 1:
        samples -= 128.0
        scale = 128.0
    else:
        scale = float(np.iinfo(dtype).max)

    if not scale:
        raise AlignmentError("Invalid audio scale.", "לא ניתן לקרוא את עוצמת האודיו.")

    samples /= scale
    frame_window = max(1, int(sample_rate * 0.004))
    hop_size = max(1, int(sample_rate * 0.001))

    # --- RMS Energy ---
    power = np.square(samples)
    kernel = np.ones(frame_window, dtype=np.float32) / frame_window
    rms = np.sqrt(np.convolve(power, kernel, mode="same"))
    energy = rms[::hop_size]

    # --- Zero-Crossing Rate ---
    sign_changes = np.abs(np.diff(np.signbit(samples).astype(np.int8)))
    zcr_raw = np.convolve(sign_changes.astype(np.float32), kernel[:frame_window], mode="same")
    zcr_padded = np.concatenate([zcr_raw, [zcr_raw[-1] if zcr_raw.size else 0.0]])
    zcr = zcr_padded[::hop_size]

    # --- Real Spectral Flux via STFT ---
    # Use a proper short-time Fourier transform to detect timbral changes.
    # This is the standard MIR approach for onset detection and is far more
    # accurate than the previous high-pass energy proxy.
    fft_size = 512
    fft_hop = hop_size  # align with energy hop for easy indexing
    hann_window = np.hanning(fft_size).astype(np.float32)

    # Pad signal for STFT
    padded = np.concatenate([np.zeros(fft_size // 2, dtype=np.float32), samples,
                             np.zeros(fft_size // 2, dtype=np.float32)])
    n_frames = max(1, (padded.size - fft_size) // fft_hop + 1)

    # Vectorized STFT: build a strided view of overlapping frames, then batch-FFT.
    # This avoids the Python-level per-frame loop and is ~50x faster on long songs.
    frame_starts = np.arange(n_frames) * fft_hop
    # Use stride_tricks only if all frames fit within the padded signal
    if n_frames > 0 and frame_starts[-1] + fft_size <= padded.size:
        strides = padded.strides
        frames = np.lib.stride_tricks.as_strided(
            padded, shape=(n_frames, fft_size), strides=(strides[0] * fft_hop, strides[0]),
        ).copy()  # copy to ensure contiguous memory for FFT
    else:
        # Fallback for edge cases
        frames = np.zeros((n_frames, fft_size), dtype=np.float32)
        for fi in range(n_frames):
            s = frame_starts[fi]
            chunk = padded[s:s + fft_size]
            frames[fi, :chunk.size] = chunk

    windowed_frames = frames * hann_window[np.newaxis, :]
    magnitudes = np.abs(np.fft.rfft(windowed_frames, axis=1))  # (n_frames, n_bins)

    # Half-wave rectified spectral difference (only positive changes = onsets)
    mag_shifted = np.vstack([np.zeros((1, magnitudes.shape[1]), dtype=np.float32), magnitudes[:-1]])
    spectral_flux = np.sum(np.maximum(magnitudes - mag_shifted, 0.0), axis=1).astype(np.float32)

    # Smooth the spectral flux slightly to reduce noise
    sf_smooth_win = max(3, int(round(0.008 / max(hop_size / sample_rate, 1e-6))))
    if sf_smooth_win % 2 == 0:
        sf_smooth_win += 1
    sf_kernel = np.ones(sf_smooth_win, dtype=np.float32) / sf_smooth_win
    spectral_flux = np.convolve(spectral_flux, sf_kernel, mode="same")

    # --- Global Onset Detection via peak-picking on spectral flux ---
    # This is the key innovation: detect ALL onsets in the entire track,
    # then snap word boundaries to these onsets. This is the standard MIR
    # approach and is far more accurate than searching for energy patterns
    # in tiny windows around Whisper's guesses.
    onsets = _detect_onsets(spectral_flux, energy[:spectral_flux.size], hop_size / sample_rate)

    # Ensure all arrays are the same length
    min_len = min(energy.size, zcr.size, spectral_flux.size)
    return AudioFeatures(
        energy=energy[:min_len],
        zcr=zcr[:min_len],
        spectral_flux=spectral_flux[:min_len],
        onsets=onsets,
        hop_seconds=hop_size / sample_rate,
    )


def _detect_onsets(
    spectral_flux: np.ndarray,
    energy: np.ndarray,
    hop_seconds: float,
) -> np.ndarray:
    """Detect onsets using adaptive thresholding on spectral flux.

    Returns an array of frame indices where onsets are detected.
    Uses a median-based adaptive threshold to handle varying dynamics,
    with an energy gate to ignore onsets in silence.
    """
    if spectral_flux.size < 5:
        return np.array([], dtype=np.int64)

    # Adaptive threshold: median of local window + offset
    # This adapts to the local dynamics of the music
    median_window = max(7, int(round(0.10 / max(hop_seconds, 1e-6))))  # ~100ms window
    if median_window % 2 == 0:
        median_window += 1

    # Compute running median using a sliding window (vectorized via stride_tricks)
    half_w = median_window // 2
    padded_sf = np.pad(spectral_flux, (half_w, half_w), mode="edge")
    sf_strides = padded_sf.strides
    windowed_view = np.lib.stride_tricks.as_strided(
        padded_sf,
        shape=(spectral_flux.size, median_window),
        strides=(sf_strides[0], sf_strides[0]),
    )
    local_median = np.median(windowed_view, axis=1).astype(np.float32)

    # Threshold = local median * multiplier + small absolute offset
    # The multiplier controls sensitivity: lower = more onsets detected
    # 1.2 (was 1.3) catches softer vocal onsets that 1.3 missed, giving
    # more onset anchors for word-start snapping.
    sf_range = float(np.percentile(spectral_flux, 95) - np.percentile(spectral_flux, 5))
    absolute_offset = sf_range * 0.05
    threshold = local_median * 1.2 + absolute_offset

    # Energy gate: ignore onsets where energy is very low (silence/noise)
    min_len = min(energy.size, spectral_flux.size)
    energy_gate = np.ones(spectral_flux.size, dtype=bool)
    if min_len > 0:
        energy_percentile_10 = float(np.percentile(energy[:min_len], 10))
        energy_gate[:min_len] = energy[:min_len] > energy_percentile_10 * 2.0

    # Peak-picking (vectorized): flux must exceed threshold AND be a local maximum
    above_threshold = spectral_flux > threshold
    local_max = np.zeros(spectral_flux.size, dtype=bool)
    local_max[1:-1] = ((spectral_flux[1:-1] >= spectral_flux[:-2])
                        & (spectral_flux[1:-1] >= spectral_flux[2:]))
    is_peak = above_threshold & local_max & energy_gate

    onset_indices = np.where(is_peak)[0]

    # Minimum inter-onset interval: ~30ms (avoid double-triggers)
    min_ioi = max(2, int(round(0.030 / max(hop_seconds, 1e-6))))
    if onset_indices.size > 1:
        filtered = [onset_indices[0]]
        for idx in onset_indices[1:]:
            if idx - filtered[-1] >= min_ioi:
                filtered.append(idx)
        onset_indices = np.array(filtered, dtype=np.int64)

    logger.debug("Detected %d onsets in %d frames (%.1fs)", onset_indices.size, spectral_flux.size,
                 spectral_flux.size * hop_seconds)
    return onset_indices


def _time_to_index(time_seconds: float, hop_seconds: float, energy_size: int) -> int:
    if energy_size <= 1:
        return 0
    return max(0, min(int(round(time_seconds / hop_seconds)), energy_size - 1))


def _index_to_time(index: int, hop_seconds: float) -> float:
    return round(max(0, index) * hop_seconds, 6)


def _local_threshold(values: np.ndarray) -> float:
    peak = float(values.max()) if values.size else 0.0
    floor = float(np.percentile(values, 25))
    return floor + (peak - floor) * 0.28


def _find_nearest_onset(
    onsets: np.ndarray,
    anchor_index: int,
    search_start: int,
    search_end: int,
    max_distance_frames: int = 0,
) -> int | None:
    """Find the nearest detected onset to anchor_index within [search_start, search_end].

    Returns the onset frame index, or None if no onset is found in range.
    If max_distance_frames > 0, rejects onsets farther than that from anchor.
    """
    if onsets.size == 0:
        return None
    mask = (onsets >= search_start) & (onsets <= search_end)
    candidates = onsets[mask]
    if candidates.size == 0:
        return None
    # Prefer onsets at or just after the anchor over ones before it.
    # An onset *before* the anchor likely belongs to the previous word's
    # tail transient, while an onset at/after the anchor is the actual
    # vocal attack for this word.  We penalize backward onsets by 40%.
    raw_distances = candidates - anchor_index  # signed: negative = before anchor
    penalties = np.where(raw_distances < 0, np.abs(raw_distances) * 1.4, np.abs(raw_distances).astype(float))
    best_idx = int(np.argmin(penalties))
    best_onset = int(candidates[best_idx])
    actual_distance = abs(int(candidates[best_idx]) - anchor_index)
    if max_distance_frames > 0 and actual_distance > max_distance_frames:
        return None
    return best_onset


def _onset_score(
    energy: np.ndarray,
    zcr: np.ndarray,
    spectral_flux: np.ndarray,
    anchor_index: int,
    start_index: int,
) -> np.ndarray:
    """Compute a combined onset-likelihood score for each frame.

    Higher score = more likely to be a word onset.
    Combines:
      - Energy rise (voiced activity starting)
      - ZCR spike (consonant attack — ת, כ, ש, ס, צ, פ, ב, ד, ג)
      - Spectral flux spike (timbral change — vowel-consonant transition)
      - Proximity to the original Whisper anchor (light bias, not dominant)
    """
    n = energy.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Energy gradient (positive = rising = onset)
    energy_grad = np.gradient(energy.astype(np.float32))
    energy_rise = np.clip(energy_grad, 0, None)
    energy_rise_norm = _normalize_curve(energy_rise) if energy_rise.max() > 1e-6 else np.zeros(n, np.float32)

    # ZCR peaks (consonant transients)
    zcr_norm = _normalize_curve(zcr) if zcr.max() > 1e-6 else np.zeros(n, np.float32)

    # Spectral flux peaks (timbral onsets) — now using real STFT-based flux
    sf_norm = _normalize_curve(spectral_flux) if spectral_flux.max() > 1e-6 else np.zeros(n, np.float32)

    # Proximity: light bias toward Whisper's guess (reduced from 0.35 → 0.15)
    distance = np.abs(np.arange(start_index, start_index + n) - anchor_index).astype(np.float32)
    max_dist = max(1.0, distance.max())
    proximity = 1.0 - (distance / max_dist)

    # Rebalanced: audio signals dominate, proximity is a tiebreaker
    score = energy_rise_norm * 0.25 + zcr_norm * 0.15 + sf_norm * 0.30 + proximity * 0.30
    return score.astype(np.float32)


def _find_word_onset(word: WordTiming, energy: np.ndarray, hop_seconds: float,
                     features: AudioFeatures | None = None) -> float:
    duration = max(word.end - word.start, MIN_WORD_DURATION_SEC)
    # Tighter backward search (0.35) to avoid snapping to previous word's
    # tail transient.  Forward search (0.4) is kept to still catch late onsets.
    conf_factor = max(0.4, 1.0 - word.confidence * 0.4) if word.confidence > 0 else 1.0
    search_back = min(BOUNDARY_SEARCH_SEC, max(0.04, duration * 0.35 * conf_factor))
    search_forward = min(BOUNDARY_SEARCH_SEC, max(0.06, duration * 0.4 * conf_factor))

    start_index = _time_to_index(max(0.0, word.start - search_back), hop_seconds, len(energy))
    end_index = _time_to_index(word.start + search_forward, hop_seconds, len(energy))
    local = energy[start_index:end_index + 1]
    if local.size == 0:
        return word.start

    anchor_index = _time_to_index(word.start, hop_seconds, len(energy))

    # PRIMARY: snap to the nearest globally-detected onset
    # This is far more accurate than searching for energy patterns in a window,
    # because onsets were detected using adaptive thresholding on the full track.
    if features is not None and features.onsets.size > 0:
        max_snap_distance = int(round(max(search_back, search_forward) / hop_seconds))
        onset_idx = _find_nearest_onset(
            features.onsets, anchor_index, start_index, end_index,
            max_distance_frames=max_snap_distance,
        )
        if onset_idx is not None:
            return _index_to_time(onset_idx, hop_seconds)

    # SECONDARY: multi-signal onset scoring (when no global onset found nearby)
    if features is not None and features.zcr.size > 0:
        local_zcr = features.zcr[start_index:end_index + 1]
        local_sf = features.spectral_flux[start_index:end_index + 1]
        if local_zcr.size < local.size:
            local_zcr = np.pad(local_zcr, (0, local.size - local_zcr.size))
        if local_sf.size < local.size:
            local_sf = np.pad(local_sf, (0, local.size - local_sf.size))
        score = _onset_score(local, local_zcr[:local.size], local_sf[:local.size], anchor_index, start_index)
        if score.size:
            best = int(np.argmax(score))
            return _index_to_time(start_index + best, hop_seconds)

    # FALLBACK: energy-only approach
    threshold = _local_threshold(local)
    valley_limit = max(1, anchor_index - start_index + 1)
    valley_index = start_index + int(np.argmin(local[:valley_limit]))
    above_threshold = np.where(energy[valley_index:end_index + 1] >= threshold)[0]
    if above_threshold.size:
        candidates = valley_index + above_threshold
        best_index = min(candidates, key=lambda idx: abs(int(idx) - anchor_index))
        return _index_to_time(int(best_index), hop_seconds)
    return _index_to_time(start_index + int(np.argmax(local)), hop_seconds)


def _find_word_offset(word: WordTiming, energy: np.ndarray, hop_seconds: float,
                      features: AudioFeatures | None = None) -> float:
    duration = max(word.end - word.start, MIN_WORD_DURATION_SEC)
    conf_factor = max(0.4, 1.0 - word.confidence * 0.4) if word.confidence > 0 else 1.0
    search_back = min(BOUNDARY_SEARCH_SEC, max(0.06, duration * 0.4 * conf_factor))
    search_forward = min(BOUNDARY_SEARCH_SEC, max(0.06, duration * 0.5 * conf_factor))

    start_index = _time_to_index(max(0.0, word.end - search_back), hop_seconds, len(energy))
    end_index = _time_to_index(word.end + search_forward, hop_seconds, len(energy))
    local = energy[start_index:end_index + 1]
    if local.size == 0:
        return word.end

    anchor_index = _time_to_index(word.end, hop_seconds, len(energy))

    # For offsets: find the energy valley between this word's body and the
    # next word's onset. Use spectral flux to identify where the next onset
    # begins — the offset should be just before it.
    if features is not None and features.zcr.size > 0:
        local_energy_norm = _normalize_curve(local)
        local_sf = features.spectral_flux[start_index:end_index + 1]
        if local_sf.size < local.size:
            local_sf = np.pad(local_sf, (0, local.size - local_sf.size))
        local_sf = local_sf[:local.size]
        sf_norm = _normalize_curve(local_sf) if local_sf.max() > 1e-6 else np.zeros_like(local)

        distance = np.abs(np.arange(start_index, start_index + local.size) - anchor_index).astype(np.float32)
        max_dist = max(1.0, distance.max())
        proximity = distance / max_dist

        # Score to minimize: want low energy + close to anchor; spectral flux
        # peaks indicate the NEXT word's onset so penalize them slightly
        offset_score = local_energy_norm * 0.35 + sf_norm * 0.20 + proximity * 0.45
        # Only consider candidates after the energy peak (don't jump backwards)
        peak_local = int(np.argmax(local[:max(1, anchor_index - start_index + 1)]))
        offset_score[:peak_local] = float('inf')
        best = int(np.argmin(offset_score))
        return _index_to_time(start_index + best, hop_seconds)

    # Fallback: energy-only approach
    threshold = _local_threshold(local)
    active_limit = max(1, anchor_index - start_index + 1)
    active = np.where(local[:active_limit] >= threshold)[0]
    active_index = start_index + (int(active[-1]) if active.size else int(np.argmax(local)))
    below_threshold = np.where(energy[active_index:end_index + 1] <= threshold)[0]
    if below_threshold.size:
        return _index_to_time(active_index + int(below_threshold[0]), hop_seconds)
    return _index_to_time(end_index, hop_seconds)


def _find_inter_word_boundary(
    previous_word: WordTiming,
    next_word: WordTiming,
    energy: np.ndarray,
    hop_seconds: float,
    features: AudioFeatures | None = None,
) -> float:
    search_start = max(
        previous_word.start + MIN_WORD_DURATION_SEC,
        min(previous_word.end, next_word.start) - BOUNDARY_SEARCH_SEC,
    )
    search_end = min(
        next_word.end - MIN_WORD_DURATION_SEC,
        max(previous_word.end, next_word.start) + BOUNDARY_SEARCH_SEC,
    )
    if search_end <= search_start:
        midpoint = max(
            previous_word.start + MIN_WORD_DURATION_SEC,
            min((previous_word.end + next_word.start) / 2, next_word.end - MIN_WORD_DURATION_SEC),
        )
        return midpoint

    start_index = _time_to_index(search_start, hop_seconds, len(energy))
    end_index = _time_to_index(search_end, hop_seconds, len(energy))
    local = energy[start_index:end_index + 1]
    if local.size == 0:
        return (search_start + search_end) / 2

    anchor_time = (previous_word.end + next_word.start) / 2
    anchor_index = _time_to_index(anchor_time, hop_seconds, len(energy))

    # PRIMARY: use globally-detected onset of the next word as boundary
    # The ideal boundary is just before the next word's consonant onset.
    if features is not None and features.onsets.size > 0:
        # Look for an onset in the search range that's closer to next_word's start
        next_word_anchor = _time_to_index(next_word.start, hop_seconds, len(energy))
        onset_idx = _find_nearest_onset(
            features.onsets, next_word_anchor, start_index, end_index,
        )
        if onset_idx is not None:
            # Place boundary slightly before the onset (the onset IS the next word)
            pre_onset = max(start_index, onset_idx - max(1, int(0.005 / hop_seconds)))
            return _index_to_time(pre_onset, hop_seconds)

    # SECONDARY: multi-signal boundary scoring
    if features is not None and features.zcr.size > 0:
        local_zcr = features.zcr[start_index:end_index + 1]
        local_sf = features.spectral_flux[start_index:end_index + 1]
        if local_zcr.size < local.size:
            local_zcr = np.pad(local_zcr, (0, local.size - local_zcr.size))
        if local_sf.size < local.size:
            local_sf = np.pad(local_sf, (0, local.size - local_sf.size))
        local_zcr = local_zcr[:local.size]
        local_sf = local_sf[:local.size]

        energy_norm = _normalize_curve(local)
        zcr_norm = _normalize_curve(local_zcr) if local_zcr.max() > 1e-6 else np.zeros_like(local)
        sf_norm = _normalize_curve(local_sf) if local_sf.max() > 1e-6 else np.zeros_like(local)

        distance = np.abs(np.arange(start_index, start_index + local.size) - anchor_index).astype(np.float32)
        max_dist = max(1.0, distance.max())
        proximity = distance / max_dist

        # Low energy is good, high sf/zcr is good (consonant onset), close to anchor is good
        boundary_score = energy_norm * 0.30 - zcr_norm * 0.10 - sf_norm * 0.15 + proximity * 0.45
        best = int(np.argmin(boundary_score))
        return _index_to_time(start_index + best, hop_seconds)

    # Fallback: energy-only
    quiet_threshold = float(np.percentile(local, 35))
    quiet_candidates = np.where(local <= quiet_threshold)[0]
    if quiet_candidates.size:
        best_offset = min(quiet_candidates, key=lambda offset: abs((start_index + int(offset)) - anchor_index))
        return _index_to_time(start_index + int(best_offset), hop_seconds)

    distances = np.abs(np.arange(start_index, end_index + 1) - anchor_index)
    penalty = local.max() - local.min() + 1e-6
    score = local + distances * (penalty / max(1, end_index - start_index))
    return _index_to_time(start_index + int(np.argmin(score)), hop_seconds)


def _stabilize_boundaries(boundaries: list[float], minimum_step: float) -> list[float]:
    if not boundaries:
        return []

    stabilized = [max(0.0, boundaries[0])]
    for boundary in boundaries[1:]:
        stabilized.append(max(boundary, stabilized[-1] + minimum_step))

    for index in range(len(stabilized) - 2, -1, -1):
        stabilized[index] = min(stabilized[index], stabilized[index + 1] - minimum_step)

    stabilized[0] = max(0.0, stabilized[0])
    for index in range(1, len(stabilized)):
        stabilized[index] = max(stabilized[index], stabilized[index - 1] + minimum_step)

    return [round(value, 6) for value in stabilized]


def _snap_boundaries_to_frames(boundaries: list[float], frame_rate: float | None) -> list[float]:
    if not frame_rate or frame_rate <= 0:
        return boundaries

    frame_duration = 1.0 / frame_rate
    snapped = []
    for index, boundary in enumerate(boundaries):
        if index == 0:
            snapped_value = math.floor(boundary / frame_duration) * frame_duration
        elif index == len(boundaries) - 1:
            snapped_value = math.ceil(boundary / frame_duration) * frame_duration
        else:
            snapped_value = round(boundary / frame_duration) * frame_duration
        snapped.append(max(0.0, snapped_value))

    return _stabilize_boundaries(snapped, frame_duration)


def _apply_frame_snapping(words: list[WordTiming], frame_rate: float | None) -> list[WordTiming]:
    if not frame_rate or frame_rate <= 0 or not words:
        return words

    frame_duration = 1.0 / frame_rate
    snapped_words = []
    cursor = max(0.0, round(words[0].start / frame_duration) * frame_duration)
    for word in words:
        # Use round (not floor) for start: floor systematically pushes every
        # word start backward by up to one frame (~40ms @25fps), making the
        # karaoke appear to start too early.
        start = max(cursor, round(word.start / frame_duration) * frame_duration)
        end = max(start + frame_duration, math.ceil(word.end / frame_duration) * frame_duration)
        snapped_words.append(
            WordTiming(
                word=word.word,
                start=round(start, 6),
                end=round(end, 6),
                confidence=word.confidence,
                source=word.source,
                aligned=word.aligned,
            )
        )
        cursor = end
    return snapped_words


def _boundary_search_radius(start_index: int, end_index: int, grapheme_count: int, hop_seconds: float) -> int:
    span = max(1, end_index - start_index)
    # Use the full average grapheme span as base (was half), so the search
    # can actually reach the correct consonant onset for stretched vowels.
    base = int(span / max(2, grapheme_count))
    # 80ms max (was 50ms) — enough to bridge a sustained vowel and find the
    # next consonant attack even when the mathematical target is far off.
    max_radius = max(1, int(0.08 / hop_seconds))
    min_radius = max(1, int(MIN_SUBWORD_DURATION_SEC / hop_seconds) * 2)
    return max(min_radius, min(max_radius, base))


def _scale_existing_subwords(source_word: WordTiming, target_word: WordTiming) -> list[SubWordTiming]:
    if not source_word.subwords:
        return []

    old_duration = max(source_word.end - source_word.start, 1e-6)
    new_duration = max(target_word.end - target_word.start, MIN_WORD_DURATION_SEC)
    remaining_count = len(source_word.subwords)
    cursor = target_word.start
    scaled: list[SubWordTiming] = []

    for index, subword in enumerate(source_word.subwords):
        start_ratio = (subword.start - source_word.start) / old_duration
        end_ratio = (subword.end - source_word.start) / old_duration
        proposed_start = target_word.start + start_ratio * new_duration
        proposed_end = target_word.start + end_ratio * new_duration

        start = target_word.start if index == 0 else max(cursor, proposed_start)
        minimum_end = start + MIN_SUBWORD_DURATION_SEC
        remaining_count -= 1
        max_end = target_word.end - remaining_count * MIN_SUBWORD_DURATION_SEC
        end = target_word.end if index == len(source_word.subwords) - 1 else min(proposed_end, max_end)
        end = max(minimum_end, end)
        end = min(target_word.end, end)

        scaled.append(
            SubWordTiming(
                text=subword.text,
                start=round(start, 6),
                end=round(end, 6),
                confidence=max(subword.confidence, source_word.confidence),
            )
        )
        cursor = end

    if scaled:
        scaled[0].start = target_word.start
        scaled[-1].end = target_word.end
    return scaled


def _subword_from_char_entries(
    text: str,
    entries: list[dict[str, object]],
    fallback_start: float,
    fallback_end: float,
    confidence: float,
) -> SubWordTiming:
    start = next((float(entry["start"]) for entry in entries if entry.get("start") is not None), fallback_start)
    end = next((float(entry["end"]) for entry in reversed(entries) if entry.get("end") is not None), fallback_end)
    if end <= start:
        end = fallback_end
    score_values = [float(entry.get("score", confidence)) for entry in entries if entry.get("score") is not None]
    return SubWordTiming(
        text=text,
        start=round(start, 6),
        end=round(end, 6),
        confidence=max(score_values) if score_values else confidence,
    )


def _uniform_subword_timings(
    graphemes: list[str],
    fallback_start: float,
    fallback_end: float,
    confidence: float,
) -> list[SubWordTiming]:
    if not graphemes:
        return []
    duration = max(fallback_end - fallback_start, MIN_WORD_DURATION_SEC)
    total_weight = sum(_grapheme_weight(grapheme) for grapheme in graphemes)
    cursor = fallback_start
    built = []
    for grapheme in graphemes:
        portion = duration * (_grapheme_weight(grapheme) / max(total_weight, 1e-6))
        next_cursor = min(fallback_end, cursor + portion)
        built.append(
            SubWordTiming(
                text=grapheme,
                start=round(cursor, 6),
                end=round(next_cursor, 6),
                confidence=confidence,
            )
        )
        cursor = next_cursor
    built[0].start = fallback_start
    built[-1].end = fallback_end
    return built


def _build_subwords_from_char_entries(
    word_text: str,
    entries: list[dict[str, object]],
    fallback_start: float,
    fallback_end: float,
    confidence: float,
) -> list[SubWordTiming]:
    graphemes = _split_graphemes(word_text)
    if not graphemes:
        return []
    if not entries:
        return _uniform_subword_timings(graphemes, fallback_start, fallback_end, confidence)

    filtered_entries = [entry for entry in entries if str(entry.get("char", "")).strip()]
    if not filtered_entries:
        return []

    cursor = 0
    built: list[SubWordTiming] = []
    for grapheme in graphemes:
        needed_codepoints = max(1, len(grapheme))
        cluster_entries: list[dict[str, object]] = []
        consumed = 0
        while cursor < len(filtered_entries) and consumed < needed_codepoints:
            entry = filtered_entries[cursor]
            cursor += 1
            entry_text = str(entry.get("char", ""))
            if not entry_text or entry_text.isspace():
                continue
            cluster_entries.append(entry)
            consumed += len(entry_text)
        if cluster_entries:
            built.append(_subword_from_char_entries(grapheme, cluster_entries, fallback_start, fallback_end, confidence))
        else:
            built.extend(
                _uniform_subword_timings(
                    graphemes[len(built):],
                    built[-1].end if built else fallback_start,
                    fallback_end,
                    confidence,
                )
            )
            break

    if not built:
        return []

    built[0].start = fallback_start
    built[-1].end = fallback_end
    for index in range(1, len(built)):
        built[index].start = max(built[index].start, built[index - 1].end)
        if built[index].end <= built[index].start:
            built[index].end = min(fallback_end, built[index].start + MIN_SUBWORD_DURATION_SEC)
    return built


def _build_subword_timings(word: WordTiming, energy: np.ndarray, hop_seconds: float,
                           features: AudioFeatures | None = None) -> list[SubWordTiming]:
    graphemes = _split_graphemes(word.word)
    if not graphemes:
        return []
    if len(graphemes) == 1:
        return [SubWordTiming(text=graphemes[0], start=word.start, end=word.end, confidence=word.confidence)]

    start_index = _time_to_index(word.start, hop_seconds, len(energy))
    end_index = _time_to_index(word.end, hop_seconds, len(energy))
    if end_index - start_index <= len(graphemes):
        duration = max(word.end - word.start, MIN_WORD_DURATION_SEC)
        cursor = word.start
        total_weight = sum(_grapheme_weight(grapheme) for grapheme in graphemes)
        subwords = []
        for grapheme in graphemes:
            part = duration * (_grapheme_weight(grapheme) / total_weight)
            next_cursor = min(word.end, cursor + part)
            subwords.append(SubWordTiming(text=grapheme, start=cursor, end=next_cursor, confidence=word.confidence))
            cursor = next_cursor
        subwords[-1].end = word.end
        return subwords

    local = energy[start_index:end_index + 1]
    activity = _activity_curve(local, hop_seconds)
    cumulative_activity = np.cumsum(np.maximum(activity, 1e-6))
    total_activity = float(cumulative_activity[-1]) if cumulative_activity.size else 0.0
    energy_curve = _normalize_curve(local.astype(np.float32))

    # Build the novelty/transition signal.
    # When AudioFeatures are available, use ZCR and real spectral flux for a richer
    # transition signal that captures consonant onsets within the word.
    if features is not None and features.zcr.size > 0:
        local_zcr = features.zcr[start_index:end_index + 1]
        local_sf = features.spectral_flux[start_index:end_index + 1]
        if local_zcr.size < local.size:
            local_zcr = np.pad(local_zcr, (0, local.size - local_zcr.size))
        if local_sf.size < local.size:
            local_sf = np.pad(local_sf, (0, local.size - local_sf.size))
        local_zcr = local_zcr[:local.size]
        local_sf = local_sf[:local.size]
        zcr_norm = _normalize_curve(local_zcr) if local_zcr.max() > 1e-6 else np.zeros(local.size, np.float32)
        sf_norm = _normalize_curve(local_sf) if local_sf.max() > 1e-6 else np.zeros(local.size, np.float32)
        energy_novelty = _normalize_curve(np.abs(np.gradient(local.astype(np.float32))))
        # ZCR is the best indicator of consonant onsets within a word (ת, כ, ש, ס, etc.)
        # Give it more weight so grapheme boundaries land on actual consonant attacks.
        transition_curve = energy_novelty * 0.15 + zcr_norm * 0.40 + sf_norm * 0.45
        transition_curve = _normalize_curve(transition_curve)
    else:
        transition_curve = _normalize_curve(np.abs(np.gradient(local.astype(np.float32))))

    # Collect onsets within this word for snap-to-onset at grapheme level
    word_onsets: list[int] = []
    if features is not None and features.onsets.size > 0:
        mask = (features.onsets >= start_index) & (features.onsets <= end_index)
        word_onsets = features.onsets[mask].tolist()

    weights = [_grapheme_weight(grapheme) for grapheme in graphemes]
    cumulative_weights = np.cumsum(weights[:-1])
    total_weight = max(1e-6, sum(weights))
    min_steps = max(1, int(math.ceil(MIN_SUBWORD_DURATION_SEC / hop_seconds)))
    search_radius = _boundary_search_radius(start_index, end_index, len(graphemes), hop_seconds)
    chosen_boundaries = [start_index]
    previous_index = start_index

    for boundary_number, cumulative_weight in enumerate(cumulative_weights, start=1):
        remaining_boundaries = len(graphemes) - boundary_number
        target_ratio = cumulative_weight / total_weight
        if total_activity > 0:
            target_local_index = int(np.searchsorted(cumulative_activity, total_activity * target_ratio, side="left"))
        else:
            target_local_index = int(round((end_index - start_index) * target_ratio))
        target_index = start_index + target_local_index
        minimum_index = previous_index + min_steps
        maximum_index = end_index - remaining_boundaries * min_steps
        if maximum_index <= minimum_index:
            chosen_index = minimum_index
        else:
            # PRIMARY: try to snap to an onset within the search radius
            snap_found = False
            if word_onsets:
                best_onset = None
                best_dist = float('inf')
                for oi in word_onsets:
                    if minimum_index <= oi <= maximum_index:
                        dist = abs(oi - target_index)
                        if dist < best_dist and dist <= search_radius * 2.5:
                            best_dist = dist
                            best_onset = oi
                if best_onset is not None:
                    chosen_index = best_onset
                    snap_found = True

            # SECONDARY: score-based search
            if not snap_found:
                search_start = max(minimum_index, target_index - search_radius)
                search_end = min(maximum_index, target_index + search_radius)
                offsets = np.arange(search_start, search_end + 1)
                search_local_start = search_start - start_index
                search_local_end = search_end - start_index + 1
                local_energy = energy_curve[search_local_start:search_local_end]
                local_transition = transition_curve[search_local_start:search_local_end]
                if local_energy.size:
                    proximity = np.abs(offsets - target_index) / max(1, search_radius)
                    # Prefer: energy dips, high transition (consonant onset), close to target
                    # Audio signals dominate — proximity is a light tiebreaker so
                    # stretched vowels and quick consonants are captured accurately.
                    score = local_energy * 0.25 + (1.0 - local_transition) * 0.55 + proximity * 0.20
                    chosen_index = int(offsets[int(np.argmin(score))])
                else:
                    chosen_index = max(minimum_index, min(target_index, maximum_index))
        chosen_boundaries.append(chosen_index)
        previous_index = chosen_index

    chosen_boundaries.append(end_index)

    # --- Energy-based post-placement adjustment ---
    # After initial boundary placement, verify each grapheme's duration is
    # proportional to its actual energy share.  If a grapheme that carries 40%
    # of the word's energy only got 15% of the duration, shift the boundaries
    # so sustained vowels are held longer and quick consonants are shorter.
    if len(chosen_boundaries) > 2 and local.size > 0:
        # Compute per-grapheme energy shares
        energy_shares: list[float] = []
        for gi in range(len(graphemes)):
            b_start_local = max(0, chosen_boundaries[gi] - start_index)
            b_end_local = min(local.size, chosen_boundaries[gi + 1] - start_index)
            if b_end_local <= b_start_local:
                energy_shares.append(0.0)
            else:
                energy_shares.append(float(np.sum(local[b_start_local:b_end_local])))
        total_energy_share = sum(energy_shares)

        if total_energy_share > 1e-6:
            total_span = chosen_boundaries[-1] - chosen_boundaries[0]
            if total_span > 0:
                # Compute target durations based on energy proportions
                energy_durations = [(e / total_energy_share) * total_span for e in energy_shares]
                # Blend: 40% energy-based, 60% current placement (avoid over-correction)
                ENERGY_BLEND = 0.40
                current_durations = [chosen_boundaries[gi + 1] - chosen_boundaries[gi] for gi in range(len(graphemes))]
                blended_durations = [
                    current_durations[gi] * (1.0 - ENERGY_BLEND) + energy_durations[gi] * ENERGY_BLEND
                    for gi in range(len(graphemes))
                ]
                # Enforce minimum duration
                for gi in range(len(blended_durations)):
                    blended_durations[gi] = max(min_steps, blended_durations[gi])
                # Rebuild boundaries from blended durations
                total_blended = sum(blended_durations)
                if total_blended > 0:
                    scale_factor = total_span / total_blended
                    cursor_idx = chosen_boundaries[0]
                    for gi in range(len(graphemes) - 1):
                        cursor_idx += blended_durations[gi] * scale_factor
                        chosen_boundaries[gi + 1] = int(round(cursor_idx))
                    # Keep first and last boundaries unchanged
                    chosen_boundaries[-1] = end_index

    chosen_boundaries = _stabilize_boundaries(
        [_index_to_time(index, hop_seconds) for index in chosen_boundaries],
        MIN_SUBWORD_DURATION_SEC,
    )

    subwords = []
    for index, grapheme in enumerate(graphemes):
        subwords.append(
            SubWordTiming(
                text=grapheme,
                start=chosen_boundaries[index],
                end=chosen_boundaries[index + 1],
                confidence=word.confidence,
            )
        )

    subwords[0].start = word.start
    subwords[-1].end = word.end
    for index in range(1, len(subwords)):
        subwords[index].start = max(subwords[index].start, subwords[index - 1].end)
    for index, subword in enumerate(subwords):
        if subword.end <= subword.start:
            subword.end = min(word.end, subword.start + MIN_SUBWORD_DURATION_SEC)
        if index == len(subwords) - 1:
            subword.end = word.end
    return subwords


def _refine_word_timings(
    audio_path: str,
    words: list[WordTiming],
    video_frame_rate: float | None = None,
) -> list[WordTiming]:
    if not words:
        return []

    try:
        features = _load_audio_features(audio_path)
        energy = features.energy
        hop_seconds = features.hop_seconds
    except Exception as exc:
        logger.warning("Audio refinement skipped for %s: %s", audio_path, exc)
        return _apply_frame_snapping(words, video_frame_rate)

    refined_words = [
        WordTiming(
            word=word.word,
            start=_find_word_onset(word, energy, hop_seconds, features=features),
            end=_find_word_offset(word, energy, hop_seconds, features=features),
            confidence=word.confidence,
            source=word.source,
            aligned=word.aligned,
        )
        for word in words
    ]

    for index in range(len(refined_words) - 1):
        boundary = _find_inter_word_boundary(
            refined_words[index], refined_words[index + 1], energy, hop_seconds, features=features,
        )
        refined_words[index].end = min(refined_words[index].end, boundary)
        refined_words[index + 1].start = max(refined_words[index + 1].start, boundary)

    stabilized_words = []
    cursor = 0.0
    minimum_step = 1.0 / video_frame_rate if video_frame_rate and video_frame_rate > 0 else MIN_WORD_DURATION_SEC
    for index, word in enumerate(refined_words):
        start = max(word.start, cursor)
        end = max(word.end, start + minimum_step)
        if index < len(refined_words) - 1 and end > refined_words[index + 1].start:
            end = max(start + MIN_WORD_DURATION_SEC, refined_words[index + 1].start)
        stabilized_words.append(
            WordTiming(
                word=word.word,
                start=round(start, 6),
                end=round(end, 6),
                confidence=word.confidence,
                source=word.source,
                aligned=word.aligned,
            )
        )
        cursor = end

    snapped_words = _apply_frame_snapping(stabilized_words, video_frame_rate)
    final_words = []
    for source_word, target_word in zip(words, snapped_words):
        # Always re-analyze subword boundaries using audio features, even
        # when the source word already has subword timings from WhisperX.
        # Linear scaling (_scale_existing_subwords) preserves proportions but
        # ignores the actual audio — energy+ZCR analysis gives more accurate
        # letter-level boundaries.
        subwords = _build_subword_timings(target_word, energy, hop_seconds, features=features)
        if not subwords:
            # Fallback: if energy analysis fails, use scaled existing subwords
            subwords = _scale_existing_subwords(source_word, target_word)
        final_words.append(
            WordTiming(
                word=target_word.word,
                start=target_word.start,
                end=target_word.end,
                confidence=target_word.confidence,
                source=target_word.source,
                aligned=target_word.aligned,
                subwords=subwords,
            )
        )
    return final_words


def _clamp_segment_to_floor(segment: TranscriptSegment, floor: float, minimum_step: float) -> TranscriptSegment:
    """Shift a segment's words forward so the segment starts at or after *floor*.

    Called when cross-segment ordering is violated (a segment starts before the
    previous segment ends).  Words that already sit past the floor are kept as-is;
    only the leading words that overlap the floor are shifted.
    """
    if not segment.words or segment.start >= floor - 1e-6:
        return segment

    shift = floor - segment.start
    clamped_words = []
    cursor = floor
    for word in segment.words:
        start = max(word.start + shift, cursor)
        end = max(word.end + shift, start + minimum_step)
        clamped_words.append(
            WordTiming(
                word=word.word,
                start=round(start, 6),
                end=round(end, 6),
                confidence=word.confidence,
                source=word.source,
                aligned=False,  # timing was adjusted; mark unaligned
                subwords=word.subwords,
            )
        )
        cursor = end

    return TranscriptSegment(
        words=clamped_words,
        text=segment.text,
        start=clamped_words[0].start,
        end=clamped_words[-1].end,
    )


def _clip_segment_overlaps(
    segments: list[TranscriptSegment],
    minimum_step: float,
) -> list[TranscriptSegment]:
    """Final pass: ensure segment[i].end <= segment[i+1].start.

    Handles both minor overlaps (last word's energy search bleeding past the
    boundary) and major overlaps (disordered segments from poor alignment).
    When a segment's end exceeds the next segment's start, the overlapping
    words and subwords are clipped.  As a safety net, if ALL words would be
    removed by clipping, the segment's end is set to the next segment's start
    and the last word's end is adjusted accordingly.
    """
    if len(segments) <= 1:
        return segments

    result = list(segments)
    for i in range(len(result) - 1):
        curr = result[i]
        nxt = result[i + 1]
        if not curr.words or curr.end <= nxt.start + 1e-6:
            continue

        new_end = nxt.start
        # Safety: new_end must be after the segment's start
        if new_end <= curr.start + minimum_step:
            # Segments are fundamentally disordered — give this segment a
            # minimal time slice ending just before the next one starts.
            new_end = max(curr.start + minimum_step, nxt.start)

        clipped_words: list[WordTiming] = []
        for word in curr.words:
            if word.start >= new_end:
                break
            word_end = max(word.start + minimum_step, min(word.end, new_end))
            clipped_subwords: list[SubWordTiming] = []
            for sw in word.subwords:
                if sw.start >= new_end:
                    break
                sw_end = max(sw.start + MIN_SUBWORD_DURATION_SEC, min(sw.end, new_end))
                clipped_subwords.append(
                    SubWordTiming(
                        text=sw.text,
                        start=sw.start,
                        end=round(sw_end, 6),
                        confidence=sw.confidence,
                    )
                )
            clipped_words.append(
                WordTiming(
                    word=word.word,
                    start=word.start,
                    end=round(word_end, 6),
                    confidence=word.confidence,
                    source=word.source,
                    aligned=word.aligned,
                    subwords=clipped_subwords,
                )
            )

        if not clipped_words:
            # All words start after new_end — compress the entire segment
            # into the available time slice.
            available = max(new_end - curr.start, minimum_step * len(curr.words))
            step = available / max(1, len(curr.words))
            for wi, word in enumerate(curr.words):
                w_start = round(curr.start + wi * step, 6)
                w_end = round(curr.start + (wi + 1) * step, 6)
                clipped_words.append(
                    WordTiming(
                        word=word.word,
                        start=w_start,
                        end=min(w_end, new_end),
                        confidence=0.1,
                        source="overlap_fix",
                        aligned=False,
                        subwords=[],
                    )
                )

        result[i] = TranscriptSegment(
            words=clipped_words,
            text=curr.text,
            start=clipped_words[0].start,
            end=clipped_words[-1].end,
        )
        logger.debug(
            "Clipped segment end: '%s' %.3f → %.3f (next starts %.3f)",
            curr.text[:30],
            curr.end,
            result[i].end,
            nxt.start,
        )

    return result


def _refine_segments(
    audio_path: str,
    segments: list[TranscriptSegment],
    video_frame_rate: float | None = None,
) -> list[TranscriptSegment]:
    minimum_step = 1.0 / video_frame_rate if video_frame_rate and video_frame_rate > 0 else MIN_WORD_DURATION_SEC
    refined_segments = []
    prev_end = 0.0
    for segment in segments:
        if not segment.words:
            refined_segments.append(segment)
            continue
        refined_words = _refine_word_timings(audio_path, segment.words, video_frame_rate=video_frame_rate)
        refined_segment = TranscriptSegment(
            words=refined_words,
            text=" ".join(word.word for word in refined_words).strip(),
            start=refined_words[0].start,
            end=refined_words[-1].end,
        )
        # Cross-segment safety net: ensure this segment starts after the previous one ends.
        if refined_segment.start < prev_end - 1e-6:
            logger.debug(
                "Cross-segment overlap detected: segment '%s' starts at %.3f but previous ended at %.3f — clamping.",
                refined_segment.text[:30],
                refined_segment.start,
                prev_end,
            )
            refined_segment = _clamp_segment_to_floor(refined_segment, prev_end, minimum_step)
        prev_end = refined_segment.end
        refined_segments.append(refined_segment)
    # Second pass: clip each segment's end to the start of the following one.
    # The first pass only guards against a segment *starting* too early; the
    # last word's energy-search can still push segment[i].end past segment[i+1].start.
    return _clip_segment_overlaps(refined_segments, minimum_step)


def _validate_final_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Final safety net: fix any remaining timing violations in the output.

    Ensures:
      1. All words within each segment are strictly ordered.
      2. All segments are strictly ordered (no overlaps).
      3. All subwords within each word are strictly ordered.
    """
    if not segments:
        return segments

    fixed = []
    global_cursor = 0.0
    for segment in segments:
        if not segment.words:
            fixed.append(segment)
            continue

        # Fix word ordering within segment
        fixed_words = []
        cursor = max(global_cursor, segment.start)
        for word in segment.words:
            start = max(cursor, word.start)
            end = max(start + MIN_WORD_DURATION_SEC, word.end)
            # Fix subword ordering within word
            fixed_subwords = []
            sw_cursor = start
            for sw in word.subwords:
                sw_start = max(sw_cursor, sw.start)
                sw_end = max(sw_start + MIN_SUBWORD_DURATION_SEC, sw.end)
                sw_end = min(sw_end, end)
                if sw_start >= end:
                    break
                fixed_subwords.append(SubWordTiming(
                    text=sw.text,
                    start=round(sw_start, 6),
                    end=round(sw_end, 6),
                    confidence=sw.confidence,
                ))
                sw_cursor = sw_end

            fixed_words.append(WordTiming(
                word=word.word,
                start=round(start, 6),
                end=round(end, 6),
                confidence=word.confidence,
                source=word.source,
                aligned=word.aligned,
                subwords=fixed_subwords,
            ))
            cursor = end

        new_seg = TranscriptSegment(
            words=fixed_words,
            text=segment.text,
            start=fixed_words[0].start,
            end=fixed_words[-1].end,
        )
        global_cursor = new_seg.end
        fixed.append(new_seg)

    return fixed


class SequenceHebrewAligner:
    name = "sequence_aligner"

    def align(
        self,
        audio_path: str,
        approved_segments: list[TranscriptSegment],
        draft_segments: list[TranscriptSegment],
        video_frame_rate: float | None = None,
    ) -> AlignedTranscript:
        approved_words, _approved_segment_indices = _flatten_words(approved_segments)
        draft_words, _draft_segment_indices = _flatten_words(draft_segments)
        if not approved_words:
            raise AlignmentError("Approved transcript is empty.", "אין טקסט מאושר ליישור.")
        if not draft_words:
            raise AlignmentError("Draft transcript is empty.", "אין טיוטת תמלול ליישור.")

        approved_norm = [_normalize_word(word.word) for word in approved_words]
        draft_norm = [_normalize_word(word.word) for word in draft_words]
        matcher = SequenceMatcher(None, approved_norm, draft_norm, autojunk=False)

        final_words: list[WordTiming | None] = [None] * len(approved_words)

        for tag, a_start, a_end, d_start, d_end in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(a_end - a_start):
                    draft_word = draft_words[d_start + offset]
                    approved_word = approved_words[a_start + offset]
                    final_words[a_start + offset] = WordTiming(
                        word=approved_word.word,
                        start=draft_word.start,
                        end=draft_word.end,
                        confidence=draft_word.confidence,
                        source="forced_aligner",
                        aligned=True,
                    )
            elif tag == "replace":
                # Ordered matching: iterate approved words and match them to
                # draft words IN CHRONOLOGICAL ORDER to avoid timing inversions.
                # Each draft word can only be used once, and we advance a cursor
                # to ensure monotonic assignment.
                used_draft: set[int] = set()
                draft_cursor = d_start
                for approved_index in range(a_start, a_end):
                    approved_token = approved_norm[approved_index]
                    best_match = None
                    best_ratio = 0.0
                    # Search forward from the cursor to maintain order
                    for draft_index in range(draft_cursor, d_end):
                        if draft_index in used_draft:
                            continue
                        ratio = SequenceMatcher(None, approved_token, draft_norm[draft_index], autojunk=False).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = draft_index
                            if ratio > 0.95:
                                break  # near-perfect match, stop searching
                    if best_match is not None and best_ratio >= 0.55:
                        draft_word = draft_words[best_match]
                        approved_word = approved_words[approved_index]
                        final_words[approved_index] = WordTiming(
                            word=approved_word.word,
                            start=draft_word.start,
                            end=draft_word.end,
                            confidence=max(draft_word.confidence, best_ratio),
                            source="forced_aligner",
                            aligned=True,
                        )
                        used_draft.add(best_match)
                        draft_cursor = best_match + 1

        # Enforce strict temporal monotonicity across all matched words.
        # Any word whose start is before the previous word's end gets reset to
        # None for the interpolation pass to place correctly.
        _prev_end = 0.0
        for _i in range(len(final_words)):
            _fw = final_words[_i]
            if _fw is not None:
                if _fw.start < _prev_end - 1e-3:
                    final_words[_i] = None  # will be re-placed by interpolation
                else:
                    _prev_end = max(_prev_end, _fw.end)

        for index, approved_word in enumerate(approved_words):
            if final_words[index] is None:
                final_words[index] = _interpolate_word(index, approved_word, final_words, approved_words)

        built_words = [word for word in final_words if word is not None]
        segments = _rebuild_segments(approved_segments, built_words)
        segments = _refine_segments(audio_path, segments, video_frame_rate=video_frame_rate)
        segments = _validate_final_segments(segments)
        unaligned_count = sum(1 for segment in segments for word in segment.words if not word.aligned)
        return AlignedTranscript(
            segments=segments,
            provider=self.name,
            fully_aligned=unaligned_count == 0,
            unaligned_word_count=unaligned_count,
        )


class WhisperXHebrewAligner:
    name = "whisperx_hebrew"
    _model = None
    _metadata = None

    def _load(self):
        if self._model is None or self._metadata is None:
            import whisperx

            self._model, self._metadata = whisperx.load_align_model(
                language_code="he",
                model_name=ALIGNMENT_MODEL_NAME,
                device="cpu",
            )
        return self._model, self._metadata

    def align(
        self,
        audio_path: str,
        approved_segments: list[TranscriptSegment],
        draft_segments: list[TranscriptSegment],
        video_frame_rate: float | None = None,
    ) -> AlignedTranscript:
        del draft_segments
        try:
            import whisperx
        except ImportError as exc:
            raise AlignmentError(str(exc), "מנוע היישור whisperx לא מותקן.") from exc

        model, metadata = self._load()
        payload = [
            {"text": segment.text, "start": segment.start, "end": segment.end}
            for segment in approved_segments
            if segment.text.strip()
        ]
        try:
            audio = whisperx.load_audio(audio_path)
            result = whisperx.align(payload, model, metadata, audio, "cpu", return_char_alignments=True)
        except Exception as exc:
            raise AlignmentError(str(exc), "whisperx נכשל ביישור הטקסט המאושר.") from exc

        final_segments = []
        unaligned_count = 0
        for source_segment, aligned_segment in zip(approved_segments, result.get("segments", [])):
            words = []
            aligned_words = aligned_segment.get("words") or []
            segment_chars = aligned_segment.get("chars") or []
            segment_char_cursor = 0
            fallback_words = source_segment.words
            for index, word_data in enumerate(aligned_words):
                fallback = fallback_words[min(index, len(fallback_words) - 1)] if fallback_words else None
                start = word_data.get("start", fallback.start if fallback else source_segment.start)
                end = word_data.get("end", fallback.end if fallback else source_segment.end)
                aligned = "start" in word_data and "end" in word_data
                if not aligned:
                    unaligned_count += 1
                word_text = str(word_data.get("word", "")).strip()
                direct_chars = word_data.get("chars") or []
                if direct_chars:
                    subwords = _build_subwords_from_char_entries(
                        word_text,
                        direct_chars,
                        float(start),
                        float(end),
                        float(word_data.get("score", 0.0)),
                    )
                else:
                    needed_codepoints = len("".join(_split_graphemes(word_text)))
                    collected_chars: list[dict[str, object]] = []
                    while segment_char_cursor < len(segment_chars) and len(collected_chars) < needed_codepoints:
                        entry = segment_chars[segment_char_cursor]
                        segment_char_cursor += 1
                        if not str(entry.get("char", "")).strip():
                            continue
                        collected_chars.append(entry)
                    subwords = _build_subwords_from_char_entries(
                        word_text,
                        collected_chars,
                        float(start),
                        float(end),
                        float(word_data.get("score", 0.0)),
                    )
                words.append(
                    WordTiming(
                        word=word_text,
                        start=float(start),
                        end=float(end),
                        confidence=float(word_data.get("score", 0.0)),
                        source="forced_aligner",
                        aligned=aligned,
                        subwords=subwords,
                    )
                )
            if not words:
                words = [
                    WordTiming(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        confidence=word.confidence,
                        source="forced_aligner",
                        aligned=False,
                    )
                    for word in source_segment.words
                ]
                unaligned_count += len(words)
            final_segments.append(
                TranscriptSegment(
                    words=words,
                    text=" ".join(word.word for word in words).strip(),
                    start=words[0].start,
                    end=words[-1].end,
                )
            )

        final_segments = _refine_segments(audio_path, final_segments, video_frame_rate=video_frame_rate)
        final_segments = _validate_final_segments(final_segments)
        return AlignedTranscript(
            segments=final_segments,
            provider=self.name,
            fully_aligned=unaligned_count == 0,
            unaligned_word_count=unaligned_count,
        )


class AutoHebrewAligner:
    name = "auto_aligner"

    def __init__(self):
        self.primary = WhisperXHebrewAligner() if importlib.util.find_spec("whisperx") is not None else None
        self.fallback = SequenceHebrewAligner()
        self.last_warning_message = ""

    def align(
        self,
        audio_path: str,
        approved_segments: list[TranscriptSegment],
        draft_segments: list[TranscriptSegment],
        video_frame_rate: float | None = None,
    ) -> AlignedTranscript:
        self.last_warning_message = ""
        if self.primary is None:
            return self.fallback.align(
                audio_path,
                approved_segments,
                draft_segments,
                video_frame_rate=video_frame_rate,
            )

        try:
            return self.primary.align(
                audio_path,
                approved_segments,
                draft_segments,
                video_frame_rate=video_frame_rate,
            )
        except AlignmentError as exc:
            self.last_warning_message = "מנוע היישור הראשי נכשל על הטקסט הזה, אז עברתי אוטומטית ליישור חלופי יציב יותר."
            logger.warning("WhisperX alignment failed, falling back to sequence aligner: %s", exc)
            return self.fallback.align(
                audio_path,
                approved_segments,
                draft_segments,
                video_frame_rate=video_frame_rate,
            )


def get_alignment_provider():
    provider = ALIGNMENT_PROVIDER.lower()
    whisperx_installed = importlib.util.find_spec("whisperx") is not None
    if provider == "whisperx":
        if not whisperx_installed:
            raise AlignmentError("whisperx is not installed", "מנוע whisperx לא זמין בסביבה.")
        return WhisperXHebrewAligner()
    if provider == "auto":
        return AutoHebrewAligner()
    return SequenceHebrewAligner()


# ---------------------------------------------------------------------------
# Timing quality validation
# ---------------------------------------------------------------------------

def validate_timing_quality(segments: list[TranscriptSegment]) -> list[str]:
    """Analyse aligned segments and return warnings for timing anomalies.

    The function checks for:
      • Words with zero or negative duration
      • Words that are suspiciously long (>3 s for a single word)
      • Segment overlaps (segment[i].end > segment[i+1].start)
      • Backwards words (word[i].start > word[i+1].start within a segment)
      • Unaligned word ratio exceeding 25 %
      • Subword timing violations (out of order, exceeding word bounds)

    Returns a list of Hebrew warning strings suitable for the user, or an
    empty list when everything looks clean.
    """
    warnings: list[str] = []
    if not segments:
        return warnings

    total_words = 0
    unaligned_words = 0
    zero_duration_words = 0
    long_words = 0
    backwards_pairs = 0
    segment_overlaps = 0
    subword_violations = 0

    for seg_idx, segment in enumerate(segments):
        # Check segment ordering
        if seg_idx > 0:
            prev = segments[seg_idx - 1]
            if segment.start < prev.end - 1e-3:
                segment_overlaps += 1

        for word_idx, word in enumerate(segment.words):
            total_words += 1
            if not word.aligned:
                unaligned_words += 1

            duration = word.end - word.start
            if duration <= 1e-6:
                zero_duration_words += 1
            elif duration > 3.0:
                long_words += 1

            # Check word ordering within segment
            if word_idx > 0:
                prev_word = segment.words[word_idx - 1]
                if word.start < prev_word.start - 1e-3:
                    backwards_pairs += 1

            # Check subword sanity
            for sw_idx, sw in enumerate(word.subwords):
                if sw.start < word.start - 0.02 or sw.end > word.end + 0.02:
                    subword_violations += 1
                if sw_idx > 0 and sw.start < word.subwords[sw_idx - 1].start - 1e-3:
                    subword_violations += 1

    if zero_duration_words:
        warnings.append(f"נמצאו {zero_duration_words} מילים עם משך אפס — ייתכן שהן לא מיושרות נכון.")

    if long_words:
        warnings.append(f"נמצאו {long_words} מילים עם משך חריג (מעל 3 שניות) — ייתכן שהטיימינג לא מדויק.")

    if segment_overlaps:
        warnings.append(f"נמצאו {segment_overlaps} חפיפות בין קטעים — שורות כתוביות עלולות להופיע יחד.")

    if backwards_pairs:
        warnings.append(f"נמצאו {backwards_pairs} מילים בסדר הפוך בתוך קטע — סימן לבעיית יישור.")

    if total_words > 0 and unaligned_words / total_words > 0.25:
        pct = int(round(unaligned_words / total_words * 100))
        warnings.append(f"{pct}% מהמילים לא יושרו ישירות — דיוק הטיימינג עלול להיות נמוך.")

    if subword_violations:
        warnings.append(f"נמצאו {subword_violations} בעיות בטיימינג תת-מילתי — עלול להשפיע על אנימציית הגרפמות.")

    return warnings
