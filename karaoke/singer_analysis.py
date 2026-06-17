"""Multi-singer analysis based on the isolated vocal track."""

from __future__ import annotations

import logging
import math
import re
from difflib import SequenceMatcher

import numpy as np

from .models import SingerAnalysisResult, SingerProfile, SingerSegmentAssignment, TranscriptSegment

logger = logging.getLogger(__name__)

ANALYSIS_SAMPLE_RATE = 16_000
MIN_SEGMENT_SECONDS = 0.45
MIN_SEGMENT_SAMPLES = int(ANALYSIS_SAMPLE_RATE * MIN_SEGMENT_SECONDS)
MAX_SINGERS = 3
PALETTE = (
    {
        "primary_color": "&H00FFF8F1",
        "secondary_color": "&H00FF8E3A",
        "outline_color": "&H00A04917",
        "shadow_color": "&H700E0E10",
    },
    {
        "primary_color": "&H00F6FFF3",
        "secondary_color": "&H0092D55A",
        "outline_color": "&H00457016",
        "shadow_color": "&H700A120A",
    },
    {
        "primary_color": "&H00F3F3FF",
        "secondary_color": "&H006868FF",
        "outline_color": "&H00262692",
        "shadow_color": "&H70101014",
    },
)


def _normalize_line(text: str) -> str:
    normalized = re.sub(r"[^\w\u0590-\u05FF]+", " ", (text or "").lower(), flags=re.UNICODE)
    return " ".join(part for part in normalized.split() if part)


def _line_similarity(left: str, right: str) -> float:
    left_norm = _normalize_line(left)
    right_norm = _normalize_line(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def _is_substantial_line(text: str) -> bool:
    normalized = _normalize_line(text)
    word_count = len(normalized.split())
    char_count = len(normalized.replace(" ", ""))
    return word_count >= 3 or char_count >= 10


def _build_repeat_anchor_map(lines: list[str]) -> dict[int, int]:
    anchors: dict[int, int] = {}
    similarity_cache: dict[tuple[int, int], float] = {}

    def _sim(left_index: int, right_index: int) -> float:
        key = (left_index, right_index)
        if key not in similarity_cache:
            similarity_cache[key] = _line_similarity(lines[left_index], lines[right_index])
        return similarity_cache[key]

    for right_start in range(1, len(lines)):
        for left_start in range(right_start):
            if _sim(left_start, right_start) < 0.84:
                continue
            run_length = 0
            while (
                right_start + run_length < len(lines)
                and left_start + run_length < right_start
                and _sim(left_start + run_length, right_start + run_length) >= 0.84
            ):
                run_length += 1
            if run_length < 2:
                continue
            for offset in range(run_length):
                anchors.setdefault(right_start + offset, left_start + offset)

    for right_index in range(1, len(lines)):
        if right_index in anchors or not _is_substantial_line(lines[right_index]):
            continue
        for left_index in range(right_index):
            if _sim(left_index, right_index) >= 0.96:
                anchors[right_index] = left_index
                break

    return anchors


def _extract_repeat_families(lines: list[str]) -> list[dict[str, object]]:
    anchor_map = _build_repeat_anchor_map(lines)
    families: dict[tuple[int, int], dict[str, object]] = {}
    index = 0
    while index < len(lines):
        source_start = anchor_map.get(index)
        if source_start is None:
            index += 1
            continue

        length = 1
        while index + length < len(lines) and anchor_map.get(index + length) == source_start + length:
            length += 1

        substantial_count = sum(1 for offset in range(length) if _is_substantial_line(lines[index + offset]))
        line_scores = [_line_similarity(lines[source_start + offset], lines[index + offset]) for offset in range(length)]
        strong_match_count = sum(1 for score in line_scores if score >= 0.96)
        average_score = sum(line_scores) / len(line_scores)
        if length >= 2 and substantial_count >= 2 and (strong_match_count >= 1 or average_score >= 0.94):
            key = (source_start, length)
            family = families.setdefault(
                key,
                {
                    "source_start": source_start,
                    "length": length,
                    "intervals": [(source_start, source_start + length)],
                },
            )
            interval = (index, index + length)
            if interval not in family["intervals"]:
                family["intervals"].append(interval)
        index += length

    ordered_families = sorted(
        families.values(),
        key=lambda item: (len(item["intervals"]) * item["length"], item["length"]),
        reverse=True,
    )
    accepted_families: list[dict[str, object]] = []
    occupied: set[int] = set()
    for family in ordered_families:
        intervals = sorted(set(family["intervals"]))
        if len(intervals) < 2:
            continue
        if any(any(line_index in occupied for line_index in range(start, end)) for start, end in intervals):
            continue
        for start, end in intervals:
            occupied.update(range(start, end))
        family["intervals"] = intervals
        accepted_families.append(family)

    accepted_families.sort(key=lambda item: min(start for start, _end in item["intervals"]))
    return accepted_families


def _detect_structural_sections(segments: list[TranscriptSegment]) -> list[dict[str, object]]:
    if not segments:
        return []

    lines = [segment.text for segment in segments]
    chorus_intervals = [
        {
            "type": "chorus",
            "family_index": family_index,
            "start": start,
            "end": end,
        }
        for family_index, family in enumerate(_extract_repeat_families(lines), start=1)
        for start, end in family["intervals"]
    ]
    chorus_intervals.sort(key=lambda item: (item["start"], item["end"]))

    sections: list[dict[str, object]] = []
    cursor = 0
    verse_counter = 0
    chorus_counter = 0
    for interval in chorus_intervals:
        start = int(interval["start"])
        end = int(interval["end"])
        if cursor < start:
            verse_counter += 1
            non_chorus_type = "intro" if not sections and start - cursor <= 2 else "verse"
            sections.append(
                {
                    "type": non_chorus_type,
                    "index": verse_counter,
                    "start": cursor,
                    "end": start,
                }
            )
        chorus_counter += 1
        sections.append(
            {
                "type": "chorus",
                "index": chorus_counter,
                "family_index": interval["family_index"],
                "start": start,
                "end": end,
            }
        )
        cursor = end

    if cursor < len(segments):
        remaining_type = "outro" if sections and sections[-1]["type"] == "chorus" and len(segments) - cursor <= 3 else "verse"
        verse_counter += 1
        sections.append(
            {
                "type": remaining_type,
                "index": verse_counter,
                "start": cursor,
                "end": len(segments),
            }
        )

    if not sections:
        sections.append({"type": "verse", "index": 1, "start": 0, "end": len(segments)})

    return [section for section in sections if int(section["end"]) > int(section["start"])]


def _split_section(section: dict[str, object], segments: list[TranscriptSegment]) -> list[tuple[int, int, int]]:
    start = int(section["start"])
    end = int(section["end"])
    segment_count = end - start
    if segment_count <= 1:
        return [(start, end, 0)]

    best_split = start + max(1, math.ceil(segment_count / 2))
    best_split = min(best_split, end - 1)

    return [
        (start, best_split, 0),
        (best_split, end, 1),
    ]


def _duet_profiles() -> tuple[list[SingerProfile], dict[tuple[int, int], SingerProfile]]:
    profiles: list[SingerProfile] = []
    by_lane_and_color: dict[tuple[int, int], SingerProfile] = {}
    for lane_index in range(2):
        for color_index, palette in enumerate(PALETTE):
            profile = SingerProfile(
                singer_id=f"duet_lane_{lane_index + 1}_color_{color_index + 1}",
                label=f"Duet {'A' if lane_index == 0 else 'B'}",
                lane_index=lane_index,
                **palette,
            )
            profiles.append(profile)
            by_lane_and_color[(lane_index, color_index)] = profile
    return profiles, by_lane_and_color


def _lazy_import_librosa():
    import librosa

    return librosa


def _title_suggests_multiple_singers(title: str) -> bool:
    head = re.split(r"\s+-\s+", title or "", maxsplit=1)[0].lower()
    return any(token in head for token in (" feat ", " ft. ", " ft ", "&", " x ", " with "))


def _safe_stat_pair(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0]
    return [float(np.mean(values)), float(np.std(values))]


def _pairwise_distances(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = data[:, None, :] - data[None, :, :]
    return np.sqrt(np.maximum(0.0, np.sum(diff * diff, axis=2))).astype(np.float32)


def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if data.shape[0] < 2 or unique_labels.size < 2:
        return -1.0

    distances = _pairwise_distances(data)
    scores: list[float] = []
    for index, label in enumerate(labels):
        same_mask = labels == label
        same_mask[index] = False
        if same_mask.any():
            intra = float(np.mean(distances[index, same_mask]))
        else:
            intra = 0.0

        inter_candidates = [
            float(np.mean(distances[index, labels == other_label]))
            for other_label in unique_labels
            if other_label != label
        ]
        if not inter_candidates:
            continue
        inter = min(inter_candidates)
        if max(intra, inter) <= 1e-6:
            scores.append(0.0)
        else:
            scores.append((inter - intra) / max(intra, inter))

    if not scores:
        return -1.0
    return float(np.mean(scores))


def _normalize_features(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (data - mean) / std


def _kmeans_plus_plus_init(data: np.ndarray, cluster_count: int, rng: np.random.Generator) -> np.ndarray:
    centers = [data[int(rng.integers(0, data.shape[0]))]]
    while len(centers) < cluster_count:
        distances = np.min(
            np.sum((data[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = float(distances.sum())
        if total <= 1e-9:
            centers.append(data[int(rng.integers(0, data.shape[0]))])
            continue
        probabilities = distances / total
        centers.append(data[int(rng.choice(data.shape[0], p=probabilities))])
    return np.asarray(centers, dtype=np.float32)


def _run_kmeans(data: np.ndarray, cluster_count: int, *, seed: int = 7, restarts: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if cluster_count <= 1 or data.shape[0] <= 1:
        return np.zeros(data.shape[0], dtype=np.int32), data[:1].copy()

    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = math.inf

    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        centers = _kmeans_plus_plus_init(data, cluster_count, rng)

        for _ in range(40):
            distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1).astype(np.int32)

            new_centers = centers.copy()
            for cluster_index in range(cluster_count):
                members = data[labels == cluster_index]
                if members.size == 0:
                    farthest_index = int(np.argmax(np.min(distances, axis=1)))
                    new_centers[cluster_index] = data[farthest_index]
                else:
                    new_centers[cluster_index] = members.mean(axis=0)

            if np.allclose(new_centers, centers, atol=1e-4):
                centers = new_centers
                break
            centers = new_centers

        distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        inertia = float(np.sum(np.min(distances, axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    if best_labels is None or best_centers is None:
        return np.zeros(data.shape[0], dtype=np.int32), data[:1].copy()
    return best_labels, best_centers


def _extract_segment_features(audio: np.ndarray, sample_rate: int, segment: TranscriptSegment) -> np.ndarray | None:
    start_sample = max(0, int(round(segment.start * sample_rate)))
    end_sample = min(audio.shape[0], int(round(segment.end * sample_rate)))
    if end_sample - start_sample < MIN_SEGMENT_SAMPLES:
        return None

    segment_audio = audio[start_sample:end_sample]
    if segment_audio.size < MIN_SEGMENT_SAMPLES:
        return None

    librosa = _lazy_import_librosa()
    try:
        trimmed, _trim_index = librosa.effects.trim(segment_audio, top_db=28)
    except Exception:
        trimmed = segment_audio
    if trimmed.size >= MIN_SEGMENT_SAMPLES // 2:
        segment_audio = trimmed

    if segment_audio.size < MIN_SEGMENT_SAMPLES // 2:
        return None

    try:
        mfcc = librosa.feature.mfcc(y=segment_audio, sr=sample_rate, n_mfcc=13, n_fft=1024, hop_length=160)
        centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sample_rate, n_fft=1024, hop_length=160)
        bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sample_rate, n_fft=1024, hop_length=160)
        rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sample_rate, n_fft=1024, hop_length=160)
        rms = librosa.feature.rms(y=segment_audio, frame_length=1024, hop_length=160)
        zcr = librosa.feature.zero_crossing_rate(segment_audio, frame_length=1024, hop_length=160)
        pitch_track = librosa.yin(
            segment_audio,
            fmin=80,
            fmax=420,
            sr=sample_rate,
            frame_length=1024,
            hop_length=160,
        )
    except Exception as exc:
        logger.debug("Singer analysis feature extraction failed for segment %.2f-%.2f: %s", segment.start, segment.end, exc)
        return None

    pitch_track = np.asarray(pitch_track, dtype=np.float32)
    finite_pitch = pitch_track[np.isfinite(pitch_track)]
    if finite_pitch.size:
        log_pitch = np.log2(np.maximum(finite_pitch, 1.0))
        pitch_features = [
            float(np.median(log_pitch)),
            float(np.subtract(*np.percentile(log_pitch, [75, 25]))),
            float(finite_pitch.size / max(1, pitch_track.size)),
        ]
    else:
        pitch_features = [0.0, 0.0, 0.0]

    feature_parts: list[float] = []
    feature_parts.extend(float(value) for value in np.mean(mfcc, axis=1))
    feature_parts.extend(float(value) for value in np.std(mfcc, axis=1))
    feature_parts.extend(_safe_stat_pair(centroid.ravel()))
    feature_parts.extend(_safe_stat_pair(bandwidth.ravel()))
    feature_parts.extend(_safe_stat_pair(rolloff.ravel()))
    feature_parts.extend(_safe_stat_pair(rms.ravel()))
    feature_parts.extend(_safe_stat_pair(zcr.ravel()))
    feature_parts.extend(pitch_features)
    feature_parts.append(float(segment.end - segment.start))
    feature_parts.append(float(len(segment.words)))
    return np.asarray(feature_parts, dtype=np.float32)


def _pick_cluster_count(
    features: np.ndarray,
    *,
    max_singers: int,
    title_hint: bool,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    if features.shape[0] < 3:
        return 1, None, None

    max_clusters = min(max_singers, features.shape[0])
    best_score = -math.inf
    best_choice: tuple[int, np.ndarray, np.ndarray] | None = None
    threshold = 0.15 if title_hint else 0.19

    for cluster_count in range(2, max_clusters + 1):
        labels, centers = _run_kmeans(features, cluster_count)
        unique_labels, counts = np.unique(labels, return_counts=True)
        if unique_labels.size < cluster_count:
            continue

        silhouette = _silhouette_score(features, labels)
        if silhouette <= -0.2:
            continue

        min_cluster_size = int(np.min(counts))
        if centers.shape[0] > 1:
            centroid_distances = _pairwise_distances(centers)
            separation = float(np.min(centroid_distances[np.triu_indices_from(centroid_distances, k=1)]))
        else:
            separation = 0.0

        score = silhouette + min(separation / 6.5, 0.28)
        if min_cluster_size == 1:
            score -= 0.12
        elif min_cluster_size == 2:
            score -= 0.04

        if score > best_score:
            best_score = score
            best_choice = (cluster_count, labels, centers)

    if best_choice is None or best_score < threshold:
        return 1, None, None
    return best_choice


def _fill_missing_labels(labels: list[int], segments: list[TranscriptSegment]) -> list[int]:
    if not labels:
        return labels
    filled = list(labels)
    first_known = next((label for label in filled if label >= 0), 0)
    for index, label in enumerate(filled):
        if label >= 0:
            continue

        prev_index = next((j for j in range(index - 1, -1, -1) if filled[j] >= 0), None)
        next_index = next((j for j in range(index + 1, len(filled)) if filled[j] >= 0), None)
        if prev_index is None and next_index is None:
            filled[index] = first_known
        elif prev_index is None:
            filled[index] = filled[next_index]
        elif next_index is None:
            filled[index] = filled[prev_index]
        else:
            prev_distance = abs(segments[index].start - segments[prev_index].end)
            next_distance = abs(segments[next_index].start - segments[index].end)
            filled[index] = filled[prev_index] if prev_distance <= next_distance else filled[next_index]
    return filled


def _smooth_labels(labels: list[int], segments: list[TranscriptSegment]) -> list[int]:
    if len(labels) < 3:
        return labels
    smoothed = list(labels)
    for index in range(1, len(labels) - 1):
        left = smoothed[index - 1]
        center = smoothed[index]
        right = smoothed[index + 1]
        short_segment = (segments[index].end - segments[index].start) <= 1.2
        if short_segment and left == right and center != left:
            smoothed[index] = left
    return smoothed


def _build_profiles(cluster_labels: list[int]) -> tuple[dict[int, SingerProfile], dict[int, int]]:
    ordered_clusters = sorted(
        set(cluster_labels),
        key=lambda cluster_index: min(
            (segment_index for segment_index, label in enumerate(cluster_labels) if label == cluster_index),
            default=10_000,
        ),
    )
    cluster_to_order = {cluster_index: order for order, cluster_index in enumerate(ordered_clusters)}
    profiles = {
        cluster_index: SingerProfile(
            singer_id=f"singer_{order + 1}",
            label=f"Singer {order + 1}",
            lane_index=order,
            **PALETTE[min(order, len(PALETTE) - 1)],
        )
        for cluster_index, order in cluster_to_order.items()
    }
    return profiles, cluster_to_order


class LibrosaSingerAnalyzer:
    name = "librosa_singer_clusters"

    def analyze(
        self,
        audio_path: str,
        segments: list[TranscriptSegment],
        *,
        title: str = "",
        max_singers: int = MAX_SINGERS,
    ) -> SingerAnalysisResult:
        if not segments:
            return SingerAnalysisResult(provider=self.name)

        librosa = _lazy_import_librosa()
        audio, sample_rate = librosa.load(audio_path, sr=ANALYSIS_SAMPLE_RATE, mono=True)
        if audio.size == 0:
            return SingerAnalysisResult(provider=self.name)

        feature_rows: list[np.ndarray] = []
        valid_segment_indices: list[int] = []
        analyzed_seconds = 0.0

        for segment_index, segment in enumerate(segments):
            features = _extract_segment_features(audio, sample_rate, segment)
            if features is None:
                continue
            feature_rows.append(features)
            valid_segment_indices.append(segment_index)
            analyzed_seconds += max(0.0, segment.end - segment.start)

        if not feature_rows:
            profile = SingerProfile(
                singer_id="singer_1",
                label="Singer 1",
                lane_index=0,
                **PALETTE[0],
            )
            assignments = [
                SingerSegmentAssignment(segment_index=index, singer_id=profile.singer_id, label=profile.label, confidence=0.0)
                for index in range(len(segments))
            ]
            return SingerAnalysisResult(
                detected_singer_count=1,
                provider=self.name,
                profiles=[profile],
                assignments=assignments,
                low_confidence_segments=len(assignments),
            )

        raw_features = np.vstack(feature_rows)
        normalized_features = _normalize_features(raw_features)
        cluster_count, valid_labels, centers = _pick_cluster_count(
            normalized_features,
            max_singers=max_singers,
            title_hint=_title_suggests_multiple_singers(title),
        )

        if cluster_count <= 1 or valid_labels is None or centers is None:
            profile = SingerProfile(
                singer_id="singer_1",
                label="Singer 1",
                lane_index=0,
                **PALETTE[0],
            )
            assignments = [
                SingerSegmentAssignment(
                    segment_index=index,
                    singer_id=profile.singer_id,
                    label=profile.label,
                    confidence=1.0 if index in valid_segment_indices else 0.2,
                )
                for index in range(len(segments))
            ]
            return SingerAnalysisResult(
                detected_singer_count=1,
                provider=self.name,
                profiles=[profile],
                assignments=assignments,
                low_confidence_segments=sum(1 for item in assignments if item.confidence < 0.5),
                analysis_window_seconds=round(analyzed_seconds, 3),
            )

        label_by_segment = [-1] * len(segments)
        confidence_by_segment = [0.0] * len(segments)
        distances = np.sqrt(np.maximum(0.0, np.sum((normalized_features[:, None, :] - centers[None, :, :]) ** 2, axis=2)))
        for row_index, segment_index in enumerate(valid_segment_indices):
            label = int(valid_labels[row_index])
            label_by_segment[segment_index] = label
            ordered = np.sort(distances[row_index])
            if ordered.size >= 2 and ordered[1] > 1e-6:
                confidence = 1.0 - float(ordered[0] / ordered[1])
            else:
                confidence = 0.7
            confidence_by_segment[segment_index] = min(0.99, max(0.05, confidence))

        label_by_segment = _fill_missing_labels(label_by_segment, segments)
        label_by_segment = _smooth_labels(label_by_segment, segments)
        profiles_by_cluster, cluster_to_order = _build_profiles(label_by_segment)

        assignments: list[SingerSegmentAssignment] = []
        low_confidence_segments = 0
        for segment_index, cluster_index in enumerate(label_by_segment):
            profile = profiles_by_cluster[cluster_index]
            confidence = confidence_by_segment[segment_index]
            if confidence <= 0.0:
                confidence = 0.32
            if confidence < 0.5:
                low_confidence_segments += 1
            assignments.append(
                SingerSegmentAssignment(
                    segment_index=segment_index,
                    singer_id=profile.singer_id,
                    label=profile.label,
                    confidence=round(float(confidence), 4),
                )
            )

        ordered_profiles = [
            profile
            for _cluster_index, profile in sorted(
                profiles_by_cluster.items(),
                key=lambda item: cluster_to_order.get(item[0], 0),
            )
        ]
        return SingerAnalysisResult(
            detected_singer_count=len(ordered_profiles),
            provider=self.name,
            profiles=ordered_profiles,
            assignments=assignments,
            low_confidence_segments=low_confidence_segments,
            analysis_window_seconds=round(analyzed_seconds, 3),
        )


class StructureDuetAnalyzer:
    name = "structure_duet_sections_v2"

    def analyze(
        self,
        audio_path: str,
        segments: list[TranscriptSegment],
        *,
        title: str = "",
        max_singers: int = MAX_SINGERS,
    ) -> SingerAnalysisResult:
        del audio_path, title, max_singers
        if not segments:
            return SingerAnalysisResult(provider=self.name)

        sections = _detect_structural_sections(segments)
        profiles, profile_map = _duet_profiles()
        assignments: list[SingerSegmentAssignment] = []
        global_half_index = 0

        for section in sections:
            for start, end, lane_index in _split_section(section, segments):
                if end <= start:
                    continue
                color_index = global_half_index % len(PALETTE)
                profile = profile_map[(lane_index, color_index)]
                confidence = 0.88 if section["type"] == "chorus" else 0.82
                if section["type"] in {"intro", "outro"}:
                    confidence = 0.72
                for segment_index in range(start, end):
                    assignments.append(
                        SingerSegmentAssignment(
                            segment_index=segment_index,
                            singer_id=profile.singer_id,
                            label=profile.label,
                            confidence=confidence,
                        )
                    )
                global_half_index += 1

        if not assignments:
            profile = profile_map[(0, 0)]
            assignments = [
                SingerSegmentAssignment(
                    segment_index=index,
                    singer_id=profile.singer_id,
                    label=profile.label,
                    confidence=0.6,
                )
                for index in range(len(segments))
            ]

        assignments.sort(key=lambda item: item.segment_index)
        lane_count = 2 if any(profile.lane_index == 1 for profile in profiles) and len(segments) > 1 else 1
        return SingerAnalysisResult(
            detected_singer_count=lane_count,
            provider=self.name,
            profiles=profiles,
            assignments=assignments,
            low_confidence_segments=sum(1 for assignment in assignments if assignment.confidence < 0.75),
            analysis_window_seconds=round(sum(max(0.0, segment.end - segment.start) for segment in segments), 3),
        )
