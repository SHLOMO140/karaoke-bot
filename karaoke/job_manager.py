"""Job lifecycle management and manifest persistence."""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_STYLE_PRESET, JOBS_DIR
from .models import (
    AlignedTranscript,
    ChordEvent,
    ErrorInfo,
    Job,
    JobManifest,
    JobStatus,
    LanguageDetectionResult,
    LyricsVerificationResult,
    ReviewStatus,
    SongAnalysis,
    SubWordTiming,
    TranscriptDraft,
    TranscriptSegment,
    VideoRequest,
    WordTiming,
)
from .styles import normalize_style_preset

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _segments_to_dict(segments: Iterable[TranscriptSegment]) -> list[dict[str, object]]:
    return [
        {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
            "words": [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "confidence": word.confidence,
                    "source": word.source,
                    "aligned": word.aligned,
                    "subwords": [
                        {
                            "text": subword.text,
                            "start": subword.start,
                            "end": subword.end,
                            "confidence": subword.confidence,
                        }
                        for subword in word.subwords
                    ],
                }
                for word in segment.words
            ],
        }
        for segment in segments
    ]


def _segments_from_dict(items: list[dict[str, object]]) -> list[TranscriptSegment]:
    segments = []
    for item in items:
        words = [
            WordTiming(
                word=str(word["word"]),
                start=float(word["start"]),
                end=float(word["end"]),
                confidence=float(word.get("confidence", 0.0)),
                source=str(word.get("source", "draft_whisper")),
                aligned=bool(word.get("aligned", False)),
                subwords=[
                    SubWordTiming(
                        text=str(subword.get("text", "")),
                        start=float(subword.get("start", 0.0)),
                        end=float(subword.get("end", 0.0)),
                        confidence=float(subword.get("confidence", 0.0)),
                    )
                    for subword in word.get("subwords", [])
                ],
            )
            for word in item.get("words", [])
        ]
        segments.append(
            TranscriptSegment(
                words=words,
                text=str(item.get("text", "")),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
            )
        )
    return segments


def _write_json(path: Path, data: dict[str, object]):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _song_analysis_to_dict(analysis: SongAnalysis) -> dict[str, object]:
    return {
        "bpm": analysis.bpm,
        "time_signature": analysis.time_signature,
        "preview_window_seconds": analysis.preview_window_seconds,
        "provider": analysis.provider,
        "source_audio": analysis.source_audio,
        "beat_times": analysis.beat_times,
        "measure_times": analysis.measure_times,
        "chord_events": [
            {
                "label": event.label,
                "start": event.start,
                "end": event.end,
                "confidence": event.confidence,
                "root": event.root,
                "quality": event.quality,
            }
            for event in analysis.chord_events
        ],
    }


def _song_analysis_from_dict(data: dict[str, object]) -> SongAnalysis:
    return SongAnalysis(
        bpm=float(data.get("bpm", 0.0) or 0.0),
        time_signature=int(data.get("time_signature", 4) or 4),
        preview_window_seconds=float(data.get("preview_window_seconds", 0.6) or 0.6),
        provider=str(data.get("provider", "")),
        source_audio=str(data.get("source_audio", "")),
        beat_times=[float(item) for item in data.get("beat_times", [])],
        measure_times=[float(item) for item in data.get("measure_times", [])],
        chord_events=[
            ChordEvent(
                label=str(item.get("label", "")),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                confidence=float(item.get("confidence", 0.0)),
                root=str(item.get("root", "")),
                quality=str(item.get("quality", "")),
            )
            for item in data.get("chord_events", [])
            if isinstance(item, dict)
        ],
    )


def _sessions_path() -> Path:
    return JOBS_DIR / "_sessions.json"


def _load_sessions() -> dict[str, str]:
    path = _sessions_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_sessions(data: dict[str, str]):
    _sessions_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _session_key(chat_id: int, user_id: int) -> str:
    return f"{chat_id}:{user_id}"


def _iter_job_dirs() -> Iterable[Path]:
    if not JOBS_DIR.exists():
        return []
    return (
        path
        for path in JOBS_DIR.iterdir()
        if path.is_dir() and path.name != "__pycache__" and (path / "job.json").exists()
    )


def _refresh_artifacts(job: Job):
    job.manifest.artifacts = {
        "manifest": job.manifest_path.name,
        "original_audio": job.original_audio_path.name if job.original_audio_path.exists() else "",
        "original_video": job.original_video_path.name if job.original_video_path.exists() else "",
        "vocals": job.vocals_path.name if job.vocals_path.exists() else "",
        "instrumental": job.instrumental_path.name if job.instrumental_path.exists() else "",
        "draft_transcript": job.draft_transcript_path.name if job.draft_transcript_path.exists() else "",
        "draft_timings": job.draft_timings_path.name if job.draft_timings_path.exists() else "",
        "review_transcript": job.review_transcript_path.name if job.review_transcript_path.exists() else "",
        "review_timings": job.review_timings_path.name if job.review_timings_path.exists() else "",
        "transcript": job.transcript_path.name if job.transcript_path.exists() else "",
        "timings": job.timings_path.name if job.timings_path.exists() else "",
        "srt": job.srt_path.name if job.srt_path.exists() else "",
        "ass": job.ass_path.name if job.ass_path.exists() else "",
        "song_analysis": job.song_analysis_path.name if job.song_analysis_path.exists() else "",
        "lyrics_with_chords": job.lyrics_with_chords_path.name if job.lyrics_with_chords_path.exists() else "",
        "thumbnail": job.thumbnail_path.name if job.thumbnail_path.exists() else "",
        "video_with_vocals": job.video_vocals_path.name if job.video_vocals_path.exists() else "",
        "video_without_vocals": job.video_instrumental_path.name if job.video_instrumental_path.exists() else "",
    }


def save_job(job: Job):
    job.manifest.updated_at = _now_iso()
    _refresh_artifacts(job)
    _write_json(job.manifest_path, asdict(job.manifest))


def create_job(
    title: str = "",
    source_url: str = "",
    input_type: str = "",
    has_video: bool = False,
    thumbnail_url: str = "",
    chat_id: int = 0,
    user_id: int = 0,
    style_preset: str = DEFAULT_STYLE_PRESET,
) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    now = _now_iso()

    manifest = JobManifest(
        job_id=job_id,
        title=title,
        source_url=source_url,
        input_type=input_type,
        has_video=has_video,
        thumbnail_url=thumbnail_url,
        chat_id=chat_id,
        user_id=user_id,
        style_preset=style_preset,
        created_at=now,
        updated_at=now,
    )
    job = Job(job_id=job_id, job_dir=job_dir, manifest=manifest)
    save_job(job)
    logger.info("Created job %s for %s", job_id, title)
    return job


def load_job(job_id: str) -> Job:
    job_dir = JOBS_DIR / job_id
    manifest_path = job_dir / "job.json"
    data = _read_json(manifest_path)
    manifest = JobManifest(**data)
    job = Job(job_id=job_id, job_dir=job_dir, manifest=manifest)
    normalized_style = normalize_style_preset(job.manifest.style_preset)
    if normalized_style != job.manifest.style_preset:
        job.manifest.style_preset = normalized_style
        save_job(job)
    return job


def find_latest_reusable_job(
    *,
    source_url: str = "",
    input_type: str = "",
    user_id: int = 0,
) -> Job | None:
    normalized_source = (source_url or "").strip()
    normalized_input = (input_type or "").strip()
    candidates: list[tuple[str, Job]] = []

    for job_dir in _iter_job_dirs():
        try:
            job = load_job(job_dir.name)
        except Exception:
            continue

        if user_id and job.manifest.user_id not in {0, user_id}:
            continue
        if normalized_input and job.input_type != normalized_input:
            continue
        if normalized_source and job.source_url.strip() != normalized_source:
            continue
        if not has_reusable_artifacts(job):
            continue
        candidates.append((job.manifest.updated_at or "", job))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def has_reusable_artifacts(job: Job) -> bool:
    return any(
        path.exists()
        for path in [
            job.original_audio_path,
            job.instrumental_path,
            job.timings_path,
            job.review_timings_path,
            job.draft_timings_path,
            job.ass_path,
            job.srt_path,
            job.video_vocals_path,
            job.video_instrumental_path,
        ]
    )


def can_rerender(job: Job) -> bool:
    return any(
        path.exists()
        for path in [
            job.timings_path,
            job.review_timings_path,
            job.draft_timings_path,
        ]
    )


def get_best_available_segments(job: Job) -> list[TranscriptSegment]:
    if job.timings_path.exists():
        return load_final_segments(job)
    if job.review_timings_path.exists():
        return load_review_segments(job)
    if job.draft_timings_path.exists():
        return load_draft_segments(job)
    return []


def update_status(job: Job, status: JobStatus, error: ErrorInfo | None = None):
    job.status = status
    if error:
        job.manifest.errors.append(asdict(error))
    save_job(job)
    logger.info("Job %s -> %s", job.job_id, status.value)


def update_review_status(job: Job, review_status: ReviewStatus):
    job.review_status = review_status
    save_job(job)


def record_provider(job: Job, provider_key: str, provider_name: str):
    job.manifest.providers[provider_key] = provider_name
    save_job(job)


def record_requested_outputs(job: Job, video_request: VideoRequest | None):
    if video_request is None:
        job.manifest.requested_outputs = {"subtitles_only": True}
    else:
        job.manifest.requested_outputs = {
            "subtitles_only": False,
            "with_vocals": video_request.with_vocals,
            "without_vocals": video_request.without_vocals,
            "quality": video_request.quality,
        }
    save_job(job)


def add_warning(job: Job, warning_message: str):
    if warning_message and warning_message not in job.manifest.warnings:
        job.manifest.warnings.append(warning_message)
        save_job(job)


def save_language_info(job: Job, language_info: LanguageDetectionResult):
    job.manifest.language_info = asdict(language_info)
    save_job(job)


def save_lyrics_verification(job: Job, verification: LyricsVerificationResult):
    job.manifest.lyrics_verification = asdict(verification)
    save_job(job)


def get_lyrics_options(job: Job) -> list[dict[str, object]]:
    verification = job.manifest.lyrics_verification or {}
    options = verification.get("options") or []
    return [option for option in options if option.get("option_id") and option.get("lines")]


def get_selected_lyrics_option_id(job: Job) -> str:
    verification = job.manifest.lyrics_verification or {}
    return str(verification.get("selected_option_id") or "draft")


def set_selected_lyrics_option(job: Job, option_id: str):
    verification = dict(job.manifest.lyrics_verification or {})
    verification["selected_option_id"] = option_id
    job.manifest.lyrics_verification = verification
    save_job(job)


def apply_lyrics_option(job: Job, option_id: str) -> dict[str, object]:
    option = next((item for item in get_lyrics_options(job) if item.get("option_id") == option_id), None)
    if option is None:
        raise ValueError(f"Unknown lyrics option: {option_id}")

    draft_segments = load_draft_segments(job)
    text = "\n".join(str(line).strip() for line in option.get("lines", []) if str(line).strip())
    if not text:
        raise ValueError(f"Lyrics option {option_id} does not contain text")

    updated_segments = update_transcript_text(draft_segments, text)
    save_review_transcript(job, updated_segments)
    set_selected_lyrics_option(job, option_id)
    return option


def save_manual_review_option(job: Job, segments: list[TranscriptSegment], label: str = "\u05ea\u05d9\u05e7\u05d5\u05df \u05d9\u05d3\u05e0\u05d9"):
    verification = dict(job.manifest.lyrics_verification or {})
    options = [dict(option) for option in verification.get("options") or [] if isinstance(option, dict)]
    lines = [segment.text.strip() for segment in segments if segment.text.strip()]
    manual_option = {
        "option_id": "manual",
        "label": label,
        "lines": lines,
        "source_url": "",
        "confidence": 1.0,
        "source_count": 0,
    }

    replaced = False
    for index, option in enumerate(options):
        if option.get("option_id") == "manual":
            options[index] = manual_option
            replaced = True
            break
    if not replaced:
        options.insert(0, manual_option)

    verification["options"] = options
    verification["selected_option_id"] = "manual"
    job.manifest.lyrics_verification = verification
    save_job(job)


def _save_segments(job: Job, txt_path: Path, json_path: Path, segments: list[TranscriptSegment], payload: dict[str, object]):
    txt_path.write_text("\n".join(segment.text for segment in segments), encoding="utf-8")
    _write_json(json_path, payload)


def save_draft_transcript(job: Job, draft: TranscriptDraft):
    payload = {
        "kind": "draft",
        "provider": draft.provider,
        "language_info": asdict(draft.language_info) if draft.language_info else {},
        "segments": _segments_to_dict(draft.segments),
    }
    _save_segments(job, job.draft_transcript_path, job.draft_timings_path, draft.segments, payload)
    if not job.review_transcript_path.exists():
        save_review_transcript(job, draft.segments)


def save_review_transcript(job: Job, segments: list[TranscriptSegment]):
    payload = {
        "kind": "review",
        "segments": _segments_to_dict(segments),
    }
    _save_segments(job, job.review_transcript_path, job.review_timings_path, segments, payload)
    update_review_status(job, ReviewStatus.AWAITING_REVIEW)


def save_final_transcript(job: Job, aligned: AlignedTranscript):
    payload = {
        "kind": "aligned",
        "provider": aligned.provider,
        "fully_aligned": aligned.fully_aligned,
        "unaligned_word_count": aligned.unaligned_word_count,
        "segments": _segments_to_dict(aligned.segments),
    }
    _save_segments(job, job.transcript_path, job.timings_path, aligned.segments, payload)


def save_song_analysis(job: Job, analysis: SongAnalysis):
    _write_json(job.song_analysis_path, _song_analysis_to_dict(analysis))
    save_job(job)


def load_song_analysis(job: Job) -> SongAnalysis:
    return _song_analysis_from_dict(_read_json(job.song_analysis_path))


def save_chord_sheet(job: Job, content: str):
    job.lyrics_with_chords_path.write_text(content, encoding="utf-8")
    save_job(job)


def load_draft_segments(job: Job) -> list[TranscriptSegment]:
    data = _read_json(job.draft_timings_path)
    return _segments_from_dict(data.get("segments", []))


def load_review_segments(job: Job) -> list[TranscriptSegment]:
    if not job.review_timings_path.exists():
        return load_draft_segments(job)
    data = _read_json(job.review_timings_path)
    return _segments_from_dict(data.get("segments", []))


def load_final_segments(job: Job) -> list[TranscriptSegment]:
    data = _read_json(job.timings_path)
    return _segments_from_dict(data.get("segments", []))


def get_review_text(job: Job) -> str:
    if job.review_transcript_path.exists():
        return job.review_transcript_path.read_text(encoding="utf-8")
    if job.draft_transcript_path.exists():
        return job.draft_transcript_path.read_text(encoding="utf-8")
    return ""


def set_active_review_job(chat_id: int, user_id: int, job_id: str):
    sessions = _load_sessions()
    sessions[_session_key(chat_id, user_id)] = job_id
    _save_sessions(sessions)


def set_review_message_id(job: Job, message_id: int):
    job.review_message_id = message_id
    save_job(job)


def clear_active_review_job(chat_id: int, user_id: int):
    sessions = _load_sessions()
    sessions.pop(_session_key(chat_id, user_id), None)
    _save_sessions(sessions)


def get_active_review_job(chat_id: int, user_id: int) -> Job | None:
    sessions = _load_sessions()
    job_id = sessions.get(_session_key(chat_id, user_id))
    if not job_id:
        return None
    try:
        job = load_job(job_id)
    except FileNotFoundError:
        sessions.pop(_session_key(chat_id, user_id), None)
        _save_sessions(sessions)
        return None
    if job.status != JobStatus.AWAITING_REVIEW or job.review_status not in {
        ReviewStatus.AWAITING_REVIEW,
        ReviewStatus.APPROVED,
        ReviewStatus.DRAFT_READY,
    }:
        return None
    return job


def _line_numbers(segments: list[TranscriptSegment]) -> list[str]:
    return [f"{index}: {segment.text}" for index, segment in enumerate(segments, 1)]


def get_display_text(segments: list[TranscriptSegment]) -> str:
    lines = _line_numbers(segments)
    if not lines:
        return "אין טקסט להצגה."
    return "\n".join(lines)


def get_display_page(segments: list[TranscriptSegment], page: int, page_size: int) -> tuple[str, int]:
    lines = _line_numbers(segments)
    if not lines:
        return "אין טקסט להצגה.", 1
    total_pages = max(1, (len(lines) + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    start = page * page_size
    end = start + page_size
    return "\n".join(lines[start:end]), total_pages


def _build_words_for_span(words_text: list[str], span_start: float, span_end: float, source: str = "review_hint") -> list[WordTiming]:
    if not words_text:
        return []
    duration = max(span_end - span_start, 0.01)
    result = []
    for index, word_text in enumerate(words_text):
        word_start = span_start + (index / len(words_text)) * duration
        word_end = span_start + ((index + 1) / len(words_text)) * duration
        result.append(
            WordTiming(
                word=word_text,
                start=word_start,
                end=word_end,
                confidence=0.0,
                source=source,
                aligned=False,
            )
        )
    return result


def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w\u0590-\u05FF]+", "", word.lower())


def _word_weight(word: str) -> float:
    normalized = _normalize_word(word)
    return float(max(1, len(normalized or word.strip() or "x")))


def _build_weighted_words_for_span(
    words_text: list[str],
    span_start: float,
    span_end: float,
    source: str = "review_hint",
    confidence: float = 0.0,
) -> list[WordTiming]:
    if not words_text:
        return []

    duration = max(span_end - span_start, 0.01)
    weights = [_word_weight(word_text) for word_text in words_text]
    total_weight = max(sum(weights), 1.0)
    cursor = span_start
    built: list[WordTiming] = []

    for index, (word_text, weight) in enumerate(zip(words_text, weights)):
        portion = duration * (weight / total_weight)
        next_cursor = span_end if index == len(words_text) - 1 else min(span_end, cursor + portion)
        built.append(
            WordTiming(
                word=word_text,
                start=round(cursor, 6),
                end=round(max(next_cursor, cursor + 0.01), 6),
                confidence=confidence,
                source=source,
                aligned=False,
            )
        )
        cursor = built[-1].end

    built[0].start = round(span_start, 6)
    built[-1].end = round(span_end, 6)
    return built


def _normalize_line_text(text: str) -> str:
    return " ".join(token for token in (_normalize_word(word) for word in text.split()) if token)


def _flatten_segment_words(segments: list[TranscriptSegment]) -> list[WordTiming]:
    return [word for segment in segments for word in segment.words]


def _span_for_line_block(
    segments: list[TranscriptSegment],
    start_index: int,
    end_index: int,
    fallback_start: float,
    fallback_end: float,
) -> tuple[float, float]:
    reference_segments = segments[start_index:end_index]
    if reference_segments:
        return reference_segments[0].start, reference_segments[-1].end

    previous_segment = segments[start_index - 1] if start_index > 0 else None
    next_segment = segments[start_index] if start_index < len(segments) else None
    span_start = previous_segment.end if previous_segment is not None else fallback_start
    span_end = next_segment.start if next_segment is not None else fallback_end
    if span_end <= span_start:
        anchor = next_segment or previous_segment
        if anchor is not None:
            return anchor.start, anchor.end
    return span_start, max(span_end, span_start + 0.01)


def _build_segment_from_line_text(
    line_text: str,
    span_start: float,
    span_end: float,
    reference_segments: list[TranscriptSegment],
) -> TranscriptSegment:
    reference_words = _flatten_segment_words(reference_segments)
    words_text = line_text.split()
    if reference_words:
        words = _align_words_to_draft(words_text, reference_words, span_start, span_end)
    else:
        words = _build_weighted_words_for_span(words_text, span_start, span_end)
    return TranscriptSegment(
        words=words,
        text=line_text.strip(),
        start=span_start,
        end=span_end,
    )


def _align_words_to_draft(
    new_words_text: list[str],
    orig_words: list[WordTiming],
    segment_start: float,
    segment_end: float,
) -> list[WordTiming]:
    """Assign timings to corrected text while preserving the original error span."""
    if not new_words_text:
        return []
    if not orig_words:
        return _build_weighted_words_for_span(new_words_text, segment_start, segment_end)

    orig_norm = [_normalize_word(w.word) for w in orig_words]
    new_norm = [_normalize_word(w) for w in new_words_text]

    result: list[WordTiming | None] = [None] * len(new_words_text)

    for tag, a0, a1, d0, d1 in SequenceMatcher(None, new_norm, orig_norm, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(a1 - a0):
                ow = orig_words[d0 + k]
                result[a0 + k] = WordTiming(
                    word=new_words_text[a0 + k],
                    start=ow.start,
                    end=ow.end,
                    confidence=ow.confidence,
                    source=ow.source,
                    aligned=ow.aligned,
                    subwords=ow.subwords,
                )
            continue

        if tag not in {"replace", "insert"}:
            continue

        n_new = a1 - a0
        n_orig = d1 - d0
        if n_new <= 0:
            continue

        if n_orig > 0:
            span_start = orig_words[d0].start
            span_end = orig_words[d1 - 1].end
            best_ratio = max(
                (
                    SequenceMatcher(None, new_norm[ai], orig_norm[di], autojunk=False).ratio()
                    for ai in range(a0, a1)
                    for di in range(d0, d1)
                ),
                default=0.0,
            )
        else:
            prev_word = orig_words[d0 - 1] if d0 > 0 else None
            next_word = orig_words[d0] if d0 < len(orig_words) else None
            span_start = prev_word.end if prev_word is not None else segment_start
            span_end = next_word.start if next_word is not None else segment_end
            if span_end <= span_start:
                anchor = next_word or prev_word or orig_words[min(d0, len(orig_words) - 1)]
                span_start = anchor.start
                span_end = anchor.end
            best_ratio = max(
                (
                    SequenceMatcher(None, new_norm[ai], orig_norm[di], autojunk=False).ratio()
                    for ai in range(a0, a1)
                    for di in range(max(0, d0 - 1), min(len(orig_words), d0 + 1))
                ),
                default=0.0,
            )

        span_words = _build_weighted_words_for_span(
            new_words_text[a0:a1],
            span_start,
            span_end,
            source="review_hint",
            confidence=best_ratio * 0.4,
        )
        for offset, word in enumerate(span_words):
            result[a0 + offset] = word

    for i in range(len(result)):
        if result[i] is not None:
            continue
        prev_idx = next((j for j in range(i - 1, -1, -1) if result[j] is not None), -1)
        next_idx = next((j for j in range(i + 1, len(result)) if result[j] is not None), len(result))
        gap_start = result[prev_idx].end if prev_idx >= 0 else segment_start
        gap_end = result[next_idx].start if next_idx < len(result) else segment_end
        none_slots = [j for j in range(prev_idx + 1, next_idx)]
        pos = none_slots.index(i)
        n = len(none_slots)
        gap_dur = max(gap_end - gap_start, 0.01)
        result[i] = WordTiming(
            word=new_words_text[i],
            start=round(gap_start + pos / n * gap_dur, 6),
            end=round(gap_start + (pos + 1) / n * gap_dur, 6),
            confidence=0.0,
            source="review_hint",
            aligned=False,
        )

    final = [w for w in result if w is not None]
    for i in range(1, len(final)):
        if final[i].start < final[i - 1].end:
            final[i] = WordTiming(
                word=final[i].word,
                start=final[i - 1].end,
                end=max(final[i].end, final[i - 1].end + 0.01),
                confidence=final[i].confidence,
                source=final[i].source,
                aligned=final[i].aligned,
            )

    return final


def update_transcript_line(segments: list[TranscriptSegment], line_number: int, corrected_line: str) -> list[TranscriptSegment]:
    index = line_number - 1
    if index < 0 or index >= len(segments):
        raise ValueError(f"מספר שורה לא תקין: {line_number}")
    updated = list(segments)
    segment = segments[index]
    words = corrected_line.strip().split()
    updated[index] = TranscriptSegment(
        words=_align_words_to_draft(words, segment.words, segment.start, segment.end),
        text=corrected_line.strip(),
        start=segment.start,
        end=segment.end,
    )
    return updated


def update_transcript_text(segments: list[TranscriptSegment], corrected_text: str) -> list[TranscriptSegment]:
    new_lines = [line.strip() for line in corrected_text.strip().splitlines() if line.strip()]
    if not new_lines:
        return segments
    total_start = segments[0].start if segments else 0.0
    total_end = segments[-1].end if segments else 0.0
    if not segments:
        total_duration = 0.01
        return [
            TranscriptSegment(
                words=_build_weighted_words_for_span(
                    line.split(),
                    total_start + (index / len(new_lines)) * total_duration,
                    total_start + ((index + 1) / len(new_lines)) * total_duration,
                ),
                text=line,
                start=total_start + (index / len(new_lines)) * total_duration,
                end=total_start + ((index + 1) / len(new_lines)) * total_duration,
            )
            for index, line in enumerate(new_lines)
        ]

    original_lines = [segment.text.strip() for segment in segments]
    original_norm = [_normalize_line_text(line) for line in original_lines]
    new_norm = [_normalize_line_text(line) for line in new_lines]

    updated_segments: list[TranscriptSegment | None] = [None] * len(new_lines)
    matcher = SequenceMatcher(None, new_norm, original_norm, autojunk=False)

    for tag, a0, a1, d0, d1 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(a1 - a0):
                reference_segment = segments[d0 + offset]
                updated_segments[a0 + offset] = _build_segment_from_line_text(
                    new_lines[a0 + offset],
                    reference_segment.start,
                    reference_segment.end,
                    [reference_segment],
                )
            continue

        if tag not in {"replace", "insert"} or a1 <= a0:
            continue

        span_start, span_end = _span_for_line_block(segments, d0, d1, total_start, total_end)
        block_reference_segments = segments[d0:d1]
        weights = [max(1, len(new_norm[index])) for index in range(a0, a1)]
        total_weight = max(sum(weights), 1)
        cursor = span_start

        for relative_index, line_index in enumerate(range(a0, a1)):
            portion = (span_end - span_start) * (weights[relative_index] / total_weight)
            next_cursor = span_end if line_index == a1 - 1 else min(span_end, cursor + portion)
            updated_segments[line_index] = _build_segment_from_line_text(
                new_lines[line_index],
                cursor,
                next_cursor,
                block_reference_segments,
            )
            cursor = next_cursor

    completed = [segment for segment in updated_segments if segment is not None]
    if completed:
        return completed

    total_duration = max(total_end - total_start, 0.01)
    new_segments = []
    for index, line in enumerate(new_lines):
        segment_start = total_start + (index / len(new_lines)) * total_duration
        segment_end = total_start + ((index + 1) / len(new_lines)) * total_duration
        new_segments.append(
            TranscriptSegment(
                words=_build_weighted_words_for_span(line.split(), segment_start, segment_end),
                text=line,
                start=segment_start,
                end=segment_end,
            )
        )
    return new_segments


def get_output_files(job: Job, video_request: VideoRequest | None = None) -> dict[str, Path]:
    files = {}
    include_vocals_video = bool(video_request and video_request.with_vocals)
    include_instrumental_video = bool(video_request and video_request.without_vocals)

    for name, path in [
        ("transcript.txt", job.transcript_path),
        ("timings.json", job.timings_path),
        ("subtitles.srt", job.srt_path),
        ("karaoke.ass", job.ass_path),
        ("song_analysis.json", job.song_analysis_path),
        ("lyrics_with_chords.txt", job.lyrics_with_chords_path),
    ]:
        if path.exists():
            files[name] = path

    if include_vocals_video and job.video_vocals_path.exists():
        files["final_video.mp4"] = job.video_vocals_path
    if include_instrumental_video and job.video_instrumental_path.exists():
        files["final_video_instrumental.mp4"] = job.video_instrumental_path
    return files


def cleanup_job(job: Job):
    try:
        shutil.rmtree(str(job.job_dir), ignore_errors=True)
        logger.info("Cleaned up job %s", job.job_id)
    except Exception as exc:
        logger.error("Cleanup failed for %s: %s", job.job_id, exc)
