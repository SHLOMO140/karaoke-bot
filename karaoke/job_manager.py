"""Job lifecycle management and manifest persistence."""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from threading import RLock
from typing import Iterable

from .config import (
    AUTO_DELETE_JOB_AFTER_DELIVERY,
    COMPLETED_JOB_RETENTION_HOURS,
    DEFAULT_STYLE_PRESET,
    JOBS_DIR,
    STALE_JOB_RETENTION_HOURS,
)
from .models import (
    AlignedTranscript,
    CharacterTiming,
    ChordEvent,
    ErrorInfo,
    Job,
    JobManifest,
    JobStatus,
    LanguageDetectionResult,
    LyricsVerificationResult,
    ReviewStatus,
    SingerAnalysisResult,
    SingerProfile,
    SingerSegmentAssignment,
    SongAnalysis,
    SubWordTiming,
    TranscriptDraft,
    TranscriptSegment,
    VideoRequest,
    WordTiming,
)
from .styles import normalize_style_preset

logger = logging.getLogger(__name__)
_STATE_LOCK = RLock()
_PROCESSING_STATUSES = {
    JobStatus.CREATED,
    JobStatus.DOWNLOADING,
    JobStatus.EXTRACTING_AUDIO,
    JobStatus.SEPARATING_VOCALS,
    JobStatus.DETECTING_LANGUAGE,
    JobStatus.TRANSCRIBING,
    JobStatus.VERIFYING_LYRICS,
    JobStatus.ALIGNING,
    JobStatus.GENERATING_SUBS,
    JobStatus.RENDERING_VIDEO,
    JobStatus.DELIVERING,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _job_last_activity(job: Job) -> datetime:
    for value in (job.manifest.updated_at, job.manifest.created_at):
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
    if job.job_dir.exists():
        return datetime.fromtimestamp(job.job_dir.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _remove_job_from_sessions(job_id: str):
    with _STATE_LOCK:
        sessions = _load_sessions()
        filtered = {key: value for key, value in sessions.items() if value != job_id}
        if len(filtered) != len(sessions):
            _save_sessions(filtered)


def _prune_expired_group_requests(
    requests: dict[str, dict[str, object]],
    *,
    now: datetime | None = None,
    stale_after_hours: int = 24,
) -> dict[str, dict[str, object]]:
    current_time = now or datetime.now(timezone.utc)
    cutoff = current_time - timedelta(hours=stale_after_hours)
    filtered: dict[str, dict[str, object]] = {}
    for token, payload in requests.items():
        created_at = _parse_timestamp(str(payload.get("created_at", "")))
        if created_at is None or created_at >= cutoff:
            filtered[token] = payload
    return filtered


def should_cleanup_delivered_job() -> bool:
    return AUTO_DELETE_JOB_AFTER_DELIVERY


def is_cleanup_candidate(
    job: Job,
    *,
    now: datetime | None = None,
    completed_after_hours: int = COMPLETED_JOB_RETENTION_HOURS,
    stale_after_hours: int = STALE_JOB_RETENTION_HOURS,
) -> str | None:
    current_time = now or datetime.now(timezone.utc)
    last_activity = _job_last_activity(job)

    if job.status == JobStatus.DONE and job.review_status == ReviewStatus.COMPLETED:
        cutoff = current_time - timedelta(hours=completed_after_hours)
        if last_activity <= cutoff:
            return "completed"

    if job.status == JobStatus.ERROR:
        cutoff = current_time - timedelta(hours=stale_after_hours)
        if last_activity <= cutoff:
            return "error"

    if job.status == JobStatus.DONE and job.review_status == ReviewStatus.APPROVED:
        cutoff = current_time - timedelta(hours=stale_after_hours)
        if last_activity <= cutoff:
            return "approved_undelivered"

    if job.status == JobStatus.AWAITING_REVIEW:
        cutoff = current_time - timedelta(hours=stale_after_hours)
        if last_activity <= cutoff:
            return "abandoned_review"

    if job.status in _PROCESSING_STATUSES:
        cutoff = current_time - timedelta(hours=stale_after_hours)
        if last_activity <= cutoff:
            return "stale"

    # Safety net: reclaim any job no rule above matched after a week,
    # so no status combination can accumulate forever.
    cutoff = current_time - timedelta(hours=max(stale_after_hours, 168))
    if last_activity <= cutoff:
        return "expired"

    return None


def find_cleanup_candidates(
    *,
    now: datetime | None = None,
    completed_after_hours: int = COMPLETED_JOB_RETENTION_HOURS,
    stale_after_hours: int = STALE_JOB_RETENTION_HOURS,
) -> list[tuple[Job, str]]:
    candidates: list[tuple[Job, str]] = []
    for job_dir in _iter_job_dirs():
        try:
            job = load_job(job_dir.name)
        except Exception:
            continue
        reason = is_cleanup_candidate(
            job,
            now=now,
            completed_after_hours=completed_after_hours,
            stale_after_hours=stale_after_hours,
        )
        if reason is not None:
            candidates.append((job, reason))
    return candidates


def cleanup_stale_jobs(
    *,
    now: datetime | None = None,
    completed_after_hours: int = COMPLETED_JOB_RETENTION_HOURS,
    stale_after_hours: int = STALE_JOB_RETENTION_HOURS,
) -> list[dict[str, str]]:
    removed: list[dict[str, str]] = []
    for job, reason in find_cleanup_candidates(
        now=now,
        completed_after_hours=completed_after_hours,
        stale_after_hours=stale_after_hours,
    ):
        cleanup_job(job)
        removed.append(
            {
                "job_id": job.job_id,
                "title": job.display_name,
                "status": job.status.value,
                "review_status": job.review_status.value,
                "reason": reason,
            }
        )
    return removed


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
                    "char_timings": [
                        {
                            "char": char_timing.char,
                            "start": char_timing.start,
                            "end": char_timing.end,
                        }
                        for char_timing in word.char_timings
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
                char_timings=[
                    CharacterTiming(
                        char=str(char_timing.get("char", "")),
                        start=float(char_timing.get("start", 0.0)),
                        end=float(char_timing.get("end", 0.0)),
                    )
                    for char_timing in word.get("char_timings", [])
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
        "original_key": analysis.original_key,
        "target_key": analysis.target_key,
        "transpose_semitones": analysis.transpose_semitones,
        "chord_sheet_text": analysis.chord_sheet_text,
        "chord_source_name": analysis.chord_source_name,
        "chord_source_url": analysis.chord_source_url,
        "original_chord_events": [
            {
                "label": event.label,
                "start": event.start,
                "end": event.end,
                "confidence": event.confidence,
                "root": event.root,
                "quality": event.quality,
            }
            for event in analysis.original_chord_events
        ],
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
        original_key=str(data.get("original_key", "")),
        target_key=str(data.get("target_key", "")),
        transpose_semitones=int(data.get("transpose_semitones", 0) or 0),
        chord_sheet_text=str(data.get("chord_sheet_text", "")),
        chord_source_name=str(data.get("chord_source_name", "")),
        chord_source_url=str(data.get("chord_source_url", "")),
        original_chord_events=[
            ChordEvent(
                label=str(item.get("label", "")),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                confidence=float(item.get("confidence", 0.0)),
                root=str(item.get("root", "")),
                quality=str(item.get("quality", "")),
            )
            for item in data.get("original_chord_events", [])
            if isinstance(item, dict)
        ],
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


def _singer_analysis_to_dict(analysis: SingerAnalysisResult) -> dict[str, object]:
    return {
        "detected_singer_count": analysis.detected_singer_count,
        "provider": analysis.provider,
        "profiles": [asdict(profile) for profile in analysis.profiles],
        "assignments": [asdict(assignment) for assignment in analysis.assignments],
        "low_confidence_segments": analysis.low_confidence_segments,
        "analysis_window_seconds": analysis.analysis_window_seconds,
    }


def _singer_analysis_from_dict(data: dict[str, object]) -> SingerAnalysisResult:
    profiles = [
        SingerProfile(
            singer_id=str(item.get("singer_id", "")),
            label=str(item.get("label", "")),
            primary_color=str(item.get("primary_color", "")),
            secondary_color=str(item.get("secondary_color", "")),
            outline_color=str(item.get("outline_color", "")),
            shadow_color=str(item.get("shadow_color", "")),
            lane_index=int(item.get("lane_index", 0) or 0),
        )
        for item in data.get("profiles", [])
        if isinstance(item, dict) and str(item.get("singer_id", "")).strip()
    ]
    assignments = [
        SingerSegmentAssignment(
            segment_index=int(item.get("segment_index", 0) or 0),
            singer_id=str(item.get("singer_id", "")),
            label=str(item.get("label", "")),
            confidence=float(item.get("confidence", 0.0) or 0.0),
        )
        for item in data.get("assignments", [])
        if isinstance(item, dict) and str(item.get("singer_id", "")).strip()
    ]
    return SingerAnalysisResult(
        detected_singer_count=int(data.get("detected_singer_count", 1) or 1),
        provider=str(data.get("provider", "")),
        profiles=profiles,
        assignments=assignments,
        low_confidence_segments=int(data.get("low_confidence_segments", 0) or 0),
        analysis_window_seconds=float(data.get("analysis_window_seconds", 0.0) or 0.0),
    )


def _sessions_path() -> Path:
    return JOBS_DIR / "_sessions.json"


def _load_sessions() -> dict[str, str]:
    path = _sessions_path()
    if not path.exists():
        return {}
    with _STATE_LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


def _save_sessions(data: dict[str, str]):
    with _STATE_LOCK:
        _sessions_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_requests_path() -> Path:
    return JOBS_DIR / "_group_requests.json"


def _load_group_requests() -> dict[str, dict[str, object]]:
    path = _group_requests_path()
    if not path.exists():
        return {}
    with _STATE_LOCK:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        str(token): item
        for token, item in payload.items()
        if isinstance(item, dict)
    }


def _save_group_requests(data: dict[str, dict[str, object]]):
    with _STATE_LOCK:
        _group_requests_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _session_key(chat_id: int, user_id: int) -> str:
    return f"{chat_id}:{user_id}"


def create_group_request(
    *,
    group_chat_id: int,
    group_message_id: int,
    user_id: int,
    request_kind: str,
    payload: dict[str, object],
) -> str:
    token = uuid.uuid4().hex[:16]
    with _STATE_LOCK:
        requests = _prune_expired_group_requests(_load_group_requests())
        requests[token] = {
            "group_chat_id": group_chat_id,
            "group_message_id": group_message_id,
            "user_id": user_id,
            "request_kind": request_kind,
            "payload": payload,
            "created_at": _now_iso(),
        }
        _save_group_requests(requests)
    return token


def claim_group_request(token: str, user_id: int) -> tuple[dict[str, object] | None, str | None]:
    with _STATE_LOCK:
        requests = _prune_expired_group_requests(_load_group_requests())
        request = requests.get(token)
        if request is None:
            _save_group_requests(requests)
            return None, "missing"
        owner_user_id = int(request.get("user_id", 0) or 0)
        if owner_user_id == 0:
            _save_group_requests(requests)
            return None, "unclaimed"
        if owner_user_id != user_id:
            _save_group_requests(requests)
            return None, "forbidden"
        requests.pop(token, None)
        _save_group_requests(requests)
        return request, None


def bind_group_request_user(token: str, user_id: int) -> tuple[dict[str, object] | None, str | None]:
    with _STATE_LOCK:
        requests = _prune_expired_group_requests(_load_group_requests())
        request = requests.get(token)
        if request is None:
            _save_group_requests(requests)
            return None, "missing"

        owner_user_id = int(request.get("user_id", 0) or 0)
        if owner_user_id not in {0, user_id}:
            _save_group_requests(requests)
            return None, "forbidden"

        request["user_id"] = user_id
        requests[token] = request
        _save_group_requests(requests)
        return request, None


def cleanup_stale_group_requests(*, now: datetime | None = None, stale_after_hours: int = 24) -> int:
    with _STATE_LOCK:
        requests = _load_group_requests()
        filtered = _prune_expired_group_requests(requests, now=now, stale_after_hours=stale_after_hours)
        if len(filtered) == len(requests):
            return 0
        _save_group_requests(filtered)
        return len(requests) - len(filtered)


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
        "singer_analysis": job.singer_analysis_path.name if job.singer_analysis_path.exists() else "",
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
    delivery_chat_id: int = 0,
    delivery_reply_to_message_id: int = 0,
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
        delivery_chat_id=delivery_chat_id or chat_id,
        delivery_reply_to_message_id=delivery_reply_to_message_id,
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


def update_job_delivery(job: Job, *, delivery_chat_id: int = 0, delivery_reply_to_message_id: int = 0):
    job.delivery_chat_id = delivery_chat_id or job.manifest.chat_id
    job.delivery_reply_to_message_id = delivery_reply_to_message_id
    save_job(job)


def update_pending_delivery(job: Job, **updates) -> dict[str, object]:
    pending = dict(job.pending_delivery or {})
    if not pending.get("created_at"):
        pending["created_at"] = _now_iso()
    for key, value in updates.items():
        if value is None:
            pending.pop(key, None)
        else:
            pending[key] = value
    pending["updated_at"] = _now_iso()
    job.pending_delivery = pending
    save_job(job)
    return dict(job.pending_delivery)


def clear_pending_delivery(job: Job):
    job.pending_delivery = {}
    save_job(job)


def _write_delivery_feedback_summary(job: Job):
    lines = [
        f"שם השיר: {job.display_name}",
        f"מזהה משימה: {job.job_id}",
        "",
        "לוג הערות איכות:",
        "",
    ]
    for index, entry in enumerate(job.quality_feedback, 1):
        created_at = str(entry.get("created_at") or "")
        source = str(entry.get("source") or "text")
        text = str(entry.get("text") or "").strip()
        lines.append(f"{index}. זמן: {created_at}")
        lines.append(f"   מקור: {source}")
        if text:
            for feedback_line in text.splitlines():
                lines.append(f"   {feedback_line}")
        lines.append("")
    job.delivery_feedback_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def append_quality_feedback(
    job: Job,
    text: str,
    *,
    source: str = "text",
    user_id: int = 0,
    chat_id: int = 0,
) -> dict[str, object]:
    entry = {
        "created_at": _now_iso(),
        "source": source,
        "text": text.strip(),
        "user_id": user_id,
        "chat_id": chat_id,
    }
    feedback = list(job.quality_feedback)
    feedback.append(entry)
    job.quality_feedback = feedback
    pending = dict(job.pending_delivery or {})
    pending["status"] = "feedback_received"
    pending["feedback_received_at"] = entry["created_at"]
    job.pending_delivery = pending
    save_job(job)
    _write_delivery_feedback_summary(job)
    return entry


def write_delivery_feedback_template(job: Job) -> Path:
    lines = [
        f"שם השיר: {job.display_name}",
        f"מזהה משימה: {job.job_id}",
        "",
        "מה לא יצא מושלם?",
        "- תאר בקצרה את הבעיה הכללית.",
        "- אם יש מילים או שורות שגויות, ציין אותן בפורמט:",
        "  שורה 12: הטקסט הנכון",
        "",
        "הערות כלליות:",
        "-",
        "",
        "שורות לתיקון:",
        "שורה 1:",
        "שורה 2:",
        "",
        "הערות על תזמון / וידאו / אקורדים:",
        "-",
    ]
    job.delivery_feedback_template_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return job.delivery_feedback_template_path


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
    if job.review_status in {ReviewStatus.AWAITING_REVIEW, ReviewStatus.APPROVED} and job.review_timings_path.exists():
        return load_review_segments(job)
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


def is_reference_lyrics_option(option: dict[str, object]) -> bool:
    option_id = str(option.get("option_id") or "").strip().lower()
    return option_id == "draft" or bool(option.get("reference_only"))


def get_selectable_lyrics_options(job: Job) -> list[dict[str, object]]:
    return [option for option in get_lyrics_options(job) if not is_reference_lyrics_option(option)]


def get_reference_lyrics_option(job: Job) -> dict[str, object] | None:
    return next((option for option in get_lyrics_options(job) if is_reference_lyrics_option(option)), None)


def get_selected_lyrics_option_id(job: Job) -> str:
    verification = job.manifest.lyrics_verification or {}
    return str(verification.get("selected_option_id") or "draft")


def is_reference_selection_active(job: Job) -> bool:
    selected_option_id = get_selected_lyrics_option_id(job)
    option = next((item for item in get_lyrics_options(job) if item.get("option_id") == selected_option_id), None)
    if option is not None:
        return is_reference_lyrics_option(option)
    return selected_option_id == "draft"


def _detect_review_shrink_from_lines(existing_lines: list[str], new_lines: list[str]) -> dict[str, float | int] | None:
    existing_words = sum(len(line.split()) for line in existing_lines if line.strip())
    new_words = sum(len(line.split()) for line in new_lines if line.strip())
    if existing_words <= 0 or new_words <= 0:
        return None

    dropped_words = existing_words - new_words
    coverage_ratio = new_words / existing_words
    if existing_words >= 40 and dropped_words >= 16 and coverage_ratio < 0.75:
        return {
            "existing_words": existing_words,
            "new_words": new_words,
            "dropped_words": dropped_words,
            "coverage_ratio": round(coverage_ratio, 3),
        }
    return None


def _recover_short_review_segments(
    reference_segments: list[TranscriptSegment],
    candidate_lines: list[str],
) -> list[TranscriptSegment] | None:
    if not reference_segments or not candidate_lines or len(candidate_lines) >= len(reference_segments):
        return None
    if len(reference_segments) - len(candidate_lines) < 2:
        return None

    total_start = float(reference_segments[0].start)
    total_end = float(reference_segments[-1].end)
    recovered = _recover_with_local_song_alignment(candidate_lines, reference_segments, total_start, total_end)
    if not recovered or len(recovered) <= len(candidate_lines):
        return None
    if _has_suspicious_segment_timing(recovered) or _has_suspicious_gap_regression(recovered, reference_segments):
        return None

    total_span = max(total_end - total_start, 0.01)
    coverage_tolerance = max(5.0, total_span * 0.08)
    if float(recovered[-1].end) < total_end - coverage_tolerance:
        return None

    recovered_lines = [segment.text.strip() for segment in recovered if segment.text.strip()]
    existing_lines = [segment.text.strip() for segment in reference_segments if segment.text.strip()]
    if _detect_review_shrink_from_lines(existing_lines, recovered_lines) is not None:
        return None
    return recovered


def detect_suspicious_review_shrink(
    segments: list[TranscriptSegment],
    corrected_text: str,
) -> dict[str, float | int] | None:
    new_lines = _prepare_review_lines(segments, corrected_text)
    if not segments or not new_lines:
        return None

    if len(new_lines) < len(segments):
        recovered_segments = _recover_short_review_segments(segments, new_lines)
        if recovered_segments is not None:
            new_lines = [segment.text.strip() for segment in recovered_segments if segment.text.strip()]

    existing_lines = [segment.text.strip() for segment in segments if segment.text.strip()]
    return _detect_review_shrink_from_lines(existing_lines, new_lines)


def set_selected_lyrics_option(job: Job, option_id: str):
    verification = dict(job.manifest.lyrics_verification or {})
    verification["selected_option_id"] = option_id
    job.manifest.lyrics_verification = verification
    save_job(job)


def apply_lyrics_option(job: Job, option_id: str) -> dict[str, object]:
    option = next((item for item in get_lyrics_options(job) if item.get("option_id") == option_id), None)
    if option is None:
        raise ValueError(f"Unknown lyrics option: {option_id}")
    if is_reference_lyrics_option(option):
        raise ValueError("התמלול המקורי זמין להשוואה בלבד ואי אפשר לבחור בו כגרסה פעילה.")

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


def save_singer_analysis(job: Job, analysis: SingerAnalysisResult):
    _write_json(job.singer_analysis_path, _singer_analysis_to_dict(analysis))
    save_job(job)


def load_singer_analysis(job: Job) -> SingerAnalysisResult:
    return _singer_analysis_from_dict(_read_json(job.singer_analysis_path))


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
    with _STATE_LOCK:
        sessions = _load_sessions()
        sessions[_session_key(chat_id, user_id)] = job_id
        _save_sessions(sessions)


def set_review_message_id(job: Job, message_id: int):
    job.review_message_id = message_id
    save_job(job)


def clear_active_review_job(chat_id: int, user_id: int):
    with _STATE_LOCK:
        sessions = _load_sessions()
        sessions.pop(_session_key(chat_id, user_id), None)
        _save_sessions(sessions)


def get_active_review_job(chat_id: int, user_id: int) -> Job | None:
    with _STATE_LOCK:
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
    if job.status not in {JobStatus.AWAITING_REVIEW, JobStatus.DONE} or job.review_status not in {
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


def _character_timings_to_subwords(
    char_timings: list[CharacterTiming],
    confidence: float,
) -> list[SubWordTiming]:
    return [
        SubWordTiming(
            text=char_timing.char,
            start=char_timing.start,
            end=char_timing.end,
            confidence=confidence,
        )
        for char_timing in char_timings
        if char_timing.char
    ]


def _attach_word_detail_timings(
    word: WordTiming,
    *,
    force_recalculate: bool = False,
) -> WordTiming:
    char_timings = list(word.char_timings)
    subwords = list(word.subwords)
    if force_recalculate or not char_timings or not subwords:
        from .transcriber import interpolate_character_timings

        char_timings = interpolate_character_timings(word)
        subwords = _character_timings_to_subwords(char_timings, word.confidence)

    return WordTiming(
        word=word.word,
        start=word.start,
        end=word.end,
        confidence=word.confidence,
        source=word.source,
        aligned=word.aligned,
        subwords=subwords,
        char_timings=char_timings,
    )


def compose_review_line_edit_text(
    segments: list[TranscriptSegment],
    line_number: int,
    corrected_line: str,
) -> str:
    index = line_number - 1
    if index < 0 or index >= len(segments):
        raise ValueError(f"מספר שורה לא תקין: {line_number}")

    lines = [segment.text.strip() for segment in segments]
    lines[index] = corrected_line.strip()
    return "\n".join(lines)


def _normalize_line_text(text: str) -> str:
    return " ".join(token for token in (_normalize_word(word) for word in text.split()) if token)


def _review_line_similarity(left: str, right: str) -> float:
    left_norm = _normalize_line_text(left)
    right_norm = _normalize_line_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def _is_substantial_review_line(line: str) -> bool:
    normalized = _normalize_line_text(line)
    word_count = len(normalized.split())
    char_count = len(normalized.replace(" ", ""))
    return word_count >= 3 or char_count >= 10


def _repeat_window_metrics(lines: list[str], left_start: int, right_start: int, length: int) -> dict[str, float | int]:
    left_window = lines[left_start:left_start + length]
    right_window = lines[right_start:right_start + length]
    line_scores = [_review_line_similarity(left, right) for left, right in zip(left_window, right_window)]

    left_norm_lines = [_normalize_line_text(line) for line in left_window]
    right_norm_lines = [_normalize_line_text(line) for line in right_window]
    left_block = "\n".join(line for line in left_norm_lines if line)
    right_block = "\n".join(line for line in right_norm_lines if line)
    if left_block and right_block:
        block_similarity = SequenceMatcher(None, left_block, right_block, autojunk=False).ratio()
    else:
        block_similarity = 0.0

    left_tokens = {token for line in left_norm_lines for token in line.split()}
    right_tokens = {token for line in right_norm_lines for token in line.split()}
    token_overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return {
        "average_line_similarity": sum(line_scores) / max(1, len(line_scores)),
        "strong_line_matches": sum(1 for score in line_scores if score >= 0.78),
        "very_strong_line_matches": sum(1 for score in line_scores if score >= 0.92),
        "block_similarity": block_similarity,
        "token_overlap": token_overlap,
    }


def _build_repeat_anchor_map(lines: list[str]) -> dict[int, int]:
    anchors: dict[int, int] = {}
    similarity_cache: dict[tuple[int, int], float] = {}

    def _sim(left_index: int, right_index: int) -> float:
        key = (left_index, right_index)
        if key not in similarity_cache:
            similarity_cache[key] = _review_line_similarity(lines[left_index], lines[right_index])
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
        if right_index in anchors or not _is_substantial_review_line(lines[right_index]):
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

        substantial_count = sum(1 for offset in range(length) if _is_substantial_review_line(lines[index + offset]))
        metrics = _repeat_window_metrics(lines, source_start, index, length)
        if (
            length >= 2
            and substantial_count >= 2
            and (
                int(metrics["very_strong_line_matches"]) >= 1
                or float(metrics["average_line_similarity"]) >= 0.94
            )
        ):
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
        family["intervals"] = intervals
        accepted_families.append(family)
        for start, end in intervals:
            occupied.update(range(start, end))

    if not accepted_families:
        return []

    for family in accepted_families:
        intervals = sorted(set(family["intervals"]))
        length = int(family["length"])
        family_occupied = {line_index for start, end in intervals for line_index in range(start, end)}
        other_occupied = occupied - family_occupied

        while True:
            best_interval: tuple[int, int] | None = None
            best_score = 0.0
            for start in range(0, len(lines) - length + 1):
                end = start + length
                interval = (start, end)
                if interval in intervals:
                    continue
                if any(line_index in other_occupied or line_index in family_occupied for line_index in range(start, end)):
                    continue
                substantial_count = sum(1 for offset in range(length) if _is_substantial_review_line(lines[start + offset]))
                if substantial_count < 2:
                    continue

                best_metrics: dict[str, float | int] | None = None
                best_candidate_score = 0.0
                for existing_start, existing_end in intervals:
                    candidate_metrics = _repeat_window_metrics(lines, existing_start, start, min(length, existing_end - existing_start))
                    candidate_score = (
                        float(candidate_metrics["block_similarity"]) * 0.58
                        + float(candidate_metrics["average_line_similarity"]) * 0.27
                        + float(candidate_metrics["token_overlap"]) * 0.15
                    )
                    if candidate_score > best_candidate_score:
                        best_candidate_score = candidate_score
                        best_metrics = candidate_metrics

                if best_metrics is None:
                    continue

                if (
                    float(best_metrics["block_similarity"]) >= 0.72
                    and (
                        float(best_metrics["average_line_similarity"]) >= 0.58
                        or (
                            int(best_metrics["strong_line_matches"]) >= max(1, min(2, length))
                            and float(best_metrics["token_overlap"]) >= 0.34
                        )
                    )
                    and best_candidate_score > best_score
                ):
                    best_score = best_candidate_score
                    best_interval = interval

            if best_interval is None:
                break

            intervals.append(best_interval)
            intervals.sort()
            family["intervals"] = intervals
            for line_index in range(best_interval[0], best_interval[1]):
                occupied.add(line_index)
                family_occupied.add(line_index)

    accepted_families.sort(key=lambda item: min(start for start, _end in item["intervals"]))
    return accepted_families


def _build_repeat_structure(lines: list[str]) -> tuple[list[str], list[int], dict[int, int]]:
    families = _extract_repeat_families(lines)
    anchor_map: dict[int, int] = {}
    for family in families:
        source_start = int(family["source_start"])
        length = int(family["length"])
        intervals = sorted(set(family["intervals"]))
        for start, _end in intervals[1:]:
            for offset in range(length):
                anchor_map[start + offset] = source_start + offset

    if not anchor_map:
        anchor_map = _build_repeat_anchor_map(lines)

    compressed_lines: list[str] = []
    compressed_existing_indices: list[int] = []

    for index, line in enumerate(lines):
        if index in anchor_map:
            continue
        compressed_existing_indices.append(index)
        compressed_lines.append(line)

    return compressed_lines, compressed_existing_indices, anchor_map


def _align_review_lines_with_context(reference_lines: list[str], candidate_lines: list[str]) -> dict[int, str] | None:
    if not reference_lines or not candidate_lines:
        return None

    ref_count = len(reference_lines)
    candidate_count = len(candidate_lines)
    match_bias = 0.15
    skip_reference_penalty = 0.72
    skip_candidate_penalty = 0.95

    scores = [[float("-inf")] * (candidate_count + 1) for _ in range(ref_count + 1)]
    backtrack: list[list[tuple[str, float] | None]] = [[None] * (candidate_count + 1) for _ in range(ref_count + 1)]
    scores[0][0] = 0.0

    for ref_index in range(1, ref_count + 1):
        scores[ref_index][0] = scores[ref_index - 1][0] - skip_reference_penalty
        backtrack[ref_index][0] = ("skip_reference", 0.0)

    for candidate_index in range(1, candidate_count + 1):
        scores[0][candidate_index] = scores[0][candidate_index - 1] - skip_candidate_penalty
        backtrack[0][candidate_index] = ("skip_candidate", 0.0)

    for ref_index in range(1, ref_count + 1):
        reference_line = reference_lines[ref_index - 1]
        for candidate_index in range(1, candidate_count + 1):
            similarity = _review_line_similarity(reference_line, candidate_lines[candidate_index - 1])
            match_score = scores[ref_index - 1][candidate_index - 1] + similarity - match_bias
            skip_reference_score = scores[ref_index - 1][candidate_index] - skip_reference_penalty
            skip_candidate_score = scores[ref_index][candidate_index - 1] - skip_candidate_penalty

            best_score = match_score
            best_step: tuple[str, float] = ("match", similarity)
            if skip_reference_score > best_score:
                best_score = skip_reference_score
                best_step = ("skip_reference", 0.0)
            if skip_candidate_score > best_score:
                best_score = skip_candidate_score
                best_step = ("skip_candidate", 0.0)

            scores[ref_index][candidate_index] = best_score
            backtrack[ref_index][candidate_index] = best_step

    assignments: dict[int, str] = {}
    matched_similarities: list[float] = []
    skipped_references = 0
    skipped_candidates = 0
    ref_index = ref_count
    candidate_index = candidate_count

    while ref_index > 0 or candidate_index > 0:
        step = backtrack[ref_index][candidate_index]
        if step is None:
            return None
        action, similarity = step
        if action == "match":
            assignments[ref_index - 1] = candidate_lines[candidate_index - 1]
            matched_similarities.append(similarity)
            ref_index -= 1
            candidate_index -= 1
        elif action == "skip_reference":
            skipped_references += 1
            ref_index -= 1
        else:
            skipped_candidates += 1
            candidate_index -= 1

    if skipped_candidates > 0 or len(assignments) != candidate_count:
        return None

    if not matched_similarities:
        return None

    strong_matches = sum(1 for similarity in matched_similarities if similarity >= 0.58)
    average_similarity = sum(matched_similarities) / len(matched_similarities)
    fully_matched = len(assignments) == ref_count == candidate_count

    if skipped_references > max(1, ref_count // 5):
        return None
    if fully_matched and average_similarity >= 0.18:
        return assignments
    if strong_matches >= (1 if candidate_count <= 2 else 2):
        return assignments
    return None


def _expand_review_lines_from_structure(
    existing_lines: list[str],
    compressed_existing_indices: list[int],
    anchor_map: dict[int, int],
    compressed_assignments: dict[int, str],
) -> list[str] | None:
    if len(compressed_assignments) != len(compressed_existing_indices):
        return None

    assigned_existing = {
        compressed_existing_indices[compressed_index]: line
        for compressed_index, line in compressed_assignments.items()
    }
    expanded: list[str] = []

    for existing_index in range(len(existing_lines)):
        assigned_line = assigned_existing.get(existing_index)
        if assigned_line is not None:
            expanded.append(assigned_line)
            continue

        source_index = anchor_map.get(existing_index)
        if source_index is None or source_index not in assigned_existing:
            return None
        expanded.append(assigned_existing[source_index])

    return expanded


def _project_review_lines_to_existing(
    existing_lines: list[str],
    anchor_map: dict[int, int],
    assignments: dict[int, str],
) -> list[str]:
    projected: list[str] = []
    for existing_index, existing_line in enumerate(existing_lines):
        assigned_line = assignments.get(existing_index)
        if assigned_line is not None:
            projected.append(assigned_line)
            continue

        source_index = anchor_map.get(existing_index)
        if source_index is not None and source_index in assignments:
            projected.append(assignments[source_index])
            continue

        projected.append(existing_line)
    return projected


def _expand_repeated_review_lines(existing_lines: list[str], new_lines: list[str]) -> list[str]:
    if len(new_lines) >= len(existing_lines):
        return new_lines

    compressed_lines, compressed_existing_indices, anchor_map = _build_repeat_structure(existing_lines)
    if not anchor_map:
        return new_lines

    expanded: list[str] = []
    assigned: dict[int, str] = {}
    new_index = 0
    greedy_failed = False

    for existing_index, existing_line in enumerate(existing_lines):
        current_line = new_lines[new_index] if new_index < len(new_lines) else None
        if current_line is not None and _review_line_similarity(current_line, existing_line) >= 0.62:
            chosen = current_line
            new_index += 1
        elif existing_index in anchor_map and anchor_map[existing_index] in assigned:
            chosen = assigned[anchor_map[existing_index]]
        else:
            greedy_failed = True
            break

        expanded.append(chosen)
        assigned[existing_index] = chosen

    if greedy_failed or new_index != len(new_lines):
        structure_assignments = _align_review_lines_with_context(compressed_lines, new_lines)
        if structure_assignments is None:
            return new_lines
        structure_expanded = _expand_review_lines_from_structure(
            existing_lines,
            compressed_existing_indices,
            anchor_map,
            structure_assignments,
        )
        return structure_expanded or new_lines
    return expanded


def _prepare_review_lines(segments: list[TranscriptSegment], corrected_text: str) -> list[str]:
    new_lines = [line.strip() for line in corrected_text.strip().splitlines() if line.strip()]
    if not new_lines:
        return []

    existing_lines = [segment.text.strip() for segment in segments if segment.text.strip()]
    if not existing_lines:
        return new_lines

    return _expand_repeated_review_lines(existing_lines, new_lines)


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
    *,
    force_recalculate_words: bool = False,
) -> TranscriptSegment:
    reference_words = _flatten_segment_words(reference_segments)
    words_text = line_text.split()
    if reference_words:
        words = _align_words_to_draft(
            words_text,
            reference_words,
            span_start,
            span_end,
            force_recalculate_words=force_recalculate_words,
        )
    else:
        words = [
            _attach_word_detail_timings(word, force_recalculate=True)
            for word in _build_weighted_words_for_span(words_text, span_start, span_end)
        ]
    return TranscriptSegment(
        words=words,
        text=line_text.strip(),
        start=span_start,
        end=span_end,
    )


def _build_fallback_segments_for_lines(
    lines: list[str],
    span_start: float,
    span_end: float,
) -> list[TranscriptSegment]:
    total_duration = max(span_end - span_start, 0.01)
    weighted_lengths = [max(1, len(_normalize_line_text(line).replace(" ", ""))) for line in lines]
    total_weight = max(sum(weighted_lengths), 1)
    cursor = span_start
    fallback_segments: list[TranscriptSegment] = []
    for index, line_text in enumerate(lines):
        portion = total_duration * (weighted_lengths[index] / total_weight)
        next_cursor = span_end if index == len(lines) - 1 else min(span_end, cursor + portion)
        fallback_segments.append(
            _build_segment_from_line_text(
                line_text,
                cursor,
                next_cursor,
                [],
                force_recalculate_words=True,
            )
        )
        cursor = next_cursor
    return fallback_segments


def _reference_span_similarity(new_words_text: list[str], candidate_words: list[WordTiming]) -> float:
    new_norm = [_normalize_word(word) for word in new_words_text if _normalize_word(word)]
    candidate_norm = [_normalize_word(word.word) for word in candidate_words if _normalize_word(word.word)]
    if not new_norm or not candidate_norm:
        return 0.0

    token_matcher = SequenceMatcher(None, new_norm, candidate_norm, autojunk=False)
    matched_tokens = sum(a1 - a0 for tag, a0, a1, _b0, _b1 in token_matcher.get_opcodes() if tag == "equal")
    token_ratio = matched_tokens / max(len(new_norm), len(candidate_norm))
    char_ratio = SequenceMatcher(None, " ".join(new_norm), " ".join(candidate_norm), autojunk=False).ratio()
    length_penalty = abs(len(candidate_norm) - len(new_norm)) / max(len(new_norm), 1)
    return token_ratio * 0.65 + char_ratio * 0.35 - length_penalty * 0.18


def _find_best_reference_word_span(
    new_words_text: list[str],
    reference_words: list[WordTiming],
    start_index: int,
) -> dict[str, float | int] | None:
    if not new_words_text or start_index >= len(reference_words):
        return None

    normalized_count = sum(1 for word in new_words_text if _normalize_word(word))
    if normalized_count <= 0:
        return None

    def _search(lookahead_limit: int) -> dict[str, float | int] | None:
        max_candidate_length = max(4, normalized_count + 8)
        best_match: dict[str, float | int] | None = None

        for candidate_start in range(start_index, lookahead_limit):
            candidate_stop_limit = min(len(reference_words), candidate_start + max_candidate_length)
            for candidate_end in range(candidate_start + 1, candidate_stop_limit + 1):
                candidate_words = reference_words[candidate_start:candidate_end]
                similarity = _reference_span_similarity(new_words_text, candidate_words)
                if similarity <= 0.0:
                    continue

                proximity_penalty = (candidate_start - start_index) * 0.01
                candidate_duration = candidate_words[-1].end - candidate_words[0].start
                max_reasonable_duration = max(4.5, normalized_count * 0.95)
                duration_penalty = 0.0
                if candidate_duration > max_reasonable_duration:
                    duration_penalty = min(0.6, (candidate_duration - max_reasonable_duration) * 0.035)

                score = similarity - proximity_penalty - duration_penalty
                if best_match is None or score > float(best_match["score"]):
                    best_match = {
                        "start_index": candidate_start,
                        "end_index": candidate_end,
                        "score": score,
                    }
        return best_match

    local_limit = min(len(reference_words), start_index + max(12, normalized_count * 2 + 4))
    best_match = _search(local_limit)
    if best_match is not None and float(best_match["score"]) >= 0.26:
        return best_match

    lookahead_limit = min(len(reference_words), start_index + max(40, normalized_count * 10))
    best_match = _search(lookahead_limit)

    if best_match is None or float(best_match["score"]) < 0.32:
        return None
    return best_match


def _segment_block_score(segments: list[TranscriptSegment]) -> float:
    if not segments:
        return float("-inf")

    total_words = sum(len(segment.words) for segment in segments)
    matched_words = sum(
        1
        for segment in segments
        for word in segment.words
        if word.source != "review_hint"
    )
    suspicious_spans = sum(
        1
        for segment in segments
        if segment.words and (segment.end - segment.start) > max(8.0, len(segment.words) * 1.35)
    )
    suspicious_gaps = sum(
        1
        for index in range(len(segments) - 1)
        if segments[index + 1].start - segments[index].end > 6.0
    )
    return (matched_words / max(total_words, 1)) - suspicious_spans * 0.4 - suspicious_gaps * 0.3


def _segment_block_covers_span_reasonably(
    segments: list[TranscriptSegment],
    span_start: float,
    span_end: float,
) -> bool:
    if not segments:
        return False

    total_span = max(span_end - span_start, 0.01)
    coverage_start = max(span_start, segments[0].start)
    coverage_end = min(span_end, segments[-1].end)
    coverage_ratio = max(0.0, coverage_end - coverage_start) / total_span
    edge_tolerance = max(2.0, total_span * 0.12)
    return (
        coverage_ratio >= 0.78
        and segments[0].start <= span_start + edge_tolerance
        and segments[-1].end >= span_end - edge_tolerance
    )


def _max_inter_segment_gap(segments: list[TranscriptSegment]) -> float:
    if len(segments) < 2:
        return 0.0
    return max(
        0.0,
        max(float(segments[index + 1].start) - float(segments[index].end) for index in range(len(segments) - 1)),
    )


def _max_word_gap_within_segment(segment: TranscriptSegment) -> float:
    if len(segment.words) < 2:
        return 0.0
    return max(
        0.0,
        max(float(segment.words[index + 1].start) - float(segment.words[index].end) for index in range(len(segment.words) - 1)),
    )


def _has_suspicious_segment_timing(segments: list[TranscriptSegment]) -> bool:
    for segment in segments:
        word_count = max(1, len(segment.words))
        span = float(segment.end) - float(segment.start)
        if span > max(12.0, word_count * 1.8):
            return True
        if _max_word_gap_within_segment(segment) > max(3.5, span * 0.42):
            return True
    return False


def _has_suspicious_gap_regression(candidate_segments: list[TranscriptSegment], reference_segments: list[TranscriptSegment]) -> bool:
    candidate_gap = _max_inter_segment_gap(candidate_segments)
    if candidate_gap <= 0.0:
        return False

    reference_gap = _max_inter_segment_gap(reference_segments)
    allowed_gap = max(12.0, reference_gap + 8.0, reference_gap * 2.2)
    return candidate_gap > allowed_gap


def _reference_coverage_ratio(reference_segment: TranscriptSegment, candidate_segments: list[TranscriptSegment]) -> float:
    reference_span = max(float(reference_segment.end) - float(reference_segment.start), 0.01)
    covered = 0.0
    for candidate in candidate_segments:
        overlap_start = max(float(reference_segment.start), float(candidate.start))
        overlap_end = min(float(reference_segment.end), float(candidate.end))
        if overlap_end > overlap_start:
            covered += overlap_end - overlap_start
    return covered / reference_span


def _find_uncovered_reference_blocks(
    candidate_segments: list[TranscriptSegment],
    reference_segments: list[TranscriptSegment],
) -> list[tuple[int, int]]:
    uncovered_blocks: list[tuple[int, int]] = []
    block_start: int | None = None

    for index, reference_segment in enumerate(reference_segments):
        coverage_ratio = _reference_coverage_ratio(reference_segment, candidate_segments)
        covered = coverage_ratio >= 0.58
        if covered:
            if block_start is not None:
                uncovered_blocks.append((block_start, index))
                block_start = None
            continue
        if block_start is None:
            block_start = index

    if block_start is not None:
        uncovered_blocks.append((block_start, len(reference_segments)))
    return uncovered_blocks


def _reference_block_text_metrics(
    candidate_lines: list[str],
    reference_block: list[TranscriptSegment],
) -> dict[str, float]:
    reference_text = " ".join(segment.text.strip() for segment in reference_block if segment.text.strip())
    candidate_text = " ".join(line.strip() for line in candidate_lines if line.strip())
    reference_norm = _normalize_line_text(reference_text)
    candidate_norm = _normalize_line_text(candidate_text)
    if not reference_norm or not candidate_norm:
        return {
            "ratio": 0.0,
            "token_overlap": 0.0,
            "score": 0.0,
        }

    reference_tokens = set(reference_norm.split())
    candidate_tokens = set(candidate_norm.split())
    shared_tokens = reference_tokens & candidate_tokens
    token_overlap = len(shared_tokens) / max(1, min(len(reference_tokens), len(candidate_tokens)))
    ratio = SequenceMatcher(None, reference_norm, candidate_norm, autojunk=False).ratio()
    return {
        "ratio": ratio,
        "token_overlap": token_overlap,
        "score": token_overlap * 0.55 + ratio * 0.45,
    }


def _recover_reference_block_from_recent_lines(
    all_lines: list[str],
    reference_block: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    if not all_lines or not reference_block:
        return []

    block_start = reference_block[0].start
    block_end = reference_block[-1].end
    total_block_span = max(block_end - block_start, 0.01)
    coverage_tolerance = max(1.5, total_block_span * 0.12)
    best_candidate: list[TranscriptSegment] = []
    best_score = float("-inf")
    max_window = min(8, len(all_lines))

    for window in range(1, max_window + 1):
        for start_index in range(0, len(all_lines) - window + 1):
            candidate_lines = all_lines[start_index:start_index + window]
            substantial_count = sum(1 for line in candidate_lines if _is_substantial_review_line(line))
            if substantial_count < 1:
                continue

            text_metrics = _reference_block_text_metrics(candidate_lines, reference_block)
            if (
                float(text_metrics["token_overlap"]) < 0.58
                or float(text_metrics["ratio"]) < 0.32
                or float(text_metrics["score"]) < 0.56
            ):
                continue

            candidate = _build_segments_from_line_block_locally(candidate_lines, block_start, block_end, reference_block)
            if not candidate:
                continue
            if candidate[0].start > block_start + coverage_tolerance:
                continue
            if candidate[-1].end < block_end - coverage_tolerance:
                continue
            if _has_suspicious_segment_timing(candidate):
                continue

            score = float(text_metrics["score"])
            score += max(0.0, (candidate[-1].end - candidate[0].start) / total_block_span) * 0.08
            score -= abs(candidate[0].start - block_start) * 0.015
            if score > best_score:
                best_score = score
                best_candidate = candidate

    return best_candidate


def _merge_recovered_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    ordered = sorted(segments, key=lambda segment: (float(segment.start), float(segment.end), segment.text))
    merged: list[TranscriptSegment] = []
    for segment in ordered:
        if merged:
            previous = merged[-1]
            same_text = _normalize_line_text(previous.text) == _normalize_line_text(segment.text)
            same_window = abs(float(previous.start) - float(segment.start)) <= 0.35 and abs(float(previous.end) - float(segment.end)) <= 0.35
            if same_text and same_window:
                continue
        merged.append(segment)
    return merged


def _recover_with_local_song_alignment(
    new_lines: list[str],
    reference_segments: list[TranscriptSegment],
    total_start: float,
    total_end: float,
) -> list[TranscriptSegment]:
    local_segments = _build_segments_from_line_block_locally(new_lines, total_start, total_end, reference_segments)
    if not local_segments:
        return []

    recovered = list(local_segments)
    for _ in range(2):
        uncovered_blocks = _find_uncovered_reference_blocks(recovered, reference_segments)
        if not uncovered_blocks:
            break

        additions: list[TranscriptSegment] = []
        for start_index, end_index in uncovered_blocks:
            block_candidate = _recover_reference_block_from_recent_lines(new_lines, reference_segments[start_index:end_index])
            if block_candidate:
                additions.extend(block_candidate)
        if not additions:
            break
        recovered = _merge_recovered_segments(recovered + additions)

    return recovered


def _build_segments_from_line_block_locally(
    lines: list[str],
    span_start: float,
    span_end: float,
    reference_segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    reference_words = _flatten_segment_words(reference_segments)
    if not lines or not reference_words:
        return []

    matches: list[dict[str, float | int] | None] = []
    reference_cursor = 0
    for line_text in lines:
        words_text = line_text.split()
        match = _find_best_reference_word_span(words_text, reference_words, reference_cursor)
        matches.append(match)
        if match is not None:
            reference_cursor = max(reference_cursor, int(match["end_index"]))

    built_segments: list[TranscriptSegment | None] = [None] * len(lines)
    for index, match in enumerate(matches):
        if match is None:
            continue
        start_index = int(match["start_index"])
        end_index = int(match["end_index"])
        candidate_words = reference_words[start_index:end_index]
        if not candidate_words:
            continue
        line_words = _align_words_to_draft(
            lines[index].split(),
            candidate_words,
            candidate_words[0].start,
            candidate_words[-1].end,
            force_recalculate_words=True,
        )
        segment_start = line_words[0].start if line_words else candidate_words[0].start
        segment_end = line_words[-1].end if line_words else candidate_words[-1].end
        built_segments[index] = TranscriptSegment(
            words=line_words,
            text=lines[index],
            start=segment_start,
            end=segment_end,
        )

    index = 0
    while index < len(built_segments):
        if built_segments[index] is not None:
            index += 1
            continue
        run_start = index
        while index < len(built_segments) and built_segments[index] is None:
            index += 1
        run_end = index
        gap_start = built_segments[run_start - 1].end if run_start > 0 and built_segments[run_start - 1] else span_start
        gap_end = built_segments[run_end].start if run_end < len(built_segments) and built_segments[run_end] else span_end
        fallback_segments = _build_fallback_segments_for_lines(lines[run_start:run_end], gap_start, gap_end)
        for offset, segment in enumerate(fallback_segments):
            built_segments[run_start + offset] = segment

    finalized: list[TranscriptSegment] = []
    cursor = span_start
    for built in built_segments:
        if built is None:
            continue
        line_start = max(cursor, built.start)
        line_end = max(line_start + 0.01, built.end)
        line_words = built.words
        if line_words:
            shift = line_start - line_words[0].start
            if abs(shift) > 1e-6:
                shifted_words: list[WordTiming] = []
                for word in line_words:
                    shifted_words.append(
                        WordTiming(
                            word=word.word,
                            start=round(word.start + shift, 6),
                            end=round(word.end + shift, 6),
                            confidence=word.confidence,
                            source=word.source,
                            aligned=word.aligned,
                            subwords=[
                                SubWordTiming(
                                    text=subword.text,
                                    start=round(subword.start + shift, 6),
                                    end=round(subword.end + shift, 6),
                                    confidence=subword.confidence,
                                )
                                for subword in word.subwords
                            ],
                            char_timings=[
                                CharacterTiming(
                                    char=char_timing.char,
                                    start=round(char_timing.start + shift, 6),
                                    end=round(char_timing.end + shift, 6),
                                )
                                for char_timing in word.char_timings
                            ],
                        )
                    )
                line_words = shifted_words
                line_end = max(line_start + 0.01, line_words[-1].end)
        finalized.append(
            TranscriptSegment(
                words=line_words,
                text=built.text,
                start=line_start,
                end=min(span_end, line_end),
            )
        )
        cursor = finalized[-1].end
    return finalized


def _build_segments_from_line_block(
    lines: list[str],
    span_start: float,
    span_end: float,
    reference_segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    if not lines:
        return []

    normalized_lines = [line.strip() for line in lines]
    reference_words = _flatten_segment_words(reference_segments)
    block_words_text: list[str] = []
    line_word_ranges: list[tuple[str, int, int]] = []

    for line_text in normalized_lines:
        words_text = line_text.split()
        start_index = len(block_words_text)
        block_words_text.extend(words_text)
        line_word_ranges.append((line_text, start_index, len(block_words_text)))

    if not block_words_text:
        return _build_fallback_segments_for_lines(normalized_lines, span_start, span_end)

    local_segments = _build_segments_from_line_block_locally(
        normalized_lines,
        span_start,
        span_end,
        reference_segments,
    )

    aligned_block_words = _align_words_to_draft(
        block_words_text,
        reference_words,
        span_start,
        span_end,
        force_recalculate_words=True,
    )

    built_segments: list[TranscriptSegment] = []
    for index, (line_text, start_index, end_index) in enumerate(line_word_ranges):
        line_words = aligned_block_words[start_index:end_index]
        if not line_words:
            line_start = built_segments[-1].end if built_segments else span_start
            line_end = span_end if index == len(line_word_ranges) - 1 else max(line_start + 0.01, line_start)
        else:
            line_start = max(span_start, line_words[0].start)
            if built_segments:
                line_start = max(line_start, built_segments[-1].end)
            line_end = min(span_end, max(line_start + 0.01, line_words[-1].end))
        built_segments.append(
            TranscriptSegment(
                words=line_words,
                text=line_text,
                start=line_start,
                end=line_end,
            )
        )

    if (
        local_segments
        and (
            len(reference_segments) <= 1
            or _segment_block_covers_span_reasonably(local_segments, span_start, span_end)
        )
        and _segment_block_score(local_segments) >= _segment_block_score(built_segments) - 0.02
    ):
        return local_segments
    return built_segments


def _build_direct_review_segments(
    segments: list[TranscriptSegment],
    new_lines: list[str],
) -> list[TranscriptSegment]:
    return [
        _build_segment_from_line_text(
            new_lines[index],
            segment.start,
            segment.end,
            [segment],
            force_recalculate_words=_normalize_line_text(new_lines[index]) != _normalize_line_text(segment.text),
        )
        for index, segment in enumerate(segments)
    ]


def _direct_review_mapping_needs_neighbor_context(
    segments: list[TranscriptSegment],
    new_lines: list[str],
) -> bool:
    if len(new_lines) != len(segments) or len(segments) < 2:
        return False

    for index, new_line in enumerate(new_lines):
        new_words = [word for word in new_line.split() if _normalize_word(word)]
        if not new_words or not segments[index].words:
            continue

        direct_words = segments[index].words
        direct_score = _reference_span_similarity(new_words, direct_words)
        direct_word_count = sum(1 for word in direct_words if _normalize_word(word.word))
        best_context_score = direct_score

        candidate_ranges: list[tuple[int, int]] = []
        if index > 0:
            candidate_ranges.append((index - 1, index + 1))
        if index + 1 < len(segments):
            candidate_ranges.append((index, index + 2))
        if index > 0 and index + 1 < len(segments):
            candidate_ranges.append((index - 1, index + 2))

        for start_index, end_index in candidate_ranges:
            candidate_words = _flatten_segment_words(segments[start_index:end_index])
            if not candidate_words:
                continue
            best_context_score = max(best_context_score, _reference_span_similarity(new_words, candidate_words))

        if (
            best_context_score >= max(0.84, direct_score + 0.18)
            and len(new_words) >= direct_word_count + 2
        ):
            return True

    return False


def _align_words_to_draft(
    new_words_text: list[str],
    orig_words: list[WordTiming],
    segment_start: float,
    segment_end: float,
    *,
    force_recalculate_words: bool = False,
) -> list[WordTiming]:
    """Assign timings to corrected text while preserving the original error span."""
    if not new_words_text:
        return []
    if not orig_words:
        return [
            _attach_word_detail_timings(word, force_recalculate=True)
            for word in _build_weighted_words_for_span(new_words_text, segment_start, segment_end)
        ]

    orig_norm = [_normalize_word(w.word) for w in orig_words]
    new_norm = [_normalize_word(w) for w in new_words_text]

    result: list[WordTiming | None] = [None] * len(new_words_text)

    for tag, a0, a1, d0, d1 in SequenceMatcher(None, new_norm, orig_norm, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(a1 - a0):
                ow = orig_words[d0 + k]
                result[a0 + k] = _attach_word_detail_timings(
                    WordTiming(
                        word=new_words_text[a0 + k],
                        start=ow.start,
                        end=ow.end,
                        confidence=ow.confidence,
                        source=ow.source,
                        aligned=ow.aligned,
                        subwords=list(ow.subwords),
                        char_timings=list(ow.char_timings),
                    ),
                    force_recalculate=new_words_text[a0 + k] != ow.word,
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
            result[a0 + offset] = _attach_word_detail_timings(word, force_recalculate=True)

    for index, word in enumerate(result):
        if word is None:
            continue
        if word.start < segment_start - 1e-3 or word.end > segment_end + 1e-3:
            result[index] = None

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

    detailed_words: list[WordTiming] = []
    for word in final:
        needs_recalculation = force_recalculate_words or word.source == "review_hint"
        detailed_words.append(_attach_word_detail_timings(word, force_recalculate=needs_recalculation))

    return detailed_words


def update_transcript_line(segments: list[TranscriptSegment], line_number: int, corrected_line: str) -> list[TranscriptSegment]:
    index = line_number - 1
    if index < 0 or index >= len(segments):
        raise ValueError(f"מספר שורה לא תקין: {line_number}")
    updated = list(segments)
    segment = segments[index]
    words = corrected_line.strip().split()
    updated[index] = TranscriptSegment(
        words=_align_words_to_draft(
            words,
            segment.words,
            segment.start,
            segment.end,
            force_recalculate_words=True,
        ),
        text=corrected_line.strip(),
        start=segment.start,
        end=segment.end,
    )
    return updated


def update_transcript_text(segments: list[TranscriptSegment], corrected_text: str) -> list[TranscriptSegment]:
    new_lines = _prepare_review_lines(segments, corrected_text)
    if not new_lines:
        return segments
    total_start = segments[0].start if segments else 0.0
    total_end = segments[-1].end if segments else 0.0
    if not segments:
        total_duration = 0.01
        return [
            TranscriptSegment(
                words=[
                    _attach_word_detail_timings(word, force_recalculate=True)
                    for word in _build_weighted_words_for_span(
                        line.split(),
                        total_start + (index / len(new_lines)) * total_duration,
                        total_start + ((index + 1) / len(new_lines)) * total_duration,
                    )
                ],
                text=line,
                start=total_start + (index / len(new_lines)) * total_duration,
                end=total_start + ((index + 1) / len(new_lines)) * total_duration,
            )
            for index, line in enumerate(new_lines)
        ]

    original_lines = [segment.text.strip() for segment in segments]
    if len(new_lines) < len(segments):
        recovered_segments = _recover_short_review_segments(segments, new_lines)
        if recovered_segments is not None:
            return recovered_segments

    if len(new_lines) < len(segments):
        compressed_lines, compressed_existing_indices, anchor_map = _build_repeat_structure(original_lines)
        if anchor_map:
            projected_lines: list[str] | None = None
            structure_assignments = _align_review_lines_with_context(compressed_lines, new_lines)
            if structure_assignments is not None:
                projected_lines = _expand_review_lines_from_structure(
                    original_lines,
                    compressed_existing_indices,
                    anchor_map,
                    structure_assignments,
                )
            if projected_lines is None:
                direct_assignments = _align_review_lines_with_context(original_lines, new_lines)
                if direct_assignments is not None:
                    projected_lines = _project_review_lines_to_existing(original_lines, anchor_map, direct_assignments)
            if projected_lines is not None and len(projected_lines) == len(segments):
                new_lines = projected_lines

    if len(new_lines) == len(segments):
        direct_segments = _build_direct_review_segments(segments, new_lines)
        if not _direct_review_mapping_needs_neighbor_context(segments, new_lines):
            return direct_segments

    original_norm = [_normalize_line_text(line) for line in original_lines]
    new_norm = [_normalize_line_text(line) for line in new_lines]

    rebuilt_segments: list[TranscriptSegment] = []
    matcher = SequenceMatcher(None, new_norm, original_norm, autojunk=False)

    for tag, a0, a1, d0, d1 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(a1 - a0):
                reference_segment = segments[d0 + offset]
                rebuilt_segments.append(
                    _build_segment_from_line_text(
                        new_lines[a0 + offset],
                        reference_segment.start,
                        reference_segment.end,
                        [reference_segment],
                        force_recalculate_words=False,
                    )
                )
            continue

        if tag == "delete":
            rebuilt_segments.extend(segments[d0:d1])
            continue

        if tag == "insert":
            rebuilt_segments.extend(segments[d0:d1])
            continue

        if tag not in {"replace"} or a1 <= a0:
            if tag == "replace" and d1 > d0:
                rebuilt_segments.extend(segments[d0:d1])
            continue

        span_start, span_end = _span_for_line_block(segments, d0, d1, total_start, total_end)
        block_reference_segments = segments[d0:d1]
        rebuilt_block = _build_segments_from_line_block(
            new_lines[a0:a1],
            span_start,
            span_end,
            block_reference_segments,
        )
        rebuilt_segments.extend(rebuilt_block)

    if rebuilt_segments:
        uncovered_blocks = _find_uncovered_reference_blocks(rebuilt_segments, segments)
        if uncovered_blocks:
            additions: list[TranscriptSegment] = []
            for start_index, end_index in uncovered_blocks:
                recovered_block = _recover_reference_block_from_recent_lines(new_lines, segments[start_index:end_index])
                if recovered_block:
                    additions.extend(recovered_block)
            if additions:
                recovered_segments = _merge_recovered_segments(rebuilt_segments + additions)
                if (
                    recovered_segments
                    and recovered_segments[-1].end >= total_end - 0.25
                    and not _has_suspicious_gap_regression(recovered_segments, segments)
                ):
                    rebuilt_segments = recovered_segments

        if _has_suspicious_segment_timing(rebuilt_segments):
            recovered_local_song = _recover_with_local_song_alignment(new_lines, segments, total_start, total_end)
            if (
                recovered_local_song
                and recovered_local_song[-1].end >= total_end - 0.25
                and not _has_suspicious_segment_timing(recovered_local_song)
                and not _has_suspicious_gap_regression(recovered_local_song, segments)
            ):
                return recovered_local_song
        if _has_suspicious_gap_regression(rebuilt_segments, segments):
            rebuilt_full_song = _build_segments_from_line_block(new_lines, total_start, total_end, segments)
            if rebuilt_full_song and (
                not _has_suspicious_gap_regression(rebuilt_full_song, segments)
                or _max_inter_segment_gap(rebuilt_full_song) + 5.0 < _max_inter_segment_gap(rebuilt_segments)
            ):
                return rebuilt_full_song
        return rebuilt_segments

    total_duration = max(total_end - total_start, 0.01)
    new_segments = []
    for index, line in enumerate(new_lines):
        segment_start = total_start + (index / len(new_lines)) * total_duration
        segment_end = total_start + ((index + 1) / len(new_lines)) * total_duration
        new_segments.append(
            TranscriptSegment(
                words=[
                    _attach_word_detail_timings(word, force_recalculate=True)
                    for word in _build_weighted_words_for_span(line.split(), segment_start, segment_end)
                ],
                text=line,
                start=segment_start,
                end=segment_end,
            )
        )
    return new_segments


def find_segment_word_text_mismatches(segments: list[TranscriptSegment]) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    for index, segment in enumerate(segments, 1):
        text_tokens = [token.strip() for token in segment.text.split() if token.strip()]
        word_tokens = [word.word.strip() for word in segment.words if word.word.strip()]
        if text_tokens != word_tokens:
            mismatches.append(
                {
                    "segment_index": index,
                    "text": segment.text,
                    "text_tokens": text_tokens,
                    "word_tokens": word_tokens,
                    "start": segment.start,
                    "end": segment.end,
                }
            )
    return mismatches


def rebuild_segments_from_authoritative_text(
    reference_segments: list[TranscriptSegment],
    authoritative_segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    authoritative_lines = [segment.text.strip() for segment in authoritative_segments if segment.text.strip()]
    if not authoritative_lines:
        return authoritative_segments
    return update_transcript_text(reference_segments, "\n".join(authoritative_lines))


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
        _remove_job_from_sessions(job.job_id)
        shutil.rmtree(str(job.job_dir), ignore_errors=True)
        logger.info("Cleaned up job %s", job.job_id)
    except Exception as exc:
        logger.error("Cleanup failed for %s: %s", job.job_id, exc)
