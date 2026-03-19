"""Typed pipeline exceptions with user-facing metadata."""

from __future__ import annotations

from dataclasses import asdict

from .models import ErrorInfo


class PipelineError(Exception):
    code = "pipeline_error"
    stage = "pipeline"
    default_user_message = "אירעה שגיאה במהלך העיבוד."

    def __init__(self, technical_message: str = "", user_message: str | None = None):
        self.info = ErrorInfo(
            code=self.code,
            stage=self.stage,
            user_message=user_message or self.default_user_message,
            technical_message=technical_message,
        )
        super().__init__(technical_message or self.info.user_message)

    def to_dict(self) -> dict[str, str]:
        return asdict(self.info)


class DownloadError(PipelineError):
    code = "download_failed"
    stage = "download"
    default_user_message = "הורדת המדיה נכשלה."


class AudioExtractionError(PipelineError):
    code = "audio_extraction_failed"
    stage = "audio_extraction"
    default_user_message = "חילוץ האודיו נכשל."


class SeparationError(PipelineError):
    code = "vocal_separation_failed"
    stage = "vocal_separation"
    default_user_message = "הפרדת הווקאל נכשלה."


class LanguageDetectionError(PipelineError):
    code = "language_detection_failed"
    stage = "language_detection"
    default_user_message = "לא זוהתה עברית דומיננטית בשיר."


class TranscriptionError(PipelineError):
    code = "transcription_failed"
    stage = "transcription"
    default_user_message = "תמלול השיר נכשל."


class AlignmentError(PipelineError):
    code = "alignment_failed"
    stage = "alignment"
    default_user_message = "יישור המילים לטיימינג נכשל."


class MusicAnalysisError(PipelineError):
    code = "music_analysis_failed"
    stage = "music_analysis"
    default_user_message = "זיהוי המקצב והאקורדים נכשל."


class SubtitleGenerationError(PipelineError):
    code = "subtitle_generation_failed"
    stage = "subtitle_generation"
    default_user_message = "יצירת קבצי הכתוביות נכשלה."


class VideoRenderError(PipelineError):
    code = "video_render_failed"
    stage = "video_render"
    default_user_message = "יצירת הווידאו הסופי נכשלה."


class DeliveryError(PipelineError):
    code = "delivery_failed"
    stage = "delivery"
    default_user_message = "שליחת הקבצים לטלגרם נכשלה."
