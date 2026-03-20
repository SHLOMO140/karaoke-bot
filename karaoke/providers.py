"""Provider protocols for pluggable pipeline components."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import (
    AlignedTranscript,
    Job,
    KaraokeStyle,
    LanguageDetectionResult,
    LyricsVerificationResult,
    TranscriptDraft,
    TranscriptSegment,
)


class SeparationProvider(Protocol):
    name: str

    def separate(self, input_audio: str, job_dir: Path) -> tuple[str, str]:
        ...


class LanguageDetector(Protocol):
    name: str

    def detect(self, audio_path: str, job_dir: Path) -> LanguageDetectionResult:
        ...


class TranscriptionProvider(Protocol):
    name: str

    def transcribe(self, audio_path: str) -> TranscriptDraft:
        ...


class AlignmentProvider(Protocol):
    name: str

    def align(
        self,
        audio_path: str,
        approved_segments: list[TranscriptSegment],
        draft_segments: list[TranscriptSegment],
        video_frame_rate: float | None = None,
    ) -> AlignedTranscript:
        ...


class LyricsVerifier(Protocol):
    name: str

    def verify(self, title: str, draft: TranscriptDraft) -> LyricsVerificationResult:
        ...


class MultiStepLyricsVerifierProtocol(Protocol):
    name: str

    def verify(self, title: str, draft: TranscriptDraft) -> LyricsVerificationResult: ...

    def post_review_steps(self, job: "Job", original_draft: TranscriptDraft) -> None: ...


class SubtitleRenderer(Protocol):
    name: str

    def render(self, segments: list[TranscriptSegment], output_path: str, style: KaraokeStyle | None = None):
        ...


class SubtitleValidator(Protocol):
    name: str

    def validate(self, segments: list[TranscriptSegment], style: KaraokeStyle) -> list[str]:
        ...
