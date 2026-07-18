"""Shared data models for the Hebrew karaoke pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class JobStatus(Enum):
    CREATED = "created"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    SEPARATING_VOCALS = "separating_vocals"
    DETECTING_LANGUAGE = "detecting_language"
    TRANSCRIBING = "transcribing"
    VERIFYING_LYRICS = "verifying_lyrics"
    AWAITING_REVIEW = "awaiting_review"
    ALIGNING = "aligning"
    GENERATING_SUBS = "generating_subs"
    RENDERING_VIDEO = "rendering_video"
    DELIVERING = "delivering"
    DONE = "done"
    ERROR = "error"


class ReviewStatus(Enum):
    NOT_STARTED = "not_started"
    DRAFT_READY = "draft_ready"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    COMPLETED = "completed"


STATUS_MESSAGES = {
    JobStatus.DOWNLOADING: "שלב 1/7: מוריד מדיה...",
    JobStatus.EXTRACTING_AUDIO: "שלב 1/7: מחלץ אודיו...",
    JobStatus.SEPARATING_VOCALS: "שלב 2/7: מפריד ווקאל מהפלייבק...",
    JobStatus.DETECTING_LANGUAGE: "שלב 3/7: מזהה שפה ודומיננטיות עברית...",
    JobStatus.TRANSCRIBING: "שלב 4/7: מתמלל את השירה לעברית...",
    JobStatus.AWAITING_REVIEW: "שלב 5/7: ממתין לאישור או תיקון הטקסט...",
    JobStatus.ALIGNING: "שלב 6/7: מיישר מילים לטיימינג סופי...",
    JobStatus.GENERATING_SUBS: "שלב 6/7: מייצר SRT ו-ASS...",
    JobStatus.RENDERING_VIDEO: "שלב 7/7: צורב כתוביות על הווידאו...",
    JobStatus.DELIVERING: "שולח את כל הקבצים בחזרה לטלגרם...",
    JobStatus.DONE: "העיבוד הושלם.",
    JobStatus.ERROR: "אירעה שגיאה במהלך העיבוד.",
}
STATUS_MESSAGES[JobStatus.VERIFYING_LYRICS] = "שלב 5/8: בודק התאמה למילות השיר לפני review..."


@dataclass
class ErrorInfo:
    code: str
    stage: str
    user_message: str
    technical_message: str = ""


@dataclass
class LanguageDetectionResult:
    language: str
    probability: float
    policy_decision: str
    warning_message: str = ""
    hebrew_ratio: float = 0.0
    sample_text: str = ""
    provider: str = ""


@dataclass
class LyricsVerificationResult:
    provider: str = ""
    llm_provider: str = ""
    verdict: str = "not_run"
    confidence: float = 0.0
    search_query: str = ""
    summary: str = ""
    matched_sources: list[str] = field(default_factory=list)
    web_excerpt: str = ""
    local_warnings: list[str] = field(default_factory=list)
    corrected_lines: list[str] = field(default_factory=list)
    correction_count: int = 0
    applied: bool = False
    selected_option_id: str = "draft"
    options: list[dict[str, object]] = field(default_factory=list)
    consensus_result: "ConsensusResult | None" = None
    source_versions: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SubWordTiming:
    text: str
    start: float
    end: float
    confidence: float = 0.0


@dataclass
class CharacterTiming:
    """Timing for a single character/grapheme cluster."""
    char: str
    start: float
    end: float


class VerificationVerdict(str, Enum):
    """Verdict for the new multi-step lyrics verification."""
    CONSENSUS = "consensus"          # 3+ sources agreed
    GROK_VERIFIED = "grok_verified"  # Grok decided
    HUMAN_APPROVED = "human_approved"    # User approved/corrected
    NO_SOURCES = "no_sources"        # No web sources found
    NOT_RUN = "not_run"


@dataclass
class DisputedLine:
    """A lyrics line where sources disagree."""
    line_number: int
    versions: dict[str, str]  # source_name → text
    grok_recommendation: str | None = None
    grok_confidence: float = 0.0


@dataclass
class ConsensusResult:
    """Result of consensus engine comparing multiple sources."""
    consensus_reached: bool
    agreed_sources: int
    lyrics: list[str]
    disputes: list[DisputedLine] = field(default_factory=list)


@dataclass
class CharChange:
    """A single character-level change."""
    position: int
    old_char: str
    new_char: str
    change_type: str  # "replaced", "added", "removed"


@dataclass
class CharDiff:
    """Character-level diff for one word."""
    word_index: int
    original_word: str
    corrected_word: str
    char_changes: list[CharChange] = field(default_factory=list)
    grok_explanation: str | None = None


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    confidence: float = 0.0
    source: str = "draft_whisper"
    aligned: bool = False
    subwords: list[SubWordTiming] = field(default_factory=list)
    char_timings: list["CharacterTiming"] = field(default_factory=list)


@dataclass
class TranscriptSegment:
    words: list[WordTiming]
    text: str = ""
    start: float = 0.0
    end: float = 0.0

    def __post_init__(self):
        if not self.text and self.words:
            self.text = " ".join(word.word for word in self.words).strip()
        if self.words and self.start == 0.0:
            self.start = self.words[0].start
        if self.words and self.end == 0.0:
            self.end = self.words[-1].end


@dataclass
class TranscriptDraft:
    segments: list[TranscriptSegment]
    language_info: LanguageDetectionResult | None = None
    provider: str = ""

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


@dataclass
class AlignedTranscript:
    segments: list[TranscriptSegment]
    provider: str = ""
    fully_aligned: bool = True
    unaligned_word_count: int = 0

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


@dataclass
class ChordEvent:
    label: str
    start: float
    end: float
    confidence: float = 0.0
    root: str = ""
    quality: str = ""


@dataclass
class SongAnalysis:
    bpm: float = 0.0
    time_signature: int = 4
    preview_window_seconds: float = 0.6
    provider: str = ""
    source_audio: str = ""
    beat_times: list[float] = field(default_factory=list)
    measure_times: list[float] = field(default_factory=list)
    original_key: str = ""
    target_key: str = ""
    transpose_semitones: int = 0
    original_chord_events: list[ChordEvent] = field(default_factory=list)
    chord_events: list[ChordEvent] = field(default_factory=list)
    chord_sheet_text: str = ""
    chord_source_name: str = ""
    chord_source_url: str = ""


@dataclass
class SingerProfile:
    singer_id: str
    label: str
    primary_color: str = ""
    secondary_color: str = ""
    outline_color: str = ""
    shadow_color: str = ""
    lane_index: int = 0


@dataclass
class SingerSegmentAssignment:
    segment_index: int
    singer_id: str
    label: str = ""
    confidence: float = 0.0


@dataclass
class SingerAnalysisResult:
    detected_singer_count: int = 1
    provider: str = ""
    profiles: list[SingerProfile] = field(default_factory=list)
    assignments: list[SingerSegmentAssignment] = field(default_factory=list)
    low_confidence_segments: int = 0
    analysis_window_seconds: float = 0.0


@dataclass
class KaraokeStyle:
    font_name: str = "Arial"
    font_size: int = 62
    primary_color: str = "&H00FFFFFF"
    secondary_color: str = "&H0000FFFF"
    outline_color: str = "&H00000000"
    shadow_color: str = "&H80000000"
    outline_width: int = 2
    shadow_depth: int = 1
    border_style: int = 1
    alignment: int = 2
    margin_v: int = 52
    margin_l: int = 28
    margin_r: int = 28
    bold: int = -1
    encoding: int = -1
    effect_mode: str = "sweep"
    max_words_per_line: int = 4
    max_chars_per_line: int = 26
    line_height_scale: float = 1.35
    word_spacing_scale: float = 1.35
    pause_gap_min_seconds: float = 0.14
    pause_gap_multiplier: float = 1.75
    grapheme_fade_ms: int = 45
    blur: float = 0.0
    base_fill_alpha: str = "55"
    active_fill_alpha: str = "00"


@dataclass
class VideoRequest:
    with_vocals: bool = False
    without_vocals: bool = False
    quality: str = "best"


@dataclass
class JobManifest:
    job_id: str
    title: str = ""
    source_url: str = ""
    input_type: str = ""
    has_video: bool = False
    thumbnail_url: str = ""
    chat_id: int = 0
    user_id: int = 0
    delivery_chat_id: int = 0
    delivery_reply_to_message_id: int = 0
    status: str = JobStatus.CREATED.value
    review_status: str = ReviewStatus.NOT_STARTED.value
    style_preset: str = "blue_outline"
    requested_outputs: dict[str, object] = field(default_factory=dict)
    pending_delivery: dict[str, object] = field(default_factory=dict)
    quality_feedback: list[dict[str, object]] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)
    timing_provider: str = ""
    timing_quality: dict[str, object] = field(default_factory=dict)
    post_review_diff: dict[str, object] = field(default_factory=dict)
    language_info: dict[str, object] = field(default_factory=dict)
    lyrics_verification: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    review_message_id: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Job:
    job_id: str
    job_dir: Path
    manifest: JobManifest

    @property
    def title(self) -> str:
        return self.manifest.title

    @title.setter
    def title(self, value: str):
        self.manifest.title = value

    @property
    def display_name(self) -> str:
        title = (self.manifest.title or "").strip()
        return title or self.job_id

    @property
    def source_url(self) -> str:
        return self.manifest.source_url

    @property
    def input_type(self) -> str:
        return self.manifest.input_type

    @property
    def has_video(self) -> bool:
        return self.manifest.has_video

    @has_video.setter
    def has_video(self, value: bool):
        self.manifest.has_video = value

    @property
    def thumbnail_url(self) -> str:
        return self.manifest.thumbnail_url

    @thumbnail_url.setter
    def thumbnail_url(self, value: str):
        self.manifest.thumbnail_url = value

    @property
    def status(self) -> JobStatus:
        return JobStatus(self.manifest.status)

    @status.setter
    def status(self, value: JobStatus):
        self.manifest.status = value.value

    @property
    def review_status(self) -> ReviewStatus:
        return ReviewStatus(self.manifest.review_status)

    @review_status.setter
    def review_status(self, value: ReviewStatus):
        self.manifest.review_status = value.value

    @property
    def review_message_id(self) -> int:
        return self.manifest.review_message_id

    @review_message_id.setter
    def review_message_id(self, value: int):
        self.manifest.review_message_id = value

    @property
    def delivery_chat_id(self) -> int:
        return self.manifest.delivery_chat_id or self.manifest.chat_id

    @delivery_chat_id.setter
    def delivery_chat_id(self, value: int):
        self.manifest.delivery_chat_id = value

    @property
    def delivery_reply_to_message_id(self) -> int:
        return self.manifest.delivery_reply_to_message_id

    @delivery_reply_to_message_id.setter
    def delivery_reply_to_message_id(self, value: int):
        self.manifest.delivery_reply_to_message_id = value

    @property
    def pending_delivery(self) -> dict[str, object]:
        return self.manifest.pending_delivery or {}

    @pending_delivery.setter
    def pending_delivery(self, value: dict[str, object]):
        self.manifest.pending_delivery = dict(value or {})

    @property
    def quality_feedback(self) -> list[dict[str, object]]:
        return list(self.manifest.quality_feedback or [])

    @quality_feedback.setter
    def quality_feedback(self, value: list[dict[str, object]]):
        self.manifest.quality_feedback = list(value or [])

    @property
    def manifest_path(self) -> Path:
        return self.job_dir / "job.json"

    @property
    def original_audio_path(self) -> Path:
        return self.job_dir / "original_audio.mp3"

    @property
    def original_video_path(self) -> Path:
        return self.job_dir / "original_video.mp4"

    @property
    def vocals_path(self) -> Path:
        return self.job_dir / "vocals.wav"

    @property
    def instrumental_path(self) -> Path:
        return self.job_dir / "instrumental.mp3"

    @property
    def vocals_16k_path(self) -> Path:
        return self.job_dir / "vocals_16k.wav"

    @property
    def language_sample_path(self) -> Path:
        return self.job_dir / "language_sample.wav"

    @property
    def draft_transcript_path(self) -> Path:
        return self.job_dir / "draft_transcript.txt"

    @property
    def draft_timings_path(self) -> Path:
        return self.job_dir / "draft_timings.json"

    @property
    def review_transcript_path(self) -> Path:
        return self.job_dir / "review_transcript.txt"

    @property
    def review_timings_path(self) -> Path:
        return self.job_dir / "review_timings.json"

    @property
    def transcript_path(self) -> Path:
        return self.job_dir / "transcript.txt"

    @property
    def timings_path(self) -> Path:
        return self.job_dir / "timings.json"

    @property
    def ass_path(self) -> Path:
        return self.job_dir / "karaoke.ass"

    @property
    def srt_path(self) -> Path:
        return self.job_dir / "subtitles.srt"

    @property
    def thumbnail_path(self) -> Path:
        return self.job_dir / "thumbnail.jpg"

    @property
    def video_vocals_path(self) -> Path:
        return self.job_dir / "final_video.mp4"

    @property
    def video_instrumental_path(self) -> Path:
        return self.job_dir / "final_video_instrumental.mp4"

    @property
    def song_analysis_path(self) -> Path:
        return self.job_dir / "song_analysis.json"

    @property
    def singer_analysis_path(self) -> Path:
        return self.job_dir / "singer_analysis.json"

    @property
    def lyrics_with_chords_path(self) -> Path:
        return self.job_dir / "lyrics_with_chords.txt"

    @property
    def delivery_feedback_path(self) -> Path:
        return self.job_dir / "delivery_feedback.txt"

    @property
    def delivery_feedback_template_path(self) -> Path:
        return self.job_dir / "delivery_feedback_template.txt"
