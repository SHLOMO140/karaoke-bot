"""Karaoke pipeline orchestrator."""

from __future__ import annotations

import inspect
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Callable

import yt_dlp

from . import job_manager
from .aligner import analyze_alignment_quality, get_alignment_provider, validate_timing_quality
from .audio_extractor import (
    convert_to_wav,
    extract_audio_from_video,
    get_audio_duration,
    get_video_frame_rate,
    transcode_to_mp3,
)
from .chord_sources import lookup_external_chord_sheet, lookup_external_chord_sheet_by_title
from .config import DEFAULT_VIDEO_FRAME_RATE, FFMPEG_PATH, YTDLP_STAGING_DIR, ytdlp_base_opts
from .exceptions import AlignmentError, AudioExtractionError, DownloadError, PipelineError
from .harmony import (
    LibrosaHarmonyAnalyzer,
    prepare_song_analysis_for_display,
    render_chord_sheet_text,
    summarize_song_analysis_quality,
)
from .job_manager import add_warning
from .language_detector import WhisperLanguageDetector
from .lyrics_verifier import HybridLyricsVerifier, MultiStepLyricsVerifier
from .models import (
    Job,
    JobStatus,
    LanguageDetectionResult,
    LyricsVerificationResult,
    SingerAnalysisResult,
    SongAnalysis,
    TranscriptDraft,
    TranscriptSegment,
    VideoRequest,
)
from .singer_analysis import StructureDuetAnalyzer
from .styles import get_style
from .subtitle_guardian import AssSubtitleGuardian
from .subtitle_generator import AssKaraokeRenderer, SrtRenderer
from .transcriber import FasterWhisperHebrewProvider
from .video_renderer import burn_subtitles, compress_video_if_needed, create_static_video
from .vocal_separator import DemucsSeparator

logger = logging.getLogger(__name__)
StatusCallback = Callable[[JobStatus], None]


class KaraokePipeline:
    def __init__(
        self,
        job: Job,
        status_callback: StatusCallback | None = None,
        separator=None,
        language_detector=None,
        transcriber=None,
        lyrics_verifier=None,
        aligner=None,
        song_analyzer=None,
        singer_analyzer=None,
        srt_renderer=None,
        ass_renderer=None,
        subtitle_validator=None,
    ):
        self.job = job
        self._status_callback = status_callback or (lambda status: None)
        self.separator = separator or DemucsSeparator()
        self.language_detector = language_detector or WhisperLanguageDetector()
        self.transcriber = transcriber or FasterWhisperHebrewProvider()
        self.lyrics_verifier = lyrics_verifier or MultiStepLyricsVerifier()
        self.aligner = aligner or get_alignment_provider()
        self.song_analyzer = song_analyzer or LibrosaHarmonyAnalyzer()
        self.singer_analyzer = singer_analyzer or StructureDuetAnalyzer()
        self.srt_renderer = srt_renderer or SrtRenderer()
        self.ass_renderer = ass_renderer or AssKaraokeRenderer()
        self.subtitle_validator = subtitle_validator or AssSubtitleGuardian()

        job_manager.record_provider(self.job, "separator", self.separator.name)
        job_manager.record_provider(self.job, "language_detector", self.language_detector.name)
        job_manager.record_provider(self.job, "transcriber", self.transcriber.name)
        job_manager.record_provider(self.job, "lyrics_verifier", self.lyrics_verifier.name)
        job_manager.record_provider(self.job, "aligner", self.aligner.name)
        job_manager.record_provider(self.job, "song_analyzer", self.song_analyzer.name)
        job_manager.record_provider(self.job, "singer_analyzer", self.singer_analyzer.name)
        job_manager.record_provider(self.job, "srt_renderer", self.srt_renderer.name)
        job_manager.record_provider(self.job, "ass_renderer", self.ass_renderer.name)
        job_manager.record_provider(self.job, "subtitle_validator", self.subtitle_validator.name)

    def _update_status(self, status: JobStatus):
        job_manager.update_status(self.job, status)
        self._status_callback(status)

    def _ensure_vocals_wav(self, vocals_path: str) -> str:
        if not self.job.vocals_16k_path.exists():
            convert_to_wav(vocals_path, str(self.job.vocals_16k_path))
        return str(self.job.vocals_16k_path)

    def step_get_audio(self, input_path: str | None = None) -> str:
        return self._get_audio(input_path)

    def step_separate_vocals(self, audio_path: str) -> tuple[str, str]:
        if self.job.vocals_path.exists() and self.job.instrumental_path.exists():
            return str(self.job.vocals_path), str(self.job.instrumental_path)
        self._update_status(JobStatus.SEPARATING_VOCALS)
        return self.separator.separate(audio_path, self.job.job_dir)

    def step_detect_language(self, vocals_path: str):
        if self.job.manifest.language_info:
            return LanguageDetectionResult(**self.job.manifest.language_info)
        self._update_status(JobStatus.DETECTING_LANGUAGE)
        vocals_wav = self._ensure_vocals_wav(vocals_path)
        result = self.language_detector.detect(vocals_wav, self.job.job_dir)
        job_manager.save_language_info(self.job, result)
        if result.warning_message:
            add_warning(self.job, result.warning_message)
        return result

    def step_transcribe(self, vocals_path: str, language_info=None) -> TranscriptDraft:
        if self.job.draft_timings_path.exists():
            draft = TranscriptDraft(
                segments=job_manager.load_draft_segments(self.job),
                language_info=language_info,
                provider=self.job.manifest.providers.get("transcriber", ""),
            )
            return draft
        self._update_status(JobStatus.TRANSCRIBING)
        vocals_wav = self._ensure_vocals_wav(vocals_path)
        transcribe_kwargs = self._build_transcription_kwargs(language_info)
        draft = self.transcriber.transcribe(vocals_wav, **transcribe_kwargs)
        draft.language_info = language_info
        job_manager.save_draft_transcript(self.job, draft)
        return draft

    def _build_transcription_kwargs(self, language_info: LanguageDetectionResult | None) -> dict[str, object]:
        try:
            parameters = inspect.signature(self.transcriber.transcribe).parameters
        except (TypeError, ValueError, AttributeError):
            return {}

        if "language" not in parameters:
            return {}

        # Let Whisper auto-detect on Hebrew-first mixed tracks so English phrases
        # are not forced through a Hebrew-only language hint.
        if language_info and language_info.policy_decision == "warn":
            return {"language": None}
        return {}

    def step_verify_lyrics(self, draft: TranscriptDraft):
        if self.job.manifest.lyrics_verification:
            verification = LyricsVerificationResult(**self.job.manifest.lyrics_verification)
            if verification.applied and verification.corrected_lines and not self.job.review_timings_path.exists():
                corrected_segments = job_manager.update_transcript_text(draft.segments, "\n".join(verification.corrected_lines))
                job_manager.save_review_transcript(self.job, corrected_segments)
            return verification
        self._update_status(JobStatus.VERIFYING_LYRICS)
        previous_source_url = getattr(self.lyrics_verifier, "_current_source_url", "")
        setattr(self.lyrics_verifier, "_current_source_url", self.job.source_url)
        try:
            verification = self.lyrics_verifier.verify(self.job.title, draft)
        finally:
            setattr(self.lyrics_verifier, "_current_source_url", previous_source_url)
        job_manager.save_lyrics_verification(self.job, verification)
        if verification.applied and verification.corrected_lines:
            corrected_segments = job_manager.update_transcript_text(draft.segments, "\n".join(verification.corrected_lines))
            job_manager.save_review_transcript(self.job, corrected_segments)
            add_warning(self.job, f"בוצעו {verification.correction_count} תיקוני מילים אוטומטיים לפני ה-review.")
        for warning in verification.local_warnings:
            add_warning(self.job, warning)
        return verification

    def step_analyze_music(self, segments: list[TranscriptSegment]) -> SongAnalysis:
        source_audio = self.job.instrumental_path if self.job.instrumental_path.exists() else self.job.original_audio_path
        delivery_target_key = ""
        if self.job.song_analysis_path.exists():
            cached_analysis = job_manager.load_song_analysis(self.job)
            if cached_analysis.provider == self.song_analyzer.name or not source_audio.exists():
                cached_quality = summarize_song_analysis_quality(cached_analysis)
                uses_transposed_delivery_key = bool((cached_analysis.target_key or "").strip())
                if source_audio.exists() and not cached_quality.reliable_for_delivery:
                    logger.info(
                        "Re-running music analysis for %s because cached chord analysis is unreliable.",
                        self.job.job_id,
                    )
                elif uses_transposed_delivery_key:
                    logger.info(
                        "Re-running music analysis for %s because cached chord delivery is transposed to %s.",
                        self.job.job_id,
                        cached_analysis.target_key,
                    )
                else:
                    if cached_analysis.chord_sheet_text.strip():
                        job_manager.save_song_analysis(self.job, cached_analysis)
                        job_manager.save_chord_sheet(self.job, cached_analysis.chord_sheet_text)
                        return cached_analysis

                    analysis = prepare_song_analysis_for_display(cached_analysis, segments, target_key=delivery_target_key)
                    job_manager.save_song_analysis(self.job, analysis)
                    job_manager.save_chord_sheet(self.job, render_chord_sheet_text(self.job.display_name, segments, analysis))
                    return analysis
            if cached_analysis.provider != self.song_analyzer.name:
                logger.info(
                    "Re-running music analysis for %s because cached provider %s is outdated.",
                    self.job.job_id,
                    cached_analysis.provider,
                )

        try:
            external_analysis = lookup_external_chord_sheet(
                self.job.display_name,
                segments,
                provider=self.song_analyzer.name,
                source_audio=str(source_audio) if source_audio.exists() else "",
                target_key=delivery_target_key,
            )
        except Exception as exc:
            logger.info("External chord lookup failed for %s: %s", self.job.job_id, exc)
            external_analysis = None
        if external_analysis is not None and external_analysis.chord_sheet_text.strip():
            job_manager.save_song_analysis(self.job, external_analysis)
            job_manager.save_chord_sheet(self.job, external_analysis.chord_sheet_text)
            return external_analysis

        lookup_title = (self.job.display_name or "").strip()
        should_try_title_only_lookup = bool(lookup_title) and (
            " - " in lookup_title or "|" in lookup_title or len(lookup_title.split()) >= 2
        )
        title_only_analysis = None
        if should_try_title_only_lookup:
            try:
                title_only_analysis = lookup_external_chord_sheet_by_title(
                    lookup_title,
                    provider=self.song_analyzer.name,
                    source_audio=str(source_audio) if source_audio.exists() else "",
                    target_key=delivery_target_key,
                )
            except Exception as exc:
                logger.info("Title-only external chord lookup failed for %s: %s", self.job.job_id, exc)
        if title_only_analysis is not None and title_only_analysis.chord_sheet_text.strip():
            job_manager.save_song_analysis(self.job, title_only_analysis)
            job_manager.save_chord_sheet(self.job, title_only_analysis.chord_sheet_text)
            return title_only_analysis

        if not source_audio.exists():
            analysis = prepare_song_analysis_for_display(
                SongAnalysis(provider=self.song_analyzer.name),
                segments,
                target_key=delivery_target_key,
            )
            job_manager.save_song_analysis(self.job, analysis)
            job_manager.save_chord_sheet(self.job, render_chord_sheet_text(self.job.display_name, segments, analysis))
            return analysis

        try:
            analysis = self.song_analyzer.analyze(str(source_audio))
        except Exception as exc:
            logger.warning("Music analysis failed for %s: %s", self.job.job_id, exc)
            add_warning(self.job, "לא הצלחתי לזהות BPM ואקורדים, אז המשכתי בלי שכבת אקורדים מלאה.")
            analysis = SongAnalysis(provider=self.song_analyzer.name, source_audio=str(source_audio))

        analysis = prepare_song_analysis_for_display(analysis, segments, target_key=delivery_target_key)
        job_manager.save_song_analysis(self.job, analysis)
        job_manager.save_chord_sheet(self.job, render_chord_sheet_text(self.job.display_name, segments, analysis))
        return analysis

    def step_analyze_singers(self, segments: list[TranscriptSegment]) -> SingerAnalysisResult:
        if self.job.singer_analysis_path.exists():
            try:
                existing = job_manager.load_singer_analysis(self.job)
            except Exception as exc:
                logger.warning("Could not read singer analysis for %s: %s", self.job.job_id, exc)
            else:
                if existing.provider == self.singer_analyzer.name and len(existing.assignments) == len(segments):
                    return existing

        if self.job.vocals_16k_path.exists():
            singer_audio = str(self.job.vocals_16k_path)
        elif self.job.vocals_path.exists():
            singer_audio = self._ensure_vocals_wav(str(self.job.vocals_path))
        else:
            return SingerAnalysisResult(provider=self.singer_analyzer.name)

        try:
            analysis = self.singer_analyzer.analyze(singer_audio, segments, title=self.job.display_name)
        except Exception as exc:
            logger.warning("Singer analysis failed for %s: %s", self.job.job_id, exc)
            add_warning(self.job, "לא הצלחתי להבחין בבטחה בין כמה זמרים, אז הקריוקי ירונדר בפריסה רגילה.")
            return SingerAnalysisResult(provider=self.singer_analyzer.name)

        if analysis.assignments:
            job_manager.save_singer_analysis(self.job, analysis)
        return analysis

    def step_post_review(self, job: Job, original_draft: TranscriptDraft, aligned_segments=None):
        """Run Steps 5-7 after human approval (char diff, LLM validate, timing fix)."""
        if hasattr(self.lyrics_verifier, 'post_review_steps'):
            self.lyrics_verifier.post_review_steps(job, original_draft, aligned_segments)

    def can_realign_after_review(self) -> bool:
        return self.job.vocals_16k_path.exists() or self.job.vocals_path.exists()

    def run_until_review(self, input_path: str | None = None) -> TranscriptDraft:
        audio_path = self.step_get_audio(input_path)
        vocals_path, _instrumental = self.step_separate_vocals(audio_path)
        language_info = self.step_detect_language(vocals_path)
        draft = self.step_transcribe(vocals_path, language_info=language_info)
        self.step_verify_lyrics(draft)
        self._update_status(JobStatus.AWAITING_REVIEW)
        return draft

    def run_after_review(self, approved_segments: list[TranscriptSegment], video_request: VideoRequest | None = None):
        draft_segments = job_manager.load_draft_segments(self.job)
        review_mismatches_before = job_manager.find_segment_word_text_mismatches(approved_segments)
        approved_segments = job_manager.rebuild_segments_from_authoritative_text(draft_segments, approved_segments)
        review_mismatches_after = job_manager.find_segment_word_text_mismatches(approved_segments)
        if review_mismatches_before and not review_mismatches_after:
            add_warning(
                self.job,
                (
                    "זוהתה אי-התאמה בין טקסט review לבין המילים המתוזמנות, "
                    "ובוצע תיקון אוטומטי לפני היישור הסופי."
                ),
            )
        if review_mismatches_after:
            raise PipelineError(
                "Approved review text could not be reconciled with the timed words.",
                "זיהיתי חוסר התאמה בין טקסט האישור לבין המילים המתוזמנות, וגם אחרי תיקון אוטומטי זה לא הסתדר. "
                "כדאי לשלוח שוב את הטקסט המלא של השיר או לעדכן את השורות הבעייתיות.",
            )
        job_manager.save_review_transcript(self.job, approved_segments)
        shrink = job_manager.detect_suspicious_review_shrink(
            draft_segments,
            "\n".join(segment.text for segment in approved_segments if segment.text.strip()),
        )
        if shrink is not None:
            raise PipelineError(
                "Approved review text is suspiciously short compared to the draft transcript.",
                "הטקסט שאושר קצר מדי ביחס לשיר, ולכן חלקים יישארו בלי כתוביות. "
                "שלח את כל מילות השיר או תקן שורות בודדות לפני יצירת הקריוקי.",
            )
        render_frame_rate = self._resolve_render_frame_rate(video_request)
        if self.job.vocals_16k_path.exists():
            alignment_audio = str(self.job.vocals_16k_path)
        elif self.job.vocals_path.exists():
            alignment_audio = self._ensure_vocals_wav(str(self.job.vocals_path))
        else:
            raise AlignmentError(
                "No isolated vocals are available for review realignment.",
                "אין קובץ ווקאל זמין כדי ליישר מחדש את הטקסט המתוקן.",
            )

        self._update_status(JobStatus.ALIGNING)
        aligned = self.aligner.align(
            alignment_audio,
            approved_segments,
            draft_segments,
            video_frame_rate=render_frame_rate,
        )
        aligner_warning = getattr(self.aligner, "last_warning_message", "")
        if aligner_warning:
            add_warning(self.job, aligner_warning)
        if aligned.unaligned_word_count:
            add_warning(self.job, f"{aligned.unaligned_word_count} מילים יושרו באינטרפולציה ולא ביישור ישיר.")
        for timing_warning in validate_timing_quality(aligned.segments):
            add_warning(self.job, timing_warning)
        quality_report = analyze_alignment_quality(aligned.segments)
        timing_provider = getattr(self.aligner, "last_provider_used", "") or self.aligner.name
        self.job.manifest.timing_provider = timing_provider
        self.job.manifest.timing_quality = {
            "score": float(quality_report.get("score", 0.0)),
            "aligned_ratio": float(quality_report.get("aligned_ratio", 0.0)),
            "char_timing_ratio": float(quality_report.get("char_timing_ratio", 0.0)),
            "critical": bool(quality_report.get("critical", False)),
            "warning_count": int(quality_report.get("warning_count", 0)),
        }
        if quality_report["critical"]:
            raise AlignmentError(
                f"Take-one quality gate failed: {quality_report}",
                "לא הצלחתי להגיע לטיימינג ברמת טייק אחד אחרי כל ניסיונות היישור, אז עצרתי לפני יצירת קבצים לא מדויקים.",
            )
        if float(quality_report["score"]) < 0.93:
            add_warning(
                self.job,
                (
                    "איכות היישור טובה אבל עדיין לא מושלמת: "
                    f"דיוק מילים {int(float(quality_report['aligned_ratio']) * 100)}%, "
                    f"דיוק אותיות {int(float(quality_report['char_timing_ratio']) * 100)}%."
                ),
            )
        try:
            self.step_post_review(
                self.job,
                TranscriptDraft(segments=draft_segments, provider=""),
                aligned.segments,
            )
        except Exception as exc:
            logger.warning("Post-review steps failed for %s: %s", self.job.job_id, exc)
        job_manager.save_final_transcript(self.job, aligned)
        analysis = self.step_analyze_music(aligned.segments)
        singer_analysis = self.step_analyze_singers(aligned.segments)

        self._update_status(JobStatus.GENERATING_SUBS)
        style = get_style(self.job.manifest.style_preset)
        self.srt_renderer.render(
            aligned.segments,
            str(self.job.srt_path),
            style=style,
            singer_analysis=singer_analysis,
        )
        self.ass_renderer.render(
            aligned.segments,
            str(self.job.ass_path),
            style=style,
            song_analysis=analysis,
            singer_analysis=singer_analysis,
            include_chord_overlays=False,
            include_hud=False,
            include_next_line_preview=True,
        )
        for warning in self.subtitle_validator.validate(aligned.segments, style, singer_analysis=singer_analysis):
            add_warning(self.job, warning)

        if video_request and (video_request.with_vocals or video_request.without_vocals):
            self._update_status(JobStatus.RENDERING_VIDEO)
            self._render_videos(video_request)

        self._update_status(JobStatus.DONE)
        return job_manager.get_output_files(self.job, video_request=video_request)

    def rerender_existing_outputs(self, video_request: VideoRequest | None = None):
        segments = job_manager.get_best_available_segments(self.job)
        if not segments:
            raise DownloadError("No saved timings available for rerender.")

        analysis = self.step_analyze_music(segments)
        singer_analysis = self.step_analyze_singers(segments)
        self._update_status(JobStatus.GENERATING_SUBS)
        style = get_style(self.job.manifest.style_preset)
        self.srt_renderer.render(
            segments,
            str(self.job.srt_path),
            style=style,
            singer_analysis=singer_analysis,
        )
        self.ass_renderer.render(
            segments,
            str(self.job.ass_path),
            style=style,
            song_analysis=analysis,
            singer_analysis=singer_analysis,
            include_chord_overlays=False,
            include_hud=False,
            include_next_line_preview=True,
        )
        for warning in self.subtitle_validator.validate(segments, style, singer_analysis=singer_analysis):
            add_warning(self.job, warning)

        if video_request and (video_request.with_vocals or video_request.without_vocals):
            self._update_status(JobStatus.RENDERING_VIDEO)
            self._render_videos(video_request)

        self._update_status(JobStatus.DONE)
        return job_manager.get_output_files(self.job, video_request=video_request)

    def _resolve_render_frame_rate(self, video_request: VideoRequest | None = None) -> float | None:
        wants_video = bool(video_request and (video_request.with_vocals or video_request.without_vocals))
        if not wants_video:
            return None

        if self.job.has_video and self.job.original_video_path.exists():
            try:
                return get_video_frame_rate(str(self.job.original_video_path))
            except Exception as exc:
                logger.warning("Falling back to default frame rate for %s: %s", self.job.job_id, exc)
                add_warning(self.job, "לא הצלחתי לקרוא את קצב הפריימים המקורי, אז השתמשתי ב-25fps ליישור.")

        return DEFAULT_VIDEO_FRAME_RATE

    def _get_stage_dir(self, media_kind: str) -> Path:
        stage_dir = YTDLP_STAGING_DIR / self.job.job_id / media_kind
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def _cleanup_stage_dir(self, stage_dir: Path):
        shutil.rmtree(stage_dir, ignore_errors=True)

    def _cleanup_download_artifacts(self, prefix: str, stage_dir: Path | None = None):
        target_dir = stage_dir or self.job.job_dir
        for path in target_dir.glob(f"{prefix}*"):
            if path.is_file():
                path.unlink(missing_ok=True)

    def _run_ytdlp_with_retry(self, ydl_opts: dict, prefix: str, stage_dir: Path | None = None):
        self._cleanup_download_artifacts(prefix, stage_dir=stage_dir)
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(self.job.source_url, download=True)
            except Exception as exc:
                last_error = exc
                logger.warning("yt-dlp attempt failed for %s (%s): %s", self.job.job_id, prefix, exc)
                self._cleanup_download_artifacts(prefix, stage_dir=stage_dir)
        raise DownloadError(str(last_error) if last_error else "")

    def _pick_download_candidate(self, stage_dir: Path, prefix: str) -> Path | None:
        candidates = [
            path
            for path in stage_dir.glob(f"{prefix}.*")
            if path.is_file() and path.suffix.lower() not in {".part", ".tmp", ".ytdl"}
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.name)
        return candidates[0]

    def _get_audio(self, input_path: str | None) -> str:
        if self.job.original_audio_path.exists():
            return str(self.job.original_audio_path)

        if input_path:
            if self.job.input_type == "video_file":
                self._update_status(JobStatus.EXTRACTING_AUDIO)
                audio_out = str(self.job.original_audio_path)
                extract_audio_from_video(input_path, audio_out)
                return audio_out

            audio_out = str(self.job.original_audio_path)
            if Path(input_path).suffix.lower() == ".mp3":
                Path(input_path).replace(self.job.original_audio_path)
            else:
                self._update_status(JobStatus.EXTRACTING_AUDIO)
                transcode_to_mp3(input_path, audio_out)
            return audio_out

        self._update_status(JobStatus.DOWNLOADING)
        return self._download_youtube_audio()

    def _download_youtube_audio(self) -> str:
        output_path = str(self.job.original_audio_path)
        stage_dir = self._get_stage_dir("audio")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str((stage_dir / "yt_audio.%(ext)s").resolve()),
            "ffmpeg_location": str(FFMPEG_PATH),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "retries": 5,
            "fragment_retries": 5,
            "windowsfilenames": True,
            **ytdlp_base_opts(),
        }

        try:
            info = self._run_ytdlp_with_retry(ydl_opts, "yt_audio", stage_dir=stage_dir)
            self.job.title = info.get("title", self.job.title)

            source_file = self._pick_download_candidate(stage_dir, "yt_audio")
            if not source_file:
                raise DownloadError("YouTube audio download did not produce an audio file.")

            transcode_to_mp3(str(source_file), output_path)
            self._download_thumbnail()
            job_manager.save_job(self.job)
            return output_path
        except AudioExtractionError:
            raise
        except DownloadError:
            raise
        except Exception as exc:
            raise DownloadError(str(exc), "הורדת האודיו מיוטיוב נכשלה.") from exc
        finally:
            self._cleanup_stage_dir(stage_dir)

    def _download_thumbnail(self):
        if not self.job.thumbnail_url and self.job.source_url:
            url = self.job.source_url
            if "youtube.com/watch?v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            else:
                return
            self.job.thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        if not self.job.thumbnail_url:
            return

        try:
            urllib.request.urlretrieve(self.job.thumbnail_url, str(self.job.thumbnail_path))
        except Exception:
            fallback = self.job.thumbnail_url.replace("maxresdefault.jpg", "hqdefault.jpg")
            try:
                urllib.request.urlretrieve(fallback, str(self.job.thumbnail_path))
                self.job.thumbnail_url = fallback
            except Exception as exc:
                logger.warning("Thumbnail download failed for %s: %s", self.job.job_id, exc)

    def download_youtube_video(self, quality: str = "best"):
        stage_dir = self._get_stage_dir("video")
        if quality == "best":
            fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        else:
            fmt = f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality}][ext=mp4]/best"

        ydl_opts = {
            "format": fmt,
            "outtmpl": str((stage_dir / "yt_video.%(ext)s").resolve()),
            "ffmpeg_location": str(FFMPEG_PATH),
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "retries": 5,
            "fragment_retries": 5,
            "windowsfilenames": True,
            **ytdlp_base_opts(),
        }

        try:
            self._run_ytdlp_with_retry(ydl_opts, "yt_video", stage_dir=stage_dir)
        except DownloadError as exc:
            raise DownloadError(exc.info.technical_message, "הורדת הווידאו מיוטיוב נכשלה.") from exc

        try:
            candidate = self._pick_download_candidate(stage_dir, "yt_video")
            if not candidate:
                raise DownloadError("YouTube video download failed to create a file.", "הווידאו לא נוצר לאחר ההורדה.")

            candidate.replace(self.job.original_video_path)
            self.job.has_video = True
            job_manager.save_job(self.job)
        finally:
            self._cleanup_stage_dir(stage_dir)

    def _render_videos(self, video_request: VideoRequest):
        ass_path = str(self.job.ass_path)
        if self.job.has_video and self.job.original_video_path.exists():
            base_video = str(self.job.original_video_path)
        elif self.job.thumbnail_path.exists():
            base_video = str(self.job.job_dir / "base_static.mp4")
            create_static_video(str(self.job.thumbnail_path), str(self.job.original_audio_path), base_video)
        else:
            logger.warning("No video source or thumbnail for job %s", self.job.job_id)
            return

        try:
            duration = get_audio_duration(base_video)
        except Exception:
            duration = 240

        if video_request.with_vocals:
            output = str(self.job.video_vocals_path)
            burn_subtitles(base_video, ass_path, output)
            compress_video_if_needed(output, duration)

        if video_request.without_vocals and self.job.instrumental_path.exists():
            output = str(self.job.video_instrumental_path)
            burn_subtitles(base_video, ass_path, output, audio_path=str(self.job.instrumental_path))
            compress_video_if_needed(output, duration)
