"""Language detection for Hebrew-first karaoke jobs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .audio_extractor import create_audio_sample
from .config import (
    HEBREW_CONFIDENT_THRESHOLD,
    HEBREW_RATIO_WARNING_THRESHOLD,
    HEBREW_WARNING_THRESHOLD,
    LANGUAGE_SAMPLE_OFFSET_SECONDS,
    LANGUAGE_SAMPLE_SECONDS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DETECT_MODEL,
    WHISPER_DEVICE,
)
from .exceptions import LanguageDetectionError
from .models import LanguageDetectionResult

logger = logging.getLogger(__name__)
_detect_model = None


def _get_detection_model():
    global _detect_model
    if _detect_model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading language detection model %s (device=%s compute=%s)",
            WHISPER_DETECT_MODEL,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _detect_model = WhisperModel(
            WHISPER_DETECT_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _detect_model


def _hebrew_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z\u0590-\u05FF]", text)
    if not letters:
        return 0.0
    hebrew_letters = re.findall(r"[\u0590-\u05FF]", text)
    return len(hebrew_letters) / len(letters)


class WhisperLanguageDetector:
    name = "faster_whisper_detect"

    def detect(self, audio_path: str, job_dir: Path) -> LanguageDetectionResult:
        sample_path = job_dir / "language_sample.wav"
        create_audio_sample(
            audio_path,
            str(sample_path),
            duration=LANGUAGE_SAMPLE_SECONDS,
            offset=LANGUAGE_SAMPLE_OFFSET_SECONDS,
        )

        model = _get_detection_model()
        segments, info = model.transcribe(
            str(sample_path),
            beam_size=1,
            vad_filter=True,
            word_timestamps=False,
        )
        sample_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        hebrew_ratio = _hebrew_ratio(sample_text)

        if info.language == "he" and info.language_probability >= HEBREW_CONFIDENT_THRESHOLD:
            decision = "allow"
            warning = ""
        elif info.language == "he" or info.language_probability >= HEBREW_WARNING_THRESHOLD or hebrew_ratio >= HEBREW_RATIO_WARNING_THRESHOLD:
            decision = "warn"
            warning = (
                f"זוהתה שפה {info.language} בהסתברות {info.language_probability:.2f}. "
                "ממשיך במסלול Hebrew-first mixed, אבל ייתכן שתידרש בדיקה ידנית."
            )
        else:
            decision = "reject"
            warning = (
                f"זוהתה שפה {info.language} בהסתברות {info.language_probability:.2f}; "
                "לא זוהתה עברית דומיננטית מספיק."
            )

        result = LanguageDetectionResult(
            language=info.language,
            probability=float(info.language_probability),
            policy_decision=decision,
            warning_message=warning,
            hebrew_ratio=hebrew_ratio,
            sample_text=sample_text,
            provider=self.name,
        )

        if decision == "reject":
            raise LanguageDetectionError(warning)
        return result
