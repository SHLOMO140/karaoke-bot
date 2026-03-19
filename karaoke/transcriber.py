"""Hebrew draft transcription using faster-whisper."""

from __future__ import annotations

import logging

from .config import WHISPER_BEAM_SIZE, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_HEBREW_MODEL, WHISPER_LANGUAGE
from .exceptions import TranscriptionError
from .models import CharacterTiming, TranscriptDraft, TranscriptSegment, WordTiming

logger = logging.getLogger(__name__)
_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Hebrew whisper model %s (device=%s compute=%s)",
            WHISPER_HEBREW_MODEL,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            WHISPER_HEBREW_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _model


class FasterWhisperHebrewProvider:
    name = "faster_whisper_hebrew"

    def transcribe(self, audio_path: str) -> TranscriptDraft:
        model = _get_model()
        try:
            segments_gen, _info = model.transcribe(
                audio_path,
                language=WHISPER_LANGUAGE,
                beam_size=WHISPER_BEAM_SIZE,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc

        segments = []
        for segment in segments_gen:
            words = [
                WordTiming(
                    word=word.word.strip(),
                    start=float(word.start),
                    end=float(word.end),
                    confidence=float(getattr(word, "probability", 0.0)),
                    source="draft_whisper",
                    aligned=False,
                )
                for word in (segment.words or [])
                if word.word and word.word.strip()
            ]
            if not words and segment.text.strip():
                words = [
                    WordTiming(
                        word=segment.text.strip(),
                        start=float(segment.start),
                        end=float(segment.end),
                        confidence=0.0,
                        source="draft_whisper",
                        aligned=False,
                    )
                ]
            if words:
                segments.append(
                    TranscriptSegment(
                        words=words,
                        text=" ".join(word.word for word in words),
                        start=words[0].start,
                        end=words[-1].end,
                    )
                )

        if not segments:
            raise TranscriptionError("No words were detected in the vocal track.", "לא זוהו מילים באודיו.")
        return TranscriptDraft(segments=segments, provider=self.name)


def interpolate_character_timings(word: WordTiming) -> list[CharacterTiming]:
    """Interpolate character-level timing from word timing using grapheme weights.

    Uses the same grapheme weight logic as aligner.py: Hebrew niqqud marks
    receive lower weight than consonants for proportional time distribution.
    """
    from .aligner import _split_graphemes, _grapheme_weight

    graphemes = _split_graphemes(word.word)
    if not graphemes:
        return []

    weights = [_grapheme_weight(g) for g in graphemes]
    total_weight = sum(weights)
    if total_weight == 0:
        total_weight = len(graphemes)
        weights = [1.0] * len(graphemes)

    duration = word.end - word.start
    timings = []
    cursor = word.start

    for grapheme, weight in zip(graphemes, weights):
        char_duration = duration * (weight / total_weight)
        timings.append(CharacterTiming(
            char=grapheme,
            start=round(cursor, 4),
            end=round(cursor + char_duration, 4),
        ))
        cursor += char_duration

    # Snap last character end to word boundary
    if timings:
        timings[-1] = CharacterTiming(
            char=timings[-1].char,
            start=timings[-1].start,
            end=word.end,
        )

    return timings


def transcribe_hebrew(audio_path: str) -> list[TranscriptSegment]:
    return FasterWhisperHebrewProvider().transcribe(audio_path).segments
