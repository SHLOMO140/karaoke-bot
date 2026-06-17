from types import SimpleNamespace

from karaoke import job_manager
from karaoke.models import LanguageDetectionResult, TranscriptDraft, TranscriptSegment, WordTiming
from karaoke.pipeline import KaraokePipeline
from karaoke.transcriber import FasterWhisperHebrewProvider


class _DummyProvider:
    def __init__(self, name: str):
        self.name = name


def _draft(provider: str = "test") -> TranscriptDraft:
    return TranscriptDraft(
        segments=[
            TranscriptSegment(
                words=[WordTiming("hello", 0.0, 0.4, confidence=0.9)],
                text="hello",
                start=0.0,
                end=0.4,
            )
        ],
        provider=provider,
    )


def _make_pipeline(tmp_path, monkeypatch, *, transcriber):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(
        title="song",
        source_url="https://www.youtube.com/watch?v=test123",
        input_type="youtube",
        chat_id=1,
        user_id=1,
    )
    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=transcriber,
        lyrics_verifier=_DummyProvider("lyrics_verifier"),
        aligner=_DummyProvider("aligner"),
        srt_renderer=_DummyProvider("srt_renderer"),
        ass_renderer=_DummyProvider("ass_renderer"),
    )
    monkeypatch.setattr(pipeline, "_ensure_vocals_wav", lambda path: path)
    return pipeline


def test_faster_whisper_provider_omits_language_hint_when_override_is_none(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeModel:
        def transcribe(self, audio_path, **kwargs):
            captured["audio_path"] = audio_path
            captured["kwargs"] = kwargs
            segment = SimpleNamespace(
                words=[SimpleNamespace(word="hello", start=0.0, end=0.4, probability=0.9)],
                text="hello",
                start=0.0,
                end=0.4,
            )
            return [segment], SimpleNamespace(language="en")

    monkeypatch.setattr("karaoke.transcriber._get_model", lambda: _FakeModel())

    draft = FasterWhisperHebrewProvider().transcribe("mixed.wav", language=None)

    assert captured["audio_path"] == "mixed.wav"
    assert "language" not in captured["kwargs"]
    assert draft.text == "hello"


def test_pipeline_uses_auto_language_for_mixed_warning_tracks(tmp_path, monkeypatch):
    class _LanguageAwareTranscriber:
        name = "transcriber"

        def __init__(self):
            self.captured_language = "unset"

        def transcribe(self, audio_path, language="unset"):
            assert audio_path == "vocals.wav"
            self.captured_language = language
            return _draft(provider=self.name)

    transcriber = _LanguageAwareTranscriber()
    pipeline = _make_pipeline(tmp_path, monkeypatch, transcriber=transcriber)

    language_info = LanguageDetectionResult(
        language="en",
        probability=0.72,
        policy_decision="warn",
        warning_message="mixed track",
        hebrew_ratio=0.34,
        provider="detector",
    )

    pipeline.step_transcribe("vocals.wav", language_info=language_info)

    assert transcriber.captured_language is None


def test_pipeline_keeps_legacy_transcribers_without_language_kwarg(tmp_path, monkeypatch):
    class _LegacyTranscriber:
        name = "legacy_transcriber"

        def transcribe(self, audio_path):
            assert audio_path == "vocals.wav"
            return _draft(provider=self.name)

    pipeline = _make_pipeline(tmp_path, monkeypatch, transcriber=_LegacyTranscriber())

    language_info = LanguageDetectionResult(
        language="en",
        probability=0.72,
        policy_decision="warn",
        warning_message="mixed track",
        hebrew_ratio=0.34,
        provider="detector",
    )

    draft = pipeline.step_transcribe("vocals.wav", language_info=language_info)

    assert draft.provider == "legacy_transcriber"
    assert draft.text == "hello"
