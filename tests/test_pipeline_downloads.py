from pathlib import Path

import pytest

from karaoke import job_manager
from karaoke.exceptions import AudioExtractionError
from karaoke.models import (
    AlignedTranscript,
    ChordEvent,
    LyricsVerificationResult,
    SingerAnalysisResult,
    SongAnalysis,
    TranscriptDraft,
    TranscriptSegment,
    WordTiming,
)
from karaoke.pipeline import KaraokePipeline


class _DummyProvider:
    def __init__(self, name: str):
        self.name = name


@pytest.fixture()
def dummy_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(
        title="song",
        source_url="https://www.youtube.com/watch?v=test",
        input_type="youtube",
        chat_id=1,
        user_id=1,
    )
    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=_DummyProvider("transcriber"),
        lyrics_verifier=_DummyProvider("lyrics_verifier"),
        aligner=_DummyProvider("aligner"),
        srt_renderer=_DummyProvider("srt_renderer"),
        ass_renderer=_DummyProvider("ass_renderer"),
    )
    monkeypatch.setattr(pipeline, "_download_thumbnail", lambda: None)
    return pipeline


def test_download_youtube_audio_uses_ascii_stage_and_transcodes(dummy_pipeline, monkeypatch):
    captured = {}

    def fake_run(ydl_opts, prefix, stage_dir=None):
        assert prefix == "yt_audio"
        assert stage_dir is not None
        assert "postprocessors" not in ydl_opts
        raw_file = Path(stage_dir) / "yt_audio.webm"
        raw_file.write_bytes(b"raw")
        captured["stage_dir"] = Path(stage_dir)
        return {"title": "Example"}

    def fake_transcode(input_path, output_path):
        captured["input_path"] = input_path
        Path(output_path).write_bytes(b"mp3")

    monkeypatch.setattr(dummy_pipeline, "_run_ytdlp_with_retry", fake_run)
    monkeypatch.setattr("karaoke.pipeline.transcode_to_mp3", fake_transcode)

    output_path = dummy_pipeline._download_youtube_audio()

    assert Path(output_path).exists()
    assert Path(captured["input_path"]).name == "yt_audio.webm"
    assert captured["stage_dir"].name == "audio"
    assert captured["stage_dir"].parent.name == dummy_pipeline.job.job_id
    assert captured["stage_dir"].parent.parent.name == "yt_dlp"
    assert not captured["stage_dir"].exists()


def test_download_youtube_audio_keeps_audio_extraction_error(dummy_pipeline, monkeypatch):
    def fake_run(ydl_opts, prefix, stage_dir=None):
        raw_file = Path(stage_dir) / "yt_audio.webm"
        raw_file.write_bytes(b"raw")
        return {"title": "Example"}

    def fake_transcode(_input_path, _output_path):
        raise AudioExtractionError("ffmpeg failed")

    monkeypatch.setattr(dummy_pipeline, "_run_ytdlp_with_retry", fake_run)
    monkeypatch.setattr("karaoke.pipeline.transcode_to_mp3", fake_transcode)

    with pytest.raises(AudioExtractionError):
        dummy_pipeline._download_youtube_audio()


def test_step_verify_lyrics_exposes_job_source_url_to_verifier(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(
        title="song",
        source_url="https://www.youtube.com/watch?v=test123",
        input_type="youtube",
        chat_id=1,
        user_id=1,
    )

    class _Verifier:
        name = "lyrics_verifier"

        def __init__(self):
            self.seen_source_url = ""

        def verify(self, title, draft):
            del title, draft
            self.seen_source_url = getattr(self, "_current_source_url", "")
            return LyricsVerificationResult(
                provider=self.name,
                verdict="matched",
                corrected_lines=[],
                selected_option_id="draft",
                options=[
                    {
                        "option_id": "draft",
                        "label": "draft",
                        "lines": [],
                    }
                ],
            )

    verifier = _Verifier()
    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=_DummyProvider("transcriber"),
        lyrics_verifier=verifier,
        aligner=_DummyProvider("aligner"),
        srt_renderer=_DummyProvider("srt_renderer"),
        ass_renderer=_DummyProvider("ass_renderer"),
    )

    draft = TranscriptDraft(
        segments=[
            TranscriptSegment(
                words=[WordTiming("שלום", 0.0, 0.5)],
                text="שלום",
                start=0.0,
                end=0.5,
            )
        ],
        provider="test",
    )

    pipeline.step_verify_lyrics(draft)

    assert verifier.seen_source_url == job.source_url


def test_step_analyze_music_preserves_original_chords_for_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    monkeypatch.setattr("karaoke.pipeline.lookup_external_chord_sheet", lambda *args, **kwargs: None)
    job = job_manager.create_job(
        title="song",
        source_url="https://www.youtube.com/watch?v=test123",
        input_type="audio_file",
        chat_id=1,
        user_id=1,
    )
    job.original_audio_path.write_bytes(b"audio")

    class _SongAnalyzer:
        name = "song_analyzer"

        def analyze(self, audio_path):
            return SongAnalysis(
                provider=self.name,
                source_audio=audio_path,
                chord_events=[
                    ChordEvent("Em", 0.0, 0.5, confidence=0.9, root="E", quality="minor"),
                    ChordEvent("Am", 0.5, 1.0, confidence=0.9, root="A", quality="minor"),
                    ChordEvent("B7", 1.0, 1.5, confidence=0.9, root="B", quality="dominant7"),
                    ChordEvent("Em", 1.5, 2.0, confidence=0.9, root="E", quality="minor"),
                ],
            )

    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=_DummyProvider("transcriber"),
        lyrics_verifier=_DummyProvider("lyrics_verifier"),
        aligner=_DummyProvider("aligner"),
        song_analyzer=_SongAnalyzer(),
        singer_analyzer=_DummyProvider("singer_analyzer"),
        srt_renderer=_DummyProvider("srt_renderer"),
        ass_renderer=_DummyProvider("ass_renderer"),
        subtitle_validator=_DummyProvider("subtitle_validator"),
    )
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("one", 0.0, 0.5),
                WordTiming("two", 0.5, 1.0),
                WordTiming("three", 1.0, 1.5),
                WordTiming("four", 1.5, 2.0),
            ],
            text="one two three four",
            start=0.0,
            end=2.0,
        )
    ]

    analysis = pipeline.step_analyze_music(segments)
    chord_sheet = job.lyrics_with_chords_path.read_text(encoding="utf-8")

    assert analysis.original_key == "Em"
    assert analysis.target_key == ""
    assert analysis.transpose_semitones == 0
    assert [event.label for event in analysis.original_chord_events] == ["Em", "Am", "B7", "Em"]
    assert [event.label for event in analysis.chord_events] == ["Em", "Am", "B7", "Em"]
    assert "Am" in chord_sheet
    assert "B7" in chord_sheet
    assert "סולם קל: Am" not in chord_sheet


def test_step_analyze_music_prefers_external_chord_sheet_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(
        title="Artist - Song Name",
        source_url="https://www.youtube.com/watch?v=test123",
        input_type="audio_file",
        chat_id=1,
        user_id=1,
    )
    job.original_audio_path.write_bytes(b"audio")

    class _SongAnalyzer:
        name = "librosa_harmony_v5"

        def __init__(self):
            self.calls = 0

        def analyze(self, audio_path):
            self.calls += 1
            return SongAnalysis(provider=self.name, source_audio=audio_path)

    monkeypatch.setattr(
        "karaoke.pipeline.lookup_external_chord_sheet",
        lambda title, segments, provider, source_audio="", target_key="": SongAnalysis(
            provider=provider,
            source_audio=source_audio,
            original_key="Em",
            target_key=target_key,
            transpose_semitones=0,
            chord_sheet_text="כותרת: song\nקצב: לא ידוע\nמשקל: 4/4\nסולם מקור: Em\n\nEm\nlyrics\n",
            chord_source_name="Tab4U",
            chord_source_url="https://www.tab4u.com/tabs/songs/1_song.html",
        ),
    )

    song_analyzer = _SongAnalyzer()
    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=_DummyProvider("transcriber"),
        lyrics_verifier=_DummyProvider("lyrics_verifier"),
        aligner=_DummyProvider("aligner"),
        song_analyzer=song_analyzer,
        singer_analyzer=_DummyProvider("singer_analyzer"),
        srt_renderer=_DummyProvider("srt_renderer"),
        ass_renderer=_DummyProvider("ass_renderer"),
        subtitle_validator=_DummyProvider("subtitle_validator"),
    )
    segments = [
        TranscriptSegment(
            words=[WordTiming("lyrics", 0.0, 0.5)],
            text="lyrics",
            start=0.0,
            end=0.5,
        )
    ]

    analysis = pipeline.step_analyze_music(segments)

    assert song_analyzer.calls == 0
    assert analysis.provider == "librosa_harmony_v5"
    assert analysis.chord_source_name == "Tab4U"
    assert "סולם קל: Am" not in job.lyrics_with_chords_path.read_text(encoding="utf-8")


def test_step_analyze_music_falls_back_to_title_only_external_chord_sheet(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(
        title="Artist - Song Name",
        source_url="https://www.youtube.com/watch?v=test123",
        input_type="audio_file",
        chat_id=1,
        user_id=1,
    )
    job.original_audio_path.write_bytes(b"audio")

    class _SongAnalyzer:
        name = "librosa_harmony_v5"

        def __init__(self):
            self.calls = 0

        def analyze(self, audio_path):
            self.calls += 1
            return SongAnalysis(provider=self.name, source_audio=audio_path)

    monkeypatch.setattr("karaoke.pipeline.lookup_external_chord_sheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "karaoke.pipeline.lookup_external_chord_sheet_by_title",
        lambda title, provider, source_audio="", target_key="": SongAnalysis(
            provider=provider,
            source_audio=source_audio,
            original_key="Bm",
            target_key=target_key,
            transpose_semitones=0,
            chord_sheet_text="כותרת: Artist - Song Name\nקצב: לא ידוע\nמשקל: 4/4\nסולם מקור: Bm\n\nBm\nlyrics\n",
            chord_source_name="Tab4U",
            chord_source_url="https://www.tab4u.com/tabs/songs/1_song.html",
        ),
    )

    song_analyzer = _SongAnalyzer()
    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=_DummyProvider("transcriber"),
        lyrics_verifier=_DummyProvider("lyrics_verifier"),
        aligner=_DummyProvider("aligner"),
        song_analyzer=song_analyzer,
        singer_analyzer=_DummyProvider("singer_analyzer"),
        srt_renderer=_DummyProvider("srt_renderer"),
        ass_renderer=_DummyProvider("ass_renderer"),
        subtitle_validator=_DummyProvider("subtitle_validator"),
    )
    segments = [
        TranscriptSegment(
            words=[WordTiming("lyrics", 0.0, 0.5)],
            text="lyrics",
            start=0.0,
            end=0.5,
        )
    ]

    analysis = pipeline.step_analyze_music(segments)

    assert song_analyzer.calls == 0
    assert analysis.chord_source_name == "Tab4U"
    assert analysis.chord_source_url.endswith("1_song.html")
    assert "סולם קל: Am" not in job.lyrics_with_chords_path.read_text(encoding="utf-8")


def test_step_analyze_music_reruns_when_cached_provider_is_outdated(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    monkeypatch.setattr("karaoke.pipeline.lookup_external_chord_sheet", lambda *args, **kwargs: None)
    job = job_manager.create_job(
        title="song",
        source_url="https://www.youtube.com/watch?v=test123",
        input_type="audio_file",
        chat_id=1,
        user_id=1,
    )
    job.original_audio_path.write_bytes(b"audio")
    job_manager.save_song_analysis(
        job,
        SongAnalysis(
            provider="librosa_harmony_v3",
            source_audio=str(job.original_audio_path),
            original_key="C",
            target_key="Am",
            transpose_semitones=9,
            chord_events=[ChordEvent("Am", 0.0, 1.0, root="A", quality="minor")],
        ),
    )

    class _SongAnalyzer:
        name = "librosa_harmony_v5"

        def __init__(self):
            self.calls = 0

        def analyze(self, audio_path):
            self.calls += 1
            return SongAnalysis(
                provider=self.name,
                source_audio=audio_path,
                chord_events=[
                    ChordEvent("Em", 0.0, 0.5, confidence=0.9, root="E", quality="minor"),
                    ChordEvent("Am", 0.5, 1.0, confidence=0.9, root="A", quality="minor"),
                    ChordEvent("B7", 1.0, 1.5, confidence=0.9, root="B", quality="dominant7"),
                    ChordEvent("Em", 1.5, 2.0, confidence=0.9, root="E", quality="minor"),
                ],
            )

    song_analyzer = _SongAnalyzer()
    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=_DummyProvider("transcriber"),
        lyrics_verifier=_DummyProvider("lyrics_verifier"),
        aligner=_DummyProvider("aligner"),
        song_analyzer=song_analyzer,
        singer_analyzer=_DummyProvider("singer_analyzer"),
        srt_renderer=_DummyProvider("srt_renderer"),
        ass_renderer=_DummyProvider("ass_renderer"),
        subtitle_validator=_DummyProvider("subtitle_validator"),
    )
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("one", 0.0, 0.5),
                WordTiming("two", 0.5, 1.0),
                WordTiming("three", 1.0, 1.5),
                WordTiming("four", 1.5, 2.0),
            ],
            text="one two three four",
            start=0.0,
            end=2.0,
        )
    ]

    analysis = pipeline.step_analyze_music(segments)

    assert song_analyzer.calls == 1
    assert analysis.provider == "librosa_harmony_v5"
    assert analysis.original_key == "Em"


def test_step_analyze_music_reruns_when_cached_audio_only_analysis_is_unreliable(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    monkeypatch.setattr("karaoke.pipeline.lookup_external_chord_sheet", lambda *args, **kwargs: None)
    job = job_manager.create_job(
        title="song",
        source_url="https://www.youtube.com/watch?v=test123",
        input_type="audio_file",
        chat_id=1,
        user_id=1,
    )
    job.original_audio_path.write_bytes(b"audio")
    job_manager.save_song_analysis(
        job,
        SongAnalysis(
            provider="librosa_harmony_v5",
            source_audio=str(job.original_audio_path),
            chord_events=[
                ChordEvent("Am", 0.0, 4.0, confidence=0.31, root="A", quality="minor"),
                ChordEvent("C", 4.0, 8.0, confidence=0.28, root="C", quality="major"),
                ChordEvent("Dm", 8.0, 12.0, confidence=0.34, root="D", quality="minor"),
            ],
        ),
    )

    class _SongAnalyzer:
        name = "librosa_harmony_v5"

        def __init__(self):
            self.calls = 0

        def analyze(self, audio_path):
            self.calls += 1
            return SongAnalysis(
                provider=self.name,
                source_audio=audio_path,
                chord_events=[
                    ChordEvent("Em", 0.0, 0.5, confidence=0.9, root="E", quality="minor"),
                    ChordEvent("Am", 0.5, 1.0, confidence=0.9, root="A", quality="minor"),
                    ChordEvent("B7", 1.0, 1.5, confidence=0.9, root="B", quality="dominant7"),
                    ChordEvent("Em", 1.5, 2.0, confidence=0.9, root="E", quality="minor"),
                ],
            )

    song_analyzer = _SongAnalyzer()
    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=_DummyProvider("transcriber"),
        lyrics_verifier=_DummyProvider("lyrics_verifier"),
        aligner=_DummyProvider("aligner"),
        song_analyzer=song_analyzer,
        singer_analyzer=_DummyProvider("singer_analyzer"),
        srt_renderer=_DummyProvider("srt_renderer"),
        ass_renderer=_DummyProvider("ass_renderer"),
        subtitle_validator=_DummyProvider("subtitle_validator"),
    )
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("one", 0.0, 0.5),
                WordTiming("two", 0.5, 1.0),
                WordTiming("three", 1.0, 1.5),
                WordTiming("four", 1.5, 2.0),
            ],
            text="one two three four",
            start=0.0,
            end=2.0,
        )
    ]

    analysis = pipeline.step_analyze_music(segments)

    assert song_analyzer.calls == 1
    assert analysis.provider == "librosa_harmony_v5"
    assert analysis.original_key == "Em"


def test_run_after_review_auto_heals_review_text_word_mismatch_before_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    monkeypatch.setattr("karaoke.pipeline.lookup_external_chord_sheet", lambda *args, **kwargs: None)
    job = job_manager.create_job(
        title="song",
        source_url="https://www.youtube.com/watch?v=test123",
        input_type="youtube",
        chat_id=1,
        user_id=1,
    )

    draft_segments = [
        TranscriptSegment(
            words=[
                WordTiming("when", 0.0, 0.3, confidence=0.9),
                WordTiming("i", 0.3, 0.5, confidence=0.9),
                WordTiming("tire", 0.5, 1.0, confidence=0.9),
            ],
            text="when i tire",
            start=0.0,
            end=1.0,
        ),
        TranscriptSegment(
            words=[
                WordTiming("dont", 1.2, 1.5, confidence=0.9),
                WordTiming("believe", 1.5, 2.0, confidence=0.9),
                WordTiming("me", 2.0, 2.4, confidence=0.9),
            ],
            text="dont believe me",
            start=1.2,
            end=2.4,
        ),
        TranscriptSegment(
            words=[
                WordTiming("because", 2.8, 3.2, confidence=0.9),
                WordTiming("its", 3.2, 3.5, confidence=0.9),
                WordTiming("not", 3.5, 3.7, confidence=0.9),
                WordTiming("me", 3.7, 4.0, confidence=0.9),
            ],
            text="because its not me",
            start=2.8,
            end=4.0,
        ),
    ]
    broken_review = [
        TranscriptSegment(
            words=list(draft_segments[0].words),
            text="when i tire dont believe me",
            start=0.0,
            end=1.0,
        ),
        TranscriptSegment(
            words=draft_segments[1].words[:2],
            text="because its",
            start=1.2,
            end=2.0,
        ),
        TranscriptSegment(
            words=draft_segments[2].words[-2:],
            text="not me",
            start=3.5,
            end=4.0,
        ),
    ]

    job_manager.save_draft_transcript(job, TranscriptDraft(segments=draft_segments, provider="test"))
    job_manager.save_review_transcript(job, broken_review)
    job.vocals_16k_path.write_bytes(b"wav")

    class _Aligner:
        name = "aligner"
        last_warning_message = ""

        def __init__(self):
            self.captured_segments = []

        def align(self, audio_path, approved_segments, draft_segments, video_frame_rate=None):
            del audio_path, draft_segments, video_frame_rate
            healed_segments = []
            for segment in approved_segments:
                healed_words = [
                    WordTiming(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        confidence=word.confidence,
                        source="forced_aligner",
                        aligned=True,
                        subwords=list(word.subwords),
                        char_timings=list(word.char_timings),
                    )
                    for word in segment.words
                ]
                healed_segments.append(
                    TranscriptSegment(
                        words=healed_words,
                        text=segment.text,
                        start=segment.start,
                        end=segment.end,
                    )
                )
            self.captured_segments = healed_segments
            return AlignedTranscript(segments=healed_segments, provider=self.name)

    class _SingerAnalyzer:
        name = "singer_analyzer"

        def analyze(self, singer_audio, segments, title=""):
            del singer_audio, segments, title
            return SingerAnalysisResult(provider=self.name)

    class _Renderer:
        def __init__(self, name: str, content: str = ""):
            self.name = name
            self.content = content

        def render(self, segments, output_path, **kwargs):
            del segments, kwargs
            Path(output_path).write_text(self.content, encoding="utf-8")

    class _Validator:
        name = "subtitle_validator"

        def validate(self, segments, style, singer_analysis=None):
            del segments, style, singer_analysis
            return []

    aligner = _Aligner()
    pipeline = KaraokePipeline(
        job,
        separator=_DummyProvider("separator"),
        language_detector=_DummyProvider("language_detector"),
        transcriber=_DummyProvider("transcriber"),
        lyrics_verifier=_DummyProvider("lyrics_verifier"),
        aligner=aligner,
        song_analyzer=_DummyProvider("song_analyzer"),
        singer_analyzer=_SingerAnalyzer(),
        srt_renderer=_Renderer("srt_renderer", "1\n00:00:00,000 --> 00:00:00,100\nx\n"),
        ass_renderer=_Renderer("ass_renderer", "[Script Info]\n"),
        subtitle_validator=_Validator(),
    )

    pipeline.run_after_review(broken_review, None)

    saved_review = job_manager.load_review_segments(job)

    assert not job_manager.find_segment_word_text_mismatches(saved_review)
    assert [word.word for word in aligner.captured_segments[0].words] == ["when", "i", "tire", "dont", "believe", "me"]
    assert saved_review[0].end == 2.4
