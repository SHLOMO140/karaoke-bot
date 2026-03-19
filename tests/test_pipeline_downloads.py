from pathlib import Path

import pytest

from karaoke import job_manager
from karaoke.exceptions import AudioExtractionError
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
