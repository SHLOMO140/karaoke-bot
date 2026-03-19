from pathlib import Path

from karaoke import job_manager
from karaoke.models import ChordEvent, JobStatus, LyricsVerificationResult, SongAnalysis, TranscriptDraft, TranscriptSegment, VideoRequest, WordTiming


def _segments():
    return [
        TranscriptSegment(
            words=[
                WordTiming("שלום", 0.0, 1.0, confidence=0.9),
                WordTiming("עולם", 1.2, 2.2, confidence=0.95),
            ],
            text="שלום עולם",
            start=0.0,
            end=2.2,
        )
    ]


def test_job_manifest_and_review_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)

    draft = TranscriptDraft(segments=_segments(), provider="fake")
    job_manager.save_draft_transcript(job, draft)
    job_manager.update_status(job, JobStatus.AWAITING_REVIEW)
    job_manager.set_active_review_job(11, 22, job.job_id)

    loaded = job_manager.get_active_review_job(11, 22)
    assert loaded is not None
    assert loaded.job_id == job.job_id
    assert loaded.manifest.providers == {}
    assert (tmp_path / job.job_id / "draft_transcript.txt").exists()
    assert (tmp_path / "_sessions.json").exists()
    assert loaded.display_name == "demo"


def test_full_text_review_rewrites_segments_without_dropping_times(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file")
    draft_segments = _segments()
    job_manager.save_review_transcript(job, draft_segments)

    updated = job_manager.update_transcript_text(draft_segments, "שלום יפה עולם")
    job_manager.save_review_transcript(job, updated)
    loaded = job_manager.load_review_segments(job)

    assert loaded[0].text == "שלום יפה עולם"
    assert len(loaded[0].words) == 3
    assert loaded[0].start == 0.0
    assert loaded[0].end == 2.2


def test_apply_lyrics_option_rewrites_review_text_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file")
    draft = TranscriptDraft(segments=_segments(), provider="fake")
    job_manager.save_draft_transcript(job, draft)
    verification = LyricsVerificationResult(
        provider="test",
        options=[
            {
                "option_id": "draft",
                "label": "draft",
                "lines": ["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"],
            },
            {
                "option_id": "verified",
                "label": "verified",
                "lines": ["\u05e9\u05dc\u05d5\u05dd \u05d9\u05e4\u05d4 \u05e2\u05d5\u05dc\u05dd"],
                "source_count": 2,
                "confidence": 0.9,
            },
        ],
    )
    job_manager.save_lyrics_verification(job, verification)

    selected = job_manager.apply_lyrics_option(job, "verified")
    loaded = job_manager.load_review_segments(job)
    reloaded = job_manager.load_job(job.job_id)

    assert selected["option_id"] == "verified"
    assert loaded[0].text == "\u05e9\u05dc\u05d5\u05dd \u05d9\u05e4\u05d4 \u05e2\u05d5\u05dc\u05dd"
    assert job_manager.get_selected_lyrics_option_id(reloaded) == "verified"


def test_update_transcript_line_splits_corrected_phrase_inside_original_error_span():
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("היהטוב", 0.4, 1.0, confidence=0.9),
                WordTiming("מאוד", 1.1, 1.6, confidence=0.95),
            ],
            text="היהטוב מאוד",
            start=0.4,
            end=1.6,
        )
    ]

    updated = job_manager.update_transcript_line(segments, 1, "כשהיה טוב מאוד")
    words = updated[0].words

    assert [word.word for word in words] == ["כשהיה", "טוב", "מאוד"]
    assert words[0].start == 0.4
    assert words[1].end <= 1.0
    assert words[2].start == 1.1
    assert words[2].end == 1.6


def test_update_transcript_text_preserves_phrase_span_even_when_line_count_changes():
    segments = [
        TranscriptSegment(
            words=[WordTiming("היהטוב", 0.4, 1.0, confidence=0.9)],
            text="היהטוב",
            start=0.4,
            end=1.0,
        ),
        TranscriptSegment(
            words=[WordTiming("מאוד", 1.1, 1.6, confidence=0.95)],
            text="מאוד",
            start=1.1,
            end=1.6,
        ),
    ]

    updated = job_manager.update_transcript_text(segments, "כשהיה טוב מאוד")
    words = updated[0].words

    assert len(updated) == 1
    assert [word.word for word in words] == ["כשהיה", "טוב", "מאוד"]
    assert words[0].start == 0.4
    assert words[1].end <= 1.0
    assert words[2].start >= 1.1
    assert updated[0].end == 1.6


def test_find_latest_reusable_job_prefers_newest_matching_source(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    older = job_manager.create_job(title="song", source_url="https://youtube.com/watch?v=1", input_type="youtube", user_id=7)
    newer = job_manager.create_job(title="song", source_url="https://youtube.com/watch?v=1", input_type="youtube", user_id=7)

    older.original_audio_path.write_text("old", encoding="utf-8")
    newer.ass_path.write_text("new", encoding="utf-8")
    job_manager.save_job(older)
    job_manager.save_job(newer)

    found = job_manager.find_latest_reusable_job(
        source_url="https://youtube.com/watch?v=1",
        input_type="youtube",
        user_id=7,
    )

    assert found is not None
    assert found.job_id == newer.job_id


def test_get_output_files_respects_requested_video_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="song", input_type="youtube")

    job.transcript_path.write_text("lyrics", encoding="utf-8")
    job.timings_path.write_text("{}", encoding="utf-8")
    job.srt_path.write_text("srt", encoding="utf-8")
    job.ass_path.write_text("ass", encoding="utf-8")
    job.song_analysis_path.write_text("{}", encoding="utf-8")
    job.lyrics_with_chords_path.write_text("C\nlyrics", encoding="utf-8")
    job.video_vocals_path.write_text("video", encoding="utf-8")
    job.video_instrumental_path.write_text("inst", encoding="utf-8")

    subs_only = job_manager.get_output_files(job)
    with_video = job_manager.get_output_files(job, video_request=VideoRequest(with_vocals=True))

    assert "song_analysis.json" in subs_only
    assert "lyrics_with_chords.txt" in subs_only
    assert "final_video.mp4" not in subs_only
    assert "final_video_instrumental.mp4" not in subs_only
    assert "final_video.mp4" in with_video
    assert "final_video_instrumental.mp4" not in with_video


def test_song_analysis_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="song", input_type="audio_file")
    analysis = SongAnalysis(
        bpm=121.5,
        preview_window_seconds=0.5,
        provider="test",
        source_audio="instrumental.mp3",
        beat_times=[0.0, 0.5, 1.0],
        chord_events=[
            ChordEvent("C", 0.0, 0.5, confidence=0.9, root="C", quality="major"),
            ChordEvent("Am", 0.5, 1.0, confidence=0.87, root="A", quality="minor"),
        ],
    )

    job_manager.save_song_analysis(job, analysis)
    job_manager.save_chord_sheet(job, "C  Am\nlyrics\n")
    loaded = job_manager.load_song_analysis(job)

    assert loaded.bpm == 121.5
    assert loaded.preview_window_seconds == 0.5
    assert [event.label for event in loaded.chord_events] == ["C", "Am"]
    assert job.lyrics_with_chords_path.read_text(encoding="utf-8").startswith("C  Am")
