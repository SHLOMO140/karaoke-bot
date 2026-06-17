from karaoke import job_manager
from karaoke.auto_repair import apply_feedback_to_review, extract_feedback_line_edits, feedback_mentions_timing_problem
from karaoke.models import TranscriptDraft, TranscriptSegment, WordTiming


def _segment_from_text(text: str, start: float, end: float) -> TranscriptSegment:
    words = text.split()
    span = max(end - start, 0.01)
    step = span / max(len(words), 1)
    timings = [
        WordTiming(word, start + index * step, start + (index + 1) * step, confidence=0.9)
        for index, word in enumerate(words)
    ]
    return TranscriptSegment(words=timings, text=text, start=start, end=end)


def test_extract_feedback_line_edits_accepts_hebrew_and_plain_numbered_lines():
    edits = extract_feedback_line_edits(
        "\n".join(
            [
                "הערות כלליות:",
                "- שורה 2: עולם מתוקן",
                "3: פזמון נכון",
                "line 4 - outro fixed",
                "שורה 5:",
                "שורה 6: -",
            ]
        )
    )

    assert [(edit.line_number, edit.text) for edit in edits] == [
        (2, "עולם מתוקן"),
        (3, "פזמון נכון"),
        (4, "outro fixed"),
    ]


def test_feedback_mentions_timing_problem_accepts_human_language():
    assert feedback_mentions_timing_problem("\u05d4\u05de\u05d9\u05dc\u05d9\u05dd \u05dc\u05d0 \u05d1\u05d6\u05de\u05df")
    assert feedback_mentions_timing_problem("\u05d4\u05db\u05ea\u05d5\u05d1\u05d9\u05d5\u05ea \u05dc\u05d0 \u05de\u05e1\u05d5\u05e0\u05db\u05e8\u05e0\u05d5\u05ea")
    assert feedback_mentions_timing_problem("the subtitles are out of sync")
    assert not feedback_mentions_timing_problem("line 2: corrected chorus")


def test_apply_feedback_to_review_updates_review_transcript_and_manual_option(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    segments = [
        _segment_from_text("hello world", 0.0, 2.0),
        _segment_from_text("old chorus", 2.0, 4.0),
        _segment_from_text("final line", 4.0, 6.0),
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=segments, provider="fake"))
    job_manager.save_review_transcript(job, segments)

    result = apply_feedback_to_review(job, "שורה 2: corrected chorus")
    loaded = job_manager.load_job(job.job_id)
    review_segments = job_manager.load_review_segments(loaded)

    assert result.applied is True
    assert result.line_numbers == [2]
    assert [segment.text for segment in review_segments] == [
        "hello world",
        "corrected chorus",
        "final line",
    ]
    assert loaded.manifest.lyrics_verification["selected_option_id"] == "manual"


def test_apply_feedback_to_review_reports_invalid_line_without_saving(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    segments = [_segment_from_text("hello world", 0.0, 2.0)]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=segments, provider="fake"))
    job_manager.save_review_transcript(job, segments)

    result = apply_feedback_to_review(job, "שורה 8: impossible")
    review_segments = job_manager.load_review_segments(job_manager.load_job(job.job_id))

    assert result.applied is False
    assert result.error
    assert [segment.text for segment in review_segments] == ["hello world"]
