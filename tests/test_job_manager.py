from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from karaoke import job_manager
from karaoke.models import (
    AlignedTranscript,
    ChordEvent,
    JobStatus,
    LyricsVerificationResult,
    ReviewStatus,
    SingerAnalysisResult,
    SingerProfile,
    SingerSegmentAssignment,
    SongAnalysis,
    TranscriptDraft,
    TranscriptSegment,
    VideoRequest,
    WordTiming,
)


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


def _segment_from_text(text: str, start: float, end: float) -> TranscriptSegment:
    words = text.split()
    if not words:
        return TranscriptSegment(words=[], text=text, start=start, end=end)

    span = max(end - start, 0.01)
    step = span / len(words)
    word_timings = []
    cursor = start
    for index, word in enumerate(words):
        word_end = end if index == len(words) - 1 else cursor + step
        word_timings.append(WordTiming(word, round(cursor, 6), round(word_end, 6), confidence=0.9))
        cursor = word_end
    return TranscriptSegment(words=word_timings, text=text, start=start, end=end)


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


def test_group_request_roundtrip_and_owner_enforcement(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)

    token = job_manager.create_group_request(
        group_chat_id=-100123,
        group_message_id=77,
        user_id=22,
        request_kind="text",
        payload={"text": "demo song"},
    )

    rejected, rejected_reason = job_manager.claim_group_request(token, 99)
    claimed, claimed_reason = job_manager.claim_group_request(token, 22)

    assert rejected is None
    assert rejected_reason == "forbidden"
    assert claimed is not None
    assert claimed_reason is None
    assert claimed["group_chat_id"] == -100123
    assert claimed["group_message_id"] == 77
    assert claimed["request_kind"] == "text"
    assert claimed["payload"]["text"] == "demo song"


def test_unclaimed_group_request_can_be_bound_then_claimed(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)

    token = job_manager.create_group_request(
        group_chat_id=-100123,
        group_message_id=88,
        user_id=0,
        request_kind="text",
        payload={"text": "anonymous demo"},
    )

    unclaimed, unclaimed_reason = job_manager.claim_group_request(token, 22)
    bound, bound_reason = job_manager.bind_group_request_user(token, 22)
    claimed, claimed_reason = job_manager.claim_group_request(token, 22)

    assert unclaimed is None
    assert unclaimed_reason == "unclaimed"
    assert bound is not None
    assert bound_reason is None
    assert int(bound["user_id"]) == 22
    assert claimed is not None
    assert claimed_reason is None
    assert claimed["payload"]["text"] == "anonymous demo"


def test_create_job_defaults_delivery_chat_to_interaction_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)

    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)

    assert job.delivery_chat_id == 11
    assert job.delivery_reply_to_message_id == 0


def test_append_quality_feedback_updates_pending_delivery_and_writes_log(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    job_manager.update_pending_delivery(job, status="awaiting_feedback", target_chat_id=-100123)

    entry = job_manager.append_quality_feedback(
        job,
        "שורה 12: המילה לא נכונה\nהפזמון לא מדויק",
        source="text",
        user_id=22,
        chat_id=11,
    )
    reloaded = job_manager.load_job(job.job_id)
    log_text = reloaded.delivery_feedback_path.read_text(encoding="utf-8")

    assert entry["source"] == "text"
    assert reloaded.pending_delivery["status"] == "feedback_received"
    assert "שורה 12: המילה לא נכונה" in log_text
    assert "הפזמון לא מדויק" in log_text


def test_write_delivery_feedback_template_creates_editable_txt(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)

    template_path = job_manager.write_delivery_feedback_template(job)
    template_text = template_path.read_text(encoding="utf-8")

    assert template_path.exists()
    assert "מה לא יצא מושלם?" in template_text
    assert "שורה 1:" in template_text


def test_done_job_can_be_reopened_for_review_session(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)

    draft = TranscriptDraft(segments=_segments(), provider="fake")
    job_manager.save_draft_transcript(job, draft)
    job_manager.update_status(job, JobStatus.DONE)
    job_manager.update_review_status(job, ReviewStatus.AWAITING_REVIEW)
    job_manager.set_active_review_job(11, 22, job.job_id)

    loaded = job_manager.get_active_review_job(11, 22)

    assert loaded is not None
    assert loaded.job_id == job.job_id


def test_get_best_available_segments_prefers_review_during_active_review(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file")

    review_segments = _segments()
    final_segments = [
        TranscriptSegment(
            words=[WordTiming("final", 0.0, 1.0, confidence=0.9)],
            text="final",
            start=0.0,
            end=1.0,
        )
    ]

    job_manager.save_review_transcript(job, review_segments)
    job_manager.save_final_transcript(job, AlignedTranscript(segments=final_segments, provider="fake"))
    job_manager.update_review_status(job, ReviewStatus.AWAITING_REVIEW)

    loaded = job_manager.get_best_available_segments(job)

    assert loaded[0].text == review_segments[0].text


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


def test_full_text_review_populates_character_and_subword_timings():
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("hello", 0.0, 1.0, confidence=0.9),
                WordTiming("world", 1.2, 2.2, confidence=0.95),
            ],
            text="hello world",
            start=0.0,
            end=2.2,
        )
    ]

    updated = job_manager.update_transcript_text(segments, "hello brave world")
    brave = updated[0].words[1]

    assert brave.word == "brave"
    assert brave.subwords
    assert brave.char_timings
    assert brave.char_timings[0].start == brave.start
    assert brave.char_timings[-1].end == brave.end


def test_review_timings_roundtrip_preserves_character_timings(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file")
    draft_segments = [
        TranscriptSegment(
            words=[
                WordTiming("hello", 0.0, 1.0, confidence=0.9),
                WordTiming("world", 1.2, 2.2, confidence=0.95),
            ],
            text="hello world",
            start=0.0,
            end=2.2,
        )
    ]

    updated = job_manager.update_transcript_text(draft_segments, "hello brave world")
    job_manager.save_review_transcript(job, updated)
    loaded = job_manager.load_review_segments(job)

    assert loaded[0].words[1].char_timings
    assert [subword.text for subword in loaded[0].words[1].subwords]


def test_singer_analysis_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file")

    analysis = SingerAnalysisResult(
        detected_singer_count=2,
        provider="test_singers",
        profiles=[
            SingerProfile(singer_id="singer_1", label="Singer 1", lane_index=0),
            SingerProfile(singer_id="singer_2", label="Singer 2", lane_index=1),
        ],
        assignments=[
            SingerSegmentAssignment(segment_index=0, singer_id="singer_1", label="Singer 1", confidence=0.81),
            SingerSegmentAssignment(segment_index=1, singer_id="singer_2", label="Singer 2", confidence=0.76),
        ],
        low_confidence_segments=0,
        analysis_window_seconds=12.4,
    )

    job_manager.save_singer_analysis(job, analysis)
    loaded = job_manager.load_singer_analysis(job)

    assert loaded.detected_singer_count == 2
    assert [profile.singer_id for profile in loaded.profiles] == ["singer_1", "singer_2"]
    assert [assignment.singer_id for assignment in loaded.assignments] == ["singer_1", "singer_2"]
    assert loaded.analysis_window_seconds == 12.4


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


def test_get_selectable_lyrics_options_hides_draft_reference_option(tmp_path, monkeypatch):
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
            },
            {
                "option_id": "source_site",
                "label": "source",
                "lines": ["\u05e9\u05dc\u05d5\u05dd \u05de\u05d3\u05d5\u05d9\u05e7 \u05e2\u05d5\u05dc\u05dd"],
            },
        ],
    )
    job_manager.save_lyrics_verification(job, verification)

    selectable = job_manager.get_selectable_lyrics_options(job)
    reference = job_manager.get_reference_lyrics_option(job)

    assert [option["option_id"] for option in selectable] == ["verified", "source_site"]
    assert reference is not None
    assert reference["option_id"] == "draft"


def test_apply_lyrics_option_rejects_draft_reference_option(tmp_path, monkeypatch):
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
            },
        ],
    )
    job_manager.save_lyrics_verification(job, verification)

    with pytest.raises(ValueError, match="\u05dc\u05d4\u05e9\u05d5\u05d5\u05d0\u05d4 \u05d1\u05dc\u05d1\u05d3"):
        job_manager.apply_lyrics_option(job, "draft")


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


def test_update_transcript_text_anchors_rebuilt_lines_to_matched_words_inside_large_block():
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("alphaaaaa", 0.0, 2.0, confidence=0.9),
                WordTiming("betaaaaa", 2.0, 4.0, confidence=0.9),
                WordTiming("pivot", 10.0, 10.5, confidence=0.95),
                WordTiming("omega", 10.5, 11.0, confidence=0.95),
                WordTiming("tail", 11.0, 12.0, confidence=0.95),
            ],
            text="alphaaaaa betaaaaa pivot omega tail",
            start=0.0,
            end=12.0,
        )
    ]

    updated = job_manager.update_transcript_text(
        segments,
        "\n".join(
            [
                "alphaaaaa betaaaaa",
                "pivot",
                "omega tail",
            ]
        ),
    )

    assert len(updated) == 3
    assert updated[0].start == 0.0
    assert updated[0].end == 4.0
    assert updated[1].start == 10.0
    assert updated[1].end == 10.5
    assert updated[2].start == 10.5
    assert updated[2].end == 12.0


def test_update_transcript_text_keeps_repeated_phrase_local_instead_of_spanning_distant_repeat():
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("fix", 10.0, 11.0, confidence=0.9),
                WordTiming("fix", 11.0, 12.0, confidence=0.9),
                WordTiming("like", 12.0, 13.0, confidence=0.9),
                WordTiming("kid", 13.0, 14.0, confidence=0.9),
                WordTiming("with", 14.0, 15.0, confidence=0.9),
                WordTiming("broken", 15.0, 16.0, confidence=0.9),
                WordTiming("life", 16.0, 17.0, confidence=0.9),
                WordTiming("passes", 17.0, 18.0, confidence=0.9),
                WordTiming("flood", 18.0, 19.0, confidence=0.9),
                WordTiming("soul", 19.0, 20.0, confidence=0.9),
                WordTiming("says", 20.0, 21.0, confidence=0.9),
                WordTiming("pray", 21.0, 22.0, confidence=0.9),
                WordTiming("and", 22.0, 23.0, confidence=0.9),
                WordTiming("truth", 23.0, 24.0, confidence=0.9),
                WordTiming("hurts", 24.0, 25.0, confidence=0.9),
                WordTiming("you", 25.0, 26.0, confidence=0.9),
                WordTiming("hold", 26.0, 27.0, confidence=0.9),
                WordTiming("my", 27.0, 28.0, confidence=0.9),
                WordTiming("hand", 28.0, 29.0, confidence=0.9),
                WordTiming("like", 40.0, 41.0, confidence=0.9),
                WordTiming("kid", 41.0, 42.0, confidence=0.9),
                WordTiming("good", 42.0, 43.0, confidence=0.9),
                WordTiming("with", 43.0, 44.0, confidence=0.9),
                WordTiming("broken", 44.0, 45.0, confidence=0.9),
                WordTiming("life", 45.0, 46.0, confidence=0.9),
                WordTiming("passes", 46.0, 47.0, confidence=0.9),
                WordTiming("flood", 47.0, 48.0, confidence=0.9),
                WordTiming("soul", 48.0, 49.0, confidence=0.9),
                WordTiming("says", 49.0, 50.0, confidence=0.9),
                WordTiming("pray", 50.0, 51.0, confidence=0.9),
                WordTiming("and", 51.0, 52.0, confidence=0.9),
                WordTiming("truth", 52.0, 53.0, confidence=0.9),
                WordTiming("hurts", 53.0, 54.0, confidence=0.9),
                WordTiming("you", 54.0, 55.0, confidence=0.9),
                WordTiming("hold", 55.0, 56.0, confidence=0.9),
                WordTiming("my", 56.0, 57.0, confidence=0.9),
                WordTiming("hand", 57.0, 58.0, confidence=0.9),
            ],
            text="fix fix like kid with broken life passes flood soul says pray and truth hurts you hold my hand like kid good with broken life passes flood soul says pray and truth hurts you hold my hand",
            start=10.0,
            end=58.0,
        )
    ]

    updated = job_manager.update_transcript_text(
        segments,
        "\n".join(
            [
                "fix fix",
                "like kid good with broken",
                "life passes flood",
                "soul says pray",
                "and truth hurts you hold my hand",
            ]
        ),
    )

    assert len(updated) == 5
    assert updated[1].start < 13.5
    assert updated[1].end < 18.0
    assert updated[2].start < 18.5
    assert updated[3].start < 22.5
    assert updated[4].start < 25.0
    assert updated[2].start - updated[1].end < 2.0


def test_align_words_to_draft_discards_matches_outside_target_segment_span():
    orig_words = [
        WordTiming("hello", 10.0, 11.0, confidence=0.9),
        WordTiming("world", 11.0, 12.0, confidence=0.9),
        WordTiming("hello", 100.0, 101.0, confidence=0.95),
        WordTiming("world", 101.0, 102.0, confidence=0.95),
    ]

    aligned = job_manager._align_words_to_draft(["hello", "world"], orig_words, 100.0, 102.0)

    assert [word.word for word in aligned] == ["hello", "world"]
    assert aligned[0].start >= 100.0
    assert aligned[-1].end <= 102.0
    assert aligned[0].end <= aligned[1].start


def test_detect_suspicious_review_shrink_flags_partial_song_replacement():
    segments = [
        TranscriptSegment(
            words=[WordTiming(f"w{i}_{j}", float(j), float(j + 1), confidence=0.9) for j in range(12)],
            text=" ".join(f"w{i}_{j}" for j in range(12)),
            start=float(i * 12),
            end=float((i + 1) * 12),
        )
        for i in range(5)
    ]

    warning = job_manager.detect_suspicious_review_shrink(
        segments,
        "one two three four\nfive six seven eight\nnine ten eleven twelve",
    )

    assert warning is not None
    assert warning["existing_words"] == 60
    assert warning["new_words"] == 12


def test_detect_suspicious_review_shrink_allows_condensed_chorus_when_full_song_can_be_recovered():
    lines = [
        "verse start carry me home",
        "verse wind above the sea",
        "color burning through night",
        "hold me in the fire",
        "we rise again tonight",
        "color blurring true night",
        "hold me in fire",
        "we rise agan tonite",
        "bridge only stars remain",
        "color turning blue night",
        "hold me in the flame",
        "we rise again tonite",
    ]
    segments = []
    cursor = 0.0
    for line in lines:
        words = []
        for index, word in enumerate(line.split()):
            words.append(WordTiming(word, cursor + index * 0.4, cursor + (index + 1) * 0.4, confidence=0.9))
        segments.append(
            TranscriptSegment(
                words=words,
                text=line,
                start=words[0].start,
                end=words[-1].end,
            )
        )
        cursor = segments[-1].end + 0.2

    corrected_text = "\n".join(
        [
            "verse start carry me home",
            "verse wind above the sea",
            "color burning through the night",
            "hold me in the fire",
            "we rise again tonight",
            "bridge only stars remain",
        ]
    )

    warning = job_manager.detect_suspicious_review_shrink(segments, corrected_text)
    updated = job_manager.update_transcript_text(segments, corrected_text)

    assert warning is None
    assert [segment.text for segment in updated] == [
        "verse start carry me home",
        "verse wind above the sea",
        "color burning through the night",
        "hold me in the fire",
        "we rise again tonight",
        "color burning through the night",
        "hold me in the fire",
        "we rise again tonight",
        "bridge only stars remain",
        "color burning through the night",
        "hold me in the fire",
        "we rise again tonight",
    ]


def test_update_transcript_text_expands_omitted_repeated_chorus_lines():
    segments = [
        TranscriptSegment(words=[WordTiming("verse", 0.0, 1.0), WordTiming("one", 1.0, 2.0)], text="verse one", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("chorus", 2.0, 3.0), WordTiming("bright", 3.0, 4.0), WordTiming("light", 4.0, 5.0)], text="chorus bright light", start=2.0, end=5.0),
        TranscriptSegment(words=[WordTiming("chorus", 5.0, 6.0), WordTiming("hold", 6.0, 7.0), WordTiming("on", 7.0, 8.0)], text="chorus hold on", start=5.0, end=8.0),
        TranscriptSegment(words=[WordTiming("second", 8.0, 9.0), WordTiming("verse", 9.0, 10.0)], text="second verse", start=8.0, end=10.0),
        TranscriptSegment(words=[WordTiming("chorus", 12.0, 13.0), WordTiming("bright", 13.0, 14.0), WordTiming("light", 14.0, 15.0)], text="chorus bright light", start=12.0, end=15.0),
        TranscriptSegment(words=[WordTiming("chorus", 15.0, 16.0), WordTiming("hold", 16.0, 17.0), WordTiming("on", 17.0, 18.0)], text="chorus hold on", start=15.0, end=18.0),
        TranscriptSegment(words=[WordTiming("final", 18.0, 19.0), WordTiming("outro", 19.0, 20.0)], text="final outro", start=18.0, end=20.0),
    ]

    updated = job_manager.update_transcript_text(
        segments,
        "\n".join(
            [
                "verse one",
                "chorus shining light",
                "chorus keep on",
                "second verse",
                "final outro",
            ]
        ),
    )

    assert [segment.text for segment in updated] == [
        "verse one",
        "chorus shining light",
        "chorus keep on",
        "second verse",
        "chorus shining light",
        "chorus keep on",
        "final outro",
    ]
    assert updated[4].start == 12.0
    assert updated[5].end == 18.0


def test_update_transcript_text_expands_repeated_chorus_from_song_structure():
    segments = [
        TranscriptSegment(words=[WordTiming("verse", 0.0, 1.0), WordTiming("one", 1.0, 2.0)], text="verse one", start=0.0, end=2.0),
        TranscriptSegment(
            words=[WordTiming("alpha", 2.0, 3.0), WordTiming("beta", 3.0, 4.0), WordTiming("gamma", 4.0, 5.0)],
            text="alpha beta gamma",
            start=2.0,
            end=5.0,
        ),
        TranscriptSegment(
            words=[WordTiming("delta", 5.0, 6.0), WordTiming("epsilon", 6.0, 7.0), WordTiming("zeta", 7.0, 8.0)],
            text="delta epsilon zeta",
            start=5.0,
            end=8.0,
        ),
        TranscriptSegment(words=[WordTiming("second", 8.0, 9.0), WordTiming("verse", 9.0, 10.0)], text="second verse", start=8.0, end=10.0),
        TranscriptSegment(
            words=[WordTiming("alpha", 12.0, 13.0), WordTiming("beta", 13.0, 14.0), WordTiming("gamma", 14.0, 15.0)],
            text="alpha beta gamma",
            start=12.0,
            end=15.0,
        ),
        TranscriptSegment(
            words=[WordTiming("delta", 15.0, 16.0), WordTiming("epsilon", 16.0, 17.0), WordTiming("zeta", 17.0, 18.0)],
            text="delta epsilon zeta",
            start=15.0,
            end=18.0,
        ),
        TranscriptSegment(words=[WordTiming("final", 18.0, 19.0), WordTiming("outro", 19.0, 20.0)], text="final outro", start=18.0, end=20.0),
    ]

    updated = job_manager.update_transcript_text(
        segments,
        "\n".join(
            [
                "verse one",
                "paint the skyline gold",
                "carry every spark home",
                "second verse",
                "final outro",
            ]
        ),
    )

    assert [segment.text for segment in updated] == [
        "verse one",
        "paint the skyline gold",
        "carry every spark home",
        "second verse",
        "paint the skyline gold",
        "carry every spark home",
        "final outro",
    ]
    assert updated[4].start == 12.0
    assert updated[5].end == 18.0


def test_update_transcript_text_expands_noisy_middle_chorus_before_double_chorus():
    segments = [
        TranscriptSegment(words=[WordTiming("verse", 0.0, 1.0), WordTiming("start", 1.0, 2.0)], text="verse start", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("verse", 2.0, 3.0), WordTiming("next", 3.0, 4.0)], text="verse next", start=2.0, end=4.0),
        TranscriptSegment(
            words=[WordTiming("glow", 4.0, 5.0), WordTiming("inside", 5.0, 6.0), WordTiming("the", 6.0, 6.5), WordTiming("night", 6.5, 7.0)],
            text="glow inside the night",
            start=4.0,
            end=7.0,
        ),
        TranscriptSegment(
            words=[WordTiming("carry", 7.0, 8.0), WordTiming("me", 8.0, 8.4), WordTiming("back", 8.4, 9.1), WordTiming("home", 9.1, 10.0)],
            text="carry me back home",
            start=7.0,
            end=10.0,
        ),
        TranscriptSegment(
            words=[WordTiming("solo", 10.0, 11.0), WordTiming("wandering", 11.0, 12.5), WordTiming("alone", 12.5, 13.5)],
            text="solo wandering alone",
            start=10.0,
            end=13.5,
        ),
        TranscriptSegment(
            words=[WordTiming("glowing", 13.5, 14.7), WordTiming("in", 14.7, 15.1), WordTiming("tonight", 15.1, 16.2)],
            text="glowing in tonight",
            start=13.5,
            end=16.2,
        ),
        TranscriptSegment(
            words=[WordTiming("carry", 16.2, 17.2), WordTiming("me", 17.2, 17.7), WordTiming("back", 17.7, 18.4), WordTiming("home", 18.4, 19.2)],
            text="carry me back home",
            start=16.2,
            end=19.2,
        ),
        TranscriptSegment(
            words=[WordTiming("glow", 19.2, 20.2), WordTiming("inside", 20.2, 21.2), WordTiming("the", 21.2, 21.7), WordTiming("night", 21.7, 22.2)],
            text="glow inside the night",
            start=19.2,
            end=22.2,
        ),
        TranscriptSegment(
            words=[WordTiming("carry", 22.2, 23.2), WordTiming("me", 23.2, 23.7), WordTiming("back", 23.7, 24.4), WordTiming("home", 24.4, 25.2)],
            text="carry me back home",
            start=22.2,
            end=25.2,
        ),
    ]

    updated = job_manager.update_transcript_text(
        segments,
        "\n".join(
            [
                "verse start",
                "verse next",
                "shine across the night",
                "bring me safely home",
                "solo wandering alone",
            ]
        ),
    )

    assert [segment.text for segment in updated] == [
        "verse start",
        "verse next",
        "shine across the night",
        "bring me safely home",
        "solo wandering alone",
        "shine across the night",
        "bring me safely home",
        "shine across the night",
        "bring me safely home",
    ]
    assert updated[5].start == 13.5
    assert updated[8].end == 25.2


def test_update_transcript_text_restores_tail_chorus_from_word_level_similarity():
    segments = [
        _segment_from_text("אני לא יודע מה עושים כאן בעולם זקן וכסיל מפריע משוטט כאן בין כולם ואומרים שיום אחד משהו יקרה", 0.0, 20.0),
        _segment_from_text("ולא תמיד יודע אם אני לאור מוכן ולא בטוח מה יקרה איתי מחר אבל בתוך תוכי יש אמונה קטנה", 20.0, 40.0),
        _segment_from_text("כי אין לי אין לי אחר מלבדו וגם לא יהיה מודה לו כל בוקר על מה שנותן לי נותן לי תקווה ואמונה אין לי מי להודות לו על מה שנותן לי את כל הדברים הטובים ממנו קיבלתי", 40.0, 72.0),
        _segment_from_text("ואין מעושן מימיהם", 72.0, 76.0),
        _segment_from_text("והוא רוצה לשמוע את קולך קורא אליו גם כסיל לא יפריע וזקן יאיר פניו וכמו בשעת הנעילה הדלת תיפתח והוא יופיע ויזכיר שלא אשכח כי אין", 78.0, 96.0),
        _segment_from_text("עוד אחר מלבדו מודה לו כל בוקר על מה שנותן לי נותן לי תקווה ואמונה להודות לו על מה את כל הדברים הטובים ממנו קיבלתי ואין מעושן ממנו", 96.0, 128.0),
    ]

    corrected_text = "\n".join(
        [
            "בית: אני לא יודע מה עושים כאן בעולם זקן וכסיל מפריע משוטט כאן בין כולם",
            "ואומרים שיום אחד משהו יקרה",
            "ולא תמיד יודע אם אני לאור מוכן ולא בטוח מה יקרה איתי מחר",
            "אבל בתוך תוכי יש אמונה קטנה",
            "כי אין לי אין לי אחר מלבדו וגם לא יהיה לי",
            "מודה לו כל בוקר על מה שנותן לי נותן לי תקווה אור ואמונה",
            "איו לי אין לי מילים להודות לו על מה שנותן לי",
            "את כל הדברים הטובים ממנו קיבלתי ואין מאושר ממני.",
            "והוא רוצה לשמוע את קולך קורא אליו גם כסיל לא יפריע וזקן יאיר פניו",
            "וכמו בשעת הנעילה הדלת תיפתח והוא יופיע ויזכיר שלא אשכח.",
        ]
    )

    updated = job_manager.update_transcript_text(segments, corrected_text)

    assert len(updated) == 14
    assert [segment.text for segment in updated[10:]] == [
        "כי אין לי אין לי אחר מלבדו וגם לא יהיה לי",
        "מודה לו כל בוקר על מה שנותן לי נותן לי תקווה אור ואמונה",
        "איו לי אין לי מילים להודות לו על מה שנותן לי",
        "את כל הדברים הטובים ממנו קיבלתי ואין מאושר ממני.",
    ]
    assert updated[10].start >= 96.0
    assert updated[-1].end == 128.0


def test_update_transcript_text_falls_back_when_line_alignment_creates_huge_mid_song_gap():
    segments = [
        _segment_from_text("כמה הדרך ארוכה", 13.36, 15.40),
        _segment_from_text("ברוך השם מחפש את התשובה", 15.59, 18.28),
        _segment_from_text("רוצה לך", 18.50, 19.82),
        _segment_from_text("כי אין מקום אחר ואין עוד זמן באמת ובתמים", 22.06, 27.44),
        _segment_from_text("כשהתעורקתי עם חיוך על הפנים לרוץ אליך", 27.86, 32.14),
        _segment_from_text("לא לפחד מחלומות הכל עובר מהר", 33.78, 39.94),
        _segment_from_text("כל עוד דולק הנר אפשר לתקן", 40.04, 44.60),
        _segment_from_text("אפשר לתקן", 46.15, 48.37),
        _segment_from_text("אז באתי אלה כמו ילד טוב עם לב שבור", 48.96, 54.29),
        _segment_from_text("חיים שלמים חולפים ברגע", 54.41, 56.97),
        _segment_from_text("כל דמעה היא כמו המבול", 57.23, 60.51),
        _segment_from_text("הנשמה שלי אומרת להתחיל להתפלל", 60.73, 65.26),
        _segment_from_text("וגם כשהאמת כואבת אתה מחזיק לי את היד", 66.68, 72.50),
        _segment_from_text("היום אני כבר לא דואג", 84.16, 86.28),
        _segment_from_text("אני אף פעם לא הייתי אסטרטג סומך עליך שתחבר אותי חזק לאדם באמת ובתמים אני לומד אני עובר פה שיעורים", 86.28, 115.73),
        _segment_from_text("אפשר לתקן כמו ילד עם לב שבור", 116.53, 124.36),
        _segment_from_text("חיים שלמים עוברים ברגע", 125.15, 127.79),
        _segment_from_text("כל דמעה היא כמו מבול", 127.89, 131.19),
        _segment_from_text("הנשמה שלי אומרת תתחיל להתפלל לבד", 131.41, 137.01),
        _segment_from_text("וגם כשהאמת כואבת אתה מחזיק לי את היד", 137.51, 142.83),
        _segment_from_text("כמו ילד עם לב שבור כמו ילד עם לב שבור חיים שלמים חולפים ברגע כל דמעה היא כמו מבול", 142.83, 176.97),
        _segment_from_text("הנשמה שלי אומרת להתחיל להתפלל לבד", 177.83, 183.18),
        _segment_from_text("וגם כשהאמת כואבת אתה מחזיק לי את היד", 183.82, 188.92),
    ]

    corrected_text = "\n".join(
        [
            "כמה הדרך ארוכה",
            "ברוך ה' מחפש את התשובה רוצה אליך",
            "כי אין מקום אחר ואין עוד זמן",
            "באמת ובתמים",
            "כשהתעוררתי עם חיוך על הפנים לרוץ אליך",
            "לא לפחד מחלומות",
            "הכל עובר מהר כל עוד דולק הנר",
            "אפשר לתקן אפשר לתקן",
            "אז באתי אליך כמו ילד טוב עם לב שבור",
            "חיים שלמים חולפים ברגע כל דמעה היא כמו מבול",
            "הנשמה שלי אומרת תתחיל להתפלל לבד",
            "וגם כשהאמת כואבת אתה מחזיק לי את היד",
            "היום אני כבר לא דואג",
            "אני אף פעם לא הייתי אסטרטג סומך עליך",
            "שתחבר אותי אותי חזק לאדמה",
            "באמת ובתמים אני לומד אני עובר פה שיעורים",
            "קרוב אליך",
            "במעשים במחשבות",
            "הכל עובר מהר כל עוד דולק הנר",
            "אפשר לתקן אפשר לתקן",
            "אז באתי אליך כמו ילד טוב עם לב שבור",
            "חיים שלמים חולפים ברגע כל דמעה היא כמו מבול",
            "הנשמה שלי אומרת תתחיל להתפלל לבד",
            "וגם כשהאמת כואבת אתה מחזיק לי את היד",
        ]
    )

    updated = job_manager.update_transcript_text(segments, corrected_text)
    max_gap = max(updated[index + 1].start - updated[index].end for index in range(len(updated) - 1))
    max_span = max(segment.end - segment.start for segment in updated)
    today_index = next(index for index, segment in enumerate(updated) if segment.text == "היום אני כבר לא דואג")
    repeated_line_count = sum(1 for segment in updated if segment.text == "וגם כשהאמת כואבת אתה מחזיק לי את היד")

    assert len(updated) >= 24
    assert max_gap < 25.0
    assert max_span < 18.0
    assert updated[today_index].start < 100.0
    assert round(updated[-1].end, 2) == 188.92
    assert repeated_line_count >= 2


def test_update_transcript_text_preserves_unmatched_tail_segments():
    segments = [
        _segment_from_text("intro line", 0.0, 2.0),
        _segment_from_text("middle line", 2.0, 4.0),
        _segment_from_text("tail line", 4.0, 6.0),
    ]

    updated = job_manager.update_transcript_text(
        segments,
        "\n".join(
            [
                "intro revised line",
                "middle line",
            ]
        ),
    )

    assert [segment.text for segment in updated] == [
        "intro revised line",
        "middle line",
        "tail line",
    ]
    assert updated[-1].start == 4.0
    assert updated[-1].end == 6.0


def test_detect_suspicious_review_shrink_allows_omitted_repeated_chorus_lines():
    segments = [
        TranscriptSegment(words=[WordTiming("verse", 0.0, 1.0), WordTiming("one", 1.0, 2.0)], text="verse one", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("chorus", 2.0, 3.0), WordTiming("bright", 3.0, 4.0), WordTiming("light", 4.0, 5.0)], text="chorus bright light", start=2.0, end=5.0),
        TranscriptSegment(words=[WordTiming("chorus", 5.0, 6.0), WordTiming("hold", 6.0, 7.0), WordTiming("on", 7.0, 8.0)], text="chorus hold on", start=5.0, end=8.0),
        TranscriptSegment(words=[WordTiming("second", 8.0, 9.0), WordTiming("verse", 9.0, 10.0)], text="second verse", start=8.0, end=10.0),
        TranscriptSegment(words=[WordTiming("chorus", 12.0, 13.0), WordTiming("bright", 13.0, 14.0), WordTiming("light", 14.0, 15.0)], text="chorus bright light", start=12.0, end=15.0),
        TranscriptSegment(words=[WordTiming("chorus", 15.0, 16.0), WordTiming("hold", 16.0, 17.0), WordTiming("on", 17.0, 18.0)], text="chorus hold on", start=15.0, end=18.0),
        TranscriptSegment(words=[WordTiming("final", 18.0, 19.0), WordTiming("outro", 19.0, 20.0)], text="final outro", start=18.0, end=20.0),
    ]

    warning = job_manager.detect_suspicious_review_shrink(
        segments,
        "\n".join(
            [
                "verse one",
                "chorus shining light",
                "chorus keep on",
                "second verse",
                "final outro",
            ]
        ),
    )

    assert warning is None


def test_detect_suspicious_review_shrink_allows_structure_aware_repeated_chorus_omission():
    segments = [
        TranscriptSegment(words=[WordTiming("verse", 0.0, 1.0), WordTiming("one", 1.0, 2.0)], text="verse one", start=0.0, end=2.0),
        TranscriptSegment(
            words=[WordTiming("alpha", 2.0, 3.0), WordTiming("beta", 3.0, 4.0), WordTiming("gamma", 4.0, 5.0)],
            text="alpha beta gamma",
            start=2.0,
            end=5.0,
        ),
        TranscriptSegment(
            words=[WordTiming("delta", 5.0, 6.0), WordTiming("epsilon", 6.0, 7.0), WordTiming("zeta", 7.0, 8.0)],
            text="delta epsilon zeta",
            start=5.0,
            end=8.0,
        ),
        TranscriptSegment(words=[WordTiming("second", 8.0, 9.0), WordTiming("verse", 9.0, 10.0)], text="second verse", start=8.0, end=10.0),
        TranscriptSegment(
            words=[WordTiming("alpha", 12.0, 13.0), WordTiming("beta", 13.0, 14.0), WordTiming("gamma", 14.0, 15.0)],
            text="alpha beta gamma",
            start=12.0,
            end=15.0,
        ),
        TranscriptSegment(
            words=[WordTiming("delta", 15.0, 16.0), WordTiming("epsilon", 16.0, 17.0), WordTiming("zeta", 17.0, 18.0)],
            text="delta epsilon zeta",
            start=15.0,
            end=18.0,
        ),
        TranscriptSegment(words=[WordTiming("final", 18.0, 19.0), WordTiming("outro", 19.0, 20.0)], text="final outro", start=18.0, end=20.0),
    ]

    warning = job_manager.detect_suspicious_review_shrink(
        segments,
        "\n".join(
            [
                "verse one",
                "paint the skyline gold",
                "carry every spark home",
                "second verse",
                "final outro",
            ]
        ),
    )

    assert warning is None


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


def test_cleanup_stale_jobs_removes_completed_and_stuck_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)

    completed = job_manager.create_job(title="done", input_type="audio_file")
    job_manager.update_status(completed, JobStatus.DONE)
    job_manager.update_review_status(completed, ReviewStatus.COMPLETED)
    completed.manifest.updated_at = (now - timedelta(hours=30)).isoformat()
    job_manager._write_json(completed.manifest_path, asdict(completed.manifest))

    failed = job_manager.create_job(title="failed", input_type="audio_file")
    job_manager.update_status(failed, JobStatus.ERROR)
    failed.manifest.updated_at = (now - timedelta(hours=96)).isoformat()
    job_manager._write_json(failed.manifest_path, asdict(failed.manifest))

    review = job_manager.create_job(title="review", input_type="audio_file")
    job_manager.update_status(review, JobStatus.AWAITING_REVIEW)
    job_manager.update_review_status(review, ReviewStatus.AWAITING_REVIEW)
    review.manifest.updated_at = (now - timedelta(hours=200)).isoformat()
    job_manager._write_json(review.manifest_path, asdict(review.manifest))

    removed = job_manager.cleanup_stale_jobs(now=now, completed_after_hours=24, stale_after_hours=72)

    assert {(item["job_id"], item["reason"]) for item in removed} == {
        (completed.job_id, "completed"),
        (failed.job_id, "error"),
    }
    assert not (tmp_path / completed.job_id).exists()
    assert not (tmp_path / failed.job_id).exists()
    assert (tmp_path / review.job_id).exists()


def test_is_cleanup_candidate_keeps_recent_review_job(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    job = job_manager.create_job(title="review", input_type="audio_file")
    job_manager.update_status(job, JobStatus.AWAITING_REVIEW)
    job_manager.update_review_status(job, ReviewStatus.AWAITING_REVIEW)
    job.manifest.updated_at = (now - timedelta(hours=2)).isoformat()
    job_manager._write_json(job.manifest_path, asdict(job.manifest))

    reason = job_manager.is_cleanup_candidate(job, now=now, completed_after_hours=24, stale_after_hours=72)

    assert reason is None


def test_song_analysis_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="song", input_type="audio_file")
    analysis = SongAnalysis(
        bpm=121.5,
        preview_window_seconds=0.5,
        provider="test",
        source_audio="instrumental.mp3",
        beat_times=[0.0, 0.5, 1.0],
        original_key="Em",
        target_key="Am",
        transpose_semitones=5,
        chord_sheet_text="כותרת: song\n\nAm\nlyrics\n",
        chord_source_name="Tab4U",
        chord_source_url="https://www.tab4u.com/tabs/songs/1_song.html",
        original_chord_events=[
            ChordEvent("Em", 0.0, 0.5, confidence=0.91, root="E", quality="minor"),
            ChordEvent("Am", 0.5, 1.0, confidence=0.88, root="A", quality="minor"),
        ],
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
    assert loaded.original_key == "Em"
    assert loaded.target_key == "Am"
    assert loaded.transpose_semitones == 5
    assert loaded.chord_sheet_text.startswith("כותרת: song")
    assert loaded.chord_source_name == "Tab4U"
    assert loaded.chord_source_url.endswith("/tabs/songs/1_song.html")
    assert [event.label for event in loaded.original_chord_events] == ["Em", "Am"]
    assert [event.label for event in loaded.chord_events] == ["C", "Am"]
    assert job.lyrics_with_chords_path.read_text(encoding="utf-8").startswith("C  Am")


def test_update_transcript_text_uses_neighboring_segments_when_equal_line_count_hides_merge():
    segments = [
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

    updated = job_manager.update_transcript_text(
        segments,
        "\n".join(
            [
                "when i tire dont believe me",
                "because its",
                "not me",
            ]
        ),
    )

    assert len(updated) == 3
    assert [word.word for word in updated[0].words] == ["when", "i", "tire", "dont", "believe", "me"]
    assert all(word.source != "review_hint" for word in updated[0].words)
    assert updated[0].start == 0.0
    assert updated[0].end == 2.4
    assert updated[0].words[3].start >= 1.2
    assert updated[1].start == 2.8
    assert updated[1].end == 3.5
    assert updated[2].start == 3.5
    assert updated[2].end == 4.0


def test_rebuild_segments_from_authoritative_text_heals_text_word_mismatches():
    reference_segments = [
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
    broken_segments = [
        TranscriptSegment(
            words=list(reference_segments[0].words),
            text="when i tire dont believe me",
            start=0.0,
            end=1.0,
        ),
        TranscriptSegment(
            words=reference_segments[1].words[:2],
            text="because its",
            start=1.2,
            end=2.0,
        ),
        TranscriptSegment(
            words=reference_segments[2].words[-2:],
            text="not me",
            start=3.5,
            end=4.0,
        ),
    ]

    mismatches_before = job_manager.find_segment_word_text_mismatches(broken_segments)
    healed = job_manager.rebuild_segments_from_authoritative_text(reference_segments, broken_segments)
    mismatches_after = job_manager.find_segment_word_text_mismatches(healed)

    assert mismatches_before
    assert not mismatches_after
    assert healed[0].text == "when i tire dont believe me"
    assert [word.word for word in healed[0].words] == ["when", "i", "tire", "dont", "believe", "me"]
    assert healed[0].end == 2.4
