import asyncio
from pathlib import Path
from types import SimpleNamespace

import bot
import pytest
from bot import (
    build_format_keyboard,
    chunk_text_for_telegram,
    filter_output_files,
    format_chord_sheet_for_telegram,
    has_current_chord_sheet,
    is_chord_sheet_chord_line,
    mirror_chord_line_for_telegram,
    normalize_chord_sheet_key_header,
    should_format_chord_sheet_for_telegram,
)
from karaoke import job_manager
from karaoke.models import ChordEvent, LyricsVerificationResult, ReviewStatus, SongAnalysis, TranscriptDraft, TranscriptSegment, WordTiming


def test_callback_job_id_extracts_job_scoped_actions():
    assert bot.callback_job_id("karaoke_review:abc123") == "abc123"
    assert bot.callback_job_id("karaoke_output:job42:subs_only") == "job42"
    assert bot.callback_job_id("format:mp3") is None


def test_build_format_keyboard_includes_direct_chords_option():
    keyboard = build_format_keyboard()
    callback_values = [button.callback_data for row in keyboard.inline_keyboard for button in row]

    assert "format:chords" in callback_values


def test_build_delivery_result_text_for_private_flow_includes_group_link():
    delivered_message = SimpleNamespace(link="https://t.me/example_group/123")

    text, parse_mode = bot.build_delivery_result_text(
        "שיר בדיקה",
        source_chat_id=6881356001,
        target_chat_id=-1003816867909,
        delivered_message=delivered_message,
    )

    assert "פתח את התוצאה בקבוצה" in text
    assert "https://t.me/example_group/123" in text
    assert "פקודות" not in text
    assert parse_mode == "HTML"


def test_build_delivery_result_text_for_same_chat_stays_plain():
    text, parse_mode = bot.build_delivery_result_text(
        "שיר בדיקה",
        source_chat_id=-1003816867909,
        target_chat_id=-1003816867909,
        delivered_message=SimpleNamespace(link="https://t.me/example_group/123"),
    )

    assert text == "הושלם בהצלחה עבור שיר בדיקה."
    assert parse_mode is None


def test_build_dynamic_review_keyboard_hides_original_transcript_option(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 1.0, confidence=0.9), WordTiming("world", 1.0, 2.0, confidence=0.95)],
            text="hello world",
            start=0.0,
            end=2.0,
        )
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=draft_segments, provider="fake"))
    job_manager.save_lyrics_verification(
        job,
        LyricsVerificationResult(
            provider="test",
            options=[
                {"option_id": "draft", "label": "draft", "lines": ["hello world"]},
                {"option_id": "verified", "label": "verified", "lines": ["hello brave world"]},
                {"option_id": "source_site", "label": "source", "lines": ["hello accurate world"]},
            ],
        ),
    )

    keyboard = bot.build_dynamic_review_keyboard(job)
    callback_values = [button.callback_data for row in keyboard.inline_keyboard for button in row]

    assert f"karaoke_option:{job.job_id}:draft" not in callback_values
    assert f"karaoke_option:{job.job_id}:verified" in callback_values
    assert f"karaoke_option:{job.job_id}:source_site" in callback_values


def test_build_review_text_marks_original_as_compare_only_when_alternatives_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 1.0, confidence=0.9), WordTiming("world", 1.0, 2.0, confidence=0.95)],
            text="hello world",
            start=0.0,
            end=2.0,
        )
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=draft_segments, provider="fake"))
    job_manager.save_lyrics_verification(
        job,
        LyricsVerificationResult(
            provider="test",
            options=[
                {"option_id": "draft", "label": "draft", "lines": ["hello world"]},
                {"option_id": "verified", "label": "verified", "lines": ["hello brave world"]},
            ],
        ),
    )

    text = bot._build_review_text(job)

    assert "להשוואה בלבד" in text
    assert "hello world" in text


def test_get_delivery_target_uses_default_group_for_private_requests(monkeypatch):
    monkeypatch.setattr(bot, "DEFAULT_DELIVERY_CHAT_ID", -1003816867909)
    monkeypatch.setattr(bot, "DEFAULT_DELIVERY_REPLY_TO_MESSAGE_ID", 0)

    target_chat_id, reply_to_message_id = bot.get_delivery_target(None, 6881356001)

    assert target_chat_id == -1003816867909
    assert reply_to_message_id == 0


def test_get_delivery_target_keeps_explicit_group_context_over_default(monkeypatch):
    monkeypatch.setattr(bot, "DEFAULT_DELIVERY_CHAT_ID", -1003816867909)
    monkeypatch.setattr(bot, "DEFAULT_DELIVERY_REPLY_TO_MESSAGE_ID", 0)

    target_chat_id, reply_to_message_id = bot.get_delivery_target(
        {"delivery_chat_id": -1002223334445, "delivery_reply_to_message_id": 77},
        6881356001,
    )

    assert target_chat_id == -1002223334445
    assert reply_to_message_id == 77


def test_requires_group_delivery_approval_only_for_private_to_group():
    assert bot.requires_group_delivery_approval(6881356001, -1003816867909) is True
    assert bot.requires_group_delivery_approval(-1003816867909, -1003816867909) is False
    assert bot.requires_group_delivery_approval(6881356001, 6881356001) is False


def test_chunk_text_for_telegram_splits_long_blocks():
    chunks = chunk_text_for_telegram(("אקורדים\n" * 1200).strip(), limit=500)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_is_chord_sheet_chord_line_accepts_chord_rows_only():
    assert is_chord_sheet_chord_line("Em            Am") is True
    assert is_chord_sheet_chord_line("E  E4   F#  F#4") is True
    assert is_chord_sheet_chord_line("פתיחה:") is False
    assert is_chord_sheet_chord_line("בדד, במשעול אל האין") is False


def test_mirror_chord_line_for_telegram_reverses_visual_order():
    assert mirror_chord_line_for_telegram("A                Bm") == "Bm                A"


def test_format_chord_sheet_for_telegram_mirrors_only_chord_lines():
    formatted = format_chord_sheet_for_telegram("A                Bm\nבדד, במשעול אל האין\n\nפתיחה:")
    lines = formatted.splitlines()

    assert lines[0] == "Bm                A"
    assert lines[1] == "בדד, במשעול אל האין"
    assert lines[2] == ""
    assert lines[3] == "פתיחה:"


def test_should_format_chord_sheet_for_telegram_only_for_hebrew_text():
    assert should_format_chord_sheet_for_telegram("A      Bm\nבדד") is True
    assert should_format_chord_sheet_for_telegram("A      Bm\nlonely road") is False


def test_normalize_chord_sheet_key_header_replaces_existing_key_lines():
    text = (
        "כותרת: demo\n"
        "קצב: לא ידוע\n"
        "משקל: 4/4\n"
        "סולם מקור: F#\n"
        "סולם קל: Am\n\n"
        "A                Bm\n"
    )

    normalized = normalize_chord_sheet_key_header(text, original_key="Eb", target_key="A")

    assert "סולם מקור: Eb" in normalized
    assert "סולם קל: A" in normalized
    assert "סולם מקור: F#" not in normalized
    assert "סולם קל: Am" not in normalized


def test_filter_output_files_keeps_only_chord_related_files_in_chords_mode():
    files = {
        "lyrics_with_chords.txt": Path("lyrics_with_chords.txt"),
        "song_analysis.json": Path("song_analysis.json"),
        "subtitles.srt": Path("subtitles.srt"),
    }

    filtered = filter_output_files(files, "chords_text")

    assert set(filtered) == {"lyrics_with_chords.txt"}


def test_filter_output_files_hides_music_artifacts_in_default_mode():
    files = {
        "transcript.txt": Path("transcript.txt"),
        "timings.json": Path("timings.json"),
        "subtitles.srt": Path("subtitles.srt"),
        "karaoke.ass": Path("karaoke.ass"),
        "song_analysis.json": Path("song_analysis.json"),
        "lyrics_with_chords.txt": Path("lyrics_with_chords.txt"),
        "final_video.mp4": Path("final_video.mp4"),
    }

    filtered = filter_output_files(files, "default")

    assert set(filtered) == {
        "transcript.txt",
        "timings.json",
        "subtitles.srt",
        "karaoke.ass",
        "final_video.mp4",
    }


def test_has_current_chord_sheet_requires_matching_analysis_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    job.lyrics_with_chords_path.write_text("Am\nlyrics", encoding="utf-8")
    job_manager.save_song_analysis(
        job,
        SongAnalysis(
            provider="librosa_harmony_v5",
            chord_events=[ChordEvent("Am", 0.0, 1.0, confidence=0.9, root="A", quality="minor")],
        ),
    )

    assert has_current_chord_sheet(job, "librosa_harmony_v5") is True
    assert has_current_chord_sheet(job, "librosa_harmony_v2") is False


def test_has_current_chord_sheet_accepts_external_chord_text_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    job.lyrics_with_chords_path.write_text("Am\nlyrics", encoding="utf-8")
    job_manager.save_song_analysis(
        job,
        SongAnalysis(
            provider="librosa_harmony_v5",
            chord_sheet_text="כותרת: demo\n\nAm\nlyrics\n",
        ),
    )

    assert has_current_chord_sheet(job, "librosa_harmony_v5") is True


def test_has_current_chord_sheet_rejects_low_confidence_audio_only_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    job.lyrics_with_chords_path.write_text("Am\nlyrics", encoding="utf-8")
    job_manager.save_song_analysis(
        job,
        SongAnalysis(
            provider="librosa_harmony_v5",
            chord_events=[
                ChordEvent("Am", 0.0, 4.0, confidence=0.31, root="A", quality="minor"),
                ChordEvent("C", 4.0, 8.0, confidence=0.28, root="C", quality="major"),
                ChordEvent("Dm", 8.0, 12.0, confidence=0.34, root="D", quality="minor"),
            ],
        ),
    )

    assert has_current_chord_sheet(job, "librosa_harmony_v5") is False


def test_send_chords_text_response_allows_low_confidence_audio_only_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    job.lyrics_with_chords_path.write_text("Am\nlyrics", encoding="utf-8")
    job_manager.save_song_analysis(
        job,
        SongAnalysis(
            provider="librosa_harmony_v5",
            chord_events=[
                ChordEvent("Am", 0.0, 4.0, confidence=0.31, root="A", quality="minor"),
                ChordEvent("C", 4.0, 8.0, confidence=0.28, root="C", quality="major"),
                ChordEvent("Dm", 8.0, 12.0, confidence=0.34, root="D", quality="minor"),
            ],
        ),
    )

    delivered: dict[str, object] = {}

    async def fake_send_document_to_chat(bot_instance, **kwargs):
        delivered["chat_id"] = kwargs["chat_id"]
        delivered["filename"] = kwargs["filename"]
        delivered["caption"] = kwargs["caption"]
        return SimpleNamespace(link="https://t.me/example_group/123")

    monkeypatch.setattr(bot, "send_document_to_chat", fake_send_document_to_chat)

    message = _FakeMessage()
    result = asyncio.run(bot.send_chords_text_response(message, job, include_preview_chunks=False))
    assert result.link == "https://t.me/example_group/123"
    assert delivered["chat_id"] == 11
    assert delivered["filename"].endswith("_lyrics_with_chords.txt")
    assert delivered["caption"]
    return

    assert "לא הצלחתי לייצר אקורדים מספיק אמינים לפרסום" in exc_info.value.info.user_message


def test_send_chords_text_response_formats_preview_chunks_for_rtl(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    job.lyrics_with_chords_path.write_text(
        "כותרת: demo\n"
        "קצב: לא ידוע\n"
        "משקל: 4/4\n"
        "סולם מקור: F#\n"
        "סולם קל: Am\n\n"
        "A                Bm\n"
        "בדד, במשעול אל האין\n",
        encoding="utf-8",
    )
    job_manager.save_song_analysis(
        job,
        SongAnalysis(
            original_key="F#",
            target_key="Am",
            transpose_semitones=6,
            original_chord_events=[
                ChordEvent("Eb", 0.0, 1.0, root="Eb"),
                ChordEvent("Fm", 1.0, 2.0, root="F"),
                ChordEvent("Eb", 2.0, 3.0, root="Eb"),
                ChordEvent("Fm", 3.0, 4.0, root="F"),
                ChordEvent("Eb", 4.0, 5.0, root="Eb"),
                ChordEvent("Ab", 5.0, 6.0, root="Ab"),
                ChordEvent("C", 6.0, 7.0, root="C"),
                ChordEvent("C#", 7.0, 8.0, root="C#"),
            ],
            chord_events=[
                ChordEvent("A", 0.0, 1.0, root="A"),
                ChordEvent("Bm", 1.0, 2.0, root="B"),
                ChordEvent("A", 2.0, 3.0, root="A"),
                ChordEvent("Bm", 3.0, 4.0, root="B"),
                ChordEvent("A", 4.0, 5.0, root="A"),
                ChordEvent("D", 5.0, 6.0, root="D"),
                ChordEvent("F#", 6.0, 7.0, root="F#"),
                ChordEvent("G", 7.0, 8.0, root="G"),
            ],
        ),
    )

    delivered: dict[str, object] = {}
    async def fake_send_document_to_chat(bot_instance, **kwargs):
        delivered["document"] = kwargs["document"].read().decode("utf-8")
        return SimpleNamespace(link="https://t.me/example_group/123")

    monkeypatch.setattr(bot, "send_document_to_chat", fake_send_document_to_chat)

    message = _FakeMessage()
    asyncio.run(bot.send_chords_text_response(message, job))

    assert message.replies
    assert message.reply_kwargs[0]["parse_mode"] == "HTML"
    assert "<pre>אקורדים + מילים עבור: demo" in message.replies[0]
    assert "Bm                A" in message.replies[0]
    assert "בדד, במשעול אל האין" in message.replies[0]
    assert "Bm                A" in delivered["document"]
    assert "בדד, במשעול אל האין" in delivered["document"]
    assert "סולם מקור: Eb" in delivered["document"]
    assert "סולם קל: A" in delivered["document"]


def test_generate_direct_chords_output_prefers_fast_external_sheet_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(bot, "run_storage_maintenance", lambda: None)

    captured: dict[str, object] = {}

    async def fake_edit_or_reply(message, text, reply_markup=None, parse_mode=None):
        captured.setdefault("messages", []).append(text)

    async def fake_send_chords_text_response(message, job, **kwargs):
        captured["job_id"] = job.job_id
        captured["title"] = job.display_name
        captured["sheet"] = job.lyrics_with_chords_path.read_text(encoding="utf-8")
        return SimpleNamespace(link="https://t.me/example_group/123")

    async def fake_show_delivery_result(message, title, *, target_chat_id, delivered_message=None):
        captured["result_title"] = title
        captured["result_target_chat_id"] = target_chat_id
        captured["result_link"] = delivered_message.link if delivered_message else ""

    monkeypatch.setattr(bot, "edit_or_reply", fake_edit_or_reply)
    monkeypatch.setattr(bot, "send_chords_text_response", fake_send_chords_text_response)
    monkeypatch.setattr(bot, "show_delivery_result", fake_show_delivery_result)
    monkeypatch.setattr(bot, "cleanup_delivered_job", lambda job: captured.setdefault("cleaned_job_id", job.job_id))
    monkeypatch.setattr(bot, "requires_group_delivery_approval", lambda source_chat_id, target_chat_id: False)
    monkeypatch.setattr(
        bot,
        "lookup_external_chord_sheet_by_title",
        lambda title, provider, target_key="": SongAnalysis(
            provider=provider,
            chord_source_name="Tab4U",
            chord_source_url="https://www.tab4u.com/tabs/songs/1_demo.html",
            original_key="Em",
            target_key=target_key,
            transpose_semitones=0,
            chord_sheet_text="כותרת: Demo Song\nקצב: לא ידוע\nמשקל: 4/4\nסולם מקור: Em\n\nEm\nlyrics\n",
        ),
    )

    class _FailPipeline:
        def __init__(self, job):
            self.job = job
            self.song_analyzer = SimpleNamespace(name="librosa_harmony_v5")

        def step_get_audio(self, *_args, **_kwargs):
            raise AssertionError("fast external chord lookup should skip audio download")

    monkeypatch.setattr(bot, "KaraokePipeline", _FailPipeline)

    context = SimpleNamespace(
        user_data={
            "chosen": {
                "title": "Demo Song",
                "url": "https://www.youtube.com/watch?v=demo123",
                "delivery_chat_id": 11,
            },
            "active_user_id": 22,
        }
    )
    message = _FakeMessage()

    asyncio.run(bot.generate_direct_chords_output(message, context))

    assert captured["title"] == "Demo Song"
    assert "סולם מקור: Em" in str(captured["sheet"])
    assert captured["result_title"] == "Demo Song"
    assert captured["result_target_chat_id"] == 11
    assert "נמצא דף אקורדים חיצוני" in "\n".join(str(item) for item in captured["messages"])
    assert job_manager.load_job(str(captured["job_id"])).song_analysis_path.exists()


def test_build_output_delivery_metadata_uses_song_title_for_video_with_vocals(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="שיר בדיקה", input_type="audio_file", chat_id=11, user_id=22)

    filename, caption = bot.build_output_delivery_metadata(job, "final_video.mp4", Path("final_video.mp4"))

    assert filename == "שיר בדיקה - כתוביות רצות.mp4"
    assert caption == "שיר בדיקה - כתוביות רצות"


def test_build_output_delivery_metadata_uses_song_title_for_instrumental_video(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="שיר בדיקה", input_type="audio_file", chat_id=11, user_id=22)

    filename, caption = bot.build_output_delivery_metadata(job, "final_video_instrumental.mp4", Path("final_video_instrumental.mp4"))

    assert filename == "שיר בדיקה - קריוקי.mp4"
    assert caption == "שיר בדיקה - קריוקי"


def test_build_legacy_audio_delivery_metadata_uses_song_title(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="אייל גולן - בית מזכוכית", input_type="youtube", chat_id=11, user_id=22)

    filename, caption = bot.build_legacy_audio_delivery_metadata(job)

    assert filename == "אייל גולן - בית מזכוכית.mp3"
    assert caption == "אייל גולן - בית מזכוכית"


def test_build_legacy_audio_delivery_metadata_marks_karaoke(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="אייל גולן - בית מזכוכית", input_type="youtube", chat_id=11, user_id=22)

    filename, caption = bot.build_legacy_audio_delivery_metadata(job, karaoke=True)

    assert filename == "אייל גולן - בית מזכוכית - קריוקי ללא ווקאל.mp3"
    assert caption == "קריוקי ללא ווקאל: אייל גולן - בית מזכוכית"


class _FakeMessage:
    def __init__(self):
        self.chat_id = 11
        self.replies: list[str] = []
        self.reply_kwargs: list[dict[str, object]] = []
        self.text = ""
        self._bot = SimpleNamespace()

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)
        self.reply_kwargs.append(kwargs)
        return SimpleNamespace(text=text, kwargs=kwargs)

    def get_bot(self):
        return self._bot


def test_handle_karaoke_correction_rebuilds_against_draft_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
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
    distorted_review = [
        TranscriptSegment(
            words=[
                WordTiming("hello", 0.0, 5.0, confidence=0.9),
                WordTiming("world", 5.0, 10.0, confidence=0.95),
            ],
            text="hello world",
            start=0.0,
            end=10.0,
        )
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=draft_segments, provider="fake"))
    job_manager.save_review_transcript(job, distorted_review)

    captured: dict[str, object] = {}

    async def fake_show_review_text(message, refreshed_job, note=None):
        captured["job_id"] = refreshed_job.job_id
        captured["note"] = note

    monkeypatch.setattr(bot, "show_review_text", fake_show_review_text)

    update = SimpleNamespace(message=_FakeMessage())
    asyncio.run(bot.handle_karaoke_correction(update, job, "1: hello brave world"))

    saved = job_manager.load_review_segments(job)
    assert saved[0].end == 2.2
    assert saved[0].words[-1].end == 2.2
    assert [word.word for word in saved[0].words] == ["hello", "brave", "world"]
    assert saved[0].words[1].char_timings
    assert captured["job_id"] == job.job_id


def test_handle_message_in_group_routes_to_private_handoff(monkeypatch):
    called: dict[str, object] = {}

    async def fake_handoff_group_request(update, context, *, request_kind, request_payload):
        called["kind"] = request_kind
        called["payload"] = request_payload

    async def fake_get_active_review_job(update):
        called["review_checked"] = True
        return None

    monkeypatch.setattr(bot, "handoff_group_request", fake_handoff_group_request)
    monkeypatch.setattr(bot, "get_active_review_job", fake_get_active_review_job)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=22),
        effective_chat=SimpleNamespace(id=-1001, type="group"),
        message=SimpleNamespace(text="demo song", chat_id=-1001),
    )
    context = SimpleNamespace(user_data={})

    asyncio.run(bot.handle_message(update, context))

    assert called["kind"] == "text"
    assert called["payload"] == {"text": "demo song"}
    assert "review_checked" not in called


def test_handle_message_in_delivery_feedback_mode_saves_text_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    captured: dict[str, object] = {}

    async def fake_save_delivery_feedback_text(message, context, passed_job, text, *, source):
        captured["job_id"] = passed_job.job_id
        captured["text"] = text
        captured["source"] = source

    async def fail_get_active_review_job(update):
        raise AssertionError("review flow should not run while waiting for delivery feedback")

    monkeypatch.setattr(bot, "save_delivery_feedback_text", fake_save_delivery_feedback_text)
    monkeypatch.setattr(bot, "get_active_review_job", fail_get_active_review_job)

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=22),
        effective_chat=SimpleNamespace(id=11, type="private"),
        message=SimpleNamespace(text="הפזמון לא מדויק", chat_id=11),
    )
    context = SimpleNamespace(user_data={"delivery_feedback_job_id": job.job_id})

    asyncio.run(bot.handle_message(update, context))

    assert captured == {
        "job_id": job.job_id,
        "text": "הפזמון לא מדויק",
        "source": "text",
    }


def test_save_delivery_feedback_text_applies_line_edit_and_regenerates_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22, delivery_chat_id=-100123)
    segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 1.0), WordTiming("world", 1.0, 2.0)],
            text="hello world",
            start=0.0,
            end=2.0,
        ),
        TranscriptSegment(
            words=[WordTiming("old", 2.0, 3.0), WordTiming("chorus", 3.0, 4.0)],
            text="old chorus",
            start=2.0,
            end=4.0,
        ),
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=segments, provider="fake"))
    job_manager.save_review_transcript(job, segments)
    job_manager.update_pending_delivery(job, status="awaiting_feedback", delivery_mode="default", target_chat_id=-100123)
    captured: dict[str, object] = {}

    async def fake_generate_karaoke_output(query, context, passed_job, video_request):
        captured["query_message"] = query.message
        captured["job_id"] = passed_job.job_id
        captured["output_mode"] = context.user_data[f"output_mode:{passed_job.job_id}"]
        captured["delivery_mode"] = context.user_data[f"delivery_mode:{passed_job.job_id}"]
        captured["video_request"] = video_request

    class _FakeMessage:
        chat_id = 11
        from_user = SimpleNamespace(id=22)

        async def reply_text(self, text, **kwargs):
            captured.setdefault("replies", []).append(text)

    monkeypatch.setattr(bot, "generate_karaoke_output", fake_generate_karaoke_output)

    context = SimpleNamespace(user_data={"delivery_feedback_job_id": job.job_id})
    message = _FakeMessage()

    asyncio.run(bot.save_delivery_feedback_text(message, context, job, "שורה 2: corrected chorus", source="text"))

    refreshed = job_manager.load_job(job.job_id)
    review_segments = job_manager.load_review_segments(refreshed)
    assert [segment.text for segment in review_segments] == ["hello world", "corrected chorus"]
    assert captured["job_id"] == job.job_id
    assert captured["query_message"] is message
    assert captured["output_mode"] == "rerender"
    assert captured["delivery_mode"] == "default"
    assert captured["video_request"] is None
    assert "delivery_feedback_job_id" not in context.user_data
    assert refreshed.pending_delivery["status"] == "repairing"
    assert refreshed.pending_delivery["auto_repair_kind"] == "review_line_edits"


def test_save_delivery_feedback_text_rerenders_when_human_feedback_mentions_timing(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22, delivery_chat_id=-100123)
    segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 1.0), WordTiming("world", 1.0, 2.0)],
            text="hello world",
            start=0.0,
            end=2.0,
        )
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=segments, provider="fake"))
    job_manager.save_review_transcript(job, segments)
    job.vocals_16k_path.write_bytes(b"fake wav")
    job_manager.update_pending_delivery(job, status="awaiting_feedback", delivery_mode="default", target_chat_id=-100123)
    captured: dict[str, object] = {}

    async def fake_generate_karaoke_output(query, context, passed_job, video_request):
        captured["job_id"] = passed_job.job_id
        captured["output_mode"] = context.user_data[f"output_mode:{passed_job.job_id}"]
        captured["delivery_mode"] = context.user_data[f"delivery_mode:{passed_job.job_id}"]
        captured["video_request"] = video_request

    class _FakeMessage:
        chat_id = 11
        from_user = SimpleNamespace(id=22)

        async def reply_text(self, text, **kwargs):
            captured.setdefault("replies", []).append(text)

    monkeypatch.setattr(bot, "generate_karaoke_output", fake_generate_karaoke_output)

    context = SimpleNamespace(user_data={"delivery_feedback_job_id": job.job_id})
    message = _FakeMessage()
    feedback = "\u05d4\u05de\u05d9\u05dc\u05d9\u05dd \u05dc\u05d0 \u05d1\u05d6\u05de\u05df"

    asyncio.run(bot.save_delivery_feedback_text(message, context, job, feedback, source="text"))

    refreshed = job_manager.load_job(job.job_id)
    review_segments = job_manager.load_review_segments(refreshed)
    assert [segment.text for segment in review_segments] == ["hello world"]
    assert captured["job_id"] == job.job_id
    assert captured["output_mode"] == "rerender"
    assert captured["delivery_mode"] == "default"
    assert captured["video_request"] is None
    assert refreshed.pending_delivery["status"] == "repairing"
    assert refreshed.pending_delivery["auto_repair_kind"] == "timing_realign"


def test_handle_callback_prevents_approving_reference_text_when_alternatives_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 1.0, confidence=0.9), WordTiming("world", 1.0, 2.0, confidence=0.95)],
            text="hello world",
            start=0.0,
            end=2.0,
        )
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=draft_segments, provider="fake"))
    job_manager.save_lyrics_verification(
        job,
        LyricsVerificationResult(
            provider="test",
            options=[
                {"option_id": "draft", "label": "draft", "lines": ["hello world"]},
                {"option_id": "verified", "label": "verified", "lines": ["hello brave world"]},
            ],
            selected_option_id="draft",
        ),
    )

    captured: dict[str, object] = {}

    async def fake_show_review_text(message, refreshed_job, note=None):
        captured["job_id"] = refreshed_job.job_id
        captured["note"] = note

    async def fake_edit_or_reply(message, text, reply_markup=None, parse_mode=None):
        captured["edit_text"] = text

    class _FakeQuery:
        def __init__(self):
            self.data = f"karaoke_approve:{job.job_id}"
            self.message = SimpleNamespace(chat_id=11)
            self.answered = False

        async def answer(self, *args, **kwargs):
            self.answered = True

    monkeypatch.setattr(bot, "show_review_text", fake_show_review_text)
    monkeypatch.setattr(bot, "edit_or_reply", fake_edit_or_reply)

    query = _FakeQuery()
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=22))
    context = SimpleNamespace(user_data={})

    asyncio.run(bot.handle_callback(update, context))

    refreshed = job_manager.load_job(job.job_id)
    assert query.answered is True
    assert refreshed.review_status == ReviewStatus.AWAITING_REVIEW
    assert "להשוואה בלבד" in str(captured.get("note", ""))
    assert "edit_text" not in captured


def test_handle_callback_delivery_reject_enters_feedback_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    job_manager.update_pending_delivery(job, status="pending_approval", target_chat_id=-100123, delivery_mode="default")
    captured: dict[str, object] = {}

    async def fake_send_delivery_feedback_template(message, passed_job):
        captured["template_job_id"] = passed_job.job_id

    async def fake_edit_or_reply(message, text, reply_markup=None, parse_mode=None):
        captured["prompt"] = text

    class _FakeQuery:
        def __init__(self):
            self.data = f"delivery_reject:{job.job_id}"
            self.message = SimpleNamespace(chat_id=11)
            self.answers: list[tuple[tuple[object, ...], dict[str, object]]] = []

        async def answer(self, *args, **kwargs):
            self.answers.append((args, kwargs))

    monkeypatch.setattr(bot, "send_delivery_feedback_template", fake_send_delivery_feedback_template)
    monkeypatch.setattr(bot, "edit_or_reply", fake_edit_or_reply)

    query = _FakeQuery()
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=22))
    context = SimpleNamespace(user_data={})

    asyncio.run(bot.handle_callback(update, context))

    refreshed = job_manager.load_job(job.job_id)
    assert context.user_data["delivery_feedback_job_id"] == job.job_id
    assert refreshed.pending_delivery["status"] == "awaiting_feedback"
    assert captured["template_job_id"] == job.job_id
    assert "לא אפרסם אותה לקבוצה עדיין" in captured["prompt"]
    assert query.answers[0][0][0] == "לא אפרסם לקבוצה עד שתשלח מה צריך לתקן."


def test_handle_callback_delivery_approve_publishes_and_clears_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(
        title="demo",
        input_type="audio_file",
        chat_id=11,
        user_id=22,
        delivery_chat_id=-100123,
    )
    job_manager.update_review_status(job, ReviewStatus.APPROVED)
    job_manager.update_pending_delivery(job, status="pending_approval", target_chat_id=-100123, delivery_mode="default")
    captured: dict[str, object] = {}

    async def fake_publish_job_to_group(message, passed_job):
        captured["published_job_id"] = passed_job.job_id
        return SimpleNamespace(link="https://t.me/example_group/321")

    async def fake_show_delivery_result(message, title, *, target_chat_id, delivered_message=None):
        captured["title"] = title
        captured["target_chat_id"] = target_chat_id
        captured["link"] = delivered_message.link

    async def fake_edit_or_reply(message, text, reply_markup=None, parse_mode=None):
        captured.setdefault("messages", []).append(text)

    def fake_cleanup_delivered_job(passed_job):
        captured["cleaned_job_id"] = passed_job.job_id

    class _FakeQuery:
        def __init__(self):
            self.data = f"delivery_approve:{job.job_id}"
            self.message = SimpleNamespace(chat_id=11)
            self.answers: list[tuple[tuple[object, ...], dict[str, object]]] = []

        async def answer(self, *args, **kwargs):
            self.answers.append((args, kwargs))

    monkeypatch.setattr(bot, "publish_job_to_group", fake_publish_job_to_group)
    monkeypatch.setattr(bot, "show_delivery_result", fake_show_delivery_result)
    monkeypatch.setattr(bot, "edit_or_reply", fake_edit_or_reply)
    monkeypatch.setattr(bot, "cleanup_delivered_job", fake_cleanup_delivered_job)

    query = _FakeQuery()
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=22))
    context = SimpleNamespace(user_data={"delivery_feedback_job_id": job.job_id})

    asyncio.run(bot.handle_callback(update, context))

    refreshed = job_manager.load_job(job.job_id)
    assert refreshed.review_status == ReviewStatus.COMPLETED
    assert refreshed.pending_delivery == {}
    assert "delivery_feedback_job_id" not in context.user_data
    assert captured["published_job_id"] == job.job_id
    assert captured["cleaned_job_id"] == job.job_id
    assert captured["target_chat_id"] == -100123
    assert captured["link"] == "https://t.me/example_group/321"


def test_handoff_group_request_anonymous_sender_requires_manual_claim(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_group_request(**kwargs):
        captured["request"] = kwargs
        return "token123"

    class _Message:
        def __init__(self):
            self.message_id = 77
            self.sender_chat = SimpleNamespace(id=-1001)
            self.replies: list[tuple[str, object]] = []

        async def reply_text(self, text, **kwargs):
            self.replies.append((text, kwargs.get("reply_markup")))
            return SimpleNamespace(text=text, kwargs=kwargs)

    monkeypatch.setattr(job_manager, "create_group_request", fake_create_group_request)
    monkeypatch.setattr(bot, "run_storage_maintenance", lambda: None)

    message = _Message()
    update = SimpleNamespace(
        message=message,
        effective_user=SimpleNamespace(id=22),
        effective_chat=SimpleNamespace(id=-1001, type="group"),
    )
    context = SimpleNamespace(bot=SimpleNamespace(get_me=lambda: None))

    async def fake_get_me():
        return SimpleNamespace(username="karaoke_bot")

    context.bot.get_me = fake_get_me

    asyncio.run(
        bot.handoff_group_request(
            update,
            context,
            request_kind="text",
            request_payload={"text": "demo song"},
        )
    )

    assert captured["request"]["user_id"] == 0
    assert message.replies
    text, reply_markup = message.replies[0]
    assert "אדמין אנונימי" in text
    assert reply_markup.inline_keyboard[0][0].callback_data == "group_claim:token123"


def test_handle_karaoke_correction_full_text_uses_draft_segments_when_review_is_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("alpha", 0.0, 1.0, confidence=0.9), WordTiming("verse", 1.0, 2.0, confidence=0.95)],
            text="alpha verse",
            start=0.0,
            end=2.0,
        ),
        TranscriptSegment(
            words=[WordTiming("beta", 2.0, 3.0, confidence=0.9), WordTiming("hook", 3.0, 4.0, confidence=0.95)],
            text="beta hook",
            start=2.0,
            end=4.0,
        ),
        TranscriptSegment(
            words=[WordTiming("gamma", 4.0, 5.0, confidence=0.9), WordTiming("outro", 5.0, 6.0, confidence=0.95)],
            text="gamma outro",
            start=4.0,
            end=6.0,
        ),
    ]
    truncated_review = draft_segments[:2]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=draft_segments, provider="fake"))
    job_manager.save_review_transcript(job, truncated_review)

    captured: dict[str, object] = {}

    async def fake_show_review_text(message, refreshed_job, note=None):
        captured["job_id"] = refreshed_job.job_id
        captured["note"] = note

    monkeypatch.setattr(bot, "show_review_text", fake_show_review_text)

    update = SimpleNamespace(message=_FakeMessage())
    asyncio.run(
        bot.handle_karaoke_correction(
            update,
            job,
            "\n".join(
                [
                    "alpha revised verse",
                    "beta revised hook",
                    "gamma revised outro",
                ]
            ),
        )
    )

    saved = job_manager.load_review_segments(job)
    assert len(saved) == 3
    assert saved[0].text == "alpha revised verse"
    assert saved[1].text == "beta revised hook"
    assert saved[2].text == "gamma revised outro"
    assert saved[2].end == 6.0
    assert captured["job_id"] == job.job_id


def test_generate_karaoke_output_rerender_realigns_when_audio_available(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 1.0), WordTiming("world", 1.0, 2.0)],
            text="hello world",
            start=0.0,
            end=2.0,
        )
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=draft_segments, provider="fake"))
    job_manager.save_review_transcript(job, draft_segments)
    job_manager.update_review_status(job, ReviewStatus.APPROVED)

    called: dict[str, object] = {}

    class _FakePipeline:
        def __init__(self, passed_job):
            self.job = passed_job

        def can_realign_after_review(self):
            return True

        def run_after_review(self, approved_segments, video_request):
            called["run_after_review"] = [segment.text for segment in approved_segments]
            return {"transcript.txt": job.transcript_path}

        def rerender_existing_outputs(self, video_request):
            called["rerender_existing_outputs"] = True
            return {}

        def download_youtube_video(self, quality):
            called["download_youtube_video"] = quality

    async def fake_edit_or_reply(message, text, reply_markup=None, parse_mode=None):
        called.setdefault("messages", []).append(text)

    async def fake_send_output_files(query, sent_job, output_files):
        called["output_files"] = dict(output_files)

    monkeypatch.setattr(bot, "KaraokePipeline", _FakePipeline)
    monkeypatch.setattr(bot, "edit_or_reply", fake_edit_or_reply)
    monkeypatch.setattr(bot, "send_output_files", fake_send_output_files)

    context = SimpleNamespace(
        user_data={
            f"output_mode:{job.job_id}": "rerender",
            f"delivery_mode:{job.job_id}": "default",
            "active_user_id": 22,
        }
    )
    query = SimpleNamespace(message=SimpleNamespace(chat_id=11))

    asyncio.run(bot.generate_karaoke_output(query, context, job, None))

    assert called["run_after_review"] == ["hello world"]
    assert "rerender_existing_outputs" not in called


def test_generate_karaoke_output_cleans_job_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file", chat_id=11, user_id=22)
    draft_segments = [
        TranscriptSegment(
            words=[WordTiming("hello", 0.0, 1.0), WordTiming("world", 1.0, 2.0)],
            text="hello world",
            start=0.0,
            end=2.0,
        )
    ]
    job_manager.save_draft_transcript(job, TranscriptDraft(segments=draft_segments, provider="fake"))
    job_manager.save_review_transcript(job, draft_segments)
    job_manager.update_review_status(job, ReviewStatus.APPROVED)

    class _FakePipeline:
        def __init__(self, passed_job):
            self.job = passed_job

        def can_realign_after_review(self):
            return True

        def run_after_review(self, approved_segments, video_request):
            return {"transcript.txt": job.transcript_path}

        def rerender_existing_outputs(self, video_request):
            return {"transcript.txt": job.transcript_path}

        def download_youtube_video(self, quality):
            return None

    async def fake_edit_or_reply(message, text, reply_markup=None, parse_mode=None):
        return None

    async def fake_send_output_files(query, sent_job, output_files):
        sent_job.transcript_path.write_text("lyrics", encoding="utf-8")

    monkeypatch.setattr(bot, "KaraokePipeline", _FakePipeline)
    monkeypatch.setattr(bot, "edit_or_reply", fake_edit_or_reply)
    monkeypatch.setattr(bot, "send_output_files", fake_send_output_files)

    context = SimpleNamespace(
        user_data={
            f"output_mode:{job.job_id}": "rerender",
            f"delivery_mode:{job.job_id}": "default",
            "active_user_id": 22,
        }
    )
    query = SimpleNamespace(message=SimpleNamespace(chat_id=11))

    asyncio.run(bot.generate_karaoke_output(query, context, job, None))

    assert not (tmp_path / job.job_id).exists()
