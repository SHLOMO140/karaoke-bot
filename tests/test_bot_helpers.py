"""Tests for pure bot helpers (no Telegram network)."""

import bot


def test_is_youtube_url():
    assert bot.is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert bot.is_youtube_url("watch here https://www.youtube.com/watch?v=dQw4w9WgXcQ now")
    assert not bot.is_youtube_url("שיר של פאר טסי")


def test_video_id_extraction():
    assert bot._video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert bot._video_id("https://www.youtube.com/watch?v=abcdefghijk") == "abcdefghijk"
    assert bot._video_id("no url here") is None


def test_format_result():
    line = bot.format_result({"title": "A", "channel": "C", "duration": "1:05"})
    assert "A" in line and "C" in line and "1:05" in line


def test_build_results_keyboard_registers_songs():
    user_data: dict = {}
    results = [{"id": "vid1", "title": "T", "channel": "C", "duration": "2:00", "url": "u"}]
    kb = bot.build_results_keyboard(results, user_data)
    assert user_data["songs"]["vid1"] == {"url": "u", "title": "T"}
    assert kb.inline_keyboard[0][0].callback_data == "pick:vid1"


def test_menu_callback_data_shapes():
    assert bot.build_song_menu("v").inline_keyboard[0][0].callback_data == "chords:v"
    assert bot.build_song_menu("v").inline_keyboard[1][0].callback_data == "dl:v"
    quality_rows = bot.build_quality_menu("v").inline_keyboard
    assert quality_rows[0][0].callback_data == "q:v:best"
