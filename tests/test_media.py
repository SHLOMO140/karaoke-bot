"""Tests for the lean media module."""

from unittest.mock import MagicMock, patch

from karaoke import media


def test_search_youtube_maps_entries():
    fake_info = {"entries": [
        {"id": "abc123", "title": "Song A", "channel": "Chan", "duration": 65},
    ]}
    fake_ydl = MagicMock()
    fake_ydl.__enter__.return_value.extract_info.return_value = fake_info
    with patch.object(media.yt_dlp, "YoutubeDL", return_value=fake_ydl):
        results = media.search_youtube("q", max_results=5)
    assert results[0]["id"] == "abc123"
    assert results[0]["duration"] == "1:05"
    assert results[0]["url"] == "https://www.youtube.com/watch?v=abc123"
    assert results[0]["channel"] == "Chan"


def test_search_youtube_skips_entries_without_id():
    fake_info = {"entries": [{"title": "no id"}, {"id": "x", "title": "ok", "duration": 0}]}
    fake_ydl = MagicMock()
    fake_ydl.__enter__.return_value.extract_info.return_value = fake_info
    with patch.object(media.yt_dlp, "YoutubeDL", return_value=fake_ydl):
        results = media.search_youtube("q")
    assert [r["id"] for r in results] == ["x"]
    assert results[0]["duration"] == "0:00"


def test_video_format_selector_maps_quality():
    assert media._video_format("best") == "bestvideo+bestaudio/best"
    assert media._video_format("720") == "bestvideo[height<=720]+bestaudio/best[height<=720]"
    assert media._video_format("1080") == "bestvideo[height<=1080]+bestaudio/best[height<=1080]"


def test_split_artist_and_title():
    assert media.split_artist_and_title("פאר טסי - סלינה") == ("פאר טסי", "סלינה")
    assert media.split_artist_and_title("justtitle") == ("", "justtitle")


def test_safe_filename_strips_illegal_chars():
    assert media.safe_filename('a/b:c*?"<>|') == "a_b_c______"
