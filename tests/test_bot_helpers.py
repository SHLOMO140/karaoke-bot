from pathlib import Path

from bot import build_format_keyboard, chunk_text_for_telegram, filter_output_files


def test_build_format_keyboard_includes_direct_chords_option():
    keyboard = build_format_keyboard()
    callback_values = [button.callback_data for row in keyboard.inline_keyboard for button in row]

    assert "format:chords" in callback_values


def test_chunk_text_for_telegram_splits_long_blocks():
    chunks = chunk_text_for_telegram(("אקורדים\n" * 1200).strip(), limit=500)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_filter_output_files_keeps_only_chord_related_files_in_chords_mode():
    files = {
        "lyrics_with_chords.txt": Path("lyrics_with_chords.txt"),
        "song_analysis.json": Path("song_analysis.json"),
        "subtitles.srt": Path("subtitles.srt"),
    }

    filtered = filter_output_files(files, "chords_text")

    assert set(filtered) == {"lyrics_with_chords.txt", "song_analysis.json"}
