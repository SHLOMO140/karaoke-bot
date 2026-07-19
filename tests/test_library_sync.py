"""Tests for the inline-[Chord] converter and best-effort Supabase upsert."""

import asyncio

from karaoke import library_sync
from karaoke.chord_sources import _ChordToken, _LyricWord, _ParsedTab4USheet


def _sheet(pairs, chord_labels=None):
    return _ParsedTab4USheet(
        source_url="u", tables=[], lyric_lines=[],
        line_word_pairs=pairs, chord_labels=chord_labels or [],
    )


def test_inline_places_chords_at_columns():
    tokens = [_ChordToken(label="Dmaj7", column=0), _ChordToken(label="F#m", column=4)]
    words = [_LyricWord(text="אני", column=0, global_index=0),
             _LyricWord(text="והיא", column=4, global_index=1)]
    out = library_sync.to_inline_chords(_sheet([(tokens, words)]))
    assert out == "[Dmaj7]אני [F#m]והיא"


def test_inline_chord_only_line_uses_pipe_separator():
    tokens = [_ChordToken(label="Bm", column=0), _ChordToken(label="A", column=6)]
    out = library_sync.to_inline_chords(_sheet([(tokens, [])]))
    assert out == "[Bm]  |  [A]"


def test_upsert_skips_when_no_credentials(monkeypatch):
    monkeypatch.setattr(library_sync, "SUPABASE_URL", "")
    monkeypatch.setattr(library_sync, "SUPABASE_ANON_KEY", "")
    monkeypatch.setattr(library_sync, "SUPABASE_SYNC_TOKEN", "")
    assert asyncio.run(library_sync.upsert_song("t", "a", "C", "content")) is None
