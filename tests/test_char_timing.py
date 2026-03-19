from karaoke.transcriber import interpolate_character_timings
from karaoke.models import WordTiming, CharacterTiming

def test_interpolate_simple_word():
    word = WordTiming(word="שלום", start=0.0, end=0.4, confidence=0.9, source="draft_whisper", aligned=False)
    chars = interpolate_character_timings(word)
    assert len(chars) == 4  # ש ל ו ם
    assert chars[0].start == 0.0
    assert chars[-1].end == 0.4
    # All chars should cover the full duration without gaps
    for i in range(len(chars) - 1):
        assert abs(chars[i].end - chars[i + 1].start) < 0.001

def test_interpolate_word_with_niqqud():
    word = WordTiming(word="שָׁלוֹם", start=0.0, end=0.6, confidence=0.9, source="draft_whisper", aligned=False)
    chars = interpolate_character_timings(word)
    # All chars should still cover the full duration without gaps
    assert chars[0].start == 0.0
    assert chars[-1].end == 0.6
    total = sum(c.end - c.start for c in chars)
    assert abs(total - 0.6) < 0.001
    # When regex lib is available, niqqud clusters with base consonants (no standalone niqqud).
    # When not available, standalone niqqud chars receive slightly higher weight (+0.18)
    # than plain consonants (1.0), reflecting their phonetic contribution.
    niqqud_durations = [c.end - c.start for c in chars if '\u05B0' <= c.char <= '\u05C8']
    if niqqud_durations:
        # Standalone niqqud graphemes exist; each should have positive duration
        assert all(d > 0 for d in niqqud_durations)

def test_interpolate_preserves_boundaries():
    word = WordTiming(word="אבג", start=1.5, end=2.1, confidence=0.9, source="draft_whisper", aligned=False)
    chars = interpolate_character_timings(word)
    assert chars[0].start == 1.5
    assert chars[-1].end == 2.1
