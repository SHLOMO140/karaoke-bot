from karaoke.aligner import realign_changed_words
from karaoke.models import WordTiming, CharacterTiming
from karaoke.transcriber import interpolate_character_timings


def test_realign_redistributes_timing_for_shorter_word():
    original = WordTiming(
        word="שלום", start=0.0, end=0.4,
        confidence=0.9, source="draft_whisper", aligned=True,
    )
    original.char_timings = interpolate_character_timings(original)

    new_timings = realign_changed_words(
        original_word=original,
        corrected_text="שלם",
        audio_path=None,  # skip audio verification in this test
    )
    assert len(new_timings) == 3  # ש ל ם
    assert new_timings[0].start == 0.0
    assert new_timings[-1].end == 0.4


def test_realign_preserves_word_boundaries():
    original = WordTiming(
        word="אבגד", start=1.0, end=2.0,
        confidence=0.9, source="draft_whisper", aligned=True,
    )
    original.char_timings = interpolate_character_timings(original)

    new_timings = realign_changed_words(
        original_word=original,
        corrected_text="אבגדה",
        audio_path=None,
    )
    assert new_timings[0].start == 1.0
    assert new_timings[-1].end == 2.0
