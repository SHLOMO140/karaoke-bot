from karaoke.aligner import AutoHebrewAligner
from karaoke.exceptions import AlignmentError
from karaoke.models import AlignedTranscript, CharacterTiming, SubWordTiming, TranscriptSegment, WordTiming


def _timed_word(text: str, start: float, end: float, *, aligned: bool = True, with_chars: bool = True) -> WordTiming:
    duration = max(end - start, 0.01)
    chars = list(text)
    subwords = []
    char_timings = []

    if with_chars and chars:
        cursor = start
        for index, char in enumerate(chars):
            next_cursor = end if index == len(chars) - 1 else start + ((index + 1) / len(chars)) * duration
            subwords.append(
                SubWordTiming(
                    text=char,
                    start=round(cursor, 6),
                    end=round(next_cursor, 6),
                    confidence=0.95,
                )
            )
            char_timings.append(
                CharacterTiming(
                    char=char,
                    start=round(cursor, 6),
                    end=round(next_cursor, 6),
                )
            )
            cursor = next_cursor

    return WordTiming(
        word=text,
        start=start,
        end=end,
        confidence=0.95 if aligned else 0.1,
        source="forced_aligner" if aligned else "review_hint",
        aligned=aligned,
        subwords=subwords,
        char_timings=char_timings,
    )


def _result(words: list[WordTiming]) -> AlignedTranscript:
    segment = TranscriptSegment(
        words=words,
        text=" ".join(word.word for word in words),
        start=words[0].start if words else 0.0,
        end=words[-1].end if words else 0.0,
    )
    return AlignedTranscript(
        segments=[segment],
        provider="test",
        fully_aligned=all(word.aligned for word in words),
        unaligned_word_count=sum(0 if word.aligned else 1 for word in words),
    )


def test_auto_aligner_falls_back_when_whisperx_raises():
    expected = _result([
        _timed_word("shalom", 0.0, 0.6),
        _timed_word("olam", 0.7, 1.2),
    ])
    auto = AutoHebrewAligner()

    class _BrokenPrimary:
        def align(self, *args, **kwargs):
            raise AlignmentError("broken", "broken")

    class _FallbackStub:
        def align(self, *args, **kwargs):
            return expected

    auto.primary = _BrokenPrimary()
    auto.fallback = _FallbackStub()

    aligned = auto.align("dummy.wav", expected.segments, expected.segments)

    assert aligned is expected
    assert auto.last_warning_message


def test_auto_aligner_keeps_primary_when_quality_is_strong():
    primary = _result([
        _timed_word("shalom", 0.0, 0.5),
        _timed_word("olam", 0.6, 1.1),
    ])
    auto = AutoHebrewAligner()
    calls = {"fallback": 0}

    class _PrimaryStub:
        def align(self, *args, **kwargs):
            return primary

    class _FallbackStub:
        def align(self, *args, **kwargs):
            calls["fallback"] += 1
            return _result([_timed_word("unused", 0.0, 0.4)])

    auto.primary = _PrimaryStub()
    auto.fallback = _FallbackStub()

    aligned = auto.align("dummy.wav", primary.segments, primary.segments)

    assert aligned is primary
    assert calls["fallback"] == 0
    assert auto.last_warning_message == ""


def test_auto_aligner_prefers_fallback_when_quality_is_better():
    weak_primary = _result([
        _timed_word("shalom", 0.0, 0.5, aligned=False, with_chars=False),
        _timed_word("olam", 0.5, 1.0, aligned=False, with_chars=False),
    ])
    strong_fallback = _result([
        _timed_word("shalom", 0.0, 0.52),
        _timed_word("olam", 0.55, 1.08),
    ])
    auto = AutoHebrewAligner()

    class _PrimaryStub:
        def align(self, *args, **kwargs):
            return weak_primary

    class _FallbackStub:
        def align(self, *args, **kwargs):
            return strong_fallback

    auto.primary = _PrimaryStub()
    auto.fallback = _FallbackStub()

    aligned = auto.align("dummy.wav", weak_primary.segments, weak_primary.segments)

    assert aligned is strong_fallback
    assert "נבחר" in auto.last_warning_message
