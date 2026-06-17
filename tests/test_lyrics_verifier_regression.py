from unittest.mock import patch

from karaoke.lyrics_verifier import MultiStepLyricsVerifier
from karaoke.models import TranscriptDraft, TranscriptSegment, WordTiming


def _draft() -> TranscriptDraft:
    first = TranscriptSegment(
        words=[
            WordTiming(word="shalom", start=0.0, end=0.3, confidence=0.9),
            WordTiming(word="olam", start=0.3, end=0.6, confidence=0.9),
        ],
        text="shalom olam",
        start=0.0,
        end=0.6,
    )
    second = TranscriptSegment(
        words=[
            WordTiming(word="halev", start=1.0, end=1.3, confidence=0.9),
            WordTiming(word="sheli", start=1.3, end=1.6, confidence=0.9),
        ],
        text="halev sheli",
        start=1.0,
        end=1.6,
    )
    return TranscriptDraft(segments=[first, second], provider="test")


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
def test_verify_passes_draft_object_into_source_search(mock_search):
    mock_search.return_value = {}
    verifier = MultiStepLyricsVerifier()
    draft = _draft()

    verifier.verify("demo song", draft)

    assert mock_search.call_count == 1
    assert mock_search.call_args.args[1] is draft
