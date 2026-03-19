"""Integration tests for the full lyrics verification pipeline."""
from unittest.mock import patch, MagicMock
from karaoke.models import TranscriptDraft, TranscriptSegment, WordTiming, VerificationVerdict
from karaoke.lyrics_verifier import MultiStepLyricsVerifier


def _make_draft(lines: list[str]) -> TranscriptDraft:
    """Create a TranscriptDraft from lines of text."""
    segments = []
    t = 0.0
    for line in lines:
        words = []
        for w in line.split():
            words.append(WordTiming(word=w, start=t, end=t + 0.3, confidence=0.9, source="draft_whisper", aligned=False))
            t += 0.3
        segments.append(TranscriptSegment(words=words))
        t += 0.5
    return TranscriptDraft(segments=segments, provider="test")


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
def test_full_consensus_flow(mock_search):
    """Happy path: 3 sources agree → consensus → high confidence."""
    lyrics = ["שלום עולם", "הלב שלי", "שיר יפה"]
    mock_search.return_value = {
        "shironet": lyrics,
        "tab4u": lyrics,
        "baneshama": lyrics,
    }
    verifier = MultiStepLyricsVerifier()
    draft = _make_draft(lyrics)
    result = verifier.verify("שיר לדוגמה", draft)

    assert result.verdict == VerificationVerdict.CONSENSUS.value
    assert result.confidence >= 0.9
    assert result.corrected_lines == lyrics


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_deep_verify")
def test_dispute_flow_with_gemini(mock_gemini, mock_search):
    """Dispute path: sources disagree → Gemini decides."""
    mock_search.return_value = {
        "shironet": ["שלום עולם", "הלב שלי"],
        "tab4u": ["שלום עולם", "הלב שלך"],
    }
    mock_gemini.return_value = (["שלום עולם", "הלב שלי"], 0.85, [])

    verifier = MultiStepLyricsVerifier()
    draft = _make_draft(["שלום עולם", "הלב שלי"])
    result = verifier.verify("שיר לדוגמה", draft)

    assert result.verdict == VerificationVerdict.GEMINI_VERIFIED.value
    assert result.corrected_lines == ["שלום עולם", "הלב שלי"]


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_knowledge_verify")
def test_no_sources_flow(mock_gemini_kb, mock_search):
    """No sources: falls back to Gemini knowledge-based check."""
    mock_search.return_value = {}
    mock_gemini_kb.return_value = (["שלום עולם"], 0.5)

    verifier = MultiStepLyricsVerifier()
    draft = _make_draft(["שלום עולם"])
    result = verifier.verify("שיר לדוגמה", draft)

    assert result.verdict == VerificationVerdict.NO_SOURCES.value
    assert result.confidence <= 0.7


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
def test_consensus_result_stored_in_verification(mock_search):
    """Verify that consensus_result is populated in the LyricsVerificationResult."""
    lyrics = ["אני שר", "לך שיר"]
    mock_search.return_value = {
        "shironet": lyrics,
        "tab4u": lyrics,
        "nagnu": lyrics,
        "baneshama": lyrics,
    }
    verifier = MultiStepLyricsVerifier()
    draft = _make_draft(lyrics)
    result = verifier.verify("שיר", draft)

    assert result.consensus_result is not None
    assert result.consensus_result.consensus_reached is True
    assert result.consensus_result.agreed_sources >= 3
    assert result.source_versions is not None
    assert len(result.source_versions) == 4
