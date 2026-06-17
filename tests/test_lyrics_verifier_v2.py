# tests/test_lyrics_verifier_v2.py
from unittest.mock import patch, MagicMock
from karaoke.models import (
    TranscriptDraft, TranscriptSegment, WordTiming,
    VerificationVerdict, ConsensusResult, DisputedLine,
)
from karaoke.lyrics_verifier import MultiStepLyricsVerifier


def _draft():
    """Create a test TranscriptDraft."""
    words1 = [
        WordTiming(word="שלום", start=0.0, end=0.3, confidence=0.9, source="draft_whisper", aligned=False),
        WordTiming(word="עולם", start=0.3, end=0.6, confidence=0.9, source="draft_whisper", aligned=False),
    ]
    words2 = [
        WordTiming(word="הלב", start=1.0, end=1.3, confidence=0.9, source="draft_whisper", aligned=False),
        WordTiming(word="שלי", start=1.3, end=1.6, confidence=0.9, source="draft_whisper", aligned=False),
    ]
    seg1 = TranscriptSegment(words=words1)
    seg2 = TranscriptSegment(words=words2)
    return TranscriptDraft(segments=[seg1, seg2], provider="test")


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
def test_consensus_reached_skips_gemini(mock_search):
    """When 3+ sources agree, Gemini step is skipped."""
    mock_search.return_value = {
        "shironet": ["שלום עולם", "הלב שלי"],
        "tab4u": ["שלום עולם", "הלב שלי"],
        "baneshama": ["שלום עולם", "הלב שלי"],
    }
    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())
    assert result.verdict == VerificationVerdict.CONSENSUS.value
    assert result.confidence >= 0.9

@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_deep_verify")
def test_no_consensus_triggers_gemini(mock_gemini, mock_search):
    """When <3 sources agree, Gemini is called."""
    mock_search.return_value = {
        "shironet": ["שלום עולם", "הלב שלי"],
        "tab4u": ["שלום עולם", "הלב שלך"],
    }
    mock_gemini.return_value = (["שלום עולם", "הלב שלי"], 0.85, [])
    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())
    mock_gemini.assert_called_once()
    assert result.verdict == VerificationVerdict.GEMINI_VERIFIED.value

@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_knowledge_verify")
def test_zero_sources_uses_whisper_with_gemini(mock_gemini_kb, mock_search):
    """When no sources found, Whisper transcript goes to Gemini knowledge-based check."""
    mock_search.return_value = {}
    mock_gemini_kb.return_value = (["שלום עולם", "הלב שלי"], 0.5)
    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())
    mock_gemini_kb.assert_called_once()
    assert result.verdict == VerificationVerdict.NO_SOURCES.value


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_knowledge_verify")
def test_zero_sources_exposes_review_options_without_auto_apply(mock_gemini_kb, mock_search):
    mock_search.return_value = {}
    mock_gemini_kb.return_value = (["שלום עולם", "הלב שלך"], 0.55)

    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())

    option_ids = {option["option_id"] for option in result.options}
    assert {"draft", "verified"} <= option_ids
    assert any(str(option_id).startswith("source_") for option_id in option_ids)
    assert result.selected_option_id == "draft"
    assert result.applied is False
    assert result.corrected_lines == ["שלום עולם", "הלב שלך"]


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_knowledge_verify")
def test_zero_sources_high_confidence_can_auto_apply_verified_words(mock_gemini_kb, mock_search):
    mock_search.return_value = {}
    mock_gemini_kb.return_value = (["שלום עולם", "הלב שלך"], 0.92)

    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())

    assert result.selected_option_id == "verified"
    assert result.applied is True
    assert result.correction_count > 0


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
def test_zero_sources_reports_actual_fallback_provider(mock_search):
    mock_search.return_value = {}
    verifier = MultiStepLyricsVerifier()
    verifier._llm_provider = "grok"
    verifier._llm_display_name = "Grok"

    def fake_knowledge_verify(_draft_text, _title):
        verifier._last_llm_provider_used = "gemini"
        verifier._last_llm_warning = "Grok לא זמין כרגע, בוצע fallback ל-Gemini."
        return ["׳©׳׳•׳ ׳¢׳•׳׳", "׳”׳׳‘ ׳©׳׳™"], 0.8

    verifier._gemini_knowledge_verify = fake_knowledge_verify
    result = verifier.verify("׳©׳™׳¨ ׳׳“׳•׳’׳׳”", _draft())

    assert result.llm_provider == "gemini"
    assert any("fallback" in warning.lower() for warning in (result.local_warnings or []))


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_deep_verify")
def test_gemini_failure_returns_best_available(mock_gemini, mock_search):
    """When Gemini API fails, return best available data with warning."""
    mock_search.return_value = {
        "shironet": ["שלום עולם", "הלב שלי"],
        "tab4u": ["שלום עולם", "הלב שלך"],
    }
    mock_gemini.side_effect = Exception("Gemini API timeout")
    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())
    assert result.corrected_lines is not None
    assert any("gemini" in w.lower() or "שגיאה" in w for w in (result.local_warnings or []))


def test_result_has_dispute_info():
    """Verify result contains dispute information for UI."""
    dispute = DisputedLine(
        line_number=1,
        versions={"shironet": "הלב שלי", "tab4u": "הלב שלך"},
        gemini_recommendation="הלב שלי",
        gemini_confidence=0.85,
    )
    consensus = ConsensusResult(
        consensus_reached=False,
        agreed_sources=2,
        lyrics=["שלום עולם", "הלב שלי"],
        disputes=[dispute],
    )
    assert len(consensus.disputes) == 1
    assert consensus.disputes[0].versions["tab4u"] == "הלב שלך"
