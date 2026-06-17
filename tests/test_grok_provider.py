from unittest.mock import patch

import karaoke.config as karaoke_config
from karaoke.lyrics_verifier import MultiStepLyricsVerifier
from karaoke.models import TranscriptDraft, TranscriptSegment, WordTiming


def _draft() -> TranscriptDraft:
    segment = TranscriptSegment(
        words=[
            WordTiming(word="hello", start=0.0, end=0.4, confidence=0.9, source="draft_whisper", aligned=False),
            WordTiming(word="world", start=0.4, end=0.8, confidence=0.9, source="draft_whisper", aligned=False),
        ],
        text="hello world",
        start=0.0,
        end=0.8,
    )
    return TranscriptDraft(segments=[segment], provider="test")


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_knowledge_verify")
def test_grok_provider_retags_result(mock_gemini_kb, mock_search, monkeypatch):
    mock_search.return_value = {}
    mock_gemini_kb.return_value = (["hello world"], 0.7)

    monkeypatch.setattr(karaoke_config, "LYRICS_LLM_PROVIDER", "grok")
    monkeypatch.setattr(karaoke_config, "XAI_API_KEY", "test-key")
    monkeypatch.setattr(karaoke_config, "XAI_MODEL", "grok-4")

    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("test song", _draft())

    option_ids = {option["option_id"] for option in result.options}
    labels = [str(option.get("label", "")) for option in result.options]

    assert result.llm_provider == "grok"
    assert "source_grok" in option_ids
    assert any("Grok" in label for label in labels)
    assert "Grok" in result.summary
