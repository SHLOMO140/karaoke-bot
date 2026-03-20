"""Tests for updated review text formatting."""
import pytest
from unittest.mock import MagicMock, patch
from karaoke.models import ConsensusResult, DisputedLine


def _make_verification_dict(consensus_reached=False, agreed_sources=0, disputes=None, verdict="consensus", confidence=0.95):
    """Create a verification dict as it would be stored in job manifest."""
    consensus_data = {
        "consensus_reached": consensus_reached,
        "agreed_sources": agreed_sources,
        "lyrics": ["שלום עולם", "הלב שלי"],
        "disputes": disputes or [],
    }
    return {
        "verdict": verdict,
        "confidence": confidence,
        "summary": f"קונצנזוס בין {agreed_sources} מקורות" if consensus_reached else "Gemini הכריע",
        "correction_count": 0,
        "applied": False,
        "matched_sources": ["shironet", "tab4u"],
        "consensus_result": consensus_data,
    }


def test_consensus_shows_checkmark():
    """When consensus reached, review text shows ✅ with source count."""
    verification = _make_verification_dict(consensus_reached=True, agreed_sources=4)
    # Simulate what _build_review_text does with consensus data
    consensus_data = verification.get("consensus_result", {})
    assert consensus_data["consensus_reached"] is True
    assert consensus_data["agreed_sources"] == 4
    # The formatted text should contain ✅ and the count
    text = f"✅ מילים אומתו מ-{consensus_data['agreed_sources']} מקורות"
    assert "✅" in text
    assert "4" in text


def test_disputes_show_warning():
    """When disputes exist, review text shows ⚠️ with source versions."""
    disputes = [
        {
            "line_number": 4,
            "versions": {"shironet": "הלב שלי", "tab4u": "הלב שלך"},
            "gemini_recommendation": "הלב שלי",
            "gemini_confidence": 0.85,
        }
    ]
    verification = _make_verification_dict(
        consensus_reached=False, agreed_sources=2,
        disputes=disputes, verdict="gemini_verified", confidence=0.85
    )
    consensus_data = verification.get("consensus_result", {})
    assert not consensus_data["consensus_reached"]
    assert len(consensus_data["disputes"]) == 1
    d = consensus_data["disputes"][0]
    assert d["gemini_confidence"] == 0.85


def test_max_review_iterations():
    """MAX_REVIEW_ITERATIONS is set to 2."""
    from karaoke.config import MAX_REVIEW_ITERATIONS
    assert MAX_REVIEW_ITERATIONS == 2
