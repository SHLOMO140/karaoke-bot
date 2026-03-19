from karaoke.models import (
    CharacterTiming, ConsensusResult, DisputedLine,
    CharDiff, CharChange, VerificationVerdict,
)

def test_character_timing_creation():
    ct = CharacterTiming(char="שׁ", start=0.0, end=0.12)
    assert ct.char == "שׁ"
    assert ct.end - ct.start == 0.12

def test_consensus_result_with_consensus():
    cr = ConsensusResult(
        consensus_reached=True,
        agreed_sources=3,
        lyrics=["שורה אחת", "שורה שתיים"],
        disputes=[],
    )
    assert cr.consensus_reached
    assert cr.agreed_sources == 3
    assert len(cr.disputes) == 0

def test_consensus_result_without_consensus():
    dispute = DisputedLine(
        line_number=5,
        versions={"shironet": "הלב שלי", "tab4u": "הלב שלך"},
    )
    cr = ConsensusResult(
        consensus_reached=False,
        agreed_sources=1,
        lyrics=["הלב שלי"],
        disputes=[dispute],
    )
    assert not cr.consensus_reached
    assert dispute.gemini_recommendation is None
    assert dispute.gemini_confidence == 0.0

def test_char_diff():
    change = CharChange(position=3, old_char="וֹ", new_char="ֵ", change_type="replaced")
    diff = CharDiff(
        word_index=0,
        original_word="שָׁלוֹם",
        corrected_word="שָׁלֵם",
        char_changes=[change],
    )
    assert diff.char_changes[0].change_type == "replaced"
    assert diff.gemini_explanation is None

def test_verification_verdict_enum():
    assert VerificationVerdict.CONSENSUS == "consensus"
    assert VerificationVerdict.GEMINI_VERIFIED == "gemini_verified"
    assert VerificationVerdict.HUMAN_APPROVED == "human_approved"
    assert VerificationVerdict.NO_SOURCES == "no_sources"
