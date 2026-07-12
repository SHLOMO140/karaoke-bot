"""Tests for the alignment-based consensus engine (L1).

Each fixture uses short dummy Hebrew lines (not real song lyrics). The gold
lyrics are identical across sources; only line SEGMENTATION differs — the
exact failure mode that made the positional engine useless in production.
"""

from karaoke.consensus import ConsensusEngine

PRIORITY = {"shironet.mako.co.il": 0, "tab4u.com": 1, "nagnu.co.il": 2}

GOLD = [
    "בוקר טוב לכולם",
    "השמש זורחת שוב",
    "הולכים ברחוב הראשי",
    "שרים את השיר הזה",
]


def test_positional_engine_fails_on_header_offset_but_aligned_succeeds():
    sources = {
        "shironet.mako.co.il": list(GOLD),
        "tab4u.com": ["מילים לשיר הזה", *GOLD],  # extra header line -> offset by 1
        "nagnu.co.il": list(GOLD),
    }

    positional = ConsensusEngine().evaluate(sources)
    aligned = ConsensusEngine().evaluate_aligned(sources, priority=PRIORITY)

    assert not positional.consensus_reached  # documents the old failure
    assert aligned.consensus_reached
    assert aligned.agreed_sources == 3
    assert aligned.lyrics == GOLD


def test_aligned_consensus_handles_merged_lines():
    merged = [GOLD[0] + " " + GOLD[1], GOLD[2], GOLD[3]]  # two lines merged into one
    sources = {
        "shironet.mako.co.il": list(GOLD),
        "tab4u.com": merged,
        "nagnu.co.il": list(GOLD),
    }

    aligned = ConsensusEngine().evaluate_aligned(sources, priority=PRIORITY)

    assert aligned.consensus_reached
    assert aligned.lyrics == GOLD


def test_aligned_consensus_ignores_niqqud_and_punctuation_variants():
    decorated = [
        "בּוֹקֶר טוֹב לְכולם,",
        "השמש זורחת שוב!",
        "הולכים ברחוב הראשי...",
        "שרים את השיר הזה.",
    ]
    sources = {
        "shironet.mako.co.il": list(GOLD),
        "tab4u.com": decorated,
        "nagnu.co.il": list(GOLD),
    }

    aligned = ConsensusEngine().evaluate_aligned(sources, priority=PRIORITY)

    assert aligned.consensus_reached
    assert aligned.agreed_sources == 3


def test_aligned_consensus_handles_collapsed_chorus():
    chorus = ["זה הפזמון שלנו", "שרים אותו ביחד"]
    full = [*GOLD[:2], *chorus, *GOLD[2:], *chorus]  # chorus sung twice
    collapsed = [*GOLD[:2], *chorus, *GOLD[2:]]  # source writes it once
    sources = {
        "shironet.mako.co.il": full,
        "tab4u.com": collapsed,
        "nagnu.co.il": full,
    }

    aligned = ConsensusEngine().evaluate_aligned(sources, priority=PRIORITY)

    assert aligned.consensus_reached
    assert aligned.lyrics == full


def test_aligned_consensus_representative_spelling_prefers_priority_domain():
    # nagnu spells with extra punctuation; shironet is the authority.
    sources = {
        "nagnu.co.il": ["בוקר טוב, לכולם!", *GOLD[1:]],
        "shironet.mako.co.il": list(GOLD),
        "tab4u.com": list(GOLD),
    }

    aligned = ConsensusEngine().evaluate_aligned(sources, priority=PRIORITY)

    assert aligned.consensus_reached
    assert aligned.lyrics[0] == GOLD[0]


def test_aligned_consensus_rejects_disagreeing_third_source():
    other_song = [
        "ערב רד על העיר",
        "הירח מציץ מלמעלה",
        "רוקדים עד אור הבוקר",
        "מחר נתחיל מחדש",
    ]
    sources = {
        "shironet.mako.co.il": list(GOLD),
        "tab4u.com": list(GOLD),
        "nagnu.co.il": other_song,
    }

    aligned = ConsensusEngine(min_sources=3).evaluate_aligned(sources, priority=PRIORITY)

    assert not aligned.consensus_reached
    assert aligned.agreed_sources == 2


def test_wrong_song_guard_rejects_dissimilar_consensus():
    from karaoke.lyrics_verifier import _consensus_matches_draft

    draft_text = "\n".join(GOLD)
    assert _consensus_matches_draft(GOLD, draft_text)
    unrelated = ["מילים אחרות לגמרי", "שאין להן שום קשר", "לשום שיר מוכר", "בעולם הזה בכלל"]
    assert not _consensus_matches_draft(unrelated, draft_text)
