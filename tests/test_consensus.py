from karaoke.consensus import ConsensusEngine

def test_consensus_reached_with_3_matching_sources():
    sources = {
        "shironet": ["שורה אחת", "שורה שתיים"],
        "tab4u": ["שורה אחת", "שורה שתיים"],
        "baneshama": ["שורה אחת", "שורה שתיים"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert result.consensus_reached
    assert result.agreed_sources == 3
    assert result.lyrics == ["שורה אחת", "שורה שתיים"]
    assert len(result.disputes) == 0

def test_no_consensus_with_2_sources():
    sources = {
        "shironet": ["שורה אחת"],
        "tab4u": ["שורה אחת"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert not result.consensus_reached
    assert result.agreed_sources == 2

def test_dispute_detected_on_disagreement():
    sources = {
        "shironet": ["הלב שלי", "שורה שתיים"],
        "tab4u": ["הלב שלך", "שורה שתיים"],
        "baneshama": ["הלב שלי", "שורה שתיים"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert not result.consensus_reached  # not 100% on all lines
    assert len(result.disputes) == 1
    assert result.disputes[0].line_number == 0
    assert "shironet" in result.disputes[0].versions

def test_normalization_ignores_niqqud():
    sources = {
        "shironet": ["שָׁלוֹם"],
        "tab4u": ["שלום"],
        "baneshama": ["שלום"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert result.consensus_reached  # same after niqqud removal

def test_normalization_ignores_punctuation_and_whitespace():
    sources = {
        "shironet": [" שלום,  עולם! "],
        "tab4u": ["שלום עולם"],
        "baneshama": ["שלום עולם."],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert result.consensus_reached

def test_empty_sources():
    engine = ConsensusEngine()
    result = engine.evaluate({})
    assert not result.consensus_reached
    assert result.agreed_sources == 0

def test_mixed_language_normalization():
    sources = {
        "shironet": ["Hello שלום"],
        "tab4u": ["hello שלום"],
        "baneshama": ["HELLO שלום"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert result.consensus_reached  # case-insensitive for Latin

def test_partial_consensus_per_line():
    sources = {
        "shironet": ["שורה אחת", "שורה שתיים", "שורה שלוש"],
        "tab4u": ["שורה אחת", "שורה אחרת", "שורה שלוש"],
        "baneshama": ["שורה אחת", "שורה שתיים", "שורה שלוש"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert not result.consensus_reached
    assert len(result.disputes) == 1
    assert result.disputes[0].line_number == 1
    assert result.lyrics[0] == "שורה אחת"
    assert result.lyrics[2] == "שורה שלוש"
