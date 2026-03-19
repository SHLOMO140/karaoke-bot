from karaoke.char_diff import compute_char_diffs
from karaoke.models import CharDiff

def test_diff_single_char_replacement():
    diffs = compute_char_diffs(
        original_words=["שלום"],
        corrected_words=["שלם"],
        word_indices=[0],
    )
    assert len(diffs) == 1
    assert diffs[0].original_word == "שלום"
    assert diffs[0].corrected_word == "שלם"
    assert any(c.change_type == "removed" for c in diffs[0].char_changes)

def test_diff_no_changes():
    diffs = compute_char_diffs(
        original_words=["שלום"],
        corrected_words=["שלום"],
        word_indices=[0],
    )
    assert len(diffs) == 0

def test_diff_multiple_words():
    diffs = compute_char_diffs(
        original_words=["הלב", "שלך"],
        corrected_words=["הלב", "שלי"],
        word_indices=[0, 1],
    )
    assert len(diffs) == 1
    assert diffs[0].word_index == 1
    assert diffs[0].original_word == "שלך"
    assert diffs[0].corrected_word == "שלי"

def test_diff_with_niqqud_change():
    diffs = compute_char_diffs(
        original_words=["שָׁלוֹם"],
        corrected_words=["שָׁלֵם"],
        word_indices=[0],
    )
    assert len(diffs) == 1
    assert len(diffs[0].char_changes) > 0
