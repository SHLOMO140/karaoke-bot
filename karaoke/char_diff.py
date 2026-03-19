"""Character-level diff engine for Hebrew text."""

import difflib

from .models import CharChange, CharDiff


def compute_char_diffs(
    original_words: list[str],
    corrected_words: list[str],
    word_indices: list[int],
) -> list[CharDiff]:
    """Compute character-level diffs between original and corrected words."""
    diffs: list[CharDiff] = []

    for orig, corrected, idx in zip(original_words, corrected_words, word_indices):
        if orig == corrected:
            continue

        changes = _diff_chars(orig, corrected)
        diffs.append(CharDiff(
            word_index=idx,
            original_word=orig,
            corrected_word=corrected,
            char_changes=changes,
        ))

    return diffs


def _diff_chars(old: str, new: str) -> list[CharChange]:
    """Compute per-character changes between two strings."""
    changes: list[CharChange] = []
    matcher = difflib.SequenceMatcher(None, old, new)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            for pos in range(max(i2 - i1, j2 - j1)):
                old_c = old[i1 + pos] if i1 + pos < i2 else ""
                new_c = new[j1 + pos] if j1 + pos < j2 else ""
                if old_c and new_c:
                    changes.append(CharChange(position=i1 + pos, old_char=old_c, new_char=new_c, change_type="replaced"))
                elif old_c:
                    changes.append(CharChange(position=i1 + pos, old_char=old_c, new_char="", change_type="removed"))
                else:
                    changes.append(CharChange(position=j1 + pos, old_char="", new_char=new_c, change_type="added"))
        elif tag == "delete":
            for pos in range(i1, i2):
                changes.append(CharChange(position=pos, old_char=old[pos], new_char="", change_type="removed"))
        elif tag == "insert":
            for pos in range(j1, j2):
                changes.append(CharChange(position=pos, old_char="", new_char=new[pos], change_type="added"))

    return changes


def format_diff_table(diffs: list[CharDiff]) -> str:
    """Format diffs as a readable table for Gemini/user display."""
    lines = []
    for d in diffs:
        lines.append(f"מילה #{d.word_index}: \"{d.original_word}\" → \"{d.corrected_word}\"")
        for c in d.char_changes:
            if c.change_type == "replaced":
                lines.append(f"  מיקום {c.position}: '{c.old_char}' → '{c.new_char}'")
            elif c.change_type == "removed":
                lines.append(f"  מיקום {c.position}: '{c.old_char}' הוסרה")
            elif c.change_type == "added":
                lines.append(f"  מיקום {c.position}: '{c.new_char}' נוספה")
    return "\n".join(lines)
