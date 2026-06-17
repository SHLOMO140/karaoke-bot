from karaoke.models import SubWordTiming, TranscriptSegment, WordTiming
from karaoke.styles import get_style, normalize_style_preset
import re

from karaoke.subtitle_generator import AssKaraokeRenderer, _layout_segment_words, _subwords_for_word


def test_legacy_classic_preset_upgrades_to_reference_blue():
    assert normalize_style_preset("classic_hebrew") == "reference_blue"

    style = get_style("classic_hebrew")

    assert style.font_size >= 90
    assert style.outline_width >= 6
    assert style.base_fill_alpha == "00"
    assert style.effect_mode == "grapheme_highlight"
    assert style.secondary_color == "&H00FF7A00"
    assert style.word_spacing_scale >= 1.5


def test_layout_segment_words_keeps_visible_gap_between_words():
    style = get_style()
    segment = TranscriptSegment(
        words=[
            WordTiming("first", 0.0, 0.4),
            WordTiming("second", 0.45, 0.9),
        ],
        text="first second",
        start=0.0,
        end=0.9,
    )

    placed_words = _layout_segment_words(segment, style)
    right_word, left_word = sorted(placed_words, key=lambda item: item[1], reverse=True)
    gap = (right_word[1] - right_word[3] / 2) - (left_word[1] + left_word[3] / 2)

    assert gap >= style.font_size * 0.20


def test_ass_renderer_highlights_graphemes_with_bold_second_layer(tmp_path):
    ass_path = tmp_path / "karaoke.ass"
    style = get_style()
    segment = TranscriptSegment(
        words=[
            WordTiming("first", 0.0, 0.5),
            WordTiming("second", 0.55, 1.2),
        ],
        text="first second",
        start=0.0,
        end=1.2,
    )

    AssKaraokeRenderer().render([segment], str(ass_path), style=style)

    ass_text = ass_path.read_text(encoding="utf-8-sig")
    base_line = next(line for line in ass_text.splitlines() if line.startswith("Dialogue: 0"))
    active_line = next(line for line in ass_text.splitlines() if line.startswith("Dialogue: 1"))

    assert "Dialogue: 0" in ass_text
    assert "Dialogue: 1" in ass_text
    assert f"\\1c{style.secondary_color}" in ass_text
    assert "\\clip(" in ass_text
    assert "\\fad(" in ass_text
    assert "\\t(" in ass_text
    assert re.search(r"\\pos\((\d+),(\d+)\)", base_line).groups() == re.search(
        r"\\pos\((\d+),(\d+)\)",
        active_line,
    ).groups()


def test_ass_renderer_keeps_completed_graphemes_on_the_blue_layer(tmp_path):
    ass_path = tmp_path / "karaoke.ass"
    style = get_style()
    segment = TranscriptSegment(
        words=[
            WordTiming(
                "go",
                0.0,
                0.4,
                subwords=[SubWordTiming("go", 0.0, 0.4)],
            ),
            WordTiming(
                "now",
                0.6,
                1.0,
                subwords=[SubWordTiming("now", 0.6, 1.0)],
            ),
        ],
        text="go now",
        start=0.0,
        end=1.0,
    )

    AssKaraokeRenderer().render([segment], str(ass_path), style=style)

    ass_text = ass_path.read_text(encoding="utf-8-sig")
    persistent_lines = [
        line
        for line in ass_text.splitlines()
        if line.startswith("Dialogue: 1,0:00:00.20,0:00:01.00")
    ]

    assert persistent_lines
    assert all(f"\\1c{style.secondary_color}" in line for line in persistent_lines)
    assert any("\\fad(" not in line for line in persistent_lines)


def test_ass_renderer_trims_micro_overlap_from_base_layer(tmp_path):
    ass_path = tmp_path / "overlap.ass"
    style = get_style()
    segments = [
        TranscriptSegment(
            words=[WordTiming("A", 0.0, 1.04)],
            text="A",
            start=0.0,
            end=1.04,
        ),
        TranscriptSegment(
            words=[WordTiming("B", 1.01, 1.60)],
            text="B",
            start=1.01,
            end=1.60,
        ),
    ]

    AssKaraokeRenderer().render(segments, str(ass_path), style=style)

    ass_text = ass_path.read_text(encoding="utf-8-sig")
    first_base = next(line for line in ass_text.splitlines() if line.startswith("Dialogue: 0") and "A" in line)

    assert "0:00:01.01" in first_base


def test_subwords_are_normalized_back_into_the_parent_word_window():
    word = WordTiming(
        "hello",
        10.0,
        11.0,
        subwords=[
            SubWordTiming("he", 50.0, 50.5),
            SubWordTiming("llo", 50.5, 51.0),
        ],
    )

    normalized = _subwords_for_word(word)

    assert normalized[0].start == 10.0
    assert normalized[-1].end == 11.0
    assert all(10.0 <= subword.start <= subword.end <= 11.0 for subword in normalized)
