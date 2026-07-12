import re

from karaoke.models import (
    ChordEvent,
    SingerAnalysisResult,
    SingerProfile,
    SingerSegmentAssignment,
    SongAnalysis,
    TranscriptSegment,
    WordTiming,
)
from karaoke.exceptions import SubtitleGenerationError
from karaoke.styles import get_style
from karaoke.subtitle_generator import (
    AssKaraokeRenderer,
    SrtRenderer,
    _build_render_chunks,
    _format_ass_interval,
    _format_srt_time,
    _split_segment_words,
)
from karaoke.subtitle_guardian import AssSubtitleGuardian


def _parse_ass_centiseconds(stamp: str) -> int:
    hours, minutes, rest = stamp.split(":")
    seconds, centis = rest.split(".")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 100 + int(centis)


def test_format_ass_interval_rounds_half_up_not_floor():
    start, end = _format_ass_interval(1.004, 1.506)
    assert start == "0:00:01.00"  # 1.004 rounds down
    assert end == "0:00:01.51"  # 1.506 rounds up (old int() floored to 1.50)


def test_format_ass_interval_enforces_min_duration():
    start, end = _format_ass_interval(2.0, 2.001)
    assert _parse_ass_centiseconds(end) == _parse_ass_centiseconds(start) + 1


def test_format_ass_interval_adjacent_words_never_overlap():
    boundaries = [0.0, 0.333, 0.6667, 1.004, 1.3333, 1.9995, 2.5049]
    previous_end = 0
    for interval_start, interval_end in zip(boundaries, boundaries[1:]):
        start, end = _format_ass_interval(interval_start, interval_end)
        start_cs = _parse_ass_centiseconds(start)
        end_cs = _parse_ass_centiseconds(end)
        assert end_cs > start_cs
        assert start_cs >= previous_end - 1  # shared boundary may round to same centisecond
        # rounding error is at most half a centisecond
        assert abs(start_cs - interval_start * 100) <= 0.5 + 1e-9
        previous_end = end_cs


def _segments():
    return [
        TranscriptSegment(
            words=[
                WordTiming("just", 0.0, 0.5, source="forced_aligner", aligned=True),
                WordTiming("needs", 0.5, 1.0, source="forced_aligner", aligned=True),
                WordTiming("to", 1.0, 1.5, source="forced_aligner", aligned=True),
                WordTiming("look", 1.5, 2.0, source="forced_aligner", aligned=True),
                WordTiming("right", 2.0, 2.5, source="forced_aligner", aligned=True),
            ],
            text="just needs to look right",
            start=0.0,
            end=2.5,
        )
    ]


def test_renderers_create_wrapped_srt_and_grapheme_highlight_ass_files(tmp_path):
    srt_path = tmp_path / "subtitles.srt"
    ass_path = tmp_path / "karaoke.ass"
    style = get_style()
    style.max_words_per_line = 2
    expected_chunks = _build_render_chunks(_segments(), style, include_next_line_preview=False)

    SrtRenderer().render(_segments(), str(srt_path), style=style)
    AssKaraokeRenderer().render(_segments(), str(ass_path), style=style)

    srt_text = srt_path.read_text(encoding="utf-8")
    ass_text = ass_path.read_text(encoding="utf-8-sig")

    assert "00:00:00,000 --> 00:00:02,500" not in srt_text
    assert len(expected_chunks) >= 2
    for chunk in expected_chunks:
        assert f"{_format_srt_time(chunk.start)} --> {_format_srt_time(chunk.end)}" in srt_text
        for line in chunk.lines:
            assert " ".join(word.word for word in line) in srt_text
    assert "\\pos(" in ass_text
    assert "Dialogue: 0" in ass_text
    assert "Dialogue: 1" in ass_text
    assert "\\clip(" in ass_text
    assert "\\fad(" in ass_text
    assert "\\kf" not in ass_text
    assert "\\blur" in ass_text


def test_default_style_is_bigger_and_uses_grapheme_highlight_mode():
    style = get_style()

    assert style.font_size >= 90
    assert style.max_words_per_line <= 4
    assert style.max_chars_per_line <= 18
    assert style.encoding == -1
    assert style.blur > 0
    assert style.effect_mode == "grapheme_highlight"


def test_ass_renderer_stacks_overlapping_segments_on_separate_rows(tmp_path):
    ass_path = tmp_path / "overlap.ass"
    style = get_style()
    segments = [
        TranscriptSegment(
            words=[WordTiming("A", 0.0, 1.2, source="forced_aligner", aligned=True)],
            text="A",
            start=0.0,
            end=2.0,
        ),
        TranscriptSegment(
            words=[WordTiming("B", 0.0, 1.1, source="forced_aligner", aligned=True)],
            text="B",
            start=0.0,
            end=2.1,
        ),
    ]

    AssKaraokeRenderer().render(segments, str(ass_path), style=style)

    ass_text = ass_path.read_text(encoding="utf-8-sig")
    first_line = next(line for line in ass_text.splitlines() if line.startswith("Dialogue: 0") and "A" in line)
    second_line = next(line for line in ass_text.splitlines() if line.startswith("Dialogue: 0") and "B" in line)
    first_y = int(re.search(r"\\pos\(\d+,(\d+)\)", first_line).group(1))
    second_y = int(re.search(r"\\pos\(\d+,(\d+)\)", second_line).group(1))

    assert second_y != first_y


def test_subtitle_guardian_allows_staggered_segments_once_boundaries_are_clipped():
    style = get_style()
    guardian = AssSubtitleGuardian()
    segments = [
        TranscriptSegment(words=[WordTiming("a", 0.0, 1.0)], text="a", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("b", 0.1, 1.1)], text="b", start=0.1, end=2.0),
        TranscriptSegment(words=[WordTiming("c", 0.2, 1.2)], text="c", start=0.2, end=2.0),
        TranscriptSegment(words=[WordTiming("d", 0.3, 1.3)], text="d", start=0.3, end=2.0),
    ]

    warnings = guardian.validate(segments, style)

    assert isinstance(warnings, list)


def test_subtitle_guardian_rejects_true_same_start_four_way_collision():
    style = get_style()
    guardian = AssSubtitleGuardian()
    segments = [
        TranscriptSegment(words=[WordTiming("a", 0.0, 1.0)], text="a", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("b", 0.0, 1.1)], text="b", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("c", 0.0, 1.2)], text="c", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("d", 0.0, 1.3)], text="d", start=0.0, end=2.0),
    ]

    try:
        guardian.validate(segments, style)
        assert False, "Expected SubtitleGenerationError"
    except SubtitleGenerationError:
        pass


def test_subtitle_guardian_ignores_zero_duration_tail_duplicates():
    style = get_style()
    guardian = AssSubtitleGuardian()
    segments = [
        TranscriptSegment(words=[WordTiming("one", 0.0, 0.8)], text="one", start=0.0, end=0.8),
        TranscriptSegment(words=[WordTiming("two", 0.8, 1.6)], text="two", start=0.8, end=1.6),
        TranscriptSegment(words=[WordTiming("two", 1.6, 1.61)], text="two", start=1.6, end=1.6),
        TranscriptSegment(words=[WordTiming("three", 1.6, 1.61)], text="three", start=1.6, end=1.6),
        TranscriptSegment(words=[WordTiming("four", 1.6, 1.61)], text="four", start=1.6, end=1.6),
    ]

    warnings = guardian.validate(segments, style)

    assert isinstance(warnings, list)


def test_split_segment_words_breaks_on_natural_pause_even_if_text_fits():
    style = get_style()
    style.max_words_per_line = 5
    style.max_chars_per_line = 40
    segment = TranscriptSegment(
        words=[
            WordTiming("first", 0.0, 0.30),
            WordTiming("phrase", 0.34, 0.62),
            WordTiming("then", 0.95, 1.15),
            WordTiming("continue", 1.18, 1.52),
        ],
        text="first phrase then continue",
        start=0.0,
        end=1.52,
    )

    lines = _split_segment_words(segment, style)

    assert [[word.word for word in line] for line in lines] == [
        ["first", "phrase"],
        ["then", "continue"],
    ]


def test_split_segment_words_prefers_larger_timing_gap_when_wrapping():
    style = get_style()
    style.max_words_per_line = 3
    style.max_chars_per_line = 40
    segment = TranscriptSegment(
        words=[
            WordTiming("one", 0.00, 0.25),
            WordTiming("two", 0.29, 0.55),
            WordTiming("three", 0.95, 1.20),
            WordTiming("four", 1.24, 1.48),
        ],
        text="one two three four",
        start=0.0,
        end=1.48,
    )

    lines = _split_segment_words(segment, style)

    assert [[word.word for word in line] for line in lines] == [
        ["one", "two"],
        ["three", "four"],
    ]


def test_split_segment_words_breaks_on_a_single_large_gap():
    style = get_style()
    style.max_words_per_line = 5
    style.max_chars_per_line = 40
    segment = TranscriptSegment(
        words=[
            WordTiming("one", 0.0, 0.30),
            WordTiming("two", 4.20, 4.45),
            WordTiming("three", 4.45, 4.75),
        ],
        text="one two three",
        start=0.0,
        end=4.75,
    )

    lines = _split_segment_words(segment, style)

    assert [[word.word for word in line] for line in lines] == [
        ["one"],
        ["two", "three"],
    ]


def test_render_chunks_do_not_stretch_lines_across_long_pause():
    style = get_style()
    segment = TranscriptSegment(
        words=[
            WordTiming("first", 0.0, 0.40),
            WordTiming("line", 0.42, 0.80),
            WordTiming("second", 3.20, 3.55),
            WordTiming("line", 3.58, 3.90),
        ],
        text="first line second line",
        start=0.0,
        end=3.90,
    )

    chunks = _build_render_chunks([segment], style)

    assert len(chunks) == 2
    assert chunks[0].end < 1.1
    assert chunks[1].start > 2.8


def test_render_chunks_keep_wrapped_lines_on_their_own_timing_windows():
    style = get_style()
    style.max_words_per_line = 2
    style.max_chars_per_line = 40
    segment = TranscriptSegment(
        words=[
            WordTiming("first", 0.0, 0.30),
            WordTiming("second", 0.31, 0.60),
            WordTiming("third", 0.61, 0.90),
            WordTiming("fourth", 0.91, 1.20),
        ],
        text="first second third fourth",
        start=0.0,
        end=1.20,
    )

    chunks = _build_render_chunks([segment], style)

    assert len(chunks) == 2
    assert len(chunks[0].lines) == 1
    assert len(chunks[1].lines) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == 0.60
    assert chunks[1].start == 0.61
    assert chunks[1].end == 1.20


def test_render_chunks_roll_preview_window_into_two_visible_lines():
    style = get_style()
    style.max_words_per_line = 2
    style.max_chars_per_line = 40
    segment = TranscriptSegment(
        words=[
            WordTiming("first", 0.0, 0.30),
            WordTiming("second", 0.31, 0.60),
            WordTiming("third", 0.61, 0.90),
            WordTiming("fourth", 0.91, 1.20),
        ],
        text="first second third fourth",
        start=0.0,
        end=1.20,
    )

    chunks = _build_render_chunks([segment], style, include_next_line_preview=True)

    assert len(chunks) == 2
    assert [[word.word for word in line] for line in chunks[0].lines] == [
        ["first", "second"],
        ["third", "fourth"],
    ]
    assert [[word.word for word in line] for line in chunks[1].lines] == [["third", "fourth"]]
    assert chunks[0].start == 0.0
    assert chunks[0].content_end == 0.60
    assert chunks[0].end == 0.61
    assert chunks[1].start == 0.61


def test_render_chunks_can_preview_across_segment_boundaries():
    style = get_style()
    segments = [
        TranscriptSegment(
            words=[WordTiming("first", 0.0, 0.45), WordTiming("line", 0.46, 0.90)],
            text="first line",
            start=0.0,
            end=0.90,
        ),
        TranscriptSegment(
            words=[WordTiming("second", 1.10, 1.45), WordTiming("line", 1.46, 1.80)],
            text="second line",
            start=1.10,
            end=1.80,
        ),
    ]

    chunks = _build_render_chunks(segments, style, include_next_line_preview=True)

    assert len(chunks) == 2
    assert [[word.word for word in line] for line in chunks[0].lines] == [
        ["first", "line"],
        ["second", "line"],
    ]
    assert chunks[0].content_end == 0.90
    assert chunks[0].end == 1.10
    assert [[word.word for word in line] for line in chunks[1].lines] == [["second", "line"]]


def test_render_chunks_do_not_preview_far_future_line_after_long_gap():
    style = get_style()
    segments = [
        TranscriptSegment(
            words=[WordTiming("first", 0.0, 0.40)],
            text="first",
            start=0.0,
            end=0.40,
        ),
        TranscriptSegment(
            words=[WordTiming("second", 6.0, 6.40)],
            text="second",
            start=6.0,
            end=6.40,
        ),
    ]

    chunks = _build_render_chunks(segments, style, include_next_line_preview=True)

    assert len(chunks) == 2
    assert [[word.word for word in line] for line in chunks[0].lines] == [["first"]]
    assert chunks[0].end == 0.40


def test_render_chunks_preview_next_visible_line_without_skipping_to_matching_lane():
    style = get_style()
    segments = [
        TranscriptSegment(
            words=[WordTiming("first", 0.0, 0.40)],
            text="first",
            start=0.0,
            end=0.40,
        ),
        TranscriptSegment(
            words=[WordTiming("second", 1.0, 1.40)],
            text="second",
            start=1.0,
            end=1.40,
        ),
        TranscriptSegment(
            words=[WordTiming("third", 8.0, 8.40)],
            text="third",
            start=8.0,
            end=8.40,
        ),
    ]
    singer_analysis = SingerAnalysisResult(
        detected_singer_count=2,
        profiles=[
            SingerProfile("lane_a", "A", lane_index=0),
            SingerProfile("lane_b", "B", lane_index=1),
        ],
        assignments=[
            SingerSegmentAssignment(segment_index=0, singer_id="lane_a"),
            SingerSegmentAssignment(segment_index=1, singer_id="lane_b"),
            SingerSegmentAssignment(segment_index=2, singer_id="lane_a"),
        ],
    )

    chunks = _build_render_chunks(
        segments,
        style,
        singer_analysis=singer_analysis,
        include_next_line_preview=True,
    )

    assert len(chunks) == 3
    assert [[word.word for word in line] for line in chunks[0].lines] == [["first"], ["second"]]
    assert chunks[0].end == 1.0


def test_render_chunks_start_exactly_on_first_word_without_lead_in():
    style = get_style()
    segment = TranscriptSegment(
        words=[
            WordTiming("first", 5.20, 5.60),
            WordTiming("second", 5.62, 6.00),
        ],
        text="first second",
        start=5.00,
        end=6.00,
    )

    chunks = _build_render_chunks([segment], style)

    assert len(chunks) == 1
    assert chunks[0].start == 5.20
    assert chunks[0].content_start == 5.20


def test_render_chunks_clip_previous_line_at_next_segment_start():
    style = get_style()
    segments = [
        TranscriptSegment(
            words=[WordTiming("first", 0.0, 1.2)],
            text="first",
            start=0.0,
            end=1.4,
        ),
        TranscriptSegment(
            words=[WordTiming("second", 1.0, 1.8)],
            text="second",
            start=1.0,
            end=1.8,
        ),
    ]

    chunks = _build_render_chunks(segments, style)

    assert len(chunks) == 2
    assert chunks[0].end == 1.0
    assert chunks[1].start == 1.0


def test_split_segment_words_avoids_trailing_single_word_line_when_balance_is_possible():
    style = get_style()
    style.max_words_per_line = 4
    style.max_chars_per_line = 40
    segment = TranscriptSegment(
        words=[
            WordTiming("one", 0.0, 0.20),
            WordTiming("two", 0.21, 0.40),
            WordTiming("three", 0.41, 0.60),
            WordTiming("four", 0.61, 0.80),
            WordTiming("five", 0.81, 1.00),
        ],
        text="one two three four five",
        start=0.0,
        end=1.00,
    )

    lines = _split_segment_words(segment, style)

    assert len(lines) == 2
    assert all(len(line) >= 2 for line in lines)


def test_ass_renderer_includes_chord_overlay_and_next_preview(tmp_path):
    ass_path = tmp_path / "karaoke_with_chords.ass"
    style = get_style()
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("alpha", 0.0, 0.45),
                WordTiming("beta", 0.46, 0.90),
                WordTiming("gamma", 0.91, 1.35),
            ],
            text="alpha beta gamma",
            start=0.0,
            end=1.35,
        )
    ]
    song_analysis = SongAnalysis(
        bpm=120.0,
        preview_window_seconds=0.5,
        original_key="Em",
        target_key="Am",
        chord_events=[
            ChordEvent("C", 0.0, 0.72),
            ChordEvent("G", 0.72, 1.35),
        ],
    )

    AssKaraokeRenderer().render(segments, str(ass_path), style=style, song_analysis=song_analysis)

    ass_text = ass_path.read_text(encoding="utf-8-sig")

    assert ",ChordNext,," in ass_text
    assert "Dialogue: 11" in ass_text
    assert "Dialogue: 4" in ass_text
    assert "\\fs" in ass_text
    assert "סולם מקור: Em" in ass_text
    assert "סולם קל: Am" in ass_text
    assert "C" in ass_text
    assert "G" in ass_text


def test_ass_renderer_adds_next_line_preview_for_wrapped_chunks(tmp_path):
    ass_path = tmp_path / "karaoke_preview.ass"
    style = get_style()
    style.max_words_per_line = 2
    style.max_chars_per_line = 40
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("first", 0.0, 0.30, source="forced_aligner", aligned=True),
                WordTiming("second", 0.31, 0.60, source="forced_aligner", aligned=True),
                WordTiming("third", 0.61, 0.90, source="forced_aligner", aligned=True),
                WordTiming("fourth", 0.91, 1.20, source="forced_aligner", aligned=True),
            ],
            text="first second third fourth",
            start=0.0,
            end=1.20,
        )
    ]

    AssKaraokeRenderer().render(segments, str(ass_path), style=style, include_hud=False, include_next_line_preview=True)

    ass_text = ass_path.read_text(encoding="utf-8-sig")
    preview_lines = [
        line
        for line in ass_text.splitlines()
        if line.startswith("Dialogue: 0") and "\\1a&H78&" in line and "third" in line and "fourth" in line
    ]
    active_line = next(
        line
        for line in ass_text.splitlines()
        if line.startswith("Dialogue: 0") and "\\1a&H78&" not in line and "first" in line and "second" in line
    )

    assert preview_lines
    active_y = int(re.search(r"\\pos\(\d+,(\d+)\)", active_line).group(1))
    preview_y = int(re.search(r"\\pos\(\d+,(\d+)\)", preview_lines[0]).group(1))

    assert active_y < preview_y


def test_ass_renderer_uses_preview_line_singer_color_for_next_line_preview(tmp_path):
    ass_path = tmp_path / "karaoke_preview_color.ass"
    style = get_style()
    style.max_words_per_line = 2
    style.max_chars_per_line = 40
    segments = [
        TranscriptSegment(
            words=[WordTiming("first", 0.0, 0.30), WordTiming("line", 0.31, 0.60)],
            text="first line",
            start=0.0,
            end=0.60,
        ),
        TranscriptSegment(
            words=[WordTiming("second", 0.90, 1.20), WordTiming("line", 1.21, 1.50)],
            text="second line",
            start=0.90,
            end=1.50,
        ),
    ]
    singer_analysis = SingerAnalysisResult(
        detected_singer_count=2,
        profiles=[
            SingerProfile(
                "lane_a",
                "A",
                primary_color="&H000000FF",
                secondary_color="&H000000FF",
                outline_color="&H00000000",
                shadow_color="&H70000000",
                lane_index=0,
            ),
            SingerProfile(
                "lane_b",
                "B",
                primary_color="&H0000FF00",
                secondary_color="&H0000FF00",
                outline_color="&H00000000",
                shadow_color="&H70000000",
                lane_index=1,
            ),
        ],
        assignments=[
            SingerSegmentAssignment(segment_index=0, singer_id="lane_a"),
            SingerSegmentAssignment(segment_index=1, singer_id="lane_b"),
        ],
    )

    AssKaraokeRenderer().render(
        segments,
        str(ass_path),
        style=style,
        singer_analysis=singer_analysis,
        include_hud=False,
        include_next_line_preview=True,
    )

    ass_text = ass_path.read_text(encoding="utf-8-sig")
    preview_line = next(
        line
        for line in ass_text.splitlines()
        if line.startswith("Dialogue: 0") and "\\1a&H78&" in line and "second" in line and "line" in line
    )

    assert "\\1c&H0000FF00" in preview_line


def test_ass_renderer_can_render_subtitle_only_output(tmp_path):
    ass_path = tmp_path / "karaoke_clean.ass"
    style = get_style()
    style.max_words_per_line = 2
    style.max_chars_per_line = 40
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("alpha", 0.0, 0.45),
                WordTiming("beta", 0.46, 0.90),
                WordTiming("gamma", 0.91, 1.35),
                WordTiming("delta", 1.36, 1.80),
            ],
            text="alpha beta gamma delta",
            start=0.0,
            end=1.80,
        )
    ]
    song_analysis = SongAnalysis(
        bpm=120.0,
        preview_window_seconds=0.5,
        original_key="Em",
        target_key="Am",
        chord_events=[
            ChordEvent("C", 0.0, 0.90),
            ChordEvent("G", 0.90, 1.80),
        ],
    )

    AssKaraokeRenderer().render(
        segments,
        str(ass_path),
        style=style,
        song_analysis=song_analysis,
        include_chord_overlays=False,
        include_hud=False,
        include_next_line_preview=False,
    )

    ass_text = ass_path.read_text(encoding="utf-8-sig")

    assert ",ChordNow,," not in ass_text
    assert ",ChordNext,," not in ass_text
    assert ",HUD,," not in ass_text
    assert "סולם מקור: Em" not in ass_text
    assert "סולם קל: Am" not in ass_text
    assert "\\1a&H78&" not in ass_text
