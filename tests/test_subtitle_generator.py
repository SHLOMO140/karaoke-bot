import re

from karaoke.models import ChordEvent, SongAnalysis, TranscriptSegment, WordTiming
from karaoke.exceptions import SubtitleGenerationError
from karaoke.styles import get_style
from karaoke.subtitle_generator import AssKaraokeRenderer, SrtRenderer, _build_render_chunks, _format_srt_time, _split_segment_words
from karaoke.subtitle_guardian import AssSubtitleGuardian


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
    expected_chunks = _build_render_chunks(_segments(), style)

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
        chord_events=[
            ChordEvent("C", 0.0, 0.72),
            ChordEvent("G", 0.72, 1.35),
        ],
    )

    AssKaraokeRenderer().render(segments, str(ass_path), style=style, song_analysis=song_analysis)

    ass_text = ass_path.read_text(encoding="utf-8-sig")

    assert "-> G" in ass_text
    assert "Dialogue: 5" in ass_text
    assert "\\fs" in ass_text
    assert "C" in ass_text
    assert "G" in ass_text
