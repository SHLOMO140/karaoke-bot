import re

from karaoke import job_manager
from karaoke.models import (
    SingerAnalysisResult,
    SingerProfile,
    SingerSegmentAssignment,
    TranscriptSegment,
    WordTiming,
)
from karaoke.styles import get_style
from karaoke.singer_analysis import StructureDuetAnalyzer
from karaoke.subtitle_generator import AssKaraokeRenderer
from karaoke.subtitle_guardian import AssSubtitleGuardian


def test_singer_analysis_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="demo", input_type="audio_file")

    analysis = SingerAnalysisResult(
        detected_singer_count=2,
        provider="test_singers",
        profiles=[
            SingerProfile(singer_id="singer_1", label="Singer 1", lane_index=0),
            SingerProfile(singer_id="singer_2", label="Singer 2", lane_index=1),
        ],
        assignments=[
            SingerSegmentAssignment(segment_index=0, singer_id="singer_1", label="Singer 1", confidence=0.81),
            SingerSegmentAssignment(segment_index=1, singer_id="singer_2", label="Singer 2", confidence=0.76),
        ],
        low_confidence_segments=0,
        analysis_window_seconds=12.4,
    )

    job_manager.save_singer_analysis(job, analysis)
    loaded = job_manager.load_singer_analysis(job)

    assert loaded.detected_singer_count == 2
    assert [profile.singer_id for profile in loaded.profiles] == ["singer_1", "singer_2"]
    assert [assignment.singer_id for assignment in loaded.assignments] == ["singer_1", "singer_2"]
    assert loaded.analysis_window_seconds == 12.4


def test_ass_renderer_uses_distinct_lanes_and_colors_for_multiple_singers(tmp_path):
    ass_path = tmp_path / "karaoke_duet.ass"
    style = get_style()
    segments = [
        TranscriptSegment(
            words=[WordTiming("alpha", 0.0, 0.8, source="forced_aligner", aligned=True)],
            text="alpha",
            start=0.0,
            end=1.2,
        ),
        TranscriptSegment(
            words=[WordTiming("beta", 0.0, 0.8, source="forced_aligner", aligned=True)],
            text="beta",
            start=0.0,
            end=1.2,
        ),
    ]
    singer_analysis = SingerAnalysisResult(
        detected_singer_count=2,
        provider="test",
        profiles=[
            SingerProfile(
                singer_id="singer_1",
                label="Singer 1",
                lane_index=0,
                primary_color="&H00FFF8F1",
                secondary_color="&H00FF8E3A",
                outline_color="&H00A04917",
                shadow_color="&H700E0E10",
            ),
            SingerProfile(
                singer_id="singer_2",
                label="Singer 2",
                lane_index=1,
                primary_color="&H00F6FFF3",
                secondary_color="&H0092D55A",
                outline_color="&H00457016",
                shadow_color="&H700A120A",
            ),
        ],
        assignments=[
            SingerSegmentAssignment(segment_index=0, singer_id="singer_1", label="Singer 1", confidence=0.91),
            SingerSegmentAssignment(segment_index=1, singer_id="singer_2", label="Singer 2", confidence=0.87),
        ],
    )

    AssKaraokeRenderer().render(
        segments,
        str(ass_path),
        style=style,
        singer_analysis=singer_analysis,
        include_hud=False,
        include_next_line_preview=False,
    )

    ass_text = ass_path.read_text(encoding="utf-8-sig")
    alpha_line = next(line for line in ass_text.splitlines() if line.startswith("Dialogue: 0") and "alpha" in line)
    beta_line = next(line for line in ass_text.splitlines() if line.startswith("Dialogue: 0") and "beta" in line)
    alpha_y = int(re.search(r"\\pos\(\d+,(\d+)\)", alpha_line).group(1))
    beta_y = int(re.search(r"\\pos\(\d+,(\d+)\)", beta_line).group(1))

    assert alpha_y < beta_y
    assert "\\1c&H00FFF8F1" in alpha_line
    assert "\\1c&H00F6FFF3" in beta_line


def test_subtitle_guardian_allows_parallel_singers_on_separate_lanes():
    style = get_style()
    guardian = AssSubtitleGuardian()
    segments = [
        TranscriptSegment(words=[WordTiming("a", 0.0, 1.0)], text="a", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("b", 0.0, 1.0)], text="b", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("c", 0.0, 1.0)], text="c", start=0.0, end=2.0),
        TranscriptSegment(words=[WordTiming("d", 0.0, 1.0)], text="d", start=0.0, end=2.0),
    ]
    singer_analysis = SingerAnalysisResult(
        detected_singer_count=2,
        provider="test",
        profiles=[
            SingerProfile(singer_id="singer_1", label="Singer 1", lane_index=0),
            SingerProfile(singer_id="singer_2", label="Singer 2", lane_index=1),
        ],
        assignments=[
            SingerSegmentAssignment(segment_index=0, singer_id="singer_1"),
            SingerSegmentAssignment(segment_index=1, singer_id="singer_2"),
            SingerSegmentAssignment(segment_index=2, singer_id="singer_1"),
            SingerSegmentAssignment(segment_index=3, singer_id="singer_2"),
        ],
    )

    warnings = guardian.validate(segments, style, singer_analysis=singer_analysis)

    assert isinstance(warnings, list)


def test_structure_duet_analyzer_splits_sections_into_halves_and_cycles_colors():
    analyzer = StructureDuetAnalyzer()
    segments = [
        TranscriptSegment(words=[WordTiming("verse", 0.0, 1.0)], text="verse one a", start=0.0, end=1.0),
        TranscriptSegment(words=[WordTiming("verse", 1.0, 2.0)], text="verse one b", start=1.0, end=2.0),
        TranscriptSegment(words=[WordTiming("verse", 2.0, 3.0)], text="verse one c", start=2.0, end=3.0),
        TranscriptSegment(words=[WordTiming("verse", 3.0, 4.0)], text="verse one d", start=3.0, end=4.0),
        TranscriptSegment(words=[WordTiming("chorus", 4.0, 5.0)], text="chorus bright light", start=4.0, end=5.0),
        TranscriptSegment(words=[WordTiming("chorus", 5.0, 6.0)], text="chorus hold on", start=5.0, end=6.0),
        TranscriptSegment(words=[WordTiming("chorus", 6.0, 7.0)], text="chorus bright sky", start=6.0, end=7.0),
        TranscriptSegment(words=[WordTiming("chorus", 7.0, 8.0)], text="chorus hold tight", start=7.0, end=8.0),
        TranscriptSegment(words=[WordTiming("verse", 8.0, 9.0)], text="verse two a", start=8.0, end=9.0),
        TranscriptSegment(words=[WordTiming("verse", 9.0, 10.0)], text="verse two b", start=9.0, end=10.0),
        TranscriptSegment(words=[WordTiming("chorus", 10.0, 11.0)], text="chorus bright light", start=10.0, end=11.0),
        TranscriptSegment(words=[WordTiming("chorus", 11.0, 12.0)], text="chorus hold on", start=11.0, end=12.0),
        TranscriptSegment(words=[WordTiming("chorus", 12.0, 13.0)], text="chorus bright sky", start=12.0, end=13.0),
        TranscriptSegment(words=[WordTiming("chorus", 13.0, 14.0)], text="chorus hold tight", start=13.0, end=14.0),
    ]

    analysis = analyzer.analyze("", segments, title="demo duet")
    by_segment = {assignment.segment_index: assignment.singer_id for assignment in analysis.assignments}

    assert analysis.provider == "structure_duet_sections_v2"
    assert analysis.detected_singer_count == 2
    assert by_segment[0] == "duet_lane_1_color_1"
    assert by_segment[1] == "duet_lane_1_color_1"
    assert by_segment[2] == "duet_lane_2_color_2"
    assert by_segment[3] == "duet_lane_2_color_2"
    assert by_segment[4] == "duet_lane_1_color_3"
    assert by_segment[5] == "duet_lane_1_color_3"
    assert by_segment[6] == "duet_lane_2_color_1"
    assert by_segment[7] == "duet_lane_2_color_1"
    assert by_segment[8] == "duet_lane_1_color_2"
    assert by_segment[9] == "duet_lane_2_color_3"
    assert by_segment[10] == "duet_lane_1_color_1"
    assert by_segment[11] == "duet_lane_1_color_1"
