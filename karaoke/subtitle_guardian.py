"""Post-layout safety checks for ASS subtitle stacking."""

from __future__ import annotations

from .exceptions import SubtitleGenerationError
from .models import KaraokeStyle, SingerAnalysisResult, TranscriptSegment
from .subtitle_generator import (
    _assign_ass_stack_offsets,
    _build_render_chunks,
    _enforce_segment_boundaries,
    _segment_line_count,
    _singer_style_map,
)


class AssSubtitleGuardian:
    """Secondary safety pass that validates stacked subtitle layout."""

    name = "ass_subtitle_guardian"

    def validate(
        self,
        segments: list[TranscriptSegment],
        style: KaraokeStyle,
        singer_analysis: SingerAnalysisResult | None = None,
    ) -> list[str]:
        render_chunks = _build_render_chunks(
            _enforce_segment_boundaries(segments),
            style,
            singer_analysis=singer_analysis,
            include_next_line_preview=True,
        )
        singer_styles = _singer_style_map(style, singer_analysis)
        stack_offsets = _assign_ass_stack_offsets(
            render_chunks,
            style,
            singer_styles=singer_styles if singer_styles else None,
        )
        warnings: list[str] = []

        for index, chunk in enumerate(render_chunks):
            if not chunk.words:
                continue

            line_count = _segment_line_count(chunk, style)
            stack_start = stack_offsets.get(index, 0)
            stack_end = stack_start + line_count
            if stack_end > 2:
                raise SubtitleGenerationError(
                    f"More than two subtitle rows are visible at chunk {index}",
                    "יותר משתי שורות כתוביות אמורות להופיע יחד על המסך. צריך לפצל או לקצר טיימינגים.",
                )

            for other_index in range(index + 1, len(render_chunks)):
                other = render_chunks[other_index]
                if not other.words:
                    continue
                if chunk.end <= other.start + 1e-6 or other.end <= chunk.start + 1e-6:
                    continue

                other_line_count = _segment_line_count(other, style)
                other_stack_start = stack_offsets.get(other_index, 0)
                other_stack_end = other_stack_start + other_line_count
                if singer_styles and getattr(chunk, "singer_id", "") != getattr(other, "singer_id", ""):
                    continue
                stack_ranges_overlap = not (
                    stack_end <= other_stack_start or other_stack_end <= stack_start
                )
                if stack_ranges_overlap:
                    raise SubtitleGenerationError(
                        f"Subtitle collision between chunks {index} and {other_index}",
                        "שתי שורות כתוביות עדיין מתנגשות על המסך אחרי תיקון הלייאאוט.",
                    )

        if any(stack_offsets.get(index, 0) == 1 for index in range(len(render_chunks))):
            warnings.append("מוצגות שתי שורות במקביל, וזה הגבול העליון המותר בלייאאוט.")

        return warnings
