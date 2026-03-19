"""SRT and ASS subtitle renderers for aligned karaoke text."""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import median

from .exceptions import SubtitleGenerationError
from .harmony import build_word_id_chord_map
from .models import KaraokeStyle, SongAnalysis, SubWordTiming, TranscriptSegment, WordTiming
from .styles import get_style

logger = logging.getLogger(__name__)

try:
    from PIL import ImageFont
except Exception:
    ImageFont = None

PLAY_RES_X = 1920
PLAY_RES_Y = 1080
RTL_MARK = "\u200F"
RTL_EMBED = "\u202B"
POP_DIRECTIONAL = "\u202C"
WINDOWS_FONT_DIR = Path("C:/Windows/Fonts")
MICRO_OVERLAP_TOLERANCE_SEC = 0.06
SUBWORD_TIMING_TOLERANCE_SEC = 0.08


@dataclass(frozen=True)
class RenderChunk:
    lines: list[list[WordTiming]]
    words: list[WordTiming]
    start: float
    end: float
    source_segment_index: int = 0
    content_start: float = 0.0
    content_end: float = 0.0


@dataclass(frozen=True)
class ChunkLayout:
    text: str
    ass_text: str
    anchor_x: float
    y: float
    line_left: float
    line_right: float
    placed_words: list[tuple[WordTiming, float, float, float]]


def _format_srt_time(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _format_ass_time(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * 100))
    return _format_ass_centiseconds(int(total_centiseconds))


def _format_ass_centiseconds(total_centiseconds: int) -> str:
    total_centiseconds = max(0, total_centiseconds)
    hours, remainder = divmod(total_centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _format_ass_interval(start_seconds: float, end_seconds: float) -> tuple[str, str]:
    start_centiseconds = max(0, int(start_seconds * 100))
    end_centiseconds = max(start_centiseconds + 1, int(end_seconds * 100))
    return (
        _format_ass_centiseconds(start_centiseconds),
        _format_ass_centiseconds(end_centiseconds),
    )


def _line_length(words: list[WordTiming]) -> int:
    return sum(len(word.word) for word in words) + max(0, len(words) - 1)


def _line_within_limits(words: list[WordTiming], style: KaraokeStyle) -> bool:
    if not words:
        return False
    if len(words) == 1:
        return True
    return len(words) <= style.max_words_per_line and _line_length(words) <= style.max_chars_per_line


def _timing_gap(words: list[WordTiming], index: int) -> float:
    if index < 0 or index >= len(words) - 1:
        return 0.0
    return max(0.0, words[index + 1].start - words[index].end)


def _gap_baseline(words: list[WordTiming]) -> float:
    gaps = sorted(_timing_gap(words, index) for index in range(len(words) - 1) if _timing_gap(words, index) > 1e-3)
    if not gaps:
        return 0.0
    sample_size = max(1, round(len(gaps) * 0.6))
    return float(median(gaps[:sample_size]))


def _pause_threshold(words: list[WordTiming], style: KaraokeStyle) -> float:
    baseline = _gap_baseline(words)
    return max(style.pause_gap_min_seconds, baseline * style.pause_gap_multiplier)


def _split_words_at_indices(words: list[WordTiming], split_indices: list[int]) -> list[list[WordTiming]]:
    lines: list[list[WordTiming]] = []
    start = 0
    for split_index in split_indices:
        if split_index <= start or split_index >= len(words):
            continue
        lines.append(words[start:split_index])
        start = split_index
    if start < len(words):
        lines.append(words[start:])
    return [line for line in lines if line]


def _pause_split_indices(words: list[WordTiming], style: KaraokeStyle) -> list[int]:
    if len(words) < 2:
        return []
    threshold = _pause_threshold(words, style)
    hard_pause_threshold = max(style.pause_gap_min_seconds * 2.5, 0.35)
    return [
        index + 1
        for index in range(len(words) - 1)
        if _timing_gap(words, index) >= threshold or _timing_gap(words, index) >= hard_pause_threshold
    ]


def _line_partition_cost(words: list[WordTiming], style: KaraokeStyle) -> float:
    max_words = max(style.max_words_per_line, 1)
    max_chars = max(style.max_chars_per_line, 1)
    word_fill = min(len(words) / max_words, 1.0)
    char_fill = min(_line_length(words) / max_chars, 1.0)
    cost = ((1.0 - char_fill) ** 2) * 1.35 + ((1.0 - word_fill) ** 2) * 0.9
    if len(words) == 1:
        cost += 0.35
    return cost


def _break_transition_cost(
    words: list[WordTiming],
    break_index: int,
    style: KaraokeStyle,
    baseline_gap: float,
    threshold_gap: float,
) -> float:
    if break_index <= 0 or break_index >= len(words):
        return 0.0

    gap = _timing_gap(words, break_index - 1)
    if gap <= 1e-3:
        return 0.55

    bonus = min(gap / max(threshold_gap, 0.01), 2.5) * 0.34
    if gap >= threshold_gap:
        bonus += 0.28
    elif baseline_gap > 0 and gap >= baseline_gap * 1.2:
        bonus += 0.10

    return max(-0.25, 0.55 - bonus)


def _split_chunk_words(words: list[WordTiming], style: KaraokeStyle) -> list[list[WordTiming]]:
    if not words:
        return []
    if _line_within_limits(words, style):
        return [words]

    baseline_gap = _gap_baseline(words)
    threshold_gap = _pause_threshold(words, style)

    @lru_cache(maxsize=None)
    def solve(start: int) -> tuple[float, tuple[int, ...]] | None:
        if start >= len(words):
            return (0.0, ())

        best_cost: float | None = None
        best_breaks: tuple[int, ...] = ()

        for end in range(start + 1, len(words) + 1):
            line_words = words[start:end]
            if not _line_within_limits(line_words, style):
                break

            remainder = solve(end)
            if remainder is None:
                continue

            remainder_cost, remainder_breaks = remainder
            transition_cost = 0.0
            if end < len(words):
                transition_cost = _break_transition_cost(
                    words,
                    end,
                    style,
                    baseline_gap,
                    threshold_gap,
                )

            total_cost = _line_partition_cost(line_words, style) + transition_cost + remainder_cost
            if best_cost is None or total_cost < best_cost:
                best_cost = total_cost
                best_breaks = (end,) + remainder_breaks

        if best_cost is None:
            return None
        return best_cost, best_breaks

    solution = solve(0)
    if solution is None:
        return [words]
    return _split_words_at_indices(words, list(solution[1]))


def _split_segment_words(segment: TranscriptSegment, style: KaraokeStyle) -> list[list[WordTiming]]:
    if not segment.words:
        return []

    lines: list[list[WordTiming]] = []
    for chunk in _split_words_at_indices(segment.words, _pause_split_indices(segment.words, style)):
        lines.extend(_rebalance_single_word_lines(_split_chunk_words(chunk, style), style))
    return lines or [segment.words]


def _rebalance_single_word_lines(lines: list[list[WordTiming]], style: KaraokeStyle) -> list[list[WordTiming]]:
    if len(lines) < 2:
        return lines

    adjusted = [list(line) for line in lines if line]
    soft_char_limit = style.max_chars_per_line + 2

    def fits_soft_limits(words: list[WordTiming]) -> bool:
        if not words:
            return False
        if len(words) == 1:
            return True
        return len(words) <= style.max_words_per_line and _line_length(words) <= soft_char_limit

    def pair_cost(current: list[WordTiming], next_line: list[WordTiming]) -> float:
        total = _line_partition_cost(current, style)
        if next_line:
            total += _line_partition_cost(next_line, style)
        return total

    improved = True
    while improved:
        improved = False
        for index in range(len(adjusted) - 1):
            current = adjusted[index]
            next_line = adjusted[index + 1]
            best_current = current
            best_next = next_line
            best_cost = pair_cost(current, next_line)

            if len(current) == 1 and len(next_line) > 1:
                candidate_current = current + [next_line[0]]
                candidate_next = next_line[1:]
                if fits_soft_limits(candidate_current) and fits_soft_limits(candidate_next):
                    candidate_cost = pair_cost(candidate_current, candidate_next)
                    if candidate_cost + 1e-6 < best_cost:
                        best_current = candidate_current
                        best_next = candidate_next
                        best_cost = candidate_cost

            if len(next_line) == 1 and len(current) > 1:
                candidate_current = current[:-1]
                candidate_next = [current[-1]] + next_line
                if fits_soft_limits(candidate_current) and fits_soft_limits(candidate_next):
                    candidate_cost = pair_cost(candidate_current, candidate_next)
                    if candidate_cost + 1e-6 < best_cost:
                        best_current = candidate_current
                        best_next = candidate_next
                        best_cost = candidate_cost

            merged_line = current + next_line
            if len(merged_line) <= style.max_words_per_line and _line_length(merged_line) <= soft_char_limit:
                candidate_cost = _line_partition_cost(merged_line, style)
                if candidate_cost + 1e-6 < best_cost:
                    best_current = merged_line
                    best_next = []
                    best_cost = candidate_cost

            if best_current != current or best_next != next_line:
                adjusted[index] = best_current
                if best_next:
                    adjusted[index + 1] = best_next
                else:
                    del adjusted[index + 1]
                improved = True
                break

    return adjusted


def _build_srt_text(lines: list[list[WordTiming]]) -> str:
    display_lines = [" ".join(word.word for word in line) for line in lines]
    return "\n".join(f"{RTL_MARK}{line}" for line in display_lines if line.strip())


def _build_ass_header(style: KaraokeStyle) -> str:
    return f"""[Script Info]
Title: Hebrew Karaoke
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{style.font_name},{style.font_size},{style.primary_color},{style.secondary_color},{style.outline_color},{style.shadow_color},{style.bold},0,0,0,100,100,0,0,{style.border_style},{style.outline_width},{style.shadow_depth},{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},{style.encoding}
Style: HUD,{style.font_name},38,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,3,3,2,7,20,20,18,-1
Style: ChordNow,{style.font_name},58,&H0000E8FF,&H0000E8FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,3,5,3,8,20,20,20,-1
Style: ChordNext,{style.font_name},40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,3,2,8,20,20,20,-1
Style: Countdown,{style.font_name},32,&H0080E0FF,&H0080E0FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,2,1,8,20,20,20,-1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", "(").replace("}", ")")


def _wrap_ass_rtl(text: str) -> str:
    return f"{RTL_MARK}{RTL_EMBED}{_escape_ass_text(text)}{POP_DIRECTIONAL}"


def _wrap_ass_rtl_parts(parts: list[str], separator: str = " ") -> str:
    escaped = separator.join(_escape_ass_text(part) for part in parts)
    return f"{RTL_MARK}{RTL_EMBED}{escaped}{POP_DIRECTIONAL}"


def _split_graphemes(text: str) -> list[str]:
    graphemes: list[str] = []
    for char in text:
        if char.isspace():
            continue
        if graphemes and unicodedata.combining(char):
            graphemes[-1] += char
        else:
            graphemes.append(char)
    return graphemes


def _fallback_subwords(word: WordTiming) -> list[SubWordTiming]:
    graphemes = _split_graphemes(word.word)
    if not graphemes:
        return []
    if len(graphemes) == 1:
        return [SubWordTiming(text=graphemes[0], start=word.start, end=word.end, confidence=word.confidence)]

    duration = max(word.end - word.start, 0.01)
    step = duration / len(graphemes)
    subwords = []
    for index, grapheme in enumerate(graphemes):
        start = word.start + index * step
        end = word.start + (index + 1) * step
        subwords.append(SubWordTiming(text=grapheme, start=start, end=end, confidence=word.confidence))
    subwords[-1].end = word.end
    return subwords


def _normalize_subwords_to_word(word: WordTiming, subwords: list[SubWordTiming]) -> list[SubWordTiming]:
    if not subwords:
        return []

    ordered = [subword for subword in subwords if subword.end > subword.start + 1e-6]
    if not ordered:
        return []

    source_start = ordered[0].start
    source_end = ordered[-1].end
    if source_end <= source_start + 1e-6:
        return []

    needs_normalization = (
        source_start < word.start - SUBWORD_TIMING_TOLERANCE_SEC
        or source_end > word.end + SUBWORD_TIMING_TOLERANCE_SEC
        or any(
            ordered[index + 1].start < ordered[index].start - 1e-6
            or ordered[index + 1].end < ordered[index + 1].start + 1e-6
            for index in range(len(ordered) - 1)
        )
    )
    if not needs_normalization:
        return ordered

    word_duration = max(word.end - word.start, 0.01)
    source_duration = max(source_end - source_start, 0.01)
    normalized: list[SubWordTiming] = []
    previous_end = word.start
    for index, subword in enumerate(ordered):
        relative_start = max(0.0, subword.start - source_start) / source_duration
        relative_end = max(0.0, subword.end - source_start) / source_duration
        start = word.start + relative_start * word_duration
        end = word.start + relative_end * word_duration
        start = max(word.start, min(word.end, start))
        end = max(start + 1e-3, min(word.end, end))
        start = max(previous_end, start)
        end = max(start + 1e-3, end)
        if index == len(ordered) - 1:
            end = word.end
        normalized.append(
            SubWordTiming(
                text=subword.text,
                start=start,
                end=min(word.end, end),
                confidence=subword.confidence,
            )
        )
        previous_end = normalized[-1].end

    if normalized:
        normalized[0].start = word.start
        normalized[-1].end = word.end
    return normalized


def _subwords_for_word(word: WordTiming) -> list[SubWordTiming]:
    normalized = _normalize_subwords_to_word(word, word.subwords)
    return normalized or _fallback_subwords(word)


def _grapheme_subwords_for_word(word: WordTiming) -> list[SubWordTiming]:
    graphemes: list[SubWordTiming] = []
    for subword in _subwords_for_word(word):
        parts = _split_graphemes(subword.text)
        if not parts:
            continue
        if len(parts) == 1:
            graphemes.append(
                SubWordTiming(
                    text=parts[0],
                    start=subword.start,
                    end=subword.end,
                    confidence=subword.confidence,
                )
            )
            continue

        duration = max(subword.end - subword.start, 0.01)
        step = duration / len(parts)
        for index, part in enumerate(parts):
            start = subword.start + index * step
            end = subword.start + (index + 1) * step
            graphemes.append(
                SubWordTiming(
                    text=part,
                    start=start,
                    end=end,
                    confidence=subword.confidence,
                )
            )

    return graphemes or _fallback_subwords(word)


def _resolve_font_path(font_name: str) -> str | None:
    candidates = [
        f"{font_name}.ttf",
        f"{font_name.lower()}.ttf",
        f"{font_name.replace(' ', '')}.ttf",
        f"{font_name.replace(' ', '').lower()}.ttf",
        "arial.ttf" if font_name.lower() == "arial" else "",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = WINDOWS_FONT_DIR / candidate
        if path.exists():
            return str(path)
    return None


@lru_cache(maxsize=16)
def _load_font(font_name: str, font_size: int):
    if ImageFont is None:
        return None
    font_path = _resolve_font_path(font_name)
    if not font_path:
        return None
    try:
        return ImageFont.truetype(font_path, font_size)
    except Exception:
        return None


def _estimate_word_width(word: str, style: KaraokeStyle) -> float:
    return _estimate_text_width(word, style)


def _estimate_text_width(text: str, style: KaraokeStyle) -> float:
    clean_word = text.strip()
    if not clean_word:
        return style.font_size * 0.46
    font = _load_font(style.font_name, style.font_size)
    if font is not None:
        try:
            return max(style.font_size * 0.52, float(font.getlength(clean_word)))
        except Exception:
            pass
    return max(style.font_size * 0.48, len(clean_word) * style.font_size * 0.48)


def _segment_line_count(segment: TranscriptSegment, style: KaraokeStyle) -> int:
    if isinstance(segment, RenderChunk):
        return max(1, len(segment.lines))
    return max(1, len(_split_segment_words(segment, style)))


def _segment_word_bounds(segment: TranscriptSegment) -> tuple[float, float]:
    if not segment.words:
        return segment.start, segment.end
    return min(segment.start, segment.words[0].start), max(segment.end, segment.words[-1].end)


def _close_intra_line_gaps(line_words: list[WordTiming]) -> list[WordTiming]:
    """Close gaps between consecutive words within a line.

    When a word ends before the next word starts (intra-line gap), extend the
    previous word's end to meet the next word's start.  This prevents "broken"
    lines where the karaoke highlight jumps or freezes between words.
    """
    if len(line_words) <= 1:
        return line_words
    fixed = list(line_words)
    for i in range(len(fixed) - 1):
        gap = fixed[i + 1].start - fixed[i].end
        if gap > 1e-3 and gap < 0.30:
            # Small gap — extend previous word to fill it
            fixed[i] = WordTiming(
                word=fixed[i].word,
                start=fixed[i].start,
                end=fixed[i + 1].start,
                confidence=fixed[i].confidence,
                source=fixed[i].source,
                aligned=fixed[i].aligned,
                subwords=fixed[i].subwords,
            )
    return fixed


def _build_render_chunks(segments: list[TranscriptSegment], style: KaraokeStyle) -> list[RenderChunk]:
    # Small lead-in before the first word starts being highlighted, so the
    # text is visible on screen for a moment before the karaoke effect begins.
    LEAD_IN_SEC = 0.12

    chunks: list[RenderChunk] = []
    for segment_index, segment in enumerate(segments):
        for line_index, line_words in enumerate(_split_segment_words(segment, style)):
            if not line_words:
                continue
            # Close intra-line gaps so words flow continuously without breaks
            line_words = _close_intra_line_gaps(line_words)
            content_start = line_words[0].start
            content_end = min(segment.end, line_words[-1].end)
            # Display the text slightly before the first word is highlighted.
            chunk_start = max(segment.start, content_start - LEAD_IN_SEC) if line_index == 0 else content_start
            chunk_end = max(content_start + 0.01, content_end)
            chunks.append(
                RenderChunk(
                    lines=[line_words],
                    words=line_words,
                    start=chunk_start,
                    end=chunk_end,
                    source_segment_index=segment_index,
                    content_start=content_start,
                    content_end=max(content_start + 0.01, content_end),
                )
            )

    chunks.sort(key=lambda chunk: (chunk.start, chunk.end, chunk.source_segment_index))

    # --- Dedup: remove chunks with identical text that overlap or nearly overlap ---
    if len(chunks) > 1:
        deduped: list[RenderChunk] = [chunks[0]]
        for chunk in chunks[1:]:
            prev = deduped[-1]
            chunk_text = " ".join(w.word for w in chunk.words)
            prev_text = " ".join(w.word for w in prev.words)
            # Same text and times overlap or are within 100ms — duplicate
            if chunk_text == prev_text and chunk.start < prev.end + 0.10:
                # Keep the one that starts earlier (already in deduped)
                logger.debug("Removed duplicate chunk: '%s' at %.3f", chunk_text[:30], chunk.start)
                continue
            deduped.append(chunk)
        chunks = deduped

    # Clip any overlapping chunks — even if the aligner's output is clean,
    # line splitting within a segment can produce chunks that slightly overlap.
    for index in range(len(chunks) - 1):
        current = chunks[index]
        next_chunk = chunks[index + 1]
        if current.end > next_chunk.start + 1e-6:
            trimmed_end = next_chunk.start
            chunks[index] = replace(current, end=max(current.start + 0.01, trimmed_end))

    # Filter out chunks that are too short to be visible or have broken timing
    return [chunk for chunk in chunks if chunk.end > chunk.start + 1e-3]


def _assign_ass_stack_offsets(segments: list[TranscriptSegment], style: KaraokeStyle) -> dict[int, int]:
    active: list[tuple[float, int, int]] = []
    offsets: dict[int, int] = {}

    indexed_segments = sorted(enumerate(segments), key=lambda item: (item[1].start, item[1].end, item[0]))
    for segment_index, segment in indexed_segments:
        if not segment.words or segment.end <= segment.start:
            offsets[segment_index] = 0
            continue

        active = [item for item in active if item[0] > segment.start + 1e-6]
        units = _segment_line_count(segment, style)
        slot = 0
        while True:
            collision = any(
                not (slot + units <= active_slot or slot >= active_slot + active_units)
                for _active_end, active_slot, active_units in active
            )
            if not collision:
                break
            slot += 1

        offsets[segment_index] = slot
        active.append((segment.end, slot, units))
        active.sort(key=lambda item: item[0])

    return offsets


def _style_event_tags(style: KaraokeStyle, fill_color: str | None = None) -> str:
    tags: list[str] = []
    if fill_color:
        tags.append(f"\\1c{fill_color}")
    if style.blur > 0:
        tags.append(f"\\blur{style.blur:g}")
    return "".join(tags)


def _grapheme_clip_padding(style: KaraokeStyle) -> tuple[int, int]:
    horizontal = max(2, int(round(style.outline_width + style.shadow_depth + style.blur * 2)))
    vertical = max(2, int(round(style.outline_width + style.shadow_depth + style.blur * 3)))
    return horizontal, vertical


def _group_display_lines(lines: list[list[WordTiming]], style: KaraokeStyle) -> list[list[list[WordTiming]]]:
    if not lines:
        return []

    groups: list[list[list[WordTiming]]] = []
    current_group: list[list[WordTiming]] = []
    for line_words in lines:
        if not line_words:
            continue
        if not current_group:
            current_group = [line_words]
            continue

        gap = max(0.0, line_words[0].start - current_group[-1][-1].end)
        if len(current_group) >= 2 or gap >= style.pause_gap_min_seconds:
            groups.append(current_group)
            current_group = [line_words]
            continue

        current_group.append(line_words)

    if current_group:
        groups.append(current_group)

    return groups


def _layout_chunk_words(
    words: list[WordTiming],
    style: KaraokeStyle,
    stack_offset_units: int = 0,
) -> list[tuple[WordTiming, float, float, float]]:
    return _layout_chunk_geometry(words, style, stack_offset_units=stack_offset_units).placed_words


def _layout_chunk_geometry(
    words: list[WordTiming],
    style: KaraokeStyle,
    stack_offset_units: int = 0,
) -> ChunkLayout:
    layouts = _layout_chunk_geometries([words], style, stack_offset_units=stack_offset_units)
    if layouts:
        return layouts[0]
    return ChunkLayout(
        text="",
        ass_text="",
        anchor_x=PLAY_RES_X / 2,
        y=PLAY_RES_Y - style.margin_v,
        line_left=PLAY_RES_X / 2,
        line_right=PLAY_RES_X / 2,
        placed_words=[],
    )


def _layout_chunk_geometries(
    lines: list[list[WordTiming]],
    style: KaraokeStyle,
    stack_offset_units: int = 0,
) -> list[ChunkLayout]:
    placed_words: list[tuple[WordTiming, float, float, float]] = []
    layouts: list[ChunkLayout] = []
    line_height = style.font_size * style.line_height_scale
    first_line_y = PLAY_RES_Y - style.margin_v - (stack_offset_units + len(lines) - 1) * line_height
    font = _load_font(style.font_name, style.font_size)
    if font is not None:
        try:
            space_width = max(style.font_size * 0.22, float(font.getlength(" ")) * style.word_spacing_scale)
        except Exception:
            space_width = style.font_size * 0.28
        try:
            base_space_width = max(style.font_size * 0.18, float(font.getlength(" ")))
        except Exception:
            base_space_width = style.font_size * 0.28
    else:
        space_width = style.font_size * 0.28
        base_space_width = style.font_size * 0.28

    gap_units = max(1, int(round(space_width / max(base_space_width, 1.0))))
    effective_space_width = base_space_width * gap_units
    gap_text = r"\h" * gap_units

    for line_index, line_words in enumerate(lines):
        word_widths = [_estimate_word_width(word.word, style) for word in line_words]
        line_width = sum(word_widths) + max(0, len(line_words) - 1) * effective_space_width
        line_left = max(style.margin_l, (PLAY_RES_X - line_width) / 2)
        cursor_right = line_left + line_width
        y_position = first_line_y + line_index * line_height
        placed_words = []

        for word, width in zip(line_words, word_widths):
            x_position = cursor_right - width / 2
            placed_words.append((word, x_position, y_position, width))
            cursor_right -= width + effective_space_width

        plain_text = " ".join(word.word for word in line_words)
        ass_text = _wrap_ass_rtl_parts([word.word for word in line_words], separator=gap_text)
        layouts.append(
            ChunkLayout(
                text=plain_text,
                ass_text=ass_text,
                anchor_x=line_left + line_width,
                y=y_position,
                line_left=line_left,
                line_right=line_left + line_width,
                placed_words=placed_words,
            )
        )

    return layouts


def _layout_segment_words(
    segment: TranscriptSegment,
    style: KaraokeStyle,
    stack_offset_units: int = 0,
) -> list[tuple[WordTiming, float, float, float]]:
    return _layout_chunk_words(segment.words, style, stack_offset_units=stack_offset_units)


def _base_chord_style_tags(style: KaraokeStyle) -> str:
    chord_size = max(26, int(round(style.font_size * 0.38)))
    chord_outline = max(2, style.outline_width - 3)
    chord_shadow = max(0, style.shadow_depth - 2)
    chord_blur = max(0.2, style.blur * 0.65)
    return (
        f"\\fs{chord_size}"
        f"\\bord{chord_outline}"
        f"\\shad{chord_shadow}"
        f"\\blur{chord_blur:g}"
        "\\b1"
        f"\\1c{style.secondary_color}"
        f"\\3c{style.outline_color}"
    )


def _chord_text_event(
    *,
    layer: int,
    start: float,
    end: float,
    text: str,
    x: float,
    y: float,
    style: KaraokeStyle,
    alpha: str = "00",
    alignment: int = 2,
) -> str | None:
    if end <= start + 1e-3 or not text.strip():
        return None
    ass_start, ass_end = _format_ass_interval(start, end)
    return (
        "Dialogue: {layer},{start},{end},Karaoke,,0,0,0,,{{\\an{alignment}\\q2\\pos({x},{y})\\1a&H{alpha}&{tags}}}{text}".format(
            layer=layer,
            start=ass_start,
            end=ass_end,
            alignment=alignment,
            x=int(round(x)),
            y=int(round(y)),
            alpha=alpha,
            tags=_base_chord_style_tags(style),
            text=_escape_ass_text(text),
        )
    )


def _word_chord_events(
    chunk_layout: ChunkLayout,
    chunk_start: float,
    chunk_end: float,
    word_chord_map: dict[int, list[object]],
    style: KaraokeStyle,
    preview_window: float,
) -> list[str]:
    dialogue_lines: list[str] = []
    y_offset = style.font_size * 0.72

    for word, x, y, _width in chunk_layout.placed_words:
        chord_events = word_chord_map.get(id(word), [])
        if not chord_events:
            continue

        label_text = " / ".join(dict.fromkeys(event.label for event in chord_events if getattr(event, "label", "")))
        if not label_text:
            continue

        base_event = _chord_text_event(
            layer=2,
            start=chunk_start,
            end=chunk_end,
            text=label_text,
            x=x,
            y=y - y_offset,
            style=style,
            alpha="66",
        )
        if base_event:
            dialogue_lines.append(base_event)

        for chord_event in chord_events:
            preview_start = max(chunk_start, chord_event.start - max(0.15, preview_window))
            preview_end = min(chunk_end, chord_event.start)
            if preview_end > preview_start + 1e-3:
                preview_event = _chord_text_event(
                    layer=3,
                    start=preview_start,
                    end=preview_end,
                    text=chord_event.label,
                    x=x,
                    y=y - y_offset,
                    style=style,
                    alpha="22",
                )
                if preview_event:
                    dialogue_lines.append(preview_event)

            active_event = _chord_text_event(
                layer=4,
                start=max(chunk_start, chord_event.start),
                end=min(chunk_end, chord_event.end),
                text=chord_event.label,
                x=x,
                y=y - y_offset,
                style=style,
                alpha="00",
            )
            if active_event:
                dialogue_lines.append(active_event)

    return dialogue_lines


def _build_hud_events(song_analysis: SongAnalysis | None, style: KaraokeStyle) -> list[str]:
    """Build the full HUD overlay: BPM/meter badge + current chord + metronome countdown.

    Layout:
    ┌─────────────────────────────────┐
    │  BPM: 120  |  4/4              │  ← HUD badge, always visible (top-left)
    │                                 │
    │         Am                      │  ← ChordNow, current chord (center-top)
    │      → Em                       │  ← ChordNext, upcoming chord (below)
    │     · · ● ·                     │  ← Metronome dots: beat position in measure
    └─────────────────────────────────┘

    The metronome runs through the FULL duration of each chord (not just a
    short preview window), counting beats grouped by time signature.  In the
    last measure before a chord change the dots switch to a countdown style
    so the viewer knows exactly when the next chord arrives.
    """
    if not song_analysis:
        return []

    dialogue_lines: list[str] = []
    bpm = song_analysis.bpm
    time_sig = max(song_analysis.time_signature, 2)
    beat_times = sorted(song_analysis.beat_times) if song_analysis.beat_times else []
    visible_events = [e for e in song_analysis.chord_events if e.label and e.label != "N"]

    # ── 1. BPM / Time Signature badge ── always visible in top-left corner ──
    if bpm > 0 or visible_events:
        first_time = 0.0
        last_time = 0.0
        if visible_events:
            first_time = max(0.0, visible_events[0].start - 2.0)
            last_time = visible_events[-1].end + 1.0
        elif beat_times:
            first_time = beat_times[0]
            last_time = beat_times[-1] + 2.0

        if last_time > first_time:
            bpm_text = f"BPM: {bpm:.0f}" if bpm > 0 else "BPM: --"
            meter_text = f"{time_sig}/4"
            hud_text = f"{bpm_text}  |  {meter_text}"
            ass_start, ass_end = _format_ass_interval(first_time, last_time)
            dialogue_lines.append(
                f"Dialogue: 10,{ass_start},{ass_end},HUD,,0,0,0,,"
                f"{{\\an7\\pos(48,38)\\bord3\\shad2\\blur0.8"
                f"\\1c&H00FFFFFF&\\3c&H00000000&\\4c&H96000000&"
                f"\\b1\\fs38}}{_escape_ass_text(hud_text)}"
            )

    if not visible_events:
        return dialogue_lines

    beat_length = 60.0 / max(bpm, 80.0) if bpm > 0 else 0.5
    center_x = PLAY_RES_X // 2
    chord_now_y = 100
    chord_next_y = 152
    metronome_y = 198

    # ── Helper: find all beats inside a time range from the real beat_times list ──
    def _beats_in_range(t_start: float, t_end: float) -> list[float]:
        """Return beat_times that fall in [t_start, t_end).  Fallback to synthetic."""
        real = [bt for bt in beat_times if t_start - 0.02 <= bt < t_end - 0.02]
        if real:
            return real
        # Synthetic fallback — generate beats from t_start at beat_length intervals
        synth: list[float] = []
        t = t_start
        while t < t_end - 0.02:
            synth.append(t)
            t += beat_length
        return synth or [t_start]

    # ── Iterate chords ──────────────────────────────────────────────────────
    for index, event in enumerate(visible_events):
        next_event = visible_events[index + 1] if index + 1 < len(visible_events) else None

        # ── 2. Current chord ── gapless, no fade-out (only fade-in) ─────────
        chord_display_end = next_event.start if next_event else event.end
        ass_start, ass_end = _format_ass_interval(event.start, chord_display_end)
        dialogue_lines.append(
            f"Dialogue: 11,{ass_start},{ass_end},ChordNow,,0,0,0,,"
            f"{{\\an8\\pos({center_x},{chord_now_y})\\bord5\\shad3\\blur1.2"
            f"\\1c&H0000E8FF&\\3c&H00000000&\\4c&H96000000&"
            f"\\b1\\fs58\\fad(120,0)}}{_escape_ass_text(event.label)}"
        )

        # ── 3. Full metronome beats throughout the chord ────────────────────
        chord_beats = _beats_in_range(event.start, chord_display_end)
        if not chord_beats:
            if next_event is None:
                continue
            # Jump to next-chord preview even without beats
            chord_beats = []

        # Group beats by time_sig to get measure position (1-based)
        for b_idx, bt in enumerate(chord_beats):
            measure_pos = (b_idx % time_sig) + 1  # 1, 2, 3, 4, 1, 2, ...
            b_end = chord_beats[b_idx + 1] if b_idx + 1 < len(chord_beats) else chord_display_end
            b_end = min(b_end, chord_display_end)

            if b_end <= bt + 0.02:
                continue

            # Build visual: dots showing position in measure
            # Current beat is ● (filled), others are · (hollow)
            dots = []
            for pos in range(1, time_sig + 1):
                if pos == measure_pos:
                    dots.append("\u25cf")  # ● filled circle
                else:
                    dots.append("\u00b7")  # · middle dot
            # RTL-safe: join with spaces
            dot_text = "  ".join(dots)

            # Is this one of the last `time_sig` beats before a chord change?
            beats_until_end = len(chord_beats) - b_idx
            is_final_measure = next_event is not None and beats_until_end <= time_sig

            # Colors: normal = subtle blue-white, final measure = warm orange
            if is_final_measure:
                dot_color = "\\1c&H0060CCFF&"  # warm orange for countdown emphasis
                dot_size = 36
            elif measure_pos == 1:
                dot_color = "\\1c&H00FFFFFF&"  # white on downbeat
                dot_size = 34
            else:
                dot_color = "\\1c&H00C0C0C0&"  # light gray for normal beats
                dot_size = 30

            ass_bs, ass_be = _format_ass_interval(bt, b_end)

            # Pulse animation: scale up then back down
            pulse_dur = min(int(beat_length * 400), 250)
            dialogue_lines.append(
                f"Dialogue: 12,{ass_bs},{ass_be},Countdown,,0,0,0,,"
                f"{{\\an8\\pos({center_x},{metronome_y})\\bord2\\shad1\\blur0.5"
                f"{dot_color}\\3c&H00000000&"
                f"\\b1\\fs{dot_size}"
                f"\\fscx110\\fscy110"
                f"\\t(0,{pulse_dur},\\fscx100\\fscy100)"
                f"\\fad(40,40)}}"
                f"{_escape_ass_text(dot_text)}"
            )

        # ── 4. Next chord preview ── appears during the final measure ───────
        if next_event is None:
            continue

        # Show next chord for the duration of the last measure
        preview_beats = min(time_sig, len(chord_beats)) if chord_beats else 2
        preview_lead = beat_length * preview_beats
        preview_lead = max(0.4, min(preview_lead, 3.0))
        preview_start = max(event.start + 0.1, next_event.start - preview_lead)
        preview_end = next_event.start

        if preview_end > preview_start + 0.05:
            ass_ps, ass_pe = _format_ass_interval(preview_start, preview_end)
            fade_in_ms = min(int(beat_length * 500), 400)
            dialogue_lines.append(
                f"Dialogue: 11,{ass_ps},{ass_pe},ChordNext,,0,0,0,,"
                f"{{\\an8\\pos({center_x},{chord_next_y})\\bord3\\shad2\\blur0.8"
                f"\\1c&H00FFFFFF&\\1a&H30&\\3c&H00000000&"
                f"\\b1\\fs40"
                f"\\fad({fade_in_ms},0)}}"
                f"{_escape_ass_text('\u2192 ' + next_event.label)}"
            )

    return dialogue_lines


def _base_word_event(
    event_start: float,
    event_end: float,
    word: WordTiming,
    x: float,
    y: float,
    style: KaraokeStyle,
) -> str:
    extra_tags = _style_event_tags(style, fill_color=style.primary_color)
    start, end = _format_ass_interval(event_start, event_end)
    return (
        "Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{{\\an5\\q2\\pos({x},{y})\\1a&H{alpha}&{extra}}}{text}".format(
            start=start,
            end=end,
            x=int(round(x)),
            y=int(round(y)),
            alpha=style.base_fill_alpha,
            extra=extra_tags,
            text=_wrap_ass_rtl(word.word),
        )
    )


def _base_line_event(
    event_start: float,
    event_end: float,
    layout: ChunkLayout,
    style: KaraokeStyle,
) -> str:
    extra_tags = _style_event_tags(style, fill_color=style.primary_color)
    start, end = _format_ass_interval(event_start, event_end)
    return (
        "Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{{\\an6\\q2\\pos({x},{y})\\1a&H{alpha}&{extra}}}{text}".format(
            start=start,
            end=end,
            x=int(round(layout.anchor_x)),
            y=int(round(layout.y)),
            alpha=style.base_fill_alpha,
            extra=extra_tags,
            text=layout.ass_text,
        )
    )


def _build_reveal_widths(word: WordTiming, word_width: float, style: KaraokeStyle) -> list[float]:
    subwords = _subwords_for_word(word)
    if not subwords:
        return []

    raw_widths = [_estimate_text_width(subword.text, style) for subword in subwords]
    total_width = sum(raw_widths)
    if total_width <= 0:
        step = word_width / len(subwords)
        return [step * (index + 1) for index in range(len(subwords))]

    scale = word_width / total_width
    widths = [max(1.0, width * scale) for width in raw_widths]
    adjustment = word_width - sum(widths)
    widths[-1] += adjustment

    cumulative = []
    running = 0.0
    for width in widths:
        running += width
        cumulative.append(min(word_width, max(1.0, running)))
    cumulative[-1] = word_width
    return cumulative


def _grapheme_positions(
    word: WordTiming,
    x: float,
    y: float,
    width: float,
    style: KaraokeStyle,
) -> list[tuple[SubWordTiming, float, float]]:
    graphemes = _grapheme_subwords_for_word(word)
    if not graphemes:
        return []

    positions: list[tuple[SubWordTiming, float, float]] = []
    prefix_widths = []
    prefix_text = ""
    for grapheme in graphemes:
        prefix_text += grapheme.text
        prefix_widths.append(_estimate_text_width(prefix_text, style))

    total_width = prefix_widths[-1] if prefix_widths else width
    if total_width <= 0:
        even_width = width / max(1, len(graphemes))
        prefix_widths = [even_width * (index + 1) for index in range(len(graphemes))]
        total_width = width

    scale = width / total_width if total_width > 0 else 1.0
    right_edge = x + width / 2
    previous_prefix_width = 0.0
    for grapheme, prefix_width in zip(graphemes, prefix_widths):
        scaled_prefix_width = prefix_width * scale
        scaled_previous_width = previous_prefix_width * scale
        grapheme_left = right_edge - scaled_prefix_width
        grapheme_right = right_edge - scaled_previous_width
        grapheme_width = max(1.0, grapheme_right - grapheme_left)
        positions.append((grapheme, grapheme_left + grapheme_width / 2, grapheme_width))
        previous_prefix_width = prefix_width
    return positions


def _erase_word_events(
    segment: TranscriptSegment,
    word: WordTiming,
    x: float,
    y: float,
    width: float,
    style: KaraokeStyle,
    segment_start: float,
    max_end: float | None = None,
) -> list[str]:
    dialogue_lines = []
    extra_tags = _style_event_tags(style, fill_color=style.primary_color)

    for grapheme, grapheme_x, _grapheme_width in _grapheme_positions(word, x, y, width, style):
        grapheme_end = min(grapheme.end, max_end) if max_end is not None else grapheme.end
        if grapheme_end <= segment_start + 1e-3:
            continue

        total_ms = max(1, int(round((grapheme_end - segment_start) * 1000)))
        fade_ms = min(style.grapheme_fade_ms, max(1, total_ms - 1))
        fade_tag = f"\\fad(0,{fade_ms})" if fade_ms > 0 else ""
        start, end = _format_ass_interval(segment_start, grapheme_end)
        dialogue_lines.append(
            "Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{{\\an5\\q2\\pos({x},{y})\\1a&H{alpha}&{fade}{extra}}}{text}".format(
                start=start,
                end=end,
                x=int(round(grapheme_x)),
                y=int(round(y)),
                alpha=style.base_fill_alpha,
                fade=fade_tag,
                extra=extra_tags,
                text=_wrap_ass_rtl(grapheme.text),
            )
        )

    return dialogue_lines


def _active_grapheme_events(
    word: WordTiming,
    x: float,
    y: float,
    width: float,
    style: KaraokeStyle,
    max_end: float | None = None,
) -> list[str]:
    dialogue_lines = []
    extra_tags = _style_event_tags(style, fill_color=style.secondary_color)
    clip_pad_x, clip_pad_y = _grapheme_clip_padding(style)
    left = int(round(x - width / 2))
    right = int(round(x + width / 2))
    top = int(round(y - style.font_size)) - clip_pad_y
    bottom = int(round(y + style.font_size * 0.6)) + clip_pad_y
    full_text = _wrap_ass_rtl(word.word)

    for grapheme, grapheme_x, grapheme_width in _grapheme_positions(word, x, y, width, style):
        grapheme_end = min(grapheme.end, max_end) if max_end is not None else grapheme.end
        if grapheme_end <= grapheme.start + 1e-3:
            continue

        total_ms = max(1, int(round((grapheme_end - grapheme.start) * 1000)))
        fade_in_ms = min(18, max(0, total_ms // 4))
        fade_out_ms = min(style.grapheme_fade_ms, max(1, total_ms - fade_in_ms - 1))
        fade_tag = f"\\fad({fade_in_ms},{fade_out_ms})" if fade_in_ms or fade_out_ms else ""
        start, end = _format_ass_interval(grapheme.start, grapheme_end)
        clip_left = max(left - clip_pad_x, int(round(grapheme_x - grapheme_width / 2)) - clip_pad_x)
        clip_right = min(right + clip_pad_x, int(round(grapheme_x + grapheme_width / 2)) + clip_pad_x)
        dialogue_lines.append(
            "Dialogue: 1,{start},{end},Karaoke,,0,0,0,,{{\\an5\\q2\\pos({x},{y})\\1a&H{alpha}&\\clip({clip_left},{top},{clip_right},{bottom}){fade}{extra}}}{text}".format(
                start=start,
                end=end,
                x=int(round(x)),
                y=int(round(y)),
                alpha=style.active_fill_alpha,
                clip_left=clip_left,
                top=top,
                clip_right=clip_right,
                bottom=bottom,
                fade=fade_tag,
                extra=extra_tags,
                text=full_text,
            )
        )

    return dialogue_lines


def _active_chunk_grapheme_events(
    layout: ChunkLayout,
    style: KaraokeStyle,
    max_end: float | None = None,
) -> list[str]:
    dialogue_lines = []
    extra_tags = _style_event_tags(style, fill_color=style.secondary_color)
    clip_pad_x, clip_pad_y = _grapheme_clip_padding(style)
    left = int(round(layout.line_left)) - clip_pad_x
    right = int(round(layout.line_right)) + clip_pad_x
    top = int(round(layout.y - style.font_size)) - clip_pad_y
    bottom = int(round(layout.y + style.font_size * 0.6)) + clip_pad_y

    for word, x, _y, width in layout.placed_words:
        for grapheme, grapheme_x, grapheme_width in _grapheme_positions(word, x, layout.y, width, style):
            grapheme_end = min(grapheme.end, max_end) if max_end is not None else grapheme.end
            if grapheme_end <= grapheme.start + 1e-3:
                continue

            total_ms = max(1, int(round((grapheme_end - grapheme.start) * 1000)))
            fade_in_ms = min(18, max(0, total_ms // 4))
            fade_out_ms = min(style.grapheme_fade_ms, max(1, total_ms - fade_in_ms - 1))
            fade_tag = f"\\fad({fade_in_ms},{fade_out_ms})" if fade_in_ms or fade_out_ms else ""
            start, end = _format_ass_interval(grapheme.start, grapheme_end)
            clip_left = max(left, int(round(grapheme_x - grapheme_width / 2)) - clip_pad_x)
            clip_right = min(right, int(round(grapheme_x + grapheme_width / 2)) + clip_pad_x)
            dialogue_lines.append(
                "Dialogue: 1,{start},{end},Karaoke,,0,0,0,,{{\\an6\\q2\\pos({x},{y})\\1a&H{alpha}&\\clip({clip_left},{top},{clip_right},{bottom}){fade}{extra}}}{text}".format(
                    start=start,
                    end=end,
                    x=int(round(layout.anchor_x)),
                    y=int(round(layout.y)),
                    alpha=style.active_fill_alpha,
                    clip_left=clip_left,
                    top=top,
                    clip_right=clip_right,
                    bottom=bottom,
                    fade=fade_tag,
                    extra=extra_tags,
                    text=layout.ass_text,
                )
            )

    return dialogue_lines


def _active_word_events(
    word: WordTiming,
    x: float,
    y: float,
    width: float,
    style: KaraokeStyle,
    max_end: float | None = None,
) -> list[str]:
    subwords = _subwords_for_word(word)
    if not subwords:
        return []

    reveal_widths = _build_reveal_widths(word, width, style)
    left = int(round(x - width / 2))
    right = int(round(x + width / 2))
    top = int(round(y - style.font_size))
    bottom = int(round(y + style.font_size * 0.6))
    full_text = _wrap_ass_rtl(word.word)
    dialogue_lines = []
    previous_reveal = 0.0
    extra_tags = _style_event_tags(style, fill_color=style.secondary_color)

    # The last subword whose start falls before max_end.
    # When it is not the naturally-last subword (i.e. some graphemes will be
    # skipped due to the boundary), we force its reveal_width to word_width so
    # the word is always fully lit before the segment disappears.
    if max_end is not None:
        last_included_idx = next(
            (j for j in range(len(subwords) - 1, -1, -1) if subwords[j].start < max_end),
            -1,
        )
    else:
        last_included_idx = len(subwords) - 1

    for j, (subword, reveal_width) in enumerate(zip(subwords, reveal_widths)):
        sw_start = subword.start
        # Clamp the subword's end so the karaoke highlight never bleeds into the
        # next segment — even if the subword timing in the data wasn't clipped.
        sw_end = min(subword.end, max_end) if max_end is not None else subword.end
        if sw_end <= sw_start:
            # Subword starts at or after the hard boundary — skip it entirely.
            # Do NOT advance previous_reveal here: the word is considered "done"
            # at the last included event which already snapped to word_width.
            continue

        # If this is the last event we will produce and some graphemes would
        # otherwise be skipped, snap the reveal to cover the entire word so
        # no characters are left permanently dimmed.
        effective_reveal = width if j == last_included_idx else reveal_width

        start, end = _format_ass_interval(sw_start, sw_end)
        start_left = max(left, int(round(right - previous_reveal)))
        end_left = max(left, int(round(right - effective_reveal)))
        transform_ms = max(1, int(round((sw_end - sw_start) * 1000)))
        transform = ""
        if end_left != start_left:
            transform = f"\\t(0,{transform_ms},\\clip({end_left},{top},{right},{bottom}))"
        dialogue_lines.append(
            "Dialogue: 1,{start},{end},Karaoke,,0,0,0,,{{\\an5\\q2\\pos({x},{y})\\1a&H{alpha}&\\clip({clip_left},{top},{right},{bottom}){transform}{extra}}}{text}".format(
                start=start,
                end=end,
                x=int(round(x)),
                y=int(round(y)),
                alpha=style.active_fill_alpha,
                clip_left=start_left,
                top=top,
                right=right,
                bottom=bottom,
                transform=transform,
                extra=extra_tags,
                text=full_text,
            )
        )
        previous_reveal = effective_reveal

    # Edge case: all subwords start at or after max_end (word is entirely past
    # the segment boundary — data quality issue but guard it here).
    # Produce a single one-frame flash so the word at least lights up once.
    if not dialogue_lines and max_end is not None:
        flash_end = max_end
        flash_start = max(word.start, flash_end - 0.033)  # ≈1 frame before end
        if flash_start < flash_end:
            dialogue_lines.append(
                "Dialogue: 1,{start},{end},Karaoke,,0,0,0,,{{\\an5\\q2\\pos({x},{y})\\1a&H{alpha}&\\clip({left},{top},{right},{bottom}){extra}}}{text}".format(
                    start=_format_ass_time(flash_start),
                    end=_format_ass_time(flash_end),
                    x=int(round(x)),
                    y=int(round(y)),
                    alpha=style.active_fill_alpha,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    extra=extra_tags,
                    text=full_text,
                )
            )

    return dialogue_lines


def _post_active_word_event(
    segment: TranscriptSegment,
    word: WordTiming,
    x: float,
    y: float,
    style: KaraokeStyle,
    max_end: float | None = None,
) -> str | None:
    """Generate an event that keeps the word fully lit after the sweep passes.

    Starts at word.end (when the karaoke sweep finishes this word) and lasts
    until segment.end, so sung words stay bright instead of reverting to the
    dimmed base layer.
    """
    post_start = word.end
    post_end = min(segment.end, max_end) if max_end is not None else segment.end
    if post_end - post_start < 0.02:
        return None
    extra_tags = _style_event_tags(style, fill_color=style.secondary_color)
    start, end = _format_ass_interval(post_start, post_end)
    return (
        "Dialogue: 1,{start},{end},Karaoke,,0,0,0,,{{\\an5\\q2\\pos({x},{y})\\1a&H{alpha}&{extra}}}{text}".format(
            start=start,
            end=end,
            x=int(round(x)),
            y=int(round(y)),
            alpha=style.active_fill_alpha,
            extra=extra_tags,
            text=_wrap_ass_rtl(word.word),
        )
    )


def _clip_words_to_boundary(words: list[WordTiming], boundary: float) -> list[WordTiming]:
    """Clip word (and subword) timings so nothing exceeds *boundary*.

    Words that start at or after the boundary are dropped entirely.
    Words that merely *end* past the boundary get their end clamped, and their
    subwords are clamped / filtered accordingly.
    """
    clipped: list[WordTiming] = []
    for word in words:
        if word.start >= boundary - 1e-6:
            break
        if word.end <= boundary + 1e-6:
            clipped.append(word)
            continue
        new_end = boundary
        new_subwords: list[SubWordTiming] = []
        for sw in word.subwords:
            if sw.start >= boundary - 1e-6:
                continue
            new_subwords.append(
                SubWordTiming(
                    text=sw.text,
                    start=sw.start,
                    end=min(sw.end, boundary),
                    confidence=sw.confidence,
                )
            )
        clipped.append(
            WordTiming(
                word=word.word,
                start=word.start,
                end=new_end,
                confidence=word.confidence,
                source=word.source,
                aligned=word.aligned,
                subwords=new_subwords,
            )
        )
    return clipped


def _enforce_segment_boundaries(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Clip overlaps so segment[i] ends no later than segment[i+1] starts.

    Only overlaps are fixed — gaps are left as-is so lines disappear naturally
    when their own timing expires, rather than being artificially extended.
      - Overlaps  → previous line disappears the moment the new one appears.
      - Gaps      → previous line keeps its original end time (no extension).
    The last segment keeps its original end time unchanged.
    Word and subword timings within a clipped segment are also clamped to
    the new boundary so downstream renderers see consistent data.
    """
    if len(segments) <= 1:
        return segments

    # Remove duplicate segments (identical text and nearly identical timing)
    deduped: list[TranscriptSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = deduped[-1]
        if seg.text == prev.text and abs(seg.start - prev.start) < 0.10:
            logger.debug("Removed duplicate segment: '%s' at %.3f", seg.text[:30], seg.start)
            continue
        deduped.append(seg)
    if not deduped:
        return segments

    result = list(deduped)
    for i in range(len(result) - 1):
        boundary = result[i + 1].start
        if boundary <= result[i].start + 1e-6:
            continue
        if result[i].end > boundary + 1e-6:
            clipped_words = _clip_words_to_boundary(result[i].words, boundary)
            result[i] = TranscriptSegment(
                words=clipped_words,
                text=result[i].text,
                start=result[i].start,
                end=boundary,
            )
    return result


class SrtRenderer:
    name = "srt_renderer"

    def render(self, segments: list[TranscriptSegment], output_path: str, style: KaraokeStyle | None = None):
        resolved_style = style or get_style()
        try:
            segments = _enforce_segment_boundaries(segments)
            render_chunks = _build_render_chunks(segments, resolved_style)
            lines = []
            for index, chunk in enumerate(render_chunks, 1):
                lines.append(str(index))
                lines.append(f"{_format_srt_time(chunk.start)} --> {_format_srt_time(chunk.end)}")
                lines.append(_build_srt_text(chunk.lines))
                lines.append("")
            Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            raise SubtitleGenerationError(str(exc), "יצירת קובץ SRT נכשלה.") from exc


class AssKaraokeRenderer:
    name = "ass_renderer"

    def render(
        self,
        segments: list[TranscriptSegment],
        output_path: str,
        style: KaraokeStyle | None = None,
        song_analysis: SongAnalysis | None = None,
    ):
        resolved_style = style or get_style()
        try:
            segments = _enforce_segment_boundaries(segments)
            render_chunks = _build_render_chunks(segments, resolved_style)
            dialogue_lines = []
            stack_offsets = _assign_ass_stack_offsets(render_chunks, resolved_style)
            preview_window = song_analysis.preview_window_seconds if song_analysis else 0.6
            word_chord_map = build_word_id_chord_map(segments, song_analysis.chord_events) if song_analysis else {}
            for chunk_index, chunk in enumerate(render_chunks):
                if not chunk.words:
                    continue

                chunk_layouts = _layout_chunk_geometries(
                    chunk.lines,
                    resolved_style,
                    stack_offset_units=stack_offsets.get(chunk_index, 0),
                )

                if resolved_style.effect_mode == "grapheme_highlight":
                    for chunk_layout in chunk_layouts:
                        dialogue_lines.append(_base_line_event(chunk.start, chunk.end, chunk_layout, resolved_style))
                        dialogue_lines.extend(_active_chunk_grapheme_events(chunk_layout, resolved_style, max_end=chunk.end))
                    continue

                for chunk_layout in chunk_layouts:
                    for word, x, y, width in chunk_layout.placed_words:
                        if resolved_style.effect_mode == "erase":
                            dialogue_lines.extend(
                                _erase_word_events(
                                    TranscriptSegment(words=chunk.words, start=chunk.start, end=chunk.end),
                                    word,
                                    x,
                                    y,
                                    width,
                                    resolved_style,
                                    segment_start=chunk.start,
                                    max_end=chunk.end,
                                )
                            )
                            continue

                        dialogue_lines.append(_base_word_event(chunk.start, chunk.end, word, x, y, resolved_style))
                        dialogue_lines.extend(_active_word_events(word, x, y, width, resolved_style, max_end=chunk.end))
                        post_event = _post_active_word_event(
                            TranscriptSegment(words=chunk.words, start=chunk.start, end=chunk.end),
                            word,
                            x,
                            y,
                            resolved_style,
                            max_end=chunk.end,
                        )
                        if post_event:
                            dialogue_lines.append(post_event)

            dialogue_lines.extend(_build_hud_events(song_analysis, resolved_style))
            content = _build_ass_header(resolved_style) + "\n".join(dialogue_lines) + "\n"
            Path(output_path).write_text(content, encoding="utf-8-sig")
        except Exception as exc:
            raise SubtitleGenerationError(str(exc), "יצירת קובץ ASS נכשלה.") from exc


def generate_srt(segments: list[TranscriptSegment], output_path: str, style: KaraokeStyle | None = None):
    SrtRenderer().render(segments, output_path, style=style)
    logger.info("Generated SRT at %s", output_path)


def generate_ass_karaoke(segments: list[TranscriptSegment], output_path: str, style: KaraokeStyle | None = None):
    AssKaraokeRenderer().render(segments, output_path, style)
    logger.info("Generated ASS at %s", output_path)
