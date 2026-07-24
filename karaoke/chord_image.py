"""Render a parsed Tab4U chord sheet to PNG images with pixel-exact placement.

Why an image and not text: Telegram messages cannot be given CSS, and its
"monospace" (<pre>) rendering does not give Hebrew and Latin glyphs identical
advance widths, so no amount of space-padding can line a Latin chord row up
with an RTL Hebrew lyric row. Here we control every pixel ourselves.

Alignment model (verified against Tab4U's live rendering, measured with DOM
Range rects): a chords row is LTR text right-aligned over an RTL lyric row,
and Tab4U pads the chords row with leading/trailing &nbsp; to exactly the
lyric row's character width. Therefore a chord whose label occupies character
columns [c, c+n) of a chords row of raw length L sits above lyric characters
[L-c-n, L-1-c] counted from the lyric string's start — i.e. the label's right
edge aligns with the right edge of lyric character index j0 = L-c-n. We anchor
each chord label's right edge at the pixel x of that lyric character boundary,
measured with the actual font (never assumed-monospace character math).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .chord_sources import _extract_chord_tokens, _transpose_row_text
from .harmony import transpose_chord_label

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_LYRIC_FONT_PATH = _FONT_DIR / "DejaVuSans.ttf"
_CHORD_FONT_PATH = _FONT_DIR / "DejaVuSans-Bold.ttf"

# Layout constants (base sizes; scaled down if a line would overflow).
_CANVAS_MIN_WIDTH = 700
_CANVAS_MAX_WIDTH = 1400
_MARGIN_X = 56
_MARGIN_TOP = 44
_MARGIN_BOTTOM = 48
_MAX_PAGE_CONTENT_HEIGHT = 2400

_LYRIC_SIZE = 34
_CHORD_SIZE = 31
_TITLE_SIZE = 44
_META_SIZE = 26
_HEADING_SIZE = 34

_BG = (255, 255, 255)
_LYRIC_COLOR = (30, 34, 44)
_CHORD_COLOR = (13, 90, 191)
_HEADING_COLOR = (90, 60, 140)
_TITLE_COLOR = (16, 16, 20)
_META_COLOR = (110, 116, 128)

_MIRROR_BRACKETS = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{", "<": ">", ">": "<"}
_HEBREW_CHAR_PATTERN = re.compile(r"[֐-׿]")


class _Fonts:
    def __init__(self, scale: float = 1.0):
        def load(path: Path, size: int) -> ImageFont.FreeTypeFont:
            return ImageFont.truetype(str(path), max(14, round(size * scale)))

        self.lyric = load(_LYRIC_FONT_PATH, _LYRIC_SIZE)
        self.chord = load(_CHORD_FONT_PATH, _CHORD_SIZE)
        self.title = load(_CHORD_FONT_PATH, _TITLE_SIZE)
        self.meta = load(_LYRIC_FONT_PATH, _META_SIZE)
        self.heading = load(_CHORD_FONT_PATH, _HEADING_SIZE)
        self._measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        self._char_cache: dict[tuple[int, str], float] = {}

    def char_width(self, font: ImageFont.FreeTypeFont, ch: str) -> float:
        key = (id(font), ch)
        width = self._char_cache.get(key)
        if width is None:
            width = self._measure.textlength(ch, font=font)
            self._char_cache[key] = width
        return width

    def text_width(self, font: ImageFont.FreeTypeFont, text: str) -> float:
        # Sum of per-char widths so it matches the char-by-char RTL renderer.
        return sum(self.char_width(font, ch) for ch in text)

    def line_height(self, font: ImageFont.FreeTypeFont) -> int:
        ascent, descent = font.getmetrics()
        return ascent + descent


def _split_bidi_runs(text: str) -> list[tuple[bool, str]]:
    """Split into (is_ltr_run, run) pieces, resolving neutrals like a browser.

    Simplified UBA with an RTL paragraph direction: Hebrew chars are strong
    RTL, ASCII letters/digits are strong LTR, everything else is neutral.
    A neutral stretch between two same-direction strong runs joins them (so
    "D6/F#: 0202x2" stays a single LTR block, exactly as a browser renders it
    inside an RTL cell); neutrals at the edges or between opposing runs take
    the paragraph (RTL) direction.
    """
    classes = []
    for ch in text:
        if _HEBREW_CHAR_PATTERN.match(ch):
            classes.append("R")
        elif ch.isascii() and ch.isalnum():
            classes.append("L")
        else:
            classes.append("N")

    resolved = list(classes)
    length = len(text)
    previous_strong = "R"  # paragraph direction
    index = 0
    while index < length:
        if classes[index] == "N":
            end = index
            while end < length and classes[end] == "N":
                end += 1
            next_strong = classes[end] if end < length else "R"
            fill = previous_strong if previous_strong == next_strong else "R"
            for position in range(index, end):
                resolved[position] = fill
            index = end
        else:
            previous_strong = classes[index]
            index += 1

    runs: list[tuple[bool, str]] = []
    for direction, ch in zip(resolved, text):
        if runs and runs[-1][0] == (direction == "L"):
            runs[-1] = (runs[-1][0], runs[-1][1] + ch)
        else:
            runs.append((direction == "L", ch))
    return runs


def _draw_rtl(draw: ImageDraw.ImageDraw, fonts: _Fonts, text: str, x_right: float, y: float,
              font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> float:
    """Draw logical-order text as RTL, anchored at x_right. Returns left x."""
    x = x_right
    for is_ltr, run in _split_bidi_runs(text):
        if is_ltr:
            width = fonts.text_width(font, run)
            draw.text((x - width, y), run, font=font, fill=fill)
            x -= width
        else:
            for ch in run:
                width = fonts.char_width(font, ch)
                draw.text((x - width, y), _MIRROR_BRACKETS.get(ch, ch), font=font, fill=fill)
                x -= width
    return x


def _draw_ltr_right_aligned(draw: ImageDraw.ImageDraw, fonts: _Fonts, text: str, x_right: float,
                            y: float, font: ImageFont.FreeTypeFont,
                            fill: tuple[int, int, int]) -> float:
    width = fonts.text_width(font, text)
    draw.text((x_right - width, y), text, font=font, fill=fill)
    return x_right - width


def _row_layout_text(row) -> str:
    return getattr(row, "layout_text", "") or row.text


def _is_heading(text: str) -> bool:
    stripped = " ".join(text.split())
    return stripped.endswith(":") and 0 < len(stripped[:-1].split()) <= 3


def _build_entries(sheet, *, title: str, key_line: str) -> list[dict]:
    """Flatten the parsed sheet into renderable line entries."""
    entries: list[dict] = [
        {"type": "title", "text": title},
    ]
    if key_line:
        entries.append({"type": "meta", "text": key_line})
    entries.append({"type": "blank"})

    for table in sheet.tables:
        consumed_index = -1
        for index, row in enumerate(table):
            if index == consumed_index:
                continue
            if row.kind == "chords":
                next_row = table[index + 1] if index + 1 < len(table) else None
                paired = (
                    row.tokens
                    and next_row is not None
                    and next_row.kind == "song"
                    and next_row.text.strip()
                    and not _is_heading(next_row.text)
                )
                if paired:
                    entries.append(
                        {
                            "type": "pair",
                            "chords": _row_layout_text(row),
                            "lyric": _row_layout_text(next_row),
                        }
                    )
                    consumed_index = index + 1
                else:
                    entries.append({"type": "chords", "text": _row_layout_text(row)})
                continue
            if _is_heading(row.text):
                entries.append({"type": "heading", "text": row.text.strip()})
            elif row.text.strip():
                entries.append({"type": "lyric", "text": _row_layout_text(row).rstrip()})
        entries.append({"type": "blank"})

    while entries and entries[-1]["type"] == "blank":
        entries.pop()
    return entries


def _entry_height(entry: dict, fonts: _Fonts) -> int:
    kind = entry["type"]
    if kind == "title":
        return fonts.line_height(fonts.title) + 10
    if kind == "meta":
        return fonts.line_height(fonts.meta) + 8
    if kind == "blank":
        return 26
    if kind == "heading":
        return fonts.line_height(fonts.heading) + 12
    if kind == "chords":
        return fonts.line_height(fonts.chord) + 8
    if kind == "lyric":
        return fonts.line_height(fonts.lyric) + 10
    # pair: chord line directly above lyric line
    return fonts.line_height(fonts.chord) + 2 + fonts.line_height(fonts.lyric) + 12


def _entry_width(entry: dict, fonts: _Fonts, semitones: int) -> float:
    kind = entry["type"]
    if kind == "blank":
        return 0.0
    if kind == "title":
        return fonts.text_width(fonts.title, entry["text"])
    if kind == "meta":
        return fonts.text_width(fonts.meta, entry["text"])
    if kind == "heading":
        return fonts.text_width(fonts.heading, entry["text"])
    if kind == "chords":
        return fonts.text_width(fonts.chord, _transpose_row_text(entry["text"], semitones))
    if kind == "lyric":
        return fonts.text_width(fonts.lyric, entry["text"])
    lyric_width = fonts.text_width(fonts.lyric, entry["lyric"].rstrip())
    # A chord anchored left of the lyric's start can stick out further.
    chord_extent = 0.0
    for x_right, label in _pair_chord_positions(entry, fonts, semitones, x_right_edge=0.0):
        chord_extent = max(chord_extent, fonts.text_width(fonts.chord, label) - x_right)
    return max(lyric_width, chord_extent)


def _pair_chord_positions(entry: dict, fonts: _Fonts, semitones: int,
                          *, x_right_edge: float) -> list[tuple[float, str]]:
    """Compute (x_right, label) for each chord in a pair entry.

    x positions are relative to x_right_edge (the shared right margin of the
    chord line and its lyric line); they grow negative leftwards.
    """
    chord_row = entry["chords"]
    lyric = entry["lyric"].rstrip()
    row_length = len(chord_row)
    space_width = fonts.char_width(fonts.lyric, " ")

    # Cumulative pixel width of the first j lyric characters.
    prefix = [0.0]
    for ch in lyric:
        prefix.append(prefix[-1] + fonts.char_width(fonts.lyric, ch))

    def prefix_width(j: int) -> float:
        if j <= len(lyric):
            return prefix[j]
        return prefix[-1] + (j - len(lyric)) * space_width

    positions: list[tuple[float, str]] = []
    for token in _extract_chord_tokens(chord_row):
        label = transpose_chord_label(token.label, semitones) if semitones else token.label
        j0 = max(0, row_length - token.column - len(token.label))
        positions.append((x_right_edge - prefix_width(j0), label))
    return positions


def _render_pair(draw: ImageDraw.ImageDraw, fonts: _Fonts, entry: dict, x_right: float, y: float,
                 semitones: int) -> int:
    chord_height = fonts.line_height(fonts.chord)
    positions = _pair_chord_positions(entry, fonts, semitones, x_right_edge=x_right)
    positions.sort(key=lambda item: -item[0])
    min_gap = fonts.char_width(fonts.chord, " ")
    previous_left: float | None = None
    for anchor_x, label in positions:
        if previous_left is not None:
            anchor_x = min(anchor_x, previous_left - min_gap)
        previous_left = _draw_ltr_right_aligned(
            draw, fonts, label, anchor_x, y, fonts.chord, _CHORD_COLOR
        )
    lyric_y = y + chord_height + 2
    _draw_rtl(draw, fonts, entry["lyric"].rstrip(), x_right, lyric_y, fonts.lyric, _LYRIC_COLOR)
    return chord_height + 2 + fonts.line_height(fonts.lyric) + 12


def _render_entry(draw: ImageDraw.ImageDraw, fonts: _Fonts, entry: dict, x_right: float, y: float,
                  semitones: int) -> int:
    kind = entry["type"]
    if kind == "blank":
        return _entry_height(entry, fonts)
    if kind == "title":
        _draw_rtl(draw, fonts, entry["text"], x_right, y, fonts.title, _TITLE_COLOR)
    elif kind == "meta":
        _draw_rtl(draw, fonts, entry["text"], x_right, y, fonts.meta, _META_COLOR)
    elif kind == "heading":
        _draw_rtl(draw, fonts, entry["text"], x_right, y, fonts.heading, _HEADING_COLOR)
    elif kind == "chords":
        text = _transpose_row_text(entry["text"], semitones)
        _draw_ltr_right_aligned(draw, fonts, text.strip(), x_right, y, fonts.chord, _CHORD_COLOR)
    elif kind == "lyric":
        _draw_rtl(draw, fonts, entry["text"], x_right, y, fonts.lyric, _LYRIC_COLOR)
    elif kind == "pair":
        return _render_pair(draw, fonts, entry, x_right, y, semitones)
    return _entry_height(entry, fonts)


def _paginate(entries: list[dict], fonts: _Fonts) -> list[list[dict]]:
    pages: list[list[dict]] = []
    current: list[dict] = []
    height = 0
    for entry in entries:
        entry_height = _entry_height(entry, fonts)
        if current and height + entry_height > _MAX_PAGE_CONTENT_HEIGHT:
            pages.append(current)
            current = []
            height = 0
        if entry["type"] == "blank" and not current:
            continue
        current.append(entry)
        height += entry_height
    if current:
        pages.append(current)
    return pages


def render_chord_sheet_images(sheet, *, title: str, original_key: str = "",
                              target_key: str = "", semitones: int = 0) -> list[bytes]:
    """Render the parsed sheet to one or more PNGs (returned as bytes).

    semitones=0 renders the original key; non-zero renders transposed labels
    at the original anchor positions.
    """
    if sheet is None or not getattr(sheet, "tables", None):
        return []

    if semitones and target_key:
        key_line = f"סולם קל: {target_key} (מקור: {original_key})" if original_key else f"סולם: {target_key}"
    elif original_key:
        key_line = f"סולם: {original_key}"
    else:
        key_line = ""

    entries = _build_entries(sheet, title=title, key_line=key_line)

    fonts = _Fonts()
    content_limit = _CANVAS_MAX_WIDTH - 2 * _MARGIN_X
    max_width = max((_entry_width(entry, fonts, semitones) for entry in entries), default=0.0)
    if max_width > content_limit:
        fonts = _Fonts(scale=content_limit / max_width)
        max_width = max((_entry_width(entry, fonts, semitones) for entry in entries), default=0.0)

    canvas_width = int(min(_CANVAS_MAX_WIDTH, max(_CANVAS_MIN_WIDTH, max_width + 2 * _MARGIN_X)))
    x_right = canvas_width - _MARGIN_X

    images: list[bytes] = []
    for page in _paginate(entries, fonts):
        page_height = sum(_entry_height(entry, fonts) for entry in page)
        image = Image.new("RGB", (canvas_width, _MARGIN_TOP + page_height + _MARGIN_BOTTOM), _BG)
        draw = ImageDraw.Draw(image)
        y = _MARGIN_TOP
        for entry in page:
            y += _render_entry(draw, fonts, entry, x_right, y, semitones)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        images.append(buffer.getvalue())
    return images
