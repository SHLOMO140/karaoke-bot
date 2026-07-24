"""Tests for the PNG chord-sheet renderer (pixel-exact chord-over-word layout).

The word-assignment expectations here are ground truth measured on Tab4U's own
live rendering of אושר כהן - כולם גנבים (DOM Range rects, July 2026): the
chords row " Ab              Fm             " (32 chars, padded with edge
&nbsp; to the lyric row's width) puts Fm above בעיניים (word index 2) and Ab
above תגלגלי (word index 5) of the RTL lyric line.
"""

import io
import re

from PIL import Image

from karaoke.chord_image import (
    _Fonts,
    _build_entries,
    _pair_chord_positions,
    render_chord_sheet_images,
)
from karaoke.chord_sources import (
    _ChordRow,
    _ChordToken,
    _ParsedTab4USheet,
    _extract_chord_tokens,
)

CHORD_LAYOUT = " Ab              Fm             "
LYRIC = "תסתכלי לי בעיניים בטח שוב תגלגלי"


def _sheet():
    rows = [
        _ChordRow(
            kind="chords",
            text=CHORD_LAYOUT.strip(),
            tokens=_extract_chord_tokens(CHORD_LAYOUT.strip()),
            layout_text=CHORD_LAYOUT,
        ),
        _ChordRow(kind="song", text=LYRIC, layout_text=LYRIC),
    ]
    return _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[LYRIC],
        line_word_pairs=[], chord_labels=["Ab", "Fm"],
    )


def _word_of_anchor(anchor_x: float, lyric: str, fonts: _Fonts) -> int:
    """Which RTL word index the anchor x lands on (x_right_edge=0)."""
    prefix = [0.0]
    for ch in lyric:
        prefix.append(prefix[-1] + fonts.char_width(fonts.lyric, ch))
    best, best_dist = None, float("inf")
    for idx, match in enumerate(re.finditer(r"\S+", lyric)):
        left, right = -prefix[match.end()], -prefix[match.start()]
        dist = 0.0 if left <= anchor_x <= right else min(abs(anchor_x - left), abs(anchor_x - right))
        if dist < best_dist:
            best_dist, best = dist, idx
    return best


def test_chords_anchor_above_the_words_tab4u_shows_them_on():
    entries = [e for e in _build_entries(_sheet(), title="t", key_line="") if e["type"] == "pair"]
    assert len(entries) == 1
    fonts = _Fonts()
    positions = {
        label: anchor_x
        for anchor_x, label in _pair_chord_positions(entries[0], fonts, 0, x_right_edge=0.0)
    }
    assert set(positions) == {"Ab", "Fm"}
    # Live Tab4U ground truth: Fm over word 2 (בעיניים), Ab over word 5 (תגלגלי).
    assert _word_of_anchor(positions["Fm"], LYRIC, fonts) == 2
    assert _word_of_anchor(positions["Ab"], LYRIC, fonts) == 5


def test_transposed_labels_keep_their_anchors():
    entries = [e for e in _build_entries(_sheet(), title="t", key_line="") if e["type"] == "pair"]
    fonts = _Fonts()
    original = sorted(x for x, _l in _pair_chord_positions(entries[0], fonts, 0, x_right_edge=0.0))
    transposed = _pair_chord_positions(entries[0], fonts, 4, x_right_edge=0.0)
    assert sorted(label for _x, label in transposed) == ["Am", "C"]  # Fm+4=Am, Ab+4=C
    assert sorted(x for x, _l in transposed) == original


def test_render_produces_decodable_png_with_sane_dimensions():
    images = render_chord_sheet_images(
        _sheet(), title="אושר כהן - כולם גנבים", original_key="Fm", target_key="Am", semitones=0
    )
    assert len(images) == 1
    image = Image.open(io.BytesIO(images[0]))
    assert image.format == "PNG"
    assert image.width >= 600
    assert 100 < image.height < 3200


def test_layout_text_fallback_to_plain_text():
    # Sheets from older code paths may lack layout_text; renderer must not crash.
    rows = [
        _ChordRow(kind="chords", text="Cm", tokens=[_ChordToken("Cm", 0)]),
        _ChordRow(kind="song", text="את מלכה אבל עדיין"),
    ]
    sheet = _ParsedTab4USheet(
        source_url="u", tables=[rows], lyric_lines=[], line_word_pairs=[], chord_labels=["Cm"],
    )
    images = render_chord_sheet_images(sheet, title="t")
    assert images and images[0].startswith(b"\x89PNG")
