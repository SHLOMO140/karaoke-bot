from karaoke.chord_sources import lookup_external_chord_sheet, lookup_external_chord_sheet_by_title
from karaoke.web_search import SearchResult
from karaoke.models import TranscriptSegment, WordTiming


TAB4U_SAMPLE_HTML = """
<div id="songContentTPL" align="right">
  <table border="0" cellspacing="0" cellpadding="0">
    <tbody>
      <tr><td class="song">פתיחה:</td></tr>
      <tr>
        <td class="chords">
          |&nbsp;<span class="c_C">Em</span>&nbsp;|&nbsp;<span class="c_C">Am</span>&nbsp;|
        </td>
      </tr>
    </tbody>
  </table>
  <br />
  <table border="0" cellspacing="0" cellpadding="0">
    <tbody>
      <tr>
        <td class="chords">
          <span class="c_C">Em</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="c_C">Am</span>
        </td>
      </tr>
      <tr><td class="song">כאן&nbsp;עוד&nbsp;יום</td></tr>
      <tr><td class="chords"><span class="c_C">B7</span></td></tr>
      <tr><td class="song">שוב&nbsp;שרים</td></tr>
    </tbody>
  </table>
</div>
"""

TAB4U_MEDLEY_PART_ONE_HTML = """
<div id="songContentTPL" align="right">
  <table border="0" cellspacing="0" cellpadding="0">
    <tbody>
      <tr><td class="chords"><span class="c_C">Am</span>     <span class="c_C">F</span></td></tr>
      <tr><td class="song">הלב שלי נפתח הלילה</td></tr>
      <tr><td class="chords"><span class="c_C">C</span></td></tr>
      <tr><td class="song">רוח טובה עוברת בי</td></tr>
      <tr><td class="chords"><span class="c_C">G</span></td></tr>
      <tr><td class="song">עוד שיר עולה מתוך שתיקה</td></tr>
    </tbody>
  </table>
</div>
"""

TAB4U_MEDLEY_PART_TWO_HTML = """
<div id="songContentTPL" align="right">
  <table border="0" cellspacing="0" cellpadding="0">
    <tbody>
      <tr><td class="chords"><span class="c_C">Dm</span>     <span class="c_C">Bb</span></td></tr>
      <tr><td class="song">בין האורות אני חוזר</td></tr>
      <tr><td class="chords"><span class="c_C">C</span></td></tr>
      <tr><td class="song">כל הרחובות שרים איתי</td></tr>
      <tr><td class="chords"><span class="c_C">Am</span></td></tr>
      <tr><td class="song">הלילה לא נגמר לעולם</td></tr>
    </tbody>
  </table>
</div>
"""


def _segments():
    return [
        TranscriptSegment(
            words=[
                WordTiming("כאן", 0.0, 0.5),
                WordTiming("עוד", 0.5, 1.0),
                WordTiming("יום", 1.0, 1.5),
            ],
            text="כאן עוד יום",
            start=0.0,
            end=1.5,
        ),
        TranscriptSegment(
            words=[
                WordTiming("שוב", 1.5, 2.0),
                WordTiming("שרים", 2.0, 2.5),
            ],
            text="שוב שרים",
            start=1.5,
            end=2.5,
        ),
    ]


def test_lookup_external_chord_sheet_parses_tab4u_and_transposes(monkeypatch):
    monkeypatch.setattr(
        "karaoke.chord_sources._search_known_site_results",
        lambda title, queries, context: [
            SearchResult(
                title="demo",
                snippet="Tab4U internal search",
                url="https://www.tab4u.com/lyrics/songs/1_demo.html",
            )
        ],
    )
    monkeypatch.setattr("karaoke.chord_sources._search_tab4u_results", lambda query: [])
    monkeypatch.setattr("karaoke.chord_sources._fetch_text", lambda url, timeout=15: TAB4U_SAMPLE_HTML)

    analysis = lookup_external_chord_sheet(
        "אמן - דמו",
        _segments(),
        provider="librosa_harmony_v5",
    )

    assert analysis is not None
    assert analysis.provider == "librosa_harmony_v5"
    assert analysis.chord_source_name == "Tab4U"
    assert analysis.chord_source_url == "https://www.tab4u.com/tabs/songs/1_demo.html"
    assert analysis.original_key == "Em"
    assert analysis.target_key == "Am"
    assert analysis.transpose_semitones == 5
    assert [event.label for event in analysis.original_chord_events] == ["Em", "Am", "B7"]
    assert [event.label for event in analysis.chord_events] == ["Am", "Dm", "E7"]
    assert "כותרת: אמן - דמו" in analysis.chord_sheet_text
    assert "סולם מקור: Em" in analysis.chord_sheet_text
    assert "סולם קל: Am" in analysis.chord_sheet_text
    assert "Am" in analysis.chord_sheet_text
    assert "Dm" in analysis.chord_sheet_text
    assert "E7" in analysis.chord_sheet_text


def test_lookup_external_chord_sheet_by_title_parses_without_segments(monkeypatch):
    monkeypatch.setattr(
        "karaoke.chord_sources._search_known_site_results",
        lambda title, queries, context: [
            SearchResult(
                title="אמן - דמו",
                snippet="Tab4U internal search",
                url="https://www.tab4u.com/lyrics/songs/1_demo.html",
            )
        ],
    )
    monkeypatch.setattr("karaoke.chord_sources._search_tab4u_results", lambda query: [])
    monkeypatch.setattr("karaoke.chord_sources._fetch_text", lambda url, timeout=15: TAB4U_SAMPLE_HTML)

    analysis = lookup_external_chord_sheet_by_title(
        "אמן - דמו",
        provider="librosa_harmony_v5",
    )

    assert analysis is not None
    assert analysis.provider == "librosa_harmony_v5"
    assert analysis.chord_source_name == "Tab4U"
    assert analysis.chord_source_url == "https://www.tab4u.com/tabs/songs/1_demo.html"
    assert analysis.chord_sheet_text.strip()
    assert "סולם מקור:" in analysis.chord_sheet_text
    assert "סולם קל:" in analysis.chord_sheet_text
    assert "Am" in analysis.chord_sheet_text
    assert "Dm" in analysis.chord_sheet_text
    assert "E7" in analysis.chord_sheet_text


def test_lookup_external_chord_sheet_stitches_medley_sections_from_multiple_sources(monkeypatch):
    segments = [
        TranscriptSegment(
            words=[
                WordTiming("הלב", 0.0, 0.3),
                WordTiming("שלי", 0.3, 0.6),
                WordTiming("נפתח", 0.6, 0.9),
                WordTiming("הלילה", 0.9, 1.3),
            ],
            text="הלב שלי נפתח הלילה",
            start=0.0,
            end=1.3,
        ),
        TranscriptSegment(
            words=[
                WordTiming("רוח", 1.4, 1.7),
                WordTiming("טובה", 1.7, 2.0),
                WordTiming("עוברת", 2.0, 2.4),
                WordTiming("בי", 2.4, 2.7),
            ],
            text="רוח טובה עוברת בי",
            start=1.4,
            end=2.7,
        ),
        TranscriptSegment(
            words=[
                WordTiming("עוד", 2.8, 3.0),
                WordTiming("שיר", 3.0, 3.3),
                WordTiming("עולה", 3.3, 3.6),
                WordTiming("מתוך", 3.6, 3.9),
                WordTiming("שתיקה", 3.9, 4.3),
            ],
            text="עוד שיר עולה מתוך שתיקה",
            start=2.8,
            end=4.3,
        ),
        TranscriptSegment(
            words=[
                WordTiming("בין", 4.5, 4.8),
                WordTiming("האורות", 4.8, 5.2),
                WordTiming("אני", 5.2, 5.5),
                WordTiming("חוזר", 5.5, 5.9),
            ],
            text="בין האורות אני חוזר",
            start=4.5,
            end=5.9,
        ),
        TranscriptSegment(
            words=[
                WordTiming("כל", 6.0, 6.2),
                WordTiming("הרחובות", 6.2, 6.7),
                WordTiming("שרים", 6.7, 7.1),
                WordTiming("איתי", 7.1, 7.5),
            ],
            text="כל הרחובות שרים איתי",
            start=6.0,
            end=7.5,
        ),
        TranscriptSegment(
            words=[
                WordTiming("הלילה", 7.6, 8.0),
                WordTiming("לא", 8.0, 8.2),
                WordTiming("נגמר", 8.2, 8.6),
                WordTiming("לעולם", 8.6, 9.0),
            ],
            text="הלילה לא נגמר לעולם",
            start=7.6,
            end=9.0,
        ),
    ]

    monkeypatch.setattr("karaoke.chord_sources._search_known_site_results", lambda title, queries, context: [])
    monkeypatch.setattr(
        "karaoke.chord_sources._search_tab4u_results",
        lambda query: [
            SearchResult(
                title="part one",
                snippet="Tab4U internal search",
                url="https://www.tab4u.com/lyrics/songs/11_part_one.html",
            ),
            SearchResult(
                title="part two",
                snippet="Tab4U internal search",
                url="https://www.tab4u.com/lyrics/songs/22_part_two.html",
            ),
        ],
    )
    monkeypatch.setattr(
        "karaoke.chord_sources._fetch_text",
        lambda url, timeout=15: (
            TAB4U_MEDLEY_PART_ONE_HTML
            if "11_part_one" in url
            else TAB4U_MEDLEY_PART_TWO_HTML
        ),
    )

    analysis = lookup_external_chord_sheet(
        "אמן | מחרוזת גדולה",
        segments,
        provider="librosa_harmony_v5",
    )

    assert analysis is not None
    assert analysis.chord_source_name == "Tab4U medley"
    assert "11_part_one" in analysis.chord_source_url
    assert "22_part_two" in analysis.chord_source_url
    assert analysis.target_key == ""
    assert analysis.transpose_semitones == 0
    assert {"Am", "F", "C", "G", "Dm", "Bb"} <= {
        event.label for event in analysis.original_chord_events
    }
    assert "הלב שלי נפתח הלילה" in analysis.chord_sheet_text
    assert "בין האורות אני חוזר" in analysis.chord_sheet_text
