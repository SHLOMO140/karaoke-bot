import urllib.parse

import karaoke.lyrics_verifier as lyrics_verifier
import pytest
import yt_dlp
from karaoke.lyrics_verifier import (
    DuckDuckGoLyricsVerifier,
    HybridLyricsVerifier,
    MultiStepLyricsVerifier,
    _build_query_variants,
    _canonicalize_lyrics_source_url,
    _extract_youtube_source_lines,
    _extract_site_specific_lyrics,
    _evaluate_candidate_text_against_draft,
    _parse_bing_results,
    _find_best_lyrics_window,
    _parse_duckduckgo_results,
    _parse_tab4u_search_results,
    _score_candidate_url,
    _search_web_results,
    _search_duckduckgo_results,
)
from karaoke.models import LyricsVerificationResult, TranscriptDraft, TranscriptSegment, WordTiming


CORRECT_LINE = "\u05db\u05db\u05d4 \u05e6\u05e8\u05d9\u05da \u05dc\u05d4\u05d9\u05d5\u05ea \u05db\u05ea\u05d5\u05d1 \u05d1\u05e2\u05d1\u05e8\u05d9\u05ea"
SECOND_LINE = "\u05d5\u05e2\u05d5\u05d3 \u05e9\u05d5\u05e8\u05d4 \u05d4\u05d2\u05d9\u05d5\u05e0\u05d9\u05ea"
THIRD_LINE = "\u05d5\u05d0\u05d6 \u05d4\u05e9\u05d9\u05e8 \u05de\u05de\u05e9\u05d9\u05da \u05e2\u05d5\u05d3"
WRONG_LINE = "\u05db\u05db\u05d4 \u05e6\u05e8\u05d9\u05dc \u05dc\u05d4\u05d9\u05d5\u05ea \u05db\u05ea\u05d5\u05d1 \u05d1\u05e2\u05d1\u05e8\u05d9\u05ea"
TITLE = "\u05e9\u05d9\u05e8 \u05dc\u05d3\u05d5\u05d2\u05de\u05d4"

SEARCH_HTML = f"""
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fshironet.mako.co.il%2Fsong">{TITLE}</a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fshironet.mako.co.il%2Fsong">
      {CORRECT_LINE}
    </a>
  </div>
</div>
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnagnu.co.il%2Flyrics">{TITLE} 2</a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnagnu.co.il%2Flyrics">
      {CORRECT_LINE}
    </a>
  </div>
</div>
"""

SINGLE_RESULT_HTML = f"""
<div class="result results_links results_links_deep web-result ">
  <div class="links_main links_deep result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fshironet.mako.co.il%2Fsong">{TITLE}</a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fshironet.mako.co.il%2Fsong">
      {CORRECT_LINE}
    </a>
  </div>
</div>
"""

LYRICS_PAGE = f"""
<html><body>
{CORRECT_LINE}<br>
{SECOND_LINE}
</body></html>
"""

YOUTUBE_EXPECTED_LINES = [CORRECT_LINE, SECOND_LINE, THIRD_LINE]
YOUTUBE_NOISY_DESCRIPTION_CASES = [
    (
        "hebrew_credits_edges",
        (
            "רן רהב תקשורת ויחסי ציבור\n"
            "תופים אבי אבידני\n"
            "קלידים ותכנותים יעקב למאי\n\n"
            f"{CORRECT_LINE}\n{SECOND_LINE}\n{THIRD_LINE}\n\n"
            "סטיילינג ישראל רחמני\n"
            "איפור בר חג׳ג׳\n"
            "הפצה דיגטלית"
        ),
    ),
    (
        "colons_and_booking",
        (
            "להזמנת הופעות: 050-1234567\n"
            "עיבוד והפקה: יעקב למאי\n"
            "מילים ולחן: דודו כהן\n\n"
            f"{CORRECT_LINE}\n{SECOND_LINE}\n{THIRD_LINE}"
        ),
    ),
    (
        "english_credit_edges",
        (
            "Public Relations Ran Rahav\n"
            "Drums Avi Avidani\n\n"
            f"{CORRECT_LINE}\n{SECOND_LINE}\n{THIRD_LINE}\n\n"
            "Styling Israel Rahmani\n"
            "Makeup Bar Hagag"
        ),
    ),
    (
        "social_links_and_urls",
        (
            "עקבו אחרי באינסטגרם @artist\n"
            "www.example.com/watch?v=1\n\n"
            f"{CORRECT_LINE}\n{SECOND_LINE}\n{THIRD_LINE}\n\n"
            "להזמנות הופעות 03-5551234"
        ),
    ),
    (
        "internal_instrument_credit",
        (
            f"{CORRECT_LINE}\n"
            "תופים אבי אבידני\n"
            f"{SECOND_LINE}\n"
            f"{THIRD_LINE}"
        ),
    ),
    (
        "internal_styling_credit",
        (
            f"{CORRECT_LINE}\n"
            f"{SECOND_LINE}\n"
            "סטיילינג ישראל רחמני\n"
            f"{THIRD_LINE}"
        ),
    ),
    (
        "mix_master_middle",
        (
            f"{CORRECT_LINE}\n"
            "Mix and Master Izik Piliya\n"
            f"{SECOND_LINE}\n"
            f"{THIRD_LINE}"
        ),
    ),
    (
        "management_and_distribution",
        (
            "ניהול אישי אבי לוי\n\n"
            f"{CORRECT_LINE}\n{SECOND_LINE}\n{THIRD_LINE}\n\n"
            "Digital Distribution Mobile1"
        ),
    ),
    (
        "photo_video_intro",
        (
            "צילום דניאל קמינסקי\n"
            "בימוי משה כהן\n\n"
            f"{CORRECT_LINE}\n{SECOND_LINE}\n{THIRD_LINE}"
        ),
    ),
    (
        "instrument_roles_english_and_hebrew",
        (
            "Piano Gal Mendel\n"
            "גיטרות מור אוזן\n\n"
            f"{CORRECT_LINE}\n{SECOND_LINE}\n{THIRD_LINE}\n\n"
            "Graphics Studio Bar"
        ),
    ),
]


def _draft() -> TranscriptDraft:
    first_line_words = WRONG_LINE.split()
    second_line_words = SECOND_LINE.split()
    first_segment = TranscriptSegment(
        words=[
            WordTiming(first_line_words[0], 0.0, 0.5),
            WordTiming(first_line_words[1], 0.5, 1.0),
            WordTiming(first_line_words[2], 1.0, 1.5),
            WordTiming(first_line_words[3], 1.5, 2.0),
            WordTiming(first_line_words[4], 2.0, 2.5),
        ],
        text=WRONG_LINE,
        start=0.0,
        end=2.5,
    )
    second_segment = TranscriptSegment(
        words=[
            WordTiming(second_line_words[0], 2.6, 3.0),
            WordTiming(second_line_words[1], 3.0, 3.5),
            WordTiming(second_line_words[2], 3.5, 4.0),
        ],
        text=SECOND_LINE,
        start=2.6,
        end=4.0,
    )
    return TranscriptDraft(segments=[first_segment, second_segment], provider="test")


def _fake_youtube_dl_with_description(description: str):
    class _FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            assert download is False
            assert "youtube.com/watch" in url
            return {
                "title": TITLE,
                "description": description,
            }

    return _FakeYoutubeDL


def test_parse_duckduckgo_results_extracts_title_snippet_and_url():
    results = _parse_duckduckgo_results(SINGLE_RESULT_HTML)

    assert len(results) == 1
    assert results[0].title == TITLE
    assert CORRECT_LINE in results[0].snippet
    assert results[0].url == "https://shironet.mako.co.il/song"


def test_query_variants_include_artist_song_and_site_specific_searches():
    queries = _build_query_variants(
        "Noam Bettan | \u05e0\u05d5\u05e2\u05dd \u05d1\u05ea\u05df - \u05d4\u05d9\u05d5\u05dd (Prod. by Doli & Penn)",
        "\u05d4\u05d9\u05d5\u05dd \u05d0\u05e0\u05d9 \u05dc\u05d0 \u05de\u05e4\u05d7\u05d3 \u05dc\u05d7\u05dc\u05d5\u05dd",
    )

    assert any("\u05e0\u05d5\u05e2\u05dd \u05d1\u05ea\u05df" in query and "\u05d4\u05d9\u05d5\u05dd" in query for query in queries)
    assert any("\u05de\u05d9\u05dc\u05d9\u05dd" in query for query in queries)
    assert any("site:lyricstranslate.com" in query for query in queries)
    assert any("site:tab4u.com" in query for query in queries)


def test_query_variants_include_lyric_snippet_from_repeated_chorus():
    queries = _build_query_variants(
        "\u05d0\u05de\u05df - \u05d4\u05d9\u05dc\u05d4",
        "\n".join(
            [
                "\u05d0\u05d5\u05e8 \u05d2\u05d3\u05d5\u05dc \u05d1\u05dc\u05d9\u05dc\u05d4",
                "\u05d0\u05e0\u05d9 \u05e2\u05d5\u05e3 \u05e8\u05d7\u05d5\u05e7",
                "\u05d0\u05d5\u05e8 \u05d2\u05d3\u05d5\u05dc \u05d1\u05dc\u05d9\u05dc\u05d4",
            ]
        ),
    )

    assert any('"\u05d0\u05d5\u05e8 \u05d2\u05d3\u05d5\u05dc \u05d1\u05dc\u05d9\u05dc\u05d4"' in query for query in queries)


def test_parse_duckduckgo_results_accepts_div_snippet_blocks():
    html = f"""
    <div class="result">
      <h2><a class="result__a extra" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ftab4u.com%2Fsong">{TITLE}</a></h2>
      <div class="result__snippet js-result-snippet">{CORRECT_LINE}</div>
    </div>
    """

    results = _parse_duckduckgo_results(html)

    assert len(results) == 1
    assert results[0].url == "https://tab4u.com/song"
    assert CORRECT_LINE in results[0].snippet


def test_parse_bing_results_decodes_redirect_url():
    html = """
    <li class="b_algo">
      <h2><a href="https://www.bing.com/ck/a?!&amp;&amp;u=a1aHR0cHM6Ly90YWI0dS5jb20vc29uZw==&amp;ntb=1">שיר לדוגמה</a></h2>
      <div class="b_caption"><p>ככה צריך להיות כתוב בעברית</p></div>
    </li>
    """

    results = _parse_bing_results(html)

    assert len(results) == 1
    assert results[0].url == "https://tab4u.com/song"
    assert results[0].title == TITLE
    assert CORRECT_LINE in results[0].snippet


def test_search_relaxation_recovers_results_for_over_strict_query(monkeypatch):
    lyrics_verifier.SEARCH_RESULT_CACHE.clear()

    def fake_fetch_text(url: str, timeout: int = 12):
        del timeout
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
        decoded_query = urllib.parse.unquote_plus(query)
        if "site:" in decoded_query or '"' in decoded_query:
            return "<html><body>no results</body></html>"
        return SINGLE_RESULT_HTML

    monkeypatch.setattr("karaoke.lyrics_verifier._fetch_text", fake_fetch_text)

    results = _search_duckduckgo_results('site:tab4u.com "\u05e9\u05d9\u05e8 \u05dc\u05d3\u05d5\u05d2\u05de\u05d4" "\u05db\u05db\u05d4 \u05e6\u05e8\u05d9\u05da \u05dc\u05d4\u05d9\u05d5\u05ea" \u05de\u05d9\u05dc\u05d9\u05dd')

    assert len(results) == 1
    assert results[0].url == "https://shironet.mako.co.il/song"


def test_search_web_results_falls_back_to_bing(monkeypatch):
    monkeypatch.setattr("karaoke.lyrics_verifier._search_duckduckgo_results", lambda query: [])
    monkeypatch.setattr(
        "karaoke.lyrics_verifier._search_bing_results",
        lambda query: [lyrics_verifier.SearchResult(title=TITLE, snippet=CORRECT_LINE, url="https://tab4u.com/song")],
    )

    results = _search_web_results("demo query")

    assert len(results) == 1
    assert results[0].url == "https://tab4u.com/song"


def test_canonicalize_lyrics_source_url_prefers_tab4u_lyrics_pages():
    url = _canonicalize_lyrics_source_url(
        "https://www.tab4u.com/tabs/songs/123_demo.html?type=piano"
    )

    assert url == "https://www.tab4u.com/lyrics/songs/123_demo.html"


def test_parse_tab4u_search_results_extracts_internal_song_links():
    html = """
    <table>
      <tr>
        <td>
          <a onmouseover="ShowIFD(this, 'song', 8135, true, 'עדן חסון');"
             href="tabs/songs/8135_%D7%A2%D7%93%D7%9F_%D7%97%D7%A1%D7%95%D7%9F_-_%D7%A9%D7%9E%D7%99%D7%A9%D7%94%D7%95_%D7%99%D7%A2%D7%A6%D7%95%D7%A8_%D7%90%D7%95%D7%AA%D7%99.html">
            שמישהו יעצור אותי / עדן חסון
          </a>
        </td>
      </tr>
    </table>
    """

    results = _parse_tab4u_search_results(html)

    assert len(results) == 1
    assert results[0].title.strip() == "שמישהו יעצור אותי / עדן חסון"
    assert results[0].url == (
        "https://www.tab4u.com/lyrics/songs/"
        "8135_%D7%A2%D7%93%D7%9F_%D7%97%D7%A1%D7%95%D7%9F_-_"
        "%D7%A9%D7%9E%D7%99%D7%A9%D7%94%D7%95_%D7%99%D7%A2%D7%A6%D7%95%D7%A8_"
        "%D7%90%D7%95%D7%AA%D7%99.html"
    )


def test_extract_site_specific_lyrics_filters_tab4u_chord_noise():
    page_html = """
    <div id="songContentTPL">
      <table>
        <tr><td class="song">D*: x54x35</td></tr>
        <tr><td class="song">פתיחה:</td></tr>
        <tr><td class="song">את יודעת אין משוגעים כמוני</td></tr>
        <tr><td class="song">אני אסע גם לאילת גם בשלוש בלילה</td></tr>
      </table>
    </div>
    """

    lyrics_html = _extract_site_specific_lyrics(
        "https://www.tab4u.com/lyrics/songs/8135_demo.html",
        page_html,
    )
    cleaned = lyrics_verifier._strip_html_preserving_lines(lyrics_html or "")

    assert "D*:" not in cleaned
    assert "פתיחה" not in cleaned
    assert "את יודעת אין משוגעים כמוני" in cleaned
    assert "אני אסע גם לאילת גם בשלוש בלילה" in cleaned


def test_find_best_lyrics_window_restores_omitted_repeated_chorus():
    lines = [
        "\u05d1\u05d9\u05ea \u05e8\u05d0\u05e9\u05d5\u05df",
        "\u05e4\u05d6\u05de\u05d5\u05df \u05d0\u05d5\u05e8 \u05d2\u05d3\u05d5\u05dc",
        "\u05e4\u05d6\u05de\u05d5\u05df \u05ea\u05d7\u05d6\u05d9\u05e7 \u05d7\u05d6\u05e7",
        "\u05d1\u05d9\u05ea \u05e9\u05e0\u05d9",
        "\u05e4\u05d6\u05de\u05d5\u05df \u05d0\u05d5\u05e8 \u05d2\u05d3\u05d5\u05dc",
        "\u05e4\u05d6\u05de\u05d5\u05df \u05ea\u05d7\u05d6\u05d9\u05e7 \u05d7\u05d6\u05e7",
        "\u05e1\u05d5\u05e3 \u05d9\u05e4\u05d4",
    ]
    segments = []
    cursor = 0.0
    for line in lines:
        words = []
        for index, word in enumerate(line.split()):
            words.append(WordTiming(word, cursor + index * 0.4, cursor + (index + 1) * 0.4))
        segments.append(
            TranscriptSegment(
                words=words,
                text=line,
                start=words[0].start,
                end=words[-1].end,
            )
        )
        cursor = segments[-1].end + 0.2
    draft = TranscriptDraft(segments=segments, provider="test")

    page_html = """
    <html><body>
    בית ראשון<br>
    פזמון אור גדול<br>
    פזמון תחזיק חזק<br>
    בית שני<br>
    סוף יפה
    </body></html>
    """

    score, corrected_lines, correction_count = _find_best_lyrics_window(page_html, draft)

    assert score >= 0.45
    assert correction_count >= 0
    assert corrected_lines == lines


def test_find_best_lyrics_window_ignores_page_boilerplate_and_repairs_only_local_words():
    draft_lines = [
        "\u05d0\u05e0\u05d9 \u05e8\u05d5\u05e6\u05d4 \u05dc\u05dc\u05d7\u05ea \u05d0\u05dc\u05d9\u05da",
        "\u05d4\u05dc\u05d9\u05dc\u05d4 \u05e4\u05ea\u05d5\u05d7",
    ]
    segments = []
    cursor = 0.0
    for line in draft_lines:
        words = []
        for index, word in enumerate(line.split()):
            words.append(WordTiming(word, cursor + index * 0.4, cursor + (index + 1) * 0.4))
        segments.append(
            TranscriptSegment(
                words=words,
                text=line,
                start=words[0].start,
                end=words[-1].end,
            )
        )
        cursor = segments[-1].end + 0.2
    draft = TranscriptDraft(segments=segments, provider="test")

    page_html = """
    <html><body>
    שיתוף בפייסבוק<br>
    כניסה למערכת<br>
    אני רוצה ללכת אליך<br>
    הלילה פתוח<br>
    מילים לשיר ותגובות
    </body></html>
    """

    score, corrected_lines, correction_count = _find_best_lyrics_window(page_html, draft)

    assert score >= 0.45
    assert correction_count == 1
    assert corrected_lines == [
        "\u05d0\u05e0\u05d9 \u05e8\u05d5\u05e6\u05d4 \u05dc\u05dc\u05db\u05ea \u05d0\u05dc\u05d9\u05da",
        "\u05d4\u05dc\u05d9\u05dc\u05d4 \u05e4\u05ea\u05d5\u05d7",
    ]


def test_find_best_source_line_window_ignores_credit_tail_lines():
    draft_lines = [
        "\u05d0\u05e0\u05d9 \u05e8\u05d5\u05e6\u05d4 \u05dc\u05dc\u05db\u05ea \u05d0\u05dc\u05d9\u05da",
        "\u05d4\u05dc\u05d9\u05dc\u05d4 \u05e4\u05ea\u05d5\u05d7",
        "\u05db\u05dc \u05d4\u05dc\u05d1 \u05e9\u05d5\u05dc\u05d7",
    ]
    segments = []
    cursor = 0.0
    for line in draft_lines:
        words = []
        for index, word in enumerate(line.split()):
            words.append(WordTiming(word, cursor + index * 0.4, cursor + (index + 1) * 0.4))
        segments.append(
            TranscriptSegment(
                words=words,
                text=line,
                start=words[0].start,
                end=words[-1].end,
            )
        )
        cursor = segments[-1].end + 0.2
    draft = TranscriptDraft(segments=segments, provider="test")

    candidate_text = "\n".join(
        draft_lines
        + [
            "\u05de\u05d9\u05dc\u05d9\u05dd \u05d5\u05dc\u05d7\u05df \u05d0\u05e8\u05d9\u05e7 \u05db\u05d4\u05df",
            "\u05e2\u05d9\u05d1\u05d5\u05d3 \u05d5\u05d4\u05e4\u05e7\u05d4 \u05d3\u05e0\u05d9 \u05dc\u05d5\u05d9",
            "\u05d8\u05dc \u05dc\u05d4\u05d5\u05e4\u05e2\u05d5\u05ea 0501234567",
        ]
    )

    corrected_lines, score = lyrics_verifier._find_best_source_line_window(candidate_text, draft)

    assert score >= 0.5
    assert corrected_lines == draft_lines


def test_lyrics_verifier_builds_verified_option_from_multiple_sources(monkeypatch):
    def fake_fetch_text(url: str, timeout: int = 12):
        del timeout
        if "duckduckgo" in url:
            return SEARCH_HTML
        return LYRICS_PAGE

    monkeypatch.setattr("karaoke.lyrics_verifier._fetch_text", fake_fetch_text)
    verifier = DuckDuckGoLyricsVerifier()

    result = verifier.verify(TITLE, _draft())

    option_ids = [option["option_id"] for option in result.options]
    assert result.verdict == "matched"
    assert result.applied is True
    assert result.selected_option_id == "verified"
    assert "verified" in option_ids
    assert "draft" in option_ids
    assert "source_shironet" in option_ids
    assert "source_nagnu" in option_ids
    assert result.corrected_lines
    assert CORRECT_LINE in result.corrected_lines[0]


def test_lyrics_verifier_keeps_source_options_even_without_auto_apply(monkeypatch):
    def fake_fetch_text(url: str, timeout: int = 12):
        del timeout
        if "duckduckgo" in url:
            return SINGLE_RESULT_HTML
        return LYRICS_PAGE

    monkeypatch.setattr("karaoke.lyrics_verifier._fetch_text", fake_fetch_text)
    verifier = DuckDuckGoLyricsVerifier()

    result = verifier.verify(TITLE, _draft())

    option_ids = [option["option_id"] for option in result.options]
    assert "verified" in option_ids
    assert "draft" in option_ids
    assert "source_shironet" in option_ids
    verified_option = next(option for option in result.options if option["option_id"] == "verified")
    assert CORRECT_LINE in verified_option["lines"][0]


def test_hybrid_lyrics_verifier_falls_back_to_gemini_only_when_preferred_sources_fail():
    hybrid = HybridLyricsVerifier()
    draft = _draft()

    hybrid.preferred.verify = lambda title, draft: LyricsVerificationResult(  # type: ignore[method-assign]
        provider="preferred",
        verdict="mismatch",
        summary="לא נמצאו מקורות מועדפים.",
        selected_option_id="draft",
        options=[
            {
                "option_id": "draft",
                "label": "draft",
                "lines": [segment.text for segment in draft.segments],
            }
        ],
    )
    hybrid.gemini.verify = lambda title, draft: LyricsVerificationResult(  # type: ignore[method-assign]
        provider="gemini",
        verdict="matched",
        summary="Gemini מצא מילים.",
        matched_sources=["gemini"],
        corrected_lines=[CORRECT_LINE, SECOND_LINE],
        applied=True,
        selected_option_id="verified",
        options=[
            {
                "option_id": "verified",
                "label": "verified",
                "lines": [CORRECT_LINE, SECOND_LINE],
            },
            {
                "option_id": "source_gemini",
                "label": "מקור: Gemini",
                "lines": [CORRECT_LINE, SECOND_LINE],
            },
        ],
    )

    result = hybrid.verify(TITLE, draft)

    assert result.provider == "hybrid_lyrics_verifier"
    assert result.verdict == "matched"
    assert "fallback ל-Gemini" in result.summary
    assert any(option["option_id"] == "source_gemini" for option in result.options)


def test_hybrid_lyrics_verifier_falls_back_to_gemini_only_when_preferred_sources_fail():
    hybrid = HybridLyricsVerifier()
    draft = _draft()

    hybrid.preferred.verify = lambda title, draft: LyricsVerificationResult(  # type: ignore[method-assign]
        provider="preferred",
        verdict="mismatch",
        summary="לא נמצאו מקורות מועדפים.",
        selected_option_id="draft",
        options=[
            {
                "option_id": "draft",
                "label": "draft",
                "lines": [segment.text for segment in draft.segments],
            }
        ],
    )
    hybrid.gemini.verify = lambda title, draft: LyricsVerificationResult(  # type: ignore[method-assign]
        provider="gemini",
        verdict="matched",
        summary="Gemini מצא מילים.",
        matched_sources=["gemini"],
        corrected_lines=[CORRECT_LINE, SECOND_LINE],
        applied=True,
        selected_option_id="verified",
        options=[
            {
                "option_id": "verified",
                "label": "verified",
                "lines": [CORRECT_LINE, SECOND_LINE],
            },
            {
                "option_id": "source_gemini",
                "label": "מקור: Gemini",
                "lines": [CORRECT_LINE, SECOND_LINE],
            },
        ],
    )

    result = hybrid.verify(TITLE, draft)

    assert result.provider == "hybrid_lyrics_verifier"
    assert result.verdict == "matched"
    assert "fallback" in result.summary
    assert any(str(option["option_id"]).startswith("source_") for option in result.options)


def test_score_candidate_url_rejects_same_title_different_artist():
    context = lyrics_verifier._extract_title_context("שיר לוי & נתי לוי - מחזיק לי את היד")

    score = _score_candidate_url(
        "https://www.tab4u.com/lyrics/songs/67458_%D7%9E%D7%95%D7%A8_-_%D7%9E%D7%97%D7%96%D7%99%D7%A7_%D7%9C%D7%99_%D7%90%D7%AA_%D7%94%D7%99%D7%93.html",
        context,
    )

    assert score == 0.0


def test_evaluate_candidate_text_against_draft_keeps_source_authored_lines():
    candidate_text = f"""
    קרדיטים

    {CORRECT_LINE}
    {SECOND_LINE}
    {THIRD_LINE}
    """

    lines, score = _evaluate_candidate_text_against_draft(_draft(), candidate_text)

    assert score >= 0.32
    assert lines == [CORRECT_LINE, SECOND_LINE, THIRD_LINE]


def test_extract_youtube_source_lines_reads_description_without_google_api(monkeypatch):
    class _FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            assert download is False
            assert "youtube.com/watch" in url
            return {
                "title": TITLE,
                "description": (
                    "רן רהב תקשורת ויחסי ציבור\n"
                    "תופים אבי אבידני\n"
                    "קלידים ותכנותים יעקב למאי\n\n"
                    f"{CORRECT_LINE}\n"
                    f"{SECOND_LINE}\n\n"
                    "סטיילינג ישראל רחמני\n"
                    "איפור בר חג׳ג׳\n"
                    "הפצה דיגטלית"
                ),
            }

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL, raising=False)

    lines, score = _extract_youtube_source_lines(
        "https://www.youtube.com/watch?v=abc123",
        TITLE,
        _draft(),
    )

    assert lines == [CORRECT_LINE, SECOND_LINE]
    assert score >= 0.45


def test_multistep_verifier_exposes_merged_search_option_without_draft_repairs():
    verifier = MultiStepLyricsVerifier()
    verifier._search_all_sources = lambda title, draft: {  # type: ignore[method-assign]
        "shironet": [CORRECT_LINE, SECOND_LINE],
        "tab4u": [CORRECT_LINE, SECOND_LINE, THIRD_LINE],
    }
    verifier._gemini_deep_verify = lambda sources, disputes, title: (  # type: ignore[method-assign]
        [CORRECT_LINE, SECOND_LINE],
        0.86,
        [],
    )

    result = verifier.verify(TITLE, _draft())

    merged_option = next(option for option in result.options if option["option_id"] == "search_merged")
    assert merged_option["lines"] == [CORRECT_LINE, SECOND_LINE, THIRD_LINE]


def test_multistep_search_uses_direct_youtube_description_when_web_search_returns_nothing(monkeypatch):
    class _FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=False):
            assert download is False
            return {
                "title": TITLE,
                "description": (
                    "רן רהב תקשורת ויחסי ציבור\n"
                    "תופים אבי אבידני\n"
                    "קלידים ותכנותים יעקב למאי\n\n"
                    f"{CORRECT_LINE}\n"
                    f"{SECOND_LINE}\n\n"
                    "סטיילינג ישראל רחמני\n"
                    "איפור בר חג׳ג׳\n"
                    "הפצה דיגטלית"
                ),
            }

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL, raising=False)
    monkeypatch.setattr("karaoke.lyrics_verifier._search_web_results", lambda query: [])
    monkeypatch.setattr("karaoke.lyrics_verifier._search_known_site_results", lambda title, queries, context: [])

    verifier = MultiStepLyricsVerifier()
    verifier._current_source_url = "https://www.youtube.com/watch?v=abc123"
    verifier._google.search = lambda query, num=10: []  # type: ignore[method-assign]
    verifier._youtube.search = lambda query, max_results=3: []  # type: ignore[method-assign]

    sources = verifier._search_all_sources(TITLE, _draft())

    assert sources == {"youtube": [CORRECT_LINE, SECOND_LINE]}


def test_multistep_search_trims_credit_lines_from_youtube_search_results():
    verifier = MultiStepLyricsVerifier()
    verifier._current_source_url = ""
    verifier._google.search = lambda query, num=10: []  # type: ignore[method-assign]
    verifier._youtube.search = lambda query, max_results=3: [  # type: ignore[method-assign]
        lyrics_verifier.SearchResult(
            title=TITLE,
            snippet=(
                "רן רהב תקשורת ויחסי ציבור\n"
                "תופים אבי אבידני\n"
                f"{CORRECT_LINE}\n"
                f"{SECOND_LINE}\n"
                "סטיילינג ישראל רחמני\n"
                "הפצה דיגטלית"
            ),
            url="https://www.youtube.com/watch?v=abc123",
        )
    ]

    sources = verifier._search_all_sources(TITLE, _draft())

    assert sources == {"youtube": [CORRECT_LINE, SECOND_LINE]}


@pytest.mark.parametrize(
    ("case_id", "description"),
    YOUTUBE_NOISY_DESCRIPTION_CASES,
    ids=[case_id for case_id, _description in YOUTUBE_NOISY_DESCRIPTION_CASES],
)
def test_extract_youtube_source_lines_filters_ten_noisy_description_patterns(
    monkeypatch,
    case_id,
    description,
):
    del case_id
    monkeypatch.setattr(
        yt_dlp,
        "YoutubeDL",
        _fake_youtube_dl_with_description(description),
        raising=False,
    )

    lines, score = _extract_youtube_source_lines(
        "https://www.youtube.com/watch?v=abc123",
        TITLE,
        _draft(),
    )

    assert lines == YOUTUBE_EXPECTED_LINES
    assert score >= 0.32


@pytest.mark.parametrize(
    ("case_id", "description"),
    YOUTUBE_NOISY_DESCRIPTION_CASES,
    ids=[f"search_{case_id}" for case_id, _description in YOUTUBE_NOISY_DESCRIPTION_CASES],
)
def test_multistep_search_filters_ten_noisy_youtube_result_patterns(
    monkeypatch,
    case_id,
    description,
):
    del case_id
    monkeypatch.setattr("karaoke.lyrics_verifier._search_web_results", lambda query: [])
    monkeypatch.setattr("karaoke.lyrics_verifier._search_known_site_results", lambda title, queries, context: [])
    verifier = MultiStepLyricsVerifier()
    verifier._current_source_url = ""
    verifier._google.search = lambda query, num=10: []  # type: ignore[method-assign]
    verifier._youtube.search = lambda query, max_results=3: [  # type: ignore[method-assign]
        lyrics_verifier.SearchResult(
            title=TITLE,
            snippet=description,
            url="https://www.youtube.com/watch?v=abc123",
        )
    ]

    sources = verifier._search_all_sources(TITLE, _draft())

    assert sources == {"youtube": YOUTUBE_EXPECTED_LINES}
