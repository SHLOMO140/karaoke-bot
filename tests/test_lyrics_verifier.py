from karaoke.lyrics_verifier import DuckDuckGoLyricsVerifier, HybridLyricsVerifier, _build_query_variants, _parse_duckduckgo_results
from karaoke.models import LyricsVerificationResult, TranscriptDraft, TranscriptSegment, WordTiming


CORRECT_LINE = "\u05db\u05db\u05d4 \u05e6\u05e8\u05d9\u05da \u05dc\u05d4\u05d9\u05d5\u05ea \u05db\u05ea\u05d5\u05d1 \u05d1\u05e2\u05d1\u05e8\u05d9\u05ea"
SECOND_LINE = "\u05d5\u05e2\u05d5\u05d3 \u05e9\u05d5\u05e8\u05d4 \u05d4\u05d2\u05d9\u05d5\u05e0\u05d9\u05ea"
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

    assert any("\u05e0\u05d5\u05e2\u05dd \u05d1\u05ea\u05df \u05d4\u05d9\u05d5\u05dd \u05de\u05d9\u05dc\u05d9\u05dd" in query for query in queries)
    assert any("site:lyricstranslate.com" in query for query in queries)
    assert any("site:tab4u.com" in query for query in queries)


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
