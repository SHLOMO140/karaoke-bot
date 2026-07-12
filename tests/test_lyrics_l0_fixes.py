"""Tests for the L0 lyrics bug fixes: nagnu spelling, Google quota error,
opcode-based correction counting, and the de-monkey-patched search method."""

import urllib.error
from unittest.mock import patch

import pytest

from karaoke import lyrics_verifier
from karaoke.google_search import GoogleSearchProvider, GoogleSearchQuotaError
from karaoke.lyrics_verifier import (
    MultiStepLyricsVerifier,
    _estimate_line_corrections,
    _search_result_title_from_url,
)
from karaoke.models import TranscriptDraft, TranscriptSegment, WordTiming


def _draft(lines):
    segments = [
        TranscriptSegment(
            words=[WordTiming(word, float(i), float(i) + 0.5) for word in line.split()],
            text=line,
            start=float(i),
            end=float(i) + 0.9,
        )
        for i, line in enumerate(lines)
    ]
    return TranscriptDraft(segments=segments, provider="test")


def test_nagnu_title_extraction_accepts_both_artist_spellings():
    with_vav = _search_result_title_from_url("https://www.nagnu.co.il/אומנים/עומר_אדם/שני_משוגעים")
    without_vav = _search_result_title_from_url("https://www.nagnu.co.il/אמנים/עומר_אדם/שני_משוגעים")
    assert with_vav == without_vav == "שני משוגעים / עומר אדם"


def test_nagnu_sitemap_filter_accepts_both_artist_spellings(monkeypatch):
    xml = (
        "<urlset>"
        "<loc>https://www.nagnu.co.il/אומנים/זמר/שיר_אחד</loc>"
        "<loc>https://www.nagnu.co.il/אמנים/זמרת/שיר_שני</loc>"
        "<loc>https://www.nagnu.co.il/מדריכים/משהו</loc>"
        "</urlset>"
    )
    monkeypatch.setattr(lyrics_verifier, "_fetch_text", lambda url, timeout=35: xml)
    lyrics_verifier._SITEMAP_URL_CACHE.pop("nagnu:sitemap:lyrics", None)
    try:
        urls = lyrics_verifier._load_nagnu_lyrics_urls()
    finally:
        lyrics_verifier._SITEMAP_URL_CACHE.pop("nagnu:sitemap:lyrics", None)

    assert len(urls) == 2
    assert all("מדריכים" not in url for url in urls)


def test_google_quota_error_raised_for_429():
    provider = GoogleSearchProvider("key", "engine")

    def _raise_429(url, timeout=15):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)

    with patch("urllib.request.urlopen", _raise_429):
        with pytest.raises(GoogleSearchQuotaError):
            provider.search("שיר לדוגמה")


def test_google_other_errors_still_return_empty():
    provider = GoogleSearchProvider("key", "engine")

    with patch("urllib.request.urlopen", side_effect=Exception("API Error")):
        assert provider.search("שיר לדוגמה") == []


def test_quota_error_triggers_web_fallback(monkeypatch):
    verifier = MultiStepLyricsVerifier()

    class _QuotaGoogle:
        def search(self, query, num=10):
            raise GoogleSearchQuotaError("429")

    fallback_queries = []

    def _fake_web_search(query):
        fallback_queries.append(query)
        return []

    monkeypatch.setattr(verifier, "_google", _QuotaGoogle())
    monkeypatch.setattr(lyrics_verifier, "_search_web_results", _fake_web_search)
    monkeypatch.setattr(lyrics_verifier, "_search_known_site_results", lambda *args: [])
    monkeypatch.setattr(
        lyrics_verifier, "_extract_youtube_source_lines", lambda url, title, draft: ([], 0.0)
    )

    class _NoYouTube:
        def search(self, query, max_results=3):
            return []

    monkeypatch.setattr(verifier, "_youtube", _NoYouTube())

    sources = verifier._search_all_sources("אמן - שיר", _draft(["שלום עולם"]))

    assert sources == {}
    assert fallback_queries, "quota error must fast-path into the web fallback"


def test_estimate_line_corrections_handles_line_splits():
    draft = _draft(["שורה אחת ארוכה מאוד", "שורה שנייה", "שורה שלישית", "שורה רביעית"])
    corrected = [
        "שורה אחת",
        "ארוכה מאוד",
        "שורה שנייה",
        "שורה שלישית",
        "שורה רביעית",
    ]
    # One line split into two: the old zip-based count reported ~5.
    assert _estimate_line_corrections(draft, corrected) <= 2


def test_search_all_sources_is_a_real_method():
    assert "_search_all_sources" in MultiStepLyricsVerifier.__dict__
    assert not hasattr(lyrics_verifier, "_multistep_search_all_sources_override")


def test_candidate_lines_preserve_non_hebrew_words():
    page = "<div>אני שר איתך yeah yeah</div><div>עוד שורה בעברית כאן</div>"
    entries = lyrics_verifier._extract_candidate_lyrics_line_entries(page)

    assert entries, "expected candidate lines from the fixture page"
    assert entries[0][0] == "אני שר איתך yeah yeah"
    assert "yeah" in entries[0][1]


def test_candidate_lines_filter_chord_label_tokens():
    page = "<div>אני שר איתך Am F#m Dm7</div>"
    entries = lyrics_verifier._extract_candidate_lyrics_line_entries(page)

    assert entries
    assert entries[0][0] == "אני שר איתך"


def test_mixed_language_line_with_long_english_word_is_not_noise():
    assert not lyrics_verifier._looks_like_noise_line("אני חושב עלייך beautiful girl")


def test_fetch_text_caches_lyric_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(lyrics_verifier, "HTTP_CACHE_DIR", tmp_path)
    calls = []

    def fake_fetch(url, timeout=12):
        calls.append(url)
        return "<html>שורה ראשונה בעברית כאן</html>"

    monkeypatch.setattr(lyrics_verifier, "_fetch_text_uncached", fake_fetch)

    url = "https://www.tab4u.com/lyrics/songs/1234_song.html"
    first = lyrics_verifier._fetch_text(url)
    second = lyrics_verifier._fetch_text(url)

    assert first == second
    assert len(calls) == 1, "second fetch must be served from the disk cache"


def test_fetch_text_does_not_cache_search_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(lyrics_verifier, "HTTP_CACHE_DIR", tmp_path)
    calls = []

    def fake_fetch(url, timeout=12):
        calls.append(url)
        return "results"

    monkeypatch.setattr(lyrics_verifier, "_fetch_text_uncached", fake_fetch)

    url = "https://html.duckduckgo.com/html/?q=song"
    lyrics_verifier._fetch_text(url)
    lyrics_verifier._fetch_text(url)

    assert len(calls) == 2


def test_fetch_text_retries_transient_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(lyrics_verifier, "HTTP_CACHE_DIR", tmp_path)
    monkeypatch.setattr(lyrics_verifier.time, "sleep", lambda seconds: None)
    attempts = []

    def flaky_fetch(url, timeout=12):
        attempts.append(url)
        if len(attempts) == 1:
            raise TimeoutError("timed out")
        return "<html>תוכן תקין לגמרי כאן</html>"

    monkeypatch.setattr(lyrics_verifier, "_fetch_text_uncached", flaky_fetch)

    text = lyrics_verifier._fetch_text("https://www.nagnu.co.il/אמנים/זמר/שיר")
    assert "תוכן תקין" in text
    assert len(attempts) == 2


def test_url_404_fallbacks_reverse_tab4u_rewrite():
    url = "https://www.tab4u.com/lyrics/songs/99_hit.html"
    assert lyrics_verifier._url_404_fallbacks(url) == [
        "https://www.tab4u.com/tabs/songs/99_hit.html"
    ]
    assert lyrics_verifier._url_404_fallbacks("https://www.shironet.mako.co.il/x") == []


def test_missing_confidence_header_defaults_low():
    lyrics, confidence, _uncertain = MultiStepLyricsVerifier._parse_gemini_response(
        "LYRICS:\nשורה ראשונה\nשורה שנייה"
    )
    assert lyrics == ["שורה ראשונה", "שורה שנייה"]
    assert confidence == 0.5


def test_effective_confidence_capped_when_llm_disagrees_with_sources():
    sources = {"tab4u.com": ["שורה אמיתית מהאתר", "עוד שורה אמיתית"]}
    invented = ["מילים שהומצאו לגמרי", "בלי קשר למקורות"]

    effective = lyrics_verifier._effective_llm_confidence(
        0.95, invented, sources=sources, draft_text="שורה אמיתית מהאתר"
    )

    assert effective < 0.72, "self-reported 0.95 must not survive zero source agreement"


def test_effective_confidence_kept_when_llm_matches_sources():
    lines = ["שורה אמיתית מהאתר", "עוד שורה אמיתית"]
    sources = {"tab4u.com": lines}

    effective = lyrics_verifier._effective_llm_confidence(
        0.9, lines, sources=sources, draft_text="שורה אמיתית מהאתר"
    )

    assert effective == pytest.approx(0.9)


def test_knowledge_confidence_capped_when_far_from_draft():
    invented = ["מילים אחרות", "שלא הושרו בכלל"]
    effective = lyrics_verifier._effective_llm_confidence(
        0.9, invented, sources=None, draft_text="שרתי משהו שונה לחלוטין הערב"
    )
    assert effective <= 0.75


def test_gemini_call_retries_on_timeout(monkeypatch):
    attempts = []

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            import json as _json

            return _json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "בסדר"}]}}]}
            ).encode("utf-8")

    def flaky_urlopen(request, timeout=0):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise TimeoutError("timed out")
        return _FakeResponse()

    monkeypatch.setattr(lyrics_verifier.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(lyrics_verifier.urllib.request, "urlopen", flaky_urlopen)

    text = lyrics_verifier._call_gemini("prompt", "key", "model")

    assert text == "בסדר"
    assert len(attempts) == 2


def test_latin_word_survives_interpolation_in_aligner():
    """Cross-track guard for L2: a Latin word the Hebrew wav2vec2 model cannot
    align must fall back to interpolation without crashing."""
    from karaoke import aligner

    approved = [
        TranscriptSegment(
            words=[
                WordTiming("שלום", 0.0, 0.5, confidence=0.9, source="review_hint"),
                WordTiming("baby", 0.5, 0.9, confidence=0.0, source="review_hint"),
                WordTiming("עולם", 0.9, 1.4, confidence=0.9, source="review_hint"),
            ],
            text="שלום baby עולם",
            start=0.0,
            end=1.4,
        )
    ]
    draft = [
        TranscriptSegment(
            words=[
                WordTiming("שלום", 0.0, 0.5, confidence=0.9),
                WordTiming("עולם", 0.9, 1.4, confidence=0.9),
            ],
            text="שלום עולם",
            start=0.0,
            end=1.4,
        )
    ]

    result = aligner.SequenceHebrewAligner().align("missing.wav", approved, draft)

    words = result.segments[0].words
    assert [word.word for word in words] == ["שלום", "baby", "עולם"]
    assert all(word.end > word.start for word in words)
