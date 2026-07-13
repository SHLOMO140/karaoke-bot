"""Tests for the lean web_search helpers extracted from lyrics_verifier."""

from karaoke.web_search import (
    SearchResult,
    _build_query_variants,
    _normalize_token,
)


def test_normalize_token_strips_and_lowers_latin():
    assert _normalize_token("  Hello!  ") == "hello"


def test_search_result_is_constructible():
    r = SearchResult(title="t", snippet="s", url="u")
    assert r.url == "u"


def test_build_query_variants_includes_title():
    variants = _build_query_variants("שיר בדיקה", "")
    assert any("שיר בדיקה" in v for v in variants)


def test_web_search_has_no_ml_dependency():
    import sys

    # Importing web_search must not drag in the heavy ML stack.
    import karaoke.web_search  # noqa: F401

    assert "torch" not in sys.modules
    assert "whisperx" not in sys.modules
