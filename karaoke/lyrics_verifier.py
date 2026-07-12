"""Multi-source lyrics verification before human review."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError
from dataclasses import dataclass
from difflib import SequenceMatcher

from .config import CONSENSUS_MODE, HTTP_CACHE_DIR, HTTP_CACHE_TTL_SECONDS, LLM_TIMEOUT_SECONDS
from .consensus import normalize_lyrics_line
from .google_search import GoogleSearchQuotaError

from .models import (
    ConsensusResult,
    LyricsVerificationResult,
    TranscriptDraft,
    TranscriptSegment,
    VerificationVerdict,
    WordTiming,
)

logger = logging.getLogger(__name__)

DUCKDUCKGO_HTML_SEARCH = "https://html.duckduckgo.com/html/?q={query}"
DUCKDUCKGO_HTML_FALLBACK = "https://duckduckgo.com/html/?q={query}"
BING_HTML_SEARCH = "https://www.bing.com/search?q={query}&setlang=he"
SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.7",
}
# nagnu.co.il uses both spellings of "artists" in URL paths across the site.
NAGNU_ARTIST_SEGMENTS = ("אמנים", "אומנים")

# Latin tokens that are really chord labels (leaking from chords sites), e.g.
# Am, F#m, Bb7, Gsus4, C/E — filtered out of mixed-language lyric lines.
_CHORD_LABEL_TOKEN_RE = re.compile(
    r"^[A-G][#b]?(?:maj7|m7b5|m7|m9|m|dim7?|aug|sus[24]|add\d+|6|7|9|11|13)?(?:/[A-G][#b]?)?$"
)

KNOWN_LYRICS_DOMAINS = (
    "shironet.mako.co.il",
    "nagnu.co.il",
    "tab4u.com",
    "lyricstranslate.com",
    "nli.org.il",
    "shirrim.com",
    "genius.com",
    "nomorelyrics.net",
    "baneshama.co.il",
    "nagina.co.il",
)
SITE_QUERY_DOMAINS = (
    "shironet.mako.co.il",
    "nagnu.co.il",
    "tab4u.com",
    "lyricstranslate.com",
    "nli.org.il",
    "genius.com",
    "baneshama.co.il",
    "nagina.co.il",
)
PREFERRED_LYRICS_DOMAINS = (
    "shironet.mako.co.il",
    "tab4u.com",
    "nagnu.co.il",
)
DOMAIN_PRIORITY = {
    "shironet.mako.co.il": 0,
    "tab4u.com": 1,
    "nagnu.co.il": 2,
    "lyricstranslate.com": 3,
    "shirrim.com": 4,
    "nli.org.il": 5,
    "genius.com": 6,
    "nomorelyrics.net": 7,
}
SOURCE_DISPLAY_NAMES = {
    "shironet.mako.co.il": "שירונט",
    "tab4u.com": "Tab4U",
    "nagnu.co.il": "נגינה",
    "gemini": "Gemini",
    "grok": "Grok",
}
STOPWORDS = {
    "של",
    "על",
    "עם",
    "את",
    "זה",
    "זאת",
    "אני",
    "אתה",
    "אתם",
    "אנחנו",
    "והוא",
    "והיא",
    "אם",
    "כי",
    "לא",
    "כן",
    "עוד",
    "הוא",
    "היא",
    "הם",
    "הן",
    "לי",
    "לך",
    "לו",
    "לה",
    "מה",
    "מי",
    "כל",
    "מן",
}

TITLE_RE = re.compile(
    r'<a(?=[^>]*class=["\'][^"\']*result__a[^"\']*["\'])(?=[^>]*href=["\'](?P<href>[^"\']+)["\'])[^>]*>(?P<title>.*?)</a>',
    re.S | re.I,
)
SNIPPET_RE = re.compile(
    r'<(?:a|div|span)(?=[^>]*class=["\'][^"\']*result__snippet[^"\']*["\'])[^>]*>(?P<snippet>.*?)</(?:a|div|span)>',
    re.S | re.I,
)
SEARCH_RESULT_CACHE: dict[str, list["SearchResult"]] = {}
SUPPORTED_LYRICS_LLM_PROVIDERS = {"gemini", "grok"}
HEBREW_LYRICS_QUERY = "\u05de\u05d9\u05dc\u05d9\u05dd"
HEBREW_LYRICS_FOR_SONG_QUERY = "\u05de\u05d9\u05dc\u05d9\u05dd \u05dc\u05e9\u05d9\u05e8"


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


@dataclass
class LyricsSourceCandidate:
    title: str
    url: str
    domain: str
    score: float
    excerpt: str
    lines: list[str]
    correction_count: int


def _normalize_lyrics_llm_provider(provider: str) -> str:
    normalized = (provider or "gemini").strip().lower()
    if normalized in {"xai", "grok"}:
        return "grok"
    if normalized not in SUPPORTED_LYRICS_LLM_PROVIDERS:
        logger.warning("Unsupported lyrics LLM provider '%s', falling back to Gemini", provider)
        return "gemini"
    return normalized


def _lyrics_llm_display_name(provider: str) -> str:
    normalized = _normalize_lyrics_llm_provider(provider)
    return SOURCE_DISPLAY_NAMES.get(normalized, normalized.title())


def _lyrics_llm_source_option_id(provider: str) -> str:
    return f"source_{_normalize_lyrics_llm_provider(provider)}"


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_html_preserving_lines(text: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|tr|h1|h2|h3|h4|section|article)>", "\n", text)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_token(token: str) -> str:
    token = re.sub(r"[\u0591-\u05C7]", "", token)
    return re.sub(r"[^\w\u0590-\u05FF]+", "", token).lower()


def _tokenize(text: str) -> list[str]:
    tokens = [_normalize_token(token) for token in re.findall(r"[\w\u0590-\u05FF']+", text)]
    return [token for token in tokens if len(token) >= 2]


def _tokenize_words(text: str) -> list[str]:
    return [token for token in re.findall(r"[\w\u0590-\u05FF']+", text) if len(_normalize_token(token)) >= 2]


def _line_signature(text: str) -> str:
    normalized_tokens = [_normalize_token(token) for token in _tokenize_words(text)]
    return " ".join(token for token in normalized_tokens if token)


def _top_keywords(text: str, limit: int = 10) -> list[str]:
    counts: dict[str, int] = {}
    for token in _tokenize(text):
        if token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [token for token, _count in ranked[:limit]]


def _draft_search_snippets(draft_text: str, limit: int = 3) -> list[str]:
    lines = [line.strip() for line in draft_text.splitlines() if line.strip()]
    if not lines:
        return []

    counts: dict[str, int] = {}
    snippets: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    token_lengths: dict[str, int] = {}

    for index, line in enumerate(lines):
        tokens = _tokenize_words(line)
        if len(tokens) < 3:
            continue
        signature = _line_signature(line)
        if not signature:
            continue
        counts[signature] = counts.get(signature, 0) + 1
        snippets.setdefault(signature, " ".join(tokens[: min(6, len(tokens))]))
        first_seen.setdefault(signature, index)
        token_lengths[signature] = max(token_lengths.get(signature, 0), len(tokens))

    ranked = sorted(
        counts,
        key=lambda signature: (-counts[signature], -token_lengths[signature], first_seen[signature]),
    )
    return [snippets[signature] for signature in ranked[:limit]]


def _normalize_title_text(text: str) -> str:
    text = re.sub(r"[\u0591-\u05C7]", "", text)
    text = re.sub(r"\[[^\]]+\]|\([^)]+\)", " ", text)
    text = re.sub(
        r"(?i)\b(prod(?:\.|uced)?\s+by|official\s+(?:video|audio|lyrics?)|lyrics?|audio|video)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -|")


def _extract_title_context(title: str) -> dict[str, str]:
    clean_title = _normalize_title_text(title)
    pipe_parts = [part.strip() for part in re.split(r"\s*\|\s*", clean_title) if part.strip()]
    primary = next((part for part in reversed(pipe_parts) if re.search(r"[\u0590-\u05FF]", part)), clean_title)

    artist = ""
    song = ""
    dash_split = re.split(r"\s*[\-\u2013\u2014:]\s*", primary, maxsplit=1)
    if len(dash_split) == 2:
        artist, song = (part.strip() for part in dash_split)
    else:
        song = primary.strip()

    latin_artist = next(
        (
            part
            for part in pipe_parts
            if re.search(r"[A-Za-z]", part) and len(_tokenize_words(part)) <= 6
        ),
        "",
    )
    hebrew_artist = artist if re.search(r"[\u0590-\u05FF]", artist) else ""
    song = song or clean_title
    return {
        "clean_title": clean_title,
        "primary": primary,
        "hebrew_artist": hebrew_artist,
        "latin_artist": latin_artist,
        "song": song,
    }


def _align_source_to_segments(
    source_tokens: list[str],
    draft: TranscriptDraft,
) -> tuple[list[str], int]:
    """Align source_tokens to draft segments using global SequenceMatcher.

    Each draft token position holds a *slot* (a list of one or more corrected
    words).  This correctly handles N:M replacements where the source has more
    words than the draft for a given range — all source words are kept, not
    just a proportionally sampled subset.

    Opcode handling:
    - equal   → slot = [source_token]   (same word, use source spelling)
    - replace → distribute all source tokens across the draft positions
                proportionally; extra source words expand the slot size
    - delete  → slot = [draft_word]     (no source match, keep draft)
    - insert  → source tokens are appended to the preceding slot so they
                appear in the correct position within the segment

    Returns (corrected_lines_per_segment, correction_count).
    """
    flat_draft_words: list[str] = []
    word_seg_idx: list[int] = []
    valid_segs: list[tuple[int, str]] = []
    for seg_idx, seg in enumerate(draft.segments):
        text = seg.text.strip()
        if not text:
            continue
        valid_segs.append((seg_idx, text))
        for w in _tokenize_words(text):
            flat_draft_words.append(w)
            word_seg_idx.append(seg_idx)

    if not flat_draft_words or not source_tokens:
        return [text for _, text in valid_segs], 0

    flat_draft_norm = [_normalize_token(w) for w in flat_draft_words]
    norm_source = [_normalize_token(t) for t in source_tokens]

    # slot[i] = list of corrected words for draft position i (may hold multiple
    # words when the source has more tokens than the draft in a replace block)
    slot: list[list[str]] = [[] for _ in range(len(flat_draft_words))]

    for tag, a0, a1, b0, b1 in SequenceMatcher(
        None, flat_draft_norm, norm_source, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            for k in range(a1 - a0):
                slot[a0 + k] = [source_tokens[b0 + k]]

        elif tag == "replace":
            src = source_tokens[b0:b1]
            n_d, n_s = a1 - a0, len(src)
            if n_d == 0 or n_s == 0:
                pass
            elif n_s <= n_d:
                # Fewer (or equal) source words: one per draft slot, proportional pick
                for di in range(n_d):
                    si = round(di * (n_s - 1) / max(1, n_d - 1)) if n_d > 1 else 0
                    slot[a0 + di] = [src[si]]
            else:
                # More source words than draft positions: distribute proportionally
                # so every source word ends up in exactly one slot.
                for di in range(n_d):
                    si_start = round(di * n_s / n_d)
                    si_end = round((di + 1) * n_s / n_d)
                    slot[a0 + di] = list(src[si_start:si_end])

        elif tag == "delete":
            # No source counterpart → keep the original draft word
            for di in range(a1 - a0):
                slot[a0 + di] = [flat_draft_words[a0 + di]]

        elif tag == "insert":
            # Source tokens with no draft position: attach to the preceding slot
            # so they appear in the right position when we reconstruct the line.
            src = source_tokens[b0:b1]
            if not src:
                continue
            if a0 > 0:
                slot[a0 - 1].extend(src)
            elif flat_draft_words:
                slot[0] = list(src) + slot[0]

    # Any slot still empty corresponds to a draft word with no opcode coverage —
    # fall back to the original draft word.
    for i in range(len(slot)):
        if not slot[i]:
            slot[i] = [flat_draft_words[i]]

    # Count draft positions where the source introduced a change
    correction_count = sum(
        1
        for i, (draft_norm, words) in enumerate(zip(flat_draft_norm, slot))
        if any(_normalize_token(w) != draft_norm for w in words)
    )

    # Collect corrected words per segment (flattening each slot in draft order)
    seg_buckets: dict[int, list[str]] = {}
    for words, seg_idx in zip(slot, word_seg_idx):
        seg_buckets.setdefault(seg_idx, []).extend(words)

    result: list[str] = []
    for seg_idx, orig_text in valid_segs:
        bucket = seg_buckets.get(seg_idx, [])
        result.append(" ".join(bucket) if bucket else orig_text)

    return result, correction_count


_BOT_BLOCK_SIGNATURES = (
    "Radware Block Page",
    "cf-browser-verification",
    "Please enable JavaScript and cookies",
    "hCaptcha",
    "ShieldSquare",
    "perfdrive.com",
)


def _is_bot_blocked(html_text: str) -> bool:
    """Return True when the page is a bot-protection/CAPTCHA block page."""
    head = html_text[:4000]
    return any(sig in head for sig in _BOT_BLOCK_SIGNATURES)


def _encode_url(url: str) -> str:
    """Re-encode non-ASCII characters in URL path/query so urllib can handle them."""
    parsed = urllib.parse.urlparse(url)
    encoded = parsed._replace(
        path=urllib.parse.quote(parsed.path, safe="/%@:!$&'()*+,;="),
        query=urllib.parse.quote(parsed.query, safe="=&%+@:!$'()*,;/?"),
    )
    return urllib.parse.urlunparse(encoded)


def _fetch_text_uncached(url: str, timeout: int = 12) -> str:
    encoded_url = _encode_url(url)
    request = urllib.request.Request(encoded_url, headers=SEARCH_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


def _http_cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return Path(HTTP_CACHE_DIR) / f"{digest}.html"


def _is_cacheable_lyrics_url(url: str) -> bool:
    domain = _domain_from_url(url)
    return any(domain.endswith(candidate) for candidate in KNOWN_LYRICS_DOMAINS)


def _fetch_text(url: str, timeout: int = 12) -> str:
    """Fetch a page with a one-shot retry and an on-disk cache for lyric pages.

    Search-engine pages are never cached (results must stay fresh); pages on
    known lyrics domains are cached for HTTP_CACHE_TTL_SECONDS so re-verifying
    a song does not re-pay timeouts, 404s and bot blocks.
    """
    cache_path = _http_cache_path(url) if _is_cacheable_lyrics_url(url) else None
    if cache_path is not None:
        try:
            if cache_path.exists() and time.time() - cache_path.stat().st_mtime < HTTP_CACHE_TTL_SECONDS:
                return cache_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass

    text = ""
    for attempt in range(2):
        try:
            text = _fetch_text_uncached(url, timeout=timeout)
            break
        except HTTPError as exc:
            # 404 is definitive; retrying will not help.
            if exc.code == 404 or attempt == 1:
                raise
            time.sleep(1.0)
        except Exception:
            if attempt == 1:
                raise
            time.sleep(1.0)

    if cache_path is not None and text and not _is_bot_blocked(text):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return text


def _decode_redirect(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    target = query.get("uddg", [url])[0]
    return urllib.parse.unquote(target)


def _parse_duckduckgo_results(html_text: str) -> list[SearchResult]:
    results = []
    title_matches = list(TITLE_RE.finditer(html_text))
    for index, title_match in enumerate(title_matches):
        block_end = title_matches[index + 1].start() if index + 1 < len(title_matches) else len(html_text)
        block = html_text[title_match.start():block_end]
        snippet_match = SNIPPET_RE.search(block)
        title = _strip_html(title_match.group("title"))
        snippet = _strip_html(snippet_match.group("snippet")) if snippet_match else ""
        url = _decode_redirect(title_match.group("href"))
        results.append(SearchResult(title=title, snippet=snippet, url=url))
    return results


def _domain_from_url(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")


def _domain_label(url: str) -> str:
    return _domain_from_url(url) or "web"


def _domain_display_name(url: str) -> str:
    domain = _domain_label(url)
    return SOURCE_DISPLAY_NAMES.get(domain, domain)


def _domain_option_key(url: str) -> str:
    domain = _domain_label(url)
    if domain.endswith("shironet.mako.co.il"):
        return "shironet"
    if domain.endswith("tab4u.com"):
        return "tab4u"
    if domain.endswith("nagnu.co.il"):
        return "nagnu"
    if domain == "gemini":
        return "gemini"
    return re.sub(r"[^a-z0-9]+", "_", domain).strip("_") or "web"


def _domain_is_allowed(url: str, allowed_domains: tuple[str, ...] | None) -> bool:
    if not allowed_domains:
        return True
    domain = _domain_from_url(url)
    return any(domain.endswith(candidate) for candidate in allowed_domains)


def _domain_priority(url: str) -> int:
    domain = _domain_from_url(url)
    for candidate, priority in DOMAIN_PRIORITY.items():
        if domain.endswith(candidate):
            return priority
    return 99


def _text_overlap_score(reference_tokens: set[str], candidate_text: str) -> float:
    candidate_tokens = {token for token in _tokenize(candidate_text) if token not in STOPWORDS}
    if not reference_tokens or not candidate_tokens:
        return 0.0
    intersection = reference_tokens & candidate_tokens
    return len(intersection) / max(1, min(len(reference_tokens), len(candidate_tokens)))


def _looks_like_lyrics_site(url: str) -> bool:
    domain = _domain_from_url(url)
    return any(domain.endswith(item) for item in KNOWN_LYRICS_DOMAINS)


def _build_draft_only_result(
    provider: str,
    draft: TranscriptDraft,
    warnings: list[str],
    summary: str,
    verdict: str = "not_run",
    search_query: str = "",
    llm_provider: str = "",
) -> LyricsVerificationResult:
    draft_lines = _draft_lines(draft)
    return LyricsVerificationResult(
        provider=provider,
        llm_provider=llm_provider,
        verdict=verdict,
        confidence=0.0,
        search_query=search_query,
        summary=summary,
        matched_sources=[],
        web_excerpt="",
        local_warnings=warnings,
        corrected_lines=[],
        correction_count=0,
        applied=False,
        selected_option_id="draft",
        options=[
            {
                "option_id": "draft",
                "label": "תמלול מקורי",
                "lines": draft_lines,
                "source_url": "",
                "confidence": 1.0,
                "source_count": 0,
            }
        ],
    )


def _build_verification_options(
    draft: TranscriptDraft,
    *,
    verified_lines: list[str] | None = None,
    verified_label: str = "גרסה מאומתת",
    verified_confidence: float = 0.0,
    verified_source_count: int = 0,
    verified_source_url: str = "",
    verified_option_id: str = "verified",
    verified_source_option_id: str | None = None,
    verified_source_label: str | None = None,
    supporting_sources: list[str] | None = None,
    extra_source_options: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    draft_lines = _draft_lines(draft)
    options: list[dict[str, object]] = [
        {
            "option_id": "draft",
            "label": "תמלול מקורי",
            "lines": draft_lines,
            "source_url": "",
            "confidence": 1.0,
            "source_count": 0,
        }
    ]

    normalized_verified_lines = [str(line).strip() for line in verified_lines or [] if str(line).strip()]
    if normalized_verified_lines:
        verified_option = {
            "option_id": verified_option_id,
            "label": verified_label,
            "lines": normalized_verified_lines,
            "source_url": verified_source_url,
            "confidence": round(verified_confidence, 3),
            "source_count": verified_source_count,
        }
        if supporting_sources:
            verified_option["supporting_sources"] = list(dict.fromkeys(supporting_sources))
        options.insert(0, verified_option)

        if verified_source_option_id and verified_source_label:
            options.append(
                {
                    "option_id": verified_source_option_id,
                    "label": verified_source_label,
                    "lines": normalized_verified_lines,
                    "source_url": verified_source_url,
                    "confidence": round(verified_confidence, 3),
                    "source_count": max(1, verified_source_count),
                }
            )

    for option in extra_source_options or []:
        if option.get("option_id") and option.get("lines"):
            options.append(option)

    return options


def _line_sets_differ(left: list[str], right: list[str]) -> bool:
    normalized_left = [line.strip() for line in left if line.strip()]
    normalized_right = [line.strip() for line in right if line.strip()]
    return normalized_left != normalized_right


def _token_stream_overlap(left_lines: list[str], right_lines: list[str]) -> float:
    """Ordered token-overlap ratio between two blocks of lines (0..1)."""
    left = [token for line in left_lines for token in normalize_lyrics_line(line).split() if token]
    right = [token for line in right_lines for token in normalize_lyrics_line(line).split() if token]
    if not left or not right:
        return 0.0
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / max(1, min(len(left), len(right)))


def _consensus_matches_draft(lyrics_lines: list[str], draft_text: str, *, threshold: float = 0.35) -> bool:
    """Sanity check that agreed source lyrics resemble the sung transcript.

    Sources can confidently agree on the WRONG song (similar titles, covers);
    if the token overlap with the Whisper draft is too low, consensus should
    demote to LLM arbitration instead of auto-verifying.
    """
    return _token_stream_overlap(draft_text.splitlines(), lyrics_lines) >= threshold


def _effective_llm_confidence(
    llm_confidence: float,
    lyrics_lines: list[str],
    *,
    sources: dict[str, list[str]] | None,
    draft_text: str,
) -> float:
    """Cap the LLM's self-reported confidence with objective agreement.

    A model may print CONFIDENCE: 0.95 while inventing lyrics; auto-apply must
    depend on how well the answer matches the search sources (or, without
    sources, at least resemble the sung transcript).
    """
    llm_confidence = max(0.0, min(1.0, float(llm_confidence or 0.0)))
    if sources:
        agreement = max(
            (
                _token_stream_overlap(lyrics_lines, source_lines)
                for source_lines in sources.values()
                if source_lines
            ),
            default=0.0,
        )
        return min(llm_confidence, 0.55 + 0.45 * agreement)

    similarity = _token_stream_overlap(lyrics_lines, draft_text.splitlines())
    if 0.35 <= similarity <= 0.95:
        return llm_confidence
    return min(llm_confidence, 0.75)


def _estimate_line_corrections(draft: TranscriptDraft, corrected_lines: list[str]) -> int:
    """Count corrected lines via opcode alignment.

    Naive zip counting inflates the number whenever segmentation changes
    (splitting one line used to count every following line as "corrected").
    """
    draft_lines = _draft_lines(draft)
    corrected = [str(line).strip() for line in corrected_lines if str(line).strip()]
    matcher = SequenceMatcher(None, draft_lines, corrected, autojunk=False)
    corrections = 0
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        corrections += max(a_end - a_start, b_end - b_start)
    return corrections


def _should_auto_apply_verified_lines(
    draft: TranscriptDraft,
    corrected_lines: list[str],
    confidence: float,
    *,
    threshold: float,
) -> bool:
    draft_lines = _draft_lines(draft)
    normalized_corrected = [str(line).strip() for line in corrected_lines if str(line).strip()]
    if not normalized_corrected:
        return False
    if not _line_sets_differ(draft_lines, normalized_corrected):
        return False
    return confidence >= threshold


def _looks_like_lyrics_result(result: SearchResult) -> bool:
    if _looks_like_lyrics_site(result.url):
        return True
    text = _normalize_title_text(f"{result.title} {result.snippet}").lower()
    return any(keyword in text for keyword in ("lyrics", "lyric", "מילים"))


def _matches_title_context(result: SearchResult, context: dict[str, str]) -> bool:
    haystack_tokens = set(_tokenize(f"{result.title} {result.snippet}"))
    artist_tokens = set(_top_keywords(context["hebrew_artist"], limit=4))
    artist_tokens.update(_top_keywords(context["latin_artist"], limit=4))
    song_tokens = set(_top_keywords(context["song"], limit=4))

    artist_hit = not artist_tokens or bool(haystack_tokens & artist_tokens)
    song_hit = not song_tokens or bool(haystack_tokens & song_tokens)
    if artist_tokens and song_tokens:
        return artist_hit and song_hit
    return artist_hit or song_hit


def _local_warnings(draft_text: str) -> list[str]:
    tokens = _tokenize(draft_text)
    if not tokens:
        return ["לא התקבל טקסט לבדיקה."]

    warnings = []
    hebrew_tokens = [token for token in tokens if re.search(r"[\u0590-\u05FF]", token)]
    if len(hebrew_tokens) / max(1, len(tokens)) < 0.75:
        warnings.append("יש יחס גבוה של מילים לא-עבריות בטיוטה.")

    short_tokens = [token for token in hebrew_tokens if len(token) == 2]
    if len(short_tokens) / max(1, len(hebrew_tokens)) > 0.45:
        warnings.append("יש הרבה מילים קצרות מאוד, ייתכן שחלק מהפלט לא יציב.")

    repeated = re.findall(r"(\b[\u0590-\u05FF]{2,}\b)(?:\s+\1){2,}", draft_text)
    if repeated:
        warnings.append("יש חזרות חשודות בטקסט שיכולות להעיד על זיהוי לא מדויק.")

    return warnings


def _draft_lines(draft: TranscriptDraft) -> list[str]:
    lines = [segment.text.strip() for segment in draft.segments if segment.text.strip()]
    if lines:
        return lines
    return [draft.text.strip()] if draft.text.strip() else []


def _count_token_corrections(reference_lines: list[str], candidate_lines: list[str]) -> int:
    reference_tokens = [_normalize_token(token) for token in _tokenize_words("\n".join(reference_lines))]
    candidate_tokens = [_normalize_token(token) for token in _tokenize_words("\n".join(candidate_lines))]
    matcher = SequenceMatcher(None, reference_tokens, candidate_tokens, autojunk=False)
    return sum(max(a2 - a1, b2 - b1, 1) for tag, a1, a2, b1, b2 in matcher.get_opcodes() if tag != "equal")


def _merge_source_candidates(
    draft: TranscriptDraft,
    candidates: list[LyricsSourceCandidate],
) -> tuple[list[str], int, list[str]]:
    draft_lines = _draft_lines(draft)
    if not draft_lines or not candidates:
        return [], 0, []

    merged_lines: list[str] = []
    supporting_sources: list[str] = []
    for line_index, draft_line in enumerate(draft_lines):
        line_candidates: list[tuple[LyricsSourceCandidate, str]] = []
        for candidate in candidates:
            if line_index < len(candidate.lines):
                line_text = candidate.lines[line_index].strip()
                if line_text:
                    line_candidates.append((candidate, line_text))

        if not line_candidates:
            merged_lines.append(draft_line)
            continue

        clusters: list[dict[str, object]] = []
        for candidate, line_text in line_candidates:
            placed = False
            for cluster in clusters:
                representative = str(cluster["members"][0][1])
                if _line_similarity(representative, line_text) >= 0.72:
                    cluster["members"].append((candidate, line_text))
                    cluster["weight"] = float(cluster["weight"]) + candidate.score
                    placed = True
                    break
            if not placed:
                clusters.append({"members": [(candidate, line_text)], "weight": candidate.score})

        clusters.sort(key=lambda item: (float(item["weight"]), len(item["members"])), reverse=True)
        best_cluster = clusters[0]
        members = list(best_cluster["members"])
        _chosen_candidate, chosen_line = max(members, key=lambda item: item[0].score)

        single_source_strong = (
            len(members) == 1
            and members[0][0].score >= 0.45
            and _line_similarity(draft_line, chosen_line) >= 0.38
        )

        if len(members) >= 2 or float(best_cluster["weight"]) >= 0.95 or single_source_strong:
            merged_lines.append(chosen_line)
            supporting_sources.extend(candidate.url for candidate, _line in members)
        else:
            merged_lines.append(draft_line)

    merged_lines = [line for line in merged_lines if line.strip()]
    corrections = _count_token_corrections(draft_lines, merged_lines)
    unique_sources = list(dict.fromkeys(supporting_sources))
    return merged_lines, corrections, unique_sources


class DuckDuckGoLyricsVerifier:
    name = "duckduckgo_lyrics_verifier"

    def __init__(
        self,
        max_results: int = 4,
        max_pages: int = 20,
        allowed_domains: tuple[str, ...] | None = None,
    ):
        self.max_results = max_results
        self.max_pages = max_pages
        self.allowed_domains = tuple(allowed_domains or ())

    def _collect_source_candidates(
        self,
        title: str,
        transcript_text: str,
        draft: TranscriptDraft,
        queries: list[str],
    ) -> list[LyricsSourceCandidate]:
        context = _extract_title_context(title)
        reference_tokens = set(_top_keywords(transcript_text, limit=16))
        reference_tokens.update(_top_keywords(context["clean_title"], limit=10))
        reference_tokens.update(_top_keywords(context["hebrew_artist"], limit=4))
        reference_tokens.update(_top_keywords(context["latin_artist"], limit=4))
        reference_tokens.update(_top_keywords(context["song"], limit=4))
        best_by_url: dict[str, LyricsSourceCandidate] = {}
        seen_urls: set[str] = set()

        for query in queries[: self.max_pages]:
            try:
                results = [
                    result
                    for result in _search_duckduckgo_results(query)
                    if (
                        _looks_like_lyrics_result(result)
                        and _matches_title_context(result, context)
                        and _domain_is_allowed(result.url, self.allowed_domains)
                    )
                ]
                results.sort(
                    key=lambda result: (
                        not _looks_like_lyrics_site(result.url),
                        _domain_priority(result.url),
                        result.url,
                    )
                )
                results = results[: self.max_results]
            except Exception as exc:
                logger.info("Lyrics search failed for query %s: %s", query, exc)
                continue

            for result in results:
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                snippet_score = _text_overlap_score(reference_tokens, f"{result.title} {result.snippet}")
                page_score = 0.0
                page_excerpt = result.snippet
                corrected_lines: list[str] = []
                correction_count = 0

                if _looks_like_lyrics_site(result.url):
                    try:
                        page_html = _fetch_text(result.url, timeout=10)
                        if _is_bot_blocked(page_html):
                            logger.info("Bot-blocked on %s — skipping page", result.url)
                        else:
                            page_text = _strip_html(page_html)
                            page_score = _text_overlap_score(reference_tokens, page_text)
                            page_excerpt = page_text[:240]
                            site_html = _extract_site_specific_lyrics(result.url, page_html)
                            effective_html = site_html if site_html else page_html
                            window_score, candidate_lines, candidate_corrections = _find_best_lyrics_window(effective_html, draft)
                            if window_score < 0.10:
                                logger.info(
                                    "Low window score %.3f for %s (Hebrew content too sparse?)",
                                    window_score, result.url,
                                )
                            page_score = max(page_score, window_score)
                            corrected_lines = candidate_lines
                            correction_count = candidate_corrections
                    except Exception as exc:
                        logger.info("Lyrics page fetch failed for %s: %s", result.url, exc)

                score = max(snippet_score, page_score)
                if not corrected_lines or score < 0.16:
                    continue

                candidate = LyricsSourceCandidate(
                    title=result.title,
                    url=result.url,
                    domain=_domain_label(result.url),
                    score=score,
                    excerpt=page_excerpt or result.snippet,
                    lines=corrected_lines,
                    correction_count=correction_count,
                )
                existing = best_by_url.get(result.url)
                if existing is None or candidate.score > existing.score:
                    best_by_url[result.url] = candidate

            if len(best_by_url) >= min(self.max_results, max(1, len(self.allowed_domains) or self.max_results)):
                break

        return sorted(best_by_url.values(), key=lambda item: item.score, reverse=True)

    def verify(self, title: str, draft: TranscriptDraft) -> LyricsVerificationResult:
        draft_text = draft.text.strip()
        draft_lines = _draft_lines(draft)
        warnings = _local_warnings(draft_text)
        queries = _build_query_variants(title, draft_text)
        search_query = queries[0] if queries else ""

        source_candidates = self._collect_source_candidates(title, draft_text, draft, queries)[: self.max_results]
        best_candidate = source_candidates[0] if source_candidates else None
        merged_lines, merged_corrections, supporting_sources = _merge_source_candidates(draft, source_candidates[:3])

        verified_confidence = 0.0
        if source_candidates and merged_lines:
            sample = source_candidates[: min(3, len(source_candidates))]
            verified_confidence = round(sum(candidate.score for candidate in sample) / len(sample), 3)

        best_single_score = source_candidates[0].score if source_candidates else 0.0
        apply_corrections = bool(
            merged_lines
            and merged_corrections > 0
            and (
                (len(supporting_sources) >= 2 and verified_confidence >= 0.32)
                or best_single_score >= 0.45
            )
        )
        selected_option_id = "verified" if apply_corrections else "draft"

        options: list[dict[str, object]] = [
            {
                "option_id": "draft",
                "label": "תמלול מקורי",
                "lines": draft_lines,
                "source_url": "",
                "confidence": 1.0,
                "source_count": 0,
            }
        ]
        if merged_lines:
            options.insert(
                0,
                {
                    "option_id": "verified",
                    "label": "אחרי אימות ותיקונים",
                    "lines": merged_lines,
                    "source_url": "",
                    "confidence": verified_confidence,
                    "source_count": len(supporting_sources),
                    "supporting_sources": supporting_sources,
                }
            )
        for candidate in source_candidates[:3]:
            options.append(
                {
                    "option_id": f"source_{_domain_option_key(candidate.url)}",
                    "label": f"מקור: {_domain_display_name(candidate.url)}",
                    "lines": candidate.lines,
                    "source_url": candidate.url,
                    "confidence": round(candidate.score, 3),
                    "source_count": 1,
                }
            )

        top_score = max([verified_confidence] + [candidate.score for candidate in source_candidates], default=0.0)
        if len(supporting_sources) >= 2 and merged_lines:
            verdict = "matched"
            summary = f"נמצאו כמה מקורות ברשת ונבנתה גרסה מאומתת מתוך {len(supporting_sources)} מקורות תומכים."
        elif best_candidate and best_candidate.score >= 0.30:
            verdict = "matched"
            summary = "נמצאה התאמה טובה למילות השיר ברשת, אבל אין עדיין הסכמה חזקה בין כמה מקורות."
        elif best_candidate and best_candidate.score >= 0.16:
            verdict = "uncertain"
            summary = "נמצאו מקורות חלקיים בלבד, לכן עדיף לבחור ידנית את הגרסה המתאימה לפני אישור."
        elif queries:
            verdict = "mismatch"
            summary = "לא נמצאה התאמה מספיק טובה למילות השיר ברשת, לכן חשוב לעבור ידנית על הטקסט."
        else:
            verdict = "not_run"
            summary = "לא היה מספיק מידע כדי לאמת את מילות השיר מול הרשת."

        if apply_corrections:
            summary += f" בוצעו {merged_corrections} תיקוני מילים אוטומטיים בגרסה המאומתת."

        return LyricsVerificationResult(
            provider=self.name,
            verdict=verdict,
            confidence=round(top_score, 3),
            search_query=search_query,
            summary=summary,
            matched_sources=[candidate.url for candidate in source_candidates[:3]],
            web_excerpt=(best_candidate.excerpt if best_candidate else "")[:240],
            local_warnings=warnings,
            corrected_lines=merged_lines,
            correction_count=merged_corrections,
            applied=apply_corrections,
            selected_option_id=selected_option_id,
            options=options,
        )

class PreferredSiteLyricsVerifier(DuckDuckGoLyricsVerifier):
    name = "preferred_site_lyrics_verifier"

    def __init__(self, max_results: int = 4, max_pages: int = 20):
        super().__init__(
            max_results=max_results,
            max_pages=max_pages,
            allowed_domains=PREFERRED_LYRICS_DOMAINS,
        )


# ---------------------------------------------------------------------------
# Gemini-based lyrics verifier
# ---------------------------------------------------------------------------

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
XAI_CHAT_API_URL = "https://api.x.ai/v1/chat/completions"

GEMINI_LYRICS_PROMPT = """\
אתה עוזר למצוא מילים מדויקות לשירים בעברית.

שם השיר: {title}

טיוטה (תמלול אוטומטי, עלולה להכיל שגיאות):
{draft_text}

בבקשה מצא ותחזיר את המילים המדויקות של השיר הזה.

כללים חשובים:
1. החזר רק את מילות השיר עצמן — בלי הסברים, כותרות, או הערות
2. כל שורה בשורה נפרדת
3. שמור על סדר השורות כמו בשיר המקורי
4. תקן שגיאות כתיב וזיהוי מהטיוטה
5. אם אתה לא בטוח לגבי השיר, החזר את הטיוטה כמו שהיא בלי שינויים
6. אל תוסיף שורות שאינן קיימות בשיר
"""


def _call_gemini(prompt: str, api_key: str, model: str, timeout: int = LLM_TIMEOUT_SECONDS) -> str:
    """Call Gemini REST API and return the generated text.

    Retries once on HTTP 429 (rate limit) after a short delay.
    """
    import time

    url = GEMINI_API_URL.format(model=model, key=api_key)
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
        },
    }).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            request = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return parts[0].get("text", "") if parts else ""
        except HTTPError as exc:
            last_error = exc
            if attempt == 0 and (exc.code == 429 or exc.code >= 500):
                logger.info("Gemini HTTP %s, retrying...", exc.code)
                time.sleep(5 if exc.code == 429 else 2)
                continue
            raise
        except Exception as exc:
            # Timeouts and transient network failures deserve one more try —
            # production logs showed deep-verify dying on a single timeout.
            last_error = exc
            if attempt == 0:
                logger.info("Gemini call failed (%s), retrying once...", exc)
                time.sleep(2)
                continue
            raise
    raise last_error  # type: ignore[misc]


def _call_grok(prompt: str, api_key: str, model: str, timeout: int = LLM_TIMEOUT_SECONDS) -> str:
    """Call xAI's OpenAI-compatible chat completions endpoint and return text."""
    import time

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Hebrew lyrics verifier. Follow the user's requested output format exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": False,
        }
    ).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            request = urllib.request.Request(
                XAI_CHAT_API_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            choices = result.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    str(part.get("text", "")).strip()
                    for part in content
                    if isinstance(part, dict) and str(part.get("type", "")).lower() == "text"
                ]
                return "\n".join(part for part in parts if part)
            return str(content or "")
        except HTTPError as exc:
            last_error = exc
            if attempt == 0 and (exc.code == 429 or exc.code >= 500):
                logger.info("Grok HTTP %s, retrying...", exc.code)
                time.sleep(5 if exc.code == 429 else 2)
                continue
            raise
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                logger.info("Grok call failed (%s), retrying once...", exc)
                time.sleep(2)
                continue
            raise
    raise last_error  # type: ignore[misc]


def _call_lyrics_llm(prompt: str, provider: str, api_key: str, model: str, timeout: int = LLM_TIMEOUT_SECONDS) -> str:
    normalized = _normalize_lyrics_llm_provider(provider)
    if normalized == "grok":
        return _call_grok(prompt, api_key, model, timeout=timeout)
    return _call_gemini(prompt, api_key, model, timeout=timeout)


class GeminiLyricsVerifier:
    """Lyrics verifier backed by a configurable LLM provider."""

    name = "gemini_lyrics_verifier"

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        fallback_verifier: DuckDuckGoLyricsVerifier | None = None,
        provider: str = "",
    ):
        from .config import GEMINI_API_KEY, GEMINI_MODEL, LYRICS_LLM_PROVIDER, XAI_API_KEY, XAI_MODEL

        self.provider = _normalize_lyrics_llm_provider(provider or LYRICS_LLM_PROVIDER)
        self.display_name = _lyrics_llm_display_name(self.provider)
        self.name = f"{self.provider}_lyrics_verifier"
        self.api_key = api_key or (XAI_API_KEY if self.provider == "grok" else GEMINI_API_KEY)
        self.model = model or (XAI_MODEL if self.provider == "grok" else GEMINI_MODEL)
        self._fallback_verifier = fallback_verifier

    def _fallback_or_draft(self, title: str, draft: TranscriptDraft, summary: str, warnings: list[str]) -> LyricsVerificationResult:
        summary = self._decorate_provider_text(summary)
        if self._fallback_verifier is not None:
            result = self._fallback_verifier.verify(title, draft)
            result.llm_provider = self.provider
            return result
        return _build_draft_only_result(
            provider=self.name,
            draft=draft,
            warnings=warnings,
            summary=summary,
            verdict="mismatch",
            search_query=f"{self.display_name}: {title}",
            llm_provider=self.provider,
        )

    def _decorate_provider_text(self, text: str) -> str:
        if self.provider == "gemini" or not text:
            return text
        return text.replace("Gemini", self.display_name)

    def _retag_provider_result(self, result: LyricsVerificationResult) -> LyricsVerificationResult:
        result.llm_provider = self.provider
        if self.provider == "gemini":
            return result

        result.search_query = self._decorate_provider_text(result.search_query)
        result.summary = self._decorate_provider_text(result.summary)
        result.matched_sources = [self.provider if source == "gemini" else source for source in result.matched_sources]

        for option in result.options:
            label = option.get("label")
            if isinstance(label, str):
                option["label"] = self._decorate_provider_text(label)
            if option.get("option_id") == "source_gemini":
                option["option_id"] = _lyrics_llm_source_option_id(self.provider)
            supporting_sources = option.get("supporting_sources")
            if isinstance(supporting_sources, list):
                option["supporting_sources"] = [
                    self.provider if source == "gemini" else source
                    for source in supporting_sources
                ]

        return result

    def _parse_gemini_lines(self, raw_text: str) -> list[str]:
        """Parse Gemini's response into clean lyrics lines."""
        text = raw_text.strip()

        for marker in ("THOUGHTS:", "Thoughts:", "thoughts:"):
            if text.startswith(marker):
                for i, line in enumerate(text.splitlines()):
                    stripped = line.strip()
                    if stripped and re.search(r"[\u0590-\u05FF]", stripped):
                        if not re.match(r"^[\(\[]", stripped):
                            text = "\n".join(text.splitlines()[i:])
                            break
                break

        lines = []
        in_thinking = False
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned.startswith(("THOUGHTS:", "Thoughts:", "thoughts:", "(Self-correction")):
                in_thinking = True
                continue
            if in_thinking:
                if re.search(r"[\u0590-\u05FF]", cleaned) and not re.match(r"^[\(\[]", cleaned):
                    in_thinking = False
                else:
                    continue
            if cleaned.startswith(("```", "##", "**", "שם השיר", "מילים:", "לחן:", "Lyrics:")):
                continue
            if re.search(r"[\u0590-\u05FF]", cleaned):
                lines.append(cleaned)
        return lines

    def verify(self, title: str, draft: TranscriptDraft) -> LyricsVerificationResult:
        draft_text = draft.text.strip()
        draft_lines = _draft_lines(draft)
        warnings = _local_warnings(draft_text)

        if not self.api_key:
            logger.info("No %s API key configured", self.display_name)
            return self._fallback_or_draft(title, draft, "Gemini לא זמין כרגע לבדיקת מילים.", warnings)

        try:
            prompt = GEMINI_LYRICS_PROMPT.format(
                title=title,
                draft_text=draft_text,
            )
            gemini_text = _call_lyrics_llm(prompt, self.provider, self.api_key, self.model)
            if not gemini_text.strip():
                logger.info("%s returned empty response", self.display_name)
                return self._fallback_or_draft(title, draft, "Gemini לא החזיר מילים שימושיות.", warnings)

            gemini_lines = self._parse_gemini_lines(gemini_text)
            if not gemini_lines:
                logger.info("%s returned no Hebrew lyrics", self.display_name)
                return self._fallback_or_draft(title, draft, "Gemini לא זיהה שורות מילים בעברית.", warnings)

            logger.info("%s returned %d lyrics lines for '%s'", self.display_name, len(gemini_lines), title[:40])
        except Exception as exc:
            logger.warning("%s lyrics request failed: %s", self.display_name, exc)
            return self._fallback_or_draft(title, draft, "Gemini נכשל בבקשת המילים לשיר.", warnings)

        gemini_tokens = [
            token
            for line in gemini_lines
            for token in _tokenize_words(line)
        ]
        corrected_lines, correction_count = _align_source_to_segments(gemini_tokens, draft)

        draft_tokens = [_normalize_token(t) for t in _tokenize_words(draft_text)]
        gemini_norm = [_normalize_token(t) for t in gemini_tokens]
        if draft_tokens and gemini_norm:
            score = SequenceMatcher(None, draft_tokens, gemini_norm, autojunk=False).ratio()
        else:
            score = 0.0

        apply_corrections = bool(
            corrected_lines
            and correction_count > 0
            and score >= 0.30
        )
        selected_option_id = "verified" if apply_corrections else "draft"

        options: list[dict[str, object]] = [
            {
                "option_id": "draft",
                "label": "תמלול מקורי",
                "lines": draft_lines,
                "source_url": "",
                "confidence": 1.0,
                "source_count": 0,
            }
        ]
        if corrected_lines:
            options.insert(
                0,
                {
                    "option_id": "verified",
                    "label": "אחרי אימות Gemini",
                    "lines": corrected_lines,
                    "source_url": "",
                    "confidence": round(score, 3),
                    "source_count": 1,
                    "supporting_sources": ["gemini"],
                },
            )
            options.append(
                {
                    "option_id": "source_gemini",
                    "label": "מקור: Gemini",
                    "lines": corrected_lines,
                    "source_url": "",
                    "confidence": round(score, 3),
                    "source_count": 1,
                }
            )

        if score >= 0.30:
            verdict = "matched"
            summary = f"Gemini מצא את מילות השיר עם ביטחון {score:.0%}."
        elif score >= 0.15:
            verdict = "uncertain"
            summary = "Gemini החזיר תוצאה חלקית, מומלץ לעבור ידנית על הטקסט."
        else:
            verdict = "mismatch"
            summary = "Gemini לא הצליח למצוא התאמה טובה למילות השיר."

        if apply_corrections:
            summary += f" בוצעו {correction_count} תיקוני מילים אוטומטיים."

        return self._retag_provider_result(LyricsVerificationResult(
            provider=self.name,
            verdict=verdict,
            confidence=round(score, 3),
            search_query=f"Gemini: {title}",
            summary=summary,
            matched_sources=["gemini"],
            web_excerpt=gemini_text[:240],
            local_warnings=warnings,
            corrected_lines=corrected_lines,
            correction_count=correction_count,
            applied=apply_corrections,
            selected_option_id=selected_option_id,
            options=options,
        ))


class HybridLyricsVerifier:
    """Prefer Shironet/Tab4U/Nagnu, and use the configured LLM only if needed."""

    name = "hybrid_lyrics_verifier"

    def __init__(self):
        self.preferred = PreferredSiteLyricsVerifier()
        self.gemini = GeminiLyricsVerifier(fallback_verifier=None)
        self.llm = self.gemini

    def verify(self, title: str, draft: TranscriptDraft) -> LyricsVerificationResult:
        preferred_result = self.preferred.verify(title, draft)
        source_options = [
            option
            for option in preferred_result.options
            if str(option.get("option_id", "")).startswith("source_")
        ]
        if source_options:
            preferred_result.provider = self.name
            return preferred_result

        gemini_result = self.gemini.verify(title, draft)
        gemini_result.provider = self.name
        preferred_warnings = preferred_result.local_warnings or []
        gemini_warnings = gemini_result.local_warnings or []
        gemini_result.local_warnings = list(dict.fromkeys([*preferred_warnings, *gemini_warnings]))
        gemini_result.summary = (
            "לא נמצאו מילים שימושיות בשירונט, Tab4U או נגינה, אז בוצע fallback ל-Gemini. "
            + gemini_result.summary
        )
        gemini_result.summary = self.gemini._decorate_provider_text(gemini_result.summary)
        if not gemini_result.search_query:
            gemini_result.search_query = preferred_result.search_query
        return gemini_result


class MultiStepLyricsVerifier:
    """Multi-step lyrics verifier with parallel source search and LLM fallback."""

    name = "multi_step_lyrics_verifier"

    def __init__(self):
        from .config import GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID, GEMINI_API_KEY, GEMINI_MODEL, LYRICS_LLM_PROVIDER, XAI_API_KEY, XAI_MODEL
        from .google_search import GoogleSearchProvider, YouTubeDescriptionProvider
        from .consensus import ConsensusEngine

        self._google = GoogleSearchProvider(api_key=GOOGLE_API_KEY, engine_id=GOOGLE_SEARCH_ENGINE_ID)
        self._youtube = YouTubeDescriptionProvider(api_key=GOOGLE_API_KEY)
        self._consensus = ConsensusEngine()
        self._llm_provider = _normalize_lyrics_llm_provider(LYRICS_LLM_PROVIDER)
        self._llm_display_name = _lyrics_llm_display_name(self._llm_provider)
        self._fallback_llm_provider = "gemini" if self._llm_provider != "gemini" and GEMINI_API_KEY else ""
        self._fallback_llm_display_name = _lyrics_llm_display_name(self._fallback_llm_provider)
        if self._llm_provider == "grok":
            self._gemini_api_key = XAI_API_KEY
            self._gemini_model = XAI_MODEL
        else:
            self._gemini_api_key = GEMINI_API_KEY
            self._gemini_model = GEMINI_MODEL
        self._fallback_api_key = GEMINI_API_KEY if self._fallback_llm_provider == "gemini" else ""
        self._fallback_model = GEMINI_MODEL if self._fallback_llm_provider == "gemini" else ""
        self._last_llm_provider_used = self._llm_provider
        self._last_llm_warning = ""

    def _decorate_llm_text(self, text: str, provider: str) -> str:
        if provider == "gemini" or not text:
            return text
        return text.replace("Gemini", _lyrics_llm_display_name(provider))

    def _retag_llm_result(
        self,
        result: LyricsVerificationResult,
        *,
        provider_used: str | None = None,
    ) -> LyricsVerificationResult:
        search_warnings = list(getattr(self, "_last_search_warnings", []) or [])
        if search_warnings:
            result.local_warnings = list(dict.fromkeys([*search_warnings, *result.local_warnings]))
        actual_provider = _normalize_lyrics_llm_provider(provider_used or self._llm_provider)
        result.llm_provider = actual_provider
        if actual_provider == "gemini":
            return result

        result.summary = self._decorate_llm_text(result.summary, actual_provider)
        result.search_query = self._decorate_llm_text(result.search_query, actual_provider)
        result.matched_sources = [actual_provider if source == "gemini" else source for source in result.matched_sources]
        result.local_warnings = [self._decorate_llm_text(warning, actual_provider) for warning in result.local_warnings]

        for option in result.options:
            label = option.get("label")
            if isinstance(label, str):
                option["label"] = self._decorate_llm_text(label, actual_provider)
            if option.get("option_id") == "source_gemini":
                option["option_id"] = _lyrics_llm_source_option_id(actual_provider)
            supporting_sources = option.get("supporting_sources")
            if isinstance(supporting_sources, list):
                option["supporting_sources"] = [
                    actual_provider if source == "gemini" else source
                    for source in supporting_sources
                ]

        return result

    def _run_llm_prompt(self, prompt: str, *, timeout: int) -> str:
        self._last_llm_provider_used = self._llm_provider
        self._last_llm_warning = ""
        try:
            return _call_lyrics_llm(
                prompt,
                self._llm_provider,
                self._gemini_api_key,
                self._gemini_model,
                timeout=timeout,
            )
        except Exception as exc:
            if self._fallback_llm_provider and self._fallback_api_key:
                logger.warning(
                    "%s failed for lyrics verification, retrying with %s: %s",
                    self._llm_display_name,
                    self._fallback_llm_display_name,
                    exc,
                )
                response_text = _call_lyrics_llm(
                    prompt,
                    self._fallback_llm_provider,
                    self._fallback_api_key,
                    self._fallback_model,
                    timeout=timeout,
                )
                self._last_llm_provider_used = self._fallback_llm_provider
                self._last_llm_warning = (
                    f"{self._llm_display_name} לא זמין כרגע, בוצע fallback ל-{self._fallback_llm_display_name}."
                )
                return response_text
            raise

    def verify(self, title: str, draft: TranscriptDraft) -> LyricsVerificationResult:
        """Main entry point: search sources, build consensus, fall back to the configured LLM if needed."""
        draft_text = draft.text
        draft_lines = _draft_lines(draft)

        try:
            sources = self._search_all_sources(title, draft)
        except Exception as exc:
            logger.warning("Source search failed: %s", exc)
            sources = {}

        if CONSENSUS_MODE == "positional":
            consensus = self._consensus.evaluate(sources)
        else:
            consensus = self._consensus.evaluate_aligned(sources, priority=DOMAIN_PRIORITY)
            if consensus.consensus_reached and not _consensus_matches_draft(consensus.lyrics, draft_text):
                # Guard against confidently agreeing on the WRONG song: the
                # sources may all describe another track with a similar title.
                logger.warning(
                    "Aligned consensus rejected: agreed lyrics are too dissimilar to the transcript."
                )
                consensus = ConsensusResult(
                    consensus_reached=False,
                    agreed_sources=consensus.agreed_sources,
                    lyrics=consensus.lyrics,
                    disputes=consensus.disputes,
                )
        merged_search_lines, merged_search_sources = _merge_search_source_versions(sources)

        if consensus.consensus_reached and consensus.agreed_sources >= 3:
            correction_count = _estimate_line_corrections(draft, consensus.lyrics)
            apply_corrections = _should_auto_apply_verified_lines(
                draft,
                consensus.lyrics,
                0.95,
                threshold=0.80,
            )
            return self._retag_llm_result(LyricsVerificationResult(
                provider=self.name,
                verdict=VerificationVerdict.CONSENSUS.value,
                confidence=0.95,
                corrected_lines=consensus.lyrics,
                correction_count=correction_count,
                applied=apply_corrections,
                selected_option_id="verified" if apply_corrections else "draft",
                options=_build_verification_options(
                    draft,
                    verified_lines=consensus.lyrics,
                    verified_label=f"קונצנזוס בין {consensus.agreed_sources} מקורות",
                    verified_confidence=0.95,
                    verified_source_count=consensus.agreed_sources,
                    supporting_sources=list(sources.keys()),
                    extra_source_options=[
                        option
                        for option in [
                            _build_search_merged_option(
                                merged_search_lines,
                                merged_search_sources,
                                confidence=0.82,
                                reference_lines=consensus.lyrics,
                            )
                        ]
                        if option
                    ],
                ),
                matched_sources=list(sources.keys()),
                summary=f"קונצנזוס בין {consensus.agreed_sources} מקורות",
                consensus_result=consensus,
                source_versions=sources,
            ))

        if not sources:
            try:
                lyrics, confidence = self._gemini_knowledge_verify(draft_text, title)
                provider_used = self._last_llm_provider_used
                fallback_warning = self._last_llm_warning
                normalized_lyrics = [str(line).strip() for line in lyrics if str(line).strip()]
                confidence = _effective_llm_confidence(
                    confidence, normalized_lyrics, sources=None, draft_text=draft_text
                )
                correction_count = _estimate_line_corrections(draft, normalized_lyrics)
                apply_corrections = _should_auto_apply_verified_lines(
                    draft,
                    normalized_lyrics,
                    confidence,
                    threshold=0.78,
                )
                return self._retag_llm_result(LyricsVerificationResult(
                    provider=self.name,
                    verdict=VerificationVerdict.NO_SOURCES.value,
                    confidence=confidence,
                    corrected_lines=normalized_lyrics,
                    correction_count=correction_count,
                    applied=apply_corrections,
                    selected_option_id="verified" if apply_corrections else "draft",
                    options=_build_verification_options(
                        draft,
                        verified_lines=normalized_lyrics,
                        verified_label="אימות Gemini ללא מקורות",
                        verified_confidence=confidence,
                        verified_source_count=1,
                        verified_source_option_id="source_gemini",
                        verified_source_label="מקור: Gemini",
                        supporting_sources=["gemini"],
                    ),
                    matched_sources=[],
                    summary="לא נמצאו מקורות אונליין, בוצע אימות על בסיס ידע Gemini",
                    local_warnings=[
                        warning
                        for warning in [
                            "לא נמצאו מקורות מילים באינטרנט",
                            fallback_warning,
                        ]
                        if warning
                    ],
                    consensus_result=consensus,
                    source_versions=sources,
                ), provider_used=provider_used)
            except Exception as exc:
                logger.warning("%s knowledge verify failed: %s", self._llm_display_name, exc)
                return self._retag_llm_result(LyricsVerificationResult(
                    provider=self.name,
                    verdict=VerificationVerdict.NO_SOURCES.value,
                    confidence=0.3,
                    corrected_lines=draft_lines,
                    correction_count=0,
                    matched_sources=[],
                    summary="לא נמצאו מקורות ו-Gemini לא זמין",
                    local_warnings=[
                        "לא נמצאו מקורות מילים באינטרנט",
                        f"שגיאה בגישה ל-Gemini: {exc}",
                    ],
                    selected_option_id="draft",
                    options=_build_verification_options(draft),
                    consensus_result=consensus,
                    source_versions=sources,
                ))

        try:
            lyrics, confidence, uncertain = self._gemini_deep_verify(
                sources, consensus.disputes, title
            )
            provider_used = self._last_llm_provider_used
            fallback_warning = self._last_llm_warning
            confidence = _effective_llm_confidence(
                confidence, lyrics, sources=sources, draft_text=draft_text
            )
            correction_count = _estimate_line_corrections(draft, lyrics)
            apply_corrections = _should_auto_apply_verified_lines(
                draft,
                lyrics,
                confidence,
                threshold=0.72,
            )
            source_options = [
                {
                    "option_id": f"source_{source_name}",
                    "label": f"מקור: {SOURCE_DISPLAY_NAMES.get(source_name, source_name)}",
                    "lines": lines,
                    "source_url": "",
                    "confidence": 0.75,
                    "source_count": 1,
                }
                for source_name, lines in sources.items()
                if lines
            ]
            search_merged_option = _build_search_merged_option(
                merged_search_lines,
                merged_search_sources,
                confidence=max(0.58, min(confidence, 0.78)),
                reference_lines=lyrics,
            )
            extra_source_options = []
            if search_merged_option:
                extra_source_options.append(search_merged_option)
            extra_source_options.extend(source_options[:3])
            return self._retag_llm_result(LyricsVerificationResult(
                provider=self.name,
                verdict=VerificationVerdict.GEMINI_VERIFIED.value,
                confidence=confidence,
                corrected_lines=lyrics,
                correction_count=correction_count,
                applied=apply_corrections,
                selected_option_id="verified" if apply_corrections else "draft",
                options=_build_verification_options(
                    draft,
                    verified_lines=lyrics,
                    verified_label="הכרעת Gemini בין מקורות",
                    verified_confidence=confidence,
                    verified_source_count=max(1, len(sources)),
                    verified_source_option_id="source_gemini",
                    verified_source_label="מקור: Gemini",
                    supporting_sources=["gemini", *sources.keys()],
                    extra_source_options=extra_source_options,
                ),
                matched_sources=list(sources.keys()),
                summary=f"Gemini הכריע בין {len(sources)} מקורות",
                local_warnings=[*uncertain, *([fallback_warning] if fallback_warning else [])],
                consensus_result=consensus,
                source_versions=sources,
            ), provider_used=provider_used)
        except Exception as exc:
            logger.warning("%s deep verify failed: %s", self._llm_display_name, exc)
            fallback_lines = merged_search_lines or consensus.lyrics
            fallback_label = (
                "תוצאת חיפוש משולבת"
                if merged_search_lines
                else "הגרסה הטובה ביותר מהמקורות"
            )
            fallback_summary = (
                "Gemini לא זמין, הוחזרה תוצאת חיפוש משולבת מהמקורות"
                if merged_search_lines
                else "Gemini לא זמין, מוחזרת התוצאה הטובה ביותר מהמקורות"
            )
            fallback_source_count = len(merged_search_sources) if merged_search_lines else len(sources)
            source_options = [
                {
                    "option_id": f"source_{source_name}",
                    "label": f"מקור: {SOURCE_DISPLAY_NAMES.get(source_name, source_name)}",
                    "lines": lines,
                    "source_url": "",
                    "confidence": 0.5,
                    "source_count": 1,
                }
                for source_name, lines in list(sources.items())[:3]
                if lines
            ]
            return self._retag_llm_result(LyricsVerificationResult(
                provider=self.name,
                verdict=VerificationVerdict.GEMINI_VERIFIED.value,
                confidence=0.5,
                corrected_lines=fallback_lines,
                correction_count=_estimate_line_corrections(draft, fallback_lines),
                matched_sources=list(sources.keys()),
                summary=fallback_summary,
                local_warnings=[f"שגיאה ב-Gemini: {exc}"],
                selected_option_id="draft",
                options=_build_verification_options(
                    draft,
                    verified_lines=fallback_lines,
                    verified_label=fallback_label,
                    verified_confidence=0.5,
                    verified_source_count=fallback_source_count,
                    extra_source_options=source_options,
                ),
                consensus_result=consensus,
                source_versions=sources,
            ))

    def _search_all_sources(self, title: str, draft_or_text) -> dict[str, list[str]]:
        """Search multiple sources in parallel and return source_name -> lyrics lines."""
        return _search_all_sources_impl(self, title, draft_or_text)

    def _gemini_deep_verify(
        self,
        sources: dict[str, list[str]],
        disputes: list,
        title: str,
    ) -> tuple[list[str], float, list[str]]:
        """Call the configured LLM to decide which version is correct among disagreeing sources."""
        source_text = ""
        for source_name, lines in sources.items():
            source_text += f"\n--- מקור: {source_name} ---\n"
            source_text += "\n".join(lines)
            source_text += "\n"

        dispute_text = ""
        if disputes:
            dispute_text = "\n\nשורות שנויות במחלוקת:\n"
            for d in disputes:
                dispute_text += f"שורה {d.line_number + 1}:\n"
                for src, ver in d.versions.items():
                    dispute_text += f"  {src}: {ver}\n"

        prompt = (
            f"אתה מומחה למילות שירים בעברית.\n"
            f"שם השיר: {title}\n\n"
            f"להלן גרסאות מילים ממקורות שונים:{source_text}"
            f"{dispute_text}\n"
            f"המשימה שלך:\n"
            f"1. השווה את כל הגרסאות\n"
            f"2. החלט איזו גרסה נכונה לכל שורה\n"
            f"3. אם אתה לא בטוח בשורה מסוימת, סמן אותה עם [?]\n\n"
            f"ודא שאתה מחזיר את המילים הנכונות בלבד, שורה אחת לכל שורת שיר.\n"
            f"בדוק את עצמך: האם המילים הגיוניות בהקשר של השיר?\n\n"
            f"החזר את התוצאה בפורמט הבא:\n"
            f"CONFIDENCE: <מספר בין 0 ל-1>\n"
            f"UNCERTAIN: <מילים לא ודאיות מופרדות בפסיקים, או NONE>\n"
            f"LYRICS:\n<המילים הנכונות, שורה לשורה>"
        )

        response_text = self._run_llm_prompt(prompt, timeout=LLM_TIMEOUT_SECONDS)
        return self._parse_gemini_response(response_text)

    def _gemini_knowledge_verify(
        self,
        whisper_text: str,
        title: str,
    ) -> tuple[list[str], float]:
        """Call the configured LLM with Whisper transcript for knowledge-based verification."""
        prompt = (
            f"אתה מומחה למילות שירים בעברית.\n"
            f"שם השיר: {title}\n\n"
            f"להלן תמלול אוטומטי (Whisper) של השיר:\n{whisper_text}\n\n"
            f"המשימה שלך:\n"
            f"1. בדוק אם אתה מכיר את השיר הזה\n"
            f"2. תקן שגיאות בתמלול על בסיס הידע שלך\n"
            f"3. אם אתה לא מכיר את השיר, החזר את התמלול כמו שהוא\n\n"
            f"החזר את התוצאה בפורמט הבא:\n"
            f"CONFIDENCE: <מספר בין 0 ל-1>\n"
            f"LYRICS:\n<המילים, שורה לשורה>"
        )

        response_text = self._run_llm_prompt(prompt, timeout=LLM_TIMEOUT_SECONDS)
        lyrics, confidence, _ = self._parse_gemini_response(response_text)
        return lyrics, confidence

    @staticmethod
    def _parse_gemini_response(response_text: str) -> tuple[list[str], float, list[str]]:
        """Parse structured Gemini response into lyrics, confidence, uncertain words."""
        # A missing CONFIDENCE header must not look almost-confident: 0.5
        # keeps it below every auto-apply threshold.
        confidence = 0.5
        uncertain: list[str] = []
        lyrics_lines: list[str] = []

        lines = response_text.strip().splitlines()
        in_lyrics = False
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(stripped.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif stripped.upper().startswith("UNCERTAIN:"):
                val = stripped.split(":", 1)[1].strip()
                if val.upper() != "NONE" and val:
                    uncertain = [w.strip() for w in val.split(",") if w.strip()]
            elif stripped.upper().startswith("LYRICS:"):
                in_lyrics = True
                remainder = stripped.split(":", 1)[1].strip()
                if remainder:
                    lyrics_lines.append(remainder)
            elif in_lyrics and stripped:
                lyrics_lines.append(stripped)

        if not lyrics_lines:
            # Fallback: treat entire response as lyrics
            lyrics_lines = [l.strip() for l in lines if l.strip()]

        return lyrics_lines, confidence, uncertain

    def post_review_steps(self, job, original_draft, aligned_segments=None):
        """Spec steps 5-7 after human approval.

        Step 5: character-level diff between the draft and the approved words,
        recorded in the manifest for observability.
        Step 6: sanity warning (optionally LLM-checked) for unusually large
        manual edits.
        Step 7: rebuild char timings for changed words that did not get a
        precise wav2vec2 measurement, using the vocals audio when available.
        """
        from . import job_manager
        from .aligner import realign_changed_words
        from .char_diff import compute_char_diffs, format_diff_table

        if aligned_segments is None:
            return
        draft_words = [
            word.word for segment in original_draft.segments for word in segment.words
        ]
        final_words = [word for segment in aligned_segments for word in segment.words]
        if not draft_words or not final_words:
            return

        draft_norm = [_normalize_token(word) for word in draft_words]
        final_norm = [_normalize_token(word.word) for word in final_words]
        changed: list[tuple[int, str, str]] = []
        matcher = SequenceMatcher(None, draft_norm, final_norm, autojunk=False)
        for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
            if tag != "replace":
                continue
            for offset in range(min(a_end - a_start, b_end - b_start)):
                changed.append(
                    (
                        b_start + offset,
                        draft_words[a_start + offset],
                        final_words[b_start + offset].word,
                    )
                )

        if not changed:
            job.manifest.post_review_diff = {"changed_words": 0, "realigned_words": 0}
            return

        # Step 7: partial re-timing for words whose boundaries were not
        # measured by wav2vec2 (edits often leave them interpolation-timed).
        audio_path = str(job.vocals_16k_path) if job.vocals_16k_path.exists() else None
        realigned = 0
        for final_index, _original, _corrected in changed:
            word = final_words[final_index]
            if word.aligned and word.source == "whisperx":
                continue
            try:
                char_timings = realign_changed_words(word, word.word, audio_path)
            except Exception as exc:
                logger.info("Partial realignment failed for '%s': %s", word.word, exc)
                continue
            if char_timings:
                word.char_timings = char_timings
                realigned += 1

        # Step 5: persist the diff for observability.
        diffs = compute_char_diffs(
            [original for _index, original, _corrected in changed],
            [corrected for _index, _original, corrected in changed],
            [index for index, _original, _corrected in changed],
        )
        job.manifest.post_review_diff = {
            "changed_words": len(changed),
            "realigned_words": realigned,
            "diff_table": format_diff_table(diffs)[:4000],
        }

        # Step 6: unusually large manual edits get a visible sanity warning.
        if len(changed) >= 12:
            job_manager.add_warning(
                job,
                f"בוצעו {len(changed)} שינויי מילים ביחס לתמלול — כדאי לוודא שהטקסט הסופי מדויק לפני השיתוף.",
            )


def _relax_search_queries(query: str) -> list[str]:
    variants: list[str] = []

    def _add(candidate: str) -> None:
        normalized = " ".join(candidate.split())
        if normalized and normalized not in variants:
            variants.append(normalized)

    _add(query)
    _add(re.sub(r'"([^"]+)"', r"\1", query))
    no_site = re.sub(r"\bsite:[^\s]+\s*", "", query)
    _add(no_site)
    _add(no_site.replace('"', ""))
    return variants


_PAGE_NOISE_KEYWORDS = {
    "אקורדים",
    "אקורד",
    "מילים",
    "מילים לשיר",
    "מילים:",
    "מילים ולחן",
    "פרסומת",
    "פרסומות",
    "תגובות",
    "תגובה",
    "שתפו",
    "שיתוף",
    "פייסבוק",
    "אינסטגרם",
    "יוטיוב",
    "וואטסאפ",
    "לחצו",
    "כניסה",
    "הרשמה",
    "חיפוש",
    "תפריט",
    "בית",
    "ראשי",
    "צפיות",
    "צפה",
    "להדפסה",
    "הדפסה",
    "האזנה",
    "הורדה",
}

_YOUTUBE_CREDIT_PHRASES = {
    "יחסי ציבור",
    "תקשורת ויחסי ציבור",
    "מילים ולחן",
    "לחן ועיבוד",
    "עיבוד והפקה",
    "עריכה דיגיטלית",
    "הפצה דיגיטלית",
    "הפצה דיגטלית",
    "ניהול אישי",
    "public relations",
    "digital editing",
    "digital distribution",
    "mix and master",
    "mix master",
}

_YOUTUBE_CREDIT_ROLE_WORDS = {
    "עיבוד",
    "הפקה",
    "תופים",
    "קלידים",
    "תכנות",
    "תכנותים",
    "גיטרות",
    "גיטרה",
    "בס",
    "בוזוקי",
    "פסנתר",
    "כינור",
    "כינורות",
    "סקסופון",
    "חצוצרה",
    "טרומבון",
    "חליל",
    "קולות",
    "סטיילינג",
    "איפור",
    "צילום",
    "בימוי",
    "עריכה",
    "גרפיקה",
    "מיקס",
    "מאסטר",
    "הפצה",
    "drums",
    "keys",
    "keyboards",
    "programming",
    "guitars",
    "guitar",
    "bass",
    "bouzouki",
    "piano",
    "violin",
    "violins",
    "saxophone",
    "trumpet",
    "trombone",
    "flute",
    "vocals",
    "styling",
    "makeup",
    "photo",
    "video",
    "editing",
    "graphics",
    "mix",
    "master",
    "management",
    "distribution",
    "producer",
    "production",
}

_YOUTUBE_LYRIC_GUARD_WORDS = {
    "אני",
    "את",
    "אתה",
    "אתם",
    "אתן",
    "הוא",
    "היא",
    "אנחנו",
    "לי",
    "לך",
    "לו",
    "לה",
    "שלי",
    "שלך",
    "שלו",
    "שלה",
    "אותי",
    "אותך",
    "אותו",
    "אותה",
    "איתי",
    "איתך",
    "עליי",
    "עליך",
    "עליו",
    "עליה",
}


def _line_token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in _line_signature(left).split() if token}
    right_tokens = {token for token in _line_signature(right).split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _looks_like_noise_line(line: str) -> bool:
    normalized = _line_signature(line)
    if not normalized:
        return True
    if len(normalized.split()) > 14 or len(normalized.replace(" ", "")) > 80:
        return True
    if re.search(r"(?:^|[^\d])(?:0\d[\s\-]*){4,}\d(?:$|[^\d])|\d{7,}", line):
        return True
    if re.search(
        r"(?:\u05de\u05d9\u05dc\u05d9\u05dd\s+\u05d5\u05dc\u05d7\u05df|"
        r"\u05dc\u05d7\u05df\s+\u05d5\u05e2\u05d9\u05d1\u05d5\u05d3|"
        r"\u05e2\u05d9\u05d1\u05d5\u05d3\s+\u05d5\u05d4\u05e4\u05e7\u05d4|"
        r"\u05dc\u05d4\u05d5\u05e4\u05e2\u05d5\u05ea|"
        r"\u05d4\u05e4\u05e7\u05d4|"
        r"\u05e6\u05d9\u05dc\u05d5\u05dd|"
        r"\u05d2\u05e8\u05e4\u05d9\u05e7\u05d4|"
        r"\u05de\u05d9\u05e7\u05e1|"
        r"\u05de\u05d0\u05e1\u05d8\u05e8)",
        normalized,
    ):
        return True
    # URLs, handles and very long Latin runs are junk; ordinary English words
    # (mixed-language lyrics) must survive, so the run threshold is high and
    # '#' is only noise when it is not part of a chord label like F#m.
    if re.search(r"https?://|www\.|@|[A-Za-z]{14,}", line):
        return True
    if "#" in re.sub(r"[A-G]#(?:m|maj7|m7|dim|sus[24]|add\d+|\d)?\b", "", line):
        return True
    return any(keyword in normalized for keyword in _PAGE_NOISE_KEYWORDS)


def _looks_like_youtube_credit_line(line: str) -> bool:
    normalized = _line_signature(line)
    if not normalized:
        return False

    words = normalized.split()
    if len(words) > 8:
        return False

    if any(phrase in normalized for phrase in _YOUTUBE_CREDIT_PHRASES):
        return True

    if "להופעות" in normalized or "להזמנות" in normalized:
        return True

    first_word = words[0]
    remaining_words = words[1:]
    if first_word in _YOUTUBE_CREDIT_ROLE_WORDS:
        if not remaining_words:
            return True
        if not any(word in _YOUTUBE_LYRIC_GUARD_WORDS for word in remaining_words):
            return True

    return False


def _trim_youtube_credit_edges(lines: list[str]) -> list[str]:
    normalized_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not normalized_lines:
        return []

    start_index = 0
    end_index = len(normalized_lines)

    while start_index < end_index and _looks_like_youtube_credit_line(normalized_lines[start_index]):
        start_index += 1

    while end_index > start_index and _looks_like_youtube_credit_line(normalized_lines[end_index - 1]):
        end_index -= 1

    if start_index >= end_index:
        return []

    return [
        line
        for line in normalized_lines[start_index:end_index]
        if not _looks_like_youtube_credit_line(line)
    ]


def _clean_youtube_description_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _looks_like_youtube_credit_line(line):
            continue
        if re.search(r"https?://|www\.|[@#]", line):
            continue
        if re.search(r"(?:^|[^\d])(?:0\d[\s\-]*){4,}\d(?:$|[^\d])|\d{7,}", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _best_draft_line_stats(candidate_line: str, draft_lines: list[str]) -> tuple[float, float]:
    if not draft_lines:
        return 0.0, 0.0
    similarities = [_line_similarity(candidate_line, draft_line) for draft_line in draft_lines]
    overlaps = [_line_token_overlap(candidate_line, draft_line) for draft_line in draft_lines]
    return max(similarities, default=0.0), max(overlaps, default=0.0)


def _extract_candidate_lyrics_line_entries(
    page_html: str,
    draft_lines: list[str] | None = None,
) -> list[tuple[str, list[str]]]:
    cleaned = _strip_html_preserving_lines(page_html)
    lines: list[tuple[str, list[str]]] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or _looks_like_noise_line(line):
            continue
        tokens = _tokenize_words(line)
        hebrew_tokens = [token for token in tokens if re.search(r"[\u0590-\u05FF]", token)]
        if len(hebrew_tokens) < 2 or len(hebrew_tokens) > 14:
            continue
        # Mixed-language songs: keep non-Hebrew words too (they used to be
        # dropped entirely). Only Latin tokens that look like chord labels
        # leaking from chords sites are filtered out.
        kept_tokens = [
            token
            for token in tokens
            if re.search(r"[\u0590-\u05FF]", token) or not _CHORD_LABEL_TOKEN_RE.match(token)
        ]
        if len(kept_tokens) > 18:
            continue
        normalized_line = " ".join(kept_tokens)
        if draft_lines:
            similarity, overlap = _best_draft_line_stats(normalized_line, draft_lines)
            if similarity < 0.22 and overlap < 0.34:
                continue
        lines.append((normalized_line, kept_tokens))
    return lines


def _line_similarity(left: str, right: str) -> float:
    left_tokens = [token for token in _line_signature(left).split() if token]
    right_tokens = [token for token in _line_signature(right).split() if token]
    if not left_tokens or not right_tokens:
        return 0.0
    return SequenceMatcher(None, left_tokens, right_tokens, autojunk=False).ratio()


def _is_substantial_lyrics_line(line: str) -> bool:
    signature = _line_signature(line)
    word_count = len(signature.split())
    char_count = len(signature.replace(" ", ""))
    return word_count >= 3 or char_count >= 10


def _build_repeat_anchor_map(lines: list[str]) -> dict[int, int]:
    anchors: dict[int, int] = {}
    similarity_cache: dict[tuple[int, int], float] = {}

    def _sim(left_index: int, right_index: int) -> float:
        key = (left_index, right_index)
        if key not in similarity_cache:
            similarity_cache[key] = _line_similarity(lines[left_index], lines[right_index])
        return similarity_cache[key]

    for right_start in range(1, len(lines)):
        for left_start in range(right_start):
            if _sim(left_start, right_start) < 0.84:
                continue
            run_length = 0
            while (
                right_start + run_length < len(lines)
                and left_start + run_length < right_start
                and _sim(left_start + run_length, right_start + run_length) >= 0.84
            ):
                run_length += 1
            if run_length < 2:
                continue
            for offset in range(run_length):
                anchors.setdefault(right_start + offset, left_start + offset)

    for right_index in range(1, len(lines)):
        if right_index in anchors or not _is_substantial_lyrics_line(lines[right_index]):
            continue
        for left_index in range(right_index):
            if _sim(left_index, right_index) >= 0.96:
                anchors[right_index] = left_index
                break

    return anchors


def _expand_repeated_candidate_lines(draft_lines: list[str], candidate_lines: list[str]) -> list[str]:
    normalized_draft = [line.strip() for line in draft_lines if line.strip()]
    normalized_candidate = [str(line).strip() for line in candidate_lines if str(line).strip()]
    if len(normalized_candidate) >= len(normalized_draft):
        return normalized_candidate

    anchor_map = _build_repeat_anchor_map(normalized_draft)
    if not anchor_map:
        return normalized_candidate

    expanded: list[str] = []
    assigned: dict[int, str] = {}
    candidate_index = 0

    for draft_index, draft_line in enumerate(normalized_draft):
        current_line = normalized_candidate[candidate_index] if candidate_index < len(normalized_candidate) else None
        if current_line is not None and _line_similarity(current_line, draft_line) >= 0.58:
            chosen = current_line
            candidate_index += 1
        elif draft_index in anchor_map and anchor_map[draft_index] in assigned:
            chosen = assigned[anchor_map[draft_index]]
        else:
            return normalized_candidate

        expanded.append(chosen)
        assigned[draft_index] = chosen

    if candidate_index != len(normalized_candidate):
        return normalized_candidate
    return expanded


def _choose_local_word_correction(
    draft_word: str,
    source_word: str,
    *,
    line_similarity: float,
) -> tuple[str, bool]:
    draft_norm = _normalize_token(draft_word)
    source_norm = _normalize_token(source_word)
    if not draft_norm:
        return draft_word, False
    if draft_norm == source_norm:
        return source_word, False

    word_similarity = SequenceMatcher(None, draft_norm, source_norm, autojunk=False).ratio()
    shared_edge = (
        draft_norm[:2] == source_norm[:2]
        or draft_norm[-2:] == source_norm[-2:]
    )

    if word_similarity >= 0.5:
        return source_word, True
    if shared_edge and word_similarity >= 0.34:
        return source_word, True
    if line_similarity >= 0.82 and word_similarity >= 0.28 and abs(len(draft_norm) - len(source_norm)) <= 2:
        return source_word, True
    return draft_word, False


def _repair_draft_line_from_source_line(draft_line: str, source_line: str) -> tuple[str, int]:
    draft_words = _tokenize_words(draft_line)
    source_words = _tokenize_words(source_line)
    if not draft_words or not source_words:
        return draft_line, 0

    line_similarity = _line_similarity(draft_line, source_line)
    overlap = _line_token_overlap(draft_line, source_line)
    if line_similarity < 0.24 and overlap < 0.34:
        return draft_line, 0

    draft_norm = [_normalize_token(word) for word in draft_words]
    source_norm = [_normalize_token(word) for word in source_words]
    corrected_words: list[str] = []

    for tag, a0, a1, b0, b1 in SequenceMatcher(None, draft_norm, source_norm, autojunk=False).get_opcodes():
        if tag == "equal":
            corrected_words.extend(source_words[b0:b1])
            continue

        if tag == "replace":
            draft_chunk = draft_words[a0:a1]
            source_chunk = source_words[b0:b1]
            if len(draft_chunk) == len(source_chunk) and len(draft_chunk) <= 3:
                replaced_chunk: list[str] = []
                safe_chunk = True
                for draft_word, source_word in zip(draft_chunk, source_chunk):
                    chosen_word, changed = _choose_local_word_correction(
                        draft_word,
                        source_word,
                        line_similarity=line_similarity,
                    )
                    if not changed and _normalize_token(draft_word) != _normalize_token(source_word):
                        safe_chunk = False
                        break
                    replaced_chunk.append(chosen_word)
                if safe_chunk:
                    corrected_words.extend(replaced_chunk)
                    continue

        if tag in {"replace", "delete"}:
            corrected_words.extend(draft_words[a0:a1])

    corrected_line = " ".join(corrected_words).strip() or draft_line.strip()
    correction_count = sum(
        1
        for left, right in zip(_tokenize_words(draft_line), _tokenize_words(corrected_line))
        if _normalize_token(left) != _normalize_token(right)
    )
    return corrected_line, correction_count


def _repair_draft_lines_from_source_lines(
    draft_lines: list[str],
    source_lines: list[str],
) -> tuple[list[str], int]:
    corrected_lines: list[str] = []
    total_corrections = 0

    for index, draft_line in enumerate(draft_lines):
        source_line = source_lines[index] if index < len(source_lines) else ""
        corrected_line, correction_count = _repair_draft_line_from_source_line(draft_line, source_line)
        corrected_lines.append(corrected_line)
        total_corrections += correction_count

    return corrected_lines, total_corrections


def _find_best_lyrics_window(page_html: str, draft: TranscriptDraft) -> tuple[float, list[str], int]:
    draft_tokens = [_normalize_token(token) for token in _tokenize_words(draft.text)]
    if not draft_tokens:
        return 0.0, [], 0

    draft_lines = _draft_lines(draft)
    raw_lines = _extract_candidate_lyrics_line_entries(page_html, draft_lines)
    if not raw_lines:
        return 0.0, [], 0

    draft_token_set = set(draft_tokens)
    target_words = len(draft_tokens)
    target_line_count = max(1, len(draft_lines))
    min_window_lines = max(1, int(target_line_count * 0.5))
    max_window_lines = max(min_window_lines, target_line_count + 5)

    def _score_window(candidate_lines: list[tuple[str, list[str]]]) -> float:
        normalized = [_normalize_token(token) for _line, tokens in candidate_lines for token in tokens]
        if not normalized:
            return 0.0
        ratio = SequenceMatcher(None, draft_tokens, normalized, autojunk=False).ratio()
        overlap = len(draft_token_set & set(normalized)) / max(1, len(draft_token_set))
        line_coverage = min(len(candidate_lines), target_line_count) / max(
            len(candidate_lines), target_line_count, 1
        )
        return ratio * 0.62 + overlap * 0.23 + line_coverage * 0.15

    def _align_candidate_window(candidate_lines: list[tuple[str, list[str]]]) -> tuple[list[str], int]:
        source_lines = [line for line, _tokens in candidate_lines]
        expanded_lines = _expand_repeated_candidate_lines(draft_lines, source_lines)
        return _repair_draft_lines_from_source_lines(draft_lines, expanded_lines)

    if min_window_lines <= len(raw_lines) <= max_window_lines:
        score = _score_window(raw_lines)
        if score >= 0.45:
            corrected_lines, correction_count = _align_candidate_window(raw_lines)
            return score, corrected_lines, correction_count

    best_score = 0.0
    best_lines: list[tuple[str, list[str]]] = []

    for start in range(len(raw_lines)):
        collected: list[tuple[str, list[str]]] = []
        collected_token_count = 0
        max_end = min(len(raw_lines), start + max_window_lines)
        for end in range(start, max_end):
            collected.append(raw_lines[end])
            collected_token_count += len(raw_lines[end][1])
            window_line_count = end - start + 1
            if collected_token_count > int(target_words * 1.65) + 10:
                break
            if window_line_count < min_window_lines:
                continue
            if collected_token_count < max(4, int(target_words * 0.5)):
                continue

            score = _score_window(collected)
            if score > best_score:
                best_score = score
                best_lines = collected[:]

    if not best_lines:
        return 0.0, [], 0

    corrected_lines, correction_count = _align_candidate_window(best_lines)
    return best_score, corrected_lines, correction_count


def _search_duckduckgo_results(query: str) -> list[SearchResult]:
    cached = SEARCH_RESULT_CACHE.get(query)
    if cached is not None:
        return cached

    last_error: Exception | None = None
    for candidate_query in _relax_search_queries(query):
        encoded_query = urllib.parse.quote_plus(candidate_query)
        for endpoint in (DUCKDUCKGO_HTML_SEARCH, DUCKDUCKGO_HTML_FALLBACK):
            try:
                html_text = _fetch_text(endpoint.format(query=encoded_query))
                results = _parse_duckduckgo_results(html_text)
                if results:
                    SEARCH_RESULT_CACHE[query] = results
                    return results
            except HTTPError as exc:
                last_error = exc
                logger.info("DuckDuckGo search endpoint failed for %s: %s", candidate_query, exc)
                continue
            except Exception as exc:
                last_error = exc
                logger.info("DuckDuckGo search parse failed for %s: %s", candidate_query, exc)
                continue

    if last_error:
        logger.info("All DuckDuckGo search endpoints failed for %s: %s", query, last_error)
    return []


def _decode_bing_result_url(url: str) -> str:
    unescaped = html.unescape(url)
    parsed = urllib.parse.urlparse(unescaped)
    params = urllib.parse.parse_qs(parsed.query)
    encoded = params.get("u", [""])[0]
    if encoded.startswith("a1"):
        payload = encoded[2:]
        padded = payload + "=" * (-len(payload) % 4)
        try:
            decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")
            return urllib.parse.unquote(decoded)
        except Exception:
            return unescaped
    return unescaped


def _parse_bing_results(html_text: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html_text, re.S):
        title_match = re.search(r'<h2[^>]*>\s*<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', block, re.S)
        if not title_match:
            continue
        snippet_match = re.search(r'<p[^>]*>(?P<snippet>.*?)</p>', block, re.S)
        url = _decode_bing_result_url(title_match.group("href"))
        title = _strip_html(title_match.group("title"))
        snippet = _strip_html(snippet_match.group("snippet")) if snippet_match else ""
        if url:
            results.append(SearchResult(title=title, snippet=snippet, url=url))
    return results


def _search_bing_results(query: str) -> list[SearchResult]:
    cache_key = f"bing:{query}"
    cached = SEARCH_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        html_text = _fetch_text(BING_HTML_SEARCH.format(query=urllib.parse.quote_plus(query)), timeout=15)
        results = _parse_bing_results(html_text)
        if results:
            SEARCH_RESULT_CACHE[cache_key] = results
            return results
    except Exception as exc:
        logger.info("Bing search failed for %s: %s", query, exc)
    return []


def _search_web_results(query: str) -> list[SearchResult]:
    results = _search_duckduckgo_results(query)
    if results:
        return results
    return _search_bing_results(query)


def _build_synthetic_draft_from_text(text: str) -> TranscriptDraft:
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        tokens = _tokenize_words(line)
        if not tokens:
            continue
        duration = max(0.3, 0.32 * len(tokens))
        step = duration / max(len(tokens), 1)
        words = [
            WordTiming(
                word=token,
                start=cursor + index * step,
                end=cursor + (index + 1) * step,
                confidence=0.0,
                source="draft_text",
            )
            for index, token in enumerate(tokens)
        ]
        segments.append(
            TranscriptSegment(
                words=words,
                text=line,
                start=cursor,
                end=cursor + duration,
            )
        )
        cursor += duration
    return TranscriptDraft(segments=segments, provider="synthetic")


def _coerce_search_draft(draft_or_text: TranscriptDraft | str) -> TranscriptDraft:
    if isinstance(draft_or_text, TranscriptDraft):
        return draft_or_text
    return _build_synthetic_draft_from_text(draft_or_text)


_DIRECT_SITE_SEARCH_CACHE: dict[str, list[SearchResult]] = {}
_SITEMAP_URL_CACHE: dict[str, list[str]] = {}
_WEAK_CONTEXT_TOKENS = {"lyrics", "lyric", "song", "מילים", "שיר"}


def _canonicalize_lyrics_source_url(url: str) -> str:
    if not url:
        return ""

    normalized_url = url.strip()
    if normalized_url.startswith(("tabs/", "lyrics/")):
        normalized_url = urllib.parse.urljoin("https://www.tab4u.com/", normalized_url)

    parsed = urllib.parse.urlparse(normalized_url)
    if not parsed.netloc:
        return normalized_url

    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    path = parsed.path or "/"
    domain = netloc.lower().removeprefix("www.")

    if domain == "tab4u.com":
        path = path.replace("/tabs/songs/", "/lyrics/songs/")
        return urllib.parse.urlunparse((scheme, netloc, path, "", "", ""))

    if domain == "nagnu.co.il":
        path = re.sub(r"/אקורדים(?:/גרסה_קלה)?/?$", "", path)
        path = re.sub(r"/(פרשנות|איך_לנגן)/?$", "", path)
        return urllib.parse.urlunparse((scheme, netloc, path, "", "", ""))

    return urllib.parse.urlunparse((scheme, netloc, path, "", parsed.query, ""))


def _url_404_fallbacks(url: str) -> list[str]:
    """Alternate URLs to try when a canonicalized URL turns out to be a 404."""
    alternates = []
    if "tab4u.com" in url and "/lyrics/songs/" in url:
        alternates.append(url.replace("/lyrics/songs/", "/tabs/songs/"))
    return alternates


def _search_result_title_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    segments = [segment for segment in urllib.parse.unquote(parsed.path).split("/") if segment]
    if not segments:
        return url

    if parsed.netloc.lower().endswith("tab4u.com") and len(segments) >= 3:
        slug = html.unescape(segments[-1]).rsplit(".", 1)[0]
        slug = re.sub(r"^\d+_", "", slug)
        return slug.replace("_", " ").strip()

    if parsed.netloc.lower().endswith("nagnu.co.il") and len(segments) >= 3 and segments[-3] in NAGNU_ARTIST_SEGMENTS:
        artist = html.unescape(segments[-2]).replace("_", " ").strip()
        song = html.unescape(segments[-1]).replace("_", " ").strip()
        return f"{song} / {artist}".strip(" /")

    return html.unescape(segments[-1]).replace("_", " ").strip()


def _dedupe_search_results(results: list[SearchResult]) -> list[SearchResult]:
    unique: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        canonical_url = _canonicalize_lyrics_source_url(getattr(result, "url", ""))
        if not canonical_url or canonical_url in seen:
            continue
        seen.add(canonical_url)
        unique.append(
            SearchResult(
                title=getattr(result, "title", "") or _search_result_title_from_url(canonical_url),
                snippet=getattr(result, "snippet", ""),
                url=canonical_url,
            )
        )
    return unique


def _parse_tab4u_search_results(html_text: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    for href, label_html in re.findall(
        r'<a[^>]+ShowIFD\(this,\s*[\'"]song[\'"][^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html_text,
        re.S | re.I,
    ):
        absolute_url = urllib.parse.urljoin("https://www.tab4u.com/", href)
        canonical_url = _canonicalize_lyrics_source_url(absolute_url)
        label = _strip_html(label_html)
        results.append(
            SearchResult(
                title=label,
                snippet="Tab4U internal search",
                url=canonical_url,
            )
        )
    return _dedupe_search_results(results)


def _search_tab4u_results(query: str) -> list[SearchResult]:
    cache_key = f"tab4u:{query}"
    cached = _DIRECT_SITE_SEARCH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    last_results: list[SearchResult] = []
    for candidate_query in _relax_search_queries(query):
        encoded_query = urllib.parse.quote_plus(candidate_query)
        try:
            html_text = _fetch_text(
                f"https://www.tab4u.com/resultsSimple?tab=songs&q={encoded_query}",
                timeout=15,
            )
        except Exception as exc:
            logger.info("Tab4U internal search failed for %s: %s", candidate_query, exc)
            continue

        parsed_results = _parse_tab4u_search_results(html_text)
        if parsed_results:
            _DIRECT_SITE_SEARCH_CACHE[cache_key] = parsed_results
            return parsed_results
        last_results = parsed_results

    _DIRECT_SITE_SEARCH_CACHE[cache_key] = last_results
    return last_results


def _context_search_terms(context: dict[str, str]) -> tuple[set[str], set[str]]:
    song_tokens = {
        token
        for token in _top_keywords(context["song"], limit=8)
        if len(token) >= 3 and token not in STOPWORDS and token not in _WEAK_CONTEXT_TOKENS
    }
    artist_tokens = {
        token
        for token in (
            _top_keywords(context["hebrew_artist"], limit=6)
            + _top_keywords(context["latin_artist"], limit=6)
        )
        if len(token) >= 3 and token not in STOPWORDS and token not in _WEAK_CONTEXT_TOKENS
    }
    return song_tokens, artist_tokens


def _score_candidate_url(url: str, context: dict[str, str]) -> float:
    song_tokens, artist_tokens = _context_search_terms(context)
    if not song_tokens and not artist_tokens:
        return 0.0

    decoded_url = html.unescape(urllib.parse.unquote(url))
    parsed = urllib.parse.urlparse(decoded_url)
    path_text = parsed.path.replace("/", " ").replace("_", " ")
    path_tokens = set(_top_keywords(path_text, limit=24))

    song_overlap = path_tokens & song_tokens
    artist_overlap = path_tokens & artist_tokens

    if song_tokens and not song_overlap:
        return 0.0
    if artist_tokens and not artist_overlap:
        return 0.0
    if not song_overlap and not artist_overlap:
        return 0.0

    score = float(len(song_overlap) * 3 + len(artist_overlap) * 2)
    if artist_tokens and len(artist_overlap) >= min(2, len(artist_tokens)):
        score += 1.0
    lowered_path = parsed.path.lower()
    if "/lyrics/" in lowered_path or "track/lyrics" in lowered_path:
        score += 1.6
    if "/tabs/" in lowered_path:
        score += 0.6
    if "type=" in parsed.query:
        score -= 2.0
    if any(marker in decoded_url for marker in ("/פרשנות", "/איך_לנגן", "/אקורדים/גרסה_קלה")):
        score -= 1.5
    if song_tokens and len(song_overlap) == len(song_tokens):
        score += 1.2
    if artist_tokens and artist_overlap:
        score += 0.4
    return score


def _sanitize_internal_site_query(query: str) -> str:
    cleaned = re.sub(r"\bsite:[^\s]+\b", " ", query, flags=re.I)
    cleaned = re.sub(r"\b(?:lyrics|lyric|מילים|לשיר)\b", " ", cleaned, flags=re.I)
    cleaned = cleaned.replace('"', " ")
    cleaned = re.sub(r"[&|/+]+", " ", cleaned)
    return " ".join(cleaned.split())


def _load_tab4u_lyrics_urls() -> list[str]:
    cache_key = "tab4u:sitemap:lyrics"
    cached = _SITEMAP_URL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    urls: list[str] = []
    try:
        index_xml = _fetch_text("https://www.tab4u.com/sitemap.xml", timeout=25)
        sitemap_urls = [
            html.unescape(match)
            for match in re.findall(r"<loc>(.*?)</loc>", index_xml)
            if "x=songs" in html.unescape(match)
        ]
        for sitemap_url in sitemap_urls:
            try:
                xml_text = _fetch_text(sitemap_url, timeout=30)
            except Exception as exc:
                logger.info("Tab4U sitemap part failed for %s: %s", sitemap_url, exc)
                continue
            for raw_url in re.findall(r"<loc>(.*?)</loc>", xml_text):
                candidate_url = html.unescape(raw_url)
                if "/lyrics/songs/" not in candidate_url or "type=" in candidate_url:
                    continue
                urls.append(_canonicalize_lyrics_source_url(candidate_url))
    except Exception as exc:
        logger.info("Tab4U sitemap fetch failed: %s", exc)

    deduped_urls = list(dict.fromkeys(urls))
    _SITEMAP_URL_CACHE[cache_key] = deduped_urls
    return deduped_urls


def _load_nagnu_lyrics_urls() -> list[str]:
    cache_key = "nagnu:sitemap:lyrics"
    cached = _SITEMAP_URL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    urls: list[str] = []
    try:
        xml_text = _fetch_text("https://www.nagnu.co.il/sitemap.xml", timeout=35)
        for raw_url in re.findall(r"<loc>(.*?)</loc>", xml_text):
            candidate_url = _canonicalize_lyrics_source_url(html.unescape(raw_url))
            decoded_url = html.unescape(urllib.parse.unquote(candidate_url))
            if not any(f"/{segment}/" in decoded_url for segment in NAGNU_ARTIST_SEGMENTS):
                continue
            if any(marker in decoded_url for marker in ("/מורים_", "/פרשנות", "/איך_לנגן", "/אקורדים")):
                continue
            urls.append(candidate_url)
    except Exception as exc:
        logger.info("Nagnu sitemap fetch failed: %s", exc)

    deduped_urls = list(dict.fromkeys(urls))
    _SITEMAP_URL_CACHE[cache_key] = deduped_urls
    return deduped_urls


def _search_sitemap_results(
    context: dict[str, str],
    url_loader,
    *,
    source_name: str,
    max_results: int = 4,
) -> list[SearchResult]:
    scored_results: list[tuple[float, SearchResult]] = []
    for candidate_url in url_loader():
        score = _score_candidate_url(candidate_url, context)
        if score < 3.0:
            continue
        scored_results.append(
            (
                score,
                SearchResult(
                    title=_search_result_title_from_url(candidate_url),
                    snippet=f"{source_name} sitemap",
                    url=candidate_url,
                ),
            )
        )

    scored_results.sort(key=lambda item: item[0], reverse=True)
    return _dedupe_search_results([result for _score, result in scored_results[:max_results]])


def _search_known_site_results(
    title: str,
    queries: list[str],
    context: dict[str, str],
) -> list[SearchResult]:
    del title

    preferred_queries = [
        f"{context['hebrew_artist']} {context['song']}".strip(),
        context["song"].strip(),
        context["clean_title"].strip(),
        f"{context['latin_artist']} {context['song']}".strip(),
    ]
    preferred_queries.extend(
        _sanitize_internal_site_query(query)
        for query in queries[:3]
        if _sanitize_internal_site_query(query)
    )
    candidate_queries = [query for query in dict.fromkeys(preferred_queries) if query]

    results: list[SearchResult] = []
    for query in candidate_queries[:4]:
        for result in _search_tab4u_results(query):
            candidate = SearchResult(
                title=getattr(result, "title", "") or _search_result_title_from_url(result.url),
                snippet=getattr(result, "snippet", ""),
                url=result.url,
            )
            if _matches_title_context(candidate, context) or _score_candidate_url(result.url, context) >= 5.0:
                results.append(result)

    results.extend(
        _search_sitemap_results(
            context,
            _load_tab4u_lyrics_urls,
            source_name="Tab4U",
            max_results=4,
        )
    )
    results.extend(
        _search_sitemap_results(
            context,
            _load_nagnu_lyrics_urls,
            source_name="Nagnu",
            max_results=4,
        )
    )

    filtered_results: list[SearchResult] = []
    for result in _dedupe_search_results(results):
        candidate = SearchResult(
            title=getattr(result, "title", "") or _search_result_title_from_url(result.url),
            snippet=getattr(result, "snippet", ""),
            url=result.url,
        )
        if _matches_title_context(candidate, context) or _score_candidate_url(result.url, context) >= 5.0:
            filtered_results.append(result)

    deduped = _dedupe_search_results(filtered_results)
    deduped.sort(
        key=lambda result: (
            _domain_priority(result.url),
            -_score_candidate_url(result.url, context),
            result.url,
        )
    )
    return deduped[:8]


def _repair_draft_from_candidate_lines(
    draft_lines: list[str],
    candidate_lines: list[str],
) -> tuple[list[str], int, float]:
    normalized_draft = [line.strip() for line in draft_lines if line.strip()]
    normalized_candidates = [line.strip() for line in candidate_lines if line.strip()]
    if not normalized_draft or not normalized_candidates:
        return [], 0, 0.0

    min_window_lines = max(1, min(len(normalized_draft), max(1, len(normalized_draft) // 2)))
    max_window_lines = max(min_window_lines, len(normalized_draft) + 6)
    best_lines: list[str] = []
    best_corrections = 0
    best_score = 0.0

    def _consider_window(lines_window: list[str]) -> None:
        nonlocal best_lines, best_corrections, best_score
        expanded_lines = _expand_repeated_candidate_lines(normalized_draft, lines_window)
        repaired_lines, correction_count = _repair_draft_lines_from_source_lines(normalized_draft, expanded_lines)
        paired_scores = [
            max(_line_similarity(left, right), _line_token_overlap(left, right))
            for left, right in zip(normalized_draft, repaired_lines)
        ]
        average_score = sum(paired_scores) / max(1, len(paired_scores))
        line_coverage = min(len(lines_window), len(normalized_draft)) / max(
            len(lines_window),
            len(normalized_draft),
            1,
        )
        combined_score = average_score * 0.88 + line_coverage * 0.12
        if correction_count <= 0 and combined_score < 0.72:
            return
        if combined_score > best_score:
            best_lines = repaired_lines
            best_corrections = correction_count
            best_score = combined_score

    _consider_window(normalized_candidates)

    if len(normalized_candidates) > len(normalized_draft) + 2:
        for start in range(len(normalized_candidates)):
            max_end = min(len(normalized_candidates), start + max_window_lines)
            for end in range(start + min_window_lines, max_end + 1):
                _consider_window(normalized_candidates[start:end])

    return best_lines, best_corrections, best_score


def _clean_tab4u_lyrics_html(page_html: str) -> str | None:
    lyrics_block: str | None = None

    match = re.search(r'id=["\']songContentTPL["\'][^>]*>(.*?)</div', page_html, re.S | re.I)
    if match:
        lyrics_block = match.group(1)

    if not lyrics_block:
        match = re.search(r'id=["\']songLyricsDiv["\'][^>]*>(.*?)</div', page_html, re.S | re.I)
        if match:
            lyrics_block = match.group(1)

    if not lyrics_block:
        song_cells = re.findall(
            r'<td[^>]+class=["\'][^"\']*\bsong\b[^"\']*["\'][^>]*>(.*?)</td>',
            page_html,
            re.S | re.I,
        )
        if song_cells:
            lyrics_block = "<br>".join(song_cells)

    if not lyrics_block:
        return None

    filtered_lines: list[str] = []
    cleaned_block = _strip_html_preserving_lines(lyrics_block)
    for raw_line in cleaned_block.splitlines():
        line = raw_line.replace("\xa0", " ").strip(" \t-")
        if not line:
            continue
        signature = _line_signature(line)
        if not signature:
            continue
        if re.fullmatch(r"[A-G][#bm0-9*+/()' -]{0,12}:\s*[x0-9]{3,}", line, re.I):
            continue
        if re.fullmatch(r"[A-G][#bm0-9*+/()' -]{1,20}", line, re.I):
            continue
        if signature in {"פתיחה", "סיום", "מעבר", "אינטרו"}:
            continue
        if "אקורדים לשיר" in line or "מילים לשיר" in line:
            continue
        filtered_lines.append(line)

    if filtered_lines:
        return "<br>".join(filtered_lines)
    return lyrics_block


def _extract_site_specific_lyrics_base(url: str, page_html: str) -> str | None:
    """חילוץ ממוקד לאתרים ידועים לפי CSS selectors אופייניים."""
    domain = _domain_from_url(url)

    if "shironet" in domain:
        match = re.search(r'id=["\']artist_lyrics(?:_text)?["\'][^>]*>(.*?)</div', page_html, re.S | re.I)
        if match:
            return match.group(1)

    if "tab4u" in domain:
        match = re.search(r'id=["\']songContentTPL["\'][^>]*>(.*?)</div', page_html, re.S | re.I)
        if match:
            return match.group(1)
        song_cells = re.findall(
            r'<td[^>]+class=["\'][^"\']*\bsong\b[^"\']*["\'][^>]*>(.*?)</td>',
            page_html,
            re.S | re.I,
        )
        if song_cells:
            return "<br>".join(song_cells)
        match = re.search(r'id=["\']songLyricsDiv["\'][^>]*>(.*?)</div', page_html, re.S | re.I)
        if match:
            return match.group(1)

    if "nagnu" in domain:
        if 'q:route="track/lyrics/' in page_html:
            return page_html
        match = re.search(r'class=["\'][^"\']*lyrics[^"\']*["\'][^>]*>(.*?)</(?:div|pre)', page_html, re.S | re.I)
        if match:
            return match.group(1)

    # shirrim.com – lyrics appear right after the "המילים של השיר:" text heading
    # Actual HTML: ">המילים של השיר: <br>\n<p>line1<br />line2..."
    if "shirrim" in domain:
        match = re.search(
            r'המילים של השיר'
            r'[^<]*(?:<[^>]+>\s*){1,3}<p>(.*?)</p>',
            page_html,
            re.S | re.I,
        )
        if match:
            return match.group(1)

    # nomorelyrics.net – lyrics inside element with id/class "songtext"
    if "nomorelyrics" in domain:
        match = re.search(r'(?:id|class)=["\']songtext["\'][^>]*>(.*?)</div', page_html, re.S | re.I)
        if match:
            return match.group(1)

    # baneshama.co.il – Hebrew lyrics site
    if "baneshama" in domain:
        match = re.search(r'class=["\'][^"\']*lyrics[^"\']*["\'][^>]*>(.*?)</(?:div|pre)', page_html, re.S | re.I)
        if match:
            return match.group(1)
        match = re.search(r'class=["\'][^"\']*song[_-]?text[^"\']*["\'][^>]*>(.*?)</(?:div|pre)', page_html, re.S | re.I)
        if match:
            return match.group(1)

    # nagina.co.il – Nagina Mizrahit, Mizrahi/Eastern music lyrics site
    # Guard against matching nagnu.co.il which has its own parser above
    if "nagina" in domain and "nagnu" not in domain:
        match = re.search(r'class=["\'][^"\']*lyrics[^"\']*["\'][^>]*>(.*?)</(?:div|pre)', page_html, re.S | re.I)
        if match:
            return match.group(1)
        match = re.search(r'class=["\'][^"\']*song[_-]?content[^"\']*["\'][^>]*>(.*?)</(?:div|pre)', page_html, re.S | re.I)
        if match:
            return match.group(1)

    return None


def _extract_site_specific_lyrics(url: str, page_html: str) -> str | None:
    domain = _domain_from_url(url)
    if "tab4u" in domain:
        cleaned_tab4u = _clean_tab4u_lyrics_html(page_html)
        if cleaned_tab4u:
            return cleaned_tab4u
    return _extract_site_specific_lyrics_base(url, page_html)


def _build_query_variants(title: str, draft_text: str) -> list[str]:
    context = _extract_title_context(title)
    clean_title = context["clean_title"]
    hebrew_artist = context["hebrew_artist"]
    latin_artist = context["latin_artist"]
    song = context["song"]
    keywords = _top_keywords(draft_text, limit=8)
    keyword_tail = " ".join(keywords[:4])
    lyric_snippets = _draft_search_snippets(draft_text, limit=3)
    hebrew_suffix = HEBREW_LYRICS_QUERY

    queries: list[str] = []
    if hebrew_artist and song:
        queries.extend(
            [
                f"{hebrew_artist} {song} {hebrew_suffix}",
                f"{song} {hebrew_artist} {hebrew_suffix}",
                f"\"{hebrew_artist}\" \"{song}\" {hebrew_suffix}",
                # Shironet is the most authoritative Hebrew source but is only
                # reachable via search engines - keep it inside the first
                # queries the providers actually run (queries[:5]).
                f"site:shironet.mako.co.il {hebrew_artist} {song}",
            ]
        )
    if latin_artist and song:
        queries.extend(
            [
                f"\"{latin_artist}\" \"{song}\" lyrics",
                f"{latin_artist} {song} lyrics",
            ]
        )
    if clean_title:
        queries.append(f"{clean_title} {hebrew_suffix}")
        queries.append(f"{clean_title} lyrics")
    if song:
        queries.append(f"\"{song}\" {hebrew_suffix}")
        queries.append(f"\"{song}\" lyrics")
    for snippet in lyric_snippets[:2]:
        if song:
            queries.append(f"\"{song}\" \"{snippet}\" {hebrew_suffix}")
        queries.append(f"\"{snippet}\" {hebrew_suffix}")
    for domain in SITE_QUERY_DOMAINS:
        suffix = "lyrics" if domain in {"genius.com", "lyricstranslate.com"} else hebrew_suffix
        if hebrew_artist and song:
            queries.append(f"site:{domain} \"{hebrew_artist}\" \"{song}\" {suffix}")
        if latin_artist and song:
            queries.append(f"site:{domain} \"{latin_artist}\" \"{song}\" {suffix}")
        for snippet in lyric_snippets[:1]:
            queries.append(f"site:{domain} \"{snippet}\" {suffix}")
    if clean_title and keyword_tail:
        queries.append(f"{clean_title} {keyword_tail}")
    elif keyword_tail:
        queries.append(f"{keyword_tail} {hebrew_suffix}")

    if keyword_tail:
        queries.append(f"{HEBREW_LYRICS_FOR_SONG_QUERY} {keyword_tail}")

    return list(dict.fromkeys(query for query in queries if query.strip()))


def _find_best_source_line_window(
    candidate_text: str,
    draft: TranscriptDraft,
) -> tuple[list[str], float]:
    draft_tokens = [_normalize_token(token) for token in _tokenize_words(draft.text)]
    if not draft_tokens:
        return [], 0.0

    draft_lines = _draft_lines(draft)
    extracted_entries = _extract_candidate_lyrics_line_entries(candidate_text)
    if not extracted_entries:
        extracted_entries = _extract_candidate_lyrics_line_entries(candidate_text, draft_lines)
    if not extracted_entries:
        return [], 0.0

    draft_token_set = set(draft_tokens)
    target_line_count = max(1, len(draft_lines))
    target_word_count = max(1, len(draft_tokens))
    min_window_lines = 1 if target_line_count <= 2 else max(2, target_line_count // 2)
    max_window_lines = max(min_window_lines, target_line_count + 6)

    best_lines: list[str] = []
    best_score = 0.0

    def _score_window(candidate_entries: list[tuple[str, list[str]]]) -> float:
        normalized_tokens = [
            _normalize_token(token)
            for _line, tokens in candidate_entries
            for token in tokens
        ]
        if not normalized_tokens:
            return 0.0

        ratio = SequenceMatcher(None, draft_tokens, normalized_tokens, autojunk=False).ratio()
        overlap = len(draft_token_set & set(normalized_tokens)) / max(
            1,
            min(len(draft_token_set), len(set(normalized_tokens))),
        )
        line_coverage = min(len(candidate_entries), target_line_count) / max(
            len(candidate_entries),
            target_line_count,
            1,
        )
        word_coverage = min(len(normalized_tokens), target_word_count) / max(
            len(normalized_tokens),
            target_word_count,
            1,
        )
        return ratio * 0.52 + overlap * 0.23 + line_coverage * 0.15 + word_coverage * 0.10

    def _consider(candidate_entries: list[tuple[str, list[str]]]) -> None:
        nonlocal best_lines, best_score
        if not candidate_entries:
            return
        candidate_lines = [line for line, _tokens in candidate_entries]
        score = _score_window(candidate_entries)
        # Near-ties go to the window whose length matches the sung draft,
        # not to the longer one (verbose pages full of credits/chords used
        # to beat the correct-but-short lyric block).
        if score > best_score or (
            abs(score - best_score) <= 0.03
            and abs(len(candidate_lines) - target_line_count)
            < abs(len(best_lines) - target_line_count)
        ):
            best_lines = candidate_lines
            best_score = score

    _consider(extracted_entries)

    if len(extracted_entries) > max_window_lines:
        for start in range(len(extracted_entries)):
            collected: list[tuple[str, list[str]]] = []
            max_end = min(len(extracted_entries), start + max_window_lines)
            for end in range(start, max_end):
                collected.append(extracted_entries[end])
                if len(collected) < min_window_lines:
                    continue

                token_count = sum(len(tokens) for _line, tokens in collected)
                if token_count < max(4, int(target_word_count * 0.35)):
                    continue
                if token_count > int(target_word_count * 1.9) + 12:
                    break

                _consider(collected)

    full_lines = [line for line, _tokens in extracted_entries]
    full_score = _score_window(extracted_entries)
    full_extends_best = (
        len(full_lines) > len(best_lines) > 0
        and all(
            max(_line_similarity(full_lines[index], best_line), _line_token_overlap(full_lines[index], best_line)) >= 0.88
            for index, best_line in enumerate(best_lines)
        )
    )
    if (
        full_lines
        and len(full_lines) > len(best_lines)
        and (
            full_score >= max(0.34, best_score - 0.06)
            or (full_extends_best and full_score >= max(0.26, best_score - 0.18))
        )
    ):
        return full_lines, full_score

    return best_lines, best_score


def _line_block_quality(lines: list[str]) -> tuple[int, int, int]:
    normalized_lines = [str(line).strip() for line in lines if str(line).strip()]
    line_count = len(normalized_lines)
    substantial_count = sum(1 for line in normalized_lines if _is_substantial_lyrics_line(line))
    total_words = sum(len(_line_signature(line).split()) for line in normalized_lines)
    return line_count, substantial_count, total_words


def _lines_are_near_duplicates(left: str, right: str) -> bool:
    if not left or not right:
        return False
    similarity = _line_similarity(left, right)
    overlap = _line_token_overlap(left, right)
    return max(similarity, overlap) >= 0.92


def _append_unique_lyrics_line(lines: list[str], line: str) -> None:
    normalized_line = str(line).strip()
    if not normalized_line:
        return
    if lines and _lines_are_near_duplicates(lines[-1], normalized_line):
        return
    lines.append(normalized_line)


def _line_blocks_are_related(base_block: list[str], candidate_block: list[str]) -> bool:
    if not base_block or not candidate_block:
        return False
    similarities = [
        max(_line_similarity(left, right), _line_token_overlap(left, right))
        for left in base_block
        for right in candidate_block
    ]
    return max(similarities, default=0.0) >= 0.46


def _choose_richer_line_block(base_block: list[str], candidate_block: list[str]) -> list[str]:
    if _line_block_quality(candidate_block) > _line_block_quality(base_block):
        return candidate_block
    return base_block


def _merge_source_line_sequences(base_lines: list[str], candidate_lines: list[str]) -> list[str]:
    normalized_base = [str(line).strip() for line in base_lines if str(line).strip()]
    normalized_candidate = [str(line).strip() for line in candidate_lines if str(line).strip()]
    if not normalized_base:
        return normalized_candidate
    if not normalized_candidate:
        return normalized_base

    base_signatures = [_line_signature(line) for line in normalized_base]
    candidate_signatures = [_line_signature(line) for line in normalized_candidate]
    merged: list[str] = []

    matcher = SequenceMatcher(None, base_signatures, candidate_signatures, autojunk=False)
    for tag, b0, b1, c0, c1 in matcher.get_opcodes():
        if tag == "equal":
            for line in normalized_base[b0:b1]:
                _append_unique_lyrics_line(merged, line)
            continue

        if tag == "insert":
            previous_line = merged[-1] if merged else ""
            next_line = normalized_base[b0] if b0 < len(normalized_base) else ""
            for line in normalized_candidate[c0:c1]:
                if _lines_are_near_duplicates(line, previous_line) or _lines_are_near_duplicates(line, next_line):
                    continue
                _append_unique_lyrics_line(merged, line)
            continue

        if tag == "delete":
            for line in normalized_base[b0:b1]:
                _append_unique_lyrics_line(merged, line)
            continue

        if tag == "replace":
            base_block = normalized_base[b0:b1]
            candidate_block = normalized_candidate[c0:c1]
            chosen_block = base_block
            if _line_blocks_are_related(base_block, candidate_block):
                chosen_block = _choose_richer_line_block(base_block, candidate_block)
            elif _line_block_quality(candidate_block) > _line_block_quality(base_block) and len(base_block) <= 1:
                chosen_block = candidate_block
            for line in chosen_block:
                _append_unique_lyrics_line(merged, line)

    return merged or normalized_base


def _merge_search_source_versions(
    sources: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    normalized_sources = [
        (source_name, [str(line).strip() for line in lines if str(line).strip()])
        for source_name, lines in sources.items()
        if lines
    ]
    if not normalized_sources:
        return [], []

    def _domain_rank(source_name: str) -> int:
        for domain, rank in DOMAIN_PRIORITY.items():
            if source_name == domain or source_name.endswith(domain):
                return rank
        return 50

    # Authoritative domains lead the merge; raw block length only breaks ties
    # within the same priority tier (length used to dominate, biasing toward
    # verbose pages).
    normalized_sources.sort(
        key=lambda item: (-_domain_rank(item[0]), _line_block_quality(item[1]), item[0]),
        reverse=True,
    )

    merged_lines = list(normalized_sources[0][1])
    supporting_sources = [normalized_sources[0][0]]

    for source_name, lines in normalized_sources[1:]:
        merged_lines = _merge_source_line_sequences(merged_lines, lines)
        supporting_sources.append(source_name)

    return merged_lines, list(dict.fromkeys(supporting_sources))


def _build_search_merged_option(
    merged_lines: list[str],
    supporting_sources: list[str],
    *,
    confidence: float,
    reference_lines: list[str] | None = None,
) -> dict[str, object] | None:
    normalized_merged = [str(line).strip() for line in merged_lines if str(line).strip()]
    if not normalized_merged:
        return None
    if reference_lines is not None and not _line_sets_differ(reference_lines, normalized_merged):
        return None
    unique_sources = list(dict.fromkeys(supporting_sources))
    option: dict[str, object] = {
        "option_id": "search_merged",
        "label": "תוצאת חיפוש משולבת",
        "lines": normalized_merged,
        "source_url": "",
        "confidence": round(confidence, 3),
        "source_count": len(unique_sources),
    }
    if unique_sources:
        option["supporting_sources"] = unique_sources
    return option


def _evaluate_candidate_text_against_draft(
    draft: TranscriptDraft,
    candidate_text: str,
) -> tuple[list[str], float]:
    candidate_lines, candidate_score = _find_best_source_line_window(candidate_text, draft)
    if candidate_lines and candidate_score >= 0.32:
        return candidate_lines, candidate_score
    return [], 0.0


def _extract_youtube_source_lines(
    source_url: str,
    title: str,
    draft: TranscriptDraft,
) -> tuple[list[str], float]:
    if not source_url:
        return [], 0.0

    source_domain = _domain_from_url(source_url)
    if not any(
        source_domain.endswith(candidate)
        for candidate in ("youtube.com", "youtu.be", "music.youtube.com")
    ):
        return [], 0.0

    try:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "skip_download": True,
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_url, download=False)
    except Exception as exc:
        logger.info("Direct YouTube lyrics fetch failed for %s: %s", source_url, exc)
        return [], 0.0

    if not isinstance(info, dict):
        return [], 0.0

    description = str(info.get("description") or "").strip()
    if not description or not re.search(r"[\u0590-\u05FF]", description):
        return [], 0.0
    cleaned_description = _clean_youtube_description_text(description)
    if cleaned_description:
        description = cleaned_description

    context = _extract_title_context(title)
    candidate = SearchResult(
        title=str(info.get("title") or title).strip(),
        snippet=description[:200],
        url=source_url,
    )
    if not _matches_title_context(candidate, context):
        return [], 0.0

    candidate_lines, candidate_score = _evaluate_candidate_text_against_draft(draft, description)
    if not candidate_lines:
        return [], 0.0

    cleaned_lines = _trim_youtube_credit_edges(candidate_lines)
    if not cleaned_lines:
        return [], 0.0

    return cleaned_lines, candidate_score


def _search_all_sources_impl(
    self,
    title: str,
    draft_or_text: TranscriptDraft | str,
) -> dict[str, list[str]]:
    import concurrent.futures

    draft = _coerce_search_draft(draft_or_text)
    queries = _build_query_variants(title, draft.text)
    context = _extract_title_context(title)
    source_url = getattr(self, "_current_source_url", "")
    sources: dict[str, tuple[float, list[str]]] = {}
    urls_by_domain: dict[str, list[str]] = {}
    search_warnings: list[str] = []
    self._last_search_warnings = search_warnings
    google_failed = False

    def _remember_result(result) -> None:
        url = _canonicalize_lyrics_source_url(getattr(result, "url", ""))
        if not url:
            return

        domain = _domain_from_url(url)
        if not any(domain.endswith(candidate) for candidate in KNOWN_LYRICS_DOMAINS):
            return

        candidate = SearchResult(
            title=getattr(result, "title", "") or _search_result_title_from_url(url),
            snippet=getattr(result, "snippet", ""),
            url=url,
        )
        if not _looks_like_lyrics_result(candidate) or not _matches_title_context(candidate, context):
            return

        domain_urls = urls_by_domain.setdefault(domain, [])
        if url not in domain_urls and len(domain_urls) < 3:
            domain_urls.append(url)

    source_domain = _domain_from_url(_canonicalize_lyrics_source_url(source_url))
    if source_url and any(source_domain.endswith(candidate) for candidate in KNOWN_LYRICS_DOMAINS):
        urls_by_domain.setdefault(source_domain, []).append(_canonicalize_lyrics_source_url(source_url))

    youtube_lines, youtube_score = _extract_youtube_source_lines(source_url, title, draft)
    if youtube_lines:
        sources["youtube"] = (youtube_score, youtube_lines)
        if youtube_score >= 0.65:
            return {"youtube": youtube_lines}

    for query in queries[:5]:
        try:
            results = self._google.search(query, num=10)
            for result in results:
                _remember_result(result)
        except GoogleSearchQuotaError:
            google_failed = True
            break
        except Exception as exc:
            logger.warning("Google search failed for query '%s': %s", query, exc)

    if google_failed or len(urls_by_domain) < 2:
        for query in queries[:5]:
            try:
                results = _search_web_results(query)
                for result in results:
                    _remember_result(result)
            except Exception as exc:
                logger.warning("Web fallback search failed: %s", exc)

    if len(urls_by_domain) < 2:
        for result in _search_known_site_results(title, queries, context):
            _remember_result(result)

    def _fetch_and_parse(url: str) -> tuple[str, list[str], float]:
        source_key = _domain_option_key(url)
        try:
            try:
                html_body = _fetch_text(url, timeout=15)
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                # Canonicalization rewrites URLs blindly (tab4u tabs->lyrics);
                # when the rewritten page does not exist, try the original form.
                html_body = ""
                for alternate_url in _url_404_fallbacks(url):
                    try:
                        html_body = _fetch_text(alternate_url, timeout=15)
                        break
                    except Exception:
                        continue
                if not html_body:
                    raise
            if _is_bot_blocked(html_body):
                if "shironet" in source_key:
                    search_warnings.append(
                        "שירונט חסם גישה אוטומטית, אז האימות רץ בלי המקור הזה."
                    )
                return source_key, [], 0.0

            specific = _extract_site_specific_lyrics(url, html_body)
            effective_html = specific if specific else html_body
            candidate_lines, candidate_score = _evaluate_candidate_text_against_draft(draft, effective_html)
            if candidate_lines:
                return source_key, candidate_lines, candidate_score
        except Exception as exc:
            logger.warning("Failed to fetch lyrics from %s: %s", url, exc)
        return source_key, [], 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_and_parse, url): url
            for domain_urls in urls_by_domain.values()
            for url in domain_urls
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                key, lines, score = future.result()
                if lines and (key not in sources or score > sources[key][0]):
                    sources[key] = (score, lines)
            except Exception as exc:
                logger.warning("Source fetch error: %s", exc)

    try:
        yt_results = self._youtube.search(f"{title} {HEBREW_LYRICS_QUERY}", max_results=3)
        for yt_result in yt_results:
            desc = _clean_youtube_description_text(yt_result.snippet)
            if desc and re.search(r"[\u0590-\u05FF]", desc):
                candidate_lines, candidate_score = _evaluate_candidate_text_against_draft(draft, desc)
                cleaned_lines = _trim_youtube_credit_edges(candidate_lines)
                if cleaned_lines and candidate_score > sources.get("youtube", (0.0, []))[0]:
                    sources["youtube"] = (candidate_score, cleaned_lines)
                    break
    except Exception as exc:
        logger.warning("YouTube search failed: %s", exc)

    return {source_name: lines for source_name, (_score, lines) in sources.items()}


SOURCE_DISPLAY_NAMES.setdefault("youtube", "YouTube")
