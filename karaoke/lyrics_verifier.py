"""Multi-source lyrics verification before human review."""

from __future__ import annotations

import html
import json
import logging
import re
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from dataclasses import dataclass
from difflib import SequenceMatcher

from .models import LyricsVerificationResult, TranscriptDraft

logger = logging.getLogger(__name__)

DUCKDUCKGO_HTML_SEARCH = "https://html.duckduckgo.com/html/?q={query}"
DUCKDUCKGO_HTML_FALLBACK = "https://duckduckgo.com/html/?q={query}"
SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.7",
}
KNOWN_LYRICS_DOMAINS = (
    "shironet.mako.co.il",
    "nagnu.co.il",
    "tab4u.com",
    "lyricstranslate.com",
    "nli.org.il",
    "shirrim.com",
    "genius.com",
    "nomorelyrics.net",
)
SITE_QUERY_DOMAINS = (
    "shironet.mako.co.il",
    "nagnu.co.il",
    "tab4u.com",
    "lyricstranslate.com",
    "nli.org.il",
    "genius.com",
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

TITLE_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.S)
SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.S)
SEARCH_RESULT_CACHE: dict[str, list["SearchResult"]] = {}


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


def _top_keywords(text: str, limit: int = 10) -> list[str]:
    counts: dict[str, int] = {}
    for token in _tokenize(text):
        if token in STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [token for token, _count in ranked[:limit]]


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


def _build_query_variants(title: str, draft_text: str) -> list[str]:
    context = _extract_title_context(title)
    clean_title = context["clean_title"]
    hebrew_artist = context["hebrew_artist"]
    latin_artist = context["latin_artist"]
    song = context["song"]
    keywords = _top_keywords(draft_text, limit=8)
    keyword_tail = " ".join(keywords[:4])

    queries = []
    if hebrew_artist and song:
        queries.extend(
            [
                f"{hebrew_artist} {song} מילים",
                f"{song} {hebrew_artist} מילים",
                f"\"{hebrew_artist}\" \"{song}\" מילים",
            ]
        )
    if latin_artist and song:
        queries.extend(
            [
                f"\"{latin_artist}\" \"{song}\" lyrics",
                f"{latin_artist} {song} lyrics",
            ]
        )
    for domain in SITE_QUERY_DOMAINS:
        suffix = "lyrics" if domain in {"genius.com", "lyricstranslate.com"} else "מילים"
        if hebrew_artist and song:
            queries.append(f"site:{domain} \"{hebrew_artist}\" \"{song}\" {suffix}")
        if latin_artist and song:
            queries.append(f"site:{domain} \"{latin_artist}\" \"{song}\" {suffix}")
    if clean_title:
        queries.append(f"{clean_title} מילים")
        queries.append(f"{clean_title} lyrics")
        if keyword_tail:
            queries.append(f"{clean_title} {keyword_tail}")
    elif keyword_tail:
        queries.append(f"{keyword_tail} מילים")

    if keyword_tail:
        queries.append(f"מילים לשיר {keyword_tail}")

    return list(dict.fromkeys(query for query in queries if query.strip()))


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


def _extract_site_specific_lyrics(url: str, page_html: str) -> str | None:
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
            r'\u05d4\u05de\u05d9\u05dc\u05d9\u05dd \u05e9\u05dc \u05d4\u05e9\u05d9\u05e8'
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

    return None


def _find_best_lyrics_window(page_html: str, draft: TranscriptDraft) -> tuple[float, list[str], int]:
    draft_tokens = [_normalize_token(token) for token in _tokenize_words(draft.text)]
    if not draft_tokens:
        return 0.0, [], 0

    cleaned = _strip_html_preserving_lines(page_html)
    raw_lines: list[list[str]] = []
    for line in cleaned.splitlines():
        tokens = _tokenize_words(line)
        hebrew_tokens = [token for token in tokens if re.search(r"[\u0590-\u05FF]", token)]
        if len(hebrew_tokens) >= 2:
            raw_lines.append(hebrew_tokens)

    if not raw_lines:
        return 0.0, [], 0

    draft_token_set = set(draft_tokens)
    target_words = len(draft_tokens)
    target_line_count = max(1, len([seg for seg in draft.segments if seg.text.strip()]))
    min_window_lines = max(1, target_line_count - 3)
    max_window_lines = max(min_window_lines, target_line_count + 4)

    def _score_window(candidate_tokens: list[str]) -> float:
        normalized = [_normalize_token(t) for t in candidate_tokens]
        ratio = SequenceMatcher(None, draft_tokens, normalized, autojunk=False).ratio()
        overlap = len(draft_token_set & set(normalized)) / max(1, len(draft_token_set))
        return ratio * 0.72 + overlap * 0.28

    # Fast path: source page has roughly the same number of lines as the draft
    if min_window_lines <= len(raw_lines) <= max_window_lines:
        flattened = [t for line in raw_lines for t in line]
        score = _score_window(flattened)
        if score >= 0.45:
            corrected_lines, correction_count = _align_source_to_segments(flattened, draft)
            return score, corrected_lines, correction_count

    # Sliding-window search: find the source sub-section that best matches the draft
    best_score = 0.0
    best_tokens: list[str] = []

    for start in range(len(raw_lines)):
        collected: list[str] = []
        max_end = min(len(raw_lines), start + max_window_lines)
        for end in range(start, max_end):
            collected.extend(raw_lines[end])
            window_line_count = end - start + 1
            if len(collected) > int(target_words * 1.5) + 8:
                break
            if window_line_count < min_window_lines:
                continue
            if len(collected) < max(4, int(target_words * 0.55)):
                continue

            score = _score_window(collected)
            if score > best_score:
                best_score = score
                best_tokens = collected[:]

    if not best_tokens:
        return 0.0, [], 0

    corrected_lines, correction_count = _align_source_to_segments(best_tokens, draft)
    return best_score, corrected_lines, correction_count


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


def _fetch_text(url: str, timeout: int = 12) -> str:
    encoded_url = _encode_url(url)
    request = urllib.request.Request(encoded_url, headers=SEARCH_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


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


def _search_duckduckgo_results(query: str) -> list[SearchResult]:
    cached = SEARCH_RESULT_CACHE.get(query)
    if cached is not None:
        return cached

    encoded_query = urllib.parse.quote_plus(query)
    last_error: Exception | None = None
    for endpoint in (DUCKDUCKGO_HTML_SEARCH, DUCKDUCKGO_HTML_FALLBACK):
        try:
            html_text = _fetch_text(endpoint.format(query=encoded_query))
            results = _parse_duckduckgo_results(html_text)
            if results:
                SEARCH_RESULT_CACHE[query] = results
                return results
        except HTTPError as exc:
            last_error = exc
            logger.info("DuckDuckGo search endpoint failed for %s: %s", query, exc)
            continue
        except Exception as exc:
            last_error = exc
            logger.info("DuckDuckGo search parse failed for %s: %s", query, exc)
            continue

    if last_error:
        logger.info("All DuckDuckGo search endpoints failed for %s: %s", query, last_error)
    return []


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
) -> LyricsVerificationResult:
    draft_lines = _draft_lines(draft)
    return LyricsVerificationResult(
        provider=provider,
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


def _line_similarity(left: str, right: str) -> float:
    left_tokens = [_normalize_token(token) for token in _tokenize_words(left)]
    right_tokens = [_normalize_token(token) for token in _tokenize_words(right)]
    if not left_tokens or not right_tokens:
        return 0.0
    return SequenceMatcher(None, left_tokens, right_tokens, autojunk=False).ratio()


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
                },
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


def _call_gemini(prompt: str, api_key: str, model: str, timeout: int = 25) -> str:
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
            if exc.code == 429 and attempt == 0:
                logger.info("Gemini rate limited, retrying in 5 seconds...")
                time.sleep(5)
                continue
            raise
        except Exception:
            raise
    raise last_error  # type: ignore[misc]


class GeminiLyricsVerifier:
    """Lyrics verifier using Google Gemini AI to find and correct song lyrics."""

    name = "gemini_lyrics_verifier"

    def __init__(self, api_key: str = "", model: str = "", fallback_verifier: DuckDuckGoLyricsVerifier | None = None):
        from .config import GEMINI_API_KEY, GEMINI_MODEL

        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self._fallback_verifier = fallback_verifier

    def _fallback_or_draft(self, title: str, draft: TranscriptDraft, summary: str, warnings: list[str]) -> LyricsVerificationResult:
        if self._fallback_verifier is not None:
            return self._fallback_verifier.verify(title, draft)
        return _build_draft_only_result(
            provider=self.name,
            draft=draft,
            warnings=warnings,
            summary=summary,
            verdict="mismatch",
            search_query=f"Gemini: {title}",
        )

    def _parse_gemini_lines(self, raw_text: str) -> list[str]:
        """Parse Gemini's response into clean lyrics lines.

        Strips any "THOUGHTS:" / reasoning prefix that some Gemini models
        (e.g. gemini-2.5-flash) include before the actual lyrics.
        """
        text = raw_text.strip()

        # Strip thinking block — everything before the actual lyrics.
        # Gemini 2.5-flash sometimes prefixes with "THOUGHTS: ..." or
        # "(Self-correction: ...)" blocks before the lyrics.
        for marker in ("THOUGHTS:", "Thoughts:", "thoughts:"):
            if text.startswith(marker):
                # Find where lyrics actually begin — first line with Hebrew
                for i, line in enumerate(text.splitlines()):
                    stripped = line.strip()
                    if stripped and re.search(r"[\u0590-\u05FF]", stripped):
                        # Check it's not part of the thinking (no English preamble)
                        if not re.match(r"^[\(\[]", stripped):
                            text = "\n".join(text.splitlines()[i:])
                            break
                break

        lines = []
        in_thinking = False
        for line in text.splitlines():
            cleaned = line.strip()
            # Skip empty lines
            if not cleaned:
                continue
            # Detect and skip thinking/reasoning blocks
            if cleaned.startswith(("THOUGHTS:", "Thoughts:", "thoughts:", "(Self-correction")):
                in_thinking = True
                continue
            # End thinking on a line that has Hebrew and no English preamble
            if in_thinking:
                if re.search(r"[\u0590-\u05FF]", cleaned) and not re.match(r"^[\(\[]", cleaned):
                    in_thinking = False
                else:
                    continue
            # Skip lines that look like headers/labels
            if cleaned.startswith(("```", "##", "**", "שם השיר", "מילים:", "לחן:", "Lyrics:")):
                continue
            # Only keep lines with Hebrew content
            if re.search(r"[\u0590-\u05FF]", cleaned):
                lines.append(cleaned)
        return lines

    def verify(self, title: str, draft: TranscriptDraft) -> LyricsVerificationResult:
        draft_text = draft.text.strip()
        draft_lines = _draft_lines(draft)
        warnings = _local_warnings(draft_text)

        if not self.api_key:
            logger.info("No Gemini API key configured")
            return self._fallback_or_draft(title, draft, "Gemini לא זמין כרגע לבדיקת מילים.", warnings)

        try:
            prompt = GEMINI_LYRICS_PROMPT.format(
                title=title,
                draft_text=draft_text,
            )
            gemini_text = _call_gemini(prompt, self.api_key, self.model)
            if not gemini_text.strip():
                logger.info("Gemini returned empty response")
                return self._fallback_or_draft(title, draft, "Gemini לא החזיר מילים שימושיות.", warnings)

            gemini_lines = self._parse_gemini_lines(gemini_text)
            if not gemini_lines:
                logger.info("Gemini returned no Hebrew lyrics")
                return self._fallback_or_draft(title, draft, "Gemini לא זיהה שורות מילים בעברית.", warnings)

            logger.info("Gemini returned %d lyrics lines for '%s'", len(gemini_lines), title[:40])

        except Exception as exc:
            logger.warning("Gemini lyrics request failed: %s", exc)
            return self._fallback_or_draft(title, draft, "Gemini נכשל בבקשת המילים לשיר.", warnings)

        # Align Gemini's output against the draft segments
        gemini_tokens = [
            token
            for line in gemini_lines
            for token in _tokenize_words(line)
        ]
        corrected_lines, correction_count = _align_source_to_segments(gemini_tokens, draft)

        # Score: compare Gemini's output to the draft
        draft_tokens = [_normalize_token(t) for t in _tokenize_words(draft_text)]
        gemini_norm = [_normalize_token(t) for t in gemini_tokens]
        if draft_tokens and gemini_norm:
            score = SequenceMatcher(None, draft_tokens, gemini_norm, autojunk=False).ratio()
        else:
            score = 0.0

        # Determine whether to apply corrections
        apply_corrections = bool(
            corrected_lines
            and correction_count > 0
            and score >= 0.30  # Gemini's output must be at least 30% similar
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

        return LyricsVerificationResult(
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
        )


class HybridLyricsVerifier:
    """Prefer Shironet/Tab4U/Nagnu, and use Gemini only if none provide usable lyrics."""

    name = "hybrid_lyrics_verifier"

    def __init__(self):
        self.preferred = PreferredSiteLyricsVerifier()
        self.gemini = GeminiLyricsVerifier(fallback_verifier=None)

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
        if not gemini_result.search_query:
            gemini_result.search_query = preferred_result.search_query
        return gemini_result
