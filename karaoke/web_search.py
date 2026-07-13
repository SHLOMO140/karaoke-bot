"""Lean HTTP fetch + lyrics/chord search helpers.

Extracted from the former lyrics_verifier module so the chord path carries no
ML dependencies. Pure stdlib + config + models.
"""

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
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError

from .config import HTTP_CACHE_DIR, HTTP_CACHE_TTL_SECONDS
from .models import TranscriptDraft

logger = logging.getLogger(__name__)

SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.7",
}

NAGNU_ARTIST_SEGMENTS = ("אמנים", "אומנים")

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

HEBREW_LYRICS_QUERY = "\u05de\u05d9\u05dc\u05d9\u05dd"

HEBREW_LYRICS_FOR_SONG_QUERY = "\u05de\u05d9\u05dc\u05d9\u05dd \u05dc\u05e9\u05d9\u05e8"

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str

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

def _domain_from_url(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")

def _domain_priority(url: str) -> int:
    domain = _domain_from_url(url)
    for candidate, priority in DOMAIN_PRIORITY.items():
        if domain.endswith(candidate):
            return priority
    return 99

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

def _draft_lines(draft: TranscriptDraft) -> list[str]:
    lines = [segment.text.strip() for segment in draft.segments if segment.text.strip()]
    if lines:
        return lines
    return [draft.text.strip()] if draft.text.strip() else []

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

def _evaluate_candidate_text_against_draft(
    draft: TranscriptDraft,
    candidate_text: str,
) -> tuple[list[str], float]:
    candidate_lines, candidate_score = _find_best_source_line_window(candidate_text, draft)
    if candidate_lines and candidate_score >= 0.32:
        return candidate_lines, candidate_score
    return [], 0.0
