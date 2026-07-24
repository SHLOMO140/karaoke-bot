"""External chord-sheet lookup and parsing helpers."""

from __future__ import annotations

import html
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .harmony import (
    EASY_KEY_TARGET,
    _parse_chord_label,
    prepare_song_analysis_for_display,
    render_chord_sheet_text,
    resolve_song_analysis_key_labels,
    transpose_chord_label,
)
from .web_search import (
    SearchResult,
    _build_query_variants,
    _evaluate_candidate_text_against_draft,
    _extract_title_context,
    _fetch_text,
    _normalize_token,
    _sanitize_internal_site_query,
    _search_known_site_results,
    _search_tab4u_results,
)
from .models import ChordEvent, SongAnalysis, TranscriptDraft, TranscriptSegment, WordTiming

logger = logging.getLogger(__name__)

_SECTIONAL_MATCH_SCORE_THRESHOLD = 0.6
_SECTIONAL_QUERY_RESULT_LIMIT = 6
_SECTIONAL_QUERY_WORD_LIMIT = 8
_SECTIONAL_QUERY_SPACING = 2
_SECTIONAL_QUERY_RADIUS = 4

_TAB4U_CONTENT_PATTERN = re.compile(
    r'<div id="songContentTPL"[^>]*>(.*?)</div>',
    re.S | re.I,
)
_TAB4U_TABLE_PATTERN = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
_TAB4U_CELL_PATTERN = re.compile(r'<td[^>]+class="([^"]*)"[^>]*>(.*?)</td>', re.S | re.I)
_TAB4U_CHORD_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-G](?:#|b)?(?:maj7|m7b5|dim7|sus2|sus4|add9|m9|m7|maj9|m6|6|dim|aug|m|7|4|2|9)?(?:/[A-G](?:#|b)?)?)"
    r"(?![A-Za-z0-9])"
)


@dataclass
class _ChordToken:
    label: str
    column: int


@dataclass
class _LyricWord:
    text: str
    column: int
    global_index: int


@dataclass
class _ChordRow:
    kind: str
    text: str
    tokens: list[_ChordToken] = field(default_factory=list)


@dataclass
class _ParsedTab4USheet:
    source_url: str
    tables: list[list[_ChordRow]]
    lyric_lines: list[str]
    line_word_pairs: list[tuple[list[_ChordToken], list[_LyricWord]]]
    chord_labels: list[str]


@dataclass
class _SectionSearchQuery:
    anchor_index: int
    query: str


@dataclass
class _SectionChordMatch:
    parsed_sheet: _ParsedTab4USheet
    score: float
    segment_start: int
    segment_end: int
    query: str


def _normalize_line_text(text: str) -> str:
    return " ".join(str(text or "").replace("\xa0", " ").split()).strip()


def _clean_visible_text(fragment: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = text.replace("\r", "").replace("\t", " ")
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _extract_chord_tokens(row_text: str) -> list[_ChordToken]:
    return [
        _ChordToken(label=match.group(1), column=match.start(1))
        for match in _TAB4U_CHORD_TOKEN_PATTERN.finditer(row_text)
    ]


def _parse_lyric_words(line_text: str, next_index: int) -> tuple[list[_LyricWord], int]:
    words: list[_LyricWord] = []
    for match in re.finditer(r"\S+", line_text):
        words.append(
            _LyricWord(
                text=match.group(0),
                column=match.start(),
                global_index=next_index,
            )
        )
        next_index += 1
    return words, next_index


def _looks_like_section_heading(text: str) -> bool:
    stripped = _normalize_line_text(text)
    if not stripped.endswith(":"):
        return False
    return 0 < len(stripped[:-1].split()) <= 3


def _parse_tab4u_sheet(page_html: str, source_url: str) -> _ParsedTab4USheet | None:
    match = _TAB4U_CONTENT_PATTERN.search(page_html)
    if match:
        content_html = match.group(1)
    elif _TAB4U_CELL_PATTERN.search(page_html):
        # Tab4U layout changes have renamed/dropped the songContentTPL
        # container before; as long as chords/song cells exist, parse the
        # whole page instead of silently returning nothing.
        content_html = page_html
    else:
        return None

    tables: list[list[_ChordRow]] = []
    lyric_lines: list[str] = []
    line_word_pairs: list[tuple[list[_ChordToken], list[_LyricWord]]] = []
    chord_labels: list[str] = []
    next_word_index = 0

    for table_html in _TAB4U_TABLE_PATTERN.findall(content_html):
        rows: list[_ChordRow] = []
        pending_chord_row: _ChordRow | None = None

        for class_name, cell_html in _TAB4U_CELL_PATTERN.findall(table_html):
            normalized_class = class_name.lower()
            cleaned_text = _clean_visible_text(cell_html)
            if not cleaned_text:
                continue

            if "chords" in normalized_class:
                tokens = _extract_chord_tokens(cleaned_text)
                chord_labels.extend(token.label for token in tokens)
                chord_row = _ChordRow(kind="chords", text=cleaned_text, tokens=tokens)
                rows.append(chord_row)
                pending_chord_row = chord_row if tokens else None
                continue

            if "song" not in normalized_class:
                continue

            row = _ChordRow(kind="song", text=cleaned_text)
            rows.append(row)

            if _looks_like_section_heading(cleaned_text):
                pending_chord_row = None
                continue

            normalized_lyric_line = _normalize_line_text(cleaned_text)
            if not normalized_lyric_line:
                pending_chord_row = None
                continue

            lyric_lines.append(normalized_lyric_line)
            lyric_words, next_word_index = _parse_lyric_words(cleaned_text, next_word_index)
            if pending_chord_row and lyric_words:
                line_word_pairs.append((pending_chord_row.tokens, lyric_words))
            pending_chord_row = None

        if rows:
            tables.append(rows)

    if not tables or not chord_labels:
        return None

    return _ParsedTab4USheet(
        source_url=source_url,
        tables=tables,
        lyric_lines=lyric_lines,
        line_word_pairs=line_word_pairs,
        chord_labels=chord_labels,
    )


def _query_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[\w\u0590-\u05FF']+", text or "")
        if len(_normalize_token(token)) >= 2
    ]


def _title_match_score(expected_title: str, candidate_title: str) -> float:
    expected_tokens = [_normalize_token(token) for token in _query_tokens(expected_title) if _normalize_token(token)]
    candidate_tokens = [_normalize_token(token) for token in _query_tokens(candidate_title) if _normalize_token(token)]
    if not expected_tokens or not candidate_tokens:
        return 0.0

    expected_set = set(expected_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(expected_set & candidate_set)
    coverage = overlap / len(expected_set)
    candidate_coverage = overlap / len(candidate_set)
    sequence_score = SequenceMatcher(
        None,
        " ".join(expected_tokens),
        " ".join(candidate_tokens),
        autojunk=False,
    ).ratio()
    return max(coverage, candidate_coverage * 0.85, sequence_score * 0.7 + coverage * 0.3)


def _looks_like_medley_title(title: str) -> bool:
    normalized = _normalize_line_text(title).lower()
    if normalized.count("|") >= 2:
        return True
    return any(marker in normalized for marker in ("מחרוזת", "medley", "mix"))


def _build_section_search_queries(
    segments: list[TranscriptSegment],
    *,
    max_queries: int = 15,
) -> list[_SectionSearchQuery]:
    candidates: list[tuple[int, int, str, str]] = []
    for index, segment in enumerate(segments):
        line = _normalize_line_text(segment.text)
        if not line:
            continue
        tokens = _query_tokens(line)
        if len(tokens) < 4:
            continue
        signature = " ".join(_normalize_token(token) for token in tokens if _normalize_token(token))
        if not signature:
            continue
        query = " ".join(tokens[:_SECTIONAL_QUERY_WORD_LIMIT]).strip()
        candidates.append((len(tokens), index, signature, query))

    if not candidates:
        return []

    selected: list[_SectionSearchQuery] = []
    seen_signatures: set[str] = set()
    bucket_size = max(1, (len(segments) + max_queries - 1) // max_queries)
    for bucket_start in range(0, len(segments), bucket_size):
        bucket_end = min(len(segments), bucket_start + bucket_size)
        bucket_candidates = [
            item
            for item in candidates
            if bucket_start <= item[1] < bucket_end and item[2] not in seen_signatures
        ]
        if not bucket_candidates:
            continue
        token_count, index, signature, query = max(
            bucket_candidates,
            key=lambda item: (item[0], -(abs(item[1] - ((bucket_start + bucket_end) // 2)))),
        )
        del token_count
        selected.append(_SectionSearchQuery(anchor_index=index, query=query))
        seen_signatures.add(signature)
        if len(selected) >= max_queries:
            break

    if len(selected) < max_queries:
        for _token_count, index, signature, query in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if signature in seen_signatures:
                continue
            if any(abs(index - item.anchor_index) <= _SECTIONAL_QUERY_SPACING for item in selected):
                continue
            selected.append(_SectionSearchQuery(anchor_index=index, query=query))
            seen_signatures.add(signature)
            if len(selected) >= max_queries:
                break

    if selected:
        selected.sort(key=lambda item: item.anchor_index)
        return selected

    _token_count, index, _signature, query = candidates[0]
    return [_SectionSearchQuery(anchor_index=index, query=query)]


def _search_tab4u_candidates_for_lyric_query(query: str) -> list[SearchResult]:
    raw_tokens = _query_tokens(query)
    fallback_query = " ".join(raw_tokens[: min(6, len(raw_tokens))]).strip()
    candidate_queries = [query.strip(), _sanitize_internal_site_query(query)]
    if fallback_query:
        candidate_queries.append(fallback_query)

    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for candidate_query in dict.fromkeys(text for text in candidate_queries if text):
        for result in _search_tab4u_results(candidate_query):
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            results.append(result)
            if len(results) >= _SECTIONAL_QUERY_RESULT_LIMIT:
                return results
    return results


def _tab4u_tabs_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")
    if domain != "tab4u.com":
        return ""
    path = parsed.path or "/"
    if "/lyrics/songs/" in path:
        path = path.replace("/lyrics/songs/", "/tabs/songs/")
    return urllib.parse.urlunparse((parsed.scheme or "https", parsed.netloc or "www.tab4u.com", path, "", "", ""))


def _search_tab4u_candidates(title: str, draft: TranscriptDraft) -> list[SearchResult]:
    context = _extract_title_context(title)
    queries = _build_query_variants(title, draft.text)

    try:
        results = _search_known_site_results(title, queries, context)
    except Exception as exc:
        logger.info("Preferred chord-source search failed for %s: %s", title, exc)
        results = []

    if results:
        return results

    fallback_queries = [
        f"{context['hebrew_artist']} {context['song']}".strip(),
        context["song"].strip(),
        context["clean_title"].strip(),
        f"{context['latin_artist']} {context['song']}".strip(),
    ]
    fallback_queries.extend(
        _sanitize_internal_site_query(query)
        for query in queries[:3]
        if _sanitize_internal_site_query(query)
    )

    deduped_queries = [query for query in dict.fromkeys(fallback_queries) if query]
    fallback_results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for query in deduped_queries[:4]:
        for result in _search_tab4u_results(query):
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            fallback_results.append(result)
    return fallback_results


def _build_metadata_from_chords(
    chord_labels: list[str],
    provider: str,
    source_audio: str,
    target_key: str,
) -> SongAnalysis:
    dummy_events: list[ChordEvent] = []
    for index, label in enumerate(chord_labels):
        parsed = _parse_chord_label(label)
        dummy_events.append(
            ChordEvent(
                label=label,
                start=float(index),
                end=float(index + 1),
                confidence=1.0,
                root=parsed[0] if parsed else "",
                quality="",
            )
        )

    return prepare_song_analysis_for_display(
        SongAnalysis(
            provider=provider,
            source_audio=source_audio,
            original_chord_events=list(dummy_events),
            chord_events=list(dummy_events),
        ),
        [],
        target_key=target_key,
    )


def _clone_chord_event(event: ChordEvent) -> ChordEvent:
    return ChordEvent(
        label=event.label,
        start=event.start,
        end=event.end,
        confidence=event.confidence,
        root=event.root,
        quality=event.quality,
    )


def _map_source_words_to_transcript(
    source_words: list[_LyricWord],
    transcript_words: list[WordTiming],
) -> dict[int, int]:
    source_tokens = [_normalize_token(word.text) for word in source_words]
    transcript_tokens = [_normalize_token(word.word) for word in transcript_words]
    mapping: dict[int, int] = {}

    for tag, left_start, left_end, right_start, right_end in SequenceMatcher(
        None,
        source_tokens,
        transcript_tokens,
        autojunk=False,
    ).get_opcodes():
        source_count = left_end - left_start
        transcript_count = right_end - right_start

        if tag == "equal":
            for offset in range(min(source_count, transcript_count)):
                mapping[source_words[left_start + offset].global_index] = right_start + offset
            continue

        if source_count <= 0 or not transcript_words:
            continue

        if transcript_count <= 0:
            fallback_index = min(max(right_start - 1, 0), len(transcript_words) - 1)
            for offset in range(source_count):
                mapping[source_words[left_start + offset].global_index] = fallback_index
            continue

        for offset in range(source_count):
            if source_count == 1:
                mapped_offset = 0
            else:
                mapped_offset = round(offset * (transcript_count - 1) / max(1, source_count - 1))
            mapping[source_words[left_start + offset].global_index] = right_start + mapped_offset

    return mapping


def _closest_lyric_word(chord: _ChordToken, words: list[_LyricWord]) -> _LyricWord | None:
    if not words:
        return None

    chosen_index = 0
    for index, word in enumerate(words):
        if word.column <= chord.column + 1:
            chosen_index = index
            continue
        break

    if chosen_index + 1 < len(words):
        current_distance = abs(words[chosen_index].column - chord.column)
        next_distance = abs(words[chosen_index + 1].column - chord.column)
        if next_distance < current_distance:
            chosen_index += 1

    return words[chosen_index]


def _build_timed_events(
    parsed_sheet: _ParsedTab4USheet,
    semitones: int,
    segments: list[TranscriptSegment],
) -> tuple[list[ChordEvent], list[ChordEvent]]:
    transcript_words = [word for segment in segments for word in segment.words]
    source_words = [word for _tokens, words in parsed_sheet.line_word_pairs for word in words]
    if not transcript_words or not source_words:
        return [], []

    source_to_transcript = _map_source_words_to_transcript(source_words, transcript_words)
    assignments: list[tuple[int, ChordEvent]] = []

    for chord_tokens, lyric_words in parsed_sheet.line_word_pairs:
        for token in chord_tokens:
            matched_word = _closest_lyric_word(token, lyric_words)
            if matched_word is None:
                continue
            transcript_index = source_to_transcript.get(matched_word.global_index)
            if transcript_index is None or transcript_index >= len(transcript_words):
                continue
            transcript_word = transcript_words[transcript_index]
            parsed = _parse_chord_label(token.label)
            assignments.append(
                (
                    transcript_index,
                    ChordEvent(
                        label=token.label,
                        start=transcript_word.start,
                        end=max(transcript_word.end, transcript_word.start + 0.05),
                        confidence=1.0,
                        root=parsed[0] if parsed else "",
                        quality="",
                    ),
                )
            )

    if not assignments:
        return [], []

    assignments.sort(key=lambda item: (item[1].start, item[0], item[1].label))
    original_events: list[ChordEvent] = []
    for index, (_transcript_index, event) in enumerate(assignments):
        next_start = assignments[index + 1][1].start if index + 1 < len(assignments) else None
        if next_start is not None and next_start > event.start + 1e-3:
            event.end = max(event.end, next_start)
        if original_events and event.label == original_events[-1].label and event.start <= original_events[-1].end + 1e-3:
            original_events[-1].end = max(original_events[-1].end, event.end)
            continue
        original_events.append(event)

    display_events: list[ChordEvent] = []
    for event in original_events:
        display_label = transpose_chord_label(event.label, semitones) if semitones else event.label
        parsed = _parse_chord_label(display_label)
        display_events.append(
            ChordEvent(
                label=display_label,
                start=event.start,
                end=event.end,
                confidence=event.confidence,
                root=parsed[0] if parsed else "",
                quality=event.quality,
            )
        )

    return original_events, display_events


def _find_best_transcript_window_for_sheet(
    parsed_sheet: _ParsedTab4USheet,
    segments: list[TranscriptSegment],
    anchor_index: int,
    query: str,
) -> _SectionChordMatch | None:
    candidate_text = "\n".join(parsed_sheet.lyric_lines)
    if not candidate_text.strip():
        return None

    total_segments = len(segments)
    best_start = 0
    best_end = 0
    best_score = 0.0
    min_window = 1 if total_segments <= 2 else 2
    max_window = min(max(min_window, 6), total_segments)
    start_min = max(0, anchor_index - _SECTIONAL_QUERY_RADIUS)
    start_max = min(anchor_index, total_segments - 1)

    for start in range(start_min, start_max + 1):
        min_end = max(anchor_index + 1, start + min_window)
        max_end = min(total_segments, start + max_window)
        for end in range(min_end, max_end + 1):
            if not (start <= anchor_index < end):
                continue
            window_segments = segments[start:end]
            if sum(len(segment.words) for segment in window_segments) < 6:
                continue
            _matched_lines, score = _evaluate_candidate_text_against_draft(
                TranscriptDraft(segments=window_segments),
                candidate_text,
            )
            if score > best_score or (
                abs(score - best_score) <= 0.02 and (end - start) > (best_end - best_start)
            ):
                best_start = start
                best_end = end
                best_score = score

    if best_score < _SECTIONAL_MATCH_SCORE_THRESHOLD:
        return None

    improved = True
    while improved:
        improved = False
        if best_start > 0:
            _matched_lines, score = _evaluate_candidate_text_against_draft(
                TranscriptDraft(segments=segments[best_start - 1 : best_end]),
                candidate_text,
            )
            if score >= max(_SECTIONAL_MATCH_SCORE_THRESHOLD, best_score - 0.04):
                best_start -= 1
                best_score = max(best_score, score)
                improved = True
        if best_end < total_segments:
            _matched_lines, score = _evaluate_candidate_text_against_draft(
                TranscriptDraft(segments=segments[best_start : best_end + 1]),
                candidate_text,
            )
            if score >= max(_SECTIONAL_MATCH_SCORE_THRESHOLD, best_score - 0.04):
                best_end += 1
                best_score = max(best_score, score)
                improved = True

    return _SectionChordMatch(
        parsed_sheet=parsed_sheet,
        score=best_score,
        segment_start=best_start,
        segment_end=best_end,
        query=query,
    )


def _merge_external_events(events: list[ChordEvent]) -> list[ChordEvent]:
    merged: list[ChordEvent] = []
    for event in sorted(events, key=lambda item: (item.start, item.end, item.label)):
        cloned = _clone_chord_event(event)
        if (
            merged
            and cloned.label == merged[-1].label
            and cloned.start <= merged[-1].end + 1e-3
        ):
            merged[-1].end = max(merged[-1].end, cloned.end)
            merged[-1].confidence = max(merged[-1].confidence, cloned.confidence)
            continue
        merged.append(cloned)
    return merged


def _build_sectional_external_analysis(
    title: str,
    segments: list[TranscriptSegment],
    *,
    provider: str,
    source_audio: str,
) -> SongAnalysis | None:
    query_specs = _build_section_search_queries(segments)
    if not query_specs:
        return None

    parsed_sheet_cache: dict[str, _ParsedTab4USheet | None] = {}
    matches_by_url: dict[str, _SectionChordMatch] = {}

    for query_spec in query_specs:
        for candidate in _search_tab4u_candidates_for_lyric_query(query_spec.query):
            chord_url = _tab4u_tabs_url(candidate.url)
            if not chord_url:
                continue

            if chord_url not in parsed_sheet_cache:
                try:
                    page_html = _fetch_text(chord_url, timeout=15)
                except Exception as exc:
                    logger.info("Tab4U chord page fetch failed for %s: %s", chord_url, exc)
                    parsed_sheet_cache[chord_url] = None
                    continue
                parsed_sheet_cache[chord_url] = _parse_tab4u_sheet(page_html, chord_url)

            parsed_sheet = parsed_sheet_cache.get(chord_url)
            if parsed_sheet is None or not parsed_sheet.lyric_lines:
                continue

            match = _find_best_transcript_window_for_sheet(
                parsed_sheet,
                segments,
                query_spec.anchor_index,
                query_spec.query,
            )
            if match is None:
                continue

            existing_match = matches_by_url.get(parsed_sheet.source_url)
            if existing_match is None or match.score > existing_match.score + 0.02 or (
                abs(match.score - existing_match.score) <= 0.02
                and (match.segment_end - match.segment_start)
                > (existing_match.segment_end - existing_match.segment_start)
            ):
                matches_by_url[parsed_sheet.source_url] = match

    candidate_matches = sorted(
        matches_by_url.values(),
        key=lambda item: (
            -item.score,
            -(item.segment_end - item.segment_start),
            item.segment_start,
            item.parsed_sheet.source_url,
        ),
    )
    selected_matches: list[_SectionChordMatch] = []
    for match in candidate_matches:
        if any(
            not (
                match.segment_end <= selected.segment_start
                or match.segment_start >= selected.segment_end
            )
            for selected in selected_matches
        ):
            continue
        selected_matches.append(match)

    if not selected_matches:
        return None

    selected_matches.sort(key=lambda item: item.segment_start)
    combined_original_events: list[ChordEvent] = []
    source_urls: list[str] = []
    for match in selected_matches:
        original_events, _display_events = _build_timed_events(
            match.parsed_sheet,
            0,
            segments[match.segment_start : match.segment_end],
        )
        if not original_events:
            continue
        combined_original_events.extend(original_events)
        source_urls.append(match.parsed_sheet.source_url)

    unique_urls = list(dict.fromkeys(source_urls))
    merged_original_events = _merge_external_events(combined_original_events)
    if not merged_original_events or not unique_urls:
        return None

    prepared = prepare_song_analysis_for_display(
        SongAnalysis(
            provider=provider,
            source_audio=source_audio,
            original_chord_events=merged_original_events,
            chord_source_name="Tab4U medley" if len(unique_urls) > 1 else "Tab4U",
            chord_source_url=", ".join(unique_urls),
        ),
        segments,
        target_key="",
    )
    analysis = SongAnalysis(
        bpm=prepared.bpm,
        time_signature=prepared.time_signature,
        preview_window_seconds=prepared.preview_window_seconds,
        provider=prepared.provider,
        source_audio=prepared.source_audio,
        beat_times=list(prepared.beat_times),
        measure_times=list(prepared.measure_times),
        original_key="",
        target_key="",
        transpose_semitones=0,
        original_chord_events=list(prepared.original_chord_events),
        chord_events=list(prepared.chord_events),
        chord_sheet_text="",
        chord_source_name="Tab4U medley" if len(unique_urls) > 1 else "Tab4U",
        chord_source_url=", ".join(unique_urls),
    )
    analysis.chord_sheet_text = render_chord_sheet_text(title, segments, analysis)

    logger.info(
        "Using sectional external chord sheet for %s from %d sources (segments=%d score=%.3f).",
        title,
        len(unique_urls),
        sum(match.segment_end - match.segment_start for match in selected_matches),
        sum(match.score for match in selected_matches) / max(1, len(selected_matches)),
    )

    return analysis


def _transpose_row_text(row_text: str, semitones: int) -> str:
    if not semitones:
        return row_text.rstrip()

    parts: list[str] = []
    cursor = 0
    for match in _TAB4U_CHORD_TOKEN_PATTERN.finditer(row_text):
        parts.append(row_text[cursor:match.start(1)])
        original_label = match.group(1)
        transposed_label = transpose_chord_label(original_label, semitones)
        if len(transposed_label) < len(original_label):
            transposed_label = transposed_label.ljust(len(original_label))
        parts.append(transposed_label)
        cursor = match.end(1)
    parts.append(row_text[cursor:])
    return "".join(parts).rstrip()


def _reverse_chord_order_in_place(row_text: str) -> str:
    """Reverse which label sits in which slot on a multi-chord row, keeping
    every slot's column position and surrounding spacing untouched — e.g.
    "Ab                Fm" -> "Fm                Ab".

    The column *positions* already line up fine once the sheet is sent as a
    monospace block (see _as_pre in bot.py); the remaining bug is that a line
    with more than one chord reads its labels in storage order (Ab then Fm)
    while the Hebrew lyric line below it reads right-to-left, so the chords
    end up matched to the wrong words. Swapping which label occupies which
    existing slot (not reflowing/padding the row) fixes the pairing without
    touching anything that was already correctly positioned. A row with 0 or 1
    tokens has no "order" to fix and is returned unchanged.
    """
    matches = list(re.finditer(r"\S+", row_text))
    if len(matches) < 2:
        return row_text
    reversed_labels = iter(m.group(0) for m in reversed(matches))
    return re.sub(r"\S+", lambda _m: next(reversed_labels), row_text)


def _render_external_chord_sheet(
    title: str,
    parsed_sheet: _ParsedTab4USheet,
    *,
    bpm: float,
    time_signature: int,
    original_key: str,
    target_key: str,
    semitones: int,
    mirror_chords_for_rtl: bool = False,
) -> str:
    header_parts = [f"כותרת: {title or 'ללא שם'}"]
    header_parts.append(f"קצב: {bpm:.0f}" if bpm > 0 else "קצב: לא ידוע")
    header_parts.append(f"משקל: {time_signature}/4")
    if original_key:
        header_parts.append(f"סולם מקור: {original_key}")
    if target_key:
        header_parts.append(f"סולם קל: {target_key}")

    lines = header_parts + [""]
    for table in parsed_sheet.tables:
        for row in table:
            if row.kind == "chords":
                text = _transpose_row_text(row.text, semitones)
                if mirror_chords_for_rtl:
                    text = _reverse_chord_order_in_place(text)
            else:
                text = row.text.rstrip()
            if text.strip():
                lines.append(text.rstrip())
        if lines[-1] != "":
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def lookup_external_chord_sheet_by_title(
    title: str,
    *,
    provider: str,
    source_audio: str = "",
    target_key: str = EASY_KEY_TARGET,
) -> SongAnalysis | None:
    normalized_title = _normalize_line_text(title)
    if not normalized_title or _looks_like_medley_title(normalized_title):
        return None

    candidates = _search_tab4u_candidates(normalized_title, TranscriptDraft(segments=[]))
    best_sheet: _ParsedTab4USheet | None = None
    best_score = 0.0

    for candidate in candidates[:6]:
        candidate_score = _title_match_score(normalized_title, getattr(candidate, "title", ""))
        if candidate_score < 0.34:
            continue

        chord_url = _tab4u_tabs_url(candidate.url)
        if not chord_url:
            continue
        try:
            page_html = _fetch_text(chord_url, timeout=15)
        except Exception as exc:
            logger.info("Tab4U chord page fetch failed for %s: %s", chord_url, exc)
            continue

        parsed_sheet = _parse_tab4u_sheet(page_html, chord_url)
        if parsed_sheet is None or not parsed_sheet.lyric_lines:
            continue

        score = max(candidate_score, _title_match_score(normalized_title, parsed_sheet.lyric_lines[0]))
        if score > best_score:
            best_score = score
            best_sheet = parsed_sheet

    if best_sheet is None or best_score < 0.45:
        return None

    analysis = _build_metadata_from_chords(
        best_sheet.chord_labels,
        provider=provider,
        source_audio=source_audio,
        target_key=target_key,
    )
    analysis.chord_source_name = "Tab4U"
    analysis.chord_source_url = best_sheet.source_url
    resolved_original_key, resolved_target_key = resolve_song_analysis_key_labels(analysis)
    analysis.original_key = resolved_original_key
    analysis.target_key = resolved_target_key
    analysis.chord_sheet_text = _render_external_chord_sheet(
        normalized_title,
        best_sheet,
        bpm=analysis.bpm,
        time_signature=analysis.time_signature,
        original_key=analysis.original_key,
        target_key=analysis.target_key,
        semitones=analysis.transpose_semitones,
    )

    logger.info(
        "Using title-only external chord sheet for %s from %s (score=%.3f).",
        normalized_title,
        best_sheet.source_url,
        best_score,
    )

    # Stash the parsed sheet so callers can re-render in the original key
    # (semitones=0) or the easy key, and build inline [Chord] library content.
    analysis.parsed_sheet = best_sheet
    return analysis


def lookup_external_chord_sheet(
    title: str,
    segments: list[TranscriptSegment],
    *,
    provider: str,
    source_audio: str = "",
    target_key: str = EASY_KEY_TARGET,
) -> SongAnalysis | None:
    draft = TranscriptDraft(segments=segments)
    if not draft.text.strip():
        return None

    candidates = _search_tab4u_candidates(title, draft)
    best_sheet: _ParsedTab4USheet | None = None
    best_score = 0.0

    for candidate in candidates[:6]:
        chord_url = _tab4u_tabs_url(candidate.url)
        if not chord_url:
            continue
        try:
            page_html = _fetch_text(chord_url, timeout=15)
        except Exception as exc:
            logger.info("Tab4U chord page fetch failed for %s: %s", chord_url, exc)
            continue

        parsed_sheet = _parse_tab4u_sheet(page_html, chord_url)
        if parsed_sheet is None or not parsed_sheet.lyric_lines:
            continue

        _matched_lines, score = _evaluate_candidate_text_against_draft(
            draft,
            "\n".join(parsed_sheet.lyric_lines),
        )
        if score > best_score:
            best_score = score
            best_sheet = parsed_sheet

    sectional_analysis = None
    if best_sheet is None or best_score < 0.55 or _looks_like_medley_title(title):
        sectional_analysis = _build_sectional_external_analysis(
            title,
            segments,
            provider=provider,
            source_audio=source_audio,
        )

    if sectional_analysis is not None and (
        best_sheet is None
        or best_score < 0.55
        or (_looks_like_medley_title(title) and best_score < 0.72)
    ):
        return sectional_analysis

    if best_sheet is None or best_score < 0.32:
        return None

    metadata = _build_metadata_from_chords(
        best_sheet.chord_labels,
        provider=provider,
        source_audio=source_audio,
        target_key=target_key,
    )
    original_events, _ = _build_timed_events(
        best_sheet,
        0,
        segments,
    )
    analysis = prepare_song_analysis_for_display(
        SongAnalysis(
            bpm=metadata.bpm,
            time_signature=metadata.time_signature,
            preview_window_seconds=metadata.preview_window_seconds,
            provider=provider,
            source_audio=source_audio,
            beat_times=list(metadata.beat_times),
            measure_times=list(metadata.measure_times),
            original_chord_events=original_events,
            chord_events=list(original_events),
            chord_sheet_text="",
            chord_source_name="Tab4U",
            chord_source_url=best_sheet.source_url,
        ),
        segments,
        target_key=target_key,
    )
    resolved_original_key, resolved_target_key = resolve_song_analysis_key_labels(analysis)
    analysis.original_key = resolved_original_key
    analysis.target_key = resolved_target_key
    analysis.chord_sheet_text = _render_external_chord_sheet(
        title,
        best_sheet,
        bpm=analysis.bpm,
        time_signature=analysis.time_signature,
        original_key=analysis.original_key,
        target_key=analysis.target_key,
        semitones=analysis.transpose_semitones,
    )

    logger.info(
        "Using external chord sheet for %s from %s (score=%.3f).",
        title,
        best_sheet.source_url,
        best_score,
    )

    return analysis
