"""Consensus engine for multi-source lyrics verification."""

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from .config import CONSENSUS_MIN_SOURCES
from .models import ConsensusResult, DisputedLine


# Hebrew niqqud Unicode range
_NIQQUD_RE = re.compile(r'[\u05B0-\u05C8]')
# Punctuation (keep Hebrew/Latin letters, digits, spaces)
_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
# Multiple whitespace
_SPACE_RE = re.compile(r'\s+')


def normalize_lyrics_line(line: str) -> str:
    """Normalize a lyrics line for comparison."""
    text = _NIQQUD_RE.sub('', line)
    text = _PUNCT_RE.sub('', text)
    text = _SPACE_RE.sub(' ', text).strip()
    text = text.lower()
    return text


class ConsensusEngine:
    """Compare lyrics from multiple sources and find consensus."""

    def __init__(self, min_sources: int = CONSENSUS_MIN_SOURCES):
        self.min_sources = min_sources

    def evaluate(self, sources: dict[str, list[str]]) -> ConsensusResult:
        if not sources:
            return ConsensusResult(
                consensus_reached=False,
                agreed_sources=0,
                lyrics=[],
                disputes=[],
            )

        max_lines = max(len(lines) for lines in sources.values())

        agreed_lyrics: list[str] = []
        disputes: list[DisputedLine] = []
        min_agreement = len(sources)

        for line_idx in range(max_lines):
            versions: dict[str, str] = {}
            normalized: dict[str, str] = {}

            for source_name, lines in sources.items():
                if line_idx < len(lines):
                    original = lines[line_idx]
                    versions[source_name] = original
                    normalized[source_name] = normalize_lyrics_line(original)

            counts = Counter(normalized.values())
            most_common_text, most_common_count = counts.most_common(1)[0]

            if most_common_count < self.min_sources:
                min_agreement = min(min_agreement, most_common_count)
                best_original = next(
                    orig for src, orig in versions.items()
                    if normalized[src] == most_common_text
                )
                agreed_lyrics.append(best_original)
                disputes.append(DisputedLine(
                    line_number=line_idx,
                    versions=versions,
                ))
            else:
                min_agreement = min(min_agreement, most_common_count)
                best_original = next(
                    orig for src, orig in versions.items()
                    if normalized[src] == most_common_text
                )
                agreed_lyrics.append(best_original)

        consensus_reached = len(disputes) == 0 and min_agreement >= self.min_sources

        return ConsensusResult(
            consensus_reached=consensus_reached,
            agreed_sources=min_agreement if sources else 0,
            lyrics=agreed_lyrics,
            disputes=disputes,
        )

    # ------------------------------------------------------------------
    # Alignment-based consensus
    # ------------------------------------------------------------------
    #
    # The positional engine above votes by raw line index, so a single extra
    # header line or different stanza segmentation in one source destroys
    # agreement even when the actual words are identical. evaluate_aligned
    # aligns every source's TOKEN STREAM to an anchor source and votes per
    # anchor line, which makes consensus robust to line offsets, merged or
    # split lines, and collapsed chorus repeats.

    AGREEMENT_TOKEN_RATIO = 0.9
    SOURCE_AGREEMENT_RATIO = 0.9

    @staticmethod
    def _line_tokens(line: str) -> list[str]:
        return [token for token in normalize_lyrics_line(line).split(" ") if token]

    def evaluate_aligned(
        self,
        sources: dict[str, list[str]],
        priority: dict[str, int] | None = None,
    ) -> ConsensusResult:
        if not sources:
            return ConsensusResult(
                consensus_reached=False, agreed_sources=0, lyrics=[], disputes=[]
            )

        priority = priority or {}

        def _rank(source_name: str) -> int:
            for domain, rank in priority.items():
                if source_name == domain or source_name.endswith(domain):
                    return rank
            return 50

        def _substance(lines: list[str]) -> int:
            return sum(len(self._line_tokens(line)) for line in lines)

        anchor_name = min(sources, key=lambda name: (_rank(name), -_substance(sources[name])))
        anchor_lines = [line for line in sources[anchor_name] if self._line_tokens(line)]
        if not anchor_lines:
            return ConsensusResult(
                consensus_reached=False, agreed_sources=0, lyrics=[], disputes=[]
            )

        # Anchor token stream with a token -> line-index map.
        anchor_tokens: list[str] = []
        anchor_token_line: list[int] = []
        for line_index, line in enumerate(anchor_lines):
            for token in self._line_tokens(line):
                anchor_tokens.append(token)
                anchor_token_line.append(line_index)
        line_token_counts = Counter(anchor_token_line)

        # Per source: which anchor lines it agrees on, and its original text
        # per anchor line (for dispute display / representative spelling).
        agreement: dict[str, set[int]] = {anchor_name: set(range(len(anchor_lines)))}
        source_line_text: dict[str, dict[int, str]] = {
            anchor_name: dict(enumerate(anchor_lines))
        }

        for source_name, raw_lines in sources.items():
            if source_name == anchor_name:
                continue
            lines = [line for line in raw_lines if self._line_tokens(line)]
            source_tokens: list[str] = []
            source_token_origin: list[tuple[int, str]] = []
            normalized_line_set = {normalize_lyrics_line(line) for line in lines}
            for line_index, line in enumerate(lines):
                for token in self._line_tokens(line):
                    source_tokens.append(token)
                    source_token_origin.append((line_index, line))

            matched_per_line: Counter = Counter()
            matched_source_tokens: dict[int, list[str]] = {}
            matcher = SequenceMatcher(None, anchor_tokens, source_tokens, autojunk=False)
            for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
                if tag != "equal":
                    continue
                for offset in range(a_end - a_start):
                    anchor_line = anchor_token_line[a_start + offset]
                    matched_per_line[anchor_line] += 1
                    matched_source_tokens.setdefault(anchor_line, []).append(
                        source_tokens[b_start + offset]
                    )

            agreed_lines: set[int] = set()
            per_line_text: dict[int, str] = {}
            for line_index, line in enumerate(anchor_lines):
                token_count = line_token_counts[line_index]
                ratio = matched_per_line[line_index] / token_count if token_count else 0.0
                normalized_anchor = normalize_lyrics_line(line)
                if ratio >= self.AGREEMENT_TOKEN_RATIO:
                    agreed_lines.add(line_index)
                elif normalized_anchor and normalized_anchor in normalized_line_set:
                    # Repeat-aware fallback: a collapsed chorus ("פזמון x2")
                    # still matches by content even when the token stream
                    # only covers the first occurrence.
                    agreed_lines.add(line_index)
                if matched_source_tokens.get(line_index):
                    per_line_text[line_index] = " ".join(matched_source_tokens[line_index])
            for line_index, line in enumerate(anchor_lines):
                normalized_anchor = normalize_lyrics_line(line)
                for original in lines:
                    if normalize_lyrics_line(original) == normalized_anchor:
                        per_line_text[line_index] = original
                        break
            agreement[source_name] = agreed_lines
            source_line_text[source_name] = per_line_text

        total_lines = len(anchor_lines)
        agreeing_sources = [
            name
            for name, lines_agreed in agreement.items()
            if len(lines_agreed) >= self.SOURCE_AGREEMENT_RATIO * total_lines
        ]
        agreed_source_count = len(agreeing_sources)

        # Representative spelling per line: highest-priority agreeing source
        # that has an exact (normalized) rendition of the line.
        ordered_sources = sorted(agreement, key=_rank)
        lyrics: list[str] = []
        disputes: list[DisputedLine] = []
        for line_index, anchor_line in enumerate(anchor_lines):
            votes = sum(1 for lines_agreed in agreement.values() if line_index in lines_agreed)
            representative = anchor_line
            for name in ordered_sources:
                if line_index in agreement[name] and source_line_text[name].get(line_index):
                    representative = source_line_text[name][line_index]
                    break
            lyrics.append(representative)
            if votes < self.min_sources:
                versions = {
                    name: source_line_text[name].get(line_index, "")
                    for name in agreement
                    if source_line_text[name].get(line_index)
                }
                disputes.append(DisputedLine(line_number=line_index, versions=versions))

        consensus_reached = agreed_source_count >= self.min_sources
        return ConsensusResult(
            consensus_reached=consensus_reached,
            agreed_sources=agreed_source_count,
            lyrics=lyrics,
            disputes=disputes,
        )
