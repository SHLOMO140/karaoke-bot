"""Consensus engine for multi-source lyrics verification."""

import re
import unicodedata
from collections import Counter

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
