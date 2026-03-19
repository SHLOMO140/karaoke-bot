# Lyrics Verification Pipeline Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign lyrics verification to achieve maximum word accuracy through multi-source consensus, Gemini deep verification, human approval with dispute highlighting, and character-level timing correction.

**Architecture:** Replace the current single-source `HybridLyricsVerifier` with a 7-step pipeline: Whisper transcription with interpolated character timing → parallel multi-source search with consensus engine → conditional Gemini deep verification → human approval with dispute UI → character-level diff with Gemini explanation → Gemini validation of replacements → partial timing re-alignment. Uses `ThreadPoolExecutor` for parallel search, Google Custom Search API as primary search engine, and existing grapheme-weight logic for character timing interpolation.

**Tech Stack:** Python 3.12, faster-whisper, wav2vec2-hebrew, Google Custom Search API, YouTube Data API v3, Google Gemini 2.5 Flash, python-telegram-bot 21.6, concurrent.futures, DuckDuckGo HTML (fallback)

**Spec:** `docs/superpowers/specs/2026-03-19-lyrics-verification-redesign.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `karaoke/models.py` | Modify | Add CharacterTiming, ConsensusResult, DisputedLine, CharDiff, CharChange dataclasses |
| `karaoke/config.py` | Modify | Add GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID, consensus/loop config |
| `karaoke/transcriber.py` | Modify | Add character-level timing interpolation after Whisper transcription |
| `karaoke/google_search.py` | Create | Google Custom Search API + YouTube Data API provider |
| `karaoke/consensus.py` | Create | Consensus engine: normalize, compare, vote across sources |
| `karaoke/char_diff.py` | Create | Character-level diff engine for Hebrew text |
| `karaoke/lyrics_verifier.py` | Major rewrite | New 7-step verification pipeline replacing HybridLyricsVerifier |
| `karaoke/aligner.py` | Modify | Add partial re-alignment support (changed words only) |
| `karaoke/providers.py` | Modify | Add GoogleSearchProvider, YouTubeProvider protocols |
| `karaoke/pipeline.py` | Modify | Wire new verification pipeline |
| `bot.py` | Modify | Update review UI with dispute highlighting, version display |
| `tests/test_models_new.py` | Create | Tests for new dataclasses |
| `tests/test_google_search.py` | Create | Tests for Google search provider |
| `tests/test_consensus.py` | Create | Tests for consensus engine |
| `tests/test_char_diff.py` | Create | Tests for character-level diff |
| `tests/test_lyrics_verifier_v2.py` | Create | Tests for new verification pipeline |

---

## Task 1: New Data Models

**Files:**
- Modify: `karaoke/models.py` (add after line 93, after SubWordTiming)
- Create: `tests/test_models_new.py`

- [ ] **Step 1: Write tests for new dataclasses**

```python
# tests/test_models_new.py
from karaoke.models import (
    CharacterTiming, ConsensusResult, DisputedLine,
    CharDiff, CharChange, VerificationVerdict,
)

def test_character_timing_creation():
    ct = CharacterTiming(char="שׁ", start=0.0, end=0.12)
    assert ct.char == "שׁ"
    assert ct.end - ct.start == 0.12

def test_consensus_result_with_consensus():
    cr = ConsensusResult(
        consensus_reached=True,
        agreed_sources=3,
        lyrics=["שורה אחת", "שורה שתיים"],
        disputes=[],
    )
    assert cr.consensus_reached
    assert cr.agreed_sources == 3
    assert len(cr.disputes) == 0

def test_consensus_result_without_consensus():
    dispute = DisputedLine(
        line_number=5,
        versions={"shironet": "הלב שלי", "tab4u": "הלב שלך"},
    )
    cr = ConsensusResult(
        consensus_reached=False,
        agreed_sources=1,
        lyrics=["הלב שלי"],
        disputes=[dispute],
    )
    assert not cr.consensus_reached
    assert dispute.gemini_recommendation is None
    assert dispute.gemini_confidence == 0.0

def test_char_diff():
    change = CharChange(position=3, old_char="וֹ", new_char="ֵ", change_type="replaced")
    diff = CharDiff(
        word_index=0,
        original_word="שָׁלוֹם",
        corrected_word="שָׁלֵם",
        char_changes=[change],
    )
    assert diff.char_changes[0].change_type == "replaced"
    assert diff.gemini_explanation is None

def test_verification_verdict_enum():
    assert VerificationVerdict.CONSENSUS == "consensus"
    assert VerificationVerdict.GEMINI_VERIFIED == "gemini_verified"
    assert VerificationVerdict.HUMAN_APPROVED == "human_approved"
    assert VerificationVerdict.NO_SOURCES == "no_sources"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_models_new.py -v`
Expected: ImportError — classes don't exist yet

- [ ] **Step 3: Implement new dataclasses in models.py**

Add after `SubWordTiming` (line 93) in `karaoke/models.py`:

```python
@dataclass
class CharacterTiming:
    """Timing for a single character/grapheme cluster."""
    char: str
    start: float
    end: float


class VerificationVerdict(str, Enum):
    """Verdict for the new multi-step lyrics verification."""
    CONSENSUS = "consensus"          # 3+ sources agreed
    GEMINI_VERIFIED = "gemini_verified"  # Gemini decided
    HUMAN_APPROVED = "human_approved"    # User approved/corrected
    NO_SOURCES = "no_sources"        # No web sources found
    NOT_RUN = "not_run"


@dataclass
class DisputedLine:
    """A lyrics line where sources disagree."""
    line_number: int
    versions: dict[str, str]  # source_name → text
    gemini_recommendation: str | None = None
    gemini_confidence: float = 0.0


@dataclass
class ConsensusResult:
    """Result of consensus engine comparing multiple sources."""
    consensus_reached: bool
    agreed_sources: int
    lyrics: list[str]
    disputes: list[DisputedLine] = field(default_factory=list)


@dataclass
class CharChange:
    """A single character-level change."""
    position: int
    old_char: str
    new_char: str
    change_type: str  # "replaced", "added", "removed"


@dataclass
class CharDiff:
    """Character-level diff for one word."""
    word_index: int
    original_word: str
    corrected_word: str
    char_changes: list[CharChange] = field(default_factory=list)
    gemini_explanation: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_models_new.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add karaoke/models.py tests/test_models_new.py
git commit -m "feat(models): add dataclasses for consensus, char diff, and verification verdict"
```

---

## Task 2: Configuration Updates

**Files:**
- Modify: `karaoke/config.py` (add after GEMINI_MODEL, around line 131)

- [ ] **Step 1: Add Google API and consensus config to config.py**

Add after `_load_gemini_api_key()` (line 127) in `karaoke/config.py`:

```python
def _load_google_api_key() -> str:
    value = os.getenv("GOOGLE_API_KEY", "").strip()
    if value:
        return value
    for candidate in (BASE_DIR / ".env", BASE_DIR / ".env.local"):
        value = _load_env_value(candidate, {"GOOGLE_API_KEY"})
        if value:
            return value
    return ""

def _load_google_search_engine_id() -> str:
    value = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "").strip()
    if value:
        return value
    for candidate in (BASE_DIR / ".env", BASE_DIR / ".env.local"):
        value = _load_env_value(candidate, {"GOOGLE_SEARCH_ENGINE_ID"})
        if value:
            return value
    return ""
```

Then add after `GEMINI_MODEL` (line 131):

```python
# Google Custom Search API
GOOGLE_API_KEY: str = _load_google_api_key()
GOOGLE_SEARCH_ENGINE_ID: str = _load_google_search_engine_id()

# YouTube Data API (uses same GOOGLE_API_KEY)
YOUTUBE_API_ENABLED: bool = bool(GOOGLE_API_KEY)

# Consensus engine
CONSENSUS_MIN_SOURCES: int = 3  # minimum sources for auto-verification

# Verification loop
MAX_REVIEW_ITERATIONS: int = 2  # max round-trips through steps 4-6
```

- [ ] **Step 2: Verify config loads correctly**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -c "from karaoke.config import GOOGLE_API_KEY, CONSENSUS_MIN_SOURCES; print('OK', CONSENSUS_MIN_SOURCES)"`
Expected: `OK 3`

- [ ] **Step 3: Commit**

```bash
git add karaoke/config.py
git commit -m "feat(config): add Google Search API and consensus configuration"
```

---

## Task 3: Character-Level Timing Interpolation

**Files:**
- Modify: `karaoke/transcriber.py` (add interpolation after WordTiming creation)
- Create: `tests/test_char_timing.py`

- [ ] **Step 1: Write test for character timing interpolation**

```python
# tests/test_char_timing.py
from karaoke.transcriber import interpolate_character_timings
from karaoke.models import WordTiming, CharacterTiming

def test_interpolate_simple_word():
    word = WordTiming(word="שלום", start=0.0, end=0.4, confidence=0.9, source="draft_whisper", aligned=False)
    chars = interpolate_character_timings(word)
    assert len(chars) == 4  # שׁ ל ו ם
    assert chars[0].start == 0.0
    assert chars[-1].end == 0.4
    # All chars should cover the full duration without gaps
    for i in range(len(chars) - 1):
        assert abs(chars[i].end - chars[i + 1].start) < 0.001

def test_interpolate_word_with_niqqud():
    word = WordTiming(word="שָׁלוֹם", start=0.0, end=0.6, confidence=0.9, source="draft_whisper", aligned=False)
    chars = interpolate_character_timings(word)
    # Niqqud marks get lower weight, so consonants should have longer duration
    consonant_durations = [c.end - c.start for c in chars if not ('\u05B0' <= c.char <= '\u05C8')]
    niqqud_durations = [c.end - c.start for c in chars if '\u05B0' <= c.char <= '\u05C8']
    if niqqud_durations:
        assert max(niqqud_durations) < min(consonant_durations)

def test_interpolate_preserves_boundaries():
    word = WordTiming(word="אבג", start=1.5, end=2.1, confidence=0.9, source="draft_whisper", aligned=False)
    chars = interpolate_character_timings(word)
    assert chars[0].start == 1.5
    assert chars[-1].end == 2.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_char_timing.py -v`
Expected: ImportError — `interpolate_character_timings` doesn't exist

- [ ] **Step 3: Implement interpolation function**

Add to `karaoke/transcriber.py` (before `transcribe_hebrew` function, around line 89):

```python
from .models import CharacterTiming

def interpolate_character_timings(word: WordTiming) -> list[CharacterTiming]:
    """Interpolate character-level timing from word timing using grapheme weights.

    Uses the same grapheme weight logic as aligner.py: Hebrew niqqud marks
    receive lower weight than consonants for proportional time distribution.
    """
    from .aligner import _split_graphemes, _grapheme_weight

    graphemes = _split_graphemes(word.word)
    if not graphemes:
        return []

    weights = [_grapheme_weight(g) for g in graphemes]
    total_weight = sum(weights)
    if total_weight == 0:
        total_weight = len(graphemes)
        weights = [1.0] * len(graphemes)

    duration = word.end - word.start
    timings = []
    cursor = word.start

    for grapheme, weight in zip(graphemes, weights):
        char_duration = duration * (weight / total_weight)
        timings.append(CharacterTiming(
            char=grapheme,
            start=round(cursor, 4),
            end=round(cursor + char_duration, 4),
        ))
        cursor += char_duration

    # Snap last character end to word boundary
    if timings:
        timings[-1] = CharacterTiming(
            char=timings[-1].char,
            start=timings[-1].start,
            end=word.end,
        )

    return timings
```

- [ ] **Step 4: Add character timings to TranscriptDraft creation in transcribe()**

In `karaoke/transcriber.py`, inside `FasterWhisperHebrewProvider.transcribe()` (around line 64, after creating WordTiming), add `char_timings` field population. First add `char_timings` field to `WordTiming` in `models.py`:

In `karaoke/models.py`, add to `WordTiming` class (line 104):
```python
    char_timings: list["CharacterTiming"] = field(default_factory=list)
```

Then in `transcriber.py`, after creating each `WordTiming` object (around line 64), add:
```python
                wt.char_timings = interpolate_character_timings(wt)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_char_timing.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Run existing tests to check for regressions**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/ -v --timeout=30`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add karaoke/transcriber.py karaoke/models.py tests/test_char_timing.py
git commit -m "feat(transcriber): add character-level timing interpolation using grapheme weights"
```

---

## Task 4: Google Search Provider

**Files:**
- Create: `karaoke/google_search.py`
- Create: `tests/test_google_search.py`

- [ ] **Step 1: Write tests for Google search provider**

```python
# tests/test_google_search.py
from unittest.mock import patch, MagicMock
from karaoke.google_search import GoogleSearchProvider, YouTubeDescriptionProvider

def _mock_google_response():
    return {
        "items": [
            {
                "title": "שיר - שירונט",
                "link": "https://shironet.mako.co.il/artist?type=lyrics&lang=1&prfid=1&wrkid=1",
                "snippet": "מילות השיר...",
            },
            {
                "title": "שיר - Tab4U",
                "link": "https://www.tab4u.com/tabs/songs/1.html",
                "snippet": "אקורדים ומילים...",
            },
        ]
    }

@patch("karaoke.google_search.urllib.request.urlopen")
def test_google_search_returns_results(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = __import__("json").dumps(_mock_google_response()).encode()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    provider = GoogleSearchProvider(api_key="test", engine_id="test")
    results = provider.search("שיר מילים")
    assert len(results) == 2
    assert "shironet" in results[0].url

@patch("karaoke.google_search.urllib.request.urlopen")
def test_google_search_handles_empty_response(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"items": []}'
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    provider = GoogleSearchProvider(api_key="test", engine_id="test")
    results = provider.search("שיר לא קיים")
    assert len(results) == 0

@patch("karaoke.google_search.urllib.request.urlopen")
def test_google_search_handles_api_error(mock_urlopen):
    mock_urlopen.side_effect = Exception("API Error")
    provider = GoogleSearchProvider(api_key="test", engine_id="test")
    results = provider.search("שיר")
    assert len(results) == 0  # graceful fallback

@patch("karaoke.google_search.urllib.request.urlopen")
def test_youtube_search_extracts_description(mock_urlopen):
    yt_response = {
        "items": [{
            "id": {"videoId": "abc123"},
            "snippet": {
                "title": "שיר - מילים",
                "description": "שורה ראשונה\nשורה שנייה\nשורה שלישית",
            }
        }]
    }
    mock_response = MagicMock()
    mock_response.read.return_value = __import__("json").dumps(yt_response).encode()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    provider = YouTubeDescriptionProvider(api_key="test")
    results = provider.search("שיר מילים")
    assert len(results) == 1
    assert "שורה ראשונה" in results[0].snippet
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_google_search.py -v`
Expected: ImportError

- [ ] **Step 3: Implement GoogleSearchProvider and YouTubeDescriptionProvider**

```python
# karaoke/google_search.py
"""Google Custom Search API and YouTube Data API providers."""

import json
import logging
import urllib.request
import urllib.parse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """A single search result from Google or YouTube."""
    title: str
    url: str
    snippet: str


class GoogleSearchProvider:
    """Search via Google Custom Search JSON API."""

    ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, engine_id: str):
        self.api_key = api_key
        self.engine_id = engine_id

    def search(self, query: str, num: int = 10) -> list[SearchResult]:
        if not self.api_key or not self.engine_id:
            logger.warning("Google API key or engine ID not configured")
            return []
        params = urllib.parse.urlencode({
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": min(num, 10),
        })
        url = f"{self.ENDPOINT}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            return [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                )
                for item in data.get("items", [])
            ]
        except Exception as e:
            logger.warning("Google search failed: %s", e)
            return []


class YouTubeDescriptionProvider:
    """Search YouTube and extract video descriptions."""

    ENDPOINT = "https://www.googleapis.com/youtube/v3/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            return []
        params = urllib.parse.urlencode({
            "key": self.api_key,
            "q": query,
            "part": "snippet",
            "type": "video",
            "maxResults": max_results,
        })
        url = f"{self.ENDPOINT}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            return [
                SearchResult(
                    title=item["snippet"].get("title", ""),
                    url=f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    snippet=item["snippet"].get("description", ""),
                )
                for item in data.get("items", [])
                if "id" in item and "videoId" in item.get("id", {})
            ]
        except Exception as e:
            logger.warning("YouTube search failed: %s", e)
            return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_google_search.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add karaoke/google_search.py tests/test_google_search.py
git commit -m "feat(search): add Google Custom Search and YouTube description providers"
```

---

## Task 5: Consensus Engine

**Files:**
- Create: `karaoke/consensus.py`
- Create: `tests/test_consensus.py`

- [ ] **Step 1: Write tests for consensus engine**

```python
# tests/test_consensus.py
from karaoke.consensus import ConsensusEngine

def test_consensus_reached_with_3_matching_sources():
    sources = {
        "shironet": ["שורה אחת", "שורה שתיים"],
        "tab4u": ["שורה אחת", "שורה שתיים"],
        "baneshama": ["שורה אחת", "שורה שתיים"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert result.consensus_reached
    assert result.agreed_sources == 3
    assert result.lyrics == ["שורה אחת", "שורה שתיים"]
    assert len(result.disputes) == 0

def test_no_consensus_with_2_sources():
    sources = {
        "shironet": ["שורה אחת"],
        "tab4u": ["שורה אחת"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert not result.consensus_reached
    assert result.agreed_sources == 2

def test_dispute_detected_on_disagreement():
    sources = {
        "shironet": ["הלב שלי", "שורה שתיים"],
        "tab4u": ["הלב שלך", "שורה שתיים"],
        "baneshama": ["הלב שלי", "שורה שתיים"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert not result.consensus_reached  # not 100% on all lines
    assert len(result.disputes) == 1
    assert result.disputes[0].line_number == 0
    assert "shironet" in result.disputes[0].versions

def test_normalization_ignores_niqqud():
    sources = {
        "shironet": ["שָׁלוֹם"],
        "tab4u": ["שלום"],
        "baneshama": ["שלום"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert result.consensus_reached  # same after niqqud removal

def test_normalization_ignores_punctuation_and_whitespace():
    sources = {
        "shironet": [" שלום,  עולם! "],
        "tab4u": ["שלום עולם"],
        "baneshama": ["שלום עולם."],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert result.consensus_reached

def test_empty_sources():
    engine = ConsensusEngine()
    result = engine.evaluate({})
    assert not result.consensus_reached
    assert result.agreed_sources == 0

def test_mixed_language_normalization():
    sources = {
        "shironet": ["Hello שלום"],
        "tab4u": ["hello שלום"],
        "baneshama": ["HELLO שלום"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert result.consensus_reached  # case-insensitive for Latin

def test_partial_consensus_per_line():
    sources = {
        "shironet": ["שורה אחת", "שורה שתיים", "שורה שלוש"],
        "tab4u": ["שורה אחת", "שורה אחרת", "שורה שלוש"],
        "baneshama": ["שורה אחת", "שורה שתיים", "שורה שלוש"],
    }
    engine = ConsensusEngine()
    result = engine.evaluate(sources)
    assert not result.consensus_reached
    # Line 0 and 2 have consensus, only line 1 is disputed
    assert len(result.disputes) == 1
    assert result.disputes[0].line_number == 1
    # The agreed lines should still be in lyrics
    assert result.lyrics[0] == "שורה אחת"
    assert result.lyrics[2] == "שורה שלוש"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_consensus.py -v`
Expected: ImportError

- [ ] **Step 3: Implement consensus engine**

```python
# karaoke/consensus.py
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
    """Normalize a lyrics line for comparison.

    Removes niqqud, punctuation, extra whitespace.
    Lowercases Latin characters (case-insensitive for mixed languages).
    """
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
        """Evaluate consensus across sources.

        Args:
            sources: Mapping of source_name → list of lyrics lines

        Returns:
            ConsensusResult with consensus status, agreed lyrics, and disputes
        """
        if not sources:
            return ConsensusResult(
                consensus_reached=False,
                agreed_sources=0,
                lyrics=[],
                disputes=[],
            )

        # Find the maximum number of lines across sources
        max_lines = max(len(lines) for lines in sources.values())

        agreed_lyrics: list[str] = []
        disputes: list[DisputedLine] = []
        min_agreement = len(sources)  # track worst-case agreement

        for line_idx in range(max_lines):
            # Collect normalized versions for this line
            versions: dict[str, str] = {}  # source → original text
            normalized: dict[str, str] = {}  # source → normalized text

            for source_name, lines in sources.items():
                if line_idx < len(lines):
                    original = lines[line_idx]
                    versions[source_name] = original
                    normalized[source_name] = normalize_lyrics_line(original)

            # Count how many sources agree on each normalized version
            counts = Counter(normalized.values())
            most_common_text, most_common_count = counts.most_common(1)[0]

            if most_common_count < self.min_sources:
                min_agreement = min(min_agreement, most_common_count)
                # Find the original text that matches the most common normalized form
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
                # Use original text from the first source that matches
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_consensus.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add karaoke/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): add multi-source consensus engine with normalization"
```

---

## Task 6: Character-Level Diff Engine

**Files:**
- Create: `karaoke/char_diff.py`
- Create: `tests/test_char_diff.py`

- [ ] **Step 1: Write tests for character diff**

```python
# tests/test_char_diff.py
from karaoke.char_diff import compute_char_diffs
from karaoke.models import CharDiff

def test_diff_single_char_replacement():
    diffs = compute_char_diffs(
        original_words=["שלום"],
        corrected_words=["שלם"],
        word_indices=[0],
    )
    assert len(diffs) == 1
    assert diffs[0].original_word == "שלום"
    assert diffs[0].corrected_word == "שלם"
    assert any(c.change_type == "removed" for c in diffs[0].char_changes)

def test_diff_no_changes():
    diffs = compute_char_diffs(
        original_words=["שלום"],
        corrected_words=["שלום"],
        word_indices=[0],
    )
    assert len(diffs) == 0  # no diff if words are identical

def test_diff_multiple_words():
    diffs = compute_char_diffs(
        original_words=["הלב", "שלך"],
        corrected_words=["הלב", "שלי"],
        word_indices=[0, 1],
    )
    assert len(diffs) == 1  # only "שלך" → "שלי" changed
    assert diffs[0].word_index == 1
    assert diffs[0].original_word == "שלך"
    assert diffs[0].corrected_word == "שלי"

def test_diff_with_niqqud_change():
    diffs = compute_char_diffs(
        original_words=["שָׁלוֹם"],
        corrected_words=["שָׁלֵם"],
        word_indices=[0],
    )
    assert len(diffs) == 1
    assert len(diffs[0].char_changes) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_char_diff.py -v`
Expected: ImportError

- [ ] **Step 3: Implement character diff engine**

```python
# karaoke/char_diff.py
"""Character-level diff engine for Hebrew text."""

import difflib

from .models import CharChange, CharDiff


def compute_char_diffs(
    original_words: list[str],
    corrected_words: list[str],
    word_indices: list[int],
) -> list[CharDiff]:
    """Compute character-level diffs between original and corrected words.

    Args:
        original_words: Original words from Whisper
        corrected_words: Corrected words (same length as original)
        word_indices: Index of each word in the full transcript

    Returns:
        List of CharDiff for words that actually changed
    """
    diffs: list[CharDiff] = []

    for orig, corrected, idx in zip(original_words, corrected_words, word_indices):
        if orig == corrected:
            continue

        changes = _diff_chars(orig, corrected)
        diffs.append(CharDiff(
            word_index=idx,
            original_word=orig,
            corrected_word=corrected,
            char_changes=changes,
        ))

    return diffs


def _diff_chars(old: str, new: str) -> list[CharChange]:
    """Compute per-character changes between two strings."""
    changes: list[CharChange] = []
    matcher = difflib.SequenceMatcher(None, old, new)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            for pos in range(max(i2 - i1, j2 - j1)):
                old_c = old[i1 + pos] if i1 + pos < i2 else ""
                new_c = new[j1 + pos] if j1 + pos < j2 else ""
                if old_c and new_c:
                    changes.append(CharChange(position=i1 + pos, old_char=old_c, new_char=new_c, change_type="replaced"))
                elif old_c:
                    changes.append(CharChange(position=i1 + pos, old_char=old_c, new_char="", change_type="removed"))
                else:
                    changes.append(CharChange(position=j1 + pos, old_char="", new_char=new_c, change_type="added"))
        elif tag == "delete":
            for pos in range(i1, i2):
                changes.append(CharChange(position=pos, old_char=old[pos], new_char="", change_type="removed"))
        elif tag == "insert":
            for pos in range(j1, j2):
                changes.append(CharChange(position=pos, old_char="", new_char=new[pos], change_type="added"))

    return changes


def format_diff_table(diffs: list[CharDiff]) -> str:
    """Format diffs as a readable table for Gemini/user display."""
    lines = []
    for d in diffs:
        lines.append(f"מילה #{d.word_index}: \"{d.original_word}\" → \"{d.corrected_word}\"")
        for c in d.char_changes:
            if c.change_type == "replaced":
                lines.append(f"  מיקום {c.position}: '{c.old_char}' → '{c.new_char}'")
            elif c.change_type == "removed":
                lines.append(f"  מיקום {c.position}: '{c.old_char}' הוסרה")
            elif c.change_type == "added":
                lines.append(f"  מיקום {c.position}: '{c.new_char}' נוספה")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_char_diff.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add karaoke/char_diff.py tests/test_char_diff.py
git commit -m "feat(diff): add character-level diff engine for Hebrew text"
```

---

## Task 7: New Lyrics Verifier — Core Pipeline

This is the largest task. It replaces `HybridLyricsVerifier` with the new 7-step pipeline.

**Files:**
- Major rewrite: `karaoke/lyrics_verifier.py`
- Create: `tests/test_lyrics_verifier_v2.py`

- [ ] **Step 1: Write tests for new verification pipeline**

```python
# tests/test_lyrics_verifier_v2.py
from unittest.mock import patch, MagicMock
from karaoke.models import (
    TranscriptDraft, TranscriptSegment, WordTiming,
    VerificationVerdict, ConsensusResult, DisputedLine,
)
from karaoke.lyrics_verifier import MultiStepLyricsVerifier


def _draft():
    """Create a test TranscriptDraft."""
    words1 = [
        WordTiming(word="שלום", start=0.0, end=0.3, confidence=0.9, source="draft_whisper", aligned=False),
        WordTiming(word="עולם", start=0.3, end=0.6, confidence=0.9, source="draft_whisper", aligned=False),
    ]
    words2 = [
        WordTiming(word="הלב", start=1.0, end=1.3, confidence=0.9, source="draft_whisper", aligned=False),
        WordTiming(word="שלי", start=1.3, end=1.6, confidence=0.9, source="draft_whisper", aligned=False),
    ]
    seg1 = TranscriptSegment(words=words1)
    seg2 = TranscriptSegment(words=words2)
    return TranscriptDraft(segments=[seg1, seg2], provider="test")


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
def test_consensus_reached_skips_gemini(mock_search):
    """When 3+ sources agree, Gemini step is skipped."""
    mock_search.return_value = {
        "shironet": ["שלום עולם", "הלב שלי"],
        "tab4u": ["שלום עולם", "הלב שלי"],
        "baneshama": ["שלום עולם", "הלב שלי"],
    }
    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())
    assert result.verdict == VerificationVerdict.CONSENSUS.value
    assert result.confidence >= 0.9

@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_deep_verify")
def test_no_consensus_triggers_gemini(mock_gemini, mock_search):
    """When <3 sources agree, Gemini is called."""
    mock_search.return_value = {
        "shironet": ["שלום עולם", "הלב שלי"],
        "tab4u": ["שלום עולם", "הלב שלך"],  # disagrees on line 2
    }
    mock_gemini.return_value = (["שלום עולם", "הלב שלי"], 0.85, [])
    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())
    mock_gemini.assert_called_once()
    assert result.verdict == VerificationVerdict.GEMINI_VERIFIED.value

@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_knowledge_verify")
def test_zero_sources_uses_whisper_with_gemini(mock_gemini_kb, mock_search):
    """When no sources found, Whisper transcript goes to Gemini knowledge-based check."""
    mock_search.return_value = {}
    mock_gemini_kb.return_value = (["שלום עולם", "הלב שלי"], 0.5)
    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())
    mock_gemini_kb.assert_called_once()
    assert result.verdict == VerificationVerdict.NO_SOURCES.value

@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_deep_verify")
def test_gemini_failure_returns_best_available(mock_gemini, mock_search):
    """When Gemini API fails, return best available data with warning."""
    mock_search.return_value = {
        "shironet": ["שלום עולם", "הלב שלי"],
        "tab4u": ["שלום עולם", "הלב שלך"],
    }
    mock_gemini.side_effect = Exception("Gemini API timeout")
    verifier = MultiStepLyricsVerifier()
    result = verifier.verify("שיר לדוגמה", _draft())
    # Should not crash — returns best available with warning
    assert result.corrected_lines is not None
    assert any("gemini" in w.lower() or "שגיאה" in w for w in (result.local_warnings or []))


def test_result_has_dispute_info():
    """Verify result contains dispute information for UI."""
    dispute = DisputedLine(
        line_number=1,
        versions={"shironet": "הלב שלי", "tab4u": "הלב שלך"},
        gemini_recommendation="הלב שלי",
        gemini_confidence=0.85,
    )
    consensus = ConsensusResult(
        consensus_reached=False,
        agreed_sources=2,
        lyrics=["שלום עולם", "הלב שלי"],
        disputes=[dispute],
    )
    assert len(consensus.disputes) == 1
    assert consensus.disputes[0].versions["tab4u"] == "הלב שלך"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_lyrics_verifier_v2.py -v`
Expected: ImportError — `MultiStepLyricsVerifier` doesn't exist

- [ ] **Step 3: Implement MultiStepLyricsVerifier class**

This is the core implementation. Add `MultiStepLyricsVerifier` class to `karaoke/lyrics_verifier.py`, keeping the existing classes intact for backward compatibility. The new class uses:
- `GoogleSearchProvider` and `YouTubeDescriptionProvider` from `google_search.py`
- `ConsensusEngine` from `consensus.py`
- Existing `_extract_site_specific_lyrics`, `_strip_html_preserving_lines` functions
- Existing `GeminiLyricsVerifier._call_gemini` for Gemini API calls
- `concurrent.futures.ThreadPoolExecutor` for parallel search
- `char_diff.py` for diff computation (used in steps 5-7, called from bot.py)

Key methods to implement:
- `verify(title, draft) -> LyricsVerificationResult` — Steps 1-3
- `_search_all_sources(title, draft_text) -> dict[str, list[str]]` — Step 2
- `_search_single_source(query, source_config) -> list[str]` — per-source search
- `_gemini_deep_verify(sources, disputes, title) -> tuple[lyrics, confidence, uncertain_words]` — Step 3
- `_gemini_knowledge_verify(whisper_text, title) -> tuple[lyrics, confidence]` — Step 3 fallback
- `post_review_steps(job, original_draft)` — Steps 5-7 (called from bot.py after user approval)
- Existing helper functions (`_build_query_variants`, `_extract_site_specific_lyrics`, etc.) are reused

**DuckDuckGo fallback:** `_search_all_sources` should:
1. Try `GoogleSearchProvider` first
2. If Google raises an HTTP 429 (quota) or returns empty due to quota, fall back to existing `_search_duckduckgo_results` with `site:` queries
3. Track which search engine was used in logs

The implementation should:
1. Add `MultiStepLyricsVerifier` after the existing `HybridLyricsVerifier` class
2. Keep `HybridLyricsVerifier` for backward compatibility
3. Store `ConsensusResult` and `disputes` in the returned `LyricsVerificationResult` (use existing fields: `summary`, `local_warnings`, `options`)
4. Store dispute data as new fields in `LyricsVerificationResult` (add `consensus_result` and `source_versions` fields to models.py)
5. Handle Gemini API failures gracefully: if Gemini fails in Step 3, skip it and present best available data with a warning

**Note:** The full implementation is ~300-400 lines. The implementer should reference the spec document for exact behavior of each step, the existing `DuckDuckGoLyricsVerifier` for search/parse patterns, and `GeminiLyricsVerifier` for Gemini API call patterns.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_lyrics_verifier_v2.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run all existing tests**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/ -v --timeout=60`
Expected: All tests pass (old and new)

- [ ] **Step 6: Commit**

```bash
git add karaoke/lyrics_verifier.py karaoke/models.py tests/test_lyrics_verifier_v2.py
git commit -m "feat(verifier): add MultiStepLyricsVerifier with parallel search and consensus"
```

---

## Task 8: New Site Parsers (Baneshama, Nagina Mizrahit)

**Files:**
- Modify: `karaoke/lyrics_verifier.py` (add to `KNOWN_LYRICS_DOMAINS` and `_extract_site_specific_lyrics`)

- [ ] **Step 1: Research Baneshama and Nagina Mizrahit HTML structure**

Visit `baneshama.co.il` and the target "nagina mizrahit" site to identify:
- URL pattern for song pages
- CSS selectors / HTML elements containing lyrics text
- Any anti-scraping measures

Document findings before writing code.

- [ ] **Step 2: Add new domains to KNOWN_LYRICS_DOMAINS (line 28-37)**

```python
# Add to KNOWN_LYRICS_DOMAINS dict
"baneshama.co.il": "baneshama",
# Add nagina mizrahit domain once confirmed
```

- [ ] **Step 3: Add domain priority (line 51-60)**

```python
# Add to DOMAIN_PRIORITY
"baneshama.co.il": 1,  # High priority (Israeli lyrics site)
```

- [ ] **Step 4: Add HTML parsing in _extract_site_specific_lyrics (line 363-412)**

Follow existing pattern:
```python
elif domain_key == "baneshama":
    # Extract lyrics using site-specific selectors
    # Pattern will depend on Step 1 research findings
    pass
```

- [ ] **Step 5: Test with a known song**

Run manual test with a popular Hebrew song to verify parsing works.

- [ ] **Step 6: Commit**

```bash
git add karaoke/lyrics_verifier.py
git commit -m "feat(parsers): add Baneshama and Nagina Mizrahit lyrics site parsers"
```

---

## Task 9: Partial Re-Alignment Support

**Files:**
- Modify: `karaoke/aligner.py` (add partial alignment method)
- Create: `tests/test_partial_alignment.py`

- [ ] **Step 1: Write test for partial re-alignment**

```python
# tests/test_partial_alignment.py
from karaoke.aligner import realign_changed_words
from karaoke.models import WordTiming, TranscriptSegment, CharacterTiming
from karaoke.transcriber import interpolate_character_timings

def test_realign_redistributes_timing_for_shorter_word():
    original = WordTiming(
        word="שלום", start=0.0, end=0.4,
        confidence=0.9, source="draft_whisper", aligned=True,
    )
    original.char_timings = interpolate_character_timings(original)

    new_timings = realign_changed_words(
        original_word=original,
        corrected_text="שלם",
        audio_path=None,  # skip audio verification in this test
    )
    assert len(new_timings) == 3  # ש ל ם
    assert new_timings[0].start == 0.0
    assert new_timings[-1].end == 0.4

def test_realign_preserves_word_boundaries():
    original = WordTiming(
        word="אבגד", start=1.0, end=2.0,
        confidence=0.9, source="draft_whisper", aligned=True,
    )
    original.char_timings = interpolate_character_timings(original)

    new_timings = realign_changed_words(
        original_word=original,
        corrected_text="אבגדה",
        audio_path=None,
    )
    assert new_timings[0].start == 1.0
    assert new_timings[-1].end == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_partial_alignment.py -v`
Expected: ImportError

- [ ] **Step 3: Implement realign_changed_words function**

Add to `karaoke/aligner.py`:

```python
def realign_changed_words(
    original_word: WordTiming,
    corrected_text: str,
    audio_path: str | None = None,
) -> list[CharacterTiming]:
    """Recalculate character-level timing for a corrected word.

    Part A: Redistribute timing proportionally based on new graphemes.
    Part B: If audio_path provided, verify against wav2vec2 and prefer
            audio timing when gap > 50ms.

    Args:
        original_word: Original WordTiming with start/end boundaries
        corrected_text: The corrected word text
        audio_path: Path to vocals audio for verification (optional)

    Returns:
        List of CharacterTiming for the corrected word
    """
    from .models import CharacterTiming
    from .transcriber import interpolate_character_timings

    # Create a temporary WordTiming for the corrected text with same boundaries
    temp_word = WordTiming(
        word=corrected_text,
        start=original_word.start,
        end=original_word.end,
        confidence=original_word.confidence,
        source="corrected",
        aligned=False,
    )

    # Part A: Interpolate using grapheme weights
    new_timings = interpolate_character_timings(temp_word)

    # Part B: Audio verification (if audio available)
    if audio_path and new_timings:
        try:
            audio_features = _load_audio_features(audio_path)
            if audio_features:
                hop_seconds = audio_features.hop_seconds
                energy = audio_features.energy

                # _find_word_onset/offset expect a WordTiming as first arg
                onset = _find_word_onset(
                    temp_word, energy, hop_seconds, features=audio_features,
                )
                offset = _find_word_offset(
                    temp_word, energy, hop_seconds, features=audio_features,
                )

                # If audio timing differs by >50ms, prefer audio
                if onset is not None and abs(onset - original_word.start) > 0.05:
                    temp_word.start = onset
                    new_timings = interpolate_character_timings(temp_word)
                if offset is not None and abs(offset - original_word.end) > 0.05:
                    temp_word.end = offset
                    new_timings = interpolate_character_timings(temp_word)
        except Exception:
            pass  # Fall back to calculated timing

    return new_timings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_partial_alignment.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Run existing alignment tests**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_aligner.py -v --timeout=30`
Expected: All existing tests pass

- [ ] **Step 6: Commit**

```bash
git add karaoke/aligner.py tests/test_partial_alignment.py
git commit -m "feat(aligner): add partial re-alignment for corrected words"
```

---

## Task 10: Update Review UI in bot.py

**Files:**
- Modify: `bot.py` (update `_build_review_text` and `show_review_text`)

- [ ] **Step 1: Update _build_review_text to show disputes**

Modify `_build_review_text()` (lines 179-246 in `bot.py`) to:
- When consensus reached: show "✅ מילים אומתו מ-X מקורות"
- When disputes exist: highlight disputed lines with all source versions
- Format disputed lines as:
  ```
  שורה 5: "הלב שלי ⚠️"
    ├─ שירונט: "הלב שלי"
    ├─ Tab4U:   "הלב שלך"
    └─ Gemini:  "הלב שלי" (85%)
  ```

The implementation should check for `consensus_result` in the job's verification result and format accordingly.

- [ ] **Step 2: Update callback handlers for new verification flow**

Add handling for Steps 5-7 post-approval:
- After user approves/corrects (existing `karaoke_approve` callback)
- If corrections made: run char diff (Step 5), Gemini validation (Step 6), timing fix (Step 7)
- If Gemini finds issues (Step 6): return to review with notes
- Track iteration count (max 2 round-trips)

- [ ] **Step 3: Write automated tests for review text formatting**

```python
# tests/test_bot_review_ui.py
from unittest.mock import MagicMock
from karaoke.models import (
    Job, LyricsVerificationResult, ConsensusResult, DisputedLine, VerificationVerdict,
)

def test_build_review_text_with_consensus(tmp_path):
    """Consensus result shows checkmark and source count."""
    job = _make_test_job(tmp_path, consensus_reached=True, agreed_sources=3)
    text = _build_review_text(job)
    assert "✅" in text
    assert "3" in text

def test_build_review_text_with_disputes(tmp_path):
    """Disputed lines show ⚠️ with all source versions."""
    job = _make_test_job(tmp_path, consensus_reached=False, disputes=[
        DisputedLine(line_number=1, versions={"shironet": "הלב שלי", "tab4u": "הלב שלך"},
                     gemini_recommendation="הלב שלי", gemini_confidence=0.85),
    ])
    text = _build_review_text(job)
    assert "⚠️" in text
    assert "שירונט" in text or "shironet" in text
    assert "85%" in text or "0.85" in text

def test_loop_counter_limits_iterations():
    """After 2 round-trips, corrections are accepted unconditionally."""
    # This tests the iteration counter logic in the callback handler
    assert MAX_REVIEW_ITERATIONS == 2
```

- [ ] **Step 4: Run tests**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_bot_review_ui.py -v`
Expected: Tests pass

- [ ] **Step 5: Test manually with bot**

Start bot and test with a known Hebrew song to verify:
1. Dispute highlighting shows correctly
2. Approve/correct flow works
3. Loop limit works (max 2 iterations)

- [ ] **Step 6: Commit**

```bash
git add bot.py tests/test_bot_review_ui.py
git commit -m "feat(bot): update review UI with dispute highlighting and multi-step verification"
```

---

## Task 11: Pipeline Integration

**Files:**
- Modify: `karaoke/pipeline.py` (wire MultiStepLyricsVerifier)
- Modify: `karaoke/providers.py` (update LyricsVerifier protocol)

- [ ] **Step 1: Update providers.py with new protocol**

Add to `karaoke/providers.py`:
```python
class MultiStepLyricsVerifierProtocol(Protocol):
    def verify(self, title: str, draft: TranscriptDraft) -> LyricsVerificationResult: ...
    def post_review_steps(self, job: "Job", original_draft: TranscriptDraft) -> None: ...
```

- [ ] **Step 2: Update pipeline.py to use MultiStepLyricsVerifier**

In `KaraokePipeline.__init__()` (line 50-84), replace:
```python
self.lyrics_verifier = HybridLyricsVerifier()
```
with:
```python
from .lyrics_verifier import MultiStepLyricsVerifier
self.lyrics_verifier = MultiStepLyricsVerifier()
```

Add a new method for post-review steps:
```python
def step_post_review(self, job: Job, original_draft: TranscriptDraft):
    """Run Steps 5-7 after human approval (char diff, Gemini validate, timing fix)."""
    self.lyrics_verifier.post_review_steps(job, original_draft)
```

- [ ] **Step 3: Run full test suite**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/ -v --timeout=60`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add karaoke/pipeline.py karaoke/providers.py
git commit -m "feat(pipeline): wire MultiStepLyricsVerifier into karaoke pipeline"
```

---

## Task 12: Integration Testing

**Files:**
- Create: `tests/test_integration_verification.py`

- [ ] **Step 1: Write end-to-end integration test**

```python
# tests/test_integration_verification.py
"""Integration tests for the full lyrics verification pipeline."""
from unittest.mock import patch, MagicMock
from karaoke.models import TranscriptDraft, TranscriptSegment, WordTiming, VerificationVerdict
from karaoke.lyrics_verifier import MultiStepLyricsVerifier


def _make_draft(lines: list[str]) -> TranscriptDraft:
    segments = []
    t = 0.0
    for line in lines:
        words = []
        for w in line.split():
            words.append(WordTiming(word=w, start=t, end=t + 0.3, confidence=0.9, source="draft_whisper", aligned=False))
            t += 0.3
        segments.append(TranscriptSegment(words=words))
        t += 0.5
    return TranscriptDraft(segments=segments, provider="test")


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
def test_full_consensus_flow(mock_search):
    """Happy path: 3 sources agree → consensus → high confidence."""
    lyrics = ["שלום עולם", "הלב שלי", "שיר יפה"]
    mock_search.return_value = {
        "shironet": lyrics,
        "tab4u": lyrics,
        "baneshama": lyrics,
    }
    verifier = MultiStepLyricsVerifier()
    draft = _make_draft(lyrics)
    result = verifier.verify("שיר לדוגמה", draft)

    assert result.verdict == VerificationVerdict.CONSENSUS.value
    assert result.confidence >= 0.9
    assert result.corrected_lines == lyrics


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_deep_verify")
def test_dispute_flow_with_gemini(mock_gemini, mock_search):
    """Dispute path: sources disagree → Gemini decides."""
    mock_search.return_value = {
        "shironet": ["שלום עולם", "הלב שלי"],
        "tab4u": ["שלום עולם", "הלב שלך"],
    }
    mock_gemini.return_value = (["שלום עולם", "הלב שלי"], 0.85, [])

    verifier = MultiStepLyricsVerifier()
    draft = _make_draft(["שלום עולם", "הלב שלי"])
    result = verifier.verify("שיר לדוגמה", draft)

    assert result.verdict == VerificationVerdict.GEMINI_VERIFIED.value
    assert result.corrected_lines == ["שלום עולם", "הלב שלי"]


@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._search_all_sources")
@patch("karaoke.lyrics_verifier.MultiStepLyricsVerifier._gemini_knowledge_verify")
def test_no_sources_flow(mock_gemini_kb, mock_search):
    """No sources: falls back to Gemini knowledge-based check."""
    mock_search.return_value = {}
    mock_gemini_kb.return_value = (["שלום עולם"], 0.5)

    verifier = MultiStepLyricsVerifier()
    draft = _make_draft(["שלום עולם"])
    result = verifier.verify("שיר לדוגמה", draft)

    assert result.verdict == VerificationVerdict.NO_SOURCES.value
    assert "ניתוח אוטומטי" in (result.summary or "") or result.confidence < 0.7
```

- [ ] **Step 2: Run integration tests**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/test_integration_verification.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Run full test suite for final validation**

Run: `cd "D:/רעיונות/בוט טלגרם" && python -m pytest tests/ -v --timeout=120`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_verification.py
git commit -m "test: add integration tests for multi-step lyrics verification"
```

---

## Summary — Task Dependencies

```
Task 1 (Models) ─────────────────────────────┐
Task 2 (Config) ─────────────────────────────┤
Task 3 (Char Timing) ───────────────────────┤
Task 4 (Google Search) ─────────────────────┤
Task 5 (Consensus) ─────────────────────────┼──→ Task 7 (Core Verifier) ──→ Task 11 (Pipeline)
Task 6 (Char Diff) ─────────────────────────┤                               ↓
Task 8 (Site Parsers) ──────────────────────┘                          Task 10 (Bot UI)
Task 9 (Partial Alignment) ─────────────────────────────────────────────────→↓
                                                                       Task 12 (Integration)
```

**Tasks 1-6, 8, 9 can be developed in parallel** (no dependencies between them).
**Task 7** depends on Tasks 1-6.
**Tasks 10-11** depend on Task 7.
**Task 12** depends on all others.
