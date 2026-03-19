# Lyrics Verification Pipeline Redesign

**Date:** 2026-03-19
**Status:** Approved
**Priority:** Accuracy of lyrics is the #1 goal

## Overview

Redesign the lyrics verification flow in `karaoke/lyrics_verifier.py` to maximize word accuracy through multi-source consensus, deep Gemini verification, human approval, and character-level timing correction.

## Current Flow (Before)

```
Whisper transcribe → DuckDuckGo search → single-source match (ratio 0.45+) → Gemini fallback → show to user
```

**Problems:**
- Relies on single source with low threshold (0.45)
- Gemini only used as fallback, not as verifier
- No consensus mechanism
- No character-level diff or timing correction after edits

## New Flow (After) — 7 Steps

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Whisper Transcription (character-level timing)     │
│  ↓                                                          │
│  Step 2: Parallel search across 8 sources                   │
│  ↓                                                          │
│  ┌─ 3+ sources agree 100%? ─── YES ──→ Skip Step 3 ────┐  │
│  │                                                       │  │
│  └─ NO ──→ Step 3: Gemini deep verification ─────────────┤  │
│                                                          ↓  │
│  Step 4: Human approval (disputes highlighted)              │
│  ↓                                                          │
│  ┌─ No corrections made? ─── YES ──→ Done (go to align) ┐  │
│  │                                                       │  │
│  └─ Corrections made ──→ Step 5: Diff + Gemini explain ──┤  │
│                          ↓                                  │
│                          Step 6: Gemini validates swaps     │
│                          ↓                                  │
│                          ┌─ Issues? → Back to Step 4 ───┐  │
│                          └─ OK → Step 7: Timing fix ─────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Whisper Transcription with Interpolated Character-Level Timing

**What changes:** After Whisper produces word-level timestamps, **interpolate** character-level timing from word boundaries. Whisper does NOT produce character-level timestamps natively — they must be synthesized.

**Interpolation method:** Use the existing grapheme-weight logic from `aligner.py` (`_split_graphemes` / `_grapheme_weight`) to distribute word duration proportionally across characters. Hebrew niqqud marks receive lower weight than consonants.

**Input:** `vocals.wav` (isolated vocal track from Demucs)
**Output:** `TranscriptDraft` with:
- Word-level timestamps (existing, from Whisper)
- **Character-level timestamps** (new, interpolated from word timing using grapheme weights)
- Raw transcript text

**Note:** These interpolated timings are approximate. Step 7 refines them via wav2vec2 audio verification.

---

## Step 2: Parallel Multi-Source Search + Consensus

### Sources (8 total, searched in parallel via ThreadPoolExecutor)

**Concurrency:** Uses `concurrent.futures.ThreadPoolExecutor` (not asyncio) to match the existing synchronous codebase architecture.

| Source | Search Method | Priority |
|--------|-------------|----------|
| Shironet | Google Custom Search API (site:shironet.mako.co.il) + HTML parse | High |
| Tab4U | Google Custom Search API (site:tab4u.com) + HTML parse | High |
| Baneshama | Google Custom Search API (site:baneshama.co.il) + HTML parse | High |
| Nagnu | Google Custom Search API (site:nagnu.co.il) + HTML parse | Medium |
| Nagina Mizrahit | Google Custom Search API + HTML parse | Medium |
| Google | Google Custom Search API (general) | Medium |
| YouTube | YouTube Data API v3 — search + extract description | Medium |
| Genius | Google Custom Search API (site:genius.com) + HTML parse | Low |

**New source parsers needed:**
- **Baneshama (baneshama.co.il):** Requires investigation of HTML structure and lyrics extraction selectors during implementation.
- **Nagina Mizrahit:** Requires confirmation of exact domain and HTML structure during implementation.
- These will be implemented following the same pattern as existing parsers in `_extract_site_specific_lyrics`.

### Search Infrastructure

- **Primary engine:** Google Custom Search JSON API (100 free searches/day)
- **Query optimization:** The Google CSE is configured with all target sites in its search engine settings, so a single search query can return results from multiple sites simultaneously. This reduces API calls from 8 (one per site) to 2-3 (batched queries), allowing ~30-50 songs/day on the free tier.
- **Fallback engine:** DuckDuckGo HTML scraping (when Google quota exhausted). DuckDuckGo fallback uses the existing `site:` query approach from the current codebase. Sources that don't return results via DuckDuckGo are skipped gracefully.
- **YouTube:** YouTube Data API v3 (10,000 units/day, ~100 searches)
- **API Keys required in .env:**
  ```
  GOOGLE_API_KEY=<your_google_api_key>
  GOOGLE_SEARCH_ENGINE_ID=<your_search_engine_id>
  ```

### Consensus Engine

1. Normalize text from each source:
   - Remove extra whitespace
   - Remove niqqud (for comparison only, preserve in output)
   - Remove punctuation
   - Normalize Unicode forms

2. Compare line-by-line across all sources

3. **Consensus rule: 3+ sources agree on 100% of words → lyrics are verified**
   - Skip Step 3 (Gemini) entirely
   - Go directly to Step 4 (human approval) with high confidence

4. If no consensus:
   - Mark words/lines with disagreement between sources
   - Pass all versions to Step 3

5. **If 0 sources return lyrics** (obscure/new song):
   - Send Whisper transcript to Gemini with a knowledge-based prompt (not arbitration): "Here is a transcription of the song [name] by [artist]. Verify and correct the lyrics based on your knowledge."
   - If Gemini is also uncertain, present raw Whisper transcript to user with warning: "⚠️ לא נמצאו מקורות — המילים מבוססות על ניתוח אוטומטי בלבד"
   - Proceed to Step 4 for human review

### Mixed-Language Songs

For songs mixing Hebrew with other languages (English, Arabic, Russian — common in Israeli music):
- Consensus comparison normalizes non-Hebrew tokens separately (case-insensitive for Latin script)
- Gemini prompt includes a note about the song's detected language mix
- Language detection from Step 1 (`language_detector.py`) informs this behavior

---

## Step 3: Gemini Deep Verification (Conditional)

**When it runs:** Only when fewer than 3 sources agree on 100% of lyrics.

**Input to Gemini:**
- All versions from web sources (NO Whisper transcript — deliberate exclusion for independent verification. Whisper data IS used in Steps 5-6 for diff explanation, but not here.)
- Song name + artist name
- Which words/lines have disagreement between sources

**Gemini behavior:**
- Model: `gemini-2.5-flash` with high `thinking_budget` (deep mode)
- Decides which version is correct from the available sources
- **Self-verification prompt:** Forces Gemini to double-check its answer:
  "Review your answer again. Are you 100% certain? If not, mark uncertain words."
- If not confident → marks words as "requires human approval"

**Output:**
- Gemini's recommended lyrics
- Confidence level per line (0.0-1.0)
- List of words Gemini is uncertain about

---

## Step 4: Human Approval via Telegram

### Display Format

**When consensus was reached (3+ sources):**
```
✅ מילים אומתו מ-X מקורות

[full lyrics displayed]

אשר ✅ | תקן ✏️
```

**When Gemini decided (no consensus):**
- Lines with agreement → displayed normally
- Lines with **dispute** → highlighted with all versions:
  ```
  שורה 5: "הלב שלי ⚠️"
    ├─ שירונט: "הלב שלי"
    ├─ Tab4U:   "הלב שלך"
    ├─ בנשמה:  "הלב שלי"
    └─ Gemini:  "הלב שלי" (85%)
  ```

### User Actions
1. ✅ Approve all
2. Correct specific line: `5: הלב שלי`
3. Upload `transcript.txt` for full replacement (plain text, one line per lyric line). Full replacement still goes through Steps 5-7 (diff against Whisper, Gemini validation). If similarity to all sources < 0.15, show warning but accept user's text.

### Flow After Approval
- **No corrections** → skip steps 5-7, proceed to alignment
- **Corrections made** → continue to Step 5

---

## Step 5: Character-Level Diff + Gemini Explanation

**When it runs:** Only if user made corrections in Step 4.

### Part A — Automatic Diff

Compare original Whisper transcript vs. approved text at **character level**:

```
Original:  "שָׁלוֹם"  →  Corrected: "שָׁלֵם"
Change: char 4: וֹ→ֵ  |  char 5: ם (unchanged)  |  removed: char 6 (ו)
```

Output: Full diff table (old word ↔ new word ↔ changed characters/niqqud)

### Part B — Gemini Explains

Gemini receives:
- The character-level diff
- Audio context (what Whisper heard)

Gemini explains **why** Whisper made each mistake:
```
"Whisper heard 'וֹ' instead of 'ֵ' because in Mizrahi pronunciation,
 tzere and holam sound similar in this context"
```

Gemini also flags if it thinks a user correction is **wrong** (user made a mistake).

**Output:**
- Diff table with explanations
- Flag if any correction seems incorrect

---

## Step 6: Gemini Validates All Replacements

**When it runs:** Immediately after Step 5, automatically.

**Input to Gemini:**
- Diff table from Step 5
- Its own explanation from Step 5
- All source versions from Step 2
- Song name + artist

**What Gemini checks:**
- Are replacements consistent with web sources?
- Was any correct word replaced by mistake?
- Are there words that should have been replaced but weren't?
- Self-verification: "Check again — are all replacements logical?"

**Output — 3 possible states:**

| State | Action |
|-------|--------|
| ✅ All valid | Continue to Step 7 |
| ⚠️ Problem found | Return to Step 4 with Gemini's notes to user (max 2 round-trips, then accept user's corrections) |
| 🔄 Additional fix suggested | Show to user for approval before continuing |

**Loop limit:** Steps 4→5→6→4 can repeat at most **2 times**. After 2 round-trips, accept the user's corrections unconditionally and proceed to Step 7. This prevents infinite loops.

---

## Step 7: Timing Correction + Audio Verification

**When it runs:** Only on words that were replaced (not the entire song).

### Part A — Character-Level Timing Recalculation

Take the **interpolated** per-character timing from Step 1 (derived from word-level Whisper timing via grapheme weights) and redistribute for the new word:

```
Original: "שָׁלוֹם" (0.00s - 0.60s)
  שׁ = 0.00-0.12s | ָ = 0.12-0.18s | ל = 0.18-0.30s | וֹ = 0.30-0.42s | ם = 0.42-0.60s

Corrected: "שָׁלֵם" (4 characters instead of 5)
  שׁ = 0.00-0.12s | ָ = 0.12-0.18s | ל = 0.18-0.30s | ֵ = 0.30-0.45s | ם = 0.45-0.60s
```

Rules:
- Preserve original word start/end boundaries
- Redistribute time proportionally based on new character count
- Preserve timing of unchanged characters at word boundaries

### Part B — Audio Verification (wav2vec2)

- Run alignment **only on changed words + neighboring words (±1)**
- The current aligner operates on full segment lists — requires modification to accept a subset of word indices and their audio time range for partial re-alignment
- Compare calculated timing (Part A) vs. audio alignment
- If gap > **50ms** → prefer audio timing over calculated timing

**Output:**
- Updated `timings.json` with corrected per-character and per-word timing
- Report: number of words corrected, timing gaps found and fixed

---

## Architecture Changes

### New/Modified Files

| File | Change |
|------|--------|
| `karaoke/lyrics_verifier.py` | Major rewrite — new 7-step pipeline |
| `karaoke/config.py` | Add Google API key, Search Engine ID, YouTube API config |
| `karaoke/providers.py` | Add Google Search provider, YouTube provider protocols |
| `karaoke/consensus.py` | **New** — consensus engine (normalize, compare, vote) |
| `karaoke/char_diff.py` | **New** — character-level diff engine |
| `karaoke/models.py` | Add CharacterTiming, DiffResult, ConsensusResult models |
| `karaoke/transcriber.py` | Add character-level timing interpolation (from word timing + grapheme weights) |
| `karaoke/aligner.py` | Support partial re-alignment (changed words only) |
| `bot.py` | Update review UI — dispute highlighting, version display |
| `.env` | Add GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID |

### New Data Models

```python
@dataclass
class CharacterTiming:
    char: str
    start: float
    end: float

@dataclass
class ConsensusResult:
    consensus_reached: bool  # 3+ sources agreed
    agreed_sources: int
    lyrics: list[str]  # agreed lyrics (or best candidate)
    disputes: list[DisputedLine]  # lines with disagreement

@dataclass
class DisputedLine:
    line_number: int
    versions: dict[str, str]  # source_name → text
    gemini_recommendation: str | None = None
    gemini_confidence: float = 0.0

@dataclass
class CharDiff:
    word_index: int
    original_word: str
    corrected_word: str
    char_changes: list[CharChange]  # per-character diffs
    gemini_explanation: str | None

@dataclass
class CharChange:
    position: int
    old_char: str
    new_char: str
    change_type: str  # "replaced", "added", "removed"
```

### Gemini API Calls Summary

| Step | Call | Condition |
|------|------|-----------|
| 3 | Deep verification | No consensus |
| 5 | Explain diff | User made corrections |
| 6 | Validate replacements | User made corrections |

**Max Gemini calls per song:** 3 (worst case), 0 (best case with consensus + no corrections)

---

## Performance Estimates

| Step | Duration | Notes |
|------|----------|-------|
| 1 | ~3-5 min | Unchanged |
| 2 | ~15-30 sec | Parallel search + HTML parsing (vs. ~60-120 sec serial today) |
| 3 | ~10-20 sec | Gemini deep mode (skipped if consensus) |
| 4 | User time | Waiting for human |
| 5 | ~5-10 sec | Diff + Gemini explain (skipped if no corrections) |
| 6 | ~5-10 sec | Gemini validate (skipped if no corrections) |
| 7 | ~5-15 sec | Partial alignment only (vs. ~60-120 sec full) |

**Best case (consensus + no corrections):** Steps 3,5,6,7 skipped → ~15-30 sec total for verification
**Worst case (no consensus + corrections):** ~90 sec total for verification

**Note:** Step 2 estimates account for full HTTP round-trips including HTML page fetching and parsing, not just API calls.

---

## Environment Variables

```env
# Existing
TELEGRAM_TOKEN=...
GEMINI_API_KEY=...

# New
GOOGLE_API_KEY=<your_google_api_key>
GOOGLE_SEARCH_ENGINE_ID=<your_search_engine_id>
```

---

## Edge Cases & Error Handling

| Scenario | Behavior |
|----------|----------|
| 0 sources return lyrics | Whisper transcript sent to Gemini for knowledge-based verification, then to user with warning |
| Google API quota exhausted | Falls back to DuckDuckGo scraping; sources unavailable via DDG are skipped |
| Gemini API fails | Skip Gemini steps, present best available data to user with warning |
| User uploads completely different transcript | Goes through diff + Gemini validation; warning if similarity < 0.15 to all sources |
| Song has no results anywhere | Raw Whisper transcript shown with "⚠️ ניתוח אוטומטי בלבד" warning |
| Mixed Hebrew + other language | Consensus normalizes non-Hebrew tokens separately; Gemini gets language hint |
| Steps 4→6 loop exceeds 2 iterations | Accept user corrections unconditionally, proceed to Step 7 |
