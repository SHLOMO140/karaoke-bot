"""Tests for spec steps 5-7 (post_review_steps) — L5."""

from karaoke import job_manager
from karaoke.lyrics_verifier import MultiStepLyricsVerifier
from karaoke.models import TranscriptDraft, TranscriptSegment, WordTiming


def _draft(words):
    segments = [
        TranscriptSegment(
            words=[WordTiming(word, float(i), float(i) + 0.4) for i, word in enumerate(words)],
            text=" ".join(words),
            start=0.0,
            end=float(len(words)),
        )
    ]
    return TranscriptDraft(segments=segments, provider="test")


def _aligned_segments(words, *, changed_index=None):
    aligned_words = []
    for i, word in enumerate(words):
        is_changed = changed_index is not None and i == changed_index
        aligned_words.append(
            WordTiming(
                word,
                float(i),
                float(i) + 0.4,
                confidence=0.3 if is_changed else 0.9,
                source="review_hint" if is_changed else "whisperx",
                aligned=not is_changed,
            )
        )
    return [
        TranscriptSegment(
            words=aligned_words,
            text=" ".join(words),
            start=0.0,
            end=float(len(words)),
        )
    ]


def test_post_review_records_diff_and_realigns_changed_word(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="שיר", input_type="audio_file")

    draft = _draft(["שלום", "עולם", "טוב"])
    aligned = _aligned_segments(["שלום", "עולמי", "טוב"], changed_index=1)

    MultiStepLyricsVerifier().post_review_steps(job, draft, aligned)

    diff = job.manifest.post_review_diff
    assert diff["changed_words"] == 1
    assert diff["realigned_words"] == 1
    assert "עולם" in diff["diff_table"] and "עולמי" in diff["diff_table"]

    changed_word = aligned[0].words[1]
    assert changed_word.char_timings, "changed word must get rebuilt char timings"
    assert [ct.char for ct in changed_word.char_timings] == list("עולמי")


def test_post_review_leaves_whisperx_words_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="שיר", input_type="audio_file")

    draft = _draft(["שלום", "עולם"])
    aligned = _aligned_segments(["שלום", "עולם"])

    MultiStepLyricsVerifier().post_review_steps(job, draft, aligned)

    assert job.manifest.post_review_diff == {"changed_words": 0, "realigned_words": 0}
    assert not aligned[0].words[0].char_timings


def test_post_review_warns_on_massive_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="שיר", input_type="audio_file")

    original = [f"מילה{i}" for i in range(15)]
    edited = [f"אחרת{i}" for i in range(15)]
    draft = _draft(original)
    aligned = _aligned_segments(edited, changed_index=None)
    for word in aligned[0].words:
        word.aligned = False
        word.source = "review_hint"

    MultiStepLyricsVerifier().post_review_steps(job, draft, aligned)

    assert job.manifest.post_review_diff["changed_words"] == 15
    assert any("שינויי מילים" in warning for warning in job.manifest.warnings)


def test_post_review_noop_without_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    job = job_manager.create_job(title="שיר", input_type="audio_file")

    MultiStepLyricsVerifier().post_review_steps(job, _draft(["שלום"]), None)

    assert job.manifest.post_review_diff == {}
