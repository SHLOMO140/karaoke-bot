"""Automatic repair helpers for delivery feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess

from karaoke import job_manager
from karaoke.config import (
    BASE_DIR,
    CODEX_AUTO_REPAIR_COMMAND,
    CODEX_AUTO_REPAIR_ENABLED,
    CODEX_AUTO_REPAIR_SANDBOX,
    CODEX_AUTO_REPAIR_TIMEOUT_SECONDS,
)
from karaoke.models import Job, TranscriptSegment


_LINE_EDIT_PATTERN = re.compile(
    r"^\s*(?:(?:line|row|\u05e9\u05d5\u05e8\u05d4)\s*)?(\d{1,4})\s*[:\-\u2013]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_LEADING_BULLET_PATTERN = re.compile(r"^\s*[-*\u2022]\s*")
_EMPTY_PLACEHOLDER_VALUES = {"", "-", "\u2014", "\u2013", "...", "."}
_TIMING_FEEDBACK_KEYWORDS = (
    "timing",
    "sync",
    "synchron",
    "out of time",
    "out-of-time",
    "early",
    "late",
    "\u05dc\u05d0 \u05d1\u05d6\u05de\u05df",
    "\u05dc\u05d0 \u05de\u05e1\u05d5\u05e0\u05db\u05e8",
    "\u05dc\u05d0 \u05de\u05e1\u05d5\u05e0\u05db\u05e8\u05df",
    "\u05de\u05e1\u05d5\u05e0\u05db\u05e8",
    "\u05de\u05e1\u05d5\u05e0\u05db\u05e8\u05df",
    "\u05e1\u05d9\u05e0\u05db\u05e8\u05d5\u05df",
    "\u05e1\u05e0\u05db\u05e8\u05d5\u05df",
    "\u05d8\u05d9\u05d9\u05de\u05d9\u05e0\u05d2",
    "\u05de\u05e7\u05d3\u05d9\u05dd",
    "\u05de\u05d0\u05d7\u05e8",
    "\u05de\u05d0\u05d5\u05d7\u05e8",
    "\u05dc\u05e4\u05e0\u05d9 \u05d4\u05e7\u05d5\u05dc",
    "\u05d0\u05d7\u05e8\u05d9 \u05d4\u05e7\u05d5\u05dc",
    "\u05dc\u05d0 \u05d9\u05d5\u05e9\u05d1",
    "\u05dc\u05d0 \u05d9\u05d5\u05e9\u05d1\u05d5\u05ea",
    "\u05dc\u05d0 \u05d9\u05d5\u05e9\u05d1\u05d9\u05dd",
)


@dataclass(frozen=True)
class FeedbackLineEdit:
    line_number: int
    text: str
    source_line: str


@dataclass
class FeedbackRepairResult:
    applied: bool
    message: str
    edit_count: int = 0
    line_numbers: list[int] = field(default_factory=list)
    error: str = ""


@dataclass
class CodexRepairResult:
    attempted: bool
    enabled: bool
    success: bool = False
    message: str = ""
    log_path: Path | None = None
    returncode: int | None = None


def extract_feedback_line_edits(feedback_text: str) -> list[FeedbackLineEdit]:
    """Parse concrete line edits from free-form delivery feedback."""
    by_line_number: dict[int, FeedbackLineEdit] = {}
    for raw_line in feedback_text.splitlines():
        line = _LEADING_BULLET_PATTERN.sub("", raw_line).strip()
        if not line:
            continue
        match = _LINE_EDIT_PATTERN.match(line)
        if not match:
            continue
        corrected_text = match.group(2).strip()
        if corrected_text in _EMPTY_PLACEHOLDER_VALUES:
            continue
        line_number = int(match.group(1))
        by_line_number[line_number] = FeedbackLineEdit(
            line_number=line_number,
            text=corrected_text,
            source_line=raw_line,
        )
    return [by_line_number[key] for key in sorted(by_line_number)]


def feedback_mentions_timing_problem(feedback_text: str) -> bool:
    normalized = " ".join((feedback_text or "").lower().split())
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _TIMING_FEEDBACK_KEYWORDS)


def apply_feedback_line_edits(
    visible_segments: list[TranscriptSegment],
    reference_segments: list[TranscriptSegment],
    edits: list[FeedbackLineEdit],
) -> list[TranscriptSegment]:
    if not edits:
        return visible_segments

    updated = list(visible_segments)
    for edit in edits:
        index = edit.line_number - 1
        if index < 0 or index >= len(updated):
            raise ValueError(f"Line {edit.line_number} is outside the current review transcript.")
        reference_segment = (
            reference_segments[index]
            if len(reference_segments) == len(visible_segments) and index < len(reference_segments)
            else updated[index]
        )
        updated[index] = job_manager.update_transcript_line([reference_segment], 1, edit.text)[0]
    return updated


def apply_feedback_to_review(job: Job, feedback_text: str) -> FeedbackRepairResult:
    """Apply concrete feedback edits to the review transcript before regeneration."""
    edits = extract_feedback_line_edits(feedback_text)
    if not edits:
        return FeedbackRepairResult(
            applied=False,
            message="No concrete line edits were found in the feedback.",
        )

    try:
        review_segments = job_manager.load_review_segments(job)
        draft_segments = job_manager.load_draft_segments(job) if job.draft_timings_path.exists() else review_segments
    except FileNotFoundError as exc:
        return FeedbackRepairResult(
            applied=False,
            message="No review transcript is available for automatic line repair.",
            error=str(exc),
            line_numbers=[edit.line_number for edit in edits],
        )
    try:
        updated_segments = apply_feedback_line_edits(review_segments, draft_segments, edits)
    except ValueError as exc:
        return FeedbackRepairResult(
            applied=False,
            message=str(exc),
            error=str(exc),
            line_numbers=[edit.line_number for edit in edits],
        )

    job_manager.save_review_transcript(job, updated_segments)
    job_manager.save_manual_review_option(job, updated_segments, label="\u05ea\u05d9\u05e7\u05d5\u05df \u05d0\u05d5\u05d8\u05d5\u05de\u05d8\u05d9 \u05de\u05de\u05e9\u05d5\u05d1")
    line_numbers = [edit.line_number for edit in edits]
    return FeedbackRepairResult(
        applied=True,
        message=f"Applied {len(edits)} feedback line edits.",
        edit_count=len(edits),
        line_numbers=line_numbers,
    )


def _build_codex_repair_prompt(job: Job, feedback_text: str) -> str:
    return f"""You are running inside a Hebrew karaoke Telegram bot repository.

A user rejected a generated output and sent this delivery feedback:

{feedback_text.strip()}

Job context:
- job_id: {job.job_id}
- title: {job.display_name}
- job_dir: {job.job_dir}
- input_type: {job.input_type}
- review_transcript: {job.review_transcript_path}
- final_transcript: {job.transcript_path}
- delivery_feedback_log: {job.delivery_feedback_path}

Task:
1. Decide whether the complaint points to a bot code bug or a data/output issue.
2. If a code fix is needed, edit the smallest safe set of files in this repo.
3. Add or update tests that prevent the issue from recurring.
4. Run targeted tests when practical.
5. Do not commit, do not delete unrelated files, and do not revert existing user changes.

Return a concise summary of what you changed and what you verified.
"""


def run_codex_auto_repair(job: Job, feedback_text: str) -> CodexRepairResult:
    """Optionally run Codex CLI for code-level repair feedback."""
    if not CODEX_AUTO_REPAIR_ENABLED:
        return CodexRepairResult(
            attempted=False,
            enabled=False,
            message="Codex auto repair is disabled.",
        )

    prompt = _build_codex_repair_prompt(job, feedback_text)
    command = [
        CODEX_AUTO_REPAIR_COMMAND,
        "-a",
        "never",
        "exec",
        "-s",
        CODEX_AUTO_REPAIR_SANDBOX,
        "-C",
        str(BASE_DIR),
        "-",
    ]
    log_path = job.job_dir / "codex_auto_repair.log"
    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=CODEX_AUTO_REPAIR_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        message = f"Codex command was not found: {CODEX_AUTO_REPAIR_COMMAND}"
        log_path.write_text(f"{message}\n{exc}\n", encoding="utf-8")
        return CodexRepairResult(True, True, False, message, log_path)
    except subprocess.TimeoutExpired as exc:
        message = "Codex auto repair timed out."
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        log_path.write_text(f"{message}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n", encoding="utf-8")
        return CodexRepairResult(True, True, False, message, log_path)

    log_path.write_text(
        "\n".join(
            [
                "COMMAND:",
                " ".join(command),
                "",
                f"RETURNCODE: {result.returncode}",
                "",
                "STDOUT:",
                result.stdout or "",
                "",
                "STDERR:",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        return CodexRepairResult(
            attempted=True,
            enabled=True,
            success=False,
            message="Codex auto repair failed. See the job log for details.",
            log_path=log_path,
            returncode=result.returncode,
        )
    return CodexRepairResult(
        attempted=True,
        enabled=True,
        success=True,
        message="Codex auto repair completed. Restart the bot to load code changes if files were edited.",
        log_path=log_path,
        returncode=result.returncode,
    )
