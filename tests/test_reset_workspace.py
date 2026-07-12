import importlib.util
import json
import re
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import bot
from karaoke import job_manager
from karaoke.models import JobStatus, ReviewStatus

_RESET_PATH = Path(__file__).resolve().parent.parent / "tools" / "reset_workspace.py"
_spec = importlib.util.spec_from_file_location("reset_workspace_under_test", _RESET_PATH)
reset_workspace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reset_workspace)


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(job_manager, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(reset_workspace, "JOBS_DIR", tmp_path)


def _make_job(*, title, status, review=None, age_hours=0.0, pending_status=None):
    job = job_manager.create_job(title=title, input_type="audio_file")
    job_manager.update_status(job, status)
    if review is not None:
        job_manager.update_review_status(job, review)
    if pending_status is not None:
        job.manifest.pending_delivery = {"status": pending_status}
    if age_hours:
        job.manifest.updated_at = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    job_manager._write_json(job.manifest_path, asdict(job.manifest))
    return job


def test_purge_jobs_skips_in_flight_and_deletes_stale(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)

    processing = _make_job(title="busy", status=JobStatus.TRANSCRIBING, age_hours=100)
    open_review = _make_job(
        title="review",
        status=JobStatus.AWAITING_REVIEW,
        review=ReviewStatus.AWAITING_REVIEW,
        age_hours=10,
    )
    pending = _make_job(
        title="pending",
        status=JobStatus.DONE,
        review=ReviewStatus.APPROVED,
        age_hours=100,
        pending_status="pending_approval",
    )
    recent = _make_job(title="recent", status=JobStatus.DONE, review=ReviewStatus.APPROVED, age_hours=1)
    stale = _make_job(title="stale", status=JobStatus.DONE, review=ReviewStatus.APPROVED, age_hours=240)

    report = reset_workspace.ResetReport(dry_run=False)
    reset_workspace.purge_jobs(report, force=False, recent_hours=6)

    assert (tmp_path / processing.job_id).exists()
    assert (tmp_path / open_review.job_id).exists()
    assert (tmp_path / pending.job_id).exists()
    assert (tmp_path / recent.job_id).exists()
    assert not (tmp_path / stale.job_id).exists()
    skipped_ids = {entry.split("(")[0] for entry in report.skipped_jobs}
    assert skipped_ids == {processing.job_id, open_review.job_id, pending.job_id, recent.job_id}


def test_purge_jobs_force_deletes_in_flight_jobs(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    processing = _make_job(title="busy", status=JobStatus.TRANSCRIBING, age_hours=1)

    report = reset_workspace.ResetReport(dry_run=False)
    reset_workspace.purge_jobs(report, force=True, recent_hours=6)

    assert not (tmp_path / processing.job_id).exists()


def test_prune_state_files_keeps_live_sessions(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    live = _make_job(title="live", status=JobStatus.DONE, review=ReviewStatus.APPROVED)
    sessions_path = tmp_path / "_sessions.json"
    sessions_path.write_text(
        json.dumps({"1:1": live.job_id, "2:2": "deadbeef0000"}),
        encoding="utf-8",
    )

    report = reset_workspace.ResetReport(dry_run=False)
    reset_workspace.prune_state_files(report)

    remaining = json.loads(sessions_path.read_text(encoding="utf-8"))
    assert remaining == {"1:1": live.job_id}


def test_assert_deletable_refuses_protected_paths():
    base = reset_workspace.BASE_DIR
    with pytest.raises(RuntimeError):
        reset_workspace._assert_deletable(base)
    with pytest.raises(RuntimeError):
        reset_workspace._assert_deletable(base / ".venv" / "Scripts")
    with pytest.raises(RuntimeError):
        reset_workspace._assert_deletable(base / ".cache" / "huggingface")
    with pytest.raises(RuntimeError):
        reset_workspace._assert_deletable(base / "bot.py")
    # Carve-outs that must stay deletable:
    reset_workspace._assert_deletable(base / "karaoke" / "__pycache__")
    reset_workspace._assert_deletable(base / ".cache" / "tmp" / "leftover.wav")


def test_every_job_scoped_callback_prefix_is_protected():
    source = Path(bot.__file__).read_text(encoding="utf-8")
    pattern = re.compile(r'callback_data=f"([a-z_]+):\{([^}]*)\}')
    prefixes = {
        prefix
        for prefix, expr in pattern.findall(source)
        if "job" in expr
    }
    assert prefixes, "expected at least one job-scoped callback prefix in bot.py"
    for prefix in prefixes:
        assert bot.callback_job_id(f"{prefix}:jobid123") == "jobid123", (
            f"callback prefix {prefix}: carries a job id but is not ownership-protected"
        )
