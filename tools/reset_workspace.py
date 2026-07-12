"""Daily workspace reset for the Hebrew karaoke bot.

Deletes stale job folders, root debug junk, transient caches and staging
directories, returning the project to its minimal operational state.
Model caches (.cache), the virtualenv, secrets, source code and git data
are never touched. Importing karaoke.config recreates every directory the
bot needs, so a run always ends in a valid "default" state.

Usage:
    python tools\\reset_workspace.py [--dry-run] [--force] [--recent-hours N]

Safe to run while the bot is up: in-flight jobs are skipped and locked
files (e.g. the open bot.log) are left alone.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from karaoke import job_manager  # noqa: E402
from karaoke.models import JobStatus  # noqa: E402
from karaoke.config import (  # noqa: E402
    BASE_DIR,
    CACHE_DIR,
    JOBS_DIR,
    TMP_DIR,
    YTDLP_STAGING_DIR,
)

PENDING_DELIVERY_ACTIVE_STATUSES = {"pending_approval", "awaiting_feedback", "repairing"}

ROOT_JUNK_PATTERNS = (
    "tmp_frame_*.png",
    "tmp_fixed_frame_*.png",
    "tmp_preview_color_*.png",
    ".debug_bot_*.txt",
    "firebase-debug*.log",
)

# Directories under BASE_DIR that must never be deleted or descended into.
PROTECTED_DIR_NAMES = {".cache", ".venv", ".git", ".claude", "docs", "tests", "karaoke", "tools"}
# File names / suffixes under BASE_DIR that must never be deleted.
PROTECTED_FILE_SUFFIXES = (".py", ".bat", ".md")
PROTECTED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".gitignore",
    "bot_token.txt",
    "cookies.txt",
    "requirements.txt",
}

CLEANUP_LOG_NAMES = {"cleanup.log", "cleanup_task.log"}


class ResetReport:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.deleted_dirs = 0
        self.deleted_files = 0
        self.bytes_freed = 0
        self.skipped_jobs: list[str] = []
        self.errors: list[str] = []

    def record(self, path: Path, size: int, is_dir: bool):
        if is_dir:
            self.deleted_dirs += 1
        else:
            self.deleted_files += 1
        self.bytes_freed += size

    def summary_line(self) -> str:
        mode = "DRY-RUN" if self.dry_run else "reset"
        skipped = ",".join(self.skipped_jobs) or "-"
        return (
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | {mode} | "
            f"dirs={self.deleted_dirs} files={self.deleted_files} "
            f"freed={self.bytes_freed / (1024 * 1024):.1f}MB | "
            f"skipped_in_flight={skipped} | errors={len(self.errors)}"
        )


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _assert_deletable(path: Path):
    """Defense in depth: refuse to delete anything on the keep-list."""
    resolved = path.resolve()
    base = BASE_DIR.resolve()
    if resolved == base:
        raise RuntimeError(f"refusing to delete project root: {path}")
    try:
        relative = resolved.relative_to(base)
    except ValueError:
        # Outside the project (runtime staging dirs) — allowed by explicit callers.
        return
    top = relative.parts[0]
    if top in PROTECTED_DIR_NAMES:
        # Only two carve-outs inside protected dirs: __pycache__ subtrees
        # and the transient .cache/tmp staging area.
        allowed = relative.parts[1:2] == ("__pycache__",) or (
            relative.parts[:2] == (".cache", "tmp") and len(relative.parts) >= 3
        )
        if not allowed:
            raise RuntimeError(f"refusing to delete protected path: {path}")
    if len(relative.parts) == 1 and resolved.is_file():
        if resolved.name in PROTECTED_FILE_NAMES or resolved.suffix in PROTECTED_FILE_SUFFIXES:
            raise RuntimeError(f"refusing to delete protected file: {path}")


def _delete_dir(path: Path, report: ResetReport):
    if not path.is_dir():
        return
    _assert_deletable(path)
    size = _dir_size(path)
    if report.dry_run:
        print(f"[dry-run] would remove dir: {path} ({size / (1024 * 1024):.1f}MB)")
        report.record(path, size, is_dir=True)
        return
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        report.errors.append(f"dir not fully removed (locked?): {path}")
    report.record(path, size, is_dir=True)


def _delete_file(path: Path, report: ResetReport):
    if not path.is_file():
        return
    _assert_deletable(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if report.dry_run:
        print(f"[dry-run] would remove file: {path}")
        report.record(path, size, is_dir=False)
        return
    try:
        path.unlink()
        report.record(path, size, is_dir=False)
    except PermissionError:
        # Locked (e.g. bot.log while the bot is running) — leave it.
        pass
    except OSError as exc:
        report.errors.append(f"{path}: {exc}")


def job_in_flight_reason(job, *, now: datetime, recent_hours: float) -> str | None:
    if job.status in job_manager._PROCESSING_STATUSES:
        return "processing"
    # An open lyrics review survives the nightly reset; the bot's own
    # maintenance reclaims it as "abandoned_review" after 72h.
    if job.status == JobStatus.AWAITING_REVIEW and job_manager._job_last_activity(job) > now - timedelta(hours=72):
        return "awaiting-review"
    pending = job.pending_delivery
    if isinstance(pending, dict) and str(pending.get("status", "")) in PENDING_DELIVERY_ACTIVE_STATUSES:
        return "pending-review"
    if job_manager._job_last_activity(job) > now - timedelta(hours=recent_hours):
        return "recent"
    return None


def purge_jobs(report: ResetReport, *, force: bool, recent_hours: float):
    now = datetime.now(timezone.utc)
    for job_dir in sorted(JOBS_DIR.iterdir() if JOBS_DIR.exists() else []):
        if not job_dir.is_dir():
            continue
        try:
            job = job_manager.load_job(job_dir.name)
        except Exception:
            # Unparseable manifest: delete only if the folder itself is old.
            mtime = datetime.fromtimestamp(job_dir.stat().st_mtime, tz=timezone.utc)
            if mtime > now - timedelta(hours=recent_hours):
                report.skipped_jobs.append(f"{job_dir.name}(unreadable-recent)")
                continue
            _delete_dir(job_dir, report)
            continue
        reason = None if force else job_in_flight_reason(job, now=now, recent_hours=recent_hours)
        if reason:
            report.skipped_jobs.append(f"{job.job_id}({reason})")
            continue
        _delete_dir(job_dir, report)


def prune_state_files(report: ResetReport):
    sessions_path = JOBS_DIR / "_sessions.json"
    if sessions_path.exists():
        try:
            sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sessions = {}
        live = {
            key: job_id
            for key, job_id in sessions.items()
            if isinstance(job_id, str) and (JOBS_DIR / job_id / "job.json").exists()
        }
        if live != sessions:
            if report.dry_run:
                print(f"[dry-run] would prune _sessions.json: {len(sessions)} -> {len(live)} entries")
            else:
                sessions_path.write_text(
                    json.dumps(live, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    if report.dry_run:
        return
    try:
        removed = job_manager.cleanup_stale_group_requests()
        if removed:
            print(f"pruned {removed} stale group request(s)")
    except Exception as exc:  # never fail the whole reset over this
        report.errors.append(f"group requests: {exc}")


def purge_root_junk(report: ResetReport):
    for pattern in ROOT_JUNK_PATTERNS:
        for path in BASE_DIR.glob(pattern):
            _delete_file(path, report)
    # "nul" is a reserved device name on Windows; needs the \\?\ long-path form.
    nul_path = BASE_DIR / "nul"
    win_nul = Path("\\\\?\\" + str(nul_path))
    if win_nul.exists():
        if report.dry_run:
            print(f"[dry-run] would remove file: {nul_path}")
            report.record(nul_path, 0, is_dir=False)
        else:
            try:
                os.remove(win_nul)
                report.record(nul_path, 0, is_dir=False)
            except OSError as exc:
                report.errors.append(f"nul: {exc}")


def purge_downloads(report: ResetReport):
    downloads = BASE_DIR / "downloads"
    if not downloads.is_dir():
        return
    for item in downloads.iterdir():
        if item.is_dir():
            _delete_dir(item, report)
        else:
            _delete_file(item, report)


def purge_logs(report: ResetReport):
    logs = BASE_DIR / "logs"
    if not logs.is_dir():
        return
    for item in logs.iterdir():
        if item.name in CLEANUP_LOG_NAMES:
            continue
        if item.is_file():
            _delete_file(item, report)


def purge_caches(report: ResetReport):
    for cache_dir in (
        BASE_DIR / "__pycache__",
        BASE_DIR / "karaoke" / "__pycache__",
        BASE_DIR / "tests" / "__pycache__",
        BASE_DIR / "tools" / "__pycache__",
        BASE_DIR / ".pytest_cache",
    ):
        _delete_dir(cache_dir, report)


def purge_runtime(report: ResetReport):
    for staging in (TMP_DIR, YTDLP_STAGING_DIR, CACHE_DIR / "tmp"):
        if not staging.is_dir():
            continue
        for item in staging.iterdir():
            if item.is_dir():
                _delete_dir(item, report)
            else:
                _delete_file(item, report)


def write_summary(report: ResetReport):
    line = report.summary_line()
    print(line)
    for error in report.errors:
        print(f"  error: {error}")
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / "cleanup.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        for error in report.errors:
            handle.write(f"  error: {error}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, delete nothing")
    parser.add_argument("--force", action="store_true", help="delete jobs even if in-flight")
    parser.add_argument(
        "--recent-hours",
        type=float,
        default=6.0,
        help="jobs with activity newer than this are kept (default: 6)",
    )
    args = parser.parse_args(argv)

    report = ResetReport(dry_run=args.dry_run)
    purge_jobs(report, force=args.force, recent_hours=args.recent_hours)
    prune_state_files(report)
    purge_root_junk(report)
    purge_downloads(report)
    purge_logs(report)
    purge_caches(report)
    purge_runtime(report)
    write_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
