"""End-to-end pipeline smoke harness — runs the karaoke pipeline without Telegram.

Produces a JSON metrics report (lyrics verdict, alignment quality, chord
quality) so changes can be A/B-compared against a baseline:

    python tools\\run_pipeline_smoke.py --audio song.mp3 --title "אמן - שיר" --json after.json
    python tools\\run_pipeline_smoke.py --audio song.mp3 --title "אמן - שיר" --baseline before.json

--audio accepts a local file or a YouTube URL. Use --jobs-dir to keep smoke
jobs out of the bot's real jobs folder (default: a smoke_jobs dir next to
this script's project).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Numeric report fields where bigger means better; everything else numeric is
# informational unless listed in _LOWER_IS_BETTER.
_HIGHER_IS_BETTER = {
    "lyrics.confidence",
    "lyrics.option_count",
    "lyrics.source_count",
    "timing.score",
    "timing.aligned_ratio",
    "timing.char_timing_ratio",
    "chords.average_confidence",
    "chords.chord_count",
}
_LOWER_IS_BETTER = {
    "timing.unaligned_word_count",
    "timing.warning_count",
    "chords.low_confidence_ratio",
}
_EPSILON = 1e-6


def _flatten(prefix: str, value, out: dict):
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def compare_reports(baseline: dict, current: dict) -> list[str]:
    flat_base: dict[str, float] = {}
    flat_current: dict[str, float] = {}
    _flatten("", baseline, flat_base)
    _flatten("", current, flat_current)
    regressions = []
    for key in sorted(set(flat_base) & set(flat_current)):
        before, after = flat_base[key], flat_current[key]
        if abs(before - after) <= _EPSILON:
            continue
        direction = "" if key not in _HIGHER_IS_BETTER and key not in _LOWER_IS_BETTER else (
            "regression" if (
                (key in _HIGHER_IS_BETTER and after < before)
                or (key in _LOWER_IS_BETTER and after > before)
            ) else "improvement"
        )
        line = f"{key}: {before:.4g} -> {after:.4g}" + (f"  [{direction}]" if direction else "")
        print(line)
        if direction == "regression":
            regressions.append(line)
    return regressions


def _summary_dict(summary) -> dict:
    if is_dataclass(summary):
        return asdict(summary)
    return {key: value for key, value in vars(summary).items() if not key.startswith("_")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="local audio/video file or YouTube URL")
    parser.add_argument("--title", required=True, help='song title, e.g. "אמן - שיר"')
    parser.add_argument("--stage", choices=("lyrics", "align", "chords", "all"), default="all")
    parser.add_argument("--json", dest="json_path", help="write the metrics report to this path")
    parser.add_argument("--baseline", help="previous report to diff against; regressions exit 1")
    parser.add_argument("--jobs-dir", default=str(PROJECT_ROOT / "smoke_jobs"))
    args = parser.parse_args(argv)

    # Must happen before importing karaoke (config reads env at import time).
    os.environ["KARAOKE_JOBS_DIR"] = args.jobs_dir
    sys.path.insert(0, str(PROJECT_ROOT))

    from karaoke import job_manager
    from karaoke.aligner import analyze_alignment_quality, validate_timing_quality
    from karaoke.harmony import summarize_song_analysis_quality
    from karaoke.pipeline import KaraokePipeline

    audio_arg = args.audio.strip()
    local_path = Path(audio_arg)
    is_local = local_path.exists()
    if is_local:
        is_video = local_path.suffix.lower() in {".mp4", ".mkv", ".avi", ".webm", ".mov"}
        job = job_manager.create_job(
            title=args.title,
            input_type="video_file" if is_video else "audio_file",
            has_video=is_video,
        )
        input_path = str(local_path.resolve())
    else:
        job = job_manager.create_job(title=args.title, source_url=audio_arg, input_type="youtube")
        input_path = None

    report: dict = {
        "job_id": job.job_id,
        "title": args.title,
        "stage": args.stage,
        "durations_sec": {},
    }
    pipeline = KaraokePipeline(job)

    started = time.perf_counter()
    draft = pipeline.run_until_review(input_path)
    report["durations_sec"]["until_review"] = round(time.perf_counter() - started, 1)

    verification = job.manifest.lyrics_verification or {}
    options = job_manager.get_selectable_lyrics_options(job)
    report["lyrics"] = {
        "verdict": verification.get("verdict", ""),
        "confidence": float(verification.get("confidence", 0.0) or 0.0),
        "applied": bool(verification.get("applied", False)),
        "correction_count": int(verification.get("correction_count", 0) or 0),
        "option_count": len(options),
        "option_ids": [option.get("option_id", "") for option in options],
        "source_count": int(verification.get("source_count", 0) or 0),
        "local_warnings": list(verification.get("local_warnings", []) or []),
    }

    if args.stage != "lyrics":
        # Approve: keep the auto-applied text if any, else apply the top option,
        # else fall back to the raw draft.
        if not job.review_timings_path.exists():
            if options:
                job_manager.apply_lyrics_option(job, options[0].get("option_id", ""))
                report["lyrics"]["applied_option"] = options[0].get("option_id", "")
            else:
                job_manager.save_review_transcript(job, draft.segments)
                report["lyrics"]["applied_option"] = "draft"
        approved = job_manager.load_review_segments(job)
        draft_segments = job_manager.load_draft_segments(job)
        approved = job_manager.rebuild_segments_from_authoritative_text(draft_segments, approved)

        started = time.perf_counter()
        aligned = pipeline.aligner.align(
            str(job.vocals_16k_path),
            approved,
            draft_segments,
            video_frame_rate=None,
        )
        report["durations_sec"]["alignment"] = round(time.perf_counter() - started, 1)
        quality = analyze_alignment_quality(aligned.segments)
        warnings = validate_timing_quality(aligned.segments)
        report["timing"] = {
            "score": float(quality.get("score", 0.0)),
            "aligned_ratio": float(quality.get("aligned_ratio", 0.0)),
            "char_timing_ratio": float(quality.get("char_timing_ratio", 0.0)),
            "critical": bool(quality.get("critical", False)),
            "unaligned_word_count": int(aligned.unaligned_word_count),
            "warning_count": len(warnings),
            "warnings": warnings,
            "provider": getattr(pipeline.aligner, "last_provider_used", "")
            or getattr(pipeline.aligner, "name", ""),
            "provider_warning": getattr(pipeline.aligner, "last_warning_message", ""),
        }
        job_manager.save_final_transcript(job, aligned)

        if args.stage in ("chords", "all"):
            started = time.perf_counter()
            analysis = pipeline.step_analyze_music(aligned.segments)
            report["durations_sec"]["chords"] = round(time.perf_counter() - started, 1)
            chord_quality = _summary_dict(summarize_song_analysis_quality(analysis))
            report["chords"] = {
                **{key: value for key, value in chord_quality.items()},
                "chord_count": len(analysis.chord_events),
                "key": analysis.original_key,
                "bpm": float(analysis.bpm or 0.0),
                "time_signature": int(getattr(analysis, "time_signature", 0) or 0),
                "source": analysis.chord_source_name or "librosa",
            }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        print("\n--- baseline comparison ---")
        regressions = compare_reports(baseline, report)
        if regressions:
            print(f"\n{len(regressions)} regression(s) detected.")
            return 1
        print("no regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
