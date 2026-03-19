"""FFmpeg-powered video rendering for karaoke output."""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

from .config import FFMPEG_EXE, MAX_TELEGRAM_FILE_SIZE
from .exceptions import VideoRenderError

logger = logging.getLogger(__name__)


def burn_subtitles(video_path: str, ass_path: str, output_path: str, audio_path: str | None = None):
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
    if audio_path:
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            f"ass='{ass_escaped}'",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    else:
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"ass='{ass_escaped}'",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    result = subprocess.run(cmd, capture_output=True, timeout=900)
    if result.returncode != 0:
        raise VideoRenderError(result.stderr.decode(errors="ignore")[-500:], "צריבת הכתוביות על הווידאו נכשלה.")
    if not Path(output_path).exists():
        raise VideoRenderError(f"Missing rendered video {output_path}")


def create_static_video(image_path: str, audio_path: str, output_path: str):
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=900)
    if result.returncode != 0:
        raise VideoRenderError(result.stderr.decode(errors="ignore")[-500:], "יצירת וידאו סטטי מהתמונה נכשלה.")
    if not Path(output_path).exists():
        raise VideoRenderError(f"Missing static video {output_path}")


def compress_video_if_needed(video_path: str, duration: float) -> str:
    size = Path(video_path).stat().st_size
    if size <= MAX_TELEGRAM_FILE_SIZE:
        return video_path

    compressed_path = str(Path(video_path).with_suffix("")) + "_compressed.mp4"
    duration = max(duration, 10)
    target_size_bits = MAX_TELEGRAM_FILE_SIZE * 8
    audio_bitrate = 128_000
    video_bitrate = max(int(target_size_bits / duration) - audio_bitrate, 150_000)
    pass_log = str(Path(video_path).parent / f"ffpass_{uuid.uuid4().hex[:8]}")

    cmd_pass1 = [
        FFMPEG_EXE,
        "-y",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-b:v",
        str(video_bitrate),
        "-passlogfile",
        pass_log,
        "-pass",
        "1",
        "-an",
        "-f",
        "null",
        "NUL",
    ]
    cmd_pass2 = [
        FFMPEG_EXE,
        "-y",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-b:v",
        str(video_bitrate),
        "-passlogfile",
        pass_log,
        "-pass",
        "2",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        compressed_path,
    ]

    result1 = subprocess.run(cmd_pass1, capture_output=True, timeout=900)
    if result1.returncode != 0:
        raise VideoRenderError(result1.stderr.decode(errors="ignore")[-500:], "דחיסת הווידאו נכשלה בשלב הראשון.")
    result2 = subprocess.run(cmd_pass2, capture_output=True, timeout=900)
    if result2.returncode != 0:
        raise VideoRenderError(result2.stderr.decode(errors="ignore")[-500:], "דחיסת הווידאו נכשלה בשלב השני.")

    for path in Path(video_path).parent.glob(f"{Path(pass_log).name}*"):
        path.unlink(missing_ok=True)

    Path(video_path).unlink(missing_ok=True)
    Path(compressed_path).rename(video_path)
    logger.info("Compressed video to %.1fMB", Path(video_path).stat().st_size / 1024 / 1024)
    return video_path
