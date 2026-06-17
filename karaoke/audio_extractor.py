"""Audio extraction and conversion helpers built on FFmpeg."""

from __future__ import annotations

import logging
import subprocess
from fractions import Fraction
from pathlib import Path

from .config import FFPROBE_EXE, FFMPEG_EXE, HIGH_QUALITY_MP3_BITRATE
from .exceptions import AudioExtractionError

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float:
    cmd = [
        FFPROBE_EXE,
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise AudioExtractionError(result.stderr[:300], "לא ניתן היה לקרוא את אורך המדיה.")
    return float(result.stdout.strip())


def get_video_frame_rate(video_path: str) -> float:
    cmd = [
        FFPROBE_EXE,
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise AudioExtractionError(result.stderr[:300], "לא ניתן לקרוא את קצב הפריימים של הווידאו.")

    for line in result.stdout.splitlines():
        value = line.strip()
        if not value or value in {"0/0", "N/A"}:
            continue
        try:
            return float(Fraction(value))
        except (ValueError, ZeroDivisionError):
            continue

    raise AudioExtractionError("ffprobe did not return a valid frame rate.", "לא התקבל קצב פריימים תקין מהווידאו.")


def extract_audio_from_video(video_path: str, output_path: str):
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        HIGH_QUALITY_MP3_BITRATE,
        str(output_path),
    ]
    logger.info("Extracting audio from %s", video_path)
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        error_text = result.stderr.decode(errors="ignore")[-400:]
        raise AudioExtractionError(error_text)
    if not Path(output_path).exists():
        raise AudioExtractionError(f"Missing extracted audio at {output_path}")


def transcode_to_mp3(
    input_path: str,
    output_path: str,
    *,
    title: str = "",
    artist: str = "",
):
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        HIGH_QUALITY_MP3_BITRATE,
    ]
    if title:
        cmd.extend(["-metadata", f"title={title}"])
    if artist:
        cmd.extend(["-metadata", f"artist={artist}"])
    cmd.extend([
        str(output_path),
    ])
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        error_text = result.stderr.decode(errors="ignore")[-400:]
        raise AudioExtractionError(error_text, "המרת קובץ האודיו ל-MP3 נכשלה.")
    if not Path(output_path).exists():
        raise AudioExtractionError(f"Missing mp3 output at {output_path}")


def convert_to_wav(input_path: str, output_path: str):
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    logger.info("Converting %s to wav", input_path)
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        error_text = result.stderr.decode(errors="ignore")[-400:]
        raise AudioExtractionError(error_text, "המרת האודיו ל-WAV נכשלה.")
    if not Path(output_path).exists():
        raise AudioExtractionError(f"Missing wav output at {output_path}")


def create_audio_sample(input_path: str, output_path: str, duration: int, offset: int = 0):
    media_duration = get_audio_duration(input_path)
    clip_duration = min(duration, max(int(media_duration), 1))
    clip_offset = min(offset, max(int(media_duration - clip_duration), 0))

    cmd = [
        FFMPEG_EXE,
        "-y",
        "-ss",
        str(clip_offset),
        "-t",
        str(clip_duration),
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=180)
    if result.returncode != 0:
        error_text = result.stderr.decode(errors="ignore")[-400:]
        raise AudioExtractionError(error_text, "חיתוך דגימת האודיו לזיהוי שפה נכשל.")
    if not Path(output_path).exists():
        raise AudioExtractionError(f"Missing language sample at {output_path}")
