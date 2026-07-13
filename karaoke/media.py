"""Lean YouTube media helpers: search, audio (mp3) and video download.

Self-contained — no ML, no vocal separation. Uses yt-dlp + ffmpeg only.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import yt_dlp

from .config import (
    DOWNLOAD_DIR,
    FFMPEG_EXE,
    HIGH_QUALITY_MP3_BITRATE,
    YTDLP_STAGING_DIR,
    ytdlp_base_opts,
)
from .exceptions import AudioExtractionError, DownloadError

STAGING_DIR = YTDLP_STAGING_DIR / "media"
STAGING_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
def safe_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", title).strip() or "output"


def split_artist_and_title(title: str) -> tuple[str, str]:
    normalized = str(title or "").strip()
    if not normalized:
        return "", ""
    for separator in (" - ", " – ", " — ", " | "):
        if separator not in normalized:
            continue
        artist, song_title = normalized.split(separator, 1)
        artist, song_title = artist.strip(), song_title.strip()
        if artist and song_title:
            return artist, song_title
    return "", normalized


# --------------------------------------------------------------------------- #
# yt-dlp plumbing
# --------------------------------------------------------------------------- #
def _run_ydlp(url: str, ydl_opts: dict) -> dict:
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=True)
        except Exception as exc:  # noqa: BLE001 - retried, then surfaced
            last_error = exc
    raise DownloadError(str(last_error) if last_error else "")


def _extract_info(url: str) -> dict:
    ydl_opts = {"quiet": True, "no_warnings": True, "noplaylist": True, **ytdlp_base_opts()}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def _stage_path(prefix: str) -> Path:
    stage_dir = STAGING_DIR / prefix
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _cleanup_stage(stage_dir: Path) -> None:
    shutil.rmtree(stage_dir, ignore_errors=True)


def _pick_stage_file(stage_dir: Path, prefix: str) -> Path | None:
    candidates = [
        path
        for path in stage_dir.glob(f"{prefix}.*")
        if path.is_file() and path.suffix.lower() not in {".part", ".tmp", ".ytdl"}
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.name)
    return candidates[0]


# --------------------------------------------------------------------------- #
# Transcode
# --------------------------------------------------------------------------- #
def transcode_to_mp3(input_path: str, output_path: str, *, title: str = "", artist: str = "") -> None:
    cmd = [FFMPEG_EXE, "-y", "-i", str(input_path), "-vn", "-c:a", "libmp3lame",
           "-b:a", HIGH_QUALITY_MP3_BITRATE]
    if title:
        cmd += ["-metadata", f"title={title}"]
    if artist:
        cmd += ["-metadata", f"artist={artist}"]
    cmd.append(str(output_path))
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise AudioExtractionError(result.stderr.decode(errors="ignore")[-400:],
                                   "המרת קובץ האודיו ל-MP3 נכשלה.")
    if not Path(output_path).exists():
        raise AudioExtractionError(f"Missing mp3 output at {output_path}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def search_youtube(query: str, max_results: int = 5) -> list[dict[str, str]]:
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True, **ytdlp_base_opts()}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    results: list[dict[str, str]] = []
    for entry in info.get("entries", []):
        if not entry.get("id"):
            continue
        minutes, seconds = divmod(int(entry.get("duration") or 0), 60)
        video_id = entry["id"]
        results.append({
            "id": video_id,
            "title": entry.get("title", "ללא שם"),
            "channel": entry.get("channel") or entry.get("uploader") or "",
            "duration": f"{minutes}:{seconds:02d}",
            "url": f"https://www.youtube.com/watch?v={video_id}",
        })
    return results


def download_audio(url: str) -> tuple[str, str]:
    """Download best audio and transcode to mp3. Returns (mp3_path, title)."""
    info = _extract_info(url)
    title = str(info.get("title") or info.get("id") or "audio")
    dst = DOWNLOAD_DIR / f"{safe_filename(title)}.mp3"
    stage_dir = _stage_path("audio")
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str((stage_dir / "audio.%(ext)s").resolve()),
            "ffmpeg_location": str(Path(FFMPEG_EXE).parent),
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "retries": 5, "fragment_retries": 5, "windowsfilenames": True,
            **ytdlp_base_opts(),
        }
        _run_ydlp(url, ydl_opts)
        src = _pick_stage_file(stage_dir, "audio")
        if src is None:
            raise DownloadError("No downloaded audio file was found.")
        if dst.exists():
            dst.unlink()
        artist, song_title = split_artist_and_title(title)
        transcode_to_mp3(str(src.resolve()), str(dst.resolve()),
                         title=song_title or title, artist=artist)
        return str(dst.resolve()), title
    finally:
        _cleanup_stage(stage_dir)


def _video_format(quality: str) -> str:
    if quality == "best":
        return "bestvideo+bestaudio/best"
    return f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]"


def download_video(url: str, quality: str) -> tuple[str, str]:
    """Download video at the requested quality, merged to mp4. Returns (mp4_path, title)."""
    info = _extract_info(url)
    title = str(info.get("title") or info.get("id") or "video")
    dst = DOWNLOAD_DIR / f"{safe_filename(title)}.mp4"
    stage_dir = _stage_path("video")
    try:
        ydl_opts = {
            "format": _video_format(quality),
            "merge_output_format": "mp4",
            "outtmpl": str((stage_dir / "video.%(ext)s").resolve()),
            "ffmpeg_location": str(Path(FFMPEG_EXE).parent),
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "retries": 5, "fragment_retries": 5, "windowsfilenames": True,
            **ytdlp_base_opts(),
        }
        _run_ydlp(url, ydl_opts)
        src = _pick_stage_file(stage_dir, "video")
        if src is None:
            raise DownloadError("No downloaded video file was found.")
        if dst.exists():
            dst.unlink()
        shutil.move(str(src.resolve()), str(dst.resolve()))
        return str(dst.resolve()), title
    finally:
        _cleanup_stage(stage_dir)
