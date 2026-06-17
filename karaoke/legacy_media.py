"""Legacy media helpers for search/download flows outside the review pipeline."""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path

import yt_dlp

from .audio_extractor import transcode_to_mp3
from .config import FFMPEG_PATH, HIGH_QUALITY_MP3_BITRATE, PYTHON_EXE, YTDLP_STAGING_DIR, ytdlp_base_opts
from .exceptions import AudioExtractionError, DownloadError

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)
MAX_SIZE = 45 * 1024 * 1024
STAGING_DIR = YTDLP_STAGING_DIR / "legacy_media"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_CACHE_REVISION = "hq-v2"


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
        artist = artist.strip()
        song_title = song_title.strip()
        if artist and song_title:
            return artist, song_title
    return "", normalized


def _cleanup(prefix: str):
    for path in DOWNLOAD_DIR.glob(f"{prefix}*"):
        if path.is_file():
            path.unlink(missing_ok=True)


def _run_ydlp(url: str, ydl_opts: dict, prefix: str):
    del prefix
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=True)
        except Exception as exc:
            last_error = exc
    raise DownloadError(str(last_error) if last_error else "")


def _extract_info(url: str) -> dict:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        **ytdlp_base_opts(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def resolve_media_title(url: str) -> str:
    info = _extract_info(url)
    return str(info.get("title") or info.get("id") or "").strip()


def _stage_path(prefix: str) -> Path:
    stage_dir = STAGING_DIR / prefix
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _cleanup_stage(stage_dir: Path):
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


def _cache_marker(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".cache")


def _has_current_cache(path: Path) -> bool:
    marker = _cache_marker(path)
    return path.exists() and path.stat().st_size > 0 and marker.exists() and marker.read_text(encoding="utf-8", errors="ignore").strip() == AUDIO_CACHE_REVISION


def _write_cache_marker(path: Path):
    _cache_marker(path).write_text(AUDIO_CACHE_REVISION, encoding="utf-8")


def _download_best_audio_source(url: str, stage_dir: Path, prefix: str) -> Path:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str((stage_dir / f"{prefix}.%(ext)s").resolve()),
        "ffmpeg_location": str(FFMPEG_PATH),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "windowsfilenames": True,
        **ytdlp_base_opts(),
    }
    _run_ydlp(url, ydl_opts, prefix)
    src = _pick_stage_file(stage_dir, prefix)
    if src is None:
        raise DownloadError("No downloaded audio file was found after the download finished.")
    return src


def search_youtube(query: str, max_results: int = 5) -> list[dict[str, str]]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        **ytdlp_base_opts(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    results = []
    for entry in info.get("entries", []):
        if not entry.get("id"):
            continue
        duration = int(entry.get("duration") or 0)
        minutes, seconds = divmod(duration, 60)
        video_id = entry["id"]
        results.append(
            {
                "id": video_id,
                "title": entry.get("title", "ללא שם"),
                "channel": entry.get("channel") or entry.get("uploader") or "",
                "duration": f"{minutes}:{seconds:02d}",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
            }
        )
    return results


def download_audio(url: str) -> tuple[str, str]:
    info = _extract_info(url)
    title = info.get("title", info.get("id", "audio"))
    dst = DOWNLOAD_DIR / f"{safe_filename(title)}.mp3"
    if _has_current_cache(dst):
        return str(dst.resolve()), title

    stage_dir = _stage_path("audio")
    try:
        src = _download_best_audio_source(url, stage_dir, "audio")
        if dst.exists():
            dst.unlink()
        artist, song_title = split_artist_and_title(str(title))
        transcode_to_mp3(
            str(src.resolve()),
            str(dst.resolve()),
            title=song_title or str(title),
            artist=artist,
        )
        _write_cache_marker(dst)
        return str(dst.resolve()), title
    except AudioExtractionError:
        raise
    finally:
        _cleanup_stage(stage_dir)


def remove_vocals_demucs(input_path: str, output_path: str):
    ffmpeg_exe = str(Path(FFMPEG_PATH) / "ffmpeg.exe")
    abs_input = str(Path(input_path).resolve())
    demucs_out = DOWNLOAD_DIR.resolve() / f"demucs_{uuid.uuid4().hex}"
    demucs_out.mkdir(exist_ok=True)
    try:
        cmd = [
            PYTHON_EXE,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-o",
            str(demucs_out),
            abs_input,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=900)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="ignore")[-600:])
        no_vocals_files = list(demucs_out.glob("**/no_vocals.mp3")) or list(demucs_out.glob("**/no_vocals.wav"))
        if not no_vocals_files:
            raise FileNotFoundError("לא נמצא no_vocals מהפרדת הווקאל.")
        no_vocals = str(no_vocals_files[0])
        if no_vocals.endswith(".wav"):
            result2 = subprocess.run(
                [ffmpeg_exe, "-y", "-i", no_vocals, "-c:a", "libmp3lame", "-b:a", HIGH_QUALITY_MP3_BITRATE, output_path],
                capture_output=True,
            )
            if result2.returncode != 0:
                raise RuntimeError(result2.stderr.decode(errors="ignore")[-400:])
        else:
            shutil.copy2(no_vocals, output_path)
    finally:
        shutil.rmtree(str(demucs_out), ignore_errors=True)


def download_audio_karaoke(url: str) -> tuple[str, str]:
    info = _extract_info(url)
    title = info.get("title", info.get("id", "karaoke"))
    karaoke_path = DOWNLOAD_DIR.resolve() / f"{safe_filename(title)}_karaoke.mp3"
    if _has_current_cache(karaoke_path):
        return str(karaoke_path), title

    stage_dir = _stage_path("karaoke")
    try:
        src = _download_best_audio_source(url, stage_dir, "karaoke_source")
        if karaoke_path.exists():
            karaoke_path.unlink()
        remove_vocals_demucs(str(src.resolve()), str(karaoke_path.resolve()))
        _write_cache_marker(karaoke_path)
    finally:
        _cleanup_stage(stage_dir)
    return str(karaoke_path), title


def compress_video(input_path: str, output_path: str, duration: float):
    duration = max(duration, 10)
    target_size_bits = MAX_SIZE * 8
    audio_bitrate = 128_000
    video_bitrate = max(int(target_size_bits / duration) - audio_bitrate, 150_000)
    ffmpeg_exe = str(Path(FFMPEG_PATH) / "ffmpeg.exe")
    pass_log = str(DOWNLOAD_DIR.resolve() / f"ffpass_{uuid.uuid4().hex}")
    cmd1 = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
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
    cmd2 = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
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
        output_path,
    ]
    result1 = subprocess.run(cmd1, capture_output=True)
    result2 = subprocess.run(cmd2, capture_output=True)
    if result1.returncode != 0 or result2.returncode != 0:
        raise RuntimeError("דחיסת הווידאו נכשלה.")
    for item in DOWNLOAD_DIR.resolve().glob(f"{Path(pass_log).name}*"):
        item.unlink(missing_ok=True)


def download_video(url: str, quality: str) -> tuple[str, str]:
    info = _extract_info(url)
    title = info.get("title", info.get("id", "video"))
    dst = DOWNLOAD_DIR / f"{safe_filename(title)}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst.resolve()), title

    stage_dir = _stage_path("video")
    if quality == "best":
        fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    else:
        fmt = f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality}][ext=mp4]/best"
    ydl_opts = {
        "format": fmt,
        "outtmpl": str((stage_dir / "video.%(ext)s").resolve()),
        "ffmpeg_location": str(FFMPEG_PATH),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "windowsfilenames": True,
        **ytdlp_base_opts(),
    }
    try:
        info = _run_ydlp(url, ydl_opts, "")
        video_id = info["id"]
        duration = float(info.get("duration") or 0)
        src = _pick_stage_file(stage_dir, "video")
        if src is None:
            raise DownloadError("No downloaded video file was found after the download finished.", "לא נמצא קובץ וידאו אחרי ההורדה.")
        if dst.exists():
            dst.unlink()
        if src.stat().st_size > MAX_SIZE:
            compressed = DOWNLOAD_DIR / f"{video_id}_compressed.mp4"
            compressed.unlink(missing_ok=True)
            compress_video(str(src.resolve()), str(compressed.resolve()), duration)
            compressed.rename(dst)
        else:
            shutil.copy2(src, dst)
        return str(dst.resolve()), title
    finally:
        _cleanup_stage(stage_dir)
