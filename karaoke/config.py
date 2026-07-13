"""Shared configuration for the lean karaoke bot (chords + YouTube media)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.getenv("KARAOKE_CACHE_DIR", BASE_DIR / ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Fetched chord/lyric pages are cached here.
HTTP_CACHE_DIR = Path(os.getenv("KARAOKE_HTTP_CACHE_DIR", str(CACHE_DIR / "http")))
HTTP_CACHE_TTL_SECONDS = int(os.getenv("KARAOKE_HTTP_CACHE_TTL_SECONDS", str(30 * 24 * 3600)))


def _default_runtime_dir() -> Path:
    if os.name == "nt" and BASE_DIR.drive:
        return Path(f"{BASE_DIR.drive}\\karaoke_runtime")
    return CACHE_DIR / "runtime"


RUNTIME_DIR = Path(os.getenv("KARAOKE_RUNTIME_DIR", _default_runtime_dir()))
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = RUNTIME_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
YTDLP_STAGING_DIR = RUNTIME_DIR / "yt_dlp"
YTDLP_STAGING_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = Path(os.getenv("KARAOKE_DOWNLOAD_DIR", "downloads"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# yt-dlp YouTube bot-detection bypass (node.js EJS solver + optional cookies).
# Set YTDLP_COOKIE_FILE or place cookies.txt next to bot.py.
# --------------------------------------------------------------------------- #
YTDLP_COOKIE_FILE: str = os.getenv("YTDLP_COOKIE_FILE", "")
YTDLP_COOKIES_FROM_BROWSER: str = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "")
_DEFAULT_COOKIE_FILE = BASE_DIR / "cookies.txt"


def _materialized_cookie_file() -> str:
    """On a public host, cookies are passed via the YTDLP_COOKIES_CONTENT secret
    (env var) instead of a committed file; write it to a runtime file once."""
    content = os.getenv("YTDLP_COOKIES_CONTENT", "")
    if not content.strip():
        return ""
    target = RUNTIME_DIR / "cookies.txt"
    try:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
        return str(target)
    except OSError:
        return ""


def ytdlp_base_opts() -> dict:
    """Return yt-dlp options for bypassing YouTube bot detection."""
    opts: dict = {"js_runtimes": {"node": {}}, "remote_components": ["ejs:github"]}
    materialized = _materialized_cookie_file()
    if YTDLP_COOKIE_FILE:
        opts["cookiefile"] = YTDLP_COOKIE_FILE
    elif materialized:
        opts["cookiefile"] = materialized
    elif _DEFAULT_COOKIE_FILE.exists():
        opts["cookiefile"] = str(_DEFAULT_COOKIE_FILE)
    elif YTDLP_COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (YTDLP_COOKIES_FROM_BROWSER,)
    return opts


# --------------------------------------------------------------------------- #
# ffmpeg (OS-aware: no ".exe" on Linux/HF Spaces).
# --------------------------------------------------------------------------- #
_EXE = ".exe" if os.name == "nt" else ""


def _discover_ffmpeg_dir() -> str:
    env_dir = os.getenv("KARAOKE_FFMPEG_DIR", "").strip()
    if env_dir:
        return env_dir
    which = shutil.which("ffmpeg")
    if which:
        return str(Path(which).parent)
    for candidate in (
        Path(r"C:\ffmpeg\bin"),
    ):
        if (candidate / f"ffmpeg{_EXE}").exists():
            return str(candidate)
    return r"C:\ffmpeg\bin" if os.name == "nt" else "/usr/bin"


FFMPEG_PATH = Path(_discover_ffmpeg_dir())
FFMPEG_EXE = str(FFMPEG_PATH / f"ffmpeg{_EXE}")
FFPROBE_EXE = str(FFMPEG_PATH / f"ffprobe{_EXE}")
if not Path(FFMPEG_EXE).exists() and not shutil.which("ffmpeg"):
    logger.warning("ffmpeg not found; media processing will fail until it is installed.")

HIGH_QUALITY_MP3_BITRATE = os.getenv("KARAOKE_HIGH_QUALITY_MP3_BITRATE", "320k")


# --------------------------------------------------------------------------- #
# Telegram bot token loading (env → bot_token.txt → .env).
# --------------------------------------------------------------------------- #
def _load_env_value(path: Path, key_names: set[str]) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() in key_names:
            return value.strip().strip('"').strip("'")
    return ""


def load_telegram_bot_token() -> str:
    for env_name in ("TELEGRAM_BOT_TOKEN", "BOT_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    for candidate in (BASE_DIR / "bot_token.txt", BASE_DIR / ".env", BASE_DIR / ".env.local"):
        if candidate.name.endswith(".txt") and candidate.exists():
            value = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            value = _load_env_value(candidate, {"TELEGRAM_BOT_TOKEN", "BOT_TOKEN"})
        if value:
            return value
    return ""


TELEGRAM_BOT_TOKEN = load_telegram_bot_token()


# --------------------------------------------------------------------------- #
# Deployment / integration config.
# --------------------------------------------------------------------------- #
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")  # e.g. https://user-space.hf.space
FILE_SERVER_PORT = int(os.getenv("PORT", "7860"))
TELEGRAM_FILE_LIMIT_BYTES = int(os.getenv("TELEGRAM_FILE_LIMIT_BYTES", str(50 * 1024 * 1024)))
LINK_TTL_SECONDS = int(os.getenv("LINK_TTL_SECONDS", str(2 * 3600)))
