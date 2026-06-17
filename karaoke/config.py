"""Shared configuration for the Hebrew karaoke pipeline."""

from __future__ import annotations

import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(os.getenv("KARAOKE_JOBS_DIR", BASE_DIR / "jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
CACHE_DIR = Path(os.getenv("KARAOKE_CACHE_DIR", BASE_DIR / ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _default_runtime_dir() -> Path:
    if BASE_DIR.drive:
        return Path(f"{BASE_DIR.drive}\\karaoke_runtime")
    return CACHE_DIR / "runtime"


RUNTIME_DIR = Path(os.getenv("KARAOKE_RUNTIME_DIR", _default_runtime_dir()))
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = RUNTIME_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
YTDLP_STAGING_DIR = RUNTIME_DIR / "yt_dlp"
YTDLP_STAGING_DIR.mkdir(parents=True, exist_ok=True)

# yt-dlp YouTube bot-detection bypass.
# Uses node.js + EJS challenge solver by default.
# Optionally set YTDLP_COOKIE_FILE or place cookies.txt next to bot.py.
YTDLP_COOKIE_FILE: str = os.getenv("YTDLP_COOKIE_FILE", "")
YTDLP_COOKIES_FROM_BROWSER: str = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "")

_DEFAULT_COOKIE_FILE = BASE_DIR / "cookies.txt"


def ytdlp_base_opts() -> dict:
    """Return yt-dlp options for bypassing YouTube bot detection."""
    opts: dict = {
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
    }
    # Add cookie auth if available (provides better access than JS solver alone).
    if YTDLP_COOKIE_FILE:
        opts["cookiefile"] = YTDLP_COOKIE_FILE
    elif _DEFAULT_COOKIE_FILE.exists():
        opts["cookiefile"] = str(_DEFAULT_COOKIE_FILE)
    elif YTDLP_COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (YTDLP_COOKIES_FROM_BROWSER,)
    return opts

for env_name, default_path in {
    "TMP": TMP_DIR,
    "TEMP": TMP_DIR,
    "PIP_CACHE_DIR": CACHE_DIR / "pip",
    "XDG_CACHE_HOME": CACHE_DIR,
    "HF_HOME": CACHE_DIR / "huggingface",
    "TRANSFORMERS_CACHE": CACHE_DIR / "transformers",
    "TORCH_HOME": CACHE_DIR / "torch",
    "TORCHINDUCTOR_CACHE_DIR": CACHE_DIR / "torchinductor",
    "MPLCONFIGDIR": CACHE_DIR / "matplotlib",
    "PYANNOTE_CACHE": CACHE_DIR / "pyannote",
    "DEMUCS_CACHE_DIR": CACHE_DIR / "demucs",
}.items():
    resolved = Path(default_path)
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ[env_name] = str(resolved)

PYTHON_EXE = os.getenv(
    "KARAOKE_PYTHON_EXE",
    str(LOCAL_VENV_PYTHON if LOCAL_VENV_PYTHON.exists() else Path(r"C:\Users\shlom\AppData\Local\Programs\Python\Python312\python.exe")),
)

FFMPEG_PATH = Path(
    os.getenv(
        "KARAOKE_FFMPEG_DIR",
        r"C:\Users\shlom\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin",
    )
)
FFMPEG_EXE = str(FFMPEG_PATH / "ffmpeg.exe")
FFPROBE_EXE = str(FFMPEG_PATH / "ffprobe.exe")

def _load_env_value(path: Path, key_names: set[str]) -> str:
    """Load a value from a .env file by matching any of the given key names."""
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


def _load_token_from_env_file(path: Path) -> str:
    return _load_env_value(path, {"TELEGRAM_BOT_TOKEN", "BOT_TOKEN"})


def _recover_token_from_pyc() -> str:
    pattern = re.compile(rb"\d{8,12}:[A-Za-z0-9_-]{20,}")
    pycache_dir = BASE_DIR / "__pycache__"
    if not pycache_dir.exists():
        return ""
    for path in sorted(pycache_dir.glob("bot*.pyc*"), reverse=True):
        match = pattern.search(path.read_bytes())
        if match:
            return match.group(0).decode("utf-8", errors="ignore")
    return ""


def load_telegram_bot_token() -> str:
    for env_name in ("TELEGRAM_BOT_TOKEN", "BOT_TOKEN"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    for candidate in (
        BASE_DIR / "bot_token.txt",
        BASE_DIR / ".env",
        BASE_DIR / ".env.local",
    ):
        if candidate.name.endswith(".txt") and candidate.exists():
            value = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            value = _load_token_from_env_file(candidate)
        if value:
            return value

    recovered = _recover_token_from_pyc()
    if recovered:
        return recovered
    return ""


TELEGRAM_BOT_TOKEN = load_telegram_bot_token()

def _load_gemini_api_key() -> str:
    value = os.getenv("GEMINI_API_KEY", "").strip()
    if value:
        return value
    for candidate in (BASE_DIR / ".env", BASE_DIR / ".env.local"):
        value = _load_env_value(candidate, {"GEMINI_API_KEY"})
        if value:
            return value
    return ""


def _load_google_api_key() -> str:
    value = os.getenv("GOOGLE_API_KEY", "").strip()
    if value:
        return value
    for candidate in (BASE_DIR / ".env", BASE_DIR / ".env.local"):
        value = _load_env_value(candidate, {"GOOGLE_API_KEY"})
        if value:
            return value
    return ""


def _load_xai_api_key() -> str:
    value = os.getenv("XAI_API_KEY", "").strip()
    if value:
        return value
    for candidate in (BASE_DIR / ".env", BASE_DIR / ".env.local"):
        value = _load_env_value(candidate, {"XAI_API_KEY"})
        if value:
            return value
    return ""


def _load_google_search_engine_id() -> str:
    value = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "").strip()
    if value:
        return value
    for candidate in (BASE_DIR / ".env", BASE_DIR / ".env.local"):
        value = _load_env_value(candidate, {"GOOGLE_SEARCH_ENGINE_ID"})
        if value:
            return value
    return ""


def _load_config_value(key_name: str, default: str = "") -> str:
    value = os.getenv(key_name, "").strip()
    if value:
        return value
    for candidate in (BASE_DIR / ".env", BASE_DIR / ".env.local"):
        value = _load_env_value(candidate, {key_name})
        if value:
            return value
    return default


def _load_bool_config(key_name: str, default: bool = False) -> bool:
    value = _load_config_value(key_name, "1" if default else "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _load_int_config(key_name: str, default: int = 0) -> int:
    value = _load_config_value(key_name, str(default)).strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


GEMINI_API_KEY = _load_gemini_api_key()
GEMINI_MODEL = _load_config_value("GEMINI_MODEL", "gemini-2.5-flash")
XAI_API_KEY = _load_xai_api_key()
XAI_MODEL = _load_config_value("XAI_MODEL", "grok-4")
LYRICS_LLM_PROVIDER = _load_config_value(
    "KARAOKE_LYRICS_LLM_PROVIDER",
    _load_config_value("LYRICS_LLM_PROVIDER", "gemini"),
)

# Google Custom Search API
GOOGLE_API_KEY: str = _load_google_api_key()
GOOGLE_SEARCH_ENGINE_ID: str = _load_google_search_engine_id()

# YouTube Data API (uses same GOOGLE_API_KEY)
YOUTUBE_API_ENABLED: bool = bool(GOOGLE_API_KEY)

# Consensus engine
CONSENSUS_MIN_SOURCES: int = 3  # minimum sources for auto-verification

# Verification loop
MAX_REVIEW_ITERATIONS: int = 2  # max round-trips through steps 4-6

WHISPER_HEBREW_MODEL = os.getenv(
    "KARAOKE_HEBREW_WHISPER_MODEL",
    "ivrit-ai/whisper-large-v3-turbo-ct2",
)
WHISPER_DETECT_MODEL = os.getenv("KARAOKE_DETECT_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("KARAOKE_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("KARAOKE_WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = "he"
WHISPER_BEAM_SIZE = int(os.getenv("KARAOKE_WHISPER_BEAM_SIZE", "5"))
LANGUAGE_SAMPLE_SECONDS = int(os.getenv("KARAOKE_LANGUAGE_SAMPLE_SECONDS", "45"))
LANGUAGE_SAMPLE_OFFSET_SECONDS = int(os.getenv("KARAOKE_LANGUAGE_SAMPLE_OFFSET_SECONDS", "20"))
HEBREW_CONFIDENT_THRESHOLD = float(os.getenv("KARAOKE_HEBREW_CONFIDENT_THRESHOLD", "0.55"))
HEBREW_WARNING_THRESHOLD = float(os.getenv("KARAOKE_HEBREW_WARNING_THRESHOLD", "0.30"))
HEBREW_RATIO_WARNING_THRESHOLD = float(os.getenv("KARAOKE_HEBREW_RATIO_WARNING_THRESHOLD", "0.25"))

ALIGNMENT_PROVIDER = os.getenv("KARAOKE_ALIGNMENT_PROVIDER", "auto")
ALIGNMENT_MODEL_NAME = os.getenv(
    "KARAOKE_ALIGNMENT_MODEL",
    "imvladikon/wav2vec2-large-xlsr-53-hebrew",
)
ALIGNMENT_DEVICE = os.getenv("KARAOKE_ALIGNMENT_DEVICE", WHISPER_DEVICE)
ALIGNMENT_COMPUTE_TYPE = os.getenv("KARAOKE_ALIGNMENT_COMPUTE_TYPE", WHISPER_COMPUTE_TYPE)
ALIGNMENT_MIN_WORD_DURATION_MS = int(os.getenv("KARAOKE_ALIGNMENT_MIN_WORD_DURATION_MS", "40"))
ALIGNMENT_BOUNDARY_SEARCH_MS = int(os.getenv("KARAOKE_ALIGNMENT_BOUNDARY_SEARCH_MS", "260"))
DEFAULT_VIDEO_FRAME_RATE = float(os.getenv("KARAOKE_DEFAULT_VIDEO_FRAME_RATE", "25"))

DEFAULT_STYLE_PRESET = os.getenv("KARAOKE_STYLE_PRESET", "blue_outline")
DEFAULT_DELIVERY_CHAT_ID = _load_int_config("KARAOKE_DEFAULT_DELIVERY_CHAT_ID", 0)
DEFAULT_DELIVERY_REPLY_TO_MESSAGE_ID = _load_int_config("KARAOKE_DEFAULT_DELIVERY_REPLY_TO_MESSAGE_ID", 0)
AUTO_DELETE_JOB_AFTER_DELIVERY = _load_bool_config("KARAOKE_AUTO_DELETE_JOB_AFTER_DELIVERY", True)
COMPLETED_JOB_RETENTION_HOURS = max(1, int(_load_config_value("KARAOKE_COMPLETED_JOB_RETENTION_HOURS", "24")))
STALE_JOB_RETENTION_HOURS = max(1, int(_load_config_value("KARAOKE_STALE_JOB_RETENTION_HOURS", "72")))
CODEX_AUTO_REPAIR_ENABLED = _load_bool_config("KARAOKE_ENABLE_CODEX_AUTO_REPAIR", False)
CODEX_AUTO_REPAIR_COMMAND = _load_config_value(
    "KARAOKE_CODEX_AUTO_REPAIR_COMMAND",
    "codex.cmd" if os.name == "nt" else "codex",
)
CODEX_AUTO_REPAIR_SANDBOX = _load_config_value("KARAOKE_CODEX_AUTO_REPAIR_SANDBOX", "workspace-write")
CODEX_AUTO_REPAIR_TIMEOUT_SECONDS = max(30, _load_int_config("KARAOKE_CODEX_AUTO_REPAIR_TIMEOUT_SECONDS", 900))

MAX_TELEGRAM_FILE_SIZE = 45 * 1024 * 1024
HIGH_QUALITY_MP3_BITRATE = os.getenv("KARAOKE_HIGH_QUALITY_MP3_BITRATE", "320k")
REVIEW_PAGE_SIZE = int(os.getenv("KARAOKE_REVIEW_PAGE_SIZE", "18"))
