import atexit
import asyncio
import contextlib
import contextvars
import ctypes
import html
import io
import json
import logging
import os
import re
import shutil
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyParameters, Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from karaoke import job_manager
from karaoke.auto_repair import apply_feedback_to_review, feedback_mentions_timing_problem, run_codex_auto_repair
from karaoke.chord_sources import lookup_external_chord_sheet_by_title
from karaoke.config import (
    BASE_DIR,
    DEFAULT_DELIVERY_CHAT_ID,
    DEFAULT_DELIVERY_REPLY_TO_MESSAGE_ID,
    MAX_REVIEW_ITERATIONS,
    TELEGRAM_BOT_TOKEN,
)
from karaoke.error_formatter import format_pipeline_error, format_unexpected_error
from karaoke.exceptions import DeliveryError, PipelineError
from karaoke.harmony import (
    prepare_song_analysis_for_display,
    render_chord_sheet_text,
    resolve_song_analysis_key_labels,
    summarize_song_analysis_quality,
)
from karaoke.legacy_media import (
    download_audio,
    download_audio_karaoke,
    download_video,
    resolve_media_title,
    safe_filename,
    search_youtube,
)
from karaoke.models import Job, JobStatus, ReviewStatus, STATUS_MESSAGES, VideoRequest, ConsensusResult, DisputedLine, VerificationVerdict
from karaoke.pipeline import KaraokePipeline

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "bot.log"

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
file_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger(__name__)

_SINGLE_INSTANCE_HANDLE = None
_WINDOWS_MUTEX_ALREADY_EXISTS = 183
GROUP_CHAT_TYPES = {"group", "supergroup"}
CHORD_LINE_TOKEN_PATTERN = re.compile(r"^[A-G](?:#|b)?(?:[A-Za-z0-9+#/()_-]*)?$")

COMMANDS_TEXT = "\n\nפקודות: /start | /clear | /stop"
YOUTUBE_URL_PATTERN = re.compile(r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+")
LINE_CORRECTION_PATTERN = re.compile(r"^\d+\s*:\s*.+")


def is_youtube_url(text: str) -> bool:
    return bool(YOUTUBE_URL_PATTERN.search(text))


def looks_like_new_request(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if is_youtube_url(normalized):
        return True
    if "\n" in normalized:
        return False
    if LINE_CORRECTION_PATTERN.match(normalized):
        return False
    words = normalized.split()
    return len(words) <= 10 and len(normalized) <= 100


def run_storage_maintenance():
    expired_group_requests = job_manager.cleanup_stale_group_requests()
    if expired_group_requests:
        logger.info("Storage cleanup removed %d expired group handoff requests.", expired_group_requests)
    removed = job_manager.cleanup_stale_jobs()
    if not removed:
        return
    summary = ", ".join(f"{item['job_id']} ({item['reason']})" for item in removed[:8])
    if len(removed) > 8:
        summary += ", ..."
    logger.info("Storage cleanup removed %d old jobs: %s", len(removed), summary)


# One demucs/whisper/ffmpeg pipeline at a time: concurrent_updates(True) would
# otherwise let several CPU-heavy jobs thrash the machine until they all time out.
HEAVY_JOB_SEMAPHORE = asyncio.Semaphore(max(1, int(os.getenv("KARAOKE_MAX_CONCURRENT_HEAVY_JOBS", "1"))))
_HEAVY_SLOT_HELD: contextvars.ContextVar[bool] = contextvars.ContextVar("heavy_slot_held", default=False)


@contextlib.asynccontextmanager
async def heavy_job_slot(message=None):
    if _HEAVY_SLOT_HELD.get():
        # Re-entrant within the same update (e.g. repair -> generate_karaoke_output).
        yield
        return
    if HEAVY_JOB_SEMAPHORE.locked() and message is not None:
        try:
            await message.reply_text("הבקשה נכנסה לתור, ממתין לסיום עבודה קודמת...")
        except Exception:
            logger.info("Could not send queue notice for a waiting heavy job.")
    async with HEAVY_JOB_SEMAPHORE:
        token = _HEAVY_SLOT_HELD.set(True)
        try:
            yield
        finally:
            _HEAVY_SLOT_HELD.reset(token)


async def run_heavy_in_executor(message, func, *args):
    async with heavy_job_slot(message):
        return await asyncio.get_running_loop().run_in_executor(None, func, *args)


def cleanup_delivered_job(job: Job):
    if not job_manager.should_cleanup_delivered_job():
        return
    job_manager.cleanup_job(job)
    run_storage_maintenance()


def _release_single_instance():
    global _SINGLE_INSTANCE_HANDLE
    if _SINGLE_INSTANCE_HANDLE is None or os.name != "nt":
        return
    ctypes.windll.kernel32.CloseHandle(_SINGLE_INSTANCE_HANDLE)
    _SINGLE_INSTANCE_HANDLE = None


def ensure_single_instance():
    global _SINGLE_INSTANCE_HANDLE
    if _SINGLE_INSTANCE_HANDLE is not None or os.name != "nt":
        return

    mutex_name = f"HebrewKaraokeBot::{BASE_DIR.resolve()}".replace("\\", "/")
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
    if not handle:
        raise RuntimeError("Failed to create a single-instance mutex for the bot.")

    last_error = ctypes.windll.kernel32.GetLastError()
    if last_error == _WINDOWS_MUTEX_ALREADY_EXISTS:
        ctypes.windll.kernel32.CloseHandle(handle)
        raise RuntimeError("Another bot instance is already running for this project.")

    _SINGLE_INSTANCE_HANDLE = handle
    atexit.register(_release_single_instance)


def is_group_chat(update: Update) -> bool:
    chat = update.effective_chat
    return bool(chat and chat.type in GROUP_CHAT_TYPES)


def build_delivery_context(chat_id: int, reply_to_message_id: int = 0) -> dict[str, int]:
    return {
        "delivery_chat_id": chat_id,
        "delivery_reply_to_message_id": reply_to_message_id,
    }


def apply_delivery_context(payload: dict[str, object], delivery_context: dict[str, int] | None) -> dict[str, object]:
    if not delivery_context:
        return dict(payload)
    merged = dict(payload)
    merged.update({key: value for key, value in delivery_context.items() if value})
    return merged


def get_default_delivery_target(default_chat_id: int, default_reply_to_message_id: int = 0) -> tuple[int, int]:
    if default_chat_id < 0 or DEFAULT_DELIVERY_CHAT_ID == 0:
        return default_chat_id, default_reply_to_message_id
    return DEFAULT_DELIVERY_CHAT_ID, DEFAULT_DELIVERY_REPLY_TO_MESSAGE_ID


def get_delivery_target(payload: dict[str, object] | None, default_chat_id: int, default_reply_to_message_id: int = 0) -> tuple[int, int]:
    fallback_chat_id, fallback_reply_to_message_id = get_default_delivery_target(default_chat_id, default_reply_to_message_id)
    if not payload:
        return fallback_chat_id, fallback_reply_to_message_id
    chat_id = int(payload.get("delivery_chat_id", 0) or fallback_chat_id)
    reply_to_message_id = int(payload.get("delivery_reply_to_message_id", 0) or fallback_reply_to_message_id)
    return chat_id, reply_to_message_id


def get_job_delivery_target(job: Job) -> tuple[int, int]:
    return job.delivery_chat_id, job.delivery_reply_to_message_id


def callback_job_id(data: str) -> str | None:
    prefixes = (
        "karaoke_page:",
        "karaoke_review:",
        "karaoke_existing:",
        "karaoke_option:",
        "karaoke_edit:",
        "karaoke_approve:",
        "karaoke_back_outputs:",
        "karaoke_output:",
        "karaoke_quality:",
        "delivery_approve:",
        "delivery_reject:",
    )
    for prefix in prefixes:
        if data.startswith(prefix):
            parts = data.split(":")
            return parts[1] if len(parts) > 1 else None
    return None


def user_owns_job(job: Job, user_id: int) -> bool:
    owner_id = int(job.manifest.user_id or 0)
    return owner_id in {0, user_id}


async def send_with_optional_reply(send_callable, *, reply_to_message_id: int = 0, **kwargs):
    # allow_sending_without_reply lets Telegram fall back server-side when the
    # replied-to message is gone, so a slow upload is never re-sent client-side
    # (a retry after TimedOut can deliver the file twice).
    if reply_to_message_id > 0:
        kwargs["reply_parameters"] = ReplyParameters(
            message_id=reply_to_message_id,
            allow_sending_without_reply=True,
        )
    return await send_callable(**kwargs)


async def send_text_to_chat(bot, *, chat_id: int, text: str, reply_to_message_id: int = 0, parse_mode=None, reply_markup=None):
    return await send_with_optional_reply(
        bot.send_message,
        chat_id=chat_id,
        text=text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
        reply_to_message_id=reply_to_message_id,
    )


async def send_document_to_chat(
    bot,
    *,
    chat_id: int,
    document,
    filename: str,
    caption: str = "",
    reply_to_message_id: int = 0,
    read_timeout: int = 120,
    write_timeout: int = 120,
):
    return await send_with_optional_reply(
        bot.send_document,
        chat_id=chat_id,
        document=document,
        filename=filename,
        caption=caption,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        reply_to_message_id=reply_to_message_id,
    )


async def send_audio_to_chat(
    bot,
    *,
    chat_id: int,
    audio,
    filename: str,
    caption: str = "",
    reply_to_message_id: int = 0,
    read_timeout: int = 300,
    write_timeout: int = 300,
    connect_timeout: int = 60,
):
    return await send_with_optional_reply(
        bot.send_audio,
        chat_id=chat_id,
        audio=audio,
        filename=filename,
        caption=caption,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        connect_timeout=connect_timeout,
        reply_to_message_id=reply_to_message_id,
    )


async def send_video_to_chat(
    bot,
    *,
    chat_id: int,
    video,
    filename: str,
    caption: str = "",
    reply_to_message_id: int = 0,
    read_timeout: int = 600,
    write_timeout: int = 600,
    connect_timeout: int = 60,
    supports_streaming: bool = True,
):
    return await send_with_optional_reply(
        bot.send_video,
        chat_id=chat_id,
        video=video,
        filename=filename,
        caption=caption,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        connect_timeout=connect_timeout,
        supports_streaming=supports_streaming,
        reply_to_message_id=reply_to_message_id,
    )


def get_delivery_message_link(delivered_message) -> str:
    link = getattr(delivered_message, "link", "") if delivered_message is not None else ""
    return link.strip() if isinstance(link, str) else ""


def build_delivery_result_text(
    title: str,
    *,
    source_chat_id: int,
    target_chat_id: int,
    delivered_message=None,
) -> tuple[str, str | None]:
    base_text = f"הושלם בהצלחה עבור {title}."
    if source_chat_id == target_chat_id:
        return base_text, None

    link = get_delivery_message_link(delivered_message)
    if not link:
        return f"{base_text}\n\nהתוצאה נשלחה לקבוצת היעד.", None

    safe_title = html.escape(title)
    safe_link = html.escape(link, quote=True)
    return (
        f"הושלם בהצלחה עבור {safe_title}.\n\n<a href=\"{safe_link}\">פתח את התוצאה בקבוצה</a>",
        "HTML",
    )


async def show_delivery_result(message, title: str, *, target_chat_id: int, delivered_message=None):
    text, parse_mode = build_delivery_result_text(
        title,
        source_chat_id=message.chat_id,
        target_chat_id=target_chat_id,
        delivered_message=delivered_message,
    )
    await edit_or_reply(message, text, parse_mode=parse_mode)


def requires_group_delivery_approval(source_chat_id: int, target_chat_id: int) -> bool:
    return source_chat_id > 0 and target_chat_id < 0 and source_chat_id != target_chat_id


def build_delivery_approval_keyboard(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("מושלם, פרסם לקבוצה", callback_data=f"delivery_approve:{job_id}")],
            [InlineKeyboardButton("לא מושלם", callback_data=f"delivery_reject:{job_id}")],
        ]
    )


def build_delivery_feedback_prompt(job: Job) -> str:
    return (
        f"בדקתי את התוצאה עבור {job.display_name} ולא אפרסם אותה לקבוצה עדיין.\n\n"
        "כתוב כאן מה לא היה מושלם, או ערוך את קובץ התבנית שאשלח לך והעלה אותו בחזרה."
    )


def build_delivery_approval_prompt(job: Job) -> str:
    return (
        f"התוצאה עבור {job.display_name} מוכנה לבדיקה.\n\n"
        "בדוק את הקבצים שקיבלת כאן בפרטי. אם הכול מושלם, אשר פרסום לקבוצה. "
        "אם לא, לחץ על 'לא מושלם' ושלח לי מה צריך לתקן."
    )


def build_video_request_from_job(job: Job) -> VideoRequest | None:
    requested = job.manifest.requested_outputs or {}
    if not requested or requested.get("subtitles_only", False):
        return None
    return VideoRequest(
        with_vocals=bool(requested.get("with_vocals")),
        without_vocals=bool(requested.get("without_vocals")),
        quality=str(requested.get("quality") or "best"),
    )


def get_legacy_artifact_path(job: Job, media_type: str) -> Path:
    suffix = ".mp4" if media_type == "video" else ".mp3"
    return job.job_dir / f"legacy_output{suffix}"


def persist_legacy_artifact(job: Job, source_path: str | Path, *, media_type: str) -> Path:
    source = Path(source_path)
    target = get_legacy_artifact_path(job, media_type)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
        try:
            source.unlink()
        except Exception:
            logger.info("Could not remove temporary legacy artifact %s", source)
    return target


def build_legacy_audio_delivery_metadata(job: Job, *, karaoke: bool = False) -> tuple[str, str]:
    title = (job.display_name or job.title or job.job_id).strip() or job.job_id
    safe_title = safe_filename(title)
    if karaoke:
        return f"{safe_title} - קריוקי ללא ווקאל.mp3", f"קריוקי ללא ווקאל: {title}"
    return f"{safe_title}.mp3", title


async def send_legacy_delivery_artifact(
    bot,
    *,
    artifact_path: Path,
    media_type: str,
    chat_id: int,
    filename: str,
    caption: str,
    reply_to_message_id: int = 0,
):
    if media_type == "video":
        with open(artifact_path, "rb") as file_handle:
            return await send_video_to_chat(
                bot,
                chat_id=chat_id,
                video=file_handle,
                filename=filename,
                caption=caption,
                reply_to_message_id=reply_to_message_id,
                read_timeout=600,
                write_timeout=600,
                connect_timeout=60,
                supports_streaming=True,
            )

    with open(artifact_path, "rb") as file_handle:
        return await send_audio_to_chat(
            bot,
            chat_id=chat_id,
            audio=file_handle,
            filename=filename,
            caption=caption,
            reply_to_message_id=reply_to_message_id,
            read_timeout=600,
            write_timeout=600,
            connect_timeout=60,
        )


def build_format_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("MP3", callback_data="format:mp3"),
                InlineKeyboardButton("קריוקי ללא ווקאל", callback_data="format:karaoke"),
            ],
            [InlineKeyboardButton("אקורדים + מילים", callback_data="format:chords")],
            [InlineKeyboardButton("קריוקי וידיאו", callback_data="format:hebrew_karaoke")],
            [
                InlineKeyboardButton("וידאו הכי טוב", callback_data="format:video:best"),
                InlineKeyboardButton("1080p", callback_data="format:video:1080"),
            ],
            [
                InlineKeyboardButton("720p", callback_data="format:video:720"),
                InlineKeyboardButton("480p", callback_data="format:video:480"),
            ],
            [InlineKeyboardButton("360p", callback_data="format:video:360")],
        ]
    )


def build_review_keyboard(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("אשר", callback_data=f"karaoke_approve:{job_id}"),
                InlineKeyboardButton("ערוך", callback_data=f"karaoke_edit:{job_id}"),
            ]
        ]
    )


def build_edit_keyboard(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("חזור", callback_data=f"karaoke_review:{job_id}")]])


def _option_button_label(option: dict[str, object], selected_option_id: str) -> str:
    label = str(option.get("label") or option.get("option_id") or "אפשרות")
    if option.get("option_id") == selected_option_id:
        return f"נבחר: {label}"[:30]
    return label[:30]


def _option_preview_text(option: dict[str, object], max_lines: int = 3, max_chars_per_line: int = 120) -> str:
    lines = [str(line).strip() for line in option.get("lines", []) if str(line).strip()]
    if not lines:
        return ""
    preview_lines = [line[:max_chars_per_line] for line in lines[:max_lines]]
    if len(lines) > max_lines:
        preview_lines.append("...")
    return "\n".join(preview_lines)


def build_dynamic_review_keyboard(job: Job) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton("אשר", callback_data=f"karaoke_approve:{job.job_id}"),
            InlineKeyboardButton("ערוך", callback_data=f"karaoke_edit:{job.job_id}"),
        ]
    ]
    selected_option_id = job_manager.get_selected_lyrics_option_id(job)
    selectable_options = job_manager.get_selectable_lyrics_options(job)
    option_buttons = [
        InlineKeyboardButton(
            _option_button_label(option, selected_option_id),
            callback_data=f"karaoke_option:{job.job_id}:{option['option_id']}",
        )
        for option in selectable_options[:5]
    ]
    for index in range(0, len(option_buttons), 2):
        rows.append(option_buttons[index:index + 2])
    return InlineKeyboardMarkup(rows)


def build_output_keyboard(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("טקסט עם אקורדים", callback_data=f"karaoke_output:{job_id}:chords_text")],
            [InlineKeyboardButton("רק כתוביות", callback_data=f"karaoke_output:{job_id}:subs_only")],
            [InlineKeyboardButton("וידאו עם ווקאל", callback_data=f"karaoke_output:{job_id}:video_vocals")],
            [InlineKeyboardButton("וידאו ללא ווקאל", callback_data=f"karaoke_output:{job_id}:video_instrumental")],
            [InlineKeyboardButton("שני הסרטונים", callback_data=f"karaoke_output:{job_id}:video_both")],
            [InlineKeyboardButton("חזור", callback_data=f"karaoke_review:{job_id}")],
        ]
    )


def build_quality_keyboard(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("הכי טוב", callback_data=f"karaoke_quality:{job_id}:best"),
                InlineKeyboardButton("1080p", callback_data=f"karaoke_quality:{job_id}:1080"),
            ],
            [
                InlineKeyboardButton("720p", callback_data=f"karaoke_quality:{job_id}:720"),
                InlineKeyboardButton("480p", callback_data=f"karaoke_quality:{job_id}:480"),
            ],
            [InlineKeyboardButton("360p", callback_data=f"karaoke_quality:{job_id}:360")],
            [InlineKeyboardButton("חזור", callback_data=f"karaoke_back_outputs:{job_id}")],
        ]
    )


def build_existing_job_keyboard(job: Job) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("להשתמש בקיים", callback_data=f"karaoke_existing:{job.job_id}:reuse")],
    ]
    if job_manager.can_rerender(job):
        rows.append([InlineKeyboardButton("רינדור מחדש", callback_data=f"karaoke_existing:{job.job_id}:rerender")])
    rows.append([InlineKeyboardButton("להתחיל מחדש", callback_data=f"karaoke_existing:{job.job_id}:new")])
    return InlineKeyboardMarkup(rows)


async def edit_or_reply(message, text: str, reply_markup=None, parse_mode=None):
    try:
        await message.edit_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
    except Exception:
        await message.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)


def build_error_message(error: PipelineError, job: Job | None = None) -> str:
    return format_pipeline_error(error, job_id=job.display_name if job else None)


def build_unexpected_error(message: str, job: Job | None = None) -> str:
    return format_unexpected_error(message, job_id=job.display_name if job else None)


async def get_active_review_job(update: Update) -> Job | None:
    if not update.effective_chat or not update.effective_user:
        return None
    return job_manager.get_active_review_job(update.effective_chat.id, update.effective_user.id)


def get_delivery_feedback_job(context: ContextTypes.DEFAULT_TYPE) -> Job | None:
    job_id = str(context.user_data.get("delivery_feedback_job_id") or "").strip()
    if not job_id:
        return None
    try:
        return job_manager.load_job(job_id)
    except FileNotFoundError:
        context.user_data.pop("delivery_feedback_job_id", None)
        return None


def extract_upload_request_payload(message) -> dict[str, object] | None:
    if message.audio:
        return {"file_id": message.audio.file_id, "input_type": "audio_file", "file_name": message.audio.file_name or "audio.mp3"}
    if message.voice:
        return {"file_id": message.voice.file_id, "input_type": "audio_file", "file_name": "voice.ogg"}
    if message.video:
        return {"file_id": message.video.file_id, "input_type": "video_file", "file_name": message.video.file_name or "video.mp4"}
    if message.video_note:
        return {"file_id": message.video_note.file_id, "input_type": "video_file", "file_name": "video_note.mp4"}
    if message.document:
        file_name = message.document.file_name or "file"
        mime = message.document.mime_type or ""
        if mime.startswith("audio/") or file_name.lower().endswith((".mp3", ".wav", ".ogg", ".m4a", ".flac")):
            input_type = "audio_file"
        elif mime.startswith("video/") or file_name.lower().endswith((".mp4", ".mkv", ".avi", ".webm", ".mov")):
            input_type = "video_file"
        else:
            return None
        return {"file_id": message.document.file_id, "input_type": input_type, "file_name": file_name}
    return None


def _build_review_text(job: Job, note: str | None = None) -> str:
    display_text = job_manager.get_display_text(job_manager.load_review_segments(job))
    verification = job.manifest.lyrics_verification or {}
    selected_option_id = job_manager.get_selected_lyrics_option_id(job)
    selectable_options = job_manager.get_selectable_lyrics_options(job)
    reference_option = job_manager.get_reference_lyrics_option(job)

    blocks = [f"משימה: {job.display_name}"]
    if note:
        blocks.append(note)

    if verification:
        llm_provider = str(verification.get("llm_provider", "") or "gemini").strip().lower()
        llm_label = "Grok" if llm_provider in {"grok", "xai"} else "Gemini"
        verdict = str(verification.get("verdict", "")).strip()
        summary = str(verification.get("summary", "")).strip()
        confidence = float(verification.get("confidence", 0.0) or 0.0)
        correction_count = int(verification.get("correction_count", 0) or 0)
        applied = bool(verification.get("applied", False))

        # New consensus-aware display
        consensus_data = verification.get("consensus_result")
        if consensus_data and isinstance(consensus_data, dict):
            if consensus_data.get("consensus_reached"):
                agreed = consensus_data.get("agreed_sources", 0)
                blocks.append(f"✅ מילים אומתו מ-{agreed} מקורות")
            else:
                # Show disputes
                disputes = consensus_data.get("disputes", [])
                if disputes:
                    dispute_lines = ["⚠️ נמצאו הבדלים בין המקורות:"]
                    for d in disputes[:5]:  # Limit to 5 disputes to avoid message overflow
                        line_num = d.get("line_number", 0) + 1
                        versions = d.get("versions", {})
                        gemini_rec = d.get("gemini_recommendation", "")
                        gemini_conf = d.get("gemini_confidence", 0.0)
                        dispute_lines.append(f"שורה {line_num}: ⚠️")
                        for src, ver in versions.items():
                            dispute_lines.append(f"  ├─ {src}: \"{ver}\"")
                        if gemini_rec:
                            pct = int(gemini_conf * 100)
                            dispute_lines.append(f"  └─ {llm_label}: \"{gemini_rec}\" ({pct}%)")
                    blocks.append("\n".join(dispute_lines))
                elif summary:
                    blocks.append(f"אימות: {summary}")
        elif summary:
            blocks.append(f"אימות לפני review: {summary}")

        if confidence:
            blocks.append(f"ציון התאמה לרשת: {confidence:.2f}")
        if applied and correction_count:
            blocks.append(f"תוקנו אוטומטית {correction_count} מילים לפני ההצגה לבדיקה.")
        sources = verification.get("matched_sources") or []
        if sources:
            blocks.append(f"מקור בדיקה ראשון: {sources[0]}")

    if selectable_options:
        option_lines = []
        for option in selectable_options[:5]:
            prefix = "->" if option.get("option_id") == selected_option_id else "-"
            meta = []
            confidence = float(option.get("confidence", 0.0) or 0.0)
            if confidence:
                meta.append(f"{confidence:.2f}")
            source_count = int(option.get("source_count", 0) or 0)
            if source_count > 1:
                meta.append(f"{source_count} מקורות")
            source_url = str(option.get("source_url", "") or "")
            if source_url:
                meta.append(source_url.replace("https://", "").replace("http://", "")[:32])
            suffix = f" ({', '.join(meta)})" if meta else ""
            option_lines.append(f"{prefix} {option.get('label', option.get('option_id', 'אפשרות'))}{suffix}")
        blocks.append("אפשרויות טקסט:\n" + "\n".join(option_lines))
        if job_manager.is_reference_selection_active(job):
            blocks.append("התמלול המקורי מוצג כרגע להשוואה בלבד. כדי להמשיך, בחר אחת מהגרסאות שמצאנו או ערוך ידנית.")
        elif reference_option is not None:
            reference_preview = _option_preview_text(reference_option)
            if reference_preview:
                blocks.append("תמלול מקורי להשוואה בלבד:\n" + reference_preview)

    blocks.append(display_text)

    if job.manifest.warnings:
        blocks.append("אזהרות:\n" + "\n".join(f"- {warning}" for warning in job.manifest.warnings[-3:]))

    blocks.append(
        "לתיקון שורה: 3: הטקסט המתוקן\n"
        "להחלפת כל הטקסט: שלח את כל הטקסט מחדש\n"
        "אפשר גם להעלות transcript.txt"
    )

    text = "\n\n".join(block for block in blocks if block.strip())
    if len(text) > 3900:
        summary_blocks = []
        for block in blocks:
            if block == display_text:
                break
            if block.strip():
                summary_blocks.append(block)
        header = "\n\n".join(summary_blocks)
        footer = "\n\nהטקסט ארוך מדי להצגה מלאה בהודעה אחת. אפשר לתקן לפי שורות או להעלות transcript.txt."
        allowed = max(900, 3900 - len(header) - len(footer) - 4)
        transcript_preview = display_text[:allowed].rstrip()
        text = f"{header}\n\n{transcript_preview}{footer}".strip()
    return text


async def show_review_text(message, job: Job, note: str | None = None):
    text = _build_review_text(job, note=note)
    bot = message.get_bot()
    if job.review_message_id:
        try:
            await bot.edit_message_text(
                chat_id=job.manifest.chat_id,
                message_id=job.review_message_id,
                text=text,
                reply_markup=build_dynamic_review_keyboard(job),
            )
            return
        except Exception:
            logger.info("Could not edit stored review message for %s, sending a new one.", job.job_id)

    sent = await message.reply_text(text, reply_markup=build_dynamic_review_keyboard(job))
    job_manager.set_review_message_id(job, sent.message_id)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat and update.effective_chat.type == "private" and context.args:
        payload = context.args[0].strip()
        if payload.startswith("group_"):
            handled = await resume_group_request_from_start(update, context, payload.removeprefix("group_"))
            if handled:
                return
    await update.message.reply_text(
        "שלום!\n\n"
        "שלח לי קישור מיוטיוב, שם שיר לחיפוש, או קובץ אודיו/וידאו.\n"
        "במסלול הקריוקי העברי הבוט ייצור טיוטה, יאמת אותה מול הרשת ככל האפשר, יחכה לאישור שלך, "
        "ואז יחזיר transcript.txt, timings.json, subtitles.srt, karaoke.ass ו-וידאו אם ביקשת."
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    message_id = update.message.message_id
    deleted = 0
    for item in range(message_id, max(message_id - 1000, 0), -1):
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=item)
            deleted += 1
        except Exception:
            pass
    confirmation = await context.bot.send_message(chat_id=chat_id, text=f"נמחקו {deleted} הודעות.")
    await asyncio.sleep(2)
    await confirmation.delete()


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active_job = await get_active_review_job(update)
    if active_job:
        job_manager.clear_active_review_job(update.effective_chat.id, update.effective_user.id)
    for key in ["chosen", "uploaded_job", "search_results", "delivery_feedback_job_id"]:
        context.user_data.pop(key, None)
    await update.message.reply_text("העיבוד הפעיל הופסק." + COMMANDS_TEXT)


def build_output_delivery_metadata(job: Job, file_name: str, file_path: Path) -> tuple[str, str]:
    title = (job.display_name or "").strip()
    suffix = file_path.suffix or Path(file_name).suffix
    video_labels = {
        "final_video.mp4": f"{title} - כתוביות רצות",
        "final_video_instrumental.mp4": f"{title} - קריוקי",
    }
    caption = video_labels.get(file_name, file_name)
    if file_name in video_labels:
        return f"{safe_filename(caption)}{suffix}", caption
    return f"{safe_filename(job.title)}_{file_name}", caption


async def send_output_files(
    query_or_message,
    job: Job,
    output_files: dict[str, Path],
    *,
    target_chat_id: int | None = None,
    target_reply_to_message_id: int | None = None,
):
    message = getattr(query_or_message, "message", query_or_message)
    bot = message.get_bot()
    delivery_chat_id, delivery_reply_to_message_id = get_job_delivery_target(job)
    if target_chat_id is not None:
        delivery_chat_id = target_chat_id
    if target_reply_to_message_id is not None:
        delivery_reply_to_message_id = target_reply_to_message_id
    first_sent_message = None
    preferred_sent_message = None
    for file_name, file_path in output_files.items():
        output_filename, output_caption = build_output_delivery_metadata(job, file_name, file_path)
        try:
            if file_name.endswith(".mp4"):
                with open(file_path, "rb") as file_handle:
                    sent_message = await send_video_to_chat(
                        bot,
                        chat_id=delivery_chat_id,
                        video=file_handle,
                        filename=output_filename,
                        caption=output_caption,
                        reply_to_message_id=delivery_reply_to_message_id,
                        read_timeout=600,
                        write_timeout=600,
                        connect_timeout=60,
                        supports_streaming=True,
                    )
                    if preferred_sent_message is None:
                        preferred_sent_message = sent_message
            else:
                with open(file_path, "rb") as file_handle:
                    sent_message = await send_document_to_chat(
                        bot,
                        chat_id=delivery_chat_id,
                        document=file_handle,
                        filename=output_filename,
                        caption=output_caption,
                        reply_to_message_id=delivery_reply_to_message_id,
                        read_timeout=120,
                        write_timeout=120,
                    )
            if first_sent_message is None:
                first_sent_message = sent_message
        except Exception as exc:
            raise DeliveryError(str(exc), f"שליחת {file_name} נכשלה.") from exc
    return preferred_sent_message or first_sent_message


def filter_output_files(output_files: dict[str, Path], delivery_mode: str) -> dict[str, Path]:
    if delivery_mode == "chords_text":
        allowed = {"lyrics_with_chords.txt"}
        return {name: path for name, path in output_files.items() if name in allowed}

    blocked = {"lyrics_with_chords.txt", "song_analysis.json"}
    return {name: path for name, path in output_files.items() if name not in blocked}


def chunk_text_for_telegram(text: str, limit: int = 3400) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    current = ""
    for block in normalized.split("\n\n"):
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        while len(block) > limit:
            split_at = block.rfind("\n", 0, limit)
            if split_at <= 0:
                split_at = limit
            chunks.append(block[:split_at].strip())
            block = block[split_at:].strip()
        current = block

    if current:
        chunks.append(current)
    return chunks


def _looks_like_chord_line_token(token: str) -> bool:
    normalized = token.strip().strip("|")
    return bool(normalized) and bool(CHORD_LINE_TOKEN_PATTERN.fullmatch(normalized))


def is_chord_sheet_chord_line(line: str) -> bool:
    tokens = line.split()
    return bool(tokens) and all(_looks_like_chord_line_token(token) for token in tokens)


def mirror_chord_line_for_telegram(line: str) -> str:
    groups = re.findall(r"\s+|\S+", line.rstrip())
    return "".join(reversed(groups)) if groups else line


def format_chord_sheet_for_telegram(text: str) -> str:
    return "\n".join(
        mirror_chord_line_for_telegram(line) if is_chord_sheet_chord_line(line) else line
        for line in text.splitlines()
    )


def should_format_chord_sheet_for_telegram(text: str) -> bool:
    return bool(re.search(r"[\u0590-\u05FF]", text or ""))


def normalize_chord_sheet_key_header(text: str, *, original_key: str, target_key: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    header_end = next((index for index, line in enumerate(lines) if not line.strip()), len(lines))
    header_lines = [
        line
        for line in lines[:header_end]
        if not line.startswith("סולם מקור:") and not line.startswith("סולם קל:")
    ]
    insert_at = next((index + 1 for index, line in enumerate(header_lines) if line.startswith("משקל:")), len(header_lines))
    if original_key:
        header_lines.insert(insert_at, f"סולם מקור: {original_key}")
        insert_at += 1
    if target_key:
        header_lines.insert(insert_at, f"סולם קל: {target_key}")

    normalized_lines = header_lines + lines[header_end:]
    normalized_text = "\n".join(normalized_lines)
    if text.endswith("\n"):
        normalized_text += "\n"
    return normalized_text


def normalize_chord_sheet_text(text: str, analysis=None) -> str:
    if analysis is None:
        return text
    original_key, target_key = resolve_song_analysis_key_labels(analysis)
    return normalize_chord_sheet_key_header(text, original_key=original_key, target_key=target_key)


def build_delivery_chord_text(job: Job, analysis=None) -> tuple[str, object]:
    chord_text = job.lyrics_with_chords_path.read_text(encoding="utf-8")
    if analysis is None:
        return chord_text, analysis

    stored_original_key = (analysis.original_key or "").strip()
    stored_target_key = (analysis.target_key or "").strip()
    resolved_original_key, resolved_target_key = resolve_song_analysis_key_labels(analysis)
    has_external_source = bool((analysis.chord_source_name or "").strip() or (analysis.chord_source_url or "").strip())
    should_rebuild = (
        has_external_source
        and bool(analysis.original_chord_events)
        and (
            (stored_original_key and resolved_original_key and stored_original_key != resolved_original_key)
            or (stored_target_key and resolved_target_key and stored_target_key != resolved_target_key)
        )
    )
    if not should_rebuild:
        return normalize_chord_sheet_key_header(
            chord_text,
            original_key=resolved_original_key,
            target_key=resolved_target_key,
        ), analysis

    try:
        segments = job_manager.load_review_segments(job)
    except Exception:
        try:
            segments = job_manager.load_draft_segments(job)
        except Exception:
            segments = []
    if not segments:
        return normalize_chord_sheet_key_header(
            chord_text,
            original_key=resolved_original_key,
            target_key=resolved_target_key,
        ), analysis

    rebuilt_analysis = prepare_song_analysis_for_display(
        analysis.__class__(
            bpm=analysis.bpm,
            time_signature=analysis.time_signature,
            preview_window_seconds=analysis.preview_window_seconds,
            provider=analysis.provider,
            source_audio=analysis.source_audio,
            beat_times=list(analysis.beat_times),
            measure_times=list(analysis.measure_times),
            original_key="",
            target_key="",
            transpose_semitones=0,
            original_chord_events=list(analysis.original_chord_events),
            chord_events=list(analysis.original_chord_events),
            chord_sheet_text="",
            chord_source_name=analysis.chord_source_name,
            chord_source_url=analysis.chord_source_url,
        ),
        segments,
        target_key=stored_target_key,
    )
    rebuilt_original_key, rebuilt_target_key = resolve_song_analysis_key_labels(rebuilt_analysis)
    rebuilt_analysis.original_key = rebuilt_original_key
    rebuilt_analysis.target_key = rebuilt_target_key
    rebuilt_text = render_chord_sheet_text(job.display_name, segments, rebuilt_analysis)
    return rebuilt_text, rebuilt_analysis


async def send_chords_text_response(
    message,
    job: Job,
    *,
    target_chat_id: int | None = None,
    target_reply_to_message_id: int | None = None,
    include_preview_chunks: bool = True,
):
    if not job.lyrics_with_chords_path.exists():
        raise DeliveryError("lyrics_with_chords.txt is missing", "קובץ האקורדים לא נוצר.")

    bpm_text = "לא זוהה"
    analysis = None
    if job.song_analysis_path.exists():
        try:
            analysis = job_manager.load_song_analysis(job)
            quality_warning = _get_chord_text_delivery_error(analysis)
            if quality_warning:
                summary = summarize_song_analysis_quality(analysis)
                logger.warning(
                    "Sending chord-text for %s despite low confidence: avg_conf=%.3f low_ratio=%.3f chords=%d external=%s",
                    job.job_id,
                    summary.average_confidence,
                    summary.low_confidence_ratio,
                    summary.visible_chord_count,
                    summary.has_external_source,
                )
            if analysis.bpm > 0:
                bpm_text = f"{analysis.bpm:.2f}"
        except Exception:
            pass

    chord_text, analysis = build_delivery_chord_text(job, analysis)
    prefixed_text = f"אקורדים + מילים עבור: {job.display_name}\nקצב: {bpm_text}\n\n{chord_text}".strip()
    preview_text = (
        format_chord_sheet_for_telegram(prefixed_text)
        if should_format_chord_sheet_for_telegram(prefixed_text)
        else prefixed_text
    )
    delivery_chat_id, delivery_reply_to_message_id = get_job_delivery_target(job)
    if target_chat_id is not None:
        delivery_chat_id = target_chat_id
    if target_reply_to_message_id is not None:
        delivery_reply_to_message_id = target_reply_to_message_id
    if include_preview_chunks:
        if message.chat_id == delivery_chat_id:
            for chunk in chunk_text_for_telegram(preview_text):
                await message.reply_text(f"<pre>{html.escape(chunk)}</pre>", parse_mode="HTML")
        else:
            for chunk in chunk_text_for_telegram(preview_text)[:2]:
                await message.reply_text(f"<pre>{html.escape(chunk)}</pre>", parse_mode="HTML")

    delivery_text = (
        format_chord_sheet_for_telegram(chord_text)
        if should_format_chord_sheet_for_telegram(chord_text)
        else chord_text
    )
    with io.BytesIO(delivery_text.encode("utf-8")) as file_handle:
        delivered_message = await send_document_to_chat(
            message.get_bot(),
            chat_id=delivery_chat_id,
            document=file_handle,
            filename=f"{safe_filename(job.title)}_lyrics_with_chords.txt",
            caption="אקורדים + מילים",
            reply_to_message_id=delivery_reply_to_message_id,
            read_timeout=120,
            write_timeout=120,
        )

    return delivered_message


async def send_delivery_feedback_template(message, job: Job):
    template_path = job_manager.write_delivery_feedback_template(job)
    with open(template_path, "rb") as file_handle:
        await send_document_to_chat(
            message.get_bot(),
            chat_id=message.chat_id,
            document=file_handle,
            filename=f"{safe_filename(job.title or job.job_id)}_feedback_template.txt",
            caption="תבנית להערות ותיקונים",
            read_timeout=120,
            write_timeout=120,
        )


async def prompt_group_delivery_approval(message, job: Job, *, delivery_mode: str):
    pending = job_manager.update_pending_delivery(
        job,
        status="pending_approval",
        source_chat_id=message.chat_id,
        target_chat_id=job.delivery_chat_id,
        target_reply_to_message_id=job.delivery_reply_to_message_id,
        delivery_mode=delivery_mode,
    )
    logger.info("Job %s waiting for delivery approval: %s", job.job_id, pending)
    await edit_or_reply(
        message,
        build_delivery_approval_prompt(job),
        reply_markup=build_delivery_approval_keyboard(job.job_id),
    )


async def repair_delivery_feedback_and_preview(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    job: Job,
    feedback_text: str,
) -> bool:
    await message.reply_text("קיבלתי את ההערה. מתחיל תיקון אוטומטי ובודק את התוצאה מחדש.")
    loop = asyncio.get_running_loop()
    repair_result = await loop.run_in_executor(None, apply_feedback_to_review, job, feedback_text)
    job = job_manager.load_job(job.job_id)

    if repair_result.applied:
        delivery_mode = str((job.pending_delivery or {}).get("delivery_mode") or "default")
        line_text = ", ".join(str(line_number) for line_number in repair_result.line_numbers)
        job_manager.update_pending_delivery(
            job,
            status="repairing",
            auto_repair_kind="review_line_edits",
            auto_repair_edit_count=repair_result.edit_count,
            auto_repair_lines=line_text,
            auto_repair_started_at=job_manager._now_iso(),
        )
        await message.reply_text(f"תיקנתי את השורות {line_text}. עכשיו מרנדר מחדש ושולח לך גרסה מתוקנת לבדיקה.")
        active_user_id = int(getattr(getattr(message, "from_user", None), "id", 0) or context.user_data.get("active_user_id", 0) or 0)
        context.user_data["active_user_id"] = active_user_id
        context.user_data[f"output_mode:{job.job_id}"] = "rerender"
        context.user_data[f"delivery_mode:{job.job_id}"] = delivery_mode
        await generate_karaoke_output(SimpleNamespace(message=message), context, job, build_video_request_from_job(job))
        return True

    if feedback_mentions_timing_problem(feedback_text):
        if not (job.vocals_16k_path.exists() or job.vocals_path.exists()):
            job_manager.update_pending_delivery(
                job,
                status="manual_review_needed",
                auto_repair_kind="timing_realign",
                auto_repair_message="Timing feedback was detected, but no isolated vocals are available for realignment.",
                auto_repair_finished_at=job_manager._now_iso(),
            )
            await message.reply_text(
                "זיהיתי שהבעיה היא תזמון, אבל אין לי קובץ ווקאל שמור ליישור מחדש. "
                "שמרתי את ההערה בלוג התיקונים לבדיקה ידנית."
            )
            return False

        delivery_mode = str((job.pending_delivery or {}).get("delivery_mode") or "default")
        job_manager.update_pending_delivery(
            job,
            status="repairing",
            auto_repair_kind="timing_realign",
            auto_repair_message="Free-form timing feedback detected; rerunning alignment and rendering.",
            auto_repair_started_at=job_manager._now_iso(),
        )
        await message.reply_text("זיהיתי בעיית תזמון/סנכרון. מיישר את המילים מחדש ומרנדר גרסה מתוקנת לבדיקה.")
        active_user_id = int(getattr(getattr(message, "from_user", None), "id", 0) or context.user_data.get("active_user_id", 0) or 0)
        context.user_data["active_user_id"] = active_user_id
        context.user_data[f"output_mode:{job.job_id}"] = "rerender"
        context.user_data[f"delivery_mode:{job.job_id}"] = delivery_mode
        await generate_karaoke_output(SimpleNamespace(message=message), context, job, build_video_request_from_job(job))
        return True

    codex_result = await loop.run_in_executor(None, run_codex_auto_repair, job, feedback_text)
    job = job_manager.load_job(job.job_id)
    job_manager.update_pending_delivery(
        job,
        status="code_repair_completed" if codex_result.success else "manual_review_needed",
        auto_repair_kind="codex" if codex_result.enabled else "feedback_only",
        auto_repair_attempted=codex_result.attempted,
        auto_repair_message=codex_result.message,
        auto_repair_log_path=str(codex_result.log_path or ""),
        auto_repair_finished_at=job_manager._now_iso(),
    )

    if not codex_result.enabled:
        await message.reply_text(
            "שמרתי את ההערה בלוג התיקונים. לא מצאתי בה תיקון שורה שאפשר להחיל אוטומטית עכשיו. "
            "לתיקון מילים שלח בפורמט: שורה 12: הטקסט הנכון."
        )
        return False

    if codex_result.success:
        await message.reply_text(
            "Codex סיים תיקון קוד לפי ההערה ושמר לוג במשימת העבודה. "
            "אם השתנו קבצי קוד, צריך להפעיל את הבוט מחדש כדי שהגרסה החדשה תרוץ."
        )
        return True

    await message.reply_text(
        "ניסיתי להפעיל תיקון קוד אוטומטי, אבל הוא לא הושלם. ההערה נשמרה בלוג התיקונים לבדיקה ידנית."
    )
    return False


async def save_delivery_feedback_text(message, context: ContextTypes.DEFAULT_TYPE, job: Job, text: str, *, source: str):
    cleaned_text = text.strip()
    if not cleaned_text:
        await message.reply_text("לא קיבלתי פירוט. כתוב מה לא היה תקין או שלח את קובץ התבנית המעודכן.")
        return
    job_manager.append_quality_feedback(
        job,
        cleaned_text,
        source=source,
        user_id=int(getattr(getattr(message, "from_user", None), "id", 0) or 0),
        chat_id=message.chat_id,
    )
    context.user_data.pop("delivery_feedback_job_id", None)
    async with heavy_job_slot(message):
        await repair_delivery_feedback_and_preview(message, context, job, cleaned_text)


async def handle_delivery_feedback_file(update: Update, context: ContextTypes.DEFAULT_TYPE, job: Job):
    document = update.message.document
    target_path = job.job_dir / "uploaded_delivery_feedback.txt"
    tg_file = await document.get_file()
    await tg_file.download_to_drive(str(target_path))
    feedback_text = target_path.read_text(encoding="utf-8", errors="ignore")
    await save_delivery_feedback_text(update.message, context, job, feedback_text, source="text_file")


async def publish_job_to_group(message, job: Job):
    pending = dict(job.pending_delivery or {})
    delivery_mode = str(pending.get("delivery_mode") or "default")
    target_chat_id = int(pending.get("target_chat_id") or job.delivery_chat_id)
    target_reply_to_message_id = int(pending.get("target_reply_to_message_id") or job.delivery_reply_to_message_id)

    if delivery_mode == "legacy_media":
        artifact_path_value = str(pending.get("artifact_path") or "").strip()
        if not artifact_path_value:
            raise DeliveryError("Missing legacy artifact path", "לא מצאתי את הקובץ שמוכן לפרסום לקבוצה.")
        artifact_path = Path(artifact_path_value)
        if not artifact_path.is_absolute():
            artifact_path = job.job_dir / artifact_path
        if not artifact_path.exists():
            raise DeliveryError("Legacy artifact is missing", "הקובץ שאושר כבר לא קיים בדיסק.")
        delivered_message = await send_legacy_delivery_artifact(
            message.get_bot(),
            artifact_path=artifact_path,
            media_type=str(pending.get("artifact_media_type") or ("video" if artifact_path.suffix.lower() == ".mp4" else "audio")),
            chat_id=target_chat_id,
            filename=str(pending.get("artifact_filename") or artifact_path.name),
            caption=str(pending.get("artifact_caption") or "הנה התוצאה שלך."),
            reply_to_message_id=target_reply_to_message_id,
        )
    elif delivery_mode == "chords_text":
        delivered_message = await send_chords_text_response(
            message,
            job,
            target_chat_id=target_chat_id,
            target_reply_to_message_id=target_reply_to_message_id,
            include_preview_chunks=False,
        )
    else:
        video_request = build_video_request_from_job(job)
        output_files = job_manager.get_output_files(job, video_request=video_request)
        output_files = filter_output_files(output_files, delivery_mode)
        if not output_files:
            raise DeliveryError("No output files available", "לא מצאתי קבצים מוכנים לפרסום לקבוצה.")
        delivered_message = await send_output_files(
            message,
            job,
            output_files,
            target_chat_id=target_chat_id,
            target_reply_to_message_id=target_reply_to_message_id,
        )

    job_manager.update_pending_delivery(
        job,
        status="published",
        published_at=job_manager._now_iso(),
        preview_chat_id=message.chat_id,
    )
    return delivered_message


def has_current_chord_sheet(job: Job, expected_provider: str) -> bool:
    if not job.lyrics_with_chords_path.exists() or not job.song_analysis_path.exists():
        return False
    try:
        analysis = job_manager.load_song_analysis(job)
    except Exception:
        return False
    if analysis.provider != expected_provider:
        return False
    if _get_chord_text_delivery_error(analysis):
        return False
    return bool(analysis.chord_sheet_text.strip() or analysis.chord_events or analysis.original_chord_events)


def _get_chord_text_delivery_error(analysis) -> str | None:
    summary = summarize_song_analysis_quality(analysis)
    if summary.reliable_for_delivery:
        return None
    if summary.visible_chord_count <= 0:
        return (
            "לא הצלחתי לייצר דף אקורדים אמין לפרסום. "
            "לא זוהו מספיק אקורדים ברורים מהאודיו."
        )
    return (
        "לא הצלחתי לייצר אקורדים מספיק אמינים לפרסום. "
        "הזיהוי האוטומטי של האקורדים יצא חלש מדי, אז עצרתי לפני שליחת טקסט שעלול להטעות."
    )


def _title_needs_resolution_for_external_lookup(title: str) -> bool:
    normalized = (title or "").strip()
    return not normalized or is_youtube_url(normalized) or normalized.startswith(("http://", "https://"))


async def resolve_chords_lookup_title(chosen: dict[str, object], loop: asyncio.AbstractEventLoop) -> str:
    current_title = str(chosen.get("title") or "").strip()
    if not _title_needs_resolution_for_external_lookup(current_title):
        return current_title

    url = str(chosen.get("url") or "").strip()
    if not url:
        return current_title

    try:
        resolved_title = await loop.run_in_executor(None, resolve_media_title, url)
    except Exception as exc:
        logger.info("Could not resolve media title for fast external lookup: %s", exc)
        return current_title

    if resolved_title:
        chosen["title"] = resolved_title
        return resolved_title
    return current_title


async def deliver_direct_chords_job(message, job: Job):
    if requires_group_delivery_approval(message.chat_id, job.delivery_chat_id):
        job_manager.update_review_status(job, ReviewStatus.APPROVED)
        await send_chords_text_response(
            message,
            job,
            target_chat_id=message.chat_id,
            target_reply_to_message_id=0,
        )
        await prompt_group_delivery_approval(message, job, delivery_mode="chords_text")
        return

    delivered_message = await send_chords_text_response(message, job)
    job_manager.update_review_status(job, ReviewStatus.COMPLETED)
    cleanup_delivered_job(job)
    await show_delivery_result(
        message,
        job.display_name,
        target_chat_id=job.delivery_chat_id,
        delivered_message=delivered_message,
    )


async def try_fast_external_chords_for_job(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    chosen: dict[str, object],
    job: Job,
    pipeline: KaraokePipeline,
) -> bool:
    if chosen.get("local_path"):
        return False

    loop = asyncio.get_running_loop()
    lookup_title = await resolve_chords_lookup_title(chosen, loop)
    if _title_needs_resolution_for_external_lookup(lookup_title):
        return False

    if job.manifest.title != lookup_title:
        job.manifest.title = lookup_title
        job_manager.save_job(job)

    await edit_or_reply(message, "מחפש דף אקורדים חיצוני לפני הורדה...")
    try:
        analysis = await loop.run_in_executor(
            None,
            lambda: lookup_external_chord_sheet_by_title(
                lookup_title,
                provider=pipeline.song_analyzer.name,
                target_key="",
            ),
        )
    except Exception as exc:
        logger.info("Fast external chord lookup failed for %s: %s", job.job_id, exc)
        return False

    if analysis is None or not analysis.chord_sheet_text.strip():
        return False

    job_manager.save_song_analysis(job, analysis)
    job_manager.save_chord_sheet(job, analysis.chord_sheet_text)
    job_manager.update_status(job, JobStatus.DONE)
    await edit_or_reply(message, f"נמצא דף אקורדים חיצוני עבור {job.display_name}, שולח בלי הורדה מיותרת...")
    await deliver_direct_chords_job(message, job)
    return True


async def generate_direct_chords_output(message, context: ContextTypes.DEFAULT_TYPE):
    chosen = context.user_data.get("chosen")
    if not chosen:
        await edit_or_reply(message, "לא נמצא קלט לעיבוד.")
        return

    loop = asyncio.get_running_loop()
    existing_job = None
    existing_pipeline = None
    delivery_chat_id, delivery_reply_to_message_id = get_delivery_target(chosen, message.chat_id)
    if chosen.get("url"):
        existing_job = job_manager.find_latest_reusable_job(
            source_url=chosen.get("url", ""),
            input_type="youtube",
            user_id=context.user_data["active_user_id"],
        )
        if existing_job is not None:
            job_manager.update_job_delivery(
                existing_job,
                delivery_chat_id=delivery_chat_id,
                delivery_reply_to_message_id=delivery_reply_to_message_id,
            )
            existing_pipeline = KaraokePipeline(existing_job)

    if existing_job and existing_pipeline and has_current_chord_sheet(existing_job, existing_pipeline.song_analyzer.name):
        await edit_or_reply(message, f"נמצא עיבוד קיים עבור {existing_job.display_name}, שולח אקורדים + מילים...")
        await deliver_direct_chords_job(message, existing_job)
        return

    if existing_job and existing_pipeline and await try_fast_external_chords_for_job(
        message,
        context,
        chosen,
        existing_job,
        existing_pipeline,
    ):
        return

    if existing_job and job_manager.can_rerender(existing_job):
        pipeline = existing_pipeline or KaraokePipeline(existing_job)
        try:
            await edit_or_reply(message, "משלים אקורדים + מילים מהחומרים הקיימים...")
            await loop.run_in_executor(None, pipeline.rerender_existing_outputs, None)
            await deliver_direct_chords_job(message, existing_job)
        except PipelineError as exc:
            logger.error("Direct chord rerender error: %s\n%s", exc, traceback.format_exc())
            job_manager.update_status(existing_job, JobStatus.ERROR, exc.info)
            await edit_or_reply(message, build_error_message(exc, existing_job))
        except Exception as exc:
            logger.error("Unexpected direct chord rerender error: %s\n%s", exc, traceback.format_exc())
            await edit_or_reply(message, build_unexpected_error(str(exc), existing_job))
        return

    if chosen.get("local_path"):
        run_storage_maintenance()
        job = context.user_data.get("uploaded_job")
        if job:
            job_manager.update_job_delivery(
                job,
                delivery_chat_id=delivery_chat_id,
                delivery_reply_to_message_id=delivery_reply_to_message_id,
            )
        if not job:
            job = job_manager.create_job(
                title=chosen.get("title", "input"),
                input_type=chosen.get("input_type", "audio_file"),
                has_video=(chosen.get("input_type") == "video_file"),
                chat_id=message.chat_id,
                user_id=context.user_data["active_user_id"],
                delivery_chat_id=delivery_chat_id,
                delivery_reply_to_message_id=delivery_reply_to_message_id,
            )
        # review rebuild belongs to the dedicated review flow, not direct chords
        note = "השורה עודכנה."
    else:
        run_storage_maintenance()
        resolved_title = await resolve_chords_lookup_title(chosen, loop)
        job = job_manager.create_job(
            title=resolved_title or chosen.get("title", "song"),
            source_url=chosen.get("url", ""),
            input_type="youtube",
            chat_id=message.chat_id,
            user_id=context.user_data["active_user_id"],
            delivery_chat_id=delivery_chat_id,
            delivery_reply_to_message_id=delivery_reply_to_message_id,
        )

    pipeline = KaraokePipeline(job)
    if await try_fast_external_chords_for_job(message, context, chosen, job, pipeline):
        return

    try:
        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.DOWNLOADING])
        audio_path = await loop.run_in_executor(None, pipeline.step_get_audio, chosen.get("local_path"))

        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.SEPARATING_VOCALS])
        vocals_path, _instrumental = await loop.run_in_executor(None, pipeline.step_separate_vocals, audio_path)

        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.DETECTING_LANGUAGE])
        language_info = await loop.run_in_executor(None, pipeline.step_detect_language, vocals_path)

        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.TRANSCRIBING])
        draft = await loop.run_in_executor(None, pipeline.step_transcribe, vocals_path, language_info)

        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.VERIFYING_LYRICS])
        await loop.run_in_executor(None, pipeline.step_verify_lyrics, draft)

        approved_segments = job_manager.load_review_segments(job)
        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.ALIGNING])
        await loop.run_in_executor(None, pipeline.run_after_review, approved_segments, None)

        await deliver_direct_chords_job(message, job)
    except PipelineError as exc:
        logger.error("Direct chord pipeline error: %s\n%s", exc, traceback.format_exc())
        job_manager.update_status(job, JobStatus.ERROR, exc.info)
        await edit_or_reply(message, build_error_message(exc, job))
    except Exception as exc:
        logger.error("Unexpected direct chord pipeline error: %s\n%s", exc, traceback.format_exc())
        await edit_or_reply(message, build_unexpected_error(str(exc), job))


async def run_karaoke_until_review(message, context: ContextTypes.DEFAULT_TYPE):
    chosen = context.user_data.get("chosen")
    if not chosen:
        await edit_or_reply(message, "לא נמצא קלט לעיבוד.")
        return

    reuse_job_id = context.user_data.pop("reuse_job_id", None)
    delivery_chat_id, delivery_reply_to_message_id = get_delivery_target(chosen, message.chat_id)
    if reuse_job_id:
        job = job_manager.load_job(reuse_job_id)
        job_manager.update_job_delivery(
            job,
            delivery_chat_id=delivery_chat_id,
            delivery_reply_to_message_id=delivery_reply_to_message_id,
        )
    elif chosen.get("local_path"):
        run_storage_maintenance()
        job = context.user_data.get("uploaded_job")
        if job:
            job_manager.update_job_delivery(
                job,
                delivery_chat_id=delivery_chat_id,
                delivery_reply_to_message_id=delivery_reply_to_message_id,
            )
        if not job:
            job = job_manager.create_job(
                title=chosen.get("title", "input"),
                input_type=chosen.get("input_type", "audio_file"),
                has_video=(chosen.get("input_type") == "video_file"),
                chat_id=message.chat_id,
                user_id=context.user_data["active_user_id"],
                delivery_chat_id=delivery_chat_id,
                delivery_reply_to_message_id=delivery_reply_to_message_id,
            )
    else:
        run_storage_maintenance()
        job = job_manager.create_job(
            title=chosen.get("title", "song"),
            source_url=chosen.get("url", ""),
            input_type="youtube",
            chat_id=message.chat_id,
            user_id=context.user_data["active_user_id"],
            delivery_chat_id=delivery_chat_id,
            delivery_reply_to_message_id=delivery_reply_to_message_id,
        )

    pipeline = KaraokePipeline(job)
    loop = asyncio.get_running_loop()
    try:
        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.DOWNLOADING])
        audio_path = await loop.run_in_executor(None, pipeline.step_get_audio, chosen.get("local_path"))

        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.SEPARATING_VOCALS])
        vocals_path, _instrumental = await loop.run_in_executor(None, pipeline.step_separate_vocals, audio_path)

        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.DETECTING_LANGUAGE])
        language_info = await loop.run_in_executor(None, pipeline.step_detect_language, vocals_path)

        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.TRANSCRIBING])
        draft = await loop.run_in_executor(None, pipeline.step_transcribe, vocals_path, language_info)

        await edit_or_reply(message, STATUS_MESSAGES[JobStatus.VERIFYING_LYRICS])
        await loop.run_in_executor(None, pipeline.step_verify_lyrics, draft)

        job_manager.update_status(job, JobStatus.AWAITING_REVIEW)
        job_manager.update_review_status(job, ReviewStatus.AWAITING_REVIEW)
        job_manager.set_active_review_job(message.chat_id, context.user_data["active_user_id"], job.job_id)
        await show_review_text(message, job)
    except PipelineError as exc:
        logger.error("Karaoke pipeline error: %s\n%s", exc, traceback.format_exc())
        job_manager.update_status(job, JobStatus.ERROR, exc.info)
        await edit_or_reply(message, build_error_message(exc, job))
    except Exception as exc:
        logger.error("Unexpected pipeline error: %s\n%s", exc, traceback.format_exc())
        await edit_or_reply(message, build_unexpected_error(str(exc), job))


async def generate_karaoke_output(query, context: ContextTypes.DEFAULT_TYPE, job: Job, video_request: VideoRequest | None):
    pipeline = KaraokePipeline(job)
    loop = asyncio.get_running_loop()
    approved_segments = job_manager.load_review_segments(job)
    job_manager.record_requested_outputs(job, video_request)
    output_mode = context.user_data.get(f"output_mode:{job.job_id}", "fresh")
    delivery_mode = context.user_data.get(f"delivery_mode:{job.job_id}", "default")

    try:
        if output_mode == "reuse":
            output_files = job_manager.get_output_files(job, video_request=video_request)
            needs_subtitles = not (job.srt_path.exists() and job.ass_path.exists())
            needs_music_outputs = delivery_mode == "chords_text" and not (
                job.song_analysis_path.exists() and job.lyrics_with_chords_path.exists()
            )
            needs_requested_video = bool(
                video_request
                and (
                    (video_request.with_vocals and not job.video_vocals_path.exists())
                    or (video_request.without_vocals and not job.video_instrumental_path.exists())
                )
            )
            if needs_subtitles or needs_requested_video or needs_music_outputs:
                output_files = await loop.run_in_executor(None, pipeline.rerender_existing_outputs, video_request)
        elif output_mode == "rerender" and pipeline.can_realign_after_review():
            if video_request and job.input_type == "youtube" and job.source_url:
                await edit_or_reply(query.message, "מוריד את הווידאו המקורי מיוטיוב...")
                await loop.run_in_executor(None, pipeline.download_youtube_video, video_request.quality)
            await edit_or_reply(query.message, STATUS_MESSAGES[JobStatus.ALIGNING])
            output_files = await loop.run_in_executor(None, pipeline.run_after_review, approved_segments, video_request)
        elif output_mode == "rerender":
            await edit_or_reply(query.message, "מרנדר מחדש מהטיימינגים הקיימים...")
            output_files = await loop.run_in_executor(None, pipeline.rerender_existing_outputs, video_request)
        else:
            if video_request and job.input_type == "youtube" and job.source_url:
                await edit_or_reply(query.message, "מוריד את הווידאו המקורי מיוטיוב...")
                await loop.run_in_executor(None, pipeline.download_youtube_video, video_request.quality)
            await edit_or_reply(query.message, STATUS_MESSAGES[JobStatus.ALIGNING])
            output_files = await loop.run_in_executor(None, pipeline.run_after_review, approved_segments, video_request)

        output_files = filter_output_files(output_files, delivery_mode)
        await edit_or_reply(query.message, STATUS_MESSAGES[JobStatus.DELIVERING])
        if requires_group_delivery_approval(query.message.chat_id, job.delivery_chat_id):
            delivered_message = await send_output_files(
                query,
                job,
                output_files,
                target_chat_id=query.message.chat_id,
                target_reply_to_message_id=0,
            )
            job_manager.update_review_status(job, ReviewStatus.APPROVED)
        else:
            delivered_message = await send_output_files(query, job, output_files)
            job_manager.update_review_status(job, ReviewStatus.COMPLETED)
        job_manager.clear_active_review_job(query.message.chat_id, context.user_data["active_user_id"])
        context.user_data.pop(f"output_mode:{job.job_id}", None)
        context.user_data.pop(f"delivery_mode:{job.job_id}", None)
        if requires_group_delivery_approval(query.message.chat_id, job.delivery_chat_id):
            await prompt_group_delivery_approval(query.message, job, delivery_mode=delivery_mode)
        else:
            cleanup_delivered_job(job)
            await show_delivery_result(
                query.message,
                job.display_name,
                target_chat_id=job.delivery_chat_id,
                delivered_message=delivered_message,
            )
    except PipelineError as exc:
        logger.error("Final generation error: %s\n%s", exc, traceback.format_exc())
        job_manager.update_status(job, JobStatus.ERROR, exc.info)
        await edit_or_reply(query.message, build_error_message(exc, job))
    except Exception as exc:
        logger.error("Unexpected final generation error: %s\n%s", exc, traceback.format_exc())
        await edit_or_reply(query.message, build_unexpected_error(str(exc), job))


async def handle_karaoke_correction(update: Update, job: Job, text: str):
    review_segments = job_manager.load_review_segments(job)
    draft_segments = job_manager.load_draft_segments(job) if job.draft_timings_path.exists() else review_segments
    line_match = re.match(r"^(\d+)\s*:\s*(.+)", text)
    if line_match:
        line_number = int(line_match.group(1))
        index = line_number - 1
        corrected_line = line_match.group(2).strip()
        visible_segments = review_segments or draft_segments
        try:
            if index < 0 or index >= len(visible_segments):
                raise ValueError(f"מספר שורה לא תקין: {line_number}")
            reference_segment = draft_segments[index] if index < len(draft_segments) else visible_segments[index]
            corrected_segment = job_manager.update_transcript_line([reference_segment], 1, corrected_line)[0]
        except ValueError as exc:
            await update.message.reply_text(f"שגיאה: {exc}")
            return
        segments = list(visible_segments)
        segments[index] = corrected_segment
        note = f"שורה {line_number} עודכנה."
    else:
        shrink = job_manager.detect_suspicious_review_shrink(draft_segments, text)
        if shrink is not None:
            await update.message.reply_text(
                "הטקסט ששלחת קצר משמעותית מהגרסה הנוכחית. "
                "אם אשמור אותו כמו שהוא, יהיו חלקים בשיר בלי כתוביות. "
                "אם רצית לתקן שורה מסוימת, שלח `מספר: טקסט`. "
                "אם רצית להחליף את כל השיר, שלח את כל המילים המלאות.",
                parse_mode="Markdown",
            )
            return
        reference_segments = draft_segments or review_segments
        segments = job_manager.update_transcript_text(reference_segments, text)
        note = "הטקסט המלא עודכן."

    job_manager.save_review_transcript(job, segments)
    job_manager.save_manual_review_option(job, segments)
    await show_review_text(update.message, job, note=note)


async def handle_app_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    details = ""
    if context.error:
        details = "".join(traceback.format_exception(type(context.error), context.error, context.error.__traceback__))
    logger.error("Unhandled Telegram error: %s\n%s", context.error, details)
    message = getattr(update, "effective_message", None)
    if message is None:
        return
    try:
        await message.reply_text("אירעה שגיאה לא צפויה בזמן הטיפול בהודעה. נסה שוב.")
    except Exception:
        logger.exception("Failed to send Telegram error message to the user.")


async def handle_review_text_file(update: Update, job: Job):
    document = update.message.document
    target_path = job.job_dir / "uploaded_review.txt"
    tg_file = await document.get_file()
    await tg_file.download_to_drive(str(target_path))
    corrected_text = target_path.read_text(encoding="utf-8")
    draft_segments = job_manager.load_draft_segments(job) if job.draft_timings_path.exists() else job_manager.load_review_segments(job)
    shrink = job_manager.detect_suspicious_review_shrink(draft_segments, corrected_text)
    if shrink is not None:
        await update.message.reply_text(
            "הקובץ שהועלה קצר משמעותית מהגרסה הנוכחית, ולכן לא שמרתי אותו כדי לא ליצור חורים בכתוביות. "
            "העלה קובץ עם כל מילות השיר או תקן שורות בודדות."
        )
        return
    reference_segments = draft_segments or job_manager.load_review_segments(job)
    segments = job_manager.update_transcript_text(reference_segments, corrected_text)
    job_manager.save_review_transcript(job, segments)
    job_manager.save_manual_review_option(job, segments)
    await show_review_text(update.message, job, note="הטקסט מתוך הקובץ נשמר.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active_user_id = int(getattr(update.effective_user, "id", 0) or 0)
    if active_user_id:
        context.user_data["active_user_id"] = active_user_id
    text = update.message.text.strip()
    if is_group_chat(update):
        await handoff_group_request(
            update,
            context,
            request_kind="text",
            request_payload={"text": text},
        )
        return
    feedback_job = get_delivery_feedback_job(context)
    if feedback_job is not None:
        await save_delivery_feedback_text(update.message, context, feedback_job, text, source="text")
        return
    review_job = await get_active_review_job(update)
    if review_job and review_job.review_status in {ReviewStatus.AWAITING_REVIEW, ReviewStatus.APPROVED}:
        if looks_like_new_request(text):
            job_manager.clear_active_review_job(update.effective_chat.id, update.effective_user.id)
            await update.message.reply_text("הייתה משימת review פתוחה מהפעם הקודמת, סגרתי אותה ומתחיל בקלט החדש.")
            review_job = None
        else:
            await handle_karaoke_correction(update, review_job, text)
            return

    await begin_song_request(update.message, context, text)


async def handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active_user_id = int(getattr(update.effective_user, "id", 0) or 0)
    if active_user_id:
        context.user_data["active_user_id"] = active_user_id
    if is_group_chat(update):
        upload_payload = extract_upload_request_payload(update.message)
        if not upload_payload:
            await update.message.reply_text("סוג קובץ לא נתמך. שלח אודיו, וידאו או transcript.txt." + COMMANDS_TEXT)
            return
        await handoff_group_request(
            update,
            context,
            request_kind="upload",
            request_payload=upload_payload,
        )
        return
    feedback_job = get_delivery_feedback_job(context)
    if feedback_job is not None:
        if update.message.document:
            file_name = (update.message.document.file_name or "").lower()
            mime = update.message.document.mime_type or ""
            if file_name.endswith(".txt") or mime == "text/plain":
                await handle_delivery_feedback_file(update, context, feedback_job)
                return
        await update.message.reply_text("כדי לתעד מה לא היה מושלם, שלח טקסט רגיל או קובץ txt.")
        return
    review_job = await get_active_review_job(update)
    if review_job and update.message.document:
        file_name = (update.message.document.file_name or "").lower()
        mime = update.message.document.mime_type or ""
        if file_name.endswith(".txt") or mime == "text/plain":
            await handle_review_text_file(update, review_job)
            return

    upload_payload = extract_upload_request_payload(update.message)
    if not upload_payload:
        await update.message.reply_text("סוג קובץ לא נתמך. שלח אודיו, וידאו או transcript.txt." + COMMANDS_TEXT)
        return
    await prepare_uploaded_choice_from_payload(
        update.message,
        context,
        user_id=update.effective_user.id,
        upload_payload=upload_payload,
    )
    return

    message = update.message
    if message.audio:
        file_obj, input_type, file_name = message.audio, "audio_file", message.audio.file_name or "audio.mp3"
    elif message.voice:
        file_obj, input_type, file_name = message.voice, "audio_file", "voice.ogg"
    elif message.video:
        file_obj, input_type, file_name = message.video, "video_file", message.video.file_name or "video.mp4"
    elif message.video_note:
        file_obj, input_type, file_name = message.video_note, "video_file", "video_note.mp4"
    elif message.document:
        file_obj = message.document
        file_name = message.document.file_name or "file"
        mime = message.document.mime_type or ""
        if mime.startswith("audio/") or file_name.lower().endswith((".mp3", ".wav", ".ogg", ".m4a", ".flac")):
            input_type = "audio_file"
        elif mime.startswith("video/") or file_name.lower().endswith((".mp4", ".mkv", ".avi", ".webm", ".mov")):
            input_type = "video_file"
        else:
            await message.reply_text("סוג קובץ לא נתמך. שלח אודיו, וידאו או transcript.txt." + COMMANDS_TEXT)
            return
    else:
        return

    status_msg = await message.reply_text(f"מוריד את הקובץ: {file_name}...")
    try:
        tg_file = await file_obj.get_file()
        run_storage_maintenance()
        job = job_manager.create_job(
            title=Path(file_name).stem,
            input_type=input_type,
            has_video=(input_type == "video_file"),
            chat_id=message.chat_id,
            user_id=update.effective_user.id,
        )
        ext = Path(file_name).suffix or (".mp3" if input_type == "audio_file" else ".mp4")
        local_path = str(job.job_dir / f"uploaded{ext}")
        await tg_file.download_to_drive(local_path)
        context.user_data["uploaded_job"] = job
        context.user_data["chosen"] = {
            "title": Path(file_name).stem,
            "input_type": input_type,
            "local_path": local_path,
        }
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("אקורדים + מילים", callback_data="format:chords")],
                [InlineKeyboardButton("קריוקי וידיאו", callback_data="format:hebrew_karaoke")],
            ]
        )
        await status_msg.edit_text(f"הקובץ נטען: {file_name}\nמה לעשות?", reply_markup=keyboard)
    except Exception as exc:
        logger.error("File upload error: %s\n%s", exc, traceback.format_exc())
        await status_msg.edit_text(build_unexpected_error(str(exc)) + COMMANDS_TEXT)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    context.user_data["active_user_id"] = update.effective_user.id
    data = query.data
    if data.startswith("group_claim:"):
        token = data.split(":", 1)[1]
        if query.message is None:
            await query.answer("אי אפשר להמשיך מההודעה הזאת כרגע.", show_alert=True)
            return
        request, error = job_manager.bind_group_request_user(token, update.effective_user.id)
        if error == "missing" or request is None:
            await query.answer("הבקשה הזאת כבר לא זמינה.", show_alert=True)
            return
        if error == "forbidden":
            await query.answer("הבקשה הזאת כבר שויכה למשתמש אחר.", show_alert=True)
            return
        me = await context.bot.get_me()
        deep_link = f"https://t.me/{me.username}?start=group_{token}"
        await query.answer("אישרתי שזה אתה. עכשיו אפשר להמשיך בפרטי.", show_alert=True)
        await edit_or_reply(
            query.message,
            "זיהיתי שזה אתה. אפשר להמשיך עכשיו בצ'אט הפרטי עם הבוט, "
            "ורק התוצאה הסופית תחזור לקבוצה.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("המשך בפרטי", url=deep_link)]]),
        )
        return
    protected_job = None
    protected_job_id = callback_job_id(data)
    if protected_job_id:
        try:
            protected_job = job_manager.load_job(protected_job_id)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            # The job folder was cleaned up (or its manifest is corrupt) —
            # e.g. the button belongs to a project the daily reset removed.
            await query.answer("הפרויקט הזה כבר נמחק. שלח את השיר מחדש כדי להתחיל.", show_alert=True)
            return
        if not user_owns_job(protected_job, update.effective_user.id):
            await query.answer("רק המשתמש שפתח את הבקשה יכול להמשיך.", show_alert=True)
            return
    if data.startswith("delivery_approve:"):
        if query.message is None or protected_job is None:
            await query.answer("אי אפשר להשלים את הפעולה כרגע.", show_alert=True)
            return
        await query.answer("מפרסם לקבוצה...")
        await edit_or_reply(query.message, "מפרסם את התוצאה לקבוצה...")
        try:
            delivered_message = await publish_job_to_group(query.message, protected_job)
        except DeliveryError as exc:
            logger.error("Delivery approval publish error: %s\n%s", exc, traceback.format_exc())
            await edit_or_reply(
                query.message,
                build_error_message(exc, protected_job),
                reply_markup=build_delivery_approval_keyboard(protected_job.job_id),
            )
            return
        except Exception as exc:
            logger.error("Unexpected delivery approval publish error: %s\n%s", exc, traceback.format_exc())
            await edit_or_reply(
                query.message,
                build_unexpected_error(str(exc), protected_job),
                reply_markup=build_delivery_approval_keyboard(protected_job.job_id),
            )
            return

        job_manager.update_review_status(protected_job, ReviewStatus.COMPLETED)
        job_manager.clear_pending_delivery(protected_job)
        context.user_data.pop("delivery_feedback_job_id", None)
        cleanup_delivered_job(protected_job)
        await show_delivery_result(
            query.message,
            protected_job.display_name,
            target_chat_id=protected_job.delivery_chat_id,
            delivered_message=delivered_message,
        )
        return
    if data.startswith("delivery_reject:"):
        if query.message is None or protected_job is None:
            await query.answer("אי אפשר להשלים את הפעולה כרגע.", show_alert=True)
            return
        await query.answer("לא אפרסם לקבוצה עד שתשלח מה צריך לתקן.")
        context.user_data["delivery_feedback_job_id"] = protected_job.job_id
        job_manager.update_pending_delivery(
            protected_job,
            status="awaiting_feedback",
            rejected_at=job_manager._now_iso(),
            preview_chat_id=query.message.chat_id,
        )
        await send_delivery_feedback_template(query.message, protected_job)
        await edit_or_reply(query.message, build_delivery_feedback_prompt(protected_job))
        return
    await query.answer()

    if data.startswith("select:"):
        index = int(data.split(":")[1])
        results = context.user_data.get("search_results", [])
        if index >= len(results):
            await edit_or_reply(query.message, "הבחירה כבר לא זמינה.")
            return
        chosen = results[index]
        context.user_data["chosen"] = chosen
        await edit_or_reply(
            query.message,
            f"נבחר: {chosen['title'][:60]}\n\nבאיזה פורמט לעבוד?",
            reply_markup=build_format_keyboard(),
        )
        return

    if data.startswith("format:"):
        parts = data.split(":")
        fmt = parts[1]
        chosen = context.user_data.get("chosen")
        if not chosen:
            await edit_or_reply(query.message, "לא נמצא פריט נבחר.")
            return
        delivery_chat_id, delivery_reply_to_message_id = get_delivery_target(chosen, query.message.chat_id)

        if fmt == "hebrew_karaoke":
            if chosen.get("url"):
                existing_job = job_manager.find_latest_reusable_job(
                    source_url=chosen.get("url", ""),
                    input_type="youtube",
                    user_id=context.user_data["active_user_id"],
                )
                if existing_job is not None:
                    job_manager.update_job_delivery(
                        existing_job,
                        delivery_chat_id=delivery_chat_id,
                        delivery_reply_to_message_id=delivery_reply_to_message_id,
                    )
                    await edit_or_reply(
                        query.message,
                        f"נמצא עיבוד קיים עבור:\n{existing_job.display_name}\n\nאפשר להשתמש במה שכבר הוכן, לרנדר מחדש, או להתחיל מחדש.",
                        reply_markup=build_existing_job_keyboard(existing_job),
                    )
                    return
            await edit_or_reply(query.message, "מתחיל עיבוד קריוקי עברי...")
            async with heavy_job_slot(query.message):
                await run_karaoke_until_review(query.message, context)
            return

        if fmt == "chords":
            await edit_or_reply(query.message, "מכין אקורדים + מילים...")
            async with heavy_job_slot(query.message):
                await generate_direct_chords_output(query.message, context)
            return

        url = chosen.get("url", "")
        bot = query.message.get_bot()
        try:
            if fmt == "mp3":
                await edit_or_reply(query.message, "מוריד MP3...")
                mp3_path, title = await run_heavy_in_executor(query.message, download_audio, url)
                if requires_group_delivery_approval(query.message.chat_id, delivery_chat_id):
                    run_storage_maintenance()
                    legacy_job = job_manager.create_job(
                        title=chosen.get("title", title or "audio"),
                        source_url=url,
                        input_type="youtube",
                        chat_id=query.message.chat_id,
                        user_id=context.user_data["active_user_id"],
                        delivery_chat_id=delivery_chat_id,
                        delivery_reply_to_message_id=delivery_reply_to_message_id,
                    )
                    artifact_path = persist_legacy_artifact(legacy_job, mp3_path, media_type="audio")
                    artifact_filename, artifact_caption = build_legacy_audio_delivery_metadata(legacy_job)
                    job_manager.update_status(legacy_job, JobStatus.DONE)
                    job_manager.update_review_status(legacy_job, ReviewStatus.APPROVED)
                    job_manager.update_pending_delivery(
                        legacy_job,
                        artifact_path=artifact_path.name,
                        artifact_media_type="audio",
                        artifact_filename=artifact_filename,
                        artifact_caption=artifact_caption,
                    )
                    await send_legacy_delivery_artifact(
                        bot,
                        artifact_path=artifact_path,
                        media_type="audio",
                        chat_id=query.message.chat_id,
                        filename=artifact_filename,
                        caption=f"בדיקה לפני פרסום לקבוצה: {artifact_caption}",
                    )
                    await prompt_group_delivery_approval(query.message, legacy_job, delivery_mode="legacy_media")
                else:
                    with open(mp3_path, "rb") as file_handle:
                        delivered_message = await send_audio_to_chat(
                            bot,
                            chat_id=delivery_chat_id,
                            audio=file_handle,
                            filename=os.path.basename(mp3_path),
                            caption="הנה הקובץ שלך." + COMMANDS_TEXT,
                            reply_to_message_id=delivery_reply_to_message_id,
                        )
                    await show_delivery_result(
                        query.message,
                        chosen.get("title", "הבקשה"),
                        target_chat_id=delivery_chat_id,
                        delivered_message=delivered_message,
                    )
            elif fmt == "karaoke":
                await edit_or_reply(query.message, "מכין פלייבק ללא ווקאל...")
                karaoke_path, title = await run_heavy_in_executor(query.message, download_audio_karaoke, url)
                if requires_group_delivery_approval(query.message.chat_id, delivery_chat_id):
                    run_storage_maintenance()
                    legacy_job = job_manager.create_job(
                        title=chosen.get("title", title or "karaoke"),
                        source_url=url,
                        input_type="youtube",
                        chat_id=query.message.chat_id,
                        user_id=context.user_data["active_user_id"],
                        delivery_chat_id=delivery_chat_id,
                        delivery_reply_to_message_id=delivery_reply_to_message_id,
                    )
                    artifact_path = persist_legacy_artifact(legacy_job, karaoke_path, media_type="audio")
                    artifact_filename, artifact_caption = build_legacy_audio_delivery_metadata(legacy_job, karaoke=True)
                    job_manager.update_status(legacy_job, JobStatus.DONE)
                    job_manager.update_review_status(legacy_job, ReviewStatus.APPROVED)
                    job_manager.update_pending_delivery(
                        legacy_job,
                        artifact_path=artifact_path.name,
                        artifact_media_type="audio",
                        artifact_filename=artifact_filename,
                        artifact_caption=artifact_caption,
                    )
                    await send_legacy_delivery_artifact(
                        bot,
                        artifact_path=artifact_path,
                        media_type="audio",
                        chat_id=query.message.chat_id,
                        filename=artifact_filename,
                        caption=f"בדיקה לפני פרסום לקבוצה: {artifact_caption}",
                    )
                    await prompt_group_delivery_approval(query.message, legacy_job, delivery_mode="legacy_media")
                else:
                    with open(karaoke_path, "rb") as file_handle:
                        delivered_message = await send_audio_to_chat(
                            bot,
                            chat_id=delivery_chat_id,
                            audio=file_handle,
                            filename=os.path.basename(karaoke_path),
                            caption=f"קריוקי ללא ווקאל: {title[:50]}" + COMMANDS_TEXT,
                            reply_to_message_id=delivery_reply_to_message_id,
                        )
                    await show_delivery_result(
                        query.message,
                        chosen.get("title", title),
                        target_chat_id=delivery_chat_id,
                        delivered_message=delivered_message,
                    )
            elif fmt == "video":
                quality = parts[2] if len(parts) > 2 else "best"
                await edit_or_reply(query.message, f"מוריד וידאו {quality}...")
                video_path, title = await run_heavy_in_executor(query.message, download_video, url, quality)
                if requires_group_delivery_approval(query.message.chat_id, delivery_chat_id):
                    run_storage_maintenance()
                    legacy_job = job_manager.create_job(
                        title=chosen.get("title", title or "video"),
                        source_url=url,
                        input_type="youtube",
                        has_video=True,
                        chat_id=query.message.chat_id,
                        user_id=context.user_data["active_user_id"],
                        delivery_chat_id=delivery_chat_id,
                        delivery_reply_to_message_id=delivery_reply_to_message_id,
                    )
                    artifact_path = persist_legacy_artifact(legacy_job, video_path, media_type="video")
                    job_manager.update_status(legacy_job, JobStatus.DONE)
                    job_manager.update_review_status(legacy_job, ReviewStatus.APPROVED)
                    job_manager.update_pending_delivery(
                        legacy_job,
                        artifact_path=artifact_path.name,
                        artifact_media_type="video",
                        artifact_filename=os.path.basename(artifact_path),
                        artifact_caption=f"וידאו {quality}",
                    )
                    await send_legacy_delivery_artifact(
                        bot,
                        artifact_path=artifact_path,
                        media_type="video",
                        chat_id=query.message.chat_id,
                        filename=os.path.basename(artifact_path),
                        caption=f"בדיקה לפני פרסום לקבוצה: וידאו {quality}",
                    )
                    await prompt_group_delivery_approval(query.message, legacy_job, delivery_mode="legacy_media")
                else:
                    with open(video_path, "rb") as file_handle:
                        delivered_message = await send_video_to_chat(
                            bot,
                            chat_id=delivery_chat_id,
                            video=file_handle,
                            filename=os.path.basename(video_path),
                            caption=f"וידאו {quality}" + COMMANDS_TEXT,
                            reply_to_message_id=delivery_reply_to_message_id,
                            read_timeout=600,
                            write_timeout=600,
                            connect_timeout=60,
                            supports_streaming=True,
                        )
                    await show_delivery_result(
                        query.message,
                        chosen.get("title", "וידאו"),
                        target_chat_id=delivery_chat_id,
                        delivered_message=delivered_message,
                    )
        except PipelineError as exc:
            logger.error("Legacy media pipeline error: %s\n%s", exc, traceback.format_exc())
            await edit_or_reply(query.message, build_error_message(exc) + COMMANDS_TEXT)
        except Exception as exc:
            logger.error("Legacy media error: %s\n%s", exc, traceback.format_exc())
            await edit_or_reply(query.message, build_unexpected_error(str(exc)) + COMMANDS_TEXT)
        return

    if data.startswith("karaoke_page:"):
        _, job_id, _page = data.split(":")
        await show_review_text(query.message, job_manager.load_job(job_id))
        return

    if data.startswith("karaoke_review:"):
        _, job_id = data.split(":")
        await show_review_text(query.message, job_manager.load_job(job_id))
        return

    if data.startswith("karaoke_existing:"):
        _, job_id, action = data.split(":")
        job = job_manager.load_job(job_id)
        if action == "reuse":
            if job.timings_path.exists():
                context.user_data[f"output_mode:{job.job_id}"] = "reuse"
                await edit_or_reply(
                    query.message,
                    f"נשתמש בתוצרים הקיימים עבור:\n{job.display_name}\n\nמה לייצר עכשיו?",
                    reply_markup=build_output_keyboard(job.job_id),
                )
                return

            context.user_data["reuse_job_id"] = job.job_id
            await edit_or_reply(query.message, f"ממשיך מהעיבוד הקיים עבור:\n{job.display_name}")
            async with heavy_job_slot(query.message):
                await run_karaoke_until_review(query.message, context)
            return

        if action == "rerender":
            if not job_manager.can_rerender(job):
                await edit_or_reply(query.message, "אין טיימינגים שמורים שאפשר לרנדר מהם מחדש.")
                return
            context.user_data[f"output_mode:{job.job_id}"] = "rerender"
            context.user_data.pop("reuse_job_id", None)
            if not job.review_timings_path.exists():
                best_segments = job_manager.get_best_available_segments(job)
                if best_segments:
                    job_manager.save_review_transcript(job, best_segments)
            job_manager.update_review_status(job, ReviewStatus.AWAITING_REVIEW)
            job_manager.set_active_review_job(query.message.chat_id, context.user_data["active_user_id"], job.job_id)
            await show_review_text(
                query.message,
                job,
                note="רינדור מחדש: בודקים ועורכים את המילים לפני היצוא.",
            )
            return

        context.user_data.pop(f"output_mode:{job.job_id}", None)
        context.user_data.pop("reuse_job_id", None)
        await edit_or_reply(query.message, "מתחיל עיבוד חדש...")
        async with heavy_job_slot(query.message):
            await run_karaoke_until_review(query.message, context)
        return

    if data.startswith("karaoke_option:"):
        _, job_id, option_id = data.split(":", 2)
        job = job_manager.load_job(job_id)
        try:
            option = job_manager.apply_lyrics_option(job, option_id)
        except ValueError as exc:
            await edit_or_reply(query.message, f"לא הצלחתי לבחור את הגרסה המבוקשת.\n\n{exc}")
            return
        label = str(option.get("label") or option_id)
        await show_review_text(query.message, job, note=f"נבחרה גרסת הטקסט: {label}")
        return

    if data.startswith("karaoke_edit:"):
        _, job_id = data.split(":")
        job = job_manager.load_job(job_id)
        await edit_or_reply(
            query.message,
            f"עריכת משימה: {job.display_name}\n\n"
            "לתיקון שורה: 3: הטקסט המתוקן\n"
            "להחלפת הכל: שלח את כל הטקסט מחדש\n"
            "אפשר גם להעלות transcript.txt",
            reply_markup=build_edit_keyboard(job_id),
        )
        return

    if data.startswith("karaoke_approve:"):
        _, job_id = data.split(":")
        job = job_manager.load_job(job_id)
        if job_manager.is_reference_selection_active(job) and job_manager.get_selectable_lyrics_options(job):
            await show_review_text(
                query.message,
                job,
                note="התמלול המקורי מיועד להשוואה בלבד. כדי להמשיך, בחר גרסה מתוקנת/חיצונית או ערוך ידנית.",
            )
            return
        if context.user_data.get(f"output_mode:{job_id}") == "rerender":
            job_manager.update_review_status(job, ReviewStatus.APPROVED)
            await edit_or_reply(query.message, "מה לייצר עכשיו?", reply_markup=build_output_keyboard(job_id))
            return
        context.user_data[f"output_mode:{job_id}"] = "fresh"
        job_manager.update_review_status(job, ReviewStatus.APPROVED)
        await edit_or_reply(query.message, "מה לייצר עכשיו?", reply_markup=build_output_keyboard(job_id))
        return

    if data.startswith("karaoke_back_outputs:"):
        _, job_id = data.split(":")
        context.user_data.setdefault(f"output_mode:{job_id}", "fresh")
        await edit_or_reply(query.message, "מה לייצר עכשיו?", reply_markup=build_output_keyboard(job_id))
        return

    if data.startswith("karaoke_output:"):
        _, job_id, choice = data.split(":")
        job = job_manager.load_job(job_id)
        if choice == "chords_text":
            context.user_data[f"delivery_mode:{job_id}"] = "chords_text"
            async with heavy_job_slot(query.message):
                await generate_karaoke_output(query, context, job, None)
        elif choice == "subs_only":
            context.user_data[f"delivery_mode:{job_id}"] = "default"
            async with heavy_job_slot(query.message):
                await generate_karaoke_output(query, context, job, None)
        else:
            context.user_data[f"delivery_mode:{job_id}"] = "default"
            if choice == "video_vocals":
                video_req = VideoRequest(with_vocals=True, without_vocals=False)
            elif choice == "video_instrumental":
                video_req = VideoRequest(with_vocals=False, without_vocals=True)
            else:
                video_req = VideoRequest(with_vocals=True, without_vocals=True)
            context.user_data[f"video_request:{job_id}"] = video_req
            await edit_or_reply(query.message, "באיזו איכות לייצר את הווידאו?", reply_markup=build_quality_keyboard(job_id))
        return

    if data.startswith("karaoke_quality:"):
        _, job_id, quality = data.split(":")
        video_req = context.user_data.get(f"video_request:{job_id}") or VideoRequest(with_vocals=True, without_vocals=True)
        video_req.quality = quality
        async with heavy_job_slot(query.message):
            await generate_karaoke_output(query, context, job_manager.load_job(job_id), video_req)


def group_request_needs_manual_claim(message) -> bool:
    return bool(message and getattr(message, "sender_chat", None))


def build_group_private_continue_keyboard(deep_link: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("המשך בפרטי", url=deep_link)]])


def build_group_claim_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("אני ביקשתי", callback_data=f"group_claim:{token}")]])


async def handoff_group_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    request_kind: str,
    request_payload: dict[str, object],
):
    message = update.message
    if message is None or update.effective_chat is None:
        return

    owner_user_id = int(getattr(update.effective_user, "id", 0) or 0)
    requires_manual_claim = group_request_needs_manual_claim(message)
    stored_user_id = 0 if requires_manual_claim else owner_user_id
    if stored_user_id == 0 and not requires_manual_claim:
        await message.reply_text("לא הצלחתי לזהות מי ביקש את העיבוד. נסה שוב מהחשבון האישי שלך.")
        return

    run_storage_maintenance()
    token = job_manager.create_group_request(
        group_chat_id=update.effective_chat.id,
        group_message_id=message.message_id,
        user_id=stored_user_id,
        request_kind=request_kind,
        payload=request_payload,
    )
    me = await context.bot.get_me()
    deep_link = f"https://t.me/{me.username}?start=group_{token}"

    if requires_manual_claim:
        await message.reply_text(
            "זיהיתי שההודעה נשלחה בשם הקבוצה או כאדמין אנונימי, ולכן טלגרם לא מוסר לי את המשתמש האמיתי.\n"
            "לחץ על 'אני ביקשתי', ואז אעביר אותך לפרטי להמשך.",
            reply_markup=build_group_claim_keyboard(token),
        )
        return

    await message.reply_text(
        "כדי שלא להעמיס על הקבוצה, המשך את הבקשה בצ'אט פרטי עם הבוט. "
        "רק אתה תראה את כל השלבים, ואת התוצאה הסופית אשלח כאן לקבוצה.",
        reply_markup=build_group_private_continue_keyboard(deep_link),
    )


async def begin_song_request(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    delivery_context: dict[str, int] | None = None,
):
    normalized = text.strip()
    if is_youtube_url(normalized):
        context.user_data["chosen"] = apply_delivery_context({"url": normalized, "title": normalized}, delivery_context)
        await message.reply_text("זוהה קישור מיוטיוב. באיזה פורמט לעבד?", reply_markup=build_format_keyboard())
        return

    status_msg = await message.reply_text(f"מחפש: {normalized}...")
    try:
        results = await asyncio.get_running_loop().run_in_executor(None, search_youtube, normalized)
        if not results:
            await status_msg.edit_text("לא נמצאו תוצאות." + COMMANDS_TEXT)
            return
        context.user_data["search_results"] = [apply_delivery_context(result, delivery_context) for result in results]
        await status_msg.delete()
        await message.reply_text(f"תוצאות עבור: {normalized}")
        for index, result in enumerate(results):
            caption = f"{index + 1}. {result['title'][:60]}\n{result['channel']}\n{result['duration']}"
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("בחר", callback_data=f"select:{index}")]])
            try:
                await message.reply_photo(result["thumbnail"], caption=caption, reply_markup=keyboard)
            except Exception:
                await message.reply_text(caption, reply_markup=keyboard)
    except Exception as exc:
        logger.error("Search error: %s", exc)
        await status_msg.edit_text(build_unexpected_error(str(exc)) + COMMANDS_TEXT)


async def prepare_uploaded_choice_from_payload(
    message,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    user_id: int,
    upload_payload: dict[str, object],
    delivery_context: dict[str, int] | None = None,
):
    file_name = str(upload_payload.get("file_name") or "file")
    input_type = str(upload_payload.get("input_type") or "audio_file")
    file_id = str(upload_payload.get("file_id") or "")
    if not file_id:
        await message.reply_text("לא נמצא קובץ תקין לעיבוד." + COMMANDS_TEXT)
        return

    status_msg = await message.reply_text(f"מוריד את הקובץ: {file_name}...")
    try:
        tg_file = await context.bot.get_file(file_id)
        run_storage_maintenance()
        delivery_chat_id, delivery_reply_to_message_id = get_delivery_target(delivery_context, message.chat_id)
        job = job_manager.create_job(
            title=Path(file_name).stem,
            input_type=input_type,
            has_video=(input_type == "video_file"),
            chat_id=message.chat_id,
            user_id=user_id,
            delivery_chat_id=delivery_chat_id,
            delivery_reply_to_message_id=delivery_reply_to_message_id,
        )
        ext = Path(file_name).suffix or (".mp3" if input_type == "audio_file" else ".mp4")
        local_path = str(job.job_dir / f"uploaded{ext}")
        await tg_file.download_to_drive(local_path)
        context.user_data["uploaded_job"] = job
        context.user_data["chosen"] = apply_delivery_context(
            {
                "title": Path(file_name).stem,
                "input_type": input_type,
                "local_path": local_path,
            },
            delivery_context,
        )
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("אקורדים + מילים", callback_data="format:chords")],
                [InlineKeyboardButton("קריוקי וידאו", callback_data="format:hebrew_karaoke")],
            ]
        )
        await status_msg.edit_text(f"הקובץ נטען: {file_name}\nמה לעשות?", reply_markup=keyboard)
    except Exception as exc:
        logger.error("File upload error: %s\n%s", exc, traceback.format_exc())
        await status_msg.edit_text(build_unexpected_error(str(exc)) + COMMANDS_TEXT)


async def resume_group_request_from_start(update: Update, context: ContextTypes.DEFAULT_TYPE, token: str) -> bool:
    if update.effective_user is None or update.message is None:
        return False

    request, error = job_manager.claim_group_request(token, update.effective_user.id)
    if error == "forbidden":
        await update.message.reply_text("הקישור הזה שייך למי שביקש את העיבוד בקבוצה.")
        return True
    if error == "unclaimed":
        await update.message.reply_text(
            "כדי לאמת שזו באמת הבקשה שלך, חזור לקבוצה ולחץ על 'אני ביקשתי', ואז פתח שוב את הקישור."
        )
        return True
    if error == "missing" or request is None:
        await update.message.reply_text("הבקשה הזאת כבר נוצלה או פגה.")
        return True

    delivery_context = build_delivery_context(
        int(request.get("group_chat_id", 0) or 0),
        int(request.get("group_message_id", 0) or 0),
    )
    payload = request.get("payload") if isinstance(request.get("payload"), dict) else {}
    request_kind = str(request.get("request_kind") or "")
    if request_kind == "text":
        request_text = str(payload.get("text") or "").strip()
        if not request_text:
            await update.message.reply_text("לא נמצא טקסט להמשך.")
            return True
        await begin_song_request(update.message, context, request_text, delivery_context=delivery_context)
        return True
    if request_kind == "upload":
        await prepare_uploaded_choice_from_payload(
            update.message,
            context,
            user_id=update.effective_user.id,
            upload_payload=payload,
            delivery_context=delivery_context,
        )
        return True

    await update.message.reply_text("לא הצלחתי לשחזר את הבקשה הזאת.")
    return True


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable.")

    try:
        ensure_single_instance()
    except RuntimeError as exc:
        logger.error("%s", exc)
        print("כבר רץ מופע אחר של הבוט בפרויקט הזה.")
        return

    run_storage_maintenance()
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        # Global request timeouts: large karaoke MP3/MP4 uploads exceed the
        # ~20s library default and used to die with httpx.WriteTimeout.
        # get_updates_* (long polling) intentionally keeps its own defaults.
        .connect_timeout(30)
        .read_timeout(60)
        .write_timeout(120)
        .pool_timeout(30)
        .media_write_timeout(600)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(
        MessageHandler(
            filters.AUDIO | filters.VIDEO | filters.VOICE | filters.VIDEO_NOTE | filters.Document.ALL,
            handle_file_upload,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_app_error)

    print("הבוט פועל.")
    app.run_polling()


if __name__ == "__main__":
    main()
