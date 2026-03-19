import asyncio
import html
import logging
import os
import re
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from karaoke import job_manager
from karaoke.config import BASE_DIR, TELEGRAM_BOT_TOKEN
from karaoke.error_formatter import format_pipeline_error, format_unexpected_error
from karaoke.exceptions import DeliveryError, PipelineError
from karaoke.legacy_media import download_audio, download_audio_karaoke, download_video, safe_filename, search_youtube
from karaoke.models import Job, JobStatus, ReviewStatus, STATUS_MESSAGES, VideoRequest
from karaoke.pipeline import KaraokePipeline

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "bot.log"

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
file_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger(__name__)

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


def build_format_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("MP3", callback_data="format:mp3"),
                InlineKeyboardButton("קריוקי ללא ווקאל", callback_data="format:karaoke"),
            ],
            [InlineKeyboardButton("אקורדים + מילים", callback_data="format:chords")],
            [InlineKeyboardButton("קריוקי עברי מלא", callback_data="format:hebrew_karaoke")],
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


def build_dynamic_review_keyboard(job: Job) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton("אשר", callback_data=f"karaoke_approve:{job.job_id}"),
            InlineKeyboardButton("ערוך", callback_data=f"karaoke_edit:{job.job_id}"),
        ]
    ]
    selected_option_id = job_manager.get_selected_lyrics_option_id(job)
    option_buttons = [
        InlineKeyboardButton(
            _option_button_label(option, selected_option_id),
            callback_data=f"karaoke_option:{job.job_id}:{option['option_id']}",
        )
        for option in job_manager.get_lyrics_options(job)[:5]
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


def _build_review_text(job: Job, note: str | None = None) -> str:
    display_text = job_manager.get_display_text(job_manager.load_review_segments(job))
    verification = job.manifest.lyrics_verification or {}
    selected_option_id = job_manager.get_selected_lyrics_option_id(job)

    blocks = [f"משימה: {job.display_name}"]
    if note:
        blocks.append(note)

    if verification:
        summary = str(verification.get("summary", "")).strip()
        confidence = float(verification.get("confidence", 0.0) or 0.0)
        correction_count = int(verification.get("correction_count", 0) or 0)
        applied = bool(verification.get("applied", False))
        if summary:
            blocks.append(f"אימות לפני review: {summary}")
        if confidence:
            blocks.append(f"ציון התאמה לרשת: {confidence:.2f}")
        if applied and correction_count:
            blocks.append(f"תוקנו אוטומטית {correction_count} מילים לפני ההצגה לבדיקה.")
        sources = verification.get("matched_sources") or []
        if sources:
            blocks.append(f"מקור בדיקה ראשון: {sources[0]}")

    options = job_manager.get_lyrics_options(job)
    if options:
        option_lines = []
        for option in options[:5]:
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
    del context
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
    for key in ["chosen", "uploaded_job", "search_results"]:
        context.user_data.pop(key, None)
    await update.message.reply_text("העיבוד הפעיל הופסק." + COMMANDS_TEXT)


async def send_output_files(query, job: Job, output_files: dict[str, Path]):
    for file_name, file_path in output_files.items():
        try:
            if file_name.endswith(".mp4"):
                with open(file_path, "rb") as file_handle:
                    await query.message.reply_video(
                        video=file_handle,
                        filename=f"{safe_filename(job.title)}_{file_name}",
                        caption=file_name,
                        read_timeout=600,
                        write_timeout=600,
                        connect_timeout=60,
                        supports_streaming=True,
                    )
            else:
                with open(file_path, "rb") as file_handle:
                    await query.message.reply_document(
                        document=file_handle,
                        filename=f"{safe_filename(job.title)}_{file_name}",
                        caption=file_name,
                        read_timeout=120,
                        write_timeout=120,
                    )
        except Exception as exc:
            raise DeliveryError(str(exc), f"שליחת {file_name} נכשלה.") from exc


def filter_output_files(output_files: dict[str, Path], delivery_mode: str) -> dict[str, Path]:
    if delivery_mode != "chords_text":
        return output_files

    allowed = {"lyrics_with_chords.txt", "song_analysis.json"}
    filtered = {name: path for name, path in output_files.items() if name in allowed}
    return filtered or output_files


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


async def send_chords_text_response(message, job: Job):
    if not job.lyrics_with_chords_path.exists():
        raise DeliveryError("lyrics_with_chords.txt is missing", "קובץ האקורדים לא נוצר.")

    bpm_text = "לא זוהה"
    if job.song_analysis_path.exists():
        try:
            analysis = job_manager.load_song_analysis(job)
            if analysis.bpm > 0:
                bpm_text = f"{analysis.bpm:.2f}"
        except Exception:
            pass

    chord_text = job.lyrics_with_chords_path.read_text(encoding="utf-8")
    prefixed_text = f"אקורדים + מילים עבור: {job.display_name}\nBPM: {bpm_text}\n\n{chord_text}".strip()
    for chunk in chunk_text_for_telegram(prefixed_text):
        await message.reply_text(f"<pre>{html.escape(chunk)}</pre>", parse_mode="HTML")

    with open(job.lyrics_with_chords_path, "rb") as file_handle:
        await message.reply_document(
            document=file_handle,
            filename=f"{safe_filename(job.title)}_lyrics_with_chords.txt",
            caption="אקורדים + מילים",
            read_timeout=120,
            write_timeout=120,
        )


async def generate_direct_chords_output(message, context: ContextTypes.DEFAULT_TYPE):
    chosen = context.user_data.get("chosen")
    if not chosen:
        await edit_or_reply(message, "לא נמצא קלט לעיבוד.")
        return

    loop = asyncio.get_running_loop()
    existing_job = None
    if chosen.get("url"):
        existing_job = job_manager.find_latest_reusable_job(
            source_url=chosen.get("url", ""),
            input_type="youtube",
            user_id=context.user_data["active_user_id"],
        )

    if existing_job and existing_job.lyrics_with_chords_path.exists():
        await edit_or_reply(message, f"נמצא עיבוד קיים עבור {existing_job.display_name}, שולח אקורדים + מילים...")
        await send_chords_text_response(message, existing_job)
        await edit_or_reply(message, f"הושלם בהצלחה עבור {existing_job.display_name}." + COMMANDS_TEXT)
        return

    if existing_job and job_manager.can_rerender(existing_job):
        pipeline = KaraokePipeline(existing_job)
        try:
            await edit_or_reply(message, "משלים אקורדים + מילים מהחומרים הקיימים...")
            await loop.run_in_executor(None, pipeline.rerender_existing_outputs, None)
            job_manager.update_review_status(existing_job, ReviewStatus.COMPLETED)
            await send_chords_text_response(message, existing_job)
            await edit_or_reply(message, f"הושלם בהצלחה עבור {existing_job.display_name}." + COMMANDS_TEXT)
        except PipelineError as exc:
            logger.error("Direct chord rerender error: %s\n%s", exc, traceback.format_exc())
            job_manager.update_status(existing_job, JobStatus.ERROR, exc.info)
            await edit_or_reply(message, build_error_message(exc, existing_job))
        except Exception as exc:
            logger.error("Unexpected direct chord rerender error: %s\n%s", exc, traceback.format_exc())
            await edit_or_reply(message, build_unexpected_error(str(exc), existing_job))
        return

    if chosen.get("local_path"):
        job = context.user_data.get("uploaded_job")
        if not job:
            job = job_manager.create_job(
                title=chosen.get("title", "input"),
                input_type=chosen.get("input_type", "audio_file"),
                has_video=(chosen.get("input_type") == "video_file"),
                chat_id=message.chat_id,
                user_id=context.user_data["active_user_id"],
            )
    else:
        job = job_manager.create_job(
            title=chosen.get("title", "song"),
            source_url=chosen.get("url", ""),
            input_type="youtube",
            chat_id=message.chat_id,
            user_id=context.user_data["active_user_id"],
        )

    pipeline = KaraokePipeline(job)
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

        await send_chords_text_response(message, job)
        job_manager.update_review_status(job, ReviewStatus.COMPLETED)
        await edit_or_reply(message, f"הושלם בהצלחה עבור {job.display_name}." + COMMANDS_TEXT)
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
    if reuse_job_id:
        job = job_manager.load_job(reuse_job_id)
    elif chosen.get("local_path"):
        job = context.user_data.get("uploaded_job")
        if not job:
            job = job_manager.create_job(
                title=chosen.get("title", "input"),
                input_type=chosen.get("input_type", "audio_file"),
                has_video=(chosen.get("input_type") == "video_file"),
                chat_id=message.chat_id,
                user_id=context.user_data["active_user_id"],
            )
    else:
        job = job_manager.create_job(
            title=chosen.get("title", "song"),
            source_url=chosen.get("url", ""),
            input_type="youtube",
            chat_id=message.chat_id,
            user_id=context.user_data["active_user_id"],
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
            needs_music_outputs = not (job.song_analysis_path.exists() and job.lyrics_with_chords_path.exists())
            needs_requested_video = bool(
                video_request
                and (
                    (video_request.with_vocals and not job.video_vocals_path.exists())
                    or (video_request.without_vocals and not job.video_instrumental_path.exists())
                )
            )
            if needs_subtitles or needs_requested_video or needs_music_outputs:
                output_files = await loop.run_in_executor(None, pipeline.rerender_existing_outputs, video_request)
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
        await send_output_files(query, job, output_files)
        job_manager.update_review_status(job, ReviewStatus.COMPLETED)
        job_manager.clear_active_review_job(query.message.chat_id, context.user_data["active_user_id"])
        context.user_data.pop(f"output_mode:{job.job_id}", None)
        context.user_data.pop(f"delivery_mode:{job.job_id}", None)
        await edit_or_reply(query.message, f"הושלם בהצלחה עבור {job.display_name}." + COMMANDS_TEXT)
    except PipelineError as exc:
        logger.error("Final generation error: %s\n%s", exc, traceback.format_exc())
        job_manager.update_status(job, JobStatus.ERROR, exc.info)
        await edit_or_reply(query.message, build_error_message(exc, job))
    except Exception as exc:
        logger.error("Unexpected final generation error: %s\n%s", exc, traceback.format_exc())
        await edit_or_reply(query.message, build_unexpected_error(str(exc), job))


async def handle_karaoke_correction(update: Update, job: Job, text: str):
    segments = job_manager.load_review_segments(job)
    line_match = re.match(r"^(\d+)\s*:\s*(.+)", text)
    if line_match:
        line_number = int(line_match.group(1))
        corrected_line = line_match.group(2).strip()
        try:
            segments = job_manager.update_transcript_line(segments, line_number, corrected_line)
        except ValueError as exc:
            await update.message.reply_text(f"שגיאה: {exc}")
            return
        note = f"שורה {line_number} עודכנה."
    else:
        segments = job_manager.update_transcript_text(segments, text)
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
    segments = job_manager.update_transcript_text(job_manager.load_review_segments(job), corrected_text)
    job_manager.save_review_transcript(job, segments)
    job_manager.save_manual_review_option(job, segments)
    await show_review_text(update.message, job, note="הטקסט מתוך הקובץ נשמר.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["active_user_id"] = update.effective_user.id
    text = update.message.text.strip()
    review_job = await get_active_review_job(update)
    if review_job and review_job.review_status in {ReviewStatus.AWAITING_REVIEW, ReviewStatus.APPROVED}:
        if looks_like_new_request(text):
            job_manager.clear_active_review_job(update.effective_chat.id, update.effective_user.id)
            await update.message.reply_text("הייתה משימת review פתוחה מהפעם הקודמת, סגרתי אותה ומתחיל בקלט החדש.")
            review_job = None
        else:
            await handle_karaoke_correction(update, review_job, text)
            return

    if is_youtube_url(text):
        context.user_data["chosen"] = {"url": text, "title": text}
        await update.message.reply_text("זוהה קישור מיוטיוב. באיזה פורמט לעבוד?", reply_markup=build_format_keyboard())
        return

    status_msg = await update.message.reply_text(f"מחפש: {text}...")
    try:
        results = await asyncio.get_running_loop().run_in_executor(None, search_youtube, text)
        if not results:
            await status_msg.edit_text("לא נמצאו תוצאות." + COMMANDS_TEXT)
            return
        context.user_data["search_results"] = results
        await status_msg.delete()
        await update.message.reply_text(f"תוצאות עבור: {text}")
        for index, result in enumerate(results):
            caption = f"{index + 1}. {result['title'][:60]}\n{result['channel']}\n{result['duration']}"
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("בחר", callback_data=f"select:{index}")]])
            try:
                await update.message.reply_photo(result["thumbnail"], caption=caption, reply_markup=keyboard)
            except Exception:
                await update.message.reply_text(caption, reply_markup=keyboard)
    except Exception as exc:
        logger.error("Search error: %s", exc)
        await status_msg.edit_text(build_unexpected_error(str(exc)) + COMMANDS_TEXT)


async def handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["active_user_id"] = update.effective_user.id
    review_job = await get_active_review_job(update)
    if review_job and update.message.document:
        file_name = (update.message.document.file_name or "").lower()
        mime = update.message.document.mime_type or ""
        if file_name.endswith(".txt") or mime == "text/plain":
            await handle_review_text_file(update, review_job)
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
                [InlineKeyboardButton("קריוקי עברי מלא", callback_data="format:hebrew_karaoke")],
            ]
        )
        await status_msg.edit_text(f"הקובץ נטען: {file_name}\nמה לעשות?", reply_markup=keyboard)
    except Exception as exc:
        logger.error("File upload error: %s\n%s", exc, traceback.format_exc())
        await status_msg.edit_text(build_unexpected_error(str(exc)) + COMMANDS_TEXT)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["active_user_id"] = update.effective_user.id
    data = query.data

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

        if fmt == "hebrew_karaoke":
            if chosen.get("url"):
                existing_job = job_manager.find_latest_reusable_job(
                    source_url=chosen.get("url", ""),
                    input_type="youtube",
                    user_id=context.user_data["active_user_id"],
                )
                if existing_job is not None:
                    await edit_or_reply(
                        query.message,
                        f"נמצא עיבוד קיים עבור:\n{existing_job.display_name}\n\nאפשר להשתמש במה שכבר הוכן, לרנדר מחדש, או להתחיל מחדש.",
                        reply_markup=build_existing_job_keyboard(existing_job),
                    )
                    return
            await edit_or_reply(query.message, "מתחיל עיבוד קריוקי עברי...")
            await run_karaoke_until_review(query.message, context)
            return

        if fmt == "chords":
            await edit_or_reply(query.message, "מכין אקורדים + מילים...")
            await generate_direct_chords_output(query.message, context)
            return

        url = chosen.get("url", "")
        try:
            if fmt == "mp3":
                await edit_or_reply(query.message, "מוריד MP3...")
                mp3_path, _title = await asyncio.get_running_loop().run_in_executor(None, download_audio, url)
                with open(mp3_path, "rb") as file_handle:
                    await query.message.reply_audio(
                        audio=file_handle,
                        filename=os.path.basename(mp3_path),
                        caption="הנה הקובץ שלך." + COMMANDS_TEXT,
                    )
            elif fmt == "karaoke":
                await edit_or_reply(query.message, "מכין פלייבק ללא ווקאל...")
                karaoke_path, title = await asyncio.get_running_loop().run_in_executor(None, download_audio_karaoke, url)
                with open(karaoke_path, "rb") as file_handle:
                    await query.message.reply_audio(
                        audio=file_handle,
                        filename=os.path.basename(karaoke_path),
                        caption=f"קריוקי ללא ווקאל: {title[:50]}" + COMMANDS_TEXT,
                    )
            elif fmt == "video":
                quality = parts[2] if len(parts) > 2 else "best"
                await edit_or_reply(query.message, f"מוריד וידאו {quality}...")
                video_path, _title = await asyncio.get_running_loop().run_in_executor(None, download_video, url, quality)
                with open(video_path, "rb") as file_handle:
                    await query.message.reply_video(
                        video=file_handle,
                        filename=os.path.basename(video_path),
                        caption=f"וידאו {quality}" + COMMANDS_TEXT,
                        read_timeout=600,
                        write_timeout=600,
                        connect_timeout=60,
                        supports_streaming=True,
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
            await run_karaoke_until_review(query.message, context)
            return

        if action == "rerender":
            if not job_manager.can_rerender(job):
                await edit_or_reply(query.message, "אין טיימינגים שמורים שאפשר לרנדר מהם מחדש.")
                return
            context.user_data[f"output_mode:{job.job_id}"] = "rerender"
            await edit_or_reply(
                query.message,
                f"רינדור מחדש עבור:\n{job.display_name}\n\nמה לייצר עכשיו?",
                reply_markup=build_output_keyboard(job.job_id),
            )
            return

        context.user_data.pop(f"output_mode:{job.job_id}", None)
        context.user_data.pop("reuse_job_id", None)
        await edit_or_reply(query.message, "מתחיל עיבוד חדש...")
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
            await generate_karaoke_output(query, context, job, None)
        elif choice == "subs_only":
            context.user_data[f"delivery_mode:{job_id}"] = "default"
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
        await generate_karaoke_output(query, context, job_manager.load_job(job_id), video_req)


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
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
