"""Lean Telegram bot: YouTube search, chords (Tab4U), MP3/video download.

Flow: user sends a song name (or a YouTube URL) -> 5 search results ->
pick a song -> [chords] / [download] -> [video|mp3] -> quality -> deliver ->
"uploaded successfully" -> local file deleted immediately. Videos over the
50MB Telegram cap are served as a temporary download link instead.
"""

from __future__ import annotations

import asyncio
import html
import io
import logging
import os
import re

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from karaoke import chords, library_sync, media
from karaoke.config import TELEGRAM_FILE_LIMIT_BYTES

logger = logging.getLogger(__name__)

TELEGRAM_TEXT_LIMIT = 4096
YOUTUBE_URL_PATTERN = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})")
QUALITIES = ["best", "1080", "720", "480", "360"]
QUALITY_LABELS = {"best": "הכי טוב", "1080": "1080p", "720": "720p", "480": "480p", "360": "360p"}


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def is_youtube_url(text: str) -> bool:
    return bool(YOUTUBE_URL_PATTERN.search(text or ""))


def _video_id(text: str) -> str | None:
    match = YOUTUBE_URL_PATTERN.search(text or "")
    return match.group(1) if match else None


def format_result(result: dict) -> str:
    parts = [result.get("title", "ללא שם")]
    if result.get("channel"):
        parts.append(result["channel"])
    if result.get("duration"):
        parts.append(result["duration"])
    return " • ".join(parts)


def build_results_keyboard(results: list[dict], user_data: dict) -> InlineKeyboardMarkup:
    songs = user_data.setdefault("songs", {})
    rows = []
    for result in results:
        songs[result["id"]] = {"url": result["url"], "title": result["title"]}
        rows.append([InlineKeyboardButton(format_result(result)[:60], callback_data=f"pick:{result['id']}")])
    return InlineKeyboardMarkup(rows)


def _pick_button(vid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("✅ בחר שיר זה", callback_data=f"pick:{vid}")]])


def build_song_menu(vid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎸 אקורדים", callback_data=f"chords:{vid}")],
        [InlineKeyboardButton("⬇️ הורדת השיר", callback_data=f"dl:{vid}")],
    ])


def build_format_menu(vid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎬 וידאו", callback_data=f"vid:{vid}"),
         InlineKeyboardButton("🎵 MP3", callback_data=f"mp3:{vid}")],
        [InlineKeyboardButton("⬅️ חזרה", callback_data=f"pick:{vid}")],
    ])


def build_quality_menu(vid: str) -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(QUALITY_LABELS[q], callback_data=f"q:{vid}:{q}") for q in QUALITIES]
    return InlineKeyboardMarkup([row[:3], row[3:], [InlineKeyboardButton("⬅️ חזרה", callback_data=f"dl:{vid}")]])


def build_chord_keyboard(vid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("מקורי", callback_data=f"ck:{vid}:original"),
         InlineKeyboardButton("גרסה קלה", callback_data=f"ck:{vid}:easy")],
        [InlineKeyboardButton("⬅️ חזרה", callback_data=f"pick:{vid}")],
    ])


def _get_song(context: ContextTypes.DEFAULT_TYPE, vid: str) -> dict | None:
    return context.user_data.get("songs", {}).get(vid)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "שלח לי שם של שיר (או קישור יוטיוב), ואביא לך אקורדים והורדה 🎸🎵"
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return

    if is_youtube_url(text):
        vid = _video_id(text)
        msg = await update.message.reply_text("⏳ טוען פרטי שיר...")
        try:
            info = await asyncio.to_thread(media._extract_info, text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("URL info failed: %s", exc)
            await msg.edit_text("לא הצלחתי לטעון את השיר.")
            return
        title = str(info.get("title") or vid)
        context.user_data.setdefault("songs", {})[vid] = {"url": text, "title": title}
        await msg.edit_text(f"🎵 {title}", reply_markup=build_song_menu(vid))
        return

    msg = await update.message.reply_text("🔍 מחפש ביוטיוב...")
    try:
        results = await asyncio.to_thread(media.search_youtube, text, 5)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Search failed: %s", exc)
        await msg.edit_text("החיפוש נכשל, נסה שוב.")
        return
    if not results:
        await msg.edit_text("לא נמצאו תוצאות. נסה ניסוח אחר.")
        return
    await msg.edit_text("👇 בחר שיר מהתוצאות:")
    await _send_results(update, context, results)


async def _send_results(update: Update, context: ContextTypes.DEFAULT_TYPE, results: list[dict]) -> None:
    """Send each result as its own thumbnail message with a select button below it."""
    songs = context.user_data.setdefault("songs", {})
    for result in results:
        vid = result["id"]
        songs[vid] = {"url": result["url"], "title": result["title"]}
        caption = format_result(result)
        markup = _pick_button(vid)
        try:
            await update.message.reply_photo(
                photo=result.get("thumbnail"), caption=caption, reply_markup=markup
            )
        except Exception as exc:  # noqa: BLE001 - thumbnail fetch can fail; fall back to text
            logger.warning("Thumbnail send failed for %s: %s", vid, exc)
            await update.message.reply_text(caption, reply_markup=markup)


async def _respond(query, text: str, reply_markup=None) -> None:
    """Edit the message in place, but if it's a photo result (no editable text),
    start a fresh text message instead so the rest of the flow can edit it."""
    if query.message.photo:
        await query.message.reply_text(text, reply_markup=reply_markup)
    else:
        await query.edit_message_text(text, reply_markup=reply_markup)


async def on_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    vid = query.data.split(":", 1)[1]
    song = _get_song(context, vid)
    if not song:
        await _respond(query, "השיר לא זמין יותר, חפש שוב.")
        return
    await _respond(query, f"🎵 {song['title']}", reply_markup=build_song_menu(vid))


async def on_chords(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    vid = query.data.split(":", 1)[1]
    song = _get_song(context, vid)
    if not song:
        await query.edit_message_text("השיר לא זמין יותר, חפש שוב.")
        return
    await query.edit_message_text(f"🎸 מחפש אקורדים ל: {song['title']}...")
    analysis = await asyncio.to_thread(chords.lookup, song["title"])
    if analysis is None:
        await query.edit_message_text(
            f"לא נמצאו אקורדים ל: {song['title']}", reply_markup=build_song_menu(vid)
        )
        return
    context.user_data.setdefault("chords", {})[vid] = analysis

    # Library sync — only when chords were found. Store a clean singer + song
    # name (not the raw YouTube upload title with its "(Official Video)" clutter).
    artist, song_name = media.split_artist_and_title(song["title"])
    sheet = getattr(analysis, "parsed_sheet", None)
    content = library_sync.to_inline_chords(sheet) if sheet else analysis.chord_sheet_text
    asyncio.create_task(
        library_sync.upsert_song(
            song_name or song["title"], artist, analysis.original_key, content
        )
    )

    await _deliver_chords(query, context, vid, "original")


async def on_chord_key(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    _, vid, mode = query.data.split(":")
    if vid not in context.user_data.get("chords", {}):
        await query.edit_message_text("האקורדים לא זמינים יותר, בחר שיר שוב.")
        return
    await _deliver_chords(query, context, vid, mode)


def _rtl_pre(text: str) -> str:
    """Wrap chords-above-lyrics text for Telegram: monospace (<pre>) so the space
    padding that lines up chords over their words actually renders at a fixed
    width, plus a Right-to-Left Mark on every line so Telegram's bidi handling
    keeps the Latin chord line anchored the same way as the Hebrew lyric line
    below it (otherwise the two lines default to opposite text directions and
    the columns drift apart)."""
    rlm = "‏"  # RIGHT-TO-LEFT MARK
    marked = "\n".join(f"{rlm}{line}" if line else line for line in text.split("\n"))
    return f"<pre>{html.escape(marked, quote=False)}</pre>"


async def _deliver_chords(query, context, vid: str, mode: str) -> None:
    song = _get_song(context, vid)
    analysis = context.user_data["chords"][vid]
    plain_text = chords.render(analysis, song["title"], mode)
    if len(plain_text) <= TELEGRAM_TEXT_LIMIT - 100:
        try:
            await query.edit_message_text(
                _rtl_pre(plain_text), parse_mode=ParseMode.HTML,
                reply_markup=build_chord_keyboard(vid),
            )
            return
        except BadRequest as exc:
            if "not modified" in str(exc).lower():
                return
    # Too long for a message — send as a file, keep the menu on the original message.
    buffer = io.BytesIO(plain_text.encode("utf-8"))
    buffer.name = f"{song['title']}.txt"
    await context.bot.send_document(chat_id=query.message.chat_id, document=buffer)
    await query.edit_message_text(f"🎸 {song['title']}", reply_markup=build_song_menu(vid))


async def on_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    vid = query.data.split(":", 1)[1]
    if not _get_song(context, vid):
        await query.edit_message_text("השיר לא זמין יותר, חפש שוב.")
        return
    await query.edit_message_text("בחר פורמט:", reply_markup=build_format_menu(vid))


async def on_video_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    vid = query.data.split(":", 1)[1]
    await query.edit_message_text("בחר איכות:", reply_markup=build_quality_menu(vid))


async def on_mp3(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    vid = query.data.split(":", 1)[1]
    song = _get_song(context, vid)
    if not song:
        await query.edit_message_text("השיר לא זמין יותר, חפש שוב.")
        return
    await query.edit_message_text(f"⏳ מוריד MP3: {song['title']}...")
    try:
        path, title = await asyncio.to_thread(media.download_audio, song["url"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("MP3 download failed: %s", exc)
        await query.edit_message_text("ההורדה נכשלה, נסה שוב.", reply_markup=build_song_menu(vid))
        return
    try:
        with open(path, "rb") as handle:
            await context.bot.send_audio(chat_id=query.message.chat_id, audio=handle, title=title)
        await context.bot.send_message(chat_id=query.message.chat_id, text="✅ הועלה בהצלחה")
    finally:
        _safe_remove(path)
    await query.edit_message_text(f"🎵 {song['title']}", reply_markup=build_song_menu(vid))


async def on_quality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    _, vid, quality = query.data.split(":")
    song = _get_song(context, vid)
    if not song:
        await query.edit_message_text("השיר לא זמין יותר, חפש שוב.")
        return
    await query.edit_message_text(
        f"⏳ מוריד וידאו {QUALITY_LABELS.get(quality, quality)}: {song['title']}..."
    )
    try:
        path, _title = await asyncio.to_thread(media.download_video, song["url"], quality)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Video download failed: %s", exc)
        await query.edit_message_text("ההורדה נכשלה, נסה שוב.", reply_markup=build_song_menu(vid))
        return

    chat_id = query.message.chat_id
    if os.path.getsize(path) <= TELEGRAM_FILE_LIMIT_BYTES:
        try:
            with open(path, "rb") as handle:
                await context.bot.send_video(chat_id=chat_id, video=handle, supports_streaming=True)
            await context.bot.send_message(chat_id=chat_id, text="✅ הועלה בהצלחה")
        finally:
            _safe_remove(path)
    else:
        # File stays on disk to be served by the link; a periodic sweep (in the
        # entrypoint) removes it after it expires.
        link_builder = context.bot_data.get("link_builder")
        link = link_builder(path) if link_builder else path
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"הקובץ גדול מ-50MB. קישור הורדה (בתוקף לשעתיים):\n{link}",
        )
        await context.bot.send_message(chat_id=chat_id, text="✅ הועלה בהצלחה")
    await query.edit_message_text(f"🎵 {song['title']}", reply_markup=build_song_menu(vid))


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
    # Downloads live in a unique per-job subdir; drop it once emptied.
    try:
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.warning("Handler error: %s", context.error)
    query = getattr(update, "callback_query", None)
    try:
        if query is not None:
            await query.answer("קרתה שגיאה, נסה שוב.", show_alert=False)
    except Exception:  # noqa: BLE001 - never let the error handler raise
        pass


def register_handlers(application: Application, link_builder=None) -> None:
    """link_builder(path)->url makes a large local file downloadable and returns its URL."""
    application.bot_data["link_builder"] = link_builder
    application.add_error_handler(on_error)
    application.add_handler(CommandHandler("start", on_start))
    application.add_handler(CallbackQueryHandler(on_pick, pattern=r"^pick:"))
    application.add_handler(CallbackQueryHandler(on_chords, pattern=r"^chords:"))
    application.add_handler(CallbackQueryHandler(on_chord_key, pattern=r"^ck:"))
    application.add_handler(CallbackQueryHandler(on_download, pattern=r"^dl:"))
    application.add_handler(CallbackQueryHandler(on_video_menu, pattern=r"^vid:"))
    application.add_handler(CallbackQueryHandler(on_mp3, pattern=r"^mp3:"))
    application.add_handler(CallbackQueryHandler(on_quality, pattern=r"^q:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
