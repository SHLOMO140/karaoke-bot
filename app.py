"""Hugging Face Gradio Space entrypoint (free tier — no Docker).

Runs the Telegram bot in a background thread and a tiny Gradio status page on
port 7860 (the only public port). Large videos that exceed Telegram's 50MB cap
are served through Gradio's built-in ``/file=`` endpoint (allowed_paths), so no
separate file server / extra port is needed. A periodic sweep deletes expired
download files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from urllib.parse import quote

import gradio as gr
from telegram.ext import ApplicationBuilder

import bot
from karaoke.config import DOWNLOAD_DIR, LINK_TTL_SECONDS, PUBLIC_BASE_URL, TELEGRAM_BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

DOWNLOAD_ROOT = os.path.abspath(str(DOWNLOAD_DIR))


def _link_builder(path: str) -> str:
    """URL for a large local file, served by Gradio's /file= endpoint."""
    base = PUBLIC_BASE_URL.rstrip("/")
    return f"{base}/file={quote(os.path.abspath(path))}"


def _sweep_downloads() -> None:
    cutoff = time.time() - LINK_TTL_SECONDS
    for root, _dirs, files in os.walk(DOWNLOAD_ROOT):
        for name in files:
            p = os.path.join(root, name)
            try:
                if os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except OSError:
                pass


def _run_bot() -> None:
    async def _main() -> None:
        if not TELEGRAM_BOT_TOKEN:
            logger.error("No Telegram bot token found; bot not started.")
            return
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        bot.register_handlers(application, _link_builder)
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot polling started")
        while True:
            await asyncio.sleep(300)
            _sweep_downloads()

    asyncio.run(_main())


with gr.Blocks(title="Karaoke Bot") as demo:
    gr.Markdown("# 🎸 הבוט פעיל\nשלח שם שיר או קישור יוטיוב לבוט בטלגרם.")

if __name__ == "__main__":
    # Start the bot in the background, then run the (blocking) Gradio server.
    threading.Thread(target=_run_bot, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[DOWNLOAD_ROOT])
