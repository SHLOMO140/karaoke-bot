"""Single-process entrypoint: Telegram long-polling + aiohttp file server.

On Hugging Face Spaces the aiohttp server binds the required web port (7860),
serves the health page (used by the platform + the external uptime ping) and
the temporary large-video download links. The Telegram bot polls in the same
event loop.
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web
from telegram.ext import ApplicationBuilder

import bot
from karaoke.config import FILE_SERVER_PORT, LINK_TTL_SECONDS, TELEGRAM_BOT_TOKEN
from karaoke.file_server import LinkRegistry, make_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

SWEEP_INTERVAL_SECONDS = 300


async def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("No Telegram bot token found (set BOT_TOKEN or bot_token.txt).")

    registry = LinkRegistry(ttl_seconds=LINK_TTL_SECONDS)

    # aiohttp file server (health + /d/<token>) on the platform's web port.
    runner = web.AppRunner(make_app(registry))
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", FILE_SERVER_PORT).start()
    logger.info("File server listening on 0.0.0.0:%s", FILE_SERVER_PORT)

    # Telegram bot polling in the same loop.
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    bot.register_handlers(application, registry)
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    logger.info("Telegram bot polling started")

    try:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            registry.sweep()
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
