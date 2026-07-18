"""Hugging Face Space entrypoint (Gradio SDK, but the app itself uses ONLY the
Python standard-library HTTP server — no gradio import — so it is immune to
gradio's dependency issues on the Space).

Runs the Telegram bot in a background thread and a tiny HTTP server on port 7860
for the health check (also the uptime-ping target) and large-video download
links. A periodic sweep deletes expired download files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from telegram.ext import ApplicationBuilder

import bot
from karaoke.config import DOWNLOAD_DIR, LINK_TTL_SECONDS, PUBLIC_BASE_URL, TELEGRAM_BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# httpx logs the full Telegram API URL (which contains the bot token) at INFO.
# On a public Space the logs are visible, so keep these quiet to avoid leaking it.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger("app")

# Satisfy ZeroGPU's "a @spaces.GPU function must exist" startup check. The bot is
# CPU-only and never calls this, so no GPU quota is consumed. No-op off ZeroGPU.
try:  # pragma: no cover - platform specific
    import spaces

    @spaces.GPU(duration=1)
    def _keep_zero_gpu_happy():
        return None
except Exception:  # noqa: BLE001
    pass

DOWNLOAD_ROOT = os.path.abspath(str(DOWNLOAD_DIR))
PORT = int(os.getenv("PORT", "7860"))

_links: dict[str, tuple[str, float]] = {}


def register_link(path: str) -> str:
    token = secrets.token_urlsafe(16)
    _links[token] = (path, time.time())
    return token


def resolve_link(token: str) -> str | None:
    item = _links.get(token)
    if not item:
        return None
    path, ts = item
    if time.time() - ts > LINK_TTL_SECONDS:
        _links.pop(token, None)
        return None
    return path


def _link_builder(path: str) -> str:
    return f"{PUBLIC_BASE_URL.rstrip('/')}/d/{register_link(path)}"


def _self_ping() -> None:
    """Hit our own public URL so the host's idle timer never fires.

    Render's free tier spins a web service down after ~15 min with no *inbound*
    HTTP traffic; the bot only makes outbound long-poll calls, so without this it
    would sleep and stop responding. Pinging our public URL every few minutes
    counts as inbound traffic and keeps the service awake 24/7 — no external
    uptime service or account needed. RENDER_EXTERNAL_URL is injected by Render.
    """
    url = os.getenv("RENDER_EXTERNAL_URL") or PUBLIC_BASE_URL
    if not url:
        return
    try:
        urllib.request.urlopen(url.rstrip("/") + "/health", timeout=10).read()
    except Exception as exc:  # noqa: BLE001 - best-effort keep-alive
        logger.debug("Self-ping failed: %s", exc)


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


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/health"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/d/"):
            token = self.path[3:].split("?", 1)[0]
            path = resolve_link(token)
            if path and os.path.exists(path):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{os.path.basename(path)}"',
                )
                self.send_header("Content-Length", str(os.path.getsize(path)))
                self.end_headers()
                with open(path, "rb") as fh:
                    shutil.copyfileobj(fh, self.wfile)
                return
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"expired or not found")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args):  # silence per-request logging
        return


def _run_server() -> None:
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


async def _bot_main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("No Telegram bot token found; bot not started.")
        return
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    bot.register_handlers(application, _link_builder)
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot polling started")
    tick = 0
    while True:
        await asyncio.sleep(240)  # self-ping every 4 min, well under Render's ~15 min idle-spindown
        await asyncio.to_thread(_self_ping)
        tick += 1
        if tick % 5 == 0:  # sweep expired downloads once every ~20 min
            _sweep_downloads()


if __name__ == "__main__":
    threading.Thread(target=_run_server, daemon=True).start()
    logger.info("HTTP server listening on 0.0.0.0:%s", PORT)
    asyncio.run(_bot_main())
