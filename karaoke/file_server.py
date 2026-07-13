"""Tiny aiohttp file server for temporary large-video download links.

Serves a health page (used by the HF Space health check + external uptime ping)
and `/d/<token>` links that stream a downloaded file once, with a TTL. Files are
removed after the TTL expires (swept) — the bot deletes small files itself.
"""

from __future__ import annotations

import logging
import os
import secrets
import time

from aiohttp import web

logger = logging.getLogger(__name__)


class LinkRegistry:
    """Maps unguessable tokens to file paths with a time-to-live."""

    def __init__(self, ttl_seconds: int, now=time.monotonic):
        self._ttl = ttl_seconds
        self._now = now
        self._map: dict[str, tuple[str, float]] = {}

    def register(self, path: str) -> str:
        token = secrets.token_urlsafe(16)
        self._map[token] = (path, self._now())
        return token

    def resolve(self, token: str) -> str | None:
        item = self._map.get(token)
        if not item:
            return None
        path, ts = item
        if self._now() - ts > self._ttl:
            self._map.pop(token, None)
            return None
        return path

    def sweep(self) -> None:
        expired = [tok for tok, (_p, ts) in self._map.items() if self._now() - ts > self._ttl]
        for tok in expired:
            path, _ = self._map.pop(tok)
            try:
                os.remove(path)
            except OSError:
                pass


def make_app(registry: LinkRegistry) -> web.Application:
    app = web.Application()

    async def health(_request):
        return web.Response(text="ok")

    async def download(request):
        token = request.match_info["token"]
        path = registry.resolve(token)
        if not path or not os.path.exists(path):
            return web.Response(status=404, text="הקישור פג תוקף או לא נמצא")
        return web.FileResponse(path)

    app.add_routes([web.get("/", health), web.get("/d/{token}", download)])
    return app
