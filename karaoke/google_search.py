"""Google Custom Search API and YouTube Data API providers."""

import json
import logging
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class GoogleSearchQuotaError(RuntimeError):
    """Google Custom Search refused the request for quota/rate reasons.

    Raised (instead of silently returning []) so callers can skip further
    Google queries and jump straight to the web-search fallbacks.
    """

@dataclass
class SearchResult:
    """A single search result from Google or YouTube."""
    title: str
    url: str
    snippet: str


class GoogleSearchProvider:
    """Search via Google Custom Search JSON API."""

    ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, engine_id: str):
        self.api_key = api_key
        self.engine_id = engine_id

    def search(self, query: str, num: int = 10) -> list[SearchResult]:
        if not self.api_key or not self.engine_id:
            logger.warning("Google API key or engine ID not configured")
            return []
        params = urllib.parse.urlencode({
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": min(num, 10),
        })
        url = f"{self.ENDPOINT}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            return [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                )
                for item in data.get("items", [])
            ]
        except urllib.error.HTTPError as e:
            if e.code in (429, 403):
                logger.warning("Google search quota/rate limited: %s", e)
                raise GoogleSearchQuotaError(str(e)) from e
            logger.warning("Google search failed: %s", e)
            return []
        except Exception as e:
            logger.warning("Google search failed: %s", e)
            return []


class YouTubeDescriptionProvider:
    """Search YouTube and extract video descriptions."""

    ENDPOINT = "https://www.googleapis.com/youtube/v3/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            return []
        params = urllib.parse.urlencode({
            "key": self.api_key,
            "q": query,
            "part": "snippet",
            "type": "video",
            "maxResults": max_results,
        })
        url = f"{self.ENDPOINT}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            return [
                SearchResult(
                    title=item["snippet"].get("title", ""),
                    url=f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    snippet=item["snippet"].get("description", ""),
                )
                for item in data.get("items", [])
                if "id" in item and "videoId" in item.get("id", {})
            ]
        except Exception as e:
            logger.warning("YouTube search failed: %s", e)
            return []
