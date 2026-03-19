from unittest.mock import patch, MagicMock
from karaoke.google_search import GoogleSearchProvider, YouTubeDescriptionProvider

def _mock_google_response():
    return {
        "items": [
            {
                "title": "שיר - שירונט",
                "link": "https://shironet.mako.co.il/artist?type=lyrics&lang=1&prfid=1&wrkid=1",
                "snippet": "מילות השיר...",
            },
            {
                "title": "שיר - Tab4U",
                "link": "https://www.tab4u.com/tabs/songs/1.html",
                "snippet": "אקורדים ומילים...",
            },
        ]
    }

@patch("karaoke.google_search.urllib.request.urlopen")
def test_google_search_returns_results(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = __import__("json").dumps(_mock_google_response()).encode()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    provider = GoogleSearchProvider(api_key="test", engine_id="test")
    results = provider.search("שיר מילים")
    assert len(results) == 2
    assert "shironet" in results[0].url

@patch("karaoke.google_search.urllib.request.urlopen")
def test_google_search_handles_empty_response(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"items": []}'
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    provider = GoogleSearchProvider(api_key="test", engine_id="test")
    results = provider.search("שיר לא קיים")
    assert len(results) == 0

@patch("karaoke.google_search.urllib.request.urlopen")
def test_google_search_handles_api_error(mock_urlopen):
    mock_urlopen.side_effect = Exception("API Error")
    provider = GoogleSearchProvider(api_key="test", engine_id="test")
    results = provider.search("שיר")
    assert len(results) == 0  # graceful fallback

@patch("karaoke.google_search.urllib.request.urlopen")
def test_youtube_search_extracts_description(mock_urlopen):
    yt_response = {
        "items": [{
            "id": {"videoId": "abc123"},
            "snippet": {
                "title": "שיר - מילים",
                "description": "שורה ראשונה\nשורה שנייה\nשורה שלישית",
            }
        }]
    }
    mock_response = MagicMock()
    mock_response.read.return_value = __import__("json").dumps(yt_response).encode()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    provider = YouTubeDescriptionProvider(api_key="test")
    results = provider.search("שיר מילים")
    assert len(results) == 1
    assert "שורה ראשונה" in results[0].snippet
