from karaoke.error_formatter import format_pipeline_error
from karaoke.exceptions import DownloadError


def test_download_error_is_explained_in_hebrew():
    error = DownloadError("Unable to download video: [Errno 22] Invalid argument")
    message = format_pipeline_error(error, job_id="abc123")

    assert "הורדת המדיה נכשלה" in message
    assert "ההורדה מיוטיוב נתקעה בזמן כתיבה של קובץ זמני" in message
    assert "Invalid argument" not in message
    assert "משימה: abc123" in message
    assert "abc123" in message
