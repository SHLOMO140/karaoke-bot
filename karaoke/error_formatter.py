"""User-facing Hebrew explanations for pipeline failures."""

from __future__ import annotations

from .exceptions import PipelineError


def _technical_to_hebrew(technical_message: str) -> str:
    text = (technical_message or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    if "unable to download video" in lowered and "invalid argument" in lowered:
        return "ההורדה מיוטיוב נתקעה בזמן כתיבה של קובץ זמני. ניקיתי את קבצי הביניים, ובדרך כלל ניסיון נוסף פותר את זה."
    if "unable to download video" in lowered:
        return "יוטיוב לא החזיר את הווידאו בצורה תקינה. נסה שוב בעוד רגע או שלח קובץ ישירות."
    if "video unavailable" in lowered:
        return "הסרטון שביקשת לא זמין כרגע ביוטיוב."
    if "http error 429" in lowered or "too many requests" in lowered:
        return "יוטיוב חסם זמנית את ההורדה בגלל עומס או יותר מדי בקשות. נסה שוב בעוד כמה דקות."
    if "sign in to confirm you're not a bot" in lowered or "sign in to confirm you’re not a bot" in lowered:
        return "יוטיוב דורש כרגע אימות אנושי לפני הורדה של הסרטון הזה."
    if "ffmpeg" in lowered and "not found" in lowered:
        return "נראה ש-FFmpeg לא זמין כרגע ולכן אי אפשר לעבד את המדיה."
    if "permission denied" in lowered:
        return "אין הרשאה לכתוב את קבצי הביניים בתיקיית העבודה."
    if "network is unreachable" in lowered or "timed out" in lowered or "connection" in lowered:
        return "נראה שהייתה בעיית רשת זמנית בזמן ההורדה או העיבוד."
    if "no such file" in lowered or "cannot find the path" in lowered:
        return "אחד מקבצי הביניים לא נוצר או לא נמצא בזמן העיבוד."
    return ""


def format_pipeline_error(error: PipelineError, job_id: str | None = None) -> str:
    parts = [error.info.user_message]
    detail = _technical_to_hebrew(error.info.technical_message)
    if detail:
        parts.append(detail)
    if job_id:
        parts.append(f"משימה: {job_id}")
    return "\n\n".join(parts)


def format_unexpected_error(message: str, job_id: str | None = None) -> str:
    detail = _technical_to_hebrew(message)
    parts = ["אירעה שגיאה לא צפויה במהלך העיבוד."]
    if detail:
        parts.append(detail)
    else:
        parts.append("אפשר לנסות שוב. אם זה חוזר, שלח לי שוב את הקישור או את הקובץ.")
    if job_id:
        parts.append(f"משימה: {job_id}")
    return "\n\n".join(parts)
