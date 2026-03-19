"""Vocal separation providers."""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path

from .config import FFMPEG_EXE, PYTHON_EXE
from .exceptions import SeparationError

logger = logging.getLogger(__name__)


def _find_demucs_file(demucs_out: Path, stem_name: str) -> str:
    files = list(demucs_out.glob(f"**/{stem_name}.mp3"))
    if not files:
        files = list(demucs_out.glob(f"**/{stem_name}.wav"))
    if not files:
        contents = [str(path) for path in demucs_out.rglob("*")]
        raise SeparationError(f"Demucs output missing {stem_name}: {contents}")
    return str(files[0])


def _copy_or_convert(src: str, dst: str):
    if src.endswith(".wav") and dst.endswith(".mp3"):
        result = subprocess.run([FFMPEG_EXE, "-y", "-i", src, "-q:a", "0", dst], capture_output=True, timeout=120)
        if result.returncode != 0:
            raise SeparationError(result.stderr.decode(errors="ignore")[-300:], "המרת הפלייבק ל-MP3 נכשלה.")
        return
    if src.endswith(".mp3") and dst.endswith(".wav"):
        result = subprocess.run(
            [FFMPEG_EXE, "-y", "-i", src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dst],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise SeparationError(result.stderr.decode(errors="ignore")[-300:], "המרת הווקאל ל-WAV נכשלה.")
        return
    shutil.copy2(src, dst)


class DemucsSeparator:
    name = "demucs"

    def separate(self, input_audio: str, job_dir: Path) -> tuple[str, str]:
        abs_input = str(Path(input_audio).resolve())
        demucs_out = job_dir / f"demucs_{uuid.uuid4().hex[:8]}"
        demucs_out.mkdir(exist_ok=True)

        try:
            cmd = [
                PYTHON_EXE,
                "-m",
                "demucs",
                "--two-stems=vocals",
                "--mp3",
                "-o",
                str(demucs_out),
                abs_input,
            ]
            logger.info("Running Demucs on %s", abs_input)
            result = subprocess.run(cmd, capture_output=True, timeout=900)
            if result.returncode != 0:
                raise SeparationError(result.stderr.decode(errors="ignore")[-600:])

            vocals_path = _find_demucs_file(demucs_out, "vocals")
            no_vocals_path = _find_demucs_file(demucs_out, "no_vocals")

            dst_vocals = str(job_dir / "vocals.wav")
            dst_instrumental = str(job_dir / "instrumental.mp3")
            _copy_or_convert(vocals_path, dst_vocals)
            _copy_or_convert(no_vocals_path, dst_instrumental)
            return dst_vocals, dst_instrumental
        finally:
            shutil.rmtree(str(demucs_out), ignore_errors=True)


def separate_vocals(input_audio: str, job_dir: Path) -> tuple[str, str]:
    return DemucsSeparator().separate(input_audio, job_dir)
