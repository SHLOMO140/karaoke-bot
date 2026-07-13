# Lean karaoke bot image for Hugging Face Spaces (Docker SDK).
FROM python:3.12-slim

# ffmpeg for audio/video, nodejs for yt-dlp's EJS challenge solver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects the web app on 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
