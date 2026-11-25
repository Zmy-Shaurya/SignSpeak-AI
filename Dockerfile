# SignSpeak AI web server (FastAPI + MediaPipe + PyTorch CPU).
#   docker build -t signspeak .
#   docker run -p 8000:8000 --device /dev/video0 -e GEMINI_API_KEY signspeak    (Linux host with a webcam)
# Then open http://localhost:8000. Without --device the UI and API still run and report "camera error".
# Docker Desktop on Windows/macOS cannot pass a webcam into a container: run `python server.py` natively there.
# Speech is Windows-only (SAPI), so in the container sentences are shown as text.
FROM python:3.11-slim

# OpenCV (pulled in by MediaPipe) needs these shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY keypoints.py train.py engine.py sentence.py server.py ./
COPY web ./web
COPY models ./models

RUN useradd --create-home app && usermod -aG video app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
