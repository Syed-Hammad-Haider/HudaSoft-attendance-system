# Hudasoft Attendance System - container image
# Build:  docker compose build      Run:  docker compose up -d
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Karachi \
    ATTENDANCE_DATA_DIR=/data \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

# System libraries: OpenCV/MediaPipe need libGL + glib; insightface needs
# a C++ compiler to build; tzdata so the 6 am work-day boundary uses
# Karachi time instead of the server's UTC.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 tzdata curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default wheel bundles ~2 GB of CUDA the
# server cannot use), then the project's own pinned requirements.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /data /root/.insightface

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:5000/healthz || exit 1

# Waitress: production server. 16 threads because every open live-video
# tab holds one thread for as long as it is open.
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--threads=16", "web_dashboard:app"]
