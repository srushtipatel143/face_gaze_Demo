# Prevent Python from writing .pyc files and enable unbuffered logging
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime shared libraries required by MediaPipe and OpenCV (YuNet/SFace)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libxcb1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with high timeout and retries
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

# Bake model weights into the image at build time (Section 10)
RUN mkdir -p /app/weights && python -c "\
import urllib.request as u; \
base = 'https://github.com/opencv/opencv_zoo/raw/main/models/'; \
files = {'face_detection_yunet_2023mar.onnx': 'face_detection_yunet/', \
         'face_recognition_sface_2021dec.onnx': 'face_recognition_sface/'}; \
[u.urlretrieve(base + d + f, '/app/weights/' + f) for f, d in files.items()]; \
import os; sizes = {f: os.path.getsize('/app/weights/' + f) for f in files}; print(sizes); \
assert all(s > 100000 for s in sizes.values()), 'weights download looks wrong (Git LFS pointer?)'"

# Copy application source code
COPY . .

# Expose FastAPI backend port
EXPOSE 8000

# Native Python container healthcheck
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Run Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
