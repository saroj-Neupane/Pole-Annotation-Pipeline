# Pole Annotation demo app — Hugging Face Docker Space image.
# Build context = repo root (locally: ./deploy/run_docker_local.sh; on HF: the Space repo).
# Pure-ONNX inference: no torch, no training code.
FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless needs glib/libgl; onnxruntime needs libgomp
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-deploy.txt

# App + ONNX SDKs. Weights ship inside sdk/ when present in the build context;
# otherwise the app downloads them from the HF model repo at startup.
COPY deploy/ ./deploy/
COPY sdk/ ./sdk/
# SDK production-version pointers (SSOT) — resolved by deploy/model_loader.py
COPY models/registry.json ./models/registry.json

# Bundled sample spans (the Space repo carries anonymized samples only)
COPY inference/ ./inference/
RUN mkdir -p ./inference/midspan/images ./inference/pole/images

# Non-root user (required by HF Spaces); needs write access for weight downloads + HF cache
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface \
    ENVIRONMENT=production \
    PORT=7860
EXPOSE 7860

CMD exec uvicorn deploy.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
