# relay — multi-stage Dockerfile (ONNX Runtime, torch-free runtime)
#
# Stage 1 (builder): install torch + optimum to export the embedding model
#   to ONNX format, then install runtime deps into a venv.
# Stage 2 (runtime): slim image with only the venv (no torch) + ONNX model +
#   app code. ~500MB vs ~2.2GB with torch, ~200-300MB RAM vs ~500MB+.

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

# Build tools needed to compile any C extensions without prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a venv outside /root so stage 2 can copy it to the same path.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# --- Step 1: Export the embedding model to ONNX format ---
# This requires torch + optimum (export-only, not in the runtime image).
# CPU-only torch to avoid pulling CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install "optimum[onnxruntime]" transformers tokenizers

# Export all-MiniLM-L6-v2 to ONNX and save the tokenizer alongside it.
# The ONNX file contains only the transformer; pooling is done in numpy
# at runtime (see app/onnx_encoder.py).
ENV HF_HOME=/opt/hf-cache
RUN python -c "\
from optimum.onnxruntime import ORTModelForFeatureExtraction as M; \
m = M.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', export=True); \
m.save_pretrained('/opt/model'); \
from transformers import AutoTokenizer; \
t = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
t.save_pretrained('/opt/model')"

# --- Step 2: Install runtime deps (no torch, no sentence-transformers) ---
# Wipe the venv and recreate with only runtime deps so torch/optimum don't
# leak into the runtime image.
RUN rm -rf /opt/venv && python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

# Runtime-only system deps. libgomp1 is required by numpy/onnxruntime on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from the builder (runtime deps only — no torch).
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the pre-exported ONNX model + tokenizer files.
COPY --from=builder /opt/model /opt/model

# Create a non-root user for security.
RUN useradd --create-home --uid 1001 relay
WORKDIR /app

# Copy app code + deploy config. config.yaml is gitignored and not in the
# build context; the deploy config (Groq-only) is baked in and selected via
# RELAY_CONFIG_PATH. Local docker-compose overrides this to mount config.yaml.
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.deploy.yaml ./config.deploy.yaml

ENV RELAY_CONFIG_PATH=config.deploy.yaml \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER relay

EXPOSE 8000

# Single worker for V1 — the in-memory cache is per-process, so multiple
# workers would each maintain their own cache (post-V1: Redis shared cache).
# PORT is injected by Render (default 10000) and other PaaS providers; fall
# back to 8000 for local docker-compose.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
