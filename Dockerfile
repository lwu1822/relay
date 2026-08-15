# relay — multi-stage Dockerfile
#
# Stage 1 (builder): install build tools + pip deps into a venv, then
#   pre-download the sentence-transformers embedding model so it is baked
#   into the image and runtime startup doesn't depend on network access.
# Stage 2 (runtime): slim image with only the venv + app code + model cache.

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

# Build tools needed to compile any C extensions without prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a venv outside /root so stage 2 can copy it to the same path.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Keep pip quiet in logs but still show errors.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install deps first (better layer caching — app code changes more often).
# Install CPU-only torch first from the PyTorch CPU index to avoid pulling
# ~2GB of CUDA libraries we don't need (the embedding model runs on CPU).
# Subsequent `pip install -r requirements.txt` sees torch is already
# satisfied and skips it.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install -r requirements.txt

# Pre-download the embedding model into a fixed cache dir so we can copy it
# to the runtime stage. This avoids a ~90MB download + network dependency on
# first request after deploy.
ENV HF_HOME=/opt/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

# Runtime-only system deps. libgomp1 is required by numpy/torch on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from the builder.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the pre-downloaded HF cache and force offline mode so the app never
# tries to download the model at runtime (fails fast if cache is missing
# instead of hanging on a network call).
COPY --from=builder /opt/hf-cache /opt/hf-cache
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

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
    PYTHONUNBUFFERED=1

USER relay

EXPOSE 8000

# Single worker for V1 — the in-memory cache is per-process, so multiple
# workers would each maintain their own cache (post-V1: Redis shared cache).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
