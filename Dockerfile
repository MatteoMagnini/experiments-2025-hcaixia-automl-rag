# syntax=docker/dockerfile:1
#
# GPU image for the AutoML-RAG experiments, meant to be converted to an enroot
# squashfs and launched with SLURM/pyxis (see slurm_run.sh + README_DOCKER.md).
# It uses the HuggingFace embedding provider (no Ollama server needed); the
# embedder models are downloaded from the HuggingFace Hub at runtime.

# --- STAGE 1: grab the uv binary --------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS uv

# --- STAGE 2: CUDA runtime ---------------------------------------------------
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS run

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    # HF caches default under /cache; overridden to the mounted /data at runtime
    # so multi-GB embedders are not re-downloaded on every job.
    HF_HOME=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf/hub

WORKDIR /app

# Python 3.12 (default on ubuntu 24.04) + git for trust_remote_code models.
# We drop NVIDIA's apt repo: the torch CUDA wheels bundle their own CUDA/cuDNN
# runtime, so we only need the Ubuntu archive here (and this avoids depending on
# NVIDIA's occasionally out-of-sync apt mirror).
RUN rm -f /etc/apt/sources.list.d/*cuda*.list /etc/apt/sources.list.d/*nvidia*.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv from stage 1.
COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv

# Install the Python dependencies first (better layer caching). We install the
# EXACT pinned versions from uv.lock (via `uv export`) rather than re-resolving
# pyproject.toml: SMAC 2.4 only works with the locked scikit-learn (1.8.0), and
# a fresh resolution pulls an incompatible newer sklearn.
COPY pyproject.toml uv.lock /app/
RUN uv export --frozen --no-emit-project --format requirements-txt -o /tmp/requirements.txt \
    && uv pip install --system --break-system-packages --no-cache -r /tmp/requirements.txt

# Project sources (the package is run with `python -m experiments` from /app,
# so it does not need to be pip-installed).
COPY . /app

RUN chmod +x /app/entrypoint.sh && sed -i 's/\r$//' /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
