# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# STAGE 1: uv
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.8.15-python3.12-trixie-slim AS uv

# ---------------------------------------------------------------------------
# STAGE 2: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS run

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf/hub

WORKDIR /app

RUN rm -f /etc/apt/sources.list.d/*cuda*.list \
    /etc/apt/sources.list.d/*nvidia*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock /app/

RUN uv export \
        --frozen \
        --no-emit-project \
        --format requirements-txt \
        -o /tmp/requirements.txt \
    && uv pip install \
        --system \
        --break-system-packages \
        --no-cache \
        -r /tmp/requirements.txt

COPY . /app

RUN chmod +x /app/entrypoint.sh \
    && sed -i 's/\r$//' /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]