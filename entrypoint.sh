#!/usr/bin/env bash

set -euo pipefail

# ---------------------------------------------------------------------------
# Required secret
# ---------------------------------------------------------------------------
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
PROVIDER="${PROVIDER:-huggingface}"

EMBEDDERS="${EMBEDDERS:-qwen3-embedding-0.6b,qwen3-embedding-4b,granite-embedding-278m,granite-embedding-107m,nomic-embed-text-v2-moe,bert-base-italian-xxl-cased}"

GEN_MODEL="${GEN_MODEL:-}"
WALLTIME_LIMIT="${WALLTIME_LIMIT:-60}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-100}"

# ---------------------------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/data}"

mkdir -p "${DATA_DIR}"

# HuggingFace cache
export HF_HOME="${HF_HOME:-${DATA_DIR}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

mkdir -p "${HF_HOME}"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Qdrant cache
QDRANT_CACHE="${QDRANT_CACHE:-${DATA_DIR}/qdrant_faq}"

mkdir -p "${QDRANT_CACHE}"

rm -rf /app/qdrant/faq
ln -s "${QDRANT_CACHE}" /app/qdrant/faq

# ---------------------------------------------------------------------------
# GPU sanity check
# ---------------------------------------------------------------------------
python3 - <<'PY'
import torch

print("[gpu-check] torch version:", torch.__version__)
print("[gpu-check] cuda available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Refusing to start the experiment."
    )

print("[gpu-check] device:", torch.cuda.get_device_name(0))
PY

cd /app

CMD=(python3 -m experiments
     --provider "${PROVIDER}"
     --embedders "${EMBEDDERS}"
     --walltime_limit "${WALLTIME_LIMIT}"
     --eval_sample_size "${EVAL_SAMPLE_SIZE}")

if [[ -n "${GEN_MODEL}" ]]; then
    CMD+=(--gen_model "${GEN_MODEL}")
fi

echo "[entrypoint] provider=${PROVIDER}"
echo "[entrypoint] embedders=${EMBEDDERS}"
echo "[entrypoint] HF_HOME=${HF_HOME}"
echo "[entrypoint] qdrant_cache=${QDRANT_CACHE}"
echo "[entrypoint] running: ${CMD[*]}"

"${CMD[@]}"

# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/automl_rag_results/${SLURM_JOB_ID:-local}}"

mkdir -p "${OUTPUT_DIR}"

cp -f /app/results/incumbents.csv \
      "${OUTPUT_DIR}/" \
      2>/dev/null || true

cp -f /app/results/*_results.csv \
      "${OUTPUT_DIR}/" \
      2>/dev/null || true

cp -f /app/results/cache/results.csv \
      "${OUTPUT_DIR}/cache_results.csv" \
      2>/dev/null || true

echo "[entrypoint] results copied to ${OUTPUT_DIR}"