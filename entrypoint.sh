#!/usr/bin/env bash
#
# Container entrypoint for the AutoML-RAG experiments.
# Everything is configured through environment variables so the same image can
# be reused from SLURM without rebuilding (see slurm_run.sh).
set -euo pipefail

# --- required secret ---------------------------------------------------------
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required (export it in the SLURM script)}"

# --- tunables (override via env) ---------------------------------------------
PROVIDER="${PROVIDER:-huggingface}"
# Public, Italian-capable embedders (no HF token needed). Override to subset.
EMBEDDERS="${EMBEDDERS:-qwen3-embedding-0.6b,qwen3-embedding-4b,granite-embedding-278m,granite-embedding-107m,nomic-embed-text-v2-moe,bert-base-italian-xxl-cased}"
# Empty -> use the in-code default list of OpenRouter Gemma/Qwen models.
GEN_MODEL="${GEN_MODEL:-}"
WALLTIME_LIMIT="${WALLTIME_LIMIT:-60}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-100}"

# --- persistence on the mounted volume ---------------------------------------
DATA_DIR="${DATA_DIR:-/data}"

# Cache HuggingFace downloads so multi-GB embedders survive across jobs.
export HF_HOME="${HF_HOME:-${DATA_DIR}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
mkdir -p "${HF_HOME}"

# An HF token is only needed for gated models; honour it if provided.
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Persist the (expensive) Qdrant embedding cache by pointing /app/qdrant/faq at
# the mounted volume.
QDRANT_CACHE="${QDRANT_CACHE:-${DATA_DIR}/qdrant_faq}"
mkdir -p "${QDRANT_CACHE}"
rm -rf /app/qdrant/faq
ln -s "${QDRANT_CACHE}" /app/qdrant/faq

cd /app

CMD=(python3 -m experiments
     --provider "${PROVIDER}"
     --embedders "${EMBEDDERS}"
     --walltime_limit "${WALLTIME_LIMIT}"
     --eval_sample_size "${EVAL_SAMPLE_SIZE}")
if [[ -n "${GEN_MODEL}" ]]; then
    CMD+=(--gen_model "${GEN_MODEL}")
fi

echo "[entrypoint] provider=${PROVIDER} embedders=${EMBEDDERS}"
echo "[entrypoint] HF_HOME=${HF_HOME} qdrant_cache=${QDRANT_CACHE}"
echo "[entrypoint] running: ${CMD[*]}"
"${CMD[@]}"

# --- collect results onto the mounted volume ---------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/automl_rag_results/${SLURM_JOB_ID:-local}}"
mkdir -p "${OUTPUT_DIR}"
cp -f /app/results/incumbents.csv          "${OUTPUT_DIR}/"                 2>/dev/null || true
cp -f /app/results/*_results.csv           "${OUTPUT_DIR}/"                 2>/dev/null || true
cp -f /app/results/cache/results.csv       "${OUTPUT_DIR}/cache_results.csv" 2>/dev/null || true
echo "[entrypoint] results copied to ${OUTPUT_DIR}"
