#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# PROJECT SETUP
# ---------------------------------------------------------------------------

PROJECT_DIR=/data/lsanna/experiments-2025-hcaixia-automl-rag/poe-retrieval-experiment
cd "$PROJECT_DIR"

mkdir -p logs

echo "============================================================"
echo "RETRIEVAL EXPERIMENT RUNNER"
echo "Time: $(date)"
echo "============================================================"


# ---------------------------------------------------------------------------
# LOAD ENV
# ---------------------------------------------------------------------------

if [ -f .env ]; then
    echo "Loading .env..."
    set -a
    source .env
    set +a
fi


# ---------------------------------------------------------------------------
# UV SYNC
# ---------------------------------------------------------------------------

echo "[$(date)] Syncing uv workspace..."
unset VIRTUAL_ENV
export UV_CACHE_DIR=/tmp/uv-cache

uv sync --all-packages
echo "[$(date)] uv sync complete"


# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------

export MODEL_SERVER_PORT=7997
export MODEL_SERVER_URL=http://localhost:$MODEL_SERVER_PORT
export HF_HOME=/tmp/models
export FASTEMBED_CACHE_PATH=/tmp/models

mkdir -p /tmp/models


# ---------------------------------------------------------------------------
# START MODEL SERVER
# ---------------------------------------------------------------------------

echo "============================================================"
echo "[$(date)] Starting model server"
echo "============================================================"

pkill -f "uvicorn model_server.server:app" 2>/dev/null || true
sleep 3

uv run python -m uvicorn model_server.server:app \
    --host 0.0.0.0 \
    --port "$MODEL_SERVER_PORT" \
    --log-level warning \
    > logs/model_server.log 2>&1 &

MODEL_PID=$!

echo "Model server PID: $MODEL_PID"

# wait for readiness
until uv run python -c "import httpx; httpx.get('http://localhost:$MODEL_SERVER_PORT/ready').raise_for_status()" \
    >/dev/null 2>&1; do
    echo "[$(date)] Waiting for model server..."
    sleep 5
done

echo "============================================================"
echo "[$(date)] MODEL SERVER READY"
echo "============================================================"


# ---------------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------------

QUERY_CSV=${QUERY_CSV:-"/data/lsanna/experiments-2025-hcaixia-automl-rag/data/test.csv"}
FULL_DATA=${FULL_DATA:-"/data/lsanna/experiments-2025-hcaixia-automl-rag/data/data_topchunks.csv"}
WORKERS=${WORKERS:-16}

BM25_COLLECTION=${BM25_COLLECTION:-"nomic-chatfaq"}
MMR_COLLECTION=${MMR_COLLECTION:-"qwen-chatfaq"}

TEST_FLAG=""
if [[ "${TEST_MODE:-0}" == "1" ]]; then
    TEST_FLAG="--test_mode"
    echo "🧪 TEST MODE ENABLED (subset execution)"
fi

echo "BM25_COLLECTION = $BM25_COLLECTION"
echo "MMR_COLLECTION  = $MMR_COLLECTION"
echo "QUERY_CSV       = $QUERY_CSV"
echo "FULL_DATA       = $FULL_DATA"
echo "WORKERS         = $WORKERS"
echo "TEST_FLAG       = $TEST_FLAG"


# ---------------------------------------------------------------------------
# EXPERIMENT FUNCTION
# ---------------------------------------------------------------------------

run_experiments () {
    DATASET_PATH=$1
    LABEL=$2

    echo "============================================================"
    echo "[$(date)] STARTING EXPERIMENTS FOR: $LABEL"
    echo "DATA: $DATASET_PATH"
    echo "============================================================"

    # ---------------- MMR ----------------
    echo "[MMR][$LABEL] Launching → $BM25_COLLECTION"

    uv run python run_retrieval_test.py \
        --strategy mmr \
        --collection "$BM25_COLLECTION" \
        --query_csv "$DATASET_PATH" \
        --dataset_name "$LABEL" \
        --embedding_model "nomic-ai/nomic-embed-text-v2-moe" \
        --workers "$WORKERS" \
        $TEST_FLAG \
        2>&1 | tee >(sed "s/^/[BM25][$LABEL] /") logs/bm25_${LABEL}.log &

    BM25_PID=$!



    # ---------------- ensemble ----------------

    echo "[ensemble][$LABEL] Launching → $MMR_COLLECTION"



    uv run python run_retrieval_test.py \
        --strategy ensemble \
        --collection "$MMR_COLLECTION" \
        --query_csv "$DATASET_PATH" \
        --dataset_name "$LABEL" \
        --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
        --workers "$WORKERS" \
        --mmr_lambda "0.11" \
        --mmr_candidates "36" \
        $TEST_FLAG \
        2>&1 | tee >(sed "s/^/[MMR][$LABEL] /") logs/mmr_${LABEL}.log &

    MMR_PID=$!


   

    wait $BM25_PID
    wait $MMR_PID

    echo "============================================================"
    echo "[$(date)] DONE: $LABEL"
    echo "============================================================"
}


# ---------------------------------------------------------------------------
# RUN EXPERIMENTS (QUERY + FULL DATA)
# ---------------------------------------------------------------------------

echo "============================================================"
echo "[$(date)] Starting A/B retrieval experiments"
echo "============================================================"

run_experiments "$QUERY_CSV" "QUERY"
run_experiments "$FULL_DATA" "FULL"


# ---------------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Shutting down model server (PID $MODEL_PID)"
echo "============================================================"

kill $MODEL_PID 2>/dev/null || true

echo "[$(date)] DONE"