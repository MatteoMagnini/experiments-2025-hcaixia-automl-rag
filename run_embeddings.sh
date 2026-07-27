#!/usr/bin/env bash
# run_embeddings.sh
# Reproducible embedding runner — creates nomic-chatfaq and qwen-chatfaq
# collections on a remote Qdrant instance.
#
# Required environment variables:
#   QDRANT_URL     — e.g. https://your-cluster.qdrant.io
#   QDRANT_API_KEY — your Qdrant API key
#
# Usage:
#   export QDRANT_URL="https://..."
#   export QDRANT_API_KEY="..."
#   bash run_embeddings.sh

# ── Load .env file ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -o allexport
  source "${SCRIPT_DIR}/.env"
  set +o allexport
fi

set -euo pipefail

# ── Validate environment ─────────────────────────────────────────────────────
: "${QDRANT_URL:?ERROR: QDRANT_URL is not set}"
: "${QDRANT_PORT:?ERROR: QDRANT_PORT is not set}"
: "${QDRANT_API_KEY:?ERROR: QDRANT_API_KEY is not set}"

echo "Target Qdrant: ${QDRANT_URL}"
echo
uv add qdrant-client

# ── Configs: embedder|provider|chunk_len|overlap|collection ──────────────────
CONFIGS=(
  "nomic-embed-text-v2-moe|huggingface|114|0.32|nomic-chatfaq"
  "qwen3-embedding-0.6b|huggingface|295|0.21|qwen-chatfaq"
)

# ── Run ──────────────────────────────────────────────────────────────────────
for config in "${CONFIGS[@]}"; do
  IFS='|' read -r EMBEDDER PROVIDER CHUNK_LEN OVERLAP COLLECTION <<< "${config}"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Embedder  : ${EMBEDDER}"
  echo "  Provider  : ${PROVIDER}"
  echo "  Chunk len : ${CHUNK_LEN}"
  echo "  Overlap   : ${OVERLAP}"
  echo "  Collection: ${COLLECTION}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

 uv run python qdrant/standalone_embed.py \
    --chunk_token_length="${CHUNK_LEN}" \
    --overlap_percentage="${OVERLAP}" \
    --embedder="${EMBEDDER}" \
    --collection_name="${COLLECTION}"

  echo "  ✓ Done: ${COLLECTION}"
  echo
done

echo "All collections embedded successfully."