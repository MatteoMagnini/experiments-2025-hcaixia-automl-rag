#!/bin/bash
#
# SLURM + pyxis/enroot launcher for the AutoML-RAG experiments.
# Modelled on the cluster's ecai_sim.sh. Adjust the marked (<-- EDIT) values to
# your cluster, then submit with:
#
#   OPENROUTER_API_KEY=sk-or-... sbatch slurm_run.sh
#
#SBATCH --job-name=automl_rag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --output=automl_rag_%j.log
#SBATCH --error=automl_rag_%j.err
#SBATCH --container-image=/storage/IDA/automl_rag.sqsh
#SBATCH --container-mounts=/storage/IDA:/data
#SBATCH --container-writable

set -euo pipefail

# ---------------------------------------------------------------------------
# Required secret
# ---------------------------------------------------------------------------
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY before submitting this job}"
export OPENROUTER_API_KEY

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
export PROVIDER=huggingface

# Persistent directory inside the mounted volume
export DATA_DIR=/data/automl_rag

export EMBEDDERS="qwen3-embedding-0.6b,qwen3-embedding-4b,granite-embedding-278m,granite-embedding-107m,nomic-embed-text-v2-moe,bert-base-italian-xxl-cased"

# SMAC optimization budget (12h) kept below the SLURM walltime (24h)
# to leave room for cleanup and result collection.
export WALLTIME_LIMIT=43200

# Number of questions used for generative evaluation.
# Set to 0 to disable.
export EVAL_SAMPLE_SIZE=100

# Optional overrides:
# export GEN_MODEL="google/gemma-3-4b-it,qwen/qwen3-8b"
# export HF_TOKEN=hf_xxx

# Already running inside the container with Pyxis.
bash /app/entrypoint.sh