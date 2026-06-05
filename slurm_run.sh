#!/bin/bash
#
# SLURM + pyxis/enroot launcher for the AutoML-RAG experiments.
# Modelled on the cluster's ecai_sim.sh. Adjust the marked (<-- EDIT) values to
# your cluster, then submit with:  OPENROUTER_API_KEY=sk-or-... sbatch slurm_run.sh
#
#SBATCH --job-name=automl_rag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=l40s                                  # <-- EDIT partition
#SBATCH --qos=normal                                      # <-- EDIT qos
#SBATCH --output=automl_rag_%j.log
#SBATCH --error=automl_rag_%j.err
#SBATCH --container-image=/storage/IDA/automl_rag.sqsh    # <-- EDIT path to the .sqsh
#SBATCH --container-mounts=/storage/IDA:/data             # <-- EDIT host:container mount
#SBATCH --container-writable

set -euo pipefail

# --- required secret ---------------------------------------------------------
# Export OPENROUTER_API_KEY in your shell BEFORE `sbatch` (it is inherited), or
# uncomment and set it here. Do NOT commit a real key.
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY before submitting this job}"
export OPENROUTER_API_KEY

# --- experiment configuration (consumed by /app/entrypoint.sh) ---------------
export PROVIDER=huggingface
export DATA_DIR=/data/automl_rag
export EMBEDDERS="qwen3-embedding-0.6b,qwen3-embedding-4b,granite-embedding-278m,granite-embedding-107m,nomic-embed-text-v2-moe,bert-base-italian-xxl-cased"
export WALLTIME_LIMIT=72000        # SMAC walltime budget in seconds (< SBATCH --time)
export EVAL_SAMPLE_SIZE=100        # questions used for the generative metrics; 0 disables them
# export GEN_MODEL="google/gemma-3-4b-it,qwen/qwen3-8b"   # optional: override OpenRouter models
# export HF_TOKEN=hf_xxx                                  # only needed for gated embedders

# With pyxis the script body already runs INSIDE the container image, so we just
# invoke the entrypoint.
bash /app/entrypoint.sh
