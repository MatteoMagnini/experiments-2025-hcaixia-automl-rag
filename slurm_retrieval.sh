#!/bin/bash
#
# SLURM + pyxis/enroot launcher for AutoML-RAG experiments
#

#SBATCH --job-name=best_pipelines
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --nodelist=maryam
#SBATCH --output=automl_rag_%j.log
#SBATCH --error=automl_rag_%j.err
#SBATCH --container-image=/storage/IDA/lsanna/experiments-2025-hcaixia-automl-rag/automl-experiments.sqsh
#SBATCH --container-mounts=/storage/IDA:/data,/storage/IDA/lsanna/experiments-2025-hcaixia-automl-rag/entrypoint.sh:/app/entrypoint.sh
#SBATCH --container-writable


set -euo pipefail

echo "============================================================"
echo "SLURM RETRIEVAL EXPERIMENT START"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================================"


# ---------------------------------------------------------------------------
# RUN EXPERIMENT SCRIPT INSIDE CONTAINER
# ---------------------------------------------------------------------------

EXPERIMENT_SCRIPT="/data/lsanna/experiments-2025-hcaixia-automl-rag/poe-retrieval-experiment/test_set.sh"

if [ ! -f "$EXPERIMENT_SCRIPT" ]; then
    echo "ERROR: Experiment script not found at $EXPERIMENT_SCRIPT"
    exit 1
fi

echo "Launching experiment script..."
bash "$EXPERIMENT_SCRIPT"


echo "============================================================"
echo "SLURM JOB COMPLETE"
echo "Time: $(date)"
echo "============================================================"