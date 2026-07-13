#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=xlong
#SBATCH --job-name=txgemma_drp
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=a100-sxm4
#SBATCH --mem=100gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=txgemma_training_%j.out
#SBATCH --error=txgemma_training_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@institution.edu

# ============================================================================
# SLURM Job Script for TxGemma Drug Response Prediction Training
# ============================================================================
#
# This script submits a training job to a SLURM-managed HPC cluster.
# 
# Usage:
#   sbatch submit_training.sbatch
#
# Customize the #SBATCH directives above according to your cluster's configuration:
#   - --partition: Queue/partition name
#   - --qos: Quality of Service
#   - --gres: GPU resources (adjust number based on availability)
#   - --constraint: Specific GPU type (e.g., a100-sxm4, rtx3090, v100)
#   - --mem: Memory allocation
#   - --mail-user: Your email for job notifications
#

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules (adjust based on your cluster)
module load Miniforge3
# module load CUDA/11.8  # Uncomment if needed

# Set cache directories to avoid filling up home directory
export WANDB_CACHE_DIR=${SCRATCH}/.cache/wandb
export TRITON_CACHE_DIR=${SCRATCH}/.triton_cache
export HF_HOME=${SCRATCH}/.cache/huggingface
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/transformers

# Create cache directories if they don't exist
mkdir -p ${WANDB_CACHE_DIR}
mkdir -p ${TRITON_CACHE_DIR}
mkdir -p ${HF_HOME}
mkdir -p ${TRANSFORMERS_CACHE}

# Navigate to project directory
# Update this path to match your project location
cd ${SCRATCH}/TxGemma_DRP/scripts || exit 1

# Verify GPU availability
echo "Available GPUs:"
nvidia-smi --list-gpus

# Execute the training script
bash train_txgemma.sh

echo "Job finished at: $(date)"
