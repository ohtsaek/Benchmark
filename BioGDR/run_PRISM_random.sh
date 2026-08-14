#!/usr/bin/env bash
# This script trains the PRISM dataset in the random split setting.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for SPLIT_IDX in 1 2 3 4 5; do
  python train.py \
    --gpu 0 \
    --dataset_dir "data/PRISM" \
    --num_workers 4 \
    --dataset_name PRISM \
    --split_type random \
    --split_idx "${SPLIT_IDX}" \
    --init_seed 42 \
    --batch_size 256 \
    --epochs 300 \
    --early_stop_epoch 15 \
    --init_lr 0.0001 \
    --weight_decay 0.00001 \
    || {
      echo "[WARN] train.py failed: PRISM random split_idx=${SPLIT_IDX}" >&2
      true
    }
done

# Notes:
# Run from the repository root, or invoke this script by path so ROOT resolves correctly.
# --gpu: GPU device id. If no GPU is available, use --cpu to force CPU.
# --dataset_dir: PRISM data root (expects Experiment/<split_type>/data_split/...).
# --save_dir: optional; if omitted, defaults to
#   data/PRISM/Experiment/random/results/split<split_idx>_initseed<init_seed>_<timestamp>/
# --dataset_name PRISM: uses split_type and split_idx to load
#   random_<split_idx>_{Training,Validation,Test}.csv under data_split/.
# Default column names match PRISM CSVs: depmap_id (cell), name (drug), auc (label); override with
#   --cl_idname, --drug_idname, --label if your tables differ.
# --split_type random: random split by pair; this script runs split_idx 1..5.
# --init_seed: 42 here; change the literal if you need another run.
# --batch_size 256, --epochs 300, --early_stop_epoch, --init_lr, --weight_decay: training hyperparameters.
# Optional: --drop_last (drop last incomplete batch), --get_attn (gene/pathway attention export).
