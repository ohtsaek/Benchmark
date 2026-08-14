#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python train.py \
--gpu 0 \
--dataset_dir 'data/Example_train/' \
--save_dir 'data/Example_train/results/' \
--dataset_name 'custom' \
--cl_idname 'depmap_id' \
--drug_idname 'name' \
--label 'auc' \
--init_seed 42 \
--num_workers 4 \
--batch_size 256 \
--epochs 3 \
--early_stop_epoch 10 \
--init_lr 0.0001 \
--weight_decay 0.00001

# --gpu: specify the GPU device.
# If no GPU device is available, you can specify --cpu  to force the model to run on the CPU.
# --dataset_dir: specify the dataset directory.
# --save_dir: specify the save directory, if not set, defaults to the "results" subdirectory of the dataset directory.
# --dataset_name: specify the dataset name, "custom" for custom dataset, "PRISM" for PRISM dataset, "GDSC" for GDSC dataset.
# --cl_idname: specify the cell line id name, default is "depmap_id" for custom dataset.
# --drug_idname: specify the drug id name, default is "name" for custom dataset.
# --label: specify the label name, default is "auc" for custom dataset.
# --init_seed: specify the initial seed, default is 42.
# You can specify --drop_last to drop the last incomplete batch, default is False.
# specify --get_attn to get gene and pathway attention score.
