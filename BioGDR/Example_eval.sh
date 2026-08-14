#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python ind_eval.py \
--gpu 0 \
--result_dir 'data/Example_train/results/' \
--target_data_dir 'data/Example_eval' \
--save_dir 'data/Example_eval/results/' \
--batch_size 256 \
--num_workers 4 

# If no GPU device is available, you can specify --cpu to force the model to run on the CPU.
# specify --get_attn to get gene and pathway attention score.