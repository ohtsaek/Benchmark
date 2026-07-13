#!/bin/bash
#
# TxGemma Training Script
# This script runs the TxGemma fine-tuning process for drug response prediction
#

# ============================================================================
# CONFIGURATION - Update these variables according to your setup
# ============================================================================

# Activate Python environment
# Replace with your actual conda environment path
CONDA_ENV_PATH="${HOME}/conda/envs/txgemma"
source activate "${CONDA_ENV_PATH}"

# Set API keys (export these as environment variables or use a secure method)
# export WANDB_API_KEY="your_key_here"
# export HUGGINGFACE_TOKEN="your_token_here"

if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY not set. Weights & Biases logging may not work."
fi

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Warning: HUGGINGFACE_TOKEN not set. Model download may fail."
fi

# Login to Weights & Biases (if API key is set)
if [ ! -z "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
fi

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Weights & Biases settings
PROJECT="TxGemma_DrugResponse"
ENTITY="your_wandb_entity"  # Update with your W&B entity name
TAGS="drug-response,txgemma,gdsc"
EPOCHS=40
INPUT_TYPE="mutations"  # Options: mutations, expression, multiomics, nothing
SPLIT_TYPE="drug-blind"  # Options: drug-blind, cell-blind, cancer-blind, random

# Data paths - Update these to match your data location
DATA_ROOT="../data"
RESULTS_ROOT="../results"
FOLDER_NAME="GDSC1_drug-blind"
INPUT_FOLDER="${DATA_ROOT}/${FOLDER_NAME}/"
OUTPUT_FOLDER="${RESULTS_ROOT}/${FOLDER_NAME}/"

# Cross-validation folds to process
SAMPLES_FOLDERS="fold1,fold2,fold3,fold4,fold5"

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================

mkdir -p "${OUTPUT_FOLDER}"
mkdir -p "${OUTPUT_FOLDER}/figures"

# ============================================================================
# TRAINING LOOP
# ============================================================================

# Loop to execute the process (set to 1 for single run, increase for multiple runs)
for iteration in {1..1}; do
    echo "========================================="
    echo "Running iteration ${iteration}..."
    echo "========================================="
    
    # Generate unique job ID
    JOB_TYPE=$(date +%s | md5sum | cut -c 1-8)
    echo "Generated Job ID: ${JOB_TYPE}"

    # Training hyperparameters
    BATCH_SIZE=10
    LEARNING_RATE=5e-6
    LORA_R=8
    MAX_SEQ_LENGTH=512
    OPTIMIZER="adamw_torch"

    # Process each fold
    IFS=',' read -ra FOLDERS <<< "$SAMPLES_FOLDERS"
    for fold in "${FOLDERS[@]}"; do
        echo "-----------------------------------------"
        echo "Processing fold: ${fold}"
        echo "-----------------------------------------"
        
        mkdir -p "${OUTPUT_FOLDER}${fold}"

        # Detect number of GPUs available
        if command -v nvidia-smi &> /dev/null; then
            NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        else
            NUM_GPUS=1
        fi
        
        echo "Using ${NUM_GPUS} GPU(s)"

        # Run training with accelerate for multi-GPU support
        accelerate launch \
            --num_processes ${NUM_GPUS} \
            --main_process_port 29518 \
            --num_machines 1 \
            --mixed_precision "fp16" \
            --dynamo_backend no \
            ../src/finetune_txgemma.py \
            --input_folder "${INPUT_FOLDER}" \
            --output_folder "${OUTPUT_FOLDER}${fold}" \
            --model_id "google/txgemma-2b-predict" \
            --save_directory "./local_txgemma_model" \
            --cell_description_file "${DATA_ROOT}/gdsc1_celllines.txt" \
            --fold "${fold}" \
            --train_file_path "${INPUT_FOLDER}${fold}/train.txt" \
            --train_jsonl_file_path "${INPUT_FOLDER}${fold}/train_prompts.jsonl" \
            --val_file_path "${INPUT_FOLDER}${fold}/validate.txt" \
            --val_jsonl_file_path "${INPUT_FOLDER}${fold}/validate_prompts.jsonl" \
            --test_file_path "${INPUT_FOLDER}${fold}/test.txt" \
            --test_jsonl_file_path "${INPUT_FOLDER}${fold}/test_prompts.jsonl" \
            --batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps 4 \
            --warmup_steps 2 \
            --num_train_epochs ${EPOCHS} \
            --learning_rate ${LEARNING_RATE} \
            --fp16 True \
            --logging_steps 5 \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --lora_r ${LORA_R} \
            --optimizer ${OPTIMIZER} \
            --wandb_project "${PROJECT}" \
            --wandb_entity "${ENTITY}" \
            --wandb_tags ${TAGS} \
            --job_type "${JOB_TYPE}" \
            --hf_token "${HUGGINGFACE_TOKEN}"
    
        echo "Completed fold: ${fold}"
        echo "  Hyperparameters:"
        echo "    - Batch size: ${BATCH_SIZE}"
        echo "    - Learning rate: ${LEARNING_RATE}"
        echo "    - LoRA rank: ${LORA_R}"
        echo "    - Max sequence length: ${MAX_SEQ_LENGTH}"
        echo "    - Optimizer: ${OPTIMIZER}"
    done

    # ============================================================================
    # COMPUTE FINAL METRICS
    # ============================================================================
    
    echo ""
    echo "========================================="
    echo "Computing final metrics..."
    echo "========================================="
    
    python ../src/compute_final_metrics.py \
        --wandb_project "${PROJECT}" \
        --wandb_entity "${ENTITY}" \
        --wandb_tags ${TAGS} \
        --job_type "${JOB_TYPE}" \
        --sweep_name "final_metrics" \
        --input_type "${INPUT_TYPE}" \
        --split_type "${SPLIT_TYPE}" \
        --input_folder "${INPUT_FOLDER}" \
        --output_folder "${OUTPUT_FOLDER}" \
        --samples_folders "${SAMPLES_FOLDERS}" \
        --predictions_name "test_predictions_" \
        --labels_name "test.txt" \
        --drugs_names "compound_names.txt" \
        --log_artifact True
    
    echo "========================================="
    echo "Iteration ${iteration} completed!"
    echo "========================================="
done

echo ""
echo "All training iterations completed successfully!"
