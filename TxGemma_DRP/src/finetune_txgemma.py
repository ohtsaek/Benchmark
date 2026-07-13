"""
TxGemma Drug Response Prediction - Fine-tuning Script

This script fine-tunes the TxGemma model for drug response prediction using
GDSC (Genomics of Drug Sensitivity in Cancer) data.

Based on: https://github.com/google-gemini/gemma-cookbook/blob/main/TxGemma/[TxGemma]Finetune_with_Hugging_Face.ipynb
"""

import os
import pandas as pd
import numpy as np
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import wandb
import argparse 
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from accelerate import Accelerator

# Import ai4clinic modules for metrics and visualization
from ai4clinic.graphs import best2worst2
from ai4clinic.metrics import corr4drug


def get_compound_names(file_name):
    """Load compound names mapping from SMILES to drug names."""
    compound_dict = {}
    
    if not os.path.exists(file_name):
        print(f"Warning: Compound names file {file_name} not found.")
        return compound_dict
    
    try:
        with open(file_name, 'r') as fi:
            for line in fi:
                tokens = line.strip().split('\t')
                if len(tokens) >= 3:
                    smiles = tokens[1]
                    drug_name = tokens[2]
                    compound_dict[smiles] = drug_name
    except Exception as e:
        print(f"Error reading compound names file: {e}")
    
    return compound_dict


def normalize_value(value, min_value=0.0, max_value=1.0, normalize=True, target_range=1000.0):
    """
    Normalizes or denormalizes a value between min and max range.
    
    Args:
        value: Value to normalize/denormalize
        min_value: Minimum value in range
        max_value: Maximum value in range
        normalize: If True, normalizes to 0-target_range. If False, denormalizes to min-max range.
        target_range: Target range for normalization (default: 1000.0)
    
    Returns:
        Normalized or denormalized value
    """
    min_value, max_value = float(min_value), float(max_value)
    
    try:
        if normalize:
            return ((float(value) - min_value) / (max_value - min_value)) * target_range
        else:
            return (float(value) / target_range) * (max_value - min_value) + min_value
    except (ValueError, ZeroDivisionError):
        return target_range / 2.0 if normalize else (max_value + min_value) / 2.0


def create_jsonl_from_df(file_path, output_path, cell_description_df):
    """Convert tab-separated data to JSONL format for training."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    column_names = ['Cell Line_ID', 'Drug', 'sensitivity']
    dataframe = pd.read_csv(file_path, delimiter='\t', names=column_names)    

    # Add description for each cell line
    dataframe['Description'] = dataframe['Cell Line_ID'].apply(
        lambda x: cell_description_df[cell_description_df['Cell line Name'] == x]['Cell description'].values[0]
        if x in cell_description_df['Cell line Name'].values else 'Unknown cell line'
    )
    
    # Input template for drug response prediction
    input_template = {
        "DRUG_RESPONSE": (
            "Instructions: Answer the following question about drug responses.\n"
            "Context: The same drug compound could have various levels of responses in different patients. "
            "To design drug for individual or a group with certain characteristics is the central goal of precision medicine. "
            "In experiments, IC50s of drugs were measured against cancer cell lines.\n"
            "Question: Given a drug SMILES string and a cell line description, predict the normalized drug sensitivity from "
            "000 to 1000, where 000 is minimum drug sensitivity and 1000 is maximum drug sensitivity.\n"
            "Drug SMILES: {drug_smiles}\n"
            "Cell line description: {cell_line}\n"
            "Answer:"
        )
    }

    # Write JSONL file
    with open(output_path, 'w') as jsonl_file:
        for index, row in dataframe.iterrows():
            template_string = input_template["DRUG_RESPONSE"]
            input_text = template_string.format(
                drug_smiles=row['Drug'],
                cell_line=f"{row['Cell Line_ID']}, {row['Description']}"
            )

            data_point = {
                "input_text": input_text,
                "output_text": str(row['sensitivity'])
            }

            jsonl_file.write(json.dumps(data_point) + '\n')
    
    print(f"Data has been successfully written to {output_path}")


def formatting_func(example):
    """Format example for training."""
    text = f"{example['input_text']} {example['output_text']}<eos>"
    return text


def evaluate_model(accelerator, model, tokenizer, test_data, args):
    """Evaluate the model on test data."""
    
    class DrugSensitivityDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            input_text = str(item['input_text'])
            output_text = float(item['output_text'])

            try:
                # Extract SMILES and cell line description from input text
                smiles_start = input_text.find('Drug SMILES: ') + len('Drug SMILES: ')
                smiles_end = input_text.find('Cell line description:')
                smiles = input_text[smiles_start:smiles_end].strip()

                cell_line_start = input_text.find('Cell line description: ') + len('Cell line description: ')
                group_start = input_text.find('Answer:')
                cell_line_description = input_text[cell_line_start:group_start].strip()
            except ValueError:
                smiles = "Unknown"
                cell_line_description = "Unknown"

            return input_text, output_text, smiles, cell_line_description
        
    dataset = DrugSensitivityDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=min(100, len(dataset)), shuffle=False)
    dataloader = accelerator.prepare(dataloader)   

    predictions = []
    actual_values = []
    smiles_list = []
    cell_lines_list = []
    
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_actuals, batch_smiles, batch_cell_lines in dataloader:
            # Tokenize inputs
            inputs = tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Move inputs to device
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Generate predictions
            try:
                outputs = model.module.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            except AttributeError:
                # Handle case where model is not wrapped in DataParallel
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode outputs
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j, output in enumerate(decoded_outputs):
                try:
                    # Extract predicted value from output
                    if 'Answer:' in output:
                        pred_value = float(output.split('Answer:')[-1].strip())
                    else:
                        pred_value = 500.0  # Default middle value

                    predictions.append(pred_value)
                    actual_values.append(batch_actuals[j])
                    smiles_list.append(batch_smiles[j])
                    cell_lines_list.append(batch_cell_lines[j])
                except (ValueError, IndexError) as e:
                    print(f"Error processing prediction {j}: {e}")
                    continue

    # Calculate metrics only on main process
    if accelerator.is_main_process:
        # Denormalize predictions and actual values
        predictions = [normalize_value(pred, min_value=0, max_value=1, normalize=False) 
                      for pred in predictions]
        actual_values = [normalize_value(act, min_value=0, max_value=1, normalize=False) 
                        for act in actual_values]

        if predictions and actual_values:
            pearson_corr, p_value = pearsonr(predictions, actual_values)
            spearman_corr, sp_value = spearmanr(predictions, actual_values)
            rmse = np.sqrt(np.mean((np.array(actual_values) - np.array(predictions))**2))
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "test/pearson_correlation": pearson_corr,
                    "test/spearman_correlation": spearman_corr,
                    "test/rmse": rmse
                })

            print(f"Test Results:")
            print(f"Pearson correlation: {pearson_corr:.3f} (p={p_value:.3f})")
            print(f"Spearman correlation: {spearman_corr:.3f} (p={sp_value:.3f})")
            print(f"RMSE: {rmse:.3f}")
            
            # Save predictions to file
            output_file = os.path.join(args.output_folder, f"test_predictions_{args.fold}.txt")
            print(f"Saving test predictions to {output_file}")
            
            predictions_df = pd.DataFrame({
                'actual': [v.item() if torch.is_tensor(v) else v for v in actual_values],
                'predicted': predictions,
                'smiles': smiles_list,
                'cell_line': cell_lines_list
            })
            predictions_df.to_csv(output_file, sep='\t', index=False)
            
            # Log predictions file to wandb if available
            if wandb.run is not None:
                artifact = wandb.Artifact(f"test_predictions_{args.fold}", type="predictions")
                artifact.add_file(output_file)
                wandb.log_artifact(artifact)

        return predictions, actual_values, smiles_list, cell_lines_list
    
    return [], [], [], []


def main(args):
    """Main training function."""
    # Initialize the accelerator
    accelerator = Accelerator()

    # Set Hugging Face token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Initialize wandb only on main process
    if accelerator.is_main_process and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.fold,
            tags=args.wandb_tags,
            job_type=args.job_type,
        )

        # Load cell description data
        if os.path.exists(args.cell_description_file):
            with open(args.cell_description_file, 'r') as fi:
                lines = [line.strip().split(',', 1) for line in fi if line.strip()]
            cells_info = pd.DataFrame(lines, columns=['Cell line Name', 'Cell description'])
            print(f"Loaded {len(cells_info)} cell descriptions")
            
            # Create JSONL files for training
            create_jsonl_from_df(args.train_file_path, args.train_jsonl_file_path, cells_info)
            create_jsonl_from_df(args.val_file_path, args.val_jsonl_file_path, cells_info)
            create_jsonl_from_df(args.test_file_path, args.test_jsonl_file_path, cells_info)
        else:
            print(f"Warning: Cell description file {args.cell_description_file} not found")

    # Wait for all processes to finish data preparation
    accelerator.wait_for_everyone()

    # Load dataset
    data = load_dataset(
        "json",
        data_files={
            "train": args.train_jsonl_file_path,
            "validation": args.val_jsonl_file_path,
            "test": args.test_jsonl_file_path
        }
    )

    # Load or download model
    if not os.path.exists(args.save_directory):
        print("Loading model from Hugging Face...")
        # Use 4-bit quantization to reduce memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=quantization_config,
            device_map={'': torch.cuda.current_device()} if torch.cuda.is_available() else 'auto',
            torch_dtype="auto",
            attn_implementation="eager",
        )

        # Save model and tokenizer locally
        os.makedirs(args.save_directory, exist_ok=True)
        model.save_pretrained(args.save_directory)
        tokenizer.save_pretrained(args.save_directory)
    else:
        print("Loading model from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(args.save_directory)
        model = AutoModelForCausalLM.from_pretrained(
            args.save_directory,
            device_map={'': torch.cuda.current_device()} if torch.cuda.is_available() else 'auto',
            torch_dtype="auto",
            attn_implementation="eager",
        )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "o_proj", "k_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    # Prepare the model for training with LoRA
    model = get_peft_model(model, lora_config)

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            max_seq_length=args.max_seq_length,
            output_dir=args.output_folder,
            optim=args.optimizer,
            report_to="wandb" if args.wandb_project else "none",
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Start training
    print("Starting training...")
    trainer.train()

    # Run evaluation
    if accelerator.is_main_process:
        print("Evaluating model...")
        predictions, actuals, smiles, cell_lines = evaluate_model(
            accelerator, model, tokenizer, data["test"], args
        )
        
        # Generate visualization
        if predictions and actuals:
            try:
                metrics, drug_correlations = best2worst2(
                    predictions=predictions,
                    real_results=actuals,
                    drugs=smiles,
                    y_range=(-0.01, 0.9),
                    x_range=(-0.01, 0.9),
                    patients=cell_lines,
                    folds=[1] * len(smiles),
                    plot_size=(5, 6),
                    num_select=2,
                    title=f"TxGemma predictions for {args.fold} fold",
                    ylabel="TxGemma predictions",
                    xlabel="Real results",
                    output_path=os.path.join(args.output_folder, f"txgemma_predictions_{args.fold}.png"),
                    display_plot=False,
                )
                
                # Log drug correlations
                print("\nDrug-wise correlations:")
                valid_corrs = []
                for drug, corr in drug_correlations.items():
                    if corr is not None:
                        print(f"{drug}: {corr:.3f}")
                        valid_corrs.append(corr)
                    else:
                        print(f"{drug}: None")

                if valid_corrs:
                    mean_drug_corr = np.mean(valid_corrs)
                    print(f"\nMean drug correlation: {mean_drug_corr:.3f}")
                    if wandb.run is not None:
                        wandb.log({"test/mean_drug_correlation": mean_drug_corr})
                
                # Upload plot to wandb
                if wandb.run is not None:
                    plot_path = os.path.join(args.output_folder, f"txgemma_predictions_{args.fold}.png")
                    if os.path.exists(plot_path):
                        artifact = wandb.Artifact(name=f"predictions_plot_{args.fold}", type="plot")
                        artifact.add_file(plot_path)
                        wandb.log_artifact(artifact)
                        wandb.log({"test/predictions_plot": wandb.Image(plot_path)})
                        
            except Exception as e:
                print(f"Error generating visualization: {e}")

    # Finalize wandb run
    if accelerator.is_main_process and wandb.run is not None:
        wandb.finish()


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description='Fine-tune TxGemma for Drug Response Prediction')

    # Data paths
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Directory where the input data is stored')
    parser.add_argument('--output_folder', type=str, default="./results",
                       help='Output directory for model checkpoints and logs')
    parser.add_argument('--fold', type=str, default="fold1",
                       help='Fold identifier for cross-validation')
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default="google/txgemma-2b-predict",
                       help='Model ID from Hugging Face')
    parser.add_argument('--save_directory', type=str, default="./local_txgemma_model",
                       help='Directory to save the model and tokenizer')
    
    # Data files
    parser.add_argument('--cell_description_file', type=str, required=True,
                       help='File containing cell line descriptions')
    parser.add_argument('--train_file_path', type=str, required=True,
                       help='Path to the training data TSV file')
    parser.add_argument('--train_jsonl_file_path', type=str, required=True,
                       help='Path to save the training data JSONL file')
    parser.add_argument('--val_file_path', type=str, required=True,
                       help='Path to the validation data TSV file')
    parser.add_argument('--val_jsonl_file_path', type=str, required=True,
                       help='Path to save the validation data JSONL file')
    parser.add_argument('--test_file_path', type=str, required=True,
                       help='Path to the test data TSV file')
    parser.add_argument('--test_jsonl_file_path', type=str, required=True,
                       help='Path to save the test data JSONL file')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Number of steps to accumulate gradients')
    parser.add_argument('--warmup_steps', type=int, default=2,
                       help='Number of warmup steps')
    parser.add_argument('--num_train_epochs', type=int, default=5,
                       help='Total number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                       help='Learning rate')
    parser.add_argument('--fp16', type=bool, default=True,
                       help='Use mixed precision training')
    parser.add_argument('--logging_steps', type=int, default=5,
                       help='Frequency of logging training information')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='Maximum sequence length for input data')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='Rank of the adaptation matrices for LoRA')
    parser.add_argument('--optimizer', type=str, default='adamw_torch',
                       help='Optimizer to use for training')

    # WandB parameters
    parser.add_argument('--wandb_project', type=str,
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str,
                       help='Weights & Biases entity name')
    parser.add_argument('--wandb_tags', type=str, nargs='+', 
                       default=['TxGemma', 'DrugResponse'],
                       help='Tags for the WandB run')
    parser.add_argument('--job_type', type=str, default='finetune',
                       help='Job type for WandB')

    # Authentication
    parser.add_argument('--hf_token', type=str,
                       help='Hugging Face token for model access')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    main(args)