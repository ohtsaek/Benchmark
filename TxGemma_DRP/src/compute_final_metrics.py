"""
This script computes the correlation between the real AUDRC and the predicted AUDRC for each drug on an individual basis.
In the case of multiple models due to k-fold cross-validation, an average correlation is derived.
It also computes the density plot of all models and its metrics.
"""

import argparse
import os 
import numpy as np
import pandas as pd
import torch
import wandb
from scipy.stats import pearsonr, spearmanr

# Import ai4clinic modules for metrics and visualization
from ai4clinic.graphs import drugs2waterfall, preds2scatter, best2worst2


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

  
def list_of_strings(arg):
    """Convert comma-separated string to list."""
    if ',' in arg:
        return arg.split(',')
    else:
        return [arg]

    
def get_compound_names(file_name):
    """Load compound names from file."""
    compounds = []
    
    if not os.path.exists(file_name):
        print(f"Warning: Compound names file {file_name} not found.")
        return compounds

    try:
        with open(file_name, 'r') as fi:
            for line in fi:
                tokens = line.strip().split('\t')
                if len(tokens) >= 3:
                    compounds.append([tokens[1], tokens[2]])
    except Exception as e:
        print(f"Error reading compound names: {e}")
    
    return compounds


def main():
    """Main function to compute and visualize metrics."""
    parser = argparse.ArgumentParser(description='TxGemma final metrics computation')

    # WandB parameters
    parser.add_argument('--wandb_project', type=str, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, help='Weights & Biases entity name')
    parser.add_argument('--wandb_tags', type=str, nargs='+', 
                       default=['TxGemma', 'FinalMetrics'], 
                       help='Tags for the WandB run')
    parser.add_argument('--job_type', type=str, default="final_metrics", 
                       help='Job type')
    parser.add_argument('--sweep_name', type=str, default="final metrics", 
                       help='Sweep name')

    # Data parameters
    parser.add_argument('--input_type', type=str, default="mutations", 
                       help='Type of omics data used')
    parser.add_argument('--split_type', type=str, default="random", 
                       help='Type of data split used (e.g., cell-blind, drug-blind)')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Directory containing the input data folders')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Directory containing the folders that have the resulting models')
    parser.add_argument('--samples_folders', type=list_of_strings, 
                       default=["fold1", "fold2", "fold3", "fold4", "fold5"],
                       help='Folders to analyze (comma-separated)')
    
    # File names
    parser.add_argument('--predictions_name', type=str, default="test_predictions_", 
                       help='Prefix for prediction files')
    parser.add_argument('--labels_name', type=str, default="test.txt", 
                       help='Name of labels file')
    parser.add_argument('--drugs_names', type=str, default="compound_names.txt", 
                       help='Drugs names and SMILES file')
    parser.add_argument('--drugs_fake_lelo', type=list_of_strings, required=False,
                       help='Drugs used in fake LELO (only if needed)')
    parser.add_argument('--log_artifact', type=bool, default=True, 
                       help='Log artifacts to wandb')
    
    opt, unknown = parser.parse_known_args()

    # Setup hyperparameters for wandb
    hyperparameters = {
        "input_type": opt.input_type,
        "split_type": opt.split_type,
    }
    
    if opt.drugs_fake_lelo:
        hyperparameters["drugs_fake_lelo"] = opt.drugs_fake_lelo

    # Initialize wandb if project is specified
    if opt.wandb_project:
        run = wandb.init(
            project=opt.wandb_project, 
            entity=opt.wandb_entity, 
            name=opt.sweep_name, 
            tags=opt.wandb_tags, 
            job_type=opt.job_type, 
            config=hyperparameters
        )
    else:
        run = None
        print("Warning: No wandb project specified. Skipping wandb logging.")

    # Load compound names
    compound_names_file = os.path.join(opt.input_folder, opt.drugs_names)
    compound_names = get_compound_names(compound_names_file)
    if compound_names and compound_names[0][0] == 'SMILES':  # Skip header if exists
        compound_names.pop(0)

    # Create output figures directory
    figures_dir = os.path.join(opt.output_folder, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Iterate over each fold in the cross-validation
    for i, fold in enumerate(opt.samples_folders):
        input_path = os.path.join(opt.input_folder, fold)
        output_path = os.path.join(opt.output_folder, fold)
        print(f"Processing fold: {fold}")
        
        # Load predictions from the unified CSV file
        predictions_file = os.path.join(output_path, f"{opt.predictions_name}{fold}.txt")
        
        if not os.path.exists(predictions_file):
            print(f"Warning: Predictions file not found: {predictions_file}")
            continue
            
        try:
            fold_data = pd.read_csv(predictions_file, delimiter="\t", header=0)
            
            fold_predictions = fold_data['predicted'].values
            fold_labels = fold_data['actual'].values
            fold_drugs = fold_data['smiles'].values
            fold_cells = fold_data['cell_line'].values
            
            # Extract fold number from fold name
            try:
                fold_num = int(''.join(filter(str.isdigit, fold)))
            except ValueError:
                fold_num = i + 1
            
            # Create tensor of fold names
            fold_names = torch.full((len(fold_predictions),), fold_num, dtype=torch.int)
            
            # Initialize or concatenate data
            if i == 0:
                all_predictions = fold_predictions
                all_labels = fold_labels
                all_drugs = fold_drugs
                all_cells = fold_cells
                all_fold_names = fold_names
            else:
                all_predictions = np.concatenate([all_predictions, fold_predictions])
                all_labels = np.concatenate([all_labels, fold_labels])
                all_drugs = np.concatenate([all_drugs, fold_drugs])
                all_cells = np.concatenate([all_cells, fold_cells])
                all_fold_names = torch.cat([all_fold_names, fold_names], dim=0)
                
        except Exception as e:
            print(f"Error processing fold {fold}: {e}")
            continue

    # Convert SMILES to drug names
    smiles_to_name = {smiles: name for smiles, name in compound_names}
    all_drug_names = [smiles_to_name.get(smiles, "Unknown") for smiles in all_drugs]
    
    # Denormalize predictions and labels
    all_predictions = [normalize_value(pred, min_value=0, max_value=1, normalize=False) 
                      for pred in all_predictions]
    all_labels = [normalize_value(label, min_value=0, max_value=1, normalize=False) 
                 for label in all_labels]

    print(f"\nTotal samples processed: {len(all_predictions)}")
    print(f"Number of unique drugs: {len(set(all_drugs))}")
    print(f"Number of unique cell lines: {len(set(all_cells))}")

    # Generate visualizations
    # Density scatter plot
    print("\nGenerating density scatter plot...")
    try:
        scatter_metrics = preds2scatter(
            all_predictions, all_labels, all_cells, all_fold_names,
            output_path=os.path.join(figures_dir, f"density_{opt.input_type}_{opt.split_type}.png"),
            density_bins=(90, 90),
            cmap='turbo',
            marker='.',
            marker_size=6,
            best_fit_line=True,
            title='',
            title_fontsize=20,
            xlabel='Real Response (AUDRC)',
            xlabel_fontsize=20,
            ylabel='Predicted Response (AUDRC)',
            ylabel_fontsize=20,
            display_plot=False,
            verbose=True,
            show_legend=True,
            legend_position=(0.46, 0.25),
            annotation_fontsize=17,
            transparent_bg=True,
            plot_size=(7, 5),
            x_range=(-0.01, 1.1),
            y_range=(0, 0.7),
            xtick_fontsize=17,
            ytick_fontsize=17
        )
        
        if run:
            run.log({
                "Average pearson cor": scatter_metrics['average_pearson'],
                "Average spearman cor": scatter_metrics['average_spearman'],
                "Overall pearson cor": scatter_metrics['overall_pearson'],
                "Overall spearman cor": scatter_metrics['overall_spearman']
            })
            
            if opt.log_artifact:
                artifact = wandb.Artifact("preds2scatter", type="graphic")
                artifact.add_file(os.path.join(figures_dir, f"density_{opt.input_type}_{opt.split_type}.png"))
                run.log_artifact(artifact)
                
        print(f"  Average Pearson: {scatter_metrics['average_pearson']:.3f}")
        print(f"  Average Spearman: {scatter_metrics['average_spearman']:.3f}")
        
    except Exception as e:
        print(f"Error generating scatter plot: {e}")

    # Best to worst drugs plot
    print("\nGenerating best-to-worst drugs plot...")
    try:
        metrics = best2worst2(
            all_predictions, all_labels, all_drug_names, all_cells, all_fold_names,
            plot_size=(11, 8),
            corr_metric='spearman',
            num_select=2,
            output_path=os.path.join(figures_dir, f"{opt.input_type}_{opt.split_type}.png"),
            marker='.',
            marker_size=7,
            best_fit_line=True,
            title='',
            title_fontsize=40,
            xlabel='Real Response (AUDRC)',
            xlabel_fontsize=30,
            ylabel='Predicted Response (AUDRC)',
            ylabel_fontsize=30,
            annotation_fontsize=30,
            worst_color_hex=None,
            best_color_hex=None,
            display_plot=False,
            verbose=True,
            show_metrics=False,
            show_legend=True,
            legend_position=(2, 0.5),
            x_range=(-0.01, 1.1),
            y_range=(0, 0.7),
            xtick_fontsize=25,
            ytick_fontsize=25,
            transparent_bg=True
        )
        
        if run and opt.log_artifact:
            artifact = wandb.Artifact("best2worst2", type="graphic")
            artifact.add_file(os.path.join(figures_dir, f"{opt.input_type}_{opt.split_type}.png"))
            run.log_artifact(artifact)
            
    except Exception as e:
        print(f"Error generating best-to-worst plot: {e}")

    # Waterfall plot
    print("\nGenerating waterfall plot...")
    try:
        mean_corr, drug_corrs = drugs2waterfall(
            all_predictions, all_labels, all_drug_names, all_cells, all_fold_names,
            output_path=os.path.join(figures_dir, f"waterfall_{opt.input_type}_{opt.split_type}.png"),
            corr_metric='spearman',
            num_select=10,
            mark_threshold=0.5,
            color=None,
            display_plot=False,
            percentage_position=(0.17, 0.5),
            percentage_fontsize=40,
            ylabel='Spearman Correlation',
            ylabel_fontsize=25,
            xlabel='',
            xlabel_fontsize=20,
            title='',
            title_fontsize=20,
            ax2_title="",
            ytick_fontsize=22,
            transparent_bg=True,
            bar_annotation_fontsize=26,
            drug_name_fontsize=24,
            plot_size=(13, 7),
            ax2_ylim=(0.03, 0.9),
            legend_position=(1, 1),
            legend_fontsize=15,
            legend=False
        )
        
        if run:
            run.log({"Average spearman correlations of all drugs": mean_corr})
            
            if opt.log_artifact:
                artifact = wandb.Artifact("drugs2waterfall", type="graphic")
                artifact.add_file(os.path.join(figures_dir, f"waterfall_{opt.input_type}_{opt.split_type}.png"))
                run.log_artifact(artifact)
                
        print(f"  Mean drug correlation: {mean_corr:.3f}")
        
    except Exception as e:
        print(f"Error generating waterfall plot: {e}")
    
    if run:
        run.finish()
        
    print("\nMetrics computation completed!")


if __name__ == "__main__":
    main()
