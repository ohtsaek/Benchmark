#Get path parameters
import os

def get_path_config(args):
    args.geneset_path = os.path.join(args.dataset_dir,"cell/geneset.gmt")

    get_prism_drug_path(args)
    args.geneset_feature_path = os.path.join(args.dataset_dir, f'cell/geneset_exp')

    dataset_name = str(args.dataset_name)
    if dataset_name in ["PRISM","GDSC"]:
        split_type = getattr(args, "split_type", None)
        split_idx = getattr(args, "split_idx", None) 
        if split_type is None:
            raise ValueError("PRISM/GDSC dataset requires args.split_type (e.g., random/drug_sim_blind/cell_sim_blind)")
        if split_idx is None:
            raise ValueError("PRISM/GDSC dataset requires args.split_idx (e.g., 1/2/3/4/5)")
            
        args.datasplit_path = os.path.join(args.dataset_dir, "Experiment", split_type, "data_split")
        file_prefix = f"{split_type}_{split_idx}"
        args.train = os.path.join(args.datasplit_path, f"{file_prefix}_Training.csv")
        args.valid = os.path.join(args.datasplit_path, f"{file_prefix}_Validation.csv")
        args.test = os.path.join(args.datasplit_path, f"{file_prefix}_Test.csv")

    elif dataset_name == "custom":
        if _is_custom_ind_eval(args):
            args.datasplit_path = None
            args.train = args.valid = args.test = None
            args.target_dataset_path = os.path.join(args.dataset_dir, "data.csv")
        else:
            args.datasplit_path = os.path.join(args.dataset_dir, "data_split")
            args.train = os.path.join(args.datasplit_path, "Training.csv")
            args.valid = os.path.join(args.datasplit_path, "Validation.csv")
            args.test = os.path.join(args.datasplit_path, "Test.csv")
    else:
        raise ValueError("dataset_name is not supported")

    _check_split_files(args)


def _is_custom_ind_eval(args):
    return (
        hasattr(args, "ind_eval")
        and args.ind_eval is True
    )


def _check_split_files(args):
    dataset_name = str(getattr(args, "dataset_name", "custom"))
    if dataset_name == "custom" and _is_custom_ind_eval(args):
        target = getattr(args, "target_dataset_path", None) or os.path.join(
            args.dataset_dir, "data.csv"
        )
        if not os.path.exists(target):
            raise FileNotFoundError(f"Evaluation data file is missing: {target}")
        return

    split_paths = [
        getattr(args, "train", None),
        getattr(args, "valid", None),
        getattr(args, "test", None),
    ]
    missing_files = [p for p in split_paths if p and not os.path.exists(p)]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(f"Split csv files are missing: {missing_text}")

def get_prism_drug_path(args):
    args.drug_feat_types = args.drug_feat_type.split('+')  
    args.drug_path = dict()
    for drug_feat in args.drug_feat_types[1:]:
        args.drug_path[drug_feat] = get_single_prism_drug_path(args,drug_feat)

def get_single_prism_drug_path(args,sdrug_feat_type):
    if sdrug_feat_type in ['afp']:
        drug_path = os.path.join(args.dataset_dir, 'drug/drug.csv')
    elif sdrug_feat_type == 'kinome':
        drug_path = os.path.join(args.dataset_dir, 'drug/kinome/kinome.csv')
    elif sdrug_feat_type in ['GATtsall']:
        drug_path = os.path.join(args.dataset_dir, 'drug/transcriptome/transcriptall.csv')
    else:
        raise ValueError('drug feature type not supported')

    if sdrug_feat_type in ['GATtsall']:
        args.drug_ppi_path = os.path.join(args.dataset_dir, 'drug/transcriptome/gene_string_mapping.csv')
    return drug_path




