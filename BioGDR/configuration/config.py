#Get input parameters
import os.path
import torch
import time
from configuration.path_config import get_path_config
from _utils.utils import makedirs
from argparse import Namespace,ArgumentParser
import pickle
import json

share = Namespace()


def parser_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--cpu',action='store_true',default=False,help="Use cpu device")
    parser.add_argument('--gpu', type=int,
                        default=0,
                        help='Which GPU to use')
    
    #data args
    parser.add_argument('--dataset_dir',type=str,default="data/train_example/",help='dataset directory')
    parser.add_argument('--save_dir',type=str,default=None,help='Specify the save path. If not set, defaults to the dataset directory.')
    parser.add_argument('--num_workers', type=int, default=0,help='The number of workers')
    parser.add_argument('--drop_last', action='store_true', default=False, help='Drop last incomplete DataLoader batch (e.g. avoid BN batch=1)')
    parser.add_argument('--dataset_name', type=str, default='custom', choices=['custom', 'PRISM','GDSC'],
                        help='Dataset source: "custom" uses fixed Training/Validation/Test files; '
                             'PRISM/GDSC use split_type/split_idx naming under Experiment/')

    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'cell_blind', 'drug_blind', 'drug_sim_blind', 'cell_sim_blind'],
                        help='Data split strategy (only used when dataset_name is not custom)')
    parser.add_argument('--split_idx', type=int, default=1,
                        help='Split index used in split filename (only used when dataset_name is not custom): '
                             'run idx for random/cell_blind/drug_blind, fold idx for sim-blind')
    parser.add_argument('--init_seed', type=int, default=42,
                        help='Seed used for model/data-loader initialization reproducibility')
    
    parser.add_argument('--cl_idname',type=str,default='depmap_id',
                        help='Column name of cell lines. For custom datasets, keep default only if your column '
                             'is depmap_id; otherwise set this to your actual cell ID column name.')
    parser.add_argument('--drug_idname',type=str,default='name',
                        help='Column name of drugs. For custom datasets, keep default only if your column '
                             'is name; otherwise set this to your actual drug ID column name.')
    parser.add_argument('--label',type=str,default='auc',
                        help='Column name of labels. For custom datasets, keep default only if your label '
                             'column is auc; otherwise set this to your actual label column name.')  

    #train args
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to task') 
    parser.add_argument('--early_stop_epoch', type=int, default=10, help='If val loss did not drop in '
                                                                         'this epochs, stop running')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight_decay')
    parser.add_argument('--init_lr', type=float, default=1e-4,help='learning rate')
    parser.add_argument('--get_attn',action='store_true',default=False,help = 'get attention score for gene and pathways')

    args = parser.parse_args()
    return args

# Configuration parameters
def get_config():
    args = parser_args()  

    args.model = "BioGDR" 
    args.save_preds = True
    args.ind_eval = False
    get_device_config(args)
    get_train_config(args)
    get_variable_config(args)
    get_model_config(args)
    get_metric_config(args)
    if args.save_dir is None: 
        get_save_dir(args)
    get_path_config(args)

    makedirs(args.save_dir)

    args.args_dir = os.path.join(args.save_dir,'args.pickle')
    save_args(args)

    return args


def get_variable_config(args): 
    args.mode = "train"  
    args.model_name = 'BioGDR' 
    share.drug_feat_type ="mFU+GATtsall+kinome+afp"
    args.drug_feat_interaction = 'linear'

    get_ori_dim_dict(args)
    get_FU_feat_type_config(args)

def get_ori_dim_dict(args):
    args.ori_dim_dict = {}
    share.drug_feat_types = share.drug_feat_type.split('+')
    for drug_feat in share.drug_feat_types[1:]:
        get_drug_ori_dim(args,drug_feat)

def get_drug_ori_dim(args,drug_feat):
    if drug_feat in ['GATtsall']:
        args.ori_dim_dict[drug_feat] = 978
    elif drug_feat in ['kinome']:
        args.ori_dim_dict[drug_feat] = 527
    elif drug_feat in ['afp']:
        args.ori_dim_dict[drug_feat] = 512 
    else:
        raise ValueError(f'The feature name "{drug_feat}" is wrong')

def get_FU_feat_type_config(args):
    ori_dim_sum = sum([ args.ori_dim_dict[feat] for feat in share.drug_feat_types[1:]])
    args.single_drug_feat_dim = 512
    args.ori_drug_feat_dim = ori_dim_sum  
    share.drug_feat_dim = 512 * len(share.drug_feat_types[1:])  

def get_device_config(args):
    args.cuda = torch.cuda.is_available()
    if not args.cuda:
        print("GPU device not found. Using CPU instead.")
    if args.cuda and not args.cpu:
        gpu_count = torch.cuda.device_count()
        if args.gpu < 0:
            raise ValueError(f"GPU index must be non-negative, but got {args.gpu}.")
        if args.gpu >= gpu_count:
            print(f"Requested GPU index {args.gpu} is unavailable. Falling back to GPU 0.")
            args.gpu = 0
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
    args.quiet=False

def get_train_config(args):
    args.loss_fn = 'mse'
    args.dropout = 0
    args.activation='ReLU'

def get_model_config(args):
    args.drug_feat_type = share.drug_feat_type
    args.drug_feat_dim = share.drug_feat_dim
    get_attentiveFP_config(args)
    get_GATts_config(args)
    
    args.cell_feat_type = 'exp'  
    args.add_self_loop = True 
    args.residual = False 
    args.g_norm = 'BatchNorm'  
    args.readout = 'AttentiveFPReadout'  
    args.gene_num_timesteps =2

    args.gene_layers = 2
    args.gene_hidden_dim = 5
    args.gene_out_dim = 1
    args.gene_mid_heads = 3
    args.gene_out_heads = 3

    args.norm = True  

def get_metric_config(args):
    args.dataset_type = 'regression'   
    args.metric = 'loss'   
    args.metric_func = [args.metric,'rmse', 'pearson','spearman']  

def get_save_dir(args):
    args.taskid = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.dataset_name in ["PRISM","GDSC"]:
        args.save_dir = os.path.join(args.dataset_dir,'Experiment',args.split_type,"results")
        args.task_file_name = f"split{args.split_idx}_initseed{args.init_seed}_{args.taskid}"
        args.save_dir = os.path.join(args.save_dir,args.task_file_name)
    elif args.dataset_name == "custom":
        args.save_dir = os.path.join(args.dataset_dir,'results')
        args.task_file_name =f"{args.taskid}"
        args.save_dir = os.path.join(args.save_dir,args.task_file_name)    
    else:
        raise ValueError("dataset_name is not supported")

def save_args(args):
    with open(args.args_dir, 'wb') as f:
        pickle.dump(args, f)

def load_args(args_dir): 
    with open(args_dir, 'rb') as f:
        store_args = pickle.load(f)
    return store_args

def get_attentiveFP_config(args):
    args.node_feat_size = 26              
    args.edge_feat_size=4                     
    args.drug_num_layers=2                    
    args.num_timesteps=2                        
    args.graph_feat_size =512                  

def get_GATts_config(args):
    args.GATts_layers = 3
    args.GATts_indim = 7
    args.GATts_hdim = 5
    args.GATts_outdim = 1
    args.GATts_head = 3


if __name__=="__main__":
    args = get_config()
    print(args.save_dir)
