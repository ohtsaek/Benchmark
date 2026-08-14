# individual evaluation
import os
import time
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from configuration.config import get_device_config, load_args, save_args
from configuration.path_config import get_path_config
from evaluation import make_prediction
from _models.models import get_model
from _utils.csv_dataset import ResponseDataset
from _utils.data_utils import feat_dict
from _utils.model_utils import get_loss_func
from _utils.utils import get_collate_fn, log_judge, makedirs, redirect_log, drug_feat_to_device


def target_eval(args, feat_dict,logger=None):
    debug, info = log_judge(logger)
    test_dataset = ResponseDataset(args, args.target_dataset_path, feat_dict)  
    collate_fn = get_collate_fn(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),   
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    model = get_model(args, feat_dict)

    args.drug_feat_func = drug_feat_to_device(args)

    args.loss_fn = get_loss_func(args)
    args.feat_dict = feat_dict

    debug(f'Moving model to device: {args.device}')
    model.load_state_dict(torch.load(args.best_model_path,map_location='cpu'))
    model = model.to(args.device,non_blocking=True)

    test_results, test_loss = make_prediction(args, model, test_loader, data_type='eval', save_preds=args.save_preds,
                                              get_attns=args.get_attn)


def parse_eval_args():
    parser = ArgumentParser()
    parser.add_argument('--cpu',action='store_true',default=False,help="Use cpu device")
    parser.add_argument('--gpu', type=int,default=0,help='Which GPU to use')
    parser.add_argument('--result_dir', type=str,default='data/Example_train/results/',
                        help='the result using for analysis')
    parser.add_argument('--target_data_dir', type=str,default='data/Example_eval',
                        help='the result using for analysis')
    parser.add_argument('--batch_size', type=int, default=256,help='The batch size')
    parser.add_argument('--save_dir',type=str,default=None,help='Specify the save path. If not set, defaults to the dataset directory.')    
    parser.add_argument('--num_workers', type=int, default=0,help='The number of workers')
    parser.add_argument('--drop_last', action='store_true', default=False, help='Drop last incomplete DataLoader batch (e.g. avoid BN batch=1)')
    parser.add_argument('--get_attn',action='store_true',default=False,help = 'get attention score for gene and pathways')
    parser.set_defaults(ind_eval=True)
    args = parser.parse_args()
    return args

def modify_ori_args_by_eval_args(ori_args,eval_args):
    #modifiy ori_args by eval_args
    ori_args.cpu = eval_args.cpu
    ori_args.gpu = eval_args.gpu
    ori_args.result_dir=eval_args.result_dir
    ori_args.batch_size=eval_args.batch_size
    ori_args.num_workers=eval_args.num_workers
    ori_args.get_attn=eval_args.get_attn
    ori_args.ind_eval = eval_args.ind_eval
    ori_args.drop_last = eval_args.drop_last


if __name__ == "__main__":

    # Get and modify evaluation parameters
    eval_args = parse_eval_args()
    target_data_dir=eval_args.target_data_dir  #dataset for evaluation
    args_path = os.path.join(eval_args.result_dir,'args.pickle')
    ori_args = load_args(args_path)
    ori_args.args_dir = args_path
    modify_ori_args_by_eval_args(ori_args,eval_args)

    #get path
    ori_args.dataset_dir = target_data_dir
    get_path_config(ori_args)   
    if getattr(ori_args, "target_dataset_path", None) is None:
        ori_args.target_dataset_path = os.path.join(target_data_dir,"data.csv")
    get_device_config(ori_args)

    #model_path
    ori_args.mode = 'eval'
    ori_args.fttaskid = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    ori_args.best_model_path = os.path.join(eval_args.result_dir,'model','best_model_param.pickle')

    #save_path
    model_file_name = os.path.basename(os.path.normpath(eval_args.result_dir))
    if eval_args.save_dir is None:
        ori_args.save_dir =  os.path.join(target_data_dir,'results',model_file_name+"_"+ori_args.fttaskid) 
    else:
        ori_args.save_dir = eval_args.save_dir
    ori_args.args_dir = os.path.join(ori_args.save_dir,'args.pickle')

   #evaluation     
    every_feat_dict = feat_dict(ori_args)  
    makedirs(ori_args.save_dir)
    save_args(ori_args)
    logger = redirect_log(ori_args, False) 
    debug, info = log_judge(logger)
    target_eval(ori_args,every_feat_dict,logger)
    print("Evaluation completed, results saved in ", ori_args.save_dir)
