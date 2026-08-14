# -*- coding: utf-8 -*-
# Main entry for model training.
import sys
import random
from tqdm import tqdm
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from collections import defaultdict
from _utils.csv_dataset import ResponseDataset
from _utils.utils import  get_collate_fn, log_judge, create_save_dir,redirect_log,drug_feat_to_device
from _utils.data_utils import feat_dict
from _utils.metrics_utils import optimization_direction
from _utils.model_utils import get_loss_func, build_optimizer
from _models.models import get_model
from evaluation import make_prediction,eval,ind_eval
from configuration.config import get_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main function for model training  
def train_main(args, use_feat_dict, logger=None):
    debug, info = log_judge(logger)
    debug('start training')
    debug('Loading data')
    train_dataset = ResponseDataset(args,args.train ,use_feat_dict, logger=logger)  
    val_dataset = ResponseDataset(args,args.valid ,use_feat_dict, logger=logger)

    collate_fn = get_collate_fn(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),     
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.drop_last,
    )

    # Initialize model
    model = get_model(args, use_feat_dict,logger)
    args.drug_feat_func = drug_feat_to_device(args)

    # loss function and optimizer
    args.loss_fn = get_loss_func(args)
    args.optimizer = build_optimizer(model, args)

    debug(f'Moving model to device: {args.device}')
    model = model.to(args.device,non_blocking=True)

    # early stopping
    best_epoch = 0
    opt_d = optimization_direction[args.metric]
    if opt_d == 1:
        best_val = -np.inf
    else:
        best_val = np.inf

    metric_dict = defaultdict(list)
    metrics_func = args.metric_func
    start_epoch = 0

    for epoch in tqdm(range(start_epoch, args.epochs), file=sys.stderr):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch_id, batch_data in enumerate(train_loader):
            drug_id, cell_id, drug_feats, CL_feats, pathways_g, labels = batch_data
            labels,  CL_feats,pathways_g =  labels.to(args.device,non_blocking=True),CL_feats.to(args.device,non_blocking=True),pathways_g.to(args.device,non_blocking=True)
            drug_feats = args.drug_feat_func(drug_feats,args.device)   
            prediction,_ = model(drug_feats, CL_feats,pathways_g) 
            loss = args.loss_fn(prediction,labels)
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()
            train_loss_sum += loss.item()
            train_steps += 1
            del drug_feats, CL_feats, pathways_g, labels, prediction, loss

        if args.device.type == 'cuda':
            torch.cuda.empty_cache()
        epoch_train_loss = train_loss_sum / train_steps

        args.feat_dict = use_feat_dict 
        epoch_val_results,epoch_val_loss = make_prediction(args, model, val_loader, data_type='val',save_preds=False)
        epoch_val_scores = eval(args,epoch_val_results, metrics_func,epoch_val_loss)


        if (opt_d * epoch_val_scores[args.metric] > opt_d * best_val) or (epoch == 0):
            best_val, best_epoch = epoch_val_scores[args.metric], epoch
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model_param.pickle"))


        metric_dict['epoch'].append(epoch)
        metric_dict['best_epoch'].append(best_epoch)
        metric_dict[f'best_val_{args.metric}'].append(best_val)

        metric_dict['train_loss'].append(epoch_train_loss)
        for metric in metrics_func:
            metric_dict['val_' + metric].append(epoch_val_scores[metric])

        metric_csv = pd.DataFrame.from_dict(metric_dict).round(4)
        metric_csv.to_csv(os.path.join(args.save_dir, f'metrics.csv'), index=False)


        if args.early_stop_epoch > 0 and (epoch - best_epoch >= args.early_stop_epoch):
            break

    args.best_epoch = best_epoch
    ind_eval(args,use_feat_dict,model)


if __name__ == "__main__":
    try:
        args = get_config()
        set_seed(args.init_seed)
        logger = redirect_log(args,False)   
        create_save_dir(args)
        use_feat_dict = feat_dict(args,logger)  
        train_main(args, use_feat_dict,logger)
        print("Training completed, results saved in ", args.save_dir)
    except Exception as e:
        print(e)
        sys.exit(1)
