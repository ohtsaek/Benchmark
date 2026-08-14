# Other components 
# -*- coding: utf-8 -*-
import sys

import dgl
import numpy as np
import random
import torch
import logging
import os
import hashlib
from collections import OrderedDict
from typing import Dict, List, Union

from _utils.metrics_utils import get_metric_func
from torch import multiprocessing

def _get_hash(key_tuple):
    _hash_func = hashlib.sha1()
    _hash_func.update(str(key_tuple).encode('utf-8'))
    return _hash_func.hexdigest()[:20]


def log_judge(logger):
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    return debug, info


def scoring(y: np.array, y_pred: np.array, dataset_type: str, metrics_func: Union[List, str] = 'default') -> Dict:
    """
    Calculate the value of each metrics based on the provided metrics and model type
    :param y:True values,numpy array
    :param y_pred:prediction values，numpy array
    :param dataset_type:‘classification’ or 'regression'
    :param metrics_func:the metrics need to be computed
    :return: a dict of every metrics
    """
    if metrics_func == 'default':
        if dataset_type == 'classification':
            metrics_func = ['roc_auc', 'matthews_corrcoef', 'recall', 'precision', 'specificity', 'prc_auc',
                            'balanced_accuracy']
        elif dataset_type == 'regression':
            metrics_func = ['rmse', 'mae', 'pearson', 'spearman']
    else:
        if not isinstance(metrics_func, List): 
            metrics_func = [metrics_func]
    return OrderedDict({m: float(get_metric_func(m)(y, y_pred)) for m in metrics_func})


def get_collate_fn(args):
    return GATtsall_kinome_afp_collate_fn

def GATtsall_kinome_afp_collate_fn(data):
    drug_id, cell_id, drug_features, cell_features, pathways_g,labels = map(list, zip(*data)) 
    GATts_feats,kinome_feats,afp_feats = map(list, zip(*drug_features))
    bdrug_features = [dgl.batch(GATts_feats),torch.stack(kinome_feats,dim=0),dgl.batch(afp_feats)]
    labels = torch.stack(labels, dim=0)
    bcell_features = dgl.batch(cell_features)
    bpathways_g  = dgl.batch(pathways_g)
    return drug_id, cell_id, bdrug_features, bcell_features,bpathways_g, labels


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False   

    fmt = "%(asctime)s PID%(process)d %(filename)s line%(lineno)d %(levelname)s | %(message)s"
    datefmt = '%Y%m%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt)

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler(stream=sys.stdout)
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_v.setFormatter(formatter)

        logger.addHandler(fh_v)

    return logger


class OutputWrapper():
    def __init__(self,file_path,if_flush=False,encoding = "utf-8",mode = 'a+'):
        self.file = open(file_path,mode=mode,encoding=encoding)
        self._if_flush = if_flush

        self.flush = self.file.flush

    def write(self, data):
        self.file.write(data)

        if self._if_flush:
            self.file.flush()


def redirect_log(args,redirect = False):
    if redirect:
        sys.stdout = OutputWrapper(os.path.join(args.save_dir,'stdout.log'), True)
        sys.stderr = OutputWrapper(os.path.join(args.save_dir,'stderr.log'), True)
    logger = create_logger(name='train', save_dir=None, quiet=args.quiet)
    return logger

def list_to_device(feats,device):
    feats = [feat.to(device,non_blocking=True) for feat in feats ]
    return feats

def drug_feat_to_device(args):
    return list_to_device


def create_save_dir(args):
    makedirs(args.save_dir)
    args.model_dir = os.path.join(args.save_dir, 'model')
    makedirs(args.model_dir)

