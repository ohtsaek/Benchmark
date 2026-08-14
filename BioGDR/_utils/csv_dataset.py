# -*- coding: utf-8 -*-
# Build dataset
import math
import sklearn
import numpy as np
import os
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import torch
from _utils.utils import log_judge
import pickle
from sklearn.model_selection import KFold, GroupKFold
from sklearn import preprocessing

class ResponseDataset(object):
    def __init__(self, args, data_path, feat_dict, logger=None):
        self.debug, self.info = log_judge(logger)

        self.args = args
        self.data_path = data_path
        self.get_data()
        self.get_dict(feat_dict)

    def get_dict(self, feat_dict):
        self.cell_features = feat_dict.cell_features
        self.drug_features = feat_dict.drug_features
        self.pathway_features = feat_dict.pathway_features

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.labels)

    def __getitem__(self, item):
        drug_id = self.drug_identifiers[item]
        cell_id = self.cl_identifiers[item]

        return  drug_id, cell_id, self.drug_features[drug_id], self.cell_features[cell_id], self.pathway_features[cell_id],self.labels[item]


    def get_data(self):
        if isinstance(self.data_path,pd.DataFrame): 
            df = self.data_path
        else:
            df = pd.read_csv(self.data_path)  

        self.cl_identifiers = df[self.args.cl_idname].tolist()
        self.drug_identifiers = df[self.args.drug_idname].tolist()
        self.labels = torch.tensor(np.array(df[[self.args.label]]).astype(np.float32),
                                   dtype=torch.float32)

