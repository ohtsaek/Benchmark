# Build input data and features
import sys
from argparse import Namespace
from logging import Logger
import os
import csv
from tqdm import tqdm
from collections import OrderedDict
import torch
import copy
import numpy as np
import pandas as pd
from _utils.utils import log_judge
import dgl
from _utils.fea import construct_bigraph_from_smiles, featurize_atoms
import pickle


def construct_pathway_graph(gene_gene_r, pathway_exp_df):
    g = dgl.graph(([], []), idtype=torch.int32)
    num_nodes = len(pathway_exp_df.columns)

    g.add_nodes(num_nodes)

    src_list = gene_gene_r[0]+gene_gene_r[1]   #No bidirectional edges in gene file, manually set bidirectional edges
    dst_list = gene_gene_r[1]+gene_gene_r[0]
    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

    return g


class feat_dict:  
    def __init__(self, args: Namespace, logger: Logger = None):
        self.args = args
        self.debug, self.info = log_judge(logger)

        self.cache_path = os.path.join(args.dataset_dir, "cache",'feat_dict.pkl')
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
            self.args = args
            self.debug, self.info = log_judge(logger)
            return

        self.get_dict()
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self, f)

    def get_dict(self):
        self.get_drug_features()
        self.get_batched_cl_features()  
        self.get_pathways_features()

    def get_batched_cl_features(self):

        GeneSet_List = []
        with open(self.args.geneset_path) as f:
            data = list(list(rec) for rec in csv.reader(f, delimiter='\t')) 
            for row in data:
                GeneSet_List.append(row[0])

        self.GeneSet_List = GeneSet_List  
        self.GeneSet_Length = len(GeneSet_List) 

        pathway_graphs = [] 
        self.GeneSet_Dict = OrderedDict()  
        GeneSet_df_Dict = OrderedDict() 
        for i, name in enumerate(GeneSet_List):
            gene_gene_r_df = pd.read_csv(os.path.join(self.args.geneset_feature_path, str(i) + name + '_idrels.csv'),
                                         index_col=0) 
            
            pathway_exp_df = pd.read_csv(os.path.join(self.args.geneset_feature_path, str(i) + name + '.csv'),
                                         index_col=0)

            self.GeneSet_Dict[name] = list(pathway_exp_df.columns)  
            GeneSet_df_Dict[name] = pathway_exp_df 

            gene_gene_r = [list(gene_gene_r_df['id1']), list(gene_gene_r_df['id2'])]
            g = construct_pathway_graph(gene_gene_r, pathway_exp_df)
            if self.args.add_self_loop:
                g = dgl.add_self_loop(g)
            pathway_graphs.append(g)

        self.pathway_graphs = pathway_graphs
        self.cell_features = OrderedDict()
        self.cl_list = list(pathway_exp_df.index)  
        for cell_identifier in tqdm(self.cl_list):
            current_gs = copy.deepcopy(pathway_graphs)
            for i, name in enumerate(GeneSet_List):
                cell_pathway_exp = torch.tensor(GeneSet_df_Dict[name].loc[cell_identifier, :], dtype=torch.float32).unsqueeze(
                    -1)
                current_gs[i].ndata['exp'] = cell_pathway_exp  
            self.cell_features[cell_identifier] = dgl.batch(current_gs) 
 
    
    def get_pathways_features(self):
        pathways_g = dgl.graph(([], []), idtype=torch.int32)
        pathways_g.add_nodes(len(self.GeneSet_List))

        if self.args.add_self_loop:
            pathways_g = dgl.add_self_loop(pathways_g)
        self.pathways_g = pathways_g

        self.pathway_features = OrderedDict()  
        for cell_identifier in tqdm(self.cl_list):
            self.pathway_features[cell_identifier] = copy.deepcopy(pathways_g)

    def get_drug_features(self):
        self.get_drug_fusion_features()
    def get_drug_single_features(self,drug_feat_type):
        self.drug_features = OrderedDict()
        if drug_feat_type == 'afp':
            self.get_drug_AFP_features()
        elif drug_feat_type == 'kinome':
            self.get_drug_kinome_features()
        elif drug_feat_type == 'GATtsall':
            self.get_drug_transcript_features(drug_feat_type)

    def get_drug_fusion_features(self):
        drug_feat_list = []
        temp_drug_path_dict = copy.deepcopy(self.args.drug_path)  
        for drug_feat in self.args.drug_feat_types[1:]:
            self.args.drug_path = temp_drug_path_dict[drug_feat]
            self.get_drug_single_features(drug_feat)
            drug_feat_list.append(copy.deepcopy(self.drug_features))
        self.args.drug_path = temp_drug_path_dict
        final_drug_feat = OrderedDict()
        co_drugids = set.intersection(*[set(i.keys()) for i in drug_feat_list])
        for k in co_drugids:
            final_drug_feat[k] = [f[k] for f in drug_feat_list]  
        self.drug_features = final_drug_feat


    def get_drug_transcript_features(self,drug_feat_type):
        drug_info = pd.read_csv(self.args.drug_path)
        drug_features = drug_info.iloc[:,1:]
        self.get_drug_ppi_graph(drug_feat_type)

        for index in tqdm(drug_info.index,file=sys.stderr):
            drug_id = drug_info.at[index, self.args.drug_idname]
            c_feats= drug_features.loc[index]
            c_feats = torch.tensor(np.array(c_feats), dtype=torch.float32)

            current_drug_ppi_g= copy.deepcopy(self.drug_ppi_g)
            current_drug_ppi_g.ndata['exp'] = c_feats.reshape(-1,978).transpose(0,1)
            self.drug_features[drug_id] = current_drug_ppi_g


    def get_drug_ppi_graph(self,feat_type):
        drug_ppi = pd.read_csv(self.args.drug_ppi_path)
        drug_ppi_g = dgl.graph(([], []), idtype=torch.int32)
        drug_ppi_g.add_nodes(self.args.ori_dim_dict[feat_type])
        src_list = list(drug_ppi['id1'])
        dst_list = list(drug_ppi['id2']) 
        drug_ppi_g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))
        if self.args.add_self_loop:
            drug_ppi_g = dgl.add_self_loop(drug_ppi_g)
        self.drug_ppi_g = drug_ppi_g

    def get_drug_kinome_features(self):
        drug_info = pd.read_csv(self.args.drug_path)
        kinome_features = drug_info.iloc[:, 1:]
        for index in tqdm(drug_info.index,file=sys.stderr):
            drug_id = drug_info.at[index, self.args.drug_idname]
            c_feats= kinome_features.loc[index] 
            self.drug_features[drug_id] = torch.tensor(np.array(c_feats), dtype=torch.float32)


    def get_drug_AFP_features(self):
        drug_full_info = pd.read_csv(self.args.drug_path)
        drug_list = list(set(list(drug_full_info[self.args.drug_idname])))
        for index in tqdm(drug_full_info.index):
            drug_id = drug_full_info.at[index, self.args.drug_idname]
            smi = drug_full_info.at[index, 'smiles']  
            if drug_id in drug_list and drug_id not in self.drug_features: 
                self.drug_features[drug_id] = construct_bigraph_from_smiles(smi, featurize_atoms)

    @property
    def drug_dim(self):
        if self.args.drug_feat_type != 'AFP':
            return len(list(self.drug_features.values())[0])
        else:
            return self.args.graph_feat_size
    

