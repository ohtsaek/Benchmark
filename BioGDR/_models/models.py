# Model assembly

from _models.drug_modules import AttentiveFPPredictor, drug_MLP
from torch import nn
import torch
from _models.base_module import GAT, MLPtorch,graphIdentity

from _models.Interaction_modules import MLP_Attention, Transform_Embedding, graph_geneset_attention
from _utils.model_utils import get_activation_function, get_loss_func, build_optimizer
from _utils.utils import log_judge
from dgllife.model.readout import AttentiveFPReadout
from layers.mlp_layer import NormLinear


def get_model(args, feat_dict,logger=None):
    model = BioGDR(args, feat_dict)
    return model


class BioGDR(nn.Module):
    def __init__(self, args, feat_dict):
        super(BioGDR, self).__init__()
        self.args = args
        self.geneset_length = feat_dict.GeneSet_Length
        self.drug_dim = args.drug_feat_dim
        self.cell_feat_type,self.drug_feat_type = args.cell_feat_type,args.drug_feat_type
        self.get_attn = args.get_attn

        self.drug_module = drug_MLP(args)
        self.dfeat_extraction = mFU(args,feat_dict)
        self.get_BioG_cell_modules(args,feat_dict)
        self.tranform_embedding = Transform_Embedding(args, self.drug_dim, 32)  
        self.gene_attention = graph_geneset_attention(args, feat_dict, 32, 4)
        self.pathway_attention = MLP_Attention(args, 32, self.geneset_length)  
        self.get_pred_modules(args,feat_dict)
        

    def forward(self, drug_feature, cell_feature, pathways_g):
        drug_feature = self.dfeat_extraction(drug_feature) 
        drug_feats = self.drug_module(drug_feature)
        trans_drug_feats = self.tranform_embedding(drug_feature) 
        geneset_feats = self.batched_genesets_GNN(cell_feature, self.cell_feat_type)
        attn_geneset_feats , gene_attns = self.gene_attention(geneset_feats, trans_drug_feats, self.get_attn) 
        cell_feature.ndata[self.cell_feat_type] = attn_geneset_feats.reshape([-1, 1])
        pathways_feats  = self.readout(cell_feature,cell_feature.ndata[self.cell_feat_type])  
        pathways_g.ndata[self.cell_feat_type] = pathways_feats
        pathways_feats = self.pathways_GNN(pathways_g, self.cell_feat_type)
        pathways_feats = pathways_feats.reshape([pathways_g.batch_size, -1])
        attn_pathways_feats,pathway_attns = self.pathway_attention(trans_drug_feats, pathways_feats, self.get_attn)  
        concat_feats = torch.cat((drug_feats, attn_pathways_feats), 1)
        prediction = self.predictor(concat_feats)
        return prediction,[gene_attns,pathway_attns]  

    def get_BioG_cell_modules(self,args,feat_dict):
        self.gene_heads = ([args.gene_mid_heads] * (args.gene_layers - 1) + [args.gene_out_heads])
        self.batched_genesets_GNN = GAT(args,
                                        args.gene_layers,
                                        1,
                                        args.gene_hidden_dim,
                                        args.gene_out_dim,
                                        self.gene_heads,
                                        get_activation_function(args.activation),
                                        args.g_norm,
                                        residual=args.residual,
                                        feat_dict=feat_dict,
                                        in_drop=args.dropout,
                                        at_drop=args.dropout
                                        )  

        self.readout = AttentiveFPReadout(feat_size=args.gene_out_dim,    
                                          num_timesteps=args.gene_num_timesteps,
                                          dropout=args.dropout)

        self.pathways_GNN = graphIdentity(args.cell_feat_type)


    def get_pred_modules(self,args,feat_dict):

        self.pred_input_dim = 128 + self.geneset_length 
        activation = get_activation_function(args.activation)
        self.predictor = nn.Sequential(nn.Linear(self.pred_input_dim,1024),
                                           activation,
                                           nn.Linear(1024,1024),
                                           activation,
                                           nn.Linear(1024,512),
                                           activation,
                                           nn.Linear(512,1),
                                            )
        

class GATts(nn.Module):
    def __init__(self,args,feat_dict):
        super(GATts, self).__init__()
        self.args = args
        self.GATts_heads = [args.GATts_head] * args.GATts_layers
        self.GATts = GAT(args,
                        args.GATts_layers,
                        args.GATts_indim,
                        args.GATts_hdim,
                        args.GATts_outdim,
                        self.GATts_heads,
                        get_activation_function(args.activation),
                        args.g_norm,
                        residual=args.residual,
                        feat_dict=feat_dict,
                        in_drop=args.dropout,
                        at_drop=args.dropout
                        ) 

    def forward(self,drug_feature):
        new_drug_feature = self.GATts(drug_feature,'exp')
        new_drug_feature = new_drug_feature.reshape([-1,978])   
        return new_drug_feature


class mFU_GATts(nn.Module):
    def __init__(self, args, feat_dict,feat_type):
        super(mFU_GATts, self).__init__()
        self.GATts = GATts(args,feat_dict)  

    def forward(self,drug_feat):
        drug_feat = self.GATts(drug_feat)
        return drug_feat


class mFU(nn.Module):
    def __init__(self,args,feat_dict):
        super(mFU, self).__init__()
        self.args = args
        self.drug_feat_interaction = self.args.drug_feat_interaction
        self.linear_flag = 'linear' in args.drug_feat_interaction

        self.mFU_modules = nn.ModuleDict()
        self.get_mFU_modules(args,feat_dict)
        if self.linear_flag:
            self.mFU_transform_modules = nn.ModuleDict()
            self.get_mFU_transform_modules(args,feat_dict)

    def get_mFU_transform_modules(self,args,feat_dict):
        for drug_feat in self.args.drug_feat_types[1:]:
            if 'linear' in self.drug_feat_interaction:
                self.mFU_transform_modules[drug_feat] = nn.Linear(args.ori_dim_dict[drug_feat], args.single_drug_feat_dim)   #dimension transform
            else:
                self.mFU_transform_modules[drug_feat]=nn.Identity()

    def get_mFU_modules(self,args,feat_dict):
        for drug_feat in self.args.drug_feat_types[1:]:
            if drug_feat in ['GATtsall']:
                self.mFU_modules[drug_feat] = mFU_GATts(args, feat_dict,drug_feat)
            elif drug_feat in ['kinome']:   
                self.mFU_modules[drug_feat] = MLPtorch(args, args.ori_dim_dict[drug_feat], args.ori_dim_dict[drug_feat], args.ori_dim_dict[drug_feat], 3, False, True, 'same')  
            elif drug_feat == 'afp':
                self.mFU_modules[drug_feat] = AttentiveFPPredictor(args)
            else:
                raise ValueError(f'unsupported drug module {drug_feat}')
        
    def forward(self,drug_features):
        new_drug_features = []
        for i,drug_feat in enumerate(self.args.drug_feat_types[1:]):
            current_drug_feature = drug_features[i]
            new_drug_feature = self.mFU_modules[drug_feat](current_drug_feature) 
            if self.linear_flag:
                new_drug_feature = self.mFU_transform_modules[drug_feat](new_drug_feature)
                
            new_drug_features.append(new_drug_feature)

        final_drug_features = torch.concat(new_drug_features,dim = 1) 
        return final_drug_features







    

