#attention module for Drug guided attention strategy
import dgl
import torch
import torch.nn as nn
from _utils.model_utils import get_activation_function
from layers.mlp_layer import NormLinear
import numpy as np



class Transform_Embedding(nn.Module):
    def __init__(self, args, source_len, after_len):
        super(Transform_Embedding, self).__init__()
        activation = get_activation_function(args.activation)
        dropout = nn.Dropout(args.dropout)
        self.transform = nn.Sequential(
            NormLinear(source_len, 128, args.norm),
            activation,
            dropout,
            NormLinear(128, after_len, args.norm),
            activation,
            dropout
        )

    def forward(self, source):
        return self.transform(source)



class MLP_Attention(nn.Module):
    def __init__(self, args, source_len, target_len, source_attn_ratio=16):  
        super(MLP_Attention, self).__init__()

        activation = get_activation_function(args.activation)
        self.dropout = nn.Dropout(args.dropout)
        self.source_len = source_len
        self.target_len = target_len

        self.source_transform_dim = int(self.target_len / source_attn_ratio) + 1  
        self.source_transform_layer = nn.Sequential(
            NormLinear(self.source_len, self.source_transform_dim, args.norm),
            activation
        )

        attn_activation = get_activation_function('tanh')

        self.concat_input_dim = self.target_len + self.source_transform_dim
        self.attention = nn.Sequential(
            NormLinear(self.concat_input_dim, self.target_len, args.norm),
            attn_activation,
            nn.Softmax(dim=-1)
        )

    def forward(self, source, target, get_attention_score=False):
        source_embedding = self.source_transform_layer(source)
        concat = torch.cat([target, source_embedding], dim=1)
        attention = self.attention(concat)
        attended_target = torch.mul(attention, target)
        attended_target = self.dropout(attended_target)
        if get_attention_score:   
            return attended_target, attention
        else:
            return attended_target,None


class graph_geneset_attention(nn.Module):
    def __init__(self, args, feat_dict, source_len, source_attn_ratio=4):
        super(graph_geneset_attention, self).__init__()
        self.args = args
        self.GeneSet_Dict = feat_dict.GeneSet_Dict
        self.GeneSet_Length = feat_dict.GeneSet_Length
        self.geneset_cnums = [0]
        c = 0
        for name, geneset in self.GeneSet_Dict.items():
            gs_num = len(geneset)
            self.add_module(name + '_MLP_attn_layer',
                            MLP_Attention(args, source_len, gs_num, source_attn_ratio)) 
            c += gs_num
            self.geneset_cnums.append(c)

    def forward(self, target_feat, t_source, get_attention=False):
        batchsize = t_source.size(0)  
        target_feat = target_feat.reshape([batchsize, -1])

        attn_targets = [] 
        attn_scores = []  
        for i, name in enumerate(self.GeneSet_Dict.keys()):
            current_feat = target_feat[:, self.geneset_cnums[i]:self.geneset_cnums[i + 1]]
            attn_target, attn_score = self._modules[name + '_MLP_attn_layer'](t_source, current_feat, get_attention)

            attn_targets.append(attn_target)
            attn_scores.append(attn_score)  

        attn_targets = torch.cat(attn_targets, dim=-1) 

        if get_attention: 
            attn_scores = torch.cat(attn_scores, dim=-1) 
            return attn_targets, attn_scores
        else:
            return attn_targets,None
