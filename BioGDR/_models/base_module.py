# Fundamental Building Blocks for GE,cell module and KI
from torch import nn
from dgl.nn.pytorch.conv import GATConv
import torch
from _utils.model_utils import get_activation_function
from layers.gat_layer import GATLayer
from layers.mlp_layer import NormLinear


class GAT(nn.Module):  
    def __init__(self, args, 
                 n_layers,  
                 in_feats,  
                 n_hidden,  
                 out_dim,
                 heads,  
                 activation,  
                 norm=None,
                 residual=False,
                 feat_dict=None,
                 in_drop=0,  
                 at_drop=0,  
                 negative_slope=0.2, 
                 ):
        super(GAT, self).__init__()

        self.args = args

        self.num_layers = n_layers
        self.activation = activation   

        self.gat_layers = nn.ModuleList()

        self.in_feats = in_feats 

        if n_layers == 1:
            self.gat_layers.append(GATLayer(
                in_feats, out_dim, heads[0],
                in_drop, at_drop, norm, negative_slope, residual, activation=self.activation))

        else:
            self.gat_layers.append(GATLayer(
                in_feats, n_hidden, heads[0],
                in_drop, at_drop, norm, negative_slope, residual, activation=self.activation))

            if n_layers > 2:
                for n in range(1, n_layers - 1):
                    self.gat_layers.append(GATLayer(
                        n_hidden * heads[n - 1], n_hidden, heads[n],
                        in_drop, at_drop, norm, negative_slope, residual, activation=self.activation))

            self.gat_layers.append(GATLayer(
                n_hidden * heads[-2], out_dim, heads[-1],
                in_drop, at_drop, norm, negative_slope, residual, activation=self.activation))


    def forward(self, cell_features, feat_name='exp', drug_features=None,
                get_attn=False):  
        self.g = cell_features
        h = self.g.ndata[feat_name]
        attns = []

        for l in range(self.num_layers - 1):
            h, attn = self.gat_layers[l](self.g, h, get_attn)
            h = h.flatten(1)
            attns.append(attn)
        h, attn = self.gat_layers[-1](self.g, h, get_attn)
        h = h.mean(1)
        attns.append(attn)

        if get_attn:
            return h, attns
        else:
            return h


class MLPtorch(nn.Module):  
    def __init__(self, args, input_dim, out_dim, hidden_dim, num_layers,start_dropout=False, end_activation=False,hidden_way = 'half'):
        super(MLPtorch, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  
        self.start_dropout = start_dropout
        self.end_activation = end_activation
        self.hidden_way = hidden_way

        self.prediction = self.create_ffn(args)

    def create_ffn(self, args):
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        if self.start_dropout:
            ffn = [dropout]
        else:
            ffn = []

        if self.num_layers == 1:
            ffn.extend([NormLinear(self.input_dim, self.out_dim,args.norm)])
        else:
            ffn.extend([NormLinear(self.input_dim, self.hidden_dim,args.norm)])
            for _ in range(self.num_layers - 2):
                if self.hidden_way == 'half':
                    next_hidden_dim = int(0.5 * self.hidden_dim)
                elif self.hidden_way == 'same':
                    next_hidden_dim = self.hidden_dim
                else:
                    raise ValueError('the transformed method not supported')

                ffn.extend([
                    activation,
                    dropout,
                    NormLinear(self.hidden_dim, next_hidden_dim,args.norm),
                ])
                self.hidden_dim = next_hidden_dim
            ffn.extend([
                activation,
                dropout,
                NormLinear(self.hidden_dim, self.out_dim, args.norm)
            ])

        if self.end_activation:
            if isinstance(self.end_activation, str):
                ffn.append(get_activation_function(self.end_activation))
            else:
                ffn.extend([activation,dropout])
        # Create FFN model
        return nn.Sequential(*ffn)  

    def forward(self, x):
        return self.prediction(x)


class graphIdentity(nn.Module):
    def __init__(self,*args, **kwargs):
        super(graphIdentity, self).__init__()
        self.Identity = torch.nn.Identity()

    def forward(self,g,feat_name):
        return self.Identity(g.ndata[feat_name])
    




