import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
from _norm.norm import LoadNorm, normalize
from dgl.nn.pytorch.utils import Identity

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""


class GATLayer(nn.Module):
    """
    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        
    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """

    def __init__(self, in_dim, out_dim, num_heads, in_drop=0., at_drop=0., norm=None, negative_slope=0.2,
                 residual=False,
                 activation=None, allow_zero_in_degree=False,
                 bias=True):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.norm = norm


        self.gatconv = GATConv(in_dim, out_dim, num_heads, in_drop, at_drop, negative_slope, residual, None,
                               allow_zero_in_degree, bias)  

        self.batchnorm_h = LoadNorm(self.norm, out_dim * num_heads, is_node=True)

    def forward(self, g, h, get_attn=False):

        h, attn = self.gatconv(g, h, True)
        size = h.size()
        h = h.flatten(1)

        if self.norm is not None:
            h = normalize(self.batchnorm_h, h, g)

        if self.activation:
            h = self.activation(h)


        h = h.reshape(size)
        if get_attn:
            return h, attn
        else:
            return h, None
