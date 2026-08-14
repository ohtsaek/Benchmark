# -*- coding: utf-8 -*-
# drug modules for structure and fusion
# AttentiveFP
# pylint: disable= no-member, arguments-differ, invalid-name
from _utils.model_utils import get_activation_function
import torch.nn as nn
from torch.nn import functional as F
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from layers.mlp_layer import NormLinear


# pylint: disable=W0221
class AttentiveFPPredictor(nn.Module):
    """AttentiveFP for regression and classification on graphs.

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for Drug
    Discovery with the Graph Attention Mechanism.
     <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """

    def __init__(self, args):
        super(AttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=args.node_feat_size,
                                  edge_feat_size=args.edge_feat_size,
                                  num_layers=args.drug_num_layers,
                                  graph_feat_size=args.graph_feat_size,
                                  dropout=args.dropout)
        self.readout = AttentiveFPReadout(feat_size=args.graph_feat_size,
                                          num_timesteps=args.num_timesteps,
                                          dropout=args.dropout)

    def forward(self, g, get_node_weight=False):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        get_node_weight : bool
            Whether to get the weights of atoms during readout. Default to False.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        node_feats, edge_feats = g.ndata.pop('hv'), g.edata.pop('he')
        graph_node_feats = self.gnn(g, node_feats, edge_feats)

        if get_node_weight:
            g_feats, node_weights = self.readout(g, graph_node_feats, get_node_weight)
            return g_feats, node_weights
        else:
            g_feats = self.readout(g, graph_node_feats, get_node_weight)
            return g_feats


class drug_MLP(nn.Module):
    def __init__(self, args):
        super(drug_MLP, self).__init__()
        activation = get_activation_function(args.activation)
        dropout = nn.Dropout(args.dropout)
        self.prediction = nn.Sequential(
            NormLinear(args.drug_feat_dim, 256, args.norm),
            activation,
            dropout,
            NormLinear(256, 128, args.norm),
            activation,
            dropout
        )

    def forward(self, x):
        return self.prediction(x)


