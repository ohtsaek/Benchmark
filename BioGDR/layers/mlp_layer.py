import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLinear(nn.Module):
    def __init__(self, in_feat, out_feat, norm=None, activation=None,bias=True,  device=None,
                 dtype=None):
        super().__init__()

        self.activation = activation

        self.Linear = nn.Linear(in_feat, out_feat, bias,
                                device, dtype)

        if norm:
            self.norm = nn.BatchNorm1d(out_feat)
        else:
            self.norm = None

    def forward(self, feats):
        out = self.Linear(feats)
        if self.norm is not None:
            out = self.norm(out)
        return out
