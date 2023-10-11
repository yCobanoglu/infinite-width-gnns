import math
from typing import Sequence

import torch
from torch import nn as nn
from torch.nn import Identity
from torch.nn import Parameter, Module
"""
# uncomment to use with figures/main.py
class GraphConv(Module):
    def __init__(self, layer, out=None, bias=True):
        super().__init__()
        if layer is None:
            self.layer = Identity()
        self.layer = layer
        if out is None:
            if isinstance(layer, Sequence):
                self.in_ = self.layer[-1].in_features
                self.out = self.layer[-1].out_features
            else:
                self.in_ = self.layer.in_features
                self.out = self.layer.out_features
        else:
            self.out = out

        self.layer = torch.nn.init.kaiming_normal_(Parameter(torch.FloatTensor(self.in_, self.out)), mode="fan_out")

        self.bias = None


    def forward(self, x, adj, *args, **kwargs):
        support = x @ self.layer
        if self.bias is None:
            self.bias = torch.nn.init.kaiming_normal_(Parameter(torch.FloatTensor(adj.shape[0], self.out)), mode="fan_out").to(x.device)
        #adj = torch.nn.functional.dropout(adj.to_dense(), 0.5, training=self.training)
        out = adj @ support
        out = out + self.bias
        return out
"""
class GraphConv(Module):
    def __init__(self, layer, out=None, bias=True):
        super().__init__()
        if layer is None:
            self.layer = Identity()
        self.layer = layer
        if out is None:
            if isinstance(layer, Sequence):
                self.out = self.layer[-1].out_features
            else:
                self.out = self.layer.out_features
        else:
            self.out = out

        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, x, adj, *args, **kwargs):
        support = self.layer(x)
        out = torch.spmm(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.layer.reset_parameters()


class Rank1Linear(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.weights = torch.nn.init.kaiming_uniform_(Parameter(torch.Tensor(1, out_features)))

    def forward(self, input):
        return input @ self.weights.broadcast_to(input.shape[1], self.weights.shape[1])
