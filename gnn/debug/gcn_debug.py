import math
from typing import Sequence

import torch
from torch import nn as nn
from torch.nn import Identity
from torch.nn import Parameter, Module
from torch.nn import functional as F, Linear, BatchNorm1d


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


class GCN(nn.Module):
    def __init__(self, num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.skip = skip
        self.batchnorm = batchnorm
        for i in range(num_layers - 1):
            if i == 0:
                self.add_module(f"layer{i}", GraphConv(Linear(nfeat, nhid, bias=False)))
                if batchnorm:
                    self.add_module(f"batchnorm{i}", BatchNorm1d(nhid))
            else:
                nhid_in = nhid * 2 if skip else nhid
                out = classes if i == num_layers - 1 else nhid
                self.add_module(f"layer{i}", GraphConv(Linear(nhid_in, out, bias=False)))
                if batchnorm:
                    self.add_module(f"batchnorm{i}", BatchNorm1d(out))
        nhid_in = nhid * 2 if skip else nhid
        self.add_module(f"layer{num_layers - 1}", GraphConv(Linear(nhid_in, classes, bias=False)))

    def forward(self, adj, x, **kwargs):
        for i in range(self.num_layers - 1):
            x_ = self._modules[f"layer{i}"](x=x, adj=adj)
            if self.batchnorm:
                x_ = self._modules[f"batchnorm{i}"](x_)
            x = F.relu(x_)
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip:
                x = torch.cat([x, x_], dim=-1)
        x = self._modules[f"layer{self.num_layers - 1}"](x=x, adj=adj)
        return x

    def reset_parameters(self):
        for i in range(self.num_layers):
            self._modules[f"layer{i}"].reset_parameters()
            if self.batchnorm and i < self.num_layers - 1:
                self._modules[f"batchnorm{i}"].reset_parameters()
