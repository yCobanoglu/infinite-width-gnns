import math
from typing import Sequence

import torch
from torch import nn as nn
from torch.nn import Identity, LazyLinear
from torch.nn import Parameter, Module
from torch.nn import functional as F, Linear, BatchNorm1d


class FCN1(nn.Module):
    def __init__(self, num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=False):
        super().__init__()
        self.l1 = Linear(nfeat, nhid, bias=True)
        self.b1 = BatchNorm1d(nhid)
        self.l3 = Linear(nhid, classes, bias=True)

    def forward(self, x, **kwargs):
        x = self.l1(x)
        x = self.b1(x)
        x = F.sigmoid(x)
        x = self.l3(x)
        return x


class FCN(nn.Module):
    def __init__(self, num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=False, nonlinearity=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.skip = skip
        self.nonlinearity = nonlinearity
        self.batchnorm = batchnorm
        for i in range(num_layers - 1):
            if i == 0:
                self.add_module(f"layer{i}", Linear(nfeat, nhid, bias=True))
                if batchnorm:
                    self.add_module(f"batchnorm{i}", BatchNorm1d(nhid))
            else:
                nhid_in = nhid * 2 if skip else nhid
                out = classes if i == num_layers - 1 else nhid
                self.add_module(f"layer{i}", LazyLinear(out, bias=True))
                if batchnorm:
                    self.add_module(f"batchnorm{i}", BatchNorm1d(out))
        nhid_in = nhid * 2 if skip else nhid
        self.add_module(f"layer{num_layers - 1}", LazyLinear(classes, bias=True))

    def forward(self, x, **kwargs):
        for i in range(self.num_layers - 1):
            x_ = self._modules[f"layer{i}"](x)
            if self.batchnorm:
                x_ = self._modules[f"batchnorm{i}"](x_)
            x_ = self.nonlinearity(x_)
            if self.dropout:
                x_ = F.dropout(x_, p=self.dropout, training=self.training)
            if self.skip:
                x_ = torch.cat([x, x_], dim=-1)
            x = x_
        x = self._modules[f"layer{self.num_layers - 1}"](x)
        return x

    def forward1(self, x, **kwargs):
        for i in range(self.num_layers - 1):
            x_ = self._modules[f"layer{i}"](x)
            x_pre_nonlinear = x_
            if self.batchnorm:
                x_ = self._modules[f"batchnorm{i}"](x_)
            if self.nonlinearity:
                x_ = F.relu(x_)
            if self.dropout:
                x_ = F.dropout(x_, p=self.dropout, training=self.training)
            if self.skip:
                x_ = torch.cat([x_pre_nonlinear, x_], dim=-1)
            x = x_
        x = self._modules[f"layer{self.num_layers - 1}"](x)
        return x
