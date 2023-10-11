import torch
from torch import nn as nn
from torch.nn import functional as F, Linear, BatchNorm1d
from gnn.models.modules import GraphConv
from gnn.my_selectors.select_nonlinearity import select_nonlinearity


class GCN(nn.Module):
    def __init__(self, num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=False, nonlinearity="relu"):
        super().__init__()
        if skip is True and num_layers <= 2:
            raise ValueError("Skip connections require at least 3 layers")
        self.nonlinearity = select_nonlinearity(nonlinearity)
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
            pre_nonlinear = x_
            if self.batchnorm:
                x_ = self._modules[f"batchnorm{i}"](x_)
            x_ = self.nonlinearity(x_)
            if self.dropout:
                x_ = F.dropout(x_, p=self.dropout, training=self.training)
            if self.skip:
                x_ = torch.cat([x_, pre_nonlinear], dim=-1)
            x = x_
        return self._modules[f"layer{self.num_layers - 1}"](x=x, adj=adj)

    def reset_parameters(self):
        for i in range(self.num_layers):
            self._modules[f"layer{i}"].reset_parameters()
            if self.batchnorm and i < self.num_layers - 1:
                self._modules[f"batchnorm{i}"].reset_parameters()
