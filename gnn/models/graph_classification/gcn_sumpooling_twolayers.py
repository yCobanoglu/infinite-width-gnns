import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear

from gnn.models.modules import GraphConv


class GCN_SumPooling_TwoLayers(nn.Module):
    """GIN"""

    def __init__(self, num_node_features, dim_h, num_classes, dropout):
        super().__init__()
        self.dropout = dropout
        self.gc1 = GraphConv(Linear(num_node_features, dim_h, bias=False))
        self.gc2 = GraphConv(Linear(dim_h, dim_h, bias=False))

        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, num_classes)

    def forward(self, x, adj, **kwargs):
        h1 = self.gc1(x, adj)
        h1 = h1.relu()
        h2 = self.gc2(h1, adj)
        h2 = h2.relu()

        h3 = torch.sum(h2, 0).unsqueeze(0)

        h = self.lin1(h3)
        h = h.relu()
        if self.dropout is not None:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        return h
