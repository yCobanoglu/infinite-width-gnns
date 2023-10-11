import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential

from gnn.models.modules import GraphConv, Rank1Linear


class GCN_SumPooling_OneLayer(torch.nn.Module):
    """GCN"""

    def __init__(self, num_node_features, dim_h, num_classes, dropout):
        super().__init__()
        self.dropout = dropout
        self.gc1 = GraphConv(Linear(num_node_features, dim_h, bias=False))
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, adj, **kwargs):
        # Node embeddings
        h1 = self.gc1(x, adj)
        h1 = F.relu(h1)
        hG = torch.sum(h1, 0).unsqueeze(0)
        if self.dropout is not None:
            hG = F.dropout(hG, p=self.dropout, training=self.training)
        h = self.lin(hG)
        return h


class GCN_SumPooling_OneLowRankLayer(torch.nn.Module):
    def __init__(self, num_classes, dropout, dim_h):
        super().__init__()
        self.dropout = dropout
        self.gc1 = GraphConv(Rank1Linear(dim_h))
        self.lin = Rank1Linear(num_classes)

    def forward(self, x, adj, **kwargs):
        # Node embeddings
        h1 = self.gc1(x, adj)
        hG = torch.sum(h1, 0).unsqueeze(0)
        if self.dropout is not None:
            hG = F.dropout(hG, p=self.dropout, training=self.training)
        h = self.lin(hG)
        return h
