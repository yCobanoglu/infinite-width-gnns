import torch
import torch.nn.functional as F
from torch.nn import Linear

from gnn.models.modules import GraphConv


class GCN_SumPool_ThreeLayers(torch.nn.Module):
    """GCN"""

    def __init__(self, num_node_features, dim_h, num_classes, dropout):
        super().__init__()
        self.dropout = dropout
        self.gc1 = GraphConv(Linear(num_node_features, dim_h, bias=False))
        self.gc2 = GraphConv(Linear(dim_h, dim_h, bias=False))
        self.gc3 = GraphConv(Linear(dim_h, dim_h, bias=False))
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, adj, **kwargs):
        # Node embeddings
        h1 = self.gc1(x, adj)
        h1 = F.relu(h1)
        h2 = self.gc2(h1, adj)
        h2 = F.relu(h2)
        h = self.gc3(h2, adj)
        h = F.relu(h)

        # Global pool
        hG = torch.sum(h, 0).unsqueeze(0)
        # Classifier
        if self.dropout is not None:
            hG = F.dropout(hG, p=self.dropout, training=self.training)
        h = self.lin(hG)
        return h
