import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import GATConv


class PytorchGeometricGAT(Module):
    def __init__(self, num_features, hidden, heads, num_classes, dropout=None):
        super().__init__()
        if dropout is None:
            self.dropout = 0

        self.conv1 = GATConv(num_features, hidden, heads=heads, dropout=self.dropout)
        self.conv2 = GATConv(hidden * heads, num_classes, concat=False, heads=1, dropout=self.dropout)

    def forward(self, adj, x, *args, **kwargs):
        if adj.is_sparse:
            edge_index = adj.coalesce().indices()
        else:
            adj = adj.to_dense()
            edge_index = adj.nonzero().t()
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
