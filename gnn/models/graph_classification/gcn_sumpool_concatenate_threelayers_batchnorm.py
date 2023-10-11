import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

from gnn.models.modules import GraphConv


class GCN_SumPool_Concatenate_ThreeLayers_BatchNorm(torch.nn.Module):
    """GIN"""

    def __init__(self, num_node_features, dim_h, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        mlp1 = Sequential(
            Linear(num_node_features, dim_h, bias=False),
            BatchNorm1d(dim_h),
            ReLU(),
            Linear(dim_h, dim_h, bias=False),
            ReLU(),
        )
        self.conv1 = GraphConv(mlp1)
        mlp2 = Sequential(
            Linear(dim_h, dim_h, bias=False),
            BatchNorm1d(dim_h),
            ReLU(),
            Linear(dim_h, dim_h, bias=False),
            ReLU(),
        )
        self.conv2 = GraphConv(mlp2)
        mlp3 = Sequential(
            Linear(dim_h, dim_h, bias=False),
            BatchNorm1d(dim_h),
            ReLU(),
            Linear(dim_h, dim_h, bias=False),
            ReLU(),
        )
        self.conv3 = GraphConv(mlp3)

        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, num_classes)

    def forward(self, x, adj, **kwargs):
        # Node embeddings
        h1 = self.conv1(x, adj)
        h2 = self.conv2(h1, adj)
        h3 = self.conv3(h2, adj)

        # Global Add pool
        h1 = torch.sum(h1, 0).unsqueeze(0)
        h2 = torch.sum(h2, 0).unsqueeze(0)
        h3 = torch.sum(h3, 0).unsqueeze(0)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        if self.dropout is not None:
            h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h
