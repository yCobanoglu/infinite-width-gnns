import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import Sequential, ReLU

from gnn.models.node_classification.gcn import GCN


class CompactHashimoto(GCN):
    def __init__(self, nfeat, nhid, nclass, dropout, nodes):
        super().__init__(nfeat, nhid, nclass, dropout)
        self.nodes = nodes // 2
        self.weight = Parameter(torch.FloatTensor(nodes * 2, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def forward(self, adj, x, **kwargs):
        x = super().forward(adj, x)
        result = self.weight * x
        half = x.shape[0] // 2
        return result[:half] + result[half:]


class CompactHashimoto1(GCN):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__(nfeat, nhid, nhid, dropout)
        self.w1 = Sequential(nn.Linear(nhid, nhid), ReLU())
        self.w2 = Sequential(nn.Linear(nhid, nhid), ReLU())
        self.w3 = nn.Linear(2 * nhid, nclass)

    def forward(self, adj, x, **kwargs):
        x = super().forward(adj, x)
        half = x.shape[0] // 2
        x1, x2 = x[:half], x[half:]
        x1, x2 = self.w1(x1), self.w2(x2)
        result = torch.cat([x1, x2], 1)
        return self.w3(result)


class CompactHashimoto2(GCN):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__(nfeat, nhid, nhid, dropout)
        self.w1 = Sequential(nn.Linear(nhid, nhid), ReLU())
        self.w1_skip = nn.Linear(nhid, nhid)
        self.w2 = Sequential(nn.Linear(nhid, nhid), ReLU())
        self.w2_skip = nn.Linear(nhid, nhid)
        self.w3 = Sequential(nn.Linear(2 * nhid, 2 * nhid), ReLU(), nn.Linear(2 * nhid, nclass))

    def forward(self, adj, x, **kwargs):
        x = super().forward(adj, x)
        half = x.shape[0] // 2
        x1, x2 = x[:half], x[half:]
        x1, x2 = self.w1(x1) + self.w1_skip(x1), self.w2(x2) + self.w2_skip(x2)
        result = torch.cat([x1, x2], 1)
        return self.w3(result)
