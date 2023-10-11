import torch
from torch.nn import ReLU


class RandomProjection(torch.nn.Module):
    def __init__(self, input, hidden):
        super().__init__()
        self.hidden = hidden
        self.weight = torch.rand((input, hidden))

    def forward(self, adj, *args, **kwargs):
        return adj @ self.weight
