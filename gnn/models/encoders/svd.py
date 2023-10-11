import torch
from torch import Tensor
from torch_geometric.nn import GAE


class SVD(GAE):
    def __init__(self, hidden):
        super().__init__(None)
        self.hidden = hidden
        self.decoderU = None

    def parameters(self):
        return []

    def reset_parameters(self):
        pass

    def encode(self, x, adj, *args, **kwargs) -> Tensor:
        adj = adj @ x
        adj = adj.to_dense()
        U, _, _ = torch.svd(adj)
        Uk = U[:, 0 : self.hidden]
        self.decoderU = Uk
        result = Uk.T @ adj
        return result

    def decoder(self, z, edge_index, *args, **kwargs) -> Tensor:
        z = self.decoderU @ z
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)
