import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge

from gnn.sparsify.call_julia_sparsify import _sparsify, find_sparse_rate
from gnn.utils import adj_from_edge_index


class RandomSparsifyGraph(BaseTransform):
    def __init__(self, sparsify_rate):
        super().__init__()
        self.sparsify_rate = sparsify_rate

    def __call__(self, data):
        if self.sparsify_rate == -1:
            self.sparsify_rate = find_sparse_rate(data.adj)
        if self.sparsify_rate < 1:
            data.edge_index, _ = dropout_edge(data.edge_index, p=self.sparsify_rate, force_undirected=True)
            data.adj = adj_from_edge_index(data.edge_index, data.num_nodes)
        else:
            raise ValueError("sparsify_rate must be less than 1 maybe graph is already sparse (n log n)")
        return data


class EffectiveResistance(BaseTransform):
    def __init__(self, ep):
        super().__init__()
        self.ep = ep

    def __call__(self, data):
        adj = _sparsify(data.adj, self.ep)
        adj = adj.coalesce()
        data.adj = torch.sparse_coo_tensor(values=torch.ones_like(adj.values()), indices=adj.indices(), size=adj.shape)
        data.edge_index = adj.indices()
        return data
