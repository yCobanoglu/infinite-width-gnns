import warnings

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops, to_torch_coo_tensor

from gnn.device import DEVICE
from gnn.transforms.gnn4cd.backtracking_graphs import (
    compact_non_backtracking,
    hashimoto,
    line_graph,
)
from gnn.transforms.gnn4cd.lgnn_utils import get_gnn, get_compact_lgnn, get_lgnn

from gnn.utils import torch_sparse_identitiy, torch_sparse_diag


def welling_normalized_laplacian(edge_index, num_nodes):
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]
    adj = to_torch_coo_tensor(edge_index, size=(num_nodes, num_nodes))
    degree = torch.sparse.sum(adj, dim=0)
    degree_normalized = torch.pow(degree, -0.5)
    degree_normalized = torch_sparse_diag(degree_normalized)
    return degree_normalized @ adj @ degree_normalized


def laplacian(edge_index, num_nodes):
    adj = to_torch_coo_tensor(edge_index, size=(num_nodes, num_nodes))
    degree = torch.sparse.sum(adj, dim=0)
    return torch_sparse_diag(degree) - adj


class AddEye(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        adj = data.adj
        data.adj = adj + torch_sparse_identitiy(adj.shape[0]).to(DEVICE)
        return data


class AddDegree(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        adj = data.adj
        d = torch.sparse.sum(adj, dim=1)
        data.adj = adj + torch_sparse_diag(d)
        return data


class WellingNormalized(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.adj = welling_normalized_laplacian(data.adj.coalesce().indices(), data.adj.shape[0])
        return data


class Default1(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        adj = data.adj
        adj = adj.to_dense()
        adj = adj + torch.eye(adj.size(0)).to(DEVICE)
        # load best_gamma.npy as np array
        best_gamma = None
        with open(
            "/home/yunus/PycharmProjects/graph-neural-networks/gnn/infinite_width/best_gamma_cora_default.npy",
            "rb",
        ) as f:
            best_gamma = np.load(f)
            best_gamma = torch.from_numpy(best_gamma)
            best_gamma = torch.diag(best_gamma)
        best_gamma = best_gamma.to(DEVICE)
        adj = best_gamma @ adj @ best_gamma
        # asd = torch.sparse_coo_tensor(adj)
        data.adj = adj
        return data


class WellingNormalized1(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        adj = welling_normalized_laplacian(data.adj.coalesce().indices(), data.adj.shape[0])
        adj = adj.to_dense()
        # load best_gamma.npy as np array
        best_gamma = None
        with open("/home/yunus/PycharmProjects/graph-neural-networks/best_gamma.npy", "rb") as f:
            best_gamma = np.load(f)
            best_gamma = torch.from_numpy(best_gamma)
            best_gamma = torch.diag(best_gamma)
        best_gamma = best_gamma.to(DEVICE)
        adj = best_gamma @ adj @ best_gamma
        # asd = torch.sparse_coo_tensor(adj)
        data.adj = adj.to_sparse_coo()
        return data


class Laplacian(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.adj = laplacian(data.adj.coalesce().indices(), data.adj.shape[0])
        return data


class MLP(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.adj = torch_sparse_identitiy(data.adj.shape[0])
        return data


class GnnTransform(BaseTransform):
    def __init__(self, hierachy):
        super().__init__()
        self.hierachy = hierachy

    def __call__(self, data):
        data.adj, data.x = get_gnn(data.adj, self.hierachy)
        return data


class Lgnn(BaseTransform):
    def __init__(self, hierachy):
        super().__init__()
        self.hierachy = hierachy

    def __call__(self, data):
        data.adj, data.x, data.adj_lg, data.adj_lg_x, data.projections = get_lgnn(data.adj, self.hierachy)
        return data


class CompactLgnn(BaseTransform):
    def __init__(self, hierachy):
        super().__init__()
        self.hierachy = hierachy

    def __call__(self, data):
        data.adj, data.x, data.adj_lg, data.adj_lg_x = get_compact_lgnn(data.adj, self.hierachy)
        half = data.adj_lg.shape[0] // 2
        eye = torch_sparse_identitiy(half)
        data.projections = torch.cat([eye, eye], dim=1).unsqueeze(2)
        return data


class Hashimoto(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.adj, data.projection = hashimoto(data.edge_index, data.num_nodes)
        data.projection = data.projection / data.projection.shape[1]
        data.x = torch_sparse_identitiy(data.adj.shape[0])
        warnings.warn(f"{self.__repr__()} will override  features i.e 'data.x'  to identity matrix {data.x.shape}")
        return data


class CompactNonBacktracking(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.adj = compact_non_backtracking(data.edge_index, data.num_nodes)
        data.x = torch_sparse_identitiy(data.adj.shape[0])
        warnings.warn(f"{self.__repr__()} will override  features i.e 'data.x'  to identity matrix {data.x.shape}")
        return data


class CompactNonBacktrackingWithProjection(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.adj = compact_non_backtracking(data.edge_index, data.num_nodes)
        data.x = torch_sparse_identitiy(data.adj.shape[0])
        warnings.warn(f"{self.__repr__()} will override  features i.e 'data.x'  to identity matrix {data.x.shape}")
        half = data.adj.shape[0] // 2
        eye = torch_sparse_identitiy(half)
        data.projections = torch.cat([eye, eye], dim=1)
        return data


class LineGraph(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.adj, data.projection = line_graph(data.edge_index, data.num_nodes)
        data.projection = data.projection / data.projection.shape[1]
        data.x = torch_sparse_identitiy(data.adj.shape[0])
        warnings.warn(f"{self.__repr__()} will override features i.e 'data.x'  to identity matrix {data.x.shape}")
        return data
