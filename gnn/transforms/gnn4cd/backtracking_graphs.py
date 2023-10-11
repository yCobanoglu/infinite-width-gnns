# https://graph-tool.skewed.de/static/doc/spectral.html#graph_tool.spectral.hashimoto
import torch
from torch_geometric.utils import to_torch_coo_tensor

from gnn.utils import torch_sparse_identitiy, torch_sparse_diag, torch_sparse_zeroes


def compact_non_backtracking(edge_index, num_nodes):
    edge_index_values = torch.ones(edge_index.shape[1])
    edge_index_ = torch.sparse_coo_tensor(edge_index, edge_index_values, (num_nodes, num_nodes))
    Q_ = torch.sparse.sum(edge_index_, dim=0)
    Q = -torch.ones((Q_.shape[0],)).to_sparse_coo() + Q_
    indicdence = -torch_sparse_identitiy(num_nodes)
    Q = torch_sparse_diag(Q)
    indices_upper_part = torch.cat([edge_index_, indicdence], 1)
    indices_lower_part = torch.cat([Q, torch_sparse_zeroes((num_nodes, num_nodes))], 1)
    return torch.cat([indices_upper_part, indices_lower_part], 0)


def incidence_matrix(edge_index, num_nodes, signed=False):
    mask = edge_index[0] + 1 <= edge_index[1]
    adj_sparse = to_torch_coo_tensor(edge_index, size=(num_nodes, num_nodes))
    adj_sparse_indices = adj_sparse.indices()
    row = adj_sparse_indices[0][mask]
    col = adj_sparse_indices[1][mask]
    edge_index_unique = torch.vstack([row, col])
    edge_number_unique = edge_index_unique.shape[1]
    edge_indices_x = torch.range(0, edge_number_unique - 1)
    indices_1 = torch.vstack([edge_indices_x, edge_index_unique[0]])
    indices_2 = torch.vstack([edge_indices_x, edge_index_unique[1]])
    indices = torch.hstack([indices_1, indices_2])
    if signed:
        # first -1 than 1 to coincide with networkx incidence implementation
        values = torch.hstack([-torch.ones(indices_2.shape[1]), torch.ones(indices_1.shape[1])])
    else:
        values = torch.ones(indices.shape[1])
    return torch.sparse_coo_tensor(indices, values, (edge_number_unique, num_nodes)).transpose(0, 1)


def to_signed_incidence(indicdence):
    indicdence = indicdence.transpose(0, 1)
    indicdence = indicdence.coalesce()
    values = torch.Tensor([-1, 1]).tile((indicdence.shape[0],))
    return torch.sparse_coo_tensor(indicdence.indices(), values, indicdence.shape).transpose(0, 1)


def line_graph(edge_index, num_nodes):
    num_edges = edge_index.shape[1] // 2
    torch_sparse_identitiy(num_edges)
    unsigned = incidence_matrix(edge_index, num_nodes)
    correction = 2 * torch_sparse_identitiy(num_edges)
    line_graph = (torch.transpose(unsigned, 0, 1) @ unsigned) - correction
    return line_graph, unsigned


def hashimoto_gnn4cd(edge_index, num_nodes):
    # used for lgnn model
    indicdence = incidence_matrix(edge_index, num_nodes)
    I_ = torch.cat([indicdence, indicdence], 1)
    S_I = to_signed_incidence(indicdence)
    S_I_ = torch.cat([-S_I, S_I], 1)
    Pf = (I_ + S_I_) / 2
    Pt = (I_ - S_I_) / 2

    X = Pt.T @ Pf
    X = X
    result = X * (torch.ones(X.shape) - X.transpose(0, 1))
    return result, torch.cat([I_.unsqueeze(2), S_I_.unsqueeze(2)], 2)


def hashimoto(edge_index, num_nodes):
    # single projection matrix which project nb adjacency back to original graph dimension
    indicdence = incidence_matrix(edge_index, num_nodes)
    I_ = torch.cat([indicdence, indicdence], 1)
    S_I = to_signed_incidence(indicdence)
    S_I_ = torch.cat([-S_I, S_I], 1)
    Pf = (I_ + S_I_) / 2
    Pt = (I_ - S_I_) / 2
    X = Pt.T @ Pf
    Y = torch.ones(X.shape) - X.transpose(0, 1).to_dense()

    # https://github.com/pytorch/pytorch/issues/90516 to dense necessary sparse mm causes wrong out
    hashimoto = X.to_dense() * Y.to_dense()
    return hashimoto.to_sparse_coo(), I_ + S_I_
