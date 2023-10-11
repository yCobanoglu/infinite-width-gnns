import numpy as np
import scipy
import torch

from gnn.device import DEVICE


def sbm_signal_to_noise(intra, inter):
    return ((intra - inter) ** 2) / (2 * (intra + inter))


def is_identitiy(x):
    x = x.detach().clone().cpu()
    eye = torch.eye(x.shape[0])
    if x.shape[0] == x.shape[1] and torch.allclose(x.to_dense(), eye):
        return True
    else:
        return False


def torch_sparse_to_scipy_coo(x):
    x = x.coalesce()
    return scipy.sparse.csr_array(scipy.sparse.coo_array((x.values().cpu().numpy(), x.indices().cpu().numpy()), shape=x.shape))


def scipy_coo_to_torch_coo(coo):
    coo = scipy.sparse.coo_array(coo)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()


def torch_sparse_zeroes(shape):
    return torch.sparse_coo_tensor(size=shape)


def torch_sparse_diag(values):
    if values.is_sparse:
        diag = values.coalesce()
        indices = diag.indices()
        indices1 = torch.vstack([indices, indices])
        return torch.sparse_coo_tensor(indices1, diag.values(), (values.shape[0], values.shape[0]))
    else:
        len_values = values.shape[0]
        indices_1 = torch.arange(0, len_values)
        indices = torch.vstack([indices_1, indices_1])
        return torch.sparse_coo_tensor(indices, values, (len_values, len_values))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adj_from_edge_index(edge_index, num_nodes):
    adj = torch.sparse_coo_tensor(
        edge_index.to(DEVICE),
        torch.ones(edge_index.shape[1]).to(DEVICE),
        (num_nodes, num_nodes),
    ).to(DEVICE)
    adj = adj.coalesce()
    return adj


def torch_sparse_identitiy(n):
    return torch_sparse_diag(torch.ones(n))
