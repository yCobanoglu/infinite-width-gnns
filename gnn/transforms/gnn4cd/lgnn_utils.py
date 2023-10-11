import contextlib

try:
    with contextlib.redirect_stdout(None):
        import bitsandbytes as bnb
except:
    import warnings

    warnings.warn("bitsandbytes not installed or not working")

import numpy as np
import torch

from gnn.device import DEVICE
from gnn.transforms.gnn4cd.backtracking_graphs import (
    compact_non_backtracking,
    hashimoto_gnn4cd,
)
from gnn.utils import torch_sparse_diag, torch_sparse_identitiy


def coo_to_sparse_tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    return torch.sparse_coo_tensor(i, v, coo.shape, dtype=torch.float64)


def compute_operators_sparse(W, J):
    # alternative implementation with cuda sparse
    W = W.to(DEVICE)
    N = W.shape[0]
    d = torch.sparse.sum(W, 1)
    D = torch_sparse_diag(d)
    QQ = W.clone()
    result = [torch_sparse_identitiy(N).unsqueeze(2).to(DEVICE)]
    for j in range(J):
        result.append(QQ.unsqueeze(2))
        QQQQ = (QQ @ QQ).coalesce()
        QQ_values = QQQQ.values()
        QQ = torch.sparse_coo_tensor(
            QQQQ.indices(),
            torch.minimum(QQ_values, torch.ones(QQ_values.shape).to(DEVICE)),
            size=QQQQ.shape,
        )
    result.append(D.unsqueeze(2))
    x = d.unsqueeze(1)
    return torch.cat([x for x in result], 2), x


def compute_operators_bnb(QQ, J):
    QQ = QQ.to_dense().to(DEVICE)
    N = QQ.shape[0]
    d = torch.sum(QQ, 1)
    D = torch.diag(d)
    result = [torch.eye(N).unsqueeze(2).to(DEVICE)]
    ones = torch.ones(QQ.shape).to(DEVICE)
    for j in range(J):
        result.append(QQ.unsqueeze(2))
        QQ = bnb.matmul(QQ, QQ)
        QQ = torch.minimum(QQ, ones)
    result.append(D.unsqueeze(2))
    x = d.unsqueeze(1)
    output = torch.cat([x for x in result], 2), x
    return output


def compute_operators(QQ, J):
    if DEVICE == "cuda":
        return compute_operators_bnb(QQ, J)
    return compute_operators_sparse(QQ, J)


def get_compact_lgnn(W, J):
    WW, x = compute_operators(W, J)
    W = W.coalesce()
    W_lg = compact_non_backtracking(W.indices(), W.shape[0])
    WW_lg, y = compute_operators(W_lg, J)
    return WW, x, WW_lg, y


def get_gnn(W, J):
    return compute_operators(W, J)


def get_lgnn(W, J):
    WW, x = compute_operators(W, J)
    # W = W.coalesce()
    W_lg, P = hashimoto_gnn4cd(W.indices(), W.shape[0])
    WW_lg, y = compute_operators(W_lg, J)
    return WW, x, WW_lg, y, P
