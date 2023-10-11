from math import log

import numpy as np
import warnings
import torch
from juliacall import Main as jl

jl.seval("using Laplacians")
jl.seval("using SparseArrays")


_WARN_ONCE = True
DEFINITE_EPSILON = None


def find_sparse_rate(adj):
    nodes = adj.shape[0]
    sparse_num_nodes = nodes * log(nodes)
    percentage = sparse_num_nodes / np.count_nonzero(adj)
    return percentage


def find_epsilon(adj, rate):
    global DEFINITE_EPSILON
    if rate == -1:
        rate = find_sparse_rate(adj)
        if rate >= 1:
            DEFINITE_EPSILON = -1
            return None
    else:
        rate = 1 - rate
    EPSILON = 4
    step = 2
    step_counter = 0
    total = np.count_nonzero(adj)

    increased = False
    decreased = False
    while True:
        sparsified = __sparsify(adj, EPSILON)
        counts = np.count_nonzero(sparsified)
        output = counts / total
        if rate - 0.01 < output < rate + 0.01:
            break
        if output > rate:
            EPSILON += step * 1 / (step_counter + 1)
            increased = True
            if decreased:
                step_counter += 1
                increased, decreased = False, False
        else:
            EPSILON -= step * 1 / (step_counter + 1)
            decreased = True
            if increased:
                step_counter += 1
                increased, decreased = False, False
    DEFINITE_EPSILON = EPSILON


def __sparsify(adj, epsilon):
    sparsified = jl.Matrix(jl.sparsify(jl.sparse(adj), ep=epsilon))
    return np.array(sparsified)


def _sparsify(adj, rate):
    global DEFINITE_EPSILON
    adj = adj.cpu().float().to_dense().numpy()
    if DEFINITE_EPSILON is None:
        find_epsilon(adj, rate)
    if DEFINITE_EPSILON == -1:
        warnings.warn(f"No Edge reduction graph is already sparse (n log n)!")
        raise ValueError("No Edge reduction graph is already sparse (n log n)!")
        return adj
    sparsified = __sparsify(adj, DEFINITE_EPSILON)
    global _WARN_ONCE
    if _WARN_ONCE:
        warnings.warn(f"Total Edges: {np.count_nonzero(adj)}")
        warnings.warn(f"Total Edges: {np.count_nonzero(sparsified)}")
        warnings.warn(f"Reduction: {1 - np.count_nonzero(sparsified) / np.count_nonzero(adj)}")
        _WARN_ONCE = False
    return torch.from_numpy(sparsified).float().to_sparse_coo()
