from time import perf_counter

import numpy as np
import scipy
from joblib import Parallel, delayed
from sparse_dot_mkl import gram_matrix_mkl, dot_product_mkl
from tqdm import tqdm

from gnn.utils import scipy_coo_to_torch_coo, torch_sparse_to_scipy_coo


def batch_elem_mul(cov):
    cov = scipy.sparse.csr_matrix(cov)

    def f(cov_adj):
        cov_vec = cov_adj.reshape((cov.shape[1], -1), order="F")
        cov_vec = scipy.sparse.csr_array(cov_vec)
        # does sparse dot product parallel but sometimes there is a bug on certain machines
        # ValueError: Input matrices to dot_product_mkl must be CSR, CSC, or BSR; COO is not supported although all arrays are csr
        if not scipy.sparse.issparse(cov) or not scipy.sparse.issparse(cov_vec):
            raise ValueError("My Error: Input matrices to dot_product_mkl must sparse")
        cov_vec = scipy.sparse.csr_matrix(cov_vec)
        return dot_product_mkl(cov, cov_vec).toarray()
    return f


"""
# much slower than batch_elem_mul_scipy
def batch_elem_mul(cov):
    cov = scipy.sparse.csr_array(cov)
    cov = scipy_coo_to_torch_coo(cov)

    def f(cov_adj):
        cov_vec = cov_adj.reshape((cov.shape[1], -1), order="F")
        cov_vec = scipy.sparse.csr_array(cov_vec)
        result = cov @ scipy_coo_to_torch_coo(cov_vec)
        return result.to_dense().numpy()
    return f
"""


def batch_mul(x, cov):
    cov = scipy.sparse.csr_array(cov)
    window = x.shape[0]
    cov_middle_block = scipy.sparse.tril(cov, window)
    cov = cov + cov_middle_block.T
    x = x.T.flatten()
    x = scipy.sparse.csr_array(x)
    f = batch_elem_mul(x)

    t = perf_counter()
    results = Parallel(n_jobs=-1)(delayed(f)(cov[n * window : (n + 1) * window]) for n in tqdm(range(window)))
    """
    # sequential for debugging
    split = [cov[n * window: (n + 1) * window] for n in tqdm(range(window))]
    results = map(f, split)
    """
    print(f"Elapsed Time Batch Mul: {round((perf_counter() - t) / 60, 4)}mins")

    results = np.vstack(list(results))
    return results + np.triu(results, 1).T


def ia_x_ia_T(adj, c=1, dtype="float32"):
    n = adj.shape[0]
    I = scipy.sparse.eye(n, dtype=dtype)
    one = scipy.sparse.coo_array(np.ones((n, 1), dtype=dtype))
    a_flat = adj.reshape(-1, 1).squeeze(1)
    a_flat_diag = scipy.sparse.diags([a_flat], [0], dtype=dtype)
    del adj
    del a_flat

    def f(x):
        x = scipy.sparse.csr_array(x, dtype=dtype)
        l = (a_flat_diag @ scipy.sparse.kron(one, I)) @ x
        l = ((a_flat_diag @ scipy.sparse.kron(I, one)) @ x) + l
        del x

        print("Calculating Adjacency Kernel Matrix")
        t1 = perf_counter()
        # torch_l = scipy_coo_to_torch_coo(scipy.sparse.coo_array(l)).requires_grad_(False)
        # kernel = torch_l @ torch_l.t()
        # result = scipy.sparse.csr_array(torch_sparse_to_scipy_coo(kernel))
        # print(f"Elapsed Time : {round((perf_counter() - t1) / 60)}mins")
        # return result

        kernel = gram_matrix_mkl(l, transpose=True)
        print(f"Elapsed Time : {round((perf_counter() - t1) / 60, 4)}mins")
        return kernel * c

    return f


def torch_matmul(x, y):
    # faster than scipy sparse mul
    result = scipy_coo_to_torch_coo(x) * scipy_coo_to_torch_coo(y)
    return scipy.sparse.csr_array(torch_sparse_to_scipy_coo(result))
