import pytest
import torch


# https://github.com/pytorch/pytorch/issues/90516
def incidence_matrix_(adj_sparse, edge_index, num_nodes):
    mask = edge_index[0] <= edge_index[1]
    adj_sparse_indices = adj_sparse.coalesce().indices()
    row = adj_sparse_indices[0][mask]
    col = adj_sparse_indices[1][mask]
    edge_index_unique = torch.vstack([row, col])
    edge_number_unique = edge_index_unique.shape[1]
    edge_indices_x = torch.range(0, edge_number_unique - 1)
    indices_1 = torch.vstack([edge_indices_x, edge_index_unique[0]])
    indices_2 = torch.vstack([edge_indices_x, edge_index_unique[1]])
    indices = torch.hstack([indices_1, indices_2])
    values = torch.ones(indices.shape[1])
    return torch.sparse_coo_tensor(indices, values, (edge_number_unique, num_nodes)).transpose(0, 1)


@pytest.mark.xfail(reason="Bug in pytorch: https://github.com/pytorch/pytorch/issues/90516")
def test_sparse_mm():
    edge_index = torch.Tensor([[1, 1, 2, 3], [2, 3, 1, 1]])
    adj_matrix = torch.Tensor(
        [
            [
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                1,
                1,
            ],
            [
                0,
                1,
                0,
                0,
            ],
            [
                0,
                1,
                0,
                0,
            ],
        ]
    ).to_sparse_coo()
    num_nodes = 4
    X = incidence_matrix_(adj_matrix, edge_index, num_nodes)
    X = X.T @ X
    Y = torch.ones(X.shape).to_sparse_coo() - X.transpose(0, 1)

    result = (X * Y).to_dense()
    expected_result = X.to_dense() * Y.to_dense()
    assert torch.allclose(result, expected_result)
