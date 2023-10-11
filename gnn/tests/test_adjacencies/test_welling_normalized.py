import numpy as np
import torch

from gnn.tests.reference_implementations_for_testing.expected_adjacenies import (
    welling_normalized_laplacian as welling_normalized_laplacian_for_testing,
)
from gnn.transforms.adjacency_transforms import welling_normalized_laplacian
from gnn.utils import torch_sparse_identitiy, torch_sparse_diag


def test_sparse_identity():
    assert torch.allclose(torch_sparse_identitiy(3).to_dense(), torch.eye(3))


def test_sparse_identitiy_2():
    exp = torch.eye(10).to_sparse_coo()
    identity = torch_sparse_identitiy(10)
    exp = exp.coalesce()
    identity = identity.coalesce()
    assert torch.allclose(exp.to_dense(), identity.to_dense())
    assert torch.allclose(exp.indices(), identity.indices())
    assert torch.allclose(exp.values(), identity.values())


def test_sparse_diag():
    diag = torch.Tensor([1, 2, 0, 0, -1])
    assert torch.allclose(torch_sparse_diag(diag).to_dense(), torch.diag(diag))


def test_welling_normalized(generate_sbm_dataset):
    # test against reference implementation
    dataset, adj, _ = generate_sbm_dataset
    expected_normalized = welling_normalized_laplacian_for_testing(adj)
    normalized = welling_normalized_laplacian(dataset.edge_index, dataset.num_nodes).to_dense().numpy()
    assert np.allclose(expected_normalized, normalized)
