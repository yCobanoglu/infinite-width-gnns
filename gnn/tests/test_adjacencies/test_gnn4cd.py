import networkx as nx
import numpy as np
import torch

from gnn.tests.conftest import fro_norm
from gnn.tests.reference_implementations_for_testing.expected_adjacenies import (
    hashimoto_networkx,
)
from gnn.tests.reference_implementations_for_testing.expected_gnn4cd import (
    get_NB_2,
    get_Pm,
    get_Pd,
)
from gnn.tests.reference_implementations_for_testing.expected_gnn4cd import (
    get_lg_inputs as expected_get_lg_inputs,
    get_P,
)
from gnn.transforms.gnn4cd.backtracking_graphs import (
    hashimoto_gnn4cd,
    incidence_matrix,
    to_signed_incidence,
)
from gnn.transforms.gnn4cd.lgnn_utils import get_lgnn


def test_get_lg_inputs(generate_sbm_dataset):
    # test against gnn4cd reference implementation
    dataset, adj, _ = generate_sbm_dataset
    expected = expected_get_lg_inputs(adj, 1)
    results = get_lgnn(dataset.adj, 1)

    for exp, result in zip(expected, results):
        a1 = exp.detach().numpy()
        a2 = result.cpu().unsqueeze(0).to_dense().numpy()
        assert np.allclose(a1, a2)


def test_hashimoto_projection(generate_sbm_dataset):
    # test against gnn4cd reference implementation
    dataset, adj, _ = generate_sbm_dataset
    expected_P = get_P(adj)
    P = hashimoto_gnn4cd(dataset.edge_index, adj.shape[0])[1].to_dense().numpy()
    assert np.allclose(expected_P, P)


def test_hashimoto(generate_sbm_dataset):
    # test against gnn4cd reference implementation
    dataset, adj, _ = generate_sbm_dataset
    expected_W_lg = get_NB_2(adj)
    W_lg = hashimoto_gnn4cd(dataset.edge_index, adj.shape[0])[0].to_dense().numpy()
    assert np.allclose(expected_W_lg, W_lg)


def test_lgnn_Pm(generate_sbm_dataset):
    dataset, adj, _ = generate_sbm_dataset
    incidence = get_Pm(adj)
    expected_incidence = nx.incidence_matrix(nx.from_numpy_array(adj)).toarray()
    assert np.allclose(np.concatenate((expected_incidence, expected_incidence), axis=1), incidence)


# L = I * I.T
def test_lgnn_Pd_laplacian(generate_sbm_dataset):
    dataset, adj, _ = generate_sbm_dataset
    incidence = get_Pd(adj)
    I_ = incidence[:, : incidence.shape[1] // 2]
    L = I_ @ I_.T
    laplacian = nx.laplacian_matrix(nx.from_numpy_array(adj)).toarray()
    assert np.allclose(laplacian, L)


# Pd are two oriented signed incidence matrix stacked next to each other up to permutation
def test_lgnn_Pd(generate_sbm_dataset):
    dataset, adj, _ = generate_sbm_dataset
    incidence = get_Pd(adj)
    nx_graph = nx.from_numpy_array(adj)
    expected_incidence = nx.incidence_matrix(nx_graph, oriented=True, nodelist=nx_graph.nodes()).toarray()
    expected_incidence = np.concatenate((-expected_incidence, expected_incidence), axis=1)
    assert fro_norm(incidence) == fro_norm(expected_incidence)
    assert np.allclose(expected_incidence, incidence)


def test_lgnn_hashimoto1(generate_sbm_dataset):
    dataset, adj, _ = generate_sbm_dataset
    H = get_NB_2(adj)
    H_1 = hashimoto_networkx(adj)[0]

    assert np.isclose(fro_norm(H), fro_norm(H_1))
    assert np.allclose(H, H_1)


def test_my_hashimoto(generate_sbm_dataset):
    # https://mathoverflow.net/questions/219752/hashimoto-matrix-non-backtracking-operator-and-the-graph-laplacian
    dataset, _, _ = generate_sbm_dataset
    adj = dataset.adj
    edge_index, num_nodes = adj.indices(), adj.shape[0]
    hashimoto, _ = hashimoto_gnn4cd(edge_index, num_nodes)

    indicdence = incidence_matrix(edge_index, num_nodes)
    signed_incidence = to_signed_incidence(indicdence).to_dense()
    # torch set all negative numbers to zero
    s_positive = torch.clone(signed_incidence)
    s_positive[signed_incidence < 0] = 0
    s_negative = torch.clone(signed_incidence)
    s_negative[signed_incidence > 0] = 0
    exp_hashimoto = s_positive.T @ s_negative
    assert torch.testing.assert_allclose(exp_hashimoto, hashimoto.to_dense())
