import networkx as nx
import numpy as np

from gnn.tests.conftest import fro_norm
from gnn.transforms.gnn4cd.backtracking_graphs import (
    incidence_matrix,
    to_signed_incidence,
)


def test_unsigned_incidence_matrix(generate_sbm_dataset):
    # test again networkx implementation
    dataset, adj, _ = generate_sbm_dataset
    graph = nx.from_numpy_array(adj)
    expected_i = nx.incidence_matrix(graph, edgelist=graph.edges).todense()
    i = incidence_matrix(dataset.edge_index, dataset.num_nodes).to_dense().numpy()
    assert np.allclose(expected_i, i)


def test_signed_incidence_matrix(generate_sbm_dataset):
    # test again networkx implementation
    dataset, adj, _ = generate_sbm_dataset
    graph = nx.from_numpy_array(adj)
    expected_i = nx.incidence_matrix(graph, edgelist=graph.edges, oriented=True).todense()
    i = incidence_matrix(dataset.edge_index, dataset.num_nodes, signed=True).to_dense().numpy()
    assert np.isclose(fro_norm(i), fro_norm(expected_i))
    assert np.allclose(expected_i, i)


def test_to_signed(generate_sbm_dataset):
    # test again networkx implementation
    dataset, adj, _ = generate_sbm_dataset
    graph = nx.from_numpy_array(adj)
    expected_i = nx.incidence_matrix(graph, edgelist=graph.edges, oriented=True).todense()
    i = incidence_matrix(dataset.edge_index, dataset.num_nodes, signed=True)
    i_ = to_signed_incidence(i).to_dense().numpy()
    assert np.allclose(expected_i, i_)
