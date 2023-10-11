import networkx as nx
import numpy as np

from gnn.tests.conftest import fro_norm
from gnn.transforms.gnn4cd.backtracking_graphs import line_graph


def test_line_graph(generate_sbm_dataset):
    # test networkx line graph against my line graph implementation
    dataset, adj, _ = generate_sbm_dataset
    graph = nx.from_numpy_array(adj)
    expected_line_graph = nx.to_numpy_array(nx.line_graph(graph))
    line_graph_adj = line_graph(dataset.edge_index, dataset.num_nodes)[0].to_dense().numpy()
    assert np.isclose(fro_norm(line_graph_adj), fro_norm(expected_line_graph))
    # assert np.allclose(line_graph, expected_line_graph) doesn't work because of the ordering of the edges


def test_line_graph_2(generate_sbm_dataset):
    # test networkx incidence I.T @ I - 2I against my line graph implementation
    dataset, adj, _ = generate_sbm_dataset
    graph = nx.from_numpy_array(adj)
    incidence_matrix = nx.incidence_matrix(graph, edgelist=graph.edges).todense()
    line_graph_adj = line_graph(dataset.edge_index, dataset.num_nodes)[0].to_dense().numpy()
    assert np.allclose(
        line_graph_adj,
        incidence_matrix.transpose() @ incidence_matrix - 2 * np.eye(line_graph_adj.shape[0]),
    )
