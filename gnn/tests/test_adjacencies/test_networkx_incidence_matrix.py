import networkx as nx
import numpy as np


# Verify networkx incidence matrices implementation


# ADJ = U U.T - D
def test_unoriented_incidence_matrix(generate_sbm_dataset):
    dataset, adj, degree_matrix = generate_sbm_dataset
    incidence_matrix = nx.incidence_matrix(nx.from_numpy_array(adj)).toarray()
    exp_adj = incidence_matrix @ incidence_matrix.T - degree_matrix
    assert np.allclose(adj, exp_adj)


# L(G) = U.T U - 2 I
# https://en.wikipedia.org/wiki/Line_graph
# https://github.com/networkx/networkx/issues/6253
def test_line_graph_from_unsigned_incidence_matrix(generate_sbm_dataset):
    _, adj, _ = generate_sbm_dataset
    graph = nx.from_numpy_array(adj)
    line_graph = nx.line_graph(graph)
    L_G = nx.to_numpy_array(line_graph)

    incidence_matrix = nx.incidence_matrix(graph, edgelist=line_graph.nodes)
    U = incidence_matrix.toarray()
    line_graph_ = U.T @ U - 2 * np.eye(U.shape[1])
    assert np.allclose(L_G, line_graph_)


# https://mathoverflow.net/questions/219752/hashimoto-matrix-non-backtracking-operator-and-the-graph-laplacian
# L = I I.T
def test_laplacian_is_signed_incidence_matrix(generate_sbm_dataset):
    _, adj, _ = generate_sbm_dataset
    graph = nx.from_numpy_array(adj)
    laplacian = nx.laplacian_matrix(graph).toarray()
    incidence_matrix = nx.incidence_matrix(graph, oriented=True, nodelist=graph.nodes).toarray()
    assert np.allclose(laplacian, incidence_matrix @ incidence_matrix.T)
