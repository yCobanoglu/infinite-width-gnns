import networkx as nx
import numpy as np
import torch


# https://mathoverflow.net/questions/219752/hashimoto-matrix-non-backtracking-operator-and-the-graph-laplacian
def hashimoto_networkx(adj):
    graph = nx.from_numpy_array(adj)
    incidence = nx.incidence_matrix(graph, nodelist=graph.nodes, edgelist=graph.edges).toarray()
    I_ = np.concatenate((incidence, incidence), axis=1)
    S_I = nx.incidence_matrix(graph, oriented=True, nodelist=graph.nodes, edgelist=graph.edges).toarray()
    S_I_ = np.concatenate((-S_I, S_I), axis=1)

    Pf = (I_ + S_I_) / 2
    Pt = (I_ - S_I_) / 2

    X = Pt.T @ Pf
    Y = np.ones((X.shape)) - X.T
    return X * Y, (incidence, S_I)


# https://graph-tool.skewed.de/static/doc/spectral.html#graph_tool.spectral.hashimoto
def compact_non_backtracking_numpy(adj):
    number_of_nodes = adj.shape[0]
    eye = -np.eye(number_of_nodes)
    Q = np.sum(adj, axis=0)
    Q = np.diag(Q) - np.eye(number_of_nodes)
    zeros = np.zeros((number_of_nodes, number_of_nodes))
    rs1 = np.concatenate([adj, eye], axis=1)
    rs2 = np.concatenate([Q, zeros], axis=1)
    rs3 = np.concatenate([rs1, rs2], axis=0)
    return torch.from_numpy(rs3)


def normalized_laplacian_networkx(adj):
    graph = nx.from_numpy_array(adj.numpy())
    return nx.normalized_laplacian_matrix(graph).toarray()


# https://github.com/abduallahmohamed/Social-STGCNN/issues/22
# https://arxiv.org/abs/1609.02907
def welling_normalized_laplacian(adj):
    """Row-normalize sparse matrix"""
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv.dot(adj)


def line_graph_networkx(adj):
    graph = nx.from_numpy_array(adj.numpy())
    incidence_matrix = nx.incidence_matrix(graph, edgelist=graph.nodes)
    U = incidence_matrix.toarray()
    return U.T @ U - 2 * np.eye(U.shape[0]), U
