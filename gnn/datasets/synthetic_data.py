import random

import numpy as np
import torch
import torch_geometric.transforms as T
from scipy.stats import multinomial
from sklearn.datasets import make_classification
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, barabasi_albert_graph

from gnn.transforms.basic_transforms import IdentitiyFeatures


def triangular_to_full_matrix(p):
    n_classes = len(p[0])
    p_square = np.zeros((n_classes, n_classes))
    for i, p_elem in enumerate(p):
        pad_width = n_classes - len(p_elem)
        p_square[i] = np.pad(np.array(p_elem), (pad_width, 0), "constant", constant_values=0)
    p = p_square + np.tril(p_square.T, -1)
    assert np.allclose(p, p.T)
    return p


def _single_sbm(block_sizes, edge_probs, num_channels, transform=None, is_undirected=True, **kwargs):
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs, directed=not is_undirected)

    num_samples = int(block_sizes.sum())
    num_classes = block_sizes.size(0)

    x = None
    if num_channels is not None:
        x, y_not_sorted = make_classification(
            n_samples=num_samples,
            n_features=num_channels,
            n_classes=num_classes,
            weights=block_sizes / num_samples,
            n_informative=2,
            **kwargs,
        )
        x = x[np.argsort(y_not_sorted)]
        x = torch.from_numpy(x).to(torch.float)

    y = torch.arange(num_classes).repeat_interleave(block_sizes)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_samples
    data.num_classes = num_classes

    if transform is not None:
        data = transform(data)
    return data


def sbm(n_nodes, prob_matrix, feature_dim=None, transform=tuple()):
    prob_matrix = triangular_to_full_matrix(prob_matrix)
    n_classes = prob_matrix[0].shape[0]
    if feature_dim is None:
        transform = [IdentitiyFeatures(), *transform]
    data = _single_sbm(
        torch.LongTensor([n_nodes // n_classes] * n_classes),
        prob_matrix,
        num_channels=feature_dim,
        is_undirected=True,
        transform=T.Compose([*transform]),
    )
    return data


def sbm_classification_generator(graphs, nodes, transforms, feature_dim):
    num_of_graphs = len(graphs)

    def sample_graph():
        multi = multinomial(1, [1 / num_of_graphs] * num_of_graphs)
        choice = np.argmax(multi.rvs(1))
        x = sbm(nodes, graphs[choice]["prob_matrix"], feature_dim)
        x.y = torch.tensor([choice.item()])
        return transforms(x)

    return sample_graph()


def barasi_albert(nodes, edges, transforms=None):
    edge_index = barabasi_albert_graph(nodes, edges)
    data = Data(num_nodes=nodes, edge_index=edge_index)
    data.num_nodes = nodes
    if transforms is not None:
        return transforms(data)
    return data


class MyDataset(Dataset):
    def __init__(self, samples, num_classes=None):
        self.samples = samples
        self.first_sample = samples[0]
        self._num_classes = num_classes

    @property
    def num_features(self):
        return self.first_sample.num_features

    def shuffle(self):
        random.shuffle(self.samples)
        return self

    @property
    def num_classes(self):
        if (
            self._num_classes is None
        ):  # inductive community detection classes are known by single data sample but for graph classification need to be set manually
            return self.first_sample.num_features
        return self._num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MyDataset(self.samples[idx], self.num_classes)
        return self.samples[idx]


def barasi_albert_classification(graphs, samples, transforms):
    num_of_graphs = len(graphs)

    def sample_graph():
        multi = multinomial(1, [1 / num_of_graphs] * num_of_graphs)
        choice = np.argmax(multi.rvs(1))
        x = barasi_albert(graphs[choice]["nodes"], graphs[choice]["edges"])
        x.y = torch.tensor([choice.item()])
        return transforms(x)

    samples = [sample_graph() for _ in range(samples)]
    return MyDataset(samples, num_classes=num_of_graphs)
