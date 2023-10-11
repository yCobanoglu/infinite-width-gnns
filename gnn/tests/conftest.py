from typing import Tuple

import numpy as np
import pytest
from torch_geometric.data import Dataset
from torch_geometric.transforms import ToUndirected

from gnn.datasets.synthetic_data import sbm
from gnn.transforms.basic_transforms import (
    SetAdj,
    RemoveSelfLoops,
)

P = [
    [0, 0.045, 0.045, 0.045, 0.045],
    [0, 0.045, 0.045, 0.045],
    [0, 0.045, 0.045],
    [0, 0.045],
    [0],
]
NODES = 400

data = sbm(NODES, P, None, [ToUndirected(), RemoveSelfLoops(), SetAdj()])


@pytest.fixture
def generate_sbm_dataset() -> Tuple[Dataset, np.ndarray, np.ndarray]:
    adj = data.adj.to_dense().numpy()
    degree = np.diag(np.sum(adj, axis=0))
    return data, adj, degree


def fro_norm(X):
    return np.linalg.norm(X, ord="fro")
