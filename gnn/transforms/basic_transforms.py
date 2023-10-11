import warnings

import numpy as np
import torch
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.utils import (
    remove_self_loops,
    to_dense_adj,
)

from gnn.datasets.dataset_split import split_data_transductive
from gnn.device import DEVICE
from gnn.utils import torch_sparse_identitiy, adj_from_edge_index


class OneHotToVector(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        asd = data.y
        data.y = torch.argmax(data.y, dim=1)
        return data


class ToFloat(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.x = data.x.float()
        return data


class NullTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return data


class ValidateInductive(BaseTransform):
    def __init__(self):
        super().__init__()

    def unset_mask(self, data, mask_name):
        if data.get(mask_name) is not None:
            data[mask_name] = None

    def __call__(self, data):
        self.unset_mask(data, "train_mask")
        self.unset_mask(data, "val_mask")
        self.unset_mask(data, "test_mask")
        return data


class LoadScalings(BaseTransform):
    def __init__(self, dataset):
        super().__init__()
        gammas = np.load(f"adjacency_scalings/{dataset}.npz")
        self.gammas = [torch.diag(torch.from_numpy(gammas[gamma])).to(DEVICE) for gamma in gammas.files]

    def __call__(self, data):
        adj = data.adj.to_dense()
        if len(self.gammas) == 1:
            adj = self.gammas[0] @ adj @ self.gammas[0]
        else:
            adj = self.gammas[0] @ adj @ self.gammas[1]
        data.adj = adj.to_sparse_coo()
        return data


class SetAdj(BaseTransform):
    def __init__(self, edge_weight=1):
        super().__init__()
        warnings.warn("Edge Weights are set to 1 automatically")
        if edge_weight != 1:
            NotImplementedError("edge_weight != 1 not implemented")

    def __call__(self, data):
        data.adj = adj_from_edge_index(data.edge_index, data.num_nodes)
        return data


class IdentitiyFeatures(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.x = torch_sparse_identitiy(data.num_nodes)
        return data


class SparseConstant(BaseTransform):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def __call__(self, data):
        sparsity = 0.5
        sparsity_mask = (torch.FloatTensor(data.num_nodes, self.num_features).uniform_() > sparsity).float()
        scale = torch.normal(torch.ones((data.num_nodes, self.num_features)), torch.Tensor([0.2]))
        x = torch.ones((data.num_nodes, self.num_features)) * sparsity_mask * scale
        data.x = x
        return data


class Constant(BaseTransform):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def __call__(self, data):
        data.num_features = self.num_features
        x = torch.normal(torch.zeros((data.num_nodes, self.num_features)), torch.Tensor([1]))
        x = torch.nn.functional.normalize(x) * 5
        data.x = x

        return data


class ConstantFeaturesAllOnes(BaseTransform):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def __call__(self, data):
        data.num_features = self.num_features
        data.x = torch.ones((data.num_nodes, self.num_features))

        return data


class RemoveSelfLoops(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.adj = adj_from_edge_index(data.edge_index, data.num_nodes)
        return data


# https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html?highlight=nell#torch_geometric.datasets.NELL
class FeaturesToSparse(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.x = data.x.to_torch_sparse_coo_tensor()
        return data


class FeaturesToDense(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.x = data.x.to_dense()
        return data


class TransductiveDataSplit(BaseTransform):
    def __init__(self, test_val_split, unlabelled=0, random_seed=0):
        super().__init__()
        self.test_val_split = test_val_split
        self.unlabelled = unlabelled
        self.LOG = True
        self.random_seed = random_seed

    def __call__(self, data):
        # data.num_nodes is not correct
        data.num_nodes = data["y"].shape[0]
        data.classes = data["y"].max().item() + 1
        features = data.get("x", None)
        if features is None:
            data.x = torch_sparse_identitiy(data.num_nodes)
        if hasattr(data, "train_mask") and hasattr(data, "test_mask") and len(data.train_mask.shape) == 1:
            data.has_dataset_dependent_split = True
            if self.test_val_split is not None:
                warnings.warn(
                    "test_val_split is ignored because train_mask and test_mask are provided by the dataset. Some datasets like Cora could not have labels for all nodes!"
                )
        else:
            data["train_mask"] = None
            data["val_mask"] = None
            data["test_mask"] = None
            data.has_dataset_dependent_split = False
            splits = split_data_transductive(data.num_nodes, self.test_val_split, self.unlabelled, self.random_seed)
            # this thing gets called for every training step so we need to make sure the randon splits are the same
            # therefore the random_seed is set once and then passed to the split_data function
            if len(splits) == 1:
                data.train_mask = splits[0]
                if hasattr(data, "train_mask"):
                    pass
            if len(splits) == 2:
                data.train_mask, data.test_mask = splits
            if len(splits) == 3:
                data.train_mask, data.val_mask, data.test_mask = splits
        if self.LOG:
            print(
                "Train Mask: ",
                (data["train_mask"].sum() / data["train_mask"].shape[0]).item(),
            )
            data.get("val_mask") is not None and print(
                "Val Mask: ",
                (data["val_mask"].sum() / data["val_mask"].shape[0]).item(),
            )
            print(
                "Test Mask: ",
                (data["test_mask"].sum() / data["test_mask"].shape[0]).item(),
            )
            self.LOG = False
        return data
