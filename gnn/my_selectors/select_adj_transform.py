from torch_geometric.transforms import AddSelfLoops

from gnn.transforms.adjacency_transforms import (
    WellingNormalized,
    MLP,
    CompactNonBacktracking,
    Hashimoto,
    LineGraph,
    GnnTransform,
    Lgnn,
    CompactLgnn,
    AddDegree,
    AddEye,
    Laplacian,
    WellingNormalized1,
    Default1,
)
from gnn.transforms.basic_transforms import NullTransform


def select_adj_transform(config_model):
    match config_model.adjacency:
        case "default+I":
            return AddEye()
        case "default+D":
            return AddDegree()
        case "mlp":
            return MLP()
        case "welling-normalized":
            return WellingNormalized()
        case "welling-normalized1":
            return WellingNormalized1()
        case "default1":
            return Default1()
        case "laplacian":
            return Laplacian()
        case "compact_hashimoto":
            return CompactNonBacktracking()
        case "hashimoto":
            return Hashimoto()
        case "line_graph":
            return LineGraph()
        case "gnn":
            return GnnTransform(config_model.hierachy - 2)
        case "lgnn":
            return Lgnn(config_model.hierachy - 2)
        case "compact_lgnn":
            return CompactLgnn(config_model.hierachy - 2)
        case "default":
            return NullTransform()
        case _:
            raise ValueError(f"Invalid Adjacency type: {config_model.adjacency}")
