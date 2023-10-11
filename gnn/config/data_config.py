from dataclasses import dataclass
from typing import Sequence, Optional, List


@dataclass
class Sparsify:
    name: str
    rate: float


class DataConfig:
    def __init__(
        self,
        dataset: str,
        data_split: Optional[Sequence[float]] = None,
        samples: Optional[int] = None,
        unlabelled: Optional[float] = None,
        nodes: Optional[int] = None,
        edges: Optional[int] = None,
        feature_dim: Optional[int] = None,
        feature_type: Optional[str] = None,
        graphs: Optional[int] = None,
        prob_matrix: Optional[List[List[float]]] = None,
        sparsify: Optional[dict] = None,
        _id: Optional[str] = None,
    ):
        self._id = _id if _id else str(id(self))
        try:
            self.name = dataset
            self.samples = samples
            self.data_split = data_split
            self.unlabelled = unlabelled
            self.nodes = nodes
            self.edges = edges
            self.feature_dim = feature_dim
            self.feature_type = feature_type
            self.graphs = graphs
            self.prob_matrix = prob_matrix
            self.sparsify = Sparsify(sparsify["name"], sparsify["rate"]) if sparsify else None
        except ValueError as e:
            raise ValueError(f"Invalid data config: {e}", e)

        self.num_classes = None
        self.num_features = None
        self.has_dataset_dependent_split = None

    @property
    def summarize_name(self):
        sparsify = f"\nsparsify: {self.sparsify.name}, {self.sparsify.rate}" if self.sparsify else ""
        return self.name + sparsify + f"\n{self._id}"
