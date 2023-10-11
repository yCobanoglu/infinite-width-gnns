import dataclasses
from typing import Optional


class OptimizerConfig:
    def __init__(
        self,
        optimizer_name: str,
        lr: float,
        weight_decay: float = 0,
        clip_gradient_norm: float = None,
    ):
        try:
            self.name = optimizer_name
            self.clip_gradient_norm = clip_gradient_norm
            self.lr = lr
            self.weight_decay = weight_decay
        except ValueError as e:
            raise ValueError(f"Invalid data config: {e}", e)

    def __str__(self):
        basic = f"{self.name} ,lr:{self.lr}"
        return basic


@dataclasses.dataclass
class Sparsifier:
    name: str
    rate: float


class ModelConfig:
    def __init__(
        self,
        _id: str,
        model_name: str,
        epochs: int,
        hidden: Optional[int] = None,
        encoder: Optional[str] = None,
        batch_size: Optional[int] = None,
        layers: Optional[int] = None,
        dropout: Optional[float] = None,
        sparsify: Optional[dict] = None,
        hierachy: Optional[int] = None,
        hierachy_projections: Optional[int] = None,
        optimizer: Optional[dict] = None,
        heads: Optional[int] = None,
        concat: Optional[bool] = None,
        symmetric: Optional[bool] = None,
        untied: Optional[bool] = None,
        adjacency: Optional[str] = None,
        embedding_topk: Optional[int] = None,
        num_layers: Optional[int] = None,
        batchnorm: Optional[bool] = None,
        nonlinearity: Optional[str] = None,
    ):
        self._id = _id
        try:
            self.name = model_name
            self.encoder = encoder
            self.epochs = epochs
            self.hidden = hidden
            self.dropout = dropout
            self.heads = heads
            self.concat = concat
            self.symmetric = symmetric
            self.untied = untied
            self.layers = layers
            self.batch_size = batch_size
            self.adjacency = adjacency
            self.optimizer = OptimizerConfig(**optimizer) if optimizer is not None else None
            self.sparsify = Sparsifier(**sparsify) if sparsify is not None else None
            self.embedding_topk = embedding_topk
            self.num_layers = num_layers
            self.batchnorm = batchnorm
            self.nonlinearity = nonlinearity

        except ValueError as e:
            raise ValueError(f"Invalid data config: {e}", e)

        self.hierachy = hierachy
        self.hierachy_projections = hierachy_projections
        self.num_features = None
        self.classes = None
        self.nodes = None
        self.parameters = None

    @property
    def summarize_name(self):
        adj = f"\n{self.adjacency}" if self.adjacency else ""
        sparsify = "\n" + str(self.sparsify) if self.sparsify else ""
        encoder = f"\n{self.encoder}" if self.encoder else ""
        return self.name + adj + encoder + "\n" + str(self.optimizer) + sparsify
