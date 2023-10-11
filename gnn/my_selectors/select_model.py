import torch

torch.random.manual_seed(0)
from torch_geometric.nn import GAE

from gnn.models.encoders.random_projection import RandomProjection
from gnn.models.encoders.spectral_embedding import (
    SpectralEmbeddingModel,
    InnerProductDecoderLinear,
)
from gnn.models.encoders.svd import SVD
from gnn.models.graph_classification.gcn_sumpool_concatenate_threelayers_batchnorm import (
    GCN_SumPool_Concatenate_ThreeLayers_BatchNorm,
)
from gnn.models.graph_classification.gcn_sumpool_threelayers import (
    GCN_SumPool_ThreeLayers,
)
from gnn.models.graph_classification.gcn_sumpooling_onelayer import (
    GCN_SumPooling_OneLayer,
    GCN_SumPooling_OneLowRankLayer,
)
from gnn.models.graph_classification.gcn_sumpooling_twolayers import (
    GCN_SumPooling_TwoLayers,
)
from gnn.models.node_classification.gat.gat import GAT
from gnn.models.node_classification.gat.my_gat import MyGAT
from gnn.models.node_classification.gat.pytorch_geometric_gat import PytorchGeometricGAT
from gnn.models.node_classification.gcn import GCN
from gnn.models.node_classification.gnn4cd.compact_hashimoto import (
    CompactHashimoto,
    CompactHashimoto2,
    CompactHashimoto1,
)
from gnn.models.node_classification.gnn4cd.gnn4cd import LineGnn, Gnn


def select_model(model_config, feature_dim, classes):
    match model_config.name:
        case "gcn":
            model = GCN(
                model_config.layers,
                feature_dim,
                model_config.hidden,
                classes,
                model_config.dropout,
                model_config.concat,
                model_config.batchnorm,
                model_config.nonlinearity,
            )
        case "reference_gat":
            model = PytorchGeometricGAT(
                feature_dim,
                model_config.hidden,
                model_config.heads,
                classes,
                model_config.dropout,
            )
        case "gat":
            model = GAT(
                feature_dim,
                model_config.hidden,
                model_config.heads,
                classes,
                model_config.dropout,
                model_config.layers,
                model_config.concat,
                model_config.untied,
            )
        case "my_gat":
            model = MyGAT(
                feature_dim,
                model_config.hidden,
                model_config.heads,
                classes,
                model_config.dropout,
                model_config.layers,
                model_config.symmetric,
                model_config.concat,
                model_config.untied,
            )
        case "compact-hashimoto":
            model = CompactHashimoto(
                feature_dim,
                model_config.hidden,
                classes,
                model_config.dropout,
                model_config.num_nodes,
            )
        case "compact-hashimoto1":
            model = CompactHashimoto1(
                feature_dim,
                model_config.hidden,
                classes,
                model_config.dropout,
            )
        case "compact-hashimoto2":
            model = CompactHashimoto2(
                feature_dim,
                model_config.hidden,
                classes,
                model_config.dropout,
            )
        case "gnn":
            model = Gnn(
                feature_dim,
                model_config.hidden,
                model_config.layers,
                classes,
                model_config.hierachy,
            )
        case "lgnn" | "compact-lgnn":
            model = LineGnn(
                feature_dim,
                model_config.hidden,
                model_config.layers,
                classes,
                model_config.hierachy,
                model_config.hierachy_projections,
            )
        case "gcn-pooling":
            model = GCN_SumPooling_OneLayer(
                feature_dim,
                model_config.hidden,
                classes,
                model_config.dropout,
            )
        case "gcn-pooling-lowrank":
            model = GCN_SumPooling_OneLowRankLayer(
                classes,
                model_config.dropout,
                model_config.hidden,
            )
        case "gcn-pooling2":
            model = GCN_SumPooling_TwoLayers(
                feature_dim,
                model_config.hidden,
                classes,
                model_config.dropout,
            )
        case "gcn-pooling3":
            model = GCN_SumPool_ThreeLayers(
                feature_dim,
                model_config.hidden,
                classes,
                model_config.dropout,
            )

        case "gcn-pooling3-2":
            model = GCN_SumPool_Concatenate_ThreeLayers_BatchNorm(
                feature_dim,
                model_config.hidden,
                classes,
                model_config.dropout,
            )
        case "gae":
            match model_config.encoder:
                case "dum":
                    encoder = RandomProjection(feature_dim, model_config.hidden)
                case "gcn-one":
                    encoder = GCN(feature_dim, model_config.hidden)
                case _:
                    raise ValueError("Encoder is not recognized")
            model = GAE(encoder)
        case "spectral":
            model = GAE(SpectralEmbeddingModel(model_config.hidden), InnerProductDecoderLinear())
        case "svd":
            model = SVD(model_config.hidden)
        case _:
            raise ValueError("Model is not recognized")
    return model
