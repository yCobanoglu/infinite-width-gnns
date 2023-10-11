import torch
from scipy.sparse import coo_matrix
from sklearn.manifold import SpectralEmbedding
from torch import Tensor

from gnn.device import DEVICE


class SpectralEmbeddingModel:
    def __init__(self, hidden):
        self.embedding = SpectralEmbedding(n_components=hidden, n_jobs=-1)

    def __call__(self, x, adj, *args, **kwargs):
        adj = adj.coalesce().cpu()
        adj_indices = adj.indices().numpy()
        adj_values = adj.values().numpy()
        adj_scipy_coo = coo_matrix((adj_values, adj_indices), shape=adj.shape)
        fitted = self.embedding.fit_transform(adj_scipy_coo)
        return torch.from_numpy(fitted).to(DEVICE)

    def parameters(self):
        return []


class InnerProductDecoderLinear(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z: Tensor, edge_index: Tensor, **args) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return value
