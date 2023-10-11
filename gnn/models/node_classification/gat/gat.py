import torch
import torch.nn.functional as F
from gnn.models.node_classification.gat.custom_gat_conv import CustomGATConv
from gnn.my_selectors.select_nonlinearity import select_nonlinearity


class GAT(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden,
        heads,
        num_classes,
        dropout,
        layers,
        concat=False,
        untied=False,
        nonlinearity="relu",
    ):
        super().__init__()
        self.nonlinear = select_nonlinearity(nonlinearity)
        self.hid = hidden
        self.head = heads
        self.layers = layers
        self.num_classes = num_classes
        self.dropout = 0.0 if dropout is None else dropout

        self.add_module(
            f"layer-0",
            CustomGATConv(
                num_features,
                self.hid,
                heads=heads,
                concat=concat,
                dropout=self.dropout,
                add_self_loops=False,
                untied=untied,
            ),
        )

        for l in range(self.layers - 1)[1:]:
            num_features = heads * self.hid if concat is True else self.hid
            self.add_module(
                f"layer-{l}",
                CustomGATConv(
                    num_features,
                    self.hid,
                    heads=heads,
                    concat=concat,
                    dropout=self.dropout,
                ),
            )

        num_features = heads * self.hid if concat is True else self.hid
        self.add_module(
            f"layer-{self.layers - 1}",
            CustomGATConv(
                num_features,
                self.num_classes,
                heads=1,
                concat=False,
                dropout=self.dropout,
            ),
        )

    def forward(self, adj, x, *args, **kwargs):
        if adj.is_sparse:
            edge_index = adj.coalesce().indices()
        else:
            adj = adj.to_dense()
            edge_index = adj.nonzero().t()
        x = x.to_dense() if x.is_sparse else x
        for l in range(self.layers):
            x = self._modules[f"layer-{l}"](x, edge_index)
            if l != self.layers - 1:
                x = self.nonlinear(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                return x
