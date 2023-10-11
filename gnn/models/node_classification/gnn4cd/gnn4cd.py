import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn


def tensor_multiply(adj_tensor, x):
    outputs = []
    x = x.to_dense()
    for index in range(adj_tensor.shape[2]):
        adj = torch.select(adj_tensor, 2, index)
        output = torch.mm(adj, x)
        outputs.append(output.unsqueeze(2))
    result = torch.cat(outputs, 2)
    return result


class GnnLastLayer(nn.Module):
    def __init__(self, in1, out):
        super().__init__()
        self.fc1 = Linear(in1, out)

    def forward(self, WW, x):
        x = tensor_multiply(WW, x)
        x = x.reshape((x.shape[0], -1))
        return self.fc1(x)


class GnnLayer(nn.Module):
    def __init__(self, in1, out):
        super().__init__()
        self.fc1 = Linear(in1, out // 2)
        self.fc2 = Linear(in1, out // 2)
        self.bn = nn.BatchNorm1d(out, track_running_stats=False)

    def forward(self, WW, x):
        x = tensor_multiply(WW, x)
        x = x.reshape((x.shape[0], -1))
        x1 = F.relu(self.fc1(x))
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        x = self.bn(x)
        return WW, x


class Gnn(nn.Module):
    def __init__(self, features_dim, hidden_dim, num_layers, n_classes, hierachy):
        super().__init__()
        self.features_dim = features_dim
        self.num_layers = num_layers
        self.layer0 = GnnLayer(hierachy * features_dim, hidden_dim)
        for i in range(num_layers):
            self.add_module(f"layer{i+1}", GnnLayer(hidden_dim * hierachy, hidden_dim))
        self.layer_final = GnnLastLayer(hidden_dim * hierachy, n_classes)

    def forward(self, adj, x, *args, **kwargs):
        cur = self.layer0(adj, x)
        for i in range(self.num_layers):
            cur = self._modules["layer{}".format(i + 1)](*cur)
        return self.layer_final(*cur)


class LineGraphModule(nn.Module):
    def __init__(self, in1, in2, out):
        super().__init__()
        self.weights_1 = Linear(in1, out)
        self.weights_2 = Linear(in2, out)
        self.weights_1_skip = Linear(in1, out)
        self.weights_2_skip = Linear(in2, out)
        self.bn = nn.BatchNorm1d(
            out * 2, out, track_running_stats=False
        )  # Normalize only across input not taking into account different samples (one input is all nodes of one graph) -> Normalize across one graph not across dataset

    def forward(self, adj, x, adj_lg_x, projections):
        y1 = tensor_multiply(adj, x)
        y1 = y1.reshape((y1.shape[0], -1))
        y2 = tensor_multiply(projections, adj_lg_x)
        y2 = y2.reshape((y2.shape[0], -1))
        non_linear = F.relu(self.weights_1(y1) + self.weights_2(y2))
        skip_conn = self.weights_1_skip(y1) + self.weights_2_skip(y2)
        output = torch.cat((non_linear, skip_conn), 1)
        return self.bn(output)


class LineGraphLayer(nn.Module):
    def __init__(self, in1, in2, out1, out2):
        super().__init__()
        self.layer1 = LineGraphModule(in1, in2, out1 // 2)
        self.layer2 = LineGraphModule(in1, 2 * out1, out2 // 2)

    def forward(self, adj, x, adj_lg, adj_lg_x, projections):
        y1 = self.layer1(adj, x, adj_lg_x, projections)
        y2 = self.layer2(adj_lg, adj_lg_x, y1, torch.transpose(projections, 1, 0))
        return adj, y1, adj_lg, y2, projections


class LineGraphLastLayer(nn.Module):
    def __init__(self, in1, in2, out):
        super().__init__()
        self.weights_1 = Linear(in1, out)
        self.weights_2 = Linear(in2, out)

    def forward(self, adj, x, adj_lg, adj_lg_x, projections):
        y1 = tensor_multiply(adj, x)
        y1 = y1.reshape((y1.shape[0], -1))
        y2 = tensor_multiply(projections, adj_lg_x)
        y2 = y2.reshape((y2.shape[0], -1))
        output = self.weights_1(y1) + self.weights_2(y2)
        return output


class LineGnn(nn.Module):
    def __init__(
        self,
        features_dim,
        hidden_dim,
        num_layers,
        n_classes,
        hierachy,
        hierachy_projections,
    ):
        super().__init__()
        self.features_dim = features_dim
        self.num_layers = num_layers
        self.layer0 = LineGraphLayer(
            features_dim * hierachy,
            features_dim * hierachy_projections,
            hidden_dim,
            hidden_dim,
        )
        for i in range(num_layers):
            dim = hidden_dim * hierachy
            self.add_module(
                f"layer{i+1}",
                LineGraphLayer(dim, hidden_dim * hierachy_projections, hidden_dim, hidden_dim),
            )
        self.layer_final = LineGraphLastLayer(dim, hidden_dim * hierachy_projections, n_classes)

    def forward(self, adj, x, adj_lg, adj_lg_x, projections, *args, **kwargs):
        cur = self.layer0(adj, x, adj_lg, adj_lg_x, projections)
        for i in range(self.num_layers):
            cur = self._modules["layer{}".format(i + 1)](*cur)
        return self.layer_final(*cur)
