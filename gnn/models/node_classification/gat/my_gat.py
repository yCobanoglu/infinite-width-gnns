import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from gnn.device import DEVICE
from gnn.utils import scipy_coo_to_torch_coo


class MyGATConv(nn.Module):
    def __init__(self, in_features, out_features, symmetric=False, untied=False, dropout=0):
        super().__init__()
        self.dropout = dropout
        self.out_features = out_features
        self.symmetric = symmetric
        self.untied = untied
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.linear1 = Linear(in_features, out_features)
        self.linear2_1 = Linear(out_features, 1)
        self.linear2_2 = Linear(out_features, 1)
        self.linear3 = Linear(in_features, out_features)
        self.I_A = None
        self.I__A = None

    def sparse_diag(self, values, size):
        index = list(range(size))
        return torch.sparse_coo_tensor(torch.Tensor([index, index]).to(DEVICE), values, (size, size)).to(DEVICE)

    def init_attention_helper_torch(self, adj):
        # slower than init_attention_helper
        n = adj.size()[0]
        I = scipy.sparse.eye(n)
        one = scipy.sparse.coo_array(np.ones((n, 1)))
        I_ = scipy_coo_to_torch_coo(scipy.sparse.coo_array(scipy.sparse.kron(one, I))).to(DEVICE)
        I__ = scipy_coo_to_torch_coo(scipy.sparse.coo_array(scipy.sparse.kron(I, one))).to(DEVICE)
        a_flat = adj.to_dense().flatten()
        a_flat_diag = self.sparse_diag(a_flat, n**2)
        I_A = a_flat_diag @ I_
        I__A = a_flat_diag @ I__
        self.I_A = I_A.to(DEVICE)
        self.I__A = I__A.to(DEVICE)

    def init_attention_helper(self, adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        n = adj.size()[0]
        I = scipy.sparse.eye(n)
        a_flat = adj.cpu().numpy().reshape(-1, 1).squeeze(1)
        a_flat_diag = scipy.sparse.diags([a_flat], [0])
        one = scipy.sparse.coo_array(np.ones((n, 1)))
        I_ = scipy.sparse.coo_array(scipy.sparse.kron(one, I))
        I_A = a_flat_diag @ I_
        I__ = scipy.sparse.coo_array(scipy.sparse.kron(I, one))
        I__A = a_flat_diag @ I__
        self.I_A = scipy_coo_to_torch_coo(scipy.sparse.coo_array(I_A)).to(DEVICE).coalesce().to_sparse_csr()
        self.I__A = scipy_coo_to_torch_coo(scipy.sparse.coo_array(I__A)).to(DEVICE).coalesce().to_sparse_csr()

    def forward(self, input, adj):
        input_ = F.tanh(input)
        input1 = F.elu(input)
        n = adj.size()[0]
        h = self.linear1(input1)
        if self.symmetric:
            g1 = self.linear2_1(h)
            g2 = self.linear2_1(h)
        else:
            g1 = self.linear2_1(h)
            g2 = self.linear2_2(h)
        if self.I_A is None and self.I__A is None:
            self.init_attention_helper(adj)
        attention_adj_flatten_1 = torch.mv(self.I_A, g1.squeeze(1))
        attention_adj_flatten_2 = torch.mv(self.I__A, g2.squeeze(1))
        attention_adj_flatten = attention_adj_flatten_1 + attention_adj_flatten_2
        attention_adj = attention_adj_flatten.to_dense().view(n, n)
        e = self.leakyrelu(attention_adj)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        if self.untied:
            return attention @ self.linear3(input_)
        return attention @ h


class MyGAT(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden,
        heads,
        num_classes,
        dropout,
        layers,
        symmetric,
        concat,
        untied=False,
    ):
        super().__init__()
        self.in_features = num_features
        self.concat = concat
        self.hid = hidden
        self.heads = heads
        self.layers = layers
        self.num_classes = num_classes
        self.dropout = 0.0 if dropout is None else dropout

        Conv = lambda inp, out: MyGATConv(inp, out, dropout=dropout, symmetric=symmetric, untied=untied)

        for l in range(layers):
            for head in range(self.heads):
                if l == 0:
                    self.add_module(f"layer-{l}-head-{head}", Conv(num_features, self.hid))
                else:
                    in_features = self.hid * self.heads if self.concat else self.hid
                    if l == layers - 1:
                        if self.concat:
                            self.add_module(
                                f"layer-{l}-head-{1}",
                                Conv(in_features, self.num_classes),
                            )
                            break
                        else:
                            self.add_module(
                                f"layer-{l}-head-{head}",
                                Conv(in_features, self.num_classes),
                            )
                    self.add_module(f"layer-{l}-head-{head}", Conv(self.in_features, self.hid))

    def forward(self, adj, x, *args, **kwargs):
        ## only to compute empirical NTK
        # if adj.is_sparse:
        # adj = adj.coalesce().to_dense()
        for l in range(self.layers):
            output = []
            if l == self.layers - 1 and self.concat:
                x = self._modules[f"layer-{l}-head-{1}"](x, adj)
                return x
            for head in range(self.heads):
                x = F.dropout(x, p=self.dropout, training=self.training)
                output.append(self._modules[f"layer-{l}-head-{head}"](x, adj))
            if self.concat:
                output = torch.cat(output, dim=1)
            else:
                output = sum(output) / self.heads
            if l != self.layers - 1:
                #output = F.elu(output)
                x = output
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                return output
