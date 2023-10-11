# adapted from https://rajatvd.github.io/NTK/ to plot figures
import gnn.device as device

device.DEVICE = "cpu"
from gnn.models.node_classification.gat.gat import GAT
from gnn.models.node_classification.gat.my_gat import MyGAT
from gnn.models.node_classification.gat.pytorch_geometric_gat import PytorchGeometricGAT


import torch
from matplotlib.pyplot import title, plot, xlabel, ylabel, legend, show, savefig
from numpy.linalg import norm
from torch import nn, vmap, optim
from torch._functorch.eager_transforms import jacrev, vjp, jvp
from torch._functorch.make_functional import make_functional

from gnn.config.data_config import DataConfig
from gnn.device import DEVICE
from gnn.models.node_classification.gcn import GCN
from gnn.my_selectors.select_data import select_dataset
from gnn.transforms.adjacency_transforms import WellingNormalized, MLP, AddEye


def empirical_ntk_ntk_vps(func, params, x):
    def get_ntk(x):
        def func_x(params):
            return func(params, x)

        output, vjp_fn = vjp(func_x, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x, (params,), vjps)
            return jvps

        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)

    return get_ntk(x)


def empirical_ntk_jacobian_contraction(fnet_single, params, x):
    # Compute J(x1)
    jac1 = jacrev(fnet_single, chunk_size=1)(params, x)
    jac1 = torch.concat([j.flatten(1).detach().cpu() for j in jac1], dim=1)
    ntk = jac1 @ jac1.T
    return ntk.detach().cpu().numpy().copy()


def fnet_forward(fnet, params, data):
    return fnet(params, **data)


if __name__ == "__main__":
    dataloaders = select_dataset(
        DataConfig(dataset="karate", data_split=[0.2, 0.0]),
        "transductive",
        1,
        [AddEye()],
    )

    data = dataloaders[0][0]
    adj = data.adj.to_dense()
    # adj = torch.eye(adj.shape[0])
    adj.requires_grad_(False)
    data.adj = adj
    x = data.x.to_dense()
    input = x.shape[1]
    y = data.y
    classes = int(y.max().item() + 1)
    y = torch.nn.functional.one_hot(y).float()

    ntks = {}
    weights_changes = {}
    losses = {}
    EPOCHS = 150
    lr = 1e-3
    WIDTHS = [8, 64, 128]
    COLORS = ["b", "y", "k"] # for plotting

    if len(COLORS) != len(WIDTHS):
        raise ValueError("COLORS and WIDTHS must have the same length")

    for width in WIDTHS:
        # MODEL_NAME = "FCN"
        # model = GCN(2, input, width, classes, None, False, False).to(DEVICE)
        #model = GCN(2, input, width, classes, None, False, False).to(DEVICE)  # Skip Concatenate GNN
        MODEL_NAME = "GAT"

        #model = GAT(input, width, heads=width, num_classes=classes, dropout=None, layers=2, concat=False).to(DEVICE)
        model = PytorchGeometricGAT(input, width, heads=width, num_classes=classes, dropout=None).to(DEVICE)
        opt = optim.SGD(model.parameters(), lr=lr)
        fnet, params = make_functional(model)
        _fnet_forward = lambda params, data: fnet_forward(fnet, params, data)
        ntk0 = empirical_ntk_jacobian_contraction(_fnet_forward, params, data.to_dict())
        weights0 = nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy().copy()
        _ntks = [0]
        weights_changes_ = [0]
        lossvals = []
        for i in range(EPOCHS):
            out = model(adj, x)
            loss = nn.MSELoss()(out, y)
            print(f"Epoch {i},  Loss {round(loss.item(),2)}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            lossvals.append(loss.item())

            fnet, params = make_functional(model)
            _fnet_forward = lambda params, data: fnet_forward(fnet, params, data)
            ntk = empirical_ntk_jacobian_contraction(_fnet_forward, params, data.to_dict())

            weights = nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy().copy()
            weights_change_ = norm(weights - weights0) / norm(weights0)
            weights_changes_.append(weights_change_)
            _ntks.append(norm(ntk - ntk0) / norm(ntk0))

        weights_changes[width] = weights_changes_.copy()
        losses[width] = lossvals.copy()
        ntks[width] = _ntks.copy()




    # plot loss changes
    title("Loss")
    for width, color in zip(WIDTHS, COLORS):
        plot(range(EPOCHS), losses[width], label=f"Width={width}", c=color)
    xlabel("Epochs")
    ylabel(f"Loss")
    legend()
    savefig(f"gnn/figures/figs/{MODEL_NAME}_loss.png", bbox_inches="tight")
    show()

    # plot weight changes
    title(f"Relative norm change of weights from initialization")
    for width, color in zip(WIDTHS, COLORS):
        plot(range(EPOCHS + 1), weights_changes[width], label=f"Width={width}", c=color)
    xlabel("Epochs")
    ylabel(r"$\frac{\Vert w(t) -  w(0) \Vert}{\Vert w(0) \Vert}$")
    legend()
    savefig(f"gnn/figures/figs/{MODEL_NAME}_weightchange.png", bbox_inches="tight")
    show()

    # plot ntks changes
    title(f"Relative norm change of empirical NTK from NTK at initialization")
    for width, color in zip(WIDTHS, COLORS):
        plot(range(EPOCHS + 1), ntks[width], label=f"Width={width}", c=color)
    xlabel("Epochs")
    ylabel(r"$\frac{\Vert NTK(t) -  NTK(0) \Vert}{\Vert NTK(0) \Vert}$")
    legend()
    savefig(f"gnn/figures/figs/{MODEL_NAME}_ntk.png", bbox_inches="tight")
    show()
