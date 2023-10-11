from gnn.infinite_width.gat.gat_ntk import gat_ntk, gat_ntk2, gat_ntk3, gat_ntk4  # caused segfault when not on top
import argparse

from gnn import device
device.Device = "cpu"
import itertools
import time
from argparse import ArgumentParser

import jax
import neural_tangents as nt
import numpy as np
import torch
from jax import numpy as jnp
from tabulate import tabulate


from gnn.config.data_config import DataConfig
from gnn.infinite_width.gnn_ntk import gnn_kernel, skip_gnn_kernel
from gnn.infinite_width.ntk import nngps_and_ntks
from gnn.my_selectors.select_data import select_dataset
from gnn.transforms.adjacency_transforms import WellingNormalized, AddEye, MLP
from gnn.transforms.sparsify_transforms import EffectiveResistance


def l2(pred, y):
    return 1 / 2 * (jnp.linalg.norm(pred - y) ** 2)


def R_squared(pred, y):
    return 1 - jnp.sum((pred - y) ** 2) / jnp.sum((y - jnp.mean(y)) ** 2)


def accuracy(pred, y):
    y_pred_classes = jnp.argmax(pred, axis=1)
    return jnp.mean(y_pred_classes == jnp.argmax(y, axis=1))


def get_data(dataset, transform):
    dataloaders = select_dataset(DataConfig(dataset, [0.2, 0.2]), "transductive", 1, transform)
    data = dataloaders[0][0]
    adj = data.cpu().adj.to_dense().numpy()
    print(f"Adj shape {adj.shape}")
    x = data.x.cpu().to_dense()
    x = x.cpu().numpy()
    y_single = data.y.cpu()
    try:
        y = torch.nn.functional.one_hot(y_single).float()
    except RuntimeError:
        print("One hot encoding not working probably regression problem")
        y = y_single.float().unsqueeze(1)
    classes = int(y.max().item() + 1)
    y = y.cpu().to_dense().numpy()
    train_mask, test_mask, val_mask = (
        data.train_mask.cpu().numpy(),
        data.test_mask.cpu().numpy(),
        data.val_mask.cpu().numpy() if data.get("val_mask") is not None else None,
    )
    y_train = y[train_mask]
    y_val = y[val_mask] if val_mask is not None else None
    y_test = y[test_mask]

    return adj, x, y_train, train_mask, y_val, val_mask, y_test, test_mask, classes, y


def check_nan(X):
    if jnp.isnan(X).any():
        raise ValueError("Nan in X")


def grid_search_on_regularization_term(kernel_train_train, kernel_test_train, y_train, y_test, task="classification"):
    best_performance = float("-inf")
    best_reg = None
    init_loss = None
    loss_during_best_performance = float("-inf")
    grid = np.logspace(-3, 1, 101)
    for counter, i in enumerate(grid):
        i = round(i, 3)
        predict_fn = nt.predict.gp_inference(kernel_train_train, y_train, diag_reg=i)
        try:
            #pred = kernel_test_train @ np.linalg.inv((kernel_train_train + np.eye(kernel_train_train.shape[0]) * i)) @ y_train
            pred = predict_fn("nngp", kernel_test_train)
            check_nan(pred)
        except (ValueError, FloatingPointError) as e:
            print(e)
            continue
        loss = 1 / 2 * (jnp.linalg.norm(pred - y_test) ** 2)
        if init_loss is None:
            init_loss = loss
        performance = accuracy(pred, y_test) if task == "classification" else R_squared(pred, y_test)
        if performance > best_performance:
            best_performance = performance
            best_reg = i
            loss_during_best_performance = loss
    log = {
        "train_loss_best_perf": round(loss_during_best_performance.item(), 3),
        "train_loss_no_reg": round(init_loss.item(), 3),
        "reg": best_reg,
    }
    if task == "regression":
        log["R_squared_train"] = round(best_performance.item(), 3)
    else:
        log["best_acc_train"] = round(best_performance.item(), 3)
    return log


def eval(
    kernel_train_train,
    kernel_test_train,
    y_train,
    y_test,
    diag_reg,
    task="classification",
    loss=False,
):
    predict_fn = nt.predict.gp_inference(kernel_train_train, y_train, diag_reg=diag_reg)
    pred = predict_fn("nngp", kernel_test_train)
    if loss:
        return l2(pred, y_test)
    return accuracy(pred, y_test) if task == "classification" else R_squared(pred, y_test)


def train_and_eval(
    kernel,
    train_mask,
    test_mask,
    y_train,
    y_test,
    val_mask=None,
    y_val=None,
    task="classification",
    reg=None,
):
    kernel_train_train = kernel[train_mask][:, train_mask]
    kernel_test_train = kernel[test_mask][:, train_mask]
    if reg is None:
        if val_mask is not None and y_val is not None:
            print("Find best regularization using val_mask")
            kernel_eval_train = kernel[val_mask][:, train_mask]
            y_eval = y_val
        else:
            print("Find best regularization using train_mask")
            kernel_eval_train = kernel_train_train
            y_eval = y_train
        results_dict = grid_search_on_regularization_term(kernel_train_train, kernel_eval_train, y_train, y_eval, task)
        reg = results_dict["reg"]
    else:
        print("Using prefixed regularization")
        results_dict = {"prefixed_reg": reg}
    performance = eval(kernel_train_train, kernel_test_train, y_train, y_test, reg, task)
    results_dict = {"kernel_fro": round(np.linalg.norm(kernel), 2)} | results_dict
    if task == "regression":
        results_dict["R_squared"] = round(performance.item(), 3)
    else:
        results_dict["Test acc"] = round(performance.item(), 3)
    return results_dict


def with_time(f):
    t1 = time.perf_counter()
    res = f()
    t2 = time.perf_counter()
    return *res, round(t2 - t1, 2)


if __name__ == "__main__":
    start = time.time()
    jax.config.update("jax_platform_name", "cpu")

    parser = ArgumentParser(description="GNNGP arguments")
    parser.add_argument("--models", type=str, default="gat")
    parser.add_argument("--linear", type=bool, default=False, action=argparse.BooleanOptionalAction)
    # nn: is Neural Network Gaussian Process and Neural Tangent Kernel
    # gnn: is Graph Neural Network Gaussian Process and Graph Neural Tangent Kernel
    # sgnn: is Skip-Concatenate Graph Neural Network Gaussian Process and Skip-Concatenate Graph Neural Tangent Kernel
    # gat: is Graph Attention Network Gaussian Process and Graph Attention Network Kernel with One Nonlinearity
    # gat2: is Graph Attention Network Gaussian Process and Graph Attention Network Kernel with Two Nonlinearities
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--ef", type=float, default=0)
    args = parser.parse_args()

    MODELS = args.models.split()

    ALL_DATASETS = [
        ("movie", "classification"),
        ("texas", "classification"),
        ("wisconsin", "classification"),
        ("cornell", "classification"),
        ("facebook", "classification"),
        ("wiki", "classification"),
        ("cora", "classification"),
        ("citeseer", "classification"),
        ("pubmed", "classification"),
        ("chameleon_c", "classification"),
        ("squirrel_c", "classification"),
        ("chameleon", "regression"),
        ("squirrel", "regression"),
        ("crocodile", "regression"),
    ]

    DATASETS = []
    if args.datasets == "":
        DATASETS = ALL_DATASETS
    else:
        for dataset in args.datasets.split():
            index = [name[0] for name in ALL_DATASETS].index(dataset)
            DATASETS.append(ALL_DATASETS[index])
    NONLINEAR = not args.linear

    LOG = {}
    TRANSFORMS = []
    if any(map(lambda model: "gat" in model, MODELS)):
        TRANSFORMS.append(AddEye())
    else:
        TRANSFORMS.append(WellingNormalized())
    if args.ef > 0:
        TRANSFORMS.append(EffectiveResistance(args.ef))
    print("EF:", args.ef)

    LAYERS = [2]

    for (DATASET, TASK), layer in itertools.product(DATASETS, LAYERS):
        #for SIGMA_W, SIGMA_B in [(1, 0), (1, 0.5), (1, 1), (1,2), (1,3)]:
        for SIGMA_W, SIGMA_B in [(1, 0)]:
            print("-" * 100)
            print("Experiment: ", TRANSFORMS)
            print("Layers:", layer)
            print("Dataset:", DATASET)
            (
                adj,
                x,
                y_train,
                train_mask,
                y_val,
                val_mask,
                y_test,
                test_mask,
                classes,
                y,
            ) = get_data(DATASET, TRANSFORMS)
            print("Number of Edges:", np.sum(adj > 0))
            print("NONLINEAR:", NONLINEAR)

            if TASK == "regression":
                # Critical for performance for Regression !
                SIGMA_B = 0.1

                #SIGMA_W = 1 is never used
            print("SIGMA_W:", SIGMA_W, "SIGMA_B:", SIGMA_B)
            models_with_names = []

            if "nn" in MODELS:
                nngp, ntk, time1 = with_time(lambda: nngps_and_ntks(layer, x, SIGMA_B, SIGMA_W, NONLINEAR))  # to do sigma b
                models_with_names.extend([(nngp, "nngp", time1), (ntk, "ntk", time1)])
            if "gnn" in MODELS:
                gnngp, gntk, time1 = with_time(lambda: gnn_kernel(adj, x, layer, SIGMA_B, SIGMA_W, NONLINEAR))
                models_with_names.extend([(gnngp, "gnngp", time1), (gntk, "gntk", time1)])
            if "sgnn" in MODELS:
                _layer = min(layer, 3)  # skip layer needs at least 3 layers for skip concatenate to take effect
                sgnngp, sgntk, time1 = with_time(lambda: skip_gnn_kernel(adj, x, _layer, SIGMA_B, SIGMA_W, NONLINEAR))
                models_with_names.extend([(sgnngp, "sgnngp", time1), (sgntk, "sgntk", time1)])
            if "gat-linear" in MODELS:
                gat_gp, gat_ntk_, time1 = with_time(lambda: gat_ntk(adj, x, layer, SIGMA_B, SIGMA_W, nonlinear=False))
                models_with_names.extend([(gat_gp, "gat_gp-linear", time1), (gat_ntk_, "gat_ntk-linear", time1)])
            if "gat" in MODELS:
                gat_gp, gat_ntk_, time1 = with_time(lambda: gat_ntk(adj, x, layer, SIGMA_B, SIGMA_W, NONLINEAR))
                models_with_names.extend([(gat_gp, "gat_gp", time1), (gat_ntk_, "gat_ntk", time1)])
            if "gat2" in MODELS:
                gat_gp2, gat_ntk_2, time1 = with_time(lambda: gat_ntk2(adj, x, layer, SIGMA_B, SIGMA_W))
                models_with_names.extend([(gat_gp2, "gat_gp2", time1), (gat_ntk_2, "gat_ntk2", time1)])
            if "gat3" in MODELS:
                gat_gp3, gat_ntk_3, time1 = with_time(lambda: gat_ntk3(adj, x, layer, SIGMA_B, SIGMA_W))
                models_with_names.extend([(gat_gp3, "gat_gp3", time1), (gat_ntk_3, "gat_ntk3", time1)])
            if "gat4" in MODELS:
                gat_gp4, gat_ntk_4, time1 = with_time(lambda: gat_ntk4(adj, x, layer, SIGMA_B, SIGMA_W))
                models_with_names.extend([(gat_gp4, "gat_gp4", time1), (gat_ntk_4, "gat_ntk4", time1)])

            if len(models_with_names) == 0:
                raise ValueError("Model not found")

            for kernel, name, runtime in models_with_names:
                results_log = train_and_eval(
                    kernel,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    y_train=y_train,
                    y_test=y_test,
                    val_mask=val_mask,
                    y_val=y_val,
                    task=TASK,
                )
                print(name + " finished")
                LOG[name] = results_log | {"time": runtime}

            pd_concise = tabulate(
                [[k, *v.items()] for k, v in LOG.items()],
                headers=[f"Dataset: f{DATASET} Layer: {layer} Adjacency: {TRANSFORMS}"],
                tablefmt="grid",
            )
            print(pd_concise)
        print("Total Time :", round((time.time() - start) / 60, 2))
