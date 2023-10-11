import torch
from torch.optim import Adam


def select_optimizer(parameters, optimizer_config):
    if optimizer_config is None:
        return None
    optimizer_name, lr, weight_decay = (
        optimizer_config.name,
        optimizer_config.lr,
        optimizer_config.weight_decay,
    )
    match optimizer_name:
        case "adam":
            opt = Adam(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
            )
        case "sgd":
            opt = torch.optim.SGD(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
            )
        case "adamax":
            opt = torch.optim.Adamax(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
            )
        case _:
            raise ValueError("Optimizer not supported")
    return opt
