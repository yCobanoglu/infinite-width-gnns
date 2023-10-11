import torch
from torch.nn import functional as F


def select_nonlinearity(nonlinearity: str):
    match nonlinearity:
        case "elu":
            nonlin = F.elu
        case "relu":
            nonlin = F.relu
        case "sigmoid":
            nonlin = F.sigmoid
        case "tanh":
            nonlin = F.tanh
        case "identity":
            nonlin = torch.nn.Identity()
        case _:
            raise ValueError(f"Unknown nonlinearity {nonlinearity}")
    return nonlin
