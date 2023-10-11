import torch
from torcheval.metrics import R2Score

from gnn.device import DEVICE
from gnn.permutation_invariant_loss import (
    PermutationInvariantLoss,
    PermutationInvariantAccuracy,
)


def metric_over_mask(metric, pred, y, mask=None):
    if mask is not None:
        return metric(pred[mask], y[mask])
    return metric(pred, y)


def default_accuracy(output, labels):
    output = torch.argmax(output, dim=1)
    return (output == labels).float().mean()


def r_squared(output, labels):
    output = output.squeeze()
    metric = R2Score()
    return metric.update(output.cpu(), labels.cpu()).compute().to(DEVICE)


def mse_loss():
    mse = torch.nn.MSELoss()

    def _loss(output, labels):
        output = output.squeeze()
        try:
            labels1 = torch.nn.functional.one_hot(labels).float()
            labels1 = torch.nn.functional.one_hot(labels, num_classes=output.shape[1]).float()
        except RuntimeError:
            labels1 = labels
        return mse(output, labels1)

    return _loss


def select_loss(loss, classes):
    match loss:
        case "mse":
            selected = mse_loss()
        case "cross_entropy":
            selected = torch.nn.CrossEntropyLoss()
        case "permutation_inv_cross_entropy":
            selected = PermutationInvariantLoss(classes, DEVICE)
        case _:
            raise ValueError(f"Unknown loss {loss}")
    return lambda pred, y, mask=None: metric_over_mask(selected, pred, y, mask)


def select_acc(acc, classes):
    match acc:
        case "r_squared":
            selected = r_squared
        case "default":
            selected = default_accuracy
        case "permutation_inv_acc":
            selected = PermutationInvariantAccuracy(classes, DEVICE)
        case _:
            raise ValueError(f"Unknown loss {acc}")
    return lambda pred, y, mask=None: metric_over_mask(selected, pred, y, mask)
