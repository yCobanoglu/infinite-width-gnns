import math
import pytest
import torch

from gnn.permutation_invariant_loss import (
    PermutationInvariantAccuracy,
    PermutationInvariantLoss,
)
from gnn.tests.reference_implementations_for_testing.expected_permutation_invariant_loss import (
    compute_loss_multiclass as expected_computed_loss_multiclass,
)
from gnn.tests.reference_implementations_for_testing.expected_permutation_invariant_loss import (
    compute_accuracy_multiclass as expected_compute_accuracy_multiclass,
)

DEVICE = "cpu"


@pytest.mark.parametrize("classes,nodes", [(2, 100), (7, 10), (4, 42)])
def test_loss(classes, nodes):
    n_classes = classes
    labels = torch.randint(n_classes, (nodes,))
    pred = torch.rand(nodes, n_classes)

    exp_pred = expected_computed_loss_multiclass(pred[None, :], labels[None, :], n_classes)
    pred = PermutationInvariantLoss(n_classes, DEVICE)(pred, labels)
    assert torch.allclose(pred, exp_pred)


@pytest.mark.parametrize("classes,nodes", [(2, 100), (7, 10), (4, 42)])
def test_accuracy(classes, nodes):
    #
    n_classes = classes
    labels = torch.randint(n_classes, (nodes,))
    pred = torch.rand(nodes, n_classes)

    exp_pred = expected_compute_accuracy_multiclass(pred[None, :], labels[None, :], n_classes)
    pred = PermutationInvariantAccuracy(n_classes, DEVICE)(pred, labels).item()
    assert math.isclose(pred, exp_pred, rel_tol=1e-5)
