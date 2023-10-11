import itertools

import torch


class PermutationInvariantMetric:
    def __init__(self, n_classes, device=None):
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.n_classes = n_classes
        permutations = list(itertools.permutations(range(n_classes), r=n_classes))
        self.permutations = torch.tensor(permutations, dtype=torch.long)
        if device == "cuda":
            self.permutations = self.permutations.cuda()


class PermutationInvariantLoss(PermutationInvariantMetric):
    def __init__(self, n_classes, device):
        super().__init__(n_classes, device)

    def __call__(self, pred, labels):
        labels_under_perm = self.permutations[:, labels].T
        expand_dim = pred.shape[0], pred.shape[1], self.permutations.shape[0]
        pred = pred.unsqueeze(2).expand(expand_dim)
        losses = self.criterion(pred, labels_under_perm)
        loss = torch.min(losses.sum(0)) / pred.shape[0]
        return loss


class PermutationInvariantAccuracy(PermutationInvariantMetric):
    def __init__(self, n_classes, device):
        super().__init__(n_classes, device)

    def __call__(self, pred, labels):
        if len(pred.shape) == 1:
            # no probabilities, just labels
            pred_labels = pred
        else:
            pred_labels = torch.argmax(pred, dim=1)
        preds = pred_labels.unsqueeze(1).expand((pred_labels.shape[0], self.permutations.shape[0]))
        labels_under_perm = self.permutations[:, labels].T
        accuracies = (preds == labels_under_perm).float().mean(0)
        acc = torch.max(accuracies)
        # TODO unsure about necessety of this line
        # https://github.com/zhengdao-chen/GNN4CD/issues/11
        acc = (acc - 1 / self.n_classes) / (1 - 1 / self.n_classes)
        return acc
