import numpy as np
import torch
from sklearn.cluster import KMeans

from gnn.metric import Metric, aggregate_list_of_metric, MetricList
from gnn.permutation_invariant_loss import PermutationInvariantAccuracy


def cluster_perm_invariant(X, labels, num_clusters, random_state=0, init="k-means++"):
    kmeans = KMeans(n_clusters=num_clusters, init=init, random_state=random_state, n_init=10).fit(X)
    predictions = kmeans.labels_
    preds = torch.from_numpy(predictions)
    labls = torch.Tensor(labels).long()
    if torch.cuda.is_available():
        preds = preds.cuda()
        labls = labls.cuda()
    return PermutationInvariantAccuracy(num_clusters, "cuda")(preds, labls)


def cluster(X, labels, num_clusters, random_state=0, init="k-means++"):
    kmeans = KMeans(n_clusters=num_clusters, init=init, random_state=random_state, n_init=10).fit(X)
    predictions = kmeans.labels_
    acc = np.mean(predictions == labels)
    return max(acc, 1 - acc)


def clustering_with_model_embedding(model, test_dataset, num_classes, run_times, cut_off=None):
    model.eval()
    metrics1 = []
    metrics2 = []
    for run in range(run_times):
        labels = []
        encodings = []
        for sample in test_dataset:
            d, label = sample, sample.y.item()
            labels.append(label)
            encoding = model.encode(**d.to_dict())
            encoding = encoding.detach().cpu().numpy().flatten()

            if cut_off is not None:
                encoding = np.partition(encoding, -cut_off)[-cut_off]
            encodings.append(encoding)

        encodings = np.vstack(encodings)
        for init in ["k-means++", "random"]:
            acc_perm_inv = cluster_perm_invariant(encodings, labels, num_classes, run, init).item()
            acc = cluster(encodings, labels, num_classes, run, init).item()
            init == "k-means++" and metrics1.append(Metric(f"{init}", {f"perm_inv_acc": acc_perm_inv, f"acc": acc}))
            init == "random" and metrics2.append(Metric(f"kmeans-{init}", {f"perm_inv_acc": acc_perm_inv, f"acc": acc}))
    aggregated1 = aggregate_list_of_metric(metrics1)
    aggregated2 = aggregate_list_of_metric(metrics2)
    return MetricList([aggregated1, aggregated2])
