from pathlib import Path

import torch
from torch_geometric.transforms import RandomLinkSplit, BaseTransform
from tqdm import tqdm

from gnn.config.config import Config
from gnn.device import DEVICE
from gnn.metric import Metric, MetricList
from gnn.my_selectors.select_data import select_dataset
from gnn.my_selectors.select_model import select_model
from gnn.my_selectors.select_optimizer import select_optimizer
from gnn.my_selectors.select_sparsification import select_sparsification
from gnn.runners.custom.custering import clustering_with_model_embedding
from gnn.utils import count_parameters


def train(model, optimizer, datasets, epochs, batch_size=1, print_verbose=True):
    model.train()
    variational = hasattr(model, "variational") and model.variational
    train_dataset, val_dataset, _ = datasets
    for epoch in tqdm(range(1, epochs + 1)):
        loss_per_epoch = 0
        for counter, sample in enumerate(train_dataset):
            z = model.encode(**sample.to_dict())
            loss = model.recon_loss(z, sample.pos_edge_label_index)
            if variational:
                loss += loss + (1 / sample.num_nodes) * model.kl_loss()
            loss_per_epoch += loss
            loss /= batch_size
            loss.backward()
            if (counter + 1) % batch_size == 0 or counter + 1 == len(train_dataset):
                optimizer.step()
                optimizer.zero_grad()

        if val_dataset is not None:
            val_auc, val_ap = eval_on_test(model, val_dataset)

        print_verbose and print(
            f"Epoch: {epoch:03d}, Loss {loss_per_epoch/ len(train_dataset):.4f}"
            + (f", AUC: {val_auc:.4f}, AP: {val_ap:.4f}" if val_dataset else "")
        )
    return Metric("train", {"loss": loss_per_epoch / len(train_dataset)})


@torch.no_grad()
def eval_on_test(model, test_dataset):
    model.eval()
    auc, ap = 0, 0
    for data in test_dataset:
        if data.get("neg_edge_label_index") is None:
            print("wtf")
            continue
        z = model.encode(**data.to_dict())
        auc_single, ap_single = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        auc += auc_single
        ap += ap_single
    return auc / len(test_dataset), ap / len(test_dataset)


class AddNegativeTrainSamples(BaseTransform):
    # add negative examples used for GAE (no need for test and val split)
    def __init__(self):
        super().__init__()
        self.t = RandomLinkSplit(
            num_val=0,
            num_test=0,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=True,
        )

    def __call__(self, data):
        return self.t(data)[0]


def run(config, print_verbose=True):
    transforms = []
    if config.data.sparsify:
        transforms = [select_sparsification(config.data.sparsify)]

    transforms = [*transforms, AddNegativeTrainSamples()]

    dataloaders = select_dataset(config.data, config.learning_task, config.model.epochs, transforms)
    train_dataset, val_dataset, test_dataset = dataloaders

    num_classes = train_dataset.num_classes
    num_features = train_dataset.num_features
    model = select_model(config.model, num_features, None)

    config.model.parameters = count_parameters(model)

    model.to(DEVICE)

    optimizer = select_optimizer(model.parameters(), config.model.optimizer)

    metrics = []
    if config.model.epochs:
        metric1 = train(
            model,
            optimizer,
            dataloaders,
            config.model.epochs,
            config.model.batch_size,
            print_verbose,
        )
        metrics.append(metric1.aggregate_single())

    test_auc, test_ap = eval_on_test(model, test_dataset)
    metric2 = Metric("test", {"AUC": test_auc, "AP": test_ap})

    assert num_classes > 1
    RUN_CLUSTERING = 10
    clustering_performance = clustering_with_model_embedding(
        model,
        test_dataset,
        num_classes,
        RUN_CLUSTERING,
        cut_off=config.model.embedding_topk,
    )
    config.set_metrics(MetricList([*metrics, metric2.aggregate_single(), *clustering_performance.metrics]))
    print_verbose and print(clustering_performance)
    return config


if __name__ == "__main__":
    logs_dir = Path(__file__).parent / "logs"
    config_path = Path(__file__).parent / "config.yaml"
    run(Config.from_file(config_path, logs_dir))
