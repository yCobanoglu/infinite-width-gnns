from pathlib import Path

import torch
from torch_geometric.transforms import RandomLinkSplit

from gnn.config.config import Config
from gnn.device import DEVICE
from gnn.metric import Metric, MetricList
from gnn.my_selectors.select_adj_transform import select_adj_transform
from gnn.my_selectors.select_data import select_dataset
from gnn.my_selectors.select_model import select_model
from gnn.my_selectors.select_optimizer import select_optimizer
from gnn.my_selectors.select_sparsification import select_sparsification
from gnn.utils import count_parameters


def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    variational = hasattr(model, "variational") and model.variational
    z = model.encode(**train_data.to_dict())
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def eval_on_test(model, data):
    model.eval()
    z = model.encode(**data.to_dict())
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


def run(config, print_verbose=True):
    num_val = config.data.data_split[1] if len(config.data.data_split) == 2 else 0

    transforms = []
    if config.data.sparsify:
        transforms = [select_sparsification(config.data.sparsify)]

    transforms = [
        *transforms,
        select_adj_transform(config.model),
        RandomLinkSplit(
            num_val=num_val,
            num_test=config.data.data_split[0],
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=False,
        ),
    ]
    dataloaders = select_dataset(config.data, config.learning_task, config.model.epochs, transforms)
    train_dataset, val_dataset, test_dataset = dataloaders[0]

    num_features = train_dataset.num_features
    model = select_model(config.model, num_features, None)

    config.model.parameters = count_parameters(model)

    model.to(DEVICE)

    optimizer = select_optimizer(model.parameters(), config.model.optimizer)

    best_auc, best_ap = None, None
    best_auc_epoch, best_ap_epoch = None, None

    metrics = MetricList()

    for epoch in range(0, config.model.epochs):
        loss = train(model, optimizer, train_dataset)
        if num_val:
            auc, ap = eval_on_test(model, val_dataset)

            if ap > (best_ap := 0):
                best_ap = ap
                best_ap_epoch = epoch
            if auc > (best_auc := 0):
                best_auc = auc
                best_auc_epoch = epoch

        print_verbose and print(f"Epoch: {epoch:03d}, Loss {loss:.4f}" + (f", AUC: {auc:.4f}, AP: {ap:.4f}" if num_val else ""))

    config.model.epochs and metrics.add(Metric("train", {"loss": loss}))
    num_val and metrics.add(
        Metric(
            "val",
            {
                "best_auc": best_auc,
                "best_ap": best_ap,
                "best_auc_epoch": best_auc_epoch,
                "best_ap_epoch": best_ap_epoch,
            },
        )
    )
    auc, ap = eval_on_test(model, test_dataset)

    metric_test = Metric(
        "test",
        {
            "auc": round(auc, 2),
            "ap": round(ap, 2),
        },
    )
    metrics.add(metric_test)
    print_verbose and print(metric_test)

    config.set_metrics(metrics)
    config.save_config_to_log_dir()
    return config


if __name__ == "__main__":
    logs_dir = Path(__file__).parent / "logs"
    config_path = Path(__file__).parent / "config.yaml"
    run(Config.from_file(config_path, logs_dir))
