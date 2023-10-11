import torch
from torch_geometric.transforms import NormalizeFeatures
import time
from pathlib import Path


from tqdm import tqdm

from gnn.config.config import Config
from gnn.device import DEVICE
from gnn.metric import MetricList, Metric
from gnn.my_selectors.select_adj_transform import select_adj_transform
from gnn.my_selectors.select_data import select_dataset
from gnn.my_selectors.select_loss_and_accuracy import select_loss, select_acc
from gnn.my_selectors.select_model import select_model
from gnn.my_selectors.select_optimizer import select_optimizer
from gnn.my_selectors.select_sparsification import select_sparsification
from gnn.utils import count_parameters


def train(
    model,
    dataloaders,
    optimizer,
    loss,
    epochs,
    logger,
    accuracy,
    gradient_clip=None,
    batch_size=1,
    print_verbose=True,
):
    train_loader, val_loader, _ = dataloaders
    best_val_acc = -1
    best_val_epoch = -1
    loss_val = -1
    acc_val = -1
    loss_per_epoch_ = -1
    acc_per_epoch_ = -1
    loss_per_batch = -1
    acc_per_batch = -1
    len_train_samples = len(train_loader)
    with tqdm(total=max(1, epochs * len_train_samples // batch_size), disable=print_verbose) as pbar:
        for epoch in range(epochs):
            start = time.time()
            loss_per_epoch = 0
            acc_per_epoch = 0
            for counter, dataset in enumerate(train_loader.shuffle()):
                model.train()
                torch.cuda.empty_cache()
                start = time.time()
                labels = dataset.y
                output = model(**dataset.to_dict())
                loss_train = loss(output, labels, dataset.get("train_mask"))
                loss_per_epoch += loss_train
                loss_train /= batch_size
                loss_per_batch += loss_train
                acc = accuracy(output, labels, dataset.get("train_mask"))
                acc_per_epoch += acc
                acc_per_batch += acc / batch_size
                loss_train.backward()
                if (counter + 1) % batch_size == 0 or counter + 1 == len(train_loader):
                    gradient_clip and torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    False and print(
                        f"\tBatch [{int(counter / batch_size):04d}/{int(len_train_samples / batch_size):04d}]| Loss: {loss_per_batch:.5f}| Acc: {acc_per_batch:.2f}| Time: {round(time.time() - start, 3)}"
                    )

                    loss_per_batch = 0
                    acc_per_batch = 0
                    pbar.update(1)

            if val_loader is not None:
                loss_val, acc_val = evaluate(
                    model,
                    val_loader,
                    loss,
                    accuracy,
                    logger,
                    val_loader[0].get("val_mask"),
                )
                model.train()
                if acc_val >= (best_val_acc := 0):
                    best_val_acc = acc_val
                    best_val_epoch = epoch

            loss_per_epoch_ = loss_per_epoch / len_train_samples
            acc_per_epoch_ = acc_per_epoch / len_train_samples
            print_verbose and print(
                f"Epoch {epoch:03d}| Loss: {loss_per_epoch_:.2f}| Acc: {acc_per_epoch_:.2f}| "
                + (f"Loss Val {loss_val:.2f}| Acc Val: {best_val_acc:.2f}| Time {time.time() - start:.2f}" if val_loader else "")
            )

            logger is not None and logger.log_training(epoch, loss_per_epoch_, acc_per_epoch_, loss_val, acc_val)
    metric = MetricList.create(Metric("train", {"loss": loss_per_epoch_, "acc": acc_per_epoch_}))
    val_loader is not None and metric.add(Metric("val", {"best_acc": best_val_acc, "epoch": best_val_epoch}))
    # plot the ntks matplolib line plot
    # plot the weights

    return metric


@torch.no_grad()
def evaluate(model, dataloader, loss, accuracy, logger, mask=None):
    model.eval()
    loss_test = 0
    acc_test = 0
    for dataset in dataloader:
        torch.cuda.empty_cache()
        output = model(**dataset.to_dict())
        labels = dataset.y
        asd = mask.sum()
        loss_test += loss(output, labels, mask)
        acc_test += accuracy(output, labels, mask)

    loss_test /= len(dataloader)
    acc_test /= len(dataloader)

    logger is not None and logger.log_test(loss_test, acc_test)
    return loss_test, acc_test


def run(config, print_verbose=True):
    # logger = TensorBoardLogger(config.log_dir)
    logger = None
    if config.loss == "mse":
        transforms = []
    else:
        transforms = [
            NormalizeFeatures()
        ]  # necessary for gat and gcn to achieve performance reported in papers but worsens performance for regression
    if config.data.sparsify:
        transforms = [select_sparsification(config.data.sparsify)]

    transforms.append(select_adj_transform(config.model))

    dataloaders = select_dataset(config.data, config.learning_task, config.model.epochs, transforms)
    train_dataset = dataloaders[0]
    config.data.has_dataset_dependent_split = train_dataset[0].get("has_dataset_dependent_split")
    num_classes = train_dataset.num_classes
    if config.loss == "mse":
        #pass
        num_classes = 1 # uncomment for using mse on classification tasks

    num_features = train_dataset.num_features
    config.model.num_features = num_features
    config.model.classes = num_classes

    model = select_model(config.model, num_features, num_classes)
    config.model.parameters = count_parameters(model)

    optimizer = select_optimizer(model.parameters(), config.model.optimizer)
    model.to(DEVICE)

    loss = select_loss(config.loss, num_classes)
    accuracy = select_acc(config.accuracy, num_classes)

    start_time = time.time()
    metrics = train(
        model,
        dataloaders,
        optimizer,
        loss,
        config.model.epochs,
        logger,
        accuracy,
        config.model.optimizer and config.model.optimizer.clip_gradient_norm,
        config.model.batch_size,
        print_verbose,
    )
    test_dataset = dataloaders[2]
    if test_dataset is not None:
        loss_test, acc_test = evaluate(
            model,
            test_dataset,
            loss,
            accuracy,
            logger,
            test_dataset[0].get("test_mask"),
        )
    end_time = time.time() - start_time
    print("Time: ", end_time)
    metrics.metrics[0].performance["time"] = end_time
    test_metric = Metric("test", {"loss": loss_test, "acc": acc_test})
    metrics.add(test_metric)
    metrics.add(Metric("params", {"params": config.model.parameters}))

    print_verbose and print("\n" + "-" * 20 + "Test" + "-" * 20 + "\n" + str(test_metric))

    config.set_metrics(metrics)
    # logger.log_dict(config.to_log_dict())
    config.save_config_to_log_dir()
    return config


if __name__ == "__main__":
    logs_dir = Path(__file__).parent / "logs"
    config_path = Path(__file__).parent / "config.yaml"
    run(Config.from_file(config_path, logs_dir), print_verbose=True)
