from typing import Tuple

import numpy as np
import torch


def split_data_inductive(
    split: Tuple[float, ...],
    data,
):
    assert len(split) <= 2
    size = len(data)
    split_ = [sum(split)]
    len(split) == 2 and split_.append(split[1])
    result = []
    start = 0
    for spli in split_:
        end = int((1 - spli) * size)
        result.append(data[start:end])
        start = end
    result.append(data[start:])
    if len(result) == 2:
        return result[0], None, result[1]
    return result


def split_data_transductive(num_nodes, test_val_split, unlabelled=0, random_seed=0):
    if len(test_val_split) == 0:
        return [torch.ones((num_nodes,), dtype=torch.bool)]
    if unlabelled is None:
        unlabelled = 0
    end_first = int(num_nodes * unlabelled)
    end_second = num_nodes - int(num_nodes * sum(test_val_split))
    end_third = num_nodes - int(num_nodes * test_val_split[0])
    splits = [end_first, end_second, end_third]
    len(test_val_split) > 1 and splits.append(num_nodes)
    assert splits[-1] == num_nodes
    np.random.seed(random_seed)
    idx = np.random.permutation(num_nodes)
    masks = []
    for i in range(len(splits) - 1):
        first, second = splits[i], splits[i + 1]
        indices = idx[first:second]
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    assert sum([x.sum() for x in masks]) == num_nodes - end_first
    return masks
