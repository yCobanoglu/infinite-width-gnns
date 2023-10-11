import random
from typing import Union, Tuple, Sequence

from torch.utils.data import Dataset
from torch_geometric.data import Data

from gnn.device import DEVICE


class GeneratorDataset(Dataset):
    def __init__(self, generate, length, to_store=True, device_for_stored=None, num_classes=None):
        self.length = length if length is not None else 1

        if device_for_stored is not None:
            self.device_for_stored = device_for_stored
        else:
            self.device_for_stored = DEVICE if self.length < 10 else "cpu"
        self.generate = generate
        self._storage = []
        self.to_store = to_store
        self._num_classes = num_classes  # if this is not set manully than num classes will be inferred from the first sample else for graph classifiation has to be set manually
        self.first_sample: Union[
            Tuple[Data, Data, Data], Data
        ] = self.generate()  # generally this is data but for link prediction due to RandomLinkSplit it is a tuple of Data

    @property
    def num_features(self):
        if isinstance(self.first_sample, tuple):
            return self.first_sample[0].num_features
        return self.first_sample.num_features

    @property
    def num_classes(self):
        if self._num_classes is None:
            if isinstance(self.first_sample, tuple):
                return self.first_sample[0].num_classes
            return self.first_sample.num_classes
        return self.first_sample.num_classes

    def shuffle(self):
        if len(self._storage):
            random.shuffle(self._storage)
        return self

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            samples = range(self.length)[idx]
            return GeneratorDataset(self.generate, len(samples), self.to_store, self.device_for_stored)
        if idx == self.length:
            raise StopIteration
        if idx < len(self._storage):
            data = self._storage[idx]
            if DEVICE == "cuda":
                data = data.clone().to(DEVICE)
                return data
        else:
            data = self.generate()
            if self.to_store:
                if isinstance(data, Sequence):
                    data_to_store = tuple(d.clone().to(self.device_for_stored) for d in data)
                else:
                    data_to_store = data.clone().to(self.device_for_stored)
                    pass
                self._storage.append(data_to_store)
        return data
