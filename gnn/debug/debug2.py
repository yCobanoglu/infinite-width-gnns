import torch

from gnn.debug.fcn_debug import FCN1
from gnn.transforms.basic_transforms import TransductiveDataSplit

torch.manual_seed(0)
import random

random.seed(0)
import numpy as np

np.random.seed(0)
torch.use_deterministic_algorithms(True)
import torch.nn.functional as F

from torch_geometric.datasets import (
    WikipediaNetwork,
)
from torch_geometric.transforms import (
    Compose,
    ToDevice,
    NormalizeFeatures,
)

from gnn.device import DEVICE

torch.manual_seed(0)

from gnn.my_selectors.select_loss_and_accuracy import r_squared

name = "chameleon"
EPOCHS = 300
transforms = [TransductiveDataSplit([0.2, 0.3]), ToDevice(DEVICE)]
dataset = WikipediaNetwork(root="/tmp/", name=name, transform=Compose(transforms), geom_gcn_preprocess=False)
data = dataset[0]
# classes = dataset.num_classes
classes = 1

num_layers = 2
nfeat = dataset.x.shape[1]
nhid = 128
skip = False
dropout = 0.5


def train(model, optimizer, data):
    model.train()
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(**data.to_dict())
        loss = F.mse_loss(out[data.train_mask, 0], data.y[data.train_mask])
        # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(**data.to_dict())

    acc = r_squared(pred[data.test_mask], data.y[data.test_mask])
    # acc = (pred[data.test_mask].argmax(dim=-1) == data.y[data.test_mask]).float().mean()
    print("Accuracy: {:.4f}".format(acc))


BATCHNORM = True

MODEL = FCN1

MODEL1 = MODEL(num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=BATCHNORM).to(DEVICE)
MODEL2 = MODEL(num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=BATCHNORM).to(DEVICE)
MODEL3 = MODEL(num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=BATCHNORM).to(DEVICE)
MODEL4 = MODEL(num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=BATCHNORM).to(DEVICE)
MODEL5 = MODEL(num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=BATCHNORM).to(DEVICE)
MODEL6 = MODEL(num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=BATCHNORM).to(DEVICE)
MODEL7 = MODEL(num_layers, nfeat, nhid, classes, dropout, skip, batchnorm=BATCHNORM).to(DEVICE)

LR = 0.01

optimizer1 = torch.optim.Adam(MODEL1.parameters(), lr=LR)
optimizer2 = torch.optim.Adam(MODEL2.parameters(), lr=LR)
optimizer3 = torch.optim.Adam(MODEL3.parameters(), lr=LR)
optimizer4 = torch.optim.Adam(MODEL4.parameters(), lr=LR)
optimizer5 = torch.optim.Adam(MODEL5.parameters(), lr=LR)
optimizer6 = torch.optim.Adam(MODEL6.parameters(), lr=LR)
optimizer7 = torch.optim.Adam(MODEL7.parameters(), lr=LR)

train(MODEL1, optimizer1, data)
train(MODEL2, optimizer2, data)
train(MODEL3, optimizer3, data)
train(MODEL4, optimizer4, data)
train(MODEL5, optimizer5, data)
train(MODEL6, optimizer6, data)
train(MODEL7, optimizer7, data)
