import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from model import Model

def train_epoch(model:Model, data_loader:DataLoader, optimizer:Optimizer) -> tuple[float, float]:
    model.train()
    device = model.device
    loss, acc = 0.0, 0.0
    times = 0
    for step, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss_, acc_ = model.forward(inputs, labels)
        loss_.backward()
        optimizer.step()

        loss += loss_.data.cpu().numpy()
        acc += acc_.data.cpu().numpy()
        times += 1
    loss /= times
    acc /= times
    return loss, acc


def test_epoch(model:Model, data_loader:DataLoader) -> tuple[float, float]:
    model.eval()
    device = model.device
    loss, acc = 0.0, 0.0
    times = 0
    for step, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        loss_, acc_ = model.forward(inputs, labels)

        loss += loss_.data.cpu().numpy()
        acc += acc_.data.cpu().numpy()

        times += 1
    loss /= times
    acc /= times
    return loss, acc
