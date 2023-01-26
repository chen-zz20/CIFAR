import torch
import torch.nn as nn
from torch import Tensor

class Model(nn.Module):
    def __init__(self, mode:str, hidden_dim=128, drop_rate=0.5, device=None):
        super().__init__()

        self.label_number = None
        if mode == "cifar10":
            self.label_number = 10
        elif mode == "cifar100":
            self.label_number = 100
        else:
            exit("mode should be in ['cifar10', 'cifar100']")
        
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool2d(2,2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(int(hidden_dim * (32/2/2) * (32/2/2)), self.label_number)
        )

        self.loss = nn.CrossEntropyLoss()


    def forward(self, x:Tensor, y=None) -> tuple[Tensor, Tensor]:
        logits = self.net(x)
        pred = torch.argmax(logits, 1)
        if y is None:
            return pred
        else:
            y = y.long()
        loss = self.loss(logits, y)
        correct_pred= (pred.int() == y.int())
        acc = torch.mean(correct_pred.float())

        return loss, acc
