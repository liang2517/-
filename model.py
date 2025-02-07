from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch

class Lua(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 5),
            nn.ReLU(),
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
    def forward(self, x):
        x = self.model(x)
        return x
