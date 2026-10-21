"""Pure tensor model, reusable without Lightning or the workflow."""

from torch import nn


class Classifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.layers(x)
