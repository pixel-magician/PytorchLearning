import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(24, 2)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out
