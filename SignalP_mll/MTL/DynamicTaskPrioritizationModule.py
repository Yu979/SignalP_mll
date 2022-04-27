import torch
import torch.nn as nn
import math

class DTPModule(nn.Module):
    """
    scope for metric is [0, 1]
    """
    def __init__(self, metric_init, gamma):
        super(DTPModule, self).__init__()
        self.gamma = gamma
        self.params = [-pow(1-m, self.gamma)*math.log(m) for m in metric_init]

    def forward(self, *x):
        Loss = 0
        for i, m in enumerate(x):
            Loss = torch.add(Loss, self.params[i]*m)

        return Loss

    def update(self, metric):
        self.params = [-pow(1-m, self.gamma)*math.log(m) for m in metric]
