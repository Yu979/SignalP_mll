import torch
import torch.nn as nn
import math

class DWAModule(nn.Module):
    """
    scope for metric is [0, 1]
    """
    def __init__(self, num_task, T=2):
        super(DWAModule, self).__init__()
        self.num_task=num_task
        self.losses = torch.ones(num_task)
        self.T = T
        self.params = torch.ones(num_task)


    def forward(self, *x):

        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += self.params[i] * loss

        return loss_sum

    def update(self, *x):
        exp_sum = 0
        rs = []

        for i, loss in enumerate(x):
            r = loss/self.losses[i]
            rs.append(r)
            exp_sum += math.exp(r/self.T)

        for i in range(self.num_task):
            self.params[i] = self.num_task * math.exp(rs[i]/self.T) / exp_sum