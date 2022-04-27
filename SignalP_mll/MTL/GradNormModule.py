# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math

class GradNormLossModule(nn.Module):
    """
    If you don't want the loss to be dependent on the initial parameters too much,
    you should use log(num_tasks) as initial values for W.

    """
    def __init__(self, num=3):

        log_num_task = math.log(num)
        super(GradNormLossModule, self).__init__()
        params = torch.ones(num, requires_grad=True)*log_num_task
        self.params = torch.nn.Parameter(params)
        self.num = num

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += torch.div(self.params[i] * loss, self.num)
        return loss_sum

