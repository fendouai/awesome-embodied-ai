# Copyright 2024 Tencent. See LICENSE.txt for licensing details.
import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self,
                 input_size,
                 momentum=0.99,
                 begin_norm_axis=0,
                 moving_mean_initializer=None,
                 moving_std_initializer=None):
        super(RunningMeanStd, self).__init__()
        self.begin_norm_axis = begin_norm_axis
        self.momentum = momentum
        self.moving_mean = nn.Parameter(torch.zeros((1, input_size))) if moving_mean_initializer is None\
            else moving_mean_initializer
        self.moving_std = nn.Parameter(torch.ones((1, input_size))) if moving_std_initializer is None\
            else moving_std_initializer

    def forward(self, x):
        mean = torch.mean(x, dim=self.begin_norm_axis, keepdim=True)
        variance = torch.var(x, dim=self.begin_norm_axis, keepdim=True)
        rms_loss = 0.5 * self.momentum * (torch.square(self.moving_mean - mean.detach())
                                          + torch.square(self.moving_std - torch.sqrt(variance.detach())))
        output = (x - self.moving_mean) / (self.moving_std + 1e-8)
        return output.detach(), rms_loss
