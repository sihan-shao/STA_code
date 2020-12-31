import torch
from torch import nn
from torch.nn.parameter import Parameter
import math


# torch.mul(torch.ones_like(x), (torch.abs(x) > thres).float() * x)


class SoftThreshold_Attention(nn.Module):
    def __init__(self,
                 channels,
                 types='se',
                 reduction=16,
                 gamma=2,
                 b=1):
        super(SoftThreshold_Attention, self).__init__()

        self.type = types
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        '''
        choose the way to caculate thresholding value
        '''
        if self.type == 'eca':
            t = int(abs((math.log(channels, 2) + b) / gamma))
            k_size = t if t % 2 else t + 1
            self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        elif self.type == 'se':
            self.block = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )
        else:
            raise ValueError("Not implemented yet")

    def forward(self, x):
        types = self.type
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y1 = torch.abs(y)

        if types == 'eca':
            z1 = self.conv1(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            z1 = self.sigmoid(z1)
            thresholding = y1 * z1
            #print('eca model', thresholding.shape)
            out = torch.mul(torch.sign(x), torch.clamp(torch.abs(x) - thresholding, min=0))

            #return out

        elif types == 'se':
            z1 = self.avg_pool(y1).view(b, c)
            z1 = self.block(z1).view(b, c, 1, 1)
            thresholding = y1 * z1
            #print('se model', thresholding.shape)
            out = torch.mul(torch.sign(x), torch.clamp(torch.abs(x) - thresholding, min=0))
            #return out

        else:
            raise ValueError("Not implemented yet")

        return out
