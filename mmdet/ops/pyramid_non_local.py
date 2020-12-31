import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

from .conv_module import ConvModule

# Pyramid non local Block
class PNB2D(nn.Module):
    """
    Pyramid non local Block

    Args:
        in_channels(int): Channels of the input feature map
        reduction(int): channels reduction ratio
        use_scale(bool): whether to scale pairwise_weight by 1/inter_channels
        conv_cfg(dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self,
                 in_channels,
                 reduction=4,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='dot_product'):
        super(PNB2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.value = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.key = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.query = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            act_cfg=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.value, self.key, self.query]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, c, h, w = x.shape
        #print('the shape of input feature map:{}'.format(x.shape))

        #value: [n, C, h', w']
        value_x = self.value(x)
        value_x = self.avgpool(x)
        #print(value_x.shape)
        value_x = self.value(value_x).view(n, self.inter_channels, -1)
        
        value_x = value_x.permute(0, 2, 1)
        # value: [n, h' x w', c]
        #print('the shape of value:{}'.format(value_x.shape))

        # in_key: [n, C, h', w']
        key_x = self.key(x)
        key_x = self.maxpool(x)
        key_x = self.key(key_x).view(n, self.inter_channels, -1)
        

        # out_key: [n, c, h' x w']
        #print('the shape of key:{}'.format(key_x.shape))

        # in_query: [n, C, h, w]
        query_x = self.query(x).view(n, self.inter_channels, -1)
        query_x = query_x.permute(0, 2, 1)
        # out_query: [n , h x w, c]
        #print('the shape of query:{}'.format(query_x.shape))

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: query_x ([N, HxW, C]) ·　key_x([N, C, H'xW'])=[N, HxW, H'xW']

        pairwise_weight = pairwise_func(query_x, key_x)
        # calculate similarity

        # y : [n, HxW, C]
        # pairwise_weight [N, HxW, H'xW'] · value_x [n, H'xW', C]
        y = torch.matmul(pairwise_weight, value_x)

        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)

        #print(output.shape)

        return output
