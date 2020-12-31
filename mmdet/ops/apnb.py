import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

from .conv_module import ConvModule
from torch.nn import functional as F

# Pyramid non local Block
class PSPModule(nn.Module):
    # downsample : [1, 3, 6, 8]
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self.__make_stage(size, dimension) for size in sizes])

    def __make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)

        if dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))

        if dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))

        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        # priors: [torch.tensor(n,c,1x1), torch.tensor(n,c,3x3), ... , torch.tensor(n,c,9x9),]
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]

        # center: [torch.tensor(n, c, 1x1+3x3+...+9x9=110)]
        center = torch.cat(priors, -1)

        return center


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory 												cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)

    '''

    def __init__(self,
                 in_channels,
                 reduction=2,
                 out_channels=None,
                 scale=1,
                 norm_cfg=None,
                 conv_cfg=None,
                 psp_size=(1, 4, 8, 12)):
        super(_SelfAttentionBlock, self).__init__()

        self.scale = scale

        self.in_channels = in_channels

        self.out_channels = out_channels

        if out_channels == None:
            self.out_channels = in_channels // reduction

        # choose the scale to downsample the input feature maps (save memory cost)
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))

        self.f_key = ConvModule(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
	    act_cfg=None)

        self.f_query = ConvModule(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
	    act_cfg=None)

        self.f_value = ConvModule(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            act_cfg=None)

        self.conv_out = ConvModule(
            self.out_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.psp = PSPModule(psp_size)

        self.init_weight()

    def init_weight(self, std=0.01, zero_init=True):
        for m in [self.f_key, self.f_query, self.f_value]:
            normal_init(m.conv, std=std)
        if zero_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        # x : [n,c,h,w]
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        if self.scale > 1:
            x = self.pool(x)

        # x -->conv1x1 --> downsample --> value:[n, h' x w', c]
        value = self.psp(self.f_value(x))
        #print(type(value))
        value = value.permute(0, 2, 1)
        # x -->conv/bn/relu --> reshape --> query: [n, c, h x w] --> [n, h x w, c]
        query = self.f_query(x).view(batch_size, self.out_channels, -1)
        query = query.permute(0, 2, 1)
        # x -->conv/bn/relu --> key : [n, c, h' x w']
        key = self.psp(self.f_key(x))
        # sim_map : [n, h x w, h' x w']
        sim_map = torch.matmul(query, key)
        sim_map = (self.out_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # context: [n, h x w, c]
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.out_channels, *x.size()[2:])
        context = self.conv_out(context)

        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):

    def __init__(self, in_channels,
                 out_channels=None,
                 scale=1,
                 reduction=2,
                 norm_cfg=None,
                 conv_cfg=None,
                 psp_size=(1, 4, 8, 12)):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   out_channels=None,
                                                   scale=1,
                                                   reduction=2,
                                                   norm_cfg=None,
                                                   conv_cfg=None,
                                                   psp_size=psp_size)

class APBN(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self,
                 in_channels,
                 reduction=2,
                 out_channels=None,
                 dropout=0.05,
                 sizes=([1]),
                 norm_cfg=None,
                 conv_cfg=None,
                 psp_size=(1, 4, 8, 12)):
        super(APBN, self).__init__()

        self.stages = []
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.psp_size = psp_size
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels,
                              out_channels,
                              reduction,
                              size) for size in sizes])

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, reduction, size):
        return SelfAttentionBlock2D(in_channels,
                                    output_channels,
                                    size,
                                    reduction,
                                    self.norm_cfg,
                                    self.conv_cfg,
                                    self.psp_size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        #print('context shape:{}'.format(context.shape))

        for i in range(1, len(priors)):
            context += priors[i]

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output
