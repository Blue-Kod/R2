from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from functools import partial


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False,
                 norm_layer=None, act_layer=None, **kwargs):
        super(BasicConv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class BasicDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 norm_layer=None, act_layer=None, **kwargs):
        super(BasicDeconv2d, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high, kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.ReLU6()(x)
        return x


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def build_correlation_volume(left_feature, right_feature, max_disp):
    B, C, H, W = left_feature.shape

    left_volume = left_feature.unsqueeze(2).expand(B, C, max_disp, H, W)
    padded_right = F.pad(right_feature, (max_disp - 1, 0, 0, 0))
    unfolded_right = padded_right.unfold(3, W, 1)
    right_volume = torch.flip(unfolded_right, [3]).permute(0, 1, 3, 2, 4)

    cost_volume = (left_volume * right_volume).mean(dim=1)
    return cost_volume.contiguous()


def context_upsample(depth_low, up_weights):
    b, c, h, w = depth_low.shape

    depth_unfold = F.unfold(depth_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    depth_unfold = F.interpolate(depth_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    depth = torch.sum(depth_unfold*up_weights, dim=1, keepdim=True)

    return depth
