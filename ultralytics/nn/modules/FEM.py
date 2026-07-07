import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.utils.misc import groups

from ultralytics.nn.modules.conv import Conv
class BasicConv_FFCA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv_FFCA, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # out = identity * a_w * a_h

        return a_w*a_h


import math


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Efficient_Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=16):
        super(Efficient_Attention_Gate, self).__init__()
        self.num_groups = num_groups
        self.grouped_conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )

        self.grouped_conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True, groups=num_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.psi(self.relu(x1 + g1))
        out = x * psi
        out += x

        return out
class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        inter_channels = in_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 11), 1, (0, 11 // 2), bias=False, groups=inter_channels),
                                   nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (11, 1), 1, (11 // 2, 0), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # 结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))  # 结合1*W和H*1的特征
        out = self.conv5(x4)
        return F.relu_(x + out)

class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)
        self.dwconv_hw5 = nn.Conv2d(gc, gc, 3, padding=5, groups=gc,dilation= 5)
        self.dwconv_hw3 = nn.Conv2d(gc, gc,3, padding=3, groups=gc,dilation= 3)
        self.strippooling = StripPooling(gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

        self.Conv = Conv(in_channels, out_channels, 1,1)
        self.bn = nn.BatchNorm2d(gc)
    def forward(self, x):
        x_id, x_hw3, x_stip, x_hw5 = torch.split(x, self.split_indexes, dim=1)
        xhw3 = self.bn(x_hw3 + x_hw5)
        xstip = self.bn(x_stip + x_hw5+ x_hw3)
        xhw5 =  self.bn(x_hw5 + x_stip)
        x = torch.cat(
            (x_id, self.dwconv_hw3(xhw3), self.strippooling(xstip), self.dwconv_hw5(xhw5)),
            dim=1,
        )
        return self.Conv(x) + x

class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.IDc = InceptionDWConv2d(in_planes,in_planes)
        self.conv1 = Conv(in_planes, in_planes // 2,1,1)
        self.conv2 = Conv(in_planes//2, in_planes//2, 3,1,1, g=in_planes // 2)
        self.conv3 = Conv(in_planes // 2, out_planes,1 ,1)

    def forward(self, x):
        x1 = self.IDc(x)
        x2 = self.conv1(x1)
        x3 =self.conv2(x2) + x2
        x4 = self.conv3(x3)

        return  x4 + x1
# class FEM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#
#         inter_planes = in_planes // map_reduce
#         self.dim_conv = 2*inter_planes
#         self.untorch = in_planes - 4*inter_planes
#         self.branch0 = nn.Sequential(
#             BasicConv_FFCA(in_planes,out_planes, kernel_size=1, stride=stride),
#
#         )
#         self.branch1 = nn.Sequential(
#
#             BasicConv_FFCA(inter_planes*2, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride,
#                            padding=(1, 0)),
#             BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
#                            relu=False)
#         )
#         self.branch2 = nn.Sequential(
#
#             BasicConv_FFCA(inter_planes*2, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride,
#                            padding=(0, 1)),
#             BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
#                            relu=False)
#         )
#
#         self.ConvLinear = BasicConv_FFCA(in_planes, in_planes // 4, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv_FFCA(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
#         self.gat = Efficient_Attention_Gate(in_planes // 4, out_planes, in_planes // 4)
#
#     def forward(self, x):
#         x_1, xid ,x_2 = torch.split(x,[self.dim_conv, self.untorch,self.dim_conv],1)
#         x0 = self.branch0(x)
#         x1 = self.branch1(x_2)
#         x2 = self.branch2(x_1)
#
#         out = torch.cat((xid, x1, x2), 1)
#
#         out = self.ConvLinear(out)
#         out = self.gat(out,x0)
#         short = self.shortcut(x)
#         out = out  + short
#         out = self.relu(out)
#
#         return out

#
# class FEM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
#         self.branch0 = nn.Sequential(
#             BasicConv_FFCA(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#             BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, groups= 2* inter_planes, relu=False)
#         )
#
#         self.conv = BasicConv_FFCA(in_planes,inter_planes,1,1)
#         self.conv_11 = BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3,(1,3),stride=stride, padding=(0,1) )
#         self.conv_12 = BasicConv_FFCA( (inter_planes // 2) * 3, (inter_planes // 2) * 3,kernel_size=(5,1),stride=stride,padding=(5//2,0),groups= (inter_planes // 2) * 3)
#         self.conv_13 = BasicConv_FFCA( (inter_planes // 2) * 3,2*inter_planes,1,1)
#
#         self.conv_21 = BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, (3, 1), stride=stride, padding=(1, 0))
#         self.conv_22 = BasicConv_FFCA((inter_planes // 2) * 3, (inter_planes // 2) * 3, kernel_size=(1, 5),
#                                       stride=stride, padding=(0, 5 // 2), groups=(inter_planes // 2) * 3)
#         self.conv_23 = BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, 1, 1)
#
#         self.conv_1 = BasicConv_FFCA(inter_planes,(inter_planes // 2) * 3,1 ,1)
#         self.conv_2 = BasicConv_FFCA((inter_planes // 2) * 3, 2*inter_planes,1,1)
#         self.dila = BasicConv_FFCA(2 * inter_planes, 2* inter_planes,kernel_size=3, stride=1,padding=5,dilation=5,groups=2 * inter_planes,relu=False)
#         self.ConvLinear_1 = BasicConv_FFCA(6*inter_planes,2*inter_planes,1,1)
#         self.ConvLinear_2 = BasicConv_FFCA(4*inter_planes,out_planes,1,1)
#         self.ConvLinear = BasicConv_FFCA(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv_FFCA(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
#         self.CA_attention =CoordAtt(in_planes, out_planes)
#
#     def forward(self, x):
#
#         x0 = self.branch0(x)
#         x10 = self.conv(x)
#         x11 = self.conv_11(x10)
#         x12 = self.conv_12(x11)
#         x13 = self.conv_13(x12)
#
#         x20 = self.conv(x)
#         x21 = self.conv_21(x20)
#         x22 = self.conv_22(x21)
#         x23 = self.conv_23(x22)
#
#         x_0 = x10 + x20
#         x_1 = self.conv_1(x_0) + x11 + x21
#         x_2 = self.conv_2(x_1) + x13 + x23
#
#         x1 = self.dila(self.ConvLinear_1(torch.cat((x13,x_2,x23),1)))
#         out = self.ConvLinear_2(torch.cat((x1,x0),1))
#
#
#         ca_scale = self.CA_attention(x)


        short = self.shortcut(x)
        out = out * ca_scale + short
        out = self.relu(out)

        return out



#--------------------源版本-----------------------

# class BasicConv_FFCA(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv_FFCA, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
# class FEM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
#         self.branch0 = nn.Sequential(
#             BasicConv_FFCA(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#             BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv_FFCA(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride,
#                            padding=(1, 0)),
#             BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
#                            relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv_FFCA(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride,
#                            padding=(0, 1)),
#             BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
#                            relu=False)
#         )
#
#         self.ConvLinear = BasicConv_FFCA(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv_FFCA(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#
#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)
#
#         return out