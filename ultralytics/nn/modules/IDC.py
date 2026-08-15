import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['C3k2_IDC']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class iAFF(nn.Module):
    '''
    澶氱壒寰佽瀺鍚?iAFF
    '''

    def __init__(self, channels=64, r=2):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 鏈湴娉ㄦ剰鍔?
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 鍏ㄥ眬娉ㄦ剰鍔?
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        # 绗簩娆℃湰鍦版敞鎰忓姏
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 绗簩娆″叏灞€娉ㄦ剰鍔?
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class DualConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, g=2):
        """
        初始化 DualConv 类。
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积步幅
        :param g: 用于 DualConv 的分组卷积组数
        """
        super(DualConv, self).__init__()
        # 分组卷积
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # 逐点卷积
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        """
        定义 DualConv 如何处理输入图像或输入特征图。
        :param input_data: 输入图像或输入特征图
        :return: 返回输出特征图
        """
        # 同时进行分组卷积和逐点卷积，然后将结果相加
        return self.gc(input_data) + self.pwc(input_data)

class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

        self.Conv = Conv(in_channels, out_channels, square_kernel_size, p)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
        return self.Conv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
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
class DASConv_1(nn.Module):
    def __init__(self, c1, c2, n_div=4,rate=1,bn_mom=0.1, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = c1 // n_div

        self.partial_conv3 = Conv(self.dim_conv3 , self.dim_conv3 , 3, 1, 1,g = self.dim_conv3 )
        self.conv = Conv(c1, c1, k=1)
        self.conv_1 = Conv(c1=c1  , c2=c2, k=1)
        self.conv_2 = Conv(c1=2*self.dim_conv3 , c2=c2, k=1)
        # self.Gsconv = GSConv(self.dim_conv3, self.dim_conv3)
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, padding=3 * rate, dilation=3 * rate, groups=self.dim_conv3 ,
                      bias=True),
            nn.BatchNorm2d(self.dim_conv3, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, padding=7 * rate, dilation=7 * rate, groups=self.dim_conv3 ,
                      bias=True),
            nn.BatchNorm2d(self.dim_conv3, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, padding=12 * rate, dilation=12 * rate, groups=self.dim_conv3 ,
                      bias=True),
            nn.BatchNorm2d(self.dim_conv3, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.iAFF = iAFF(2 * self.dim_conv3)  # x_l/x_r 各为 2*dim_conv3 通道


    def forward(self, x):
        # for training/inference
        x_1, x_2, x_3, x_4 = torch.split(x, [self.dim_conv3,self.dim_conv3, self.dim_conv3,self.dim_conv3], dim=1)

        x1 = self.partial_conv3(x_1)
        x2 = self.branch1(x_2)
        x_l = torch.cat((x1, x2),1)
        x3 = self.branch2(x_3)
        x4 = self.branch3(x_4)
        x_r = torch.cat((x3,x4),1)
        x_iaaf = self.iAFF(x_l,x_r)
        # x1_12 = torch.cat((x1,x2,x3,x4), 1)
        # x_iaff = self.iAFF(x,x1_12)

        return self.conv_2(x_iaaf) + self.conv_1(x)

class Bottleneck_IDC(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1 ,c_,1,1)
        self.cv1 = Conv(c_, c1, k[0],1)
        self.DASCONv = DASConv_1(c1, c_,n_div=4)
        self.stripool = StripPooling(c_)
        self.cv2 = Conv(c_, c_,k[0],1)
        self.conv2 = Conv(c_*2,c2,1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        x1 = self.conv1(x)
        x1 = self.cv1(x1)
        x2 = self.DASCONv(x1)
        x3 = self.stripool(x2)
        x4 = self.cv2(x2)
        x5 = self.conv2(torch.cat((x3,x4),1))
        return x + x5 if self.add else x5

#-------------------版本一---------------------
# class Bottleneck_IDC(nn.Module):
#     """Standard bottleneck."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = InceptionDWConv2d(c_, c2)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """Applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# 在c3k=True时，使用普通的Bottleneck，为false的时候我们使用Bottleneck_IDC提取特征
class C3k2_IDC(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_IDC(self.c, self.c, shortcut, g) for _ in
            range(n)
        )


if __name__ == '__main__':
    IDC = InceptionDWConv2d(256, 128)
    # 创建一个输入张量
    batch_size = 8
    input_tensor = torch.randn(batch_size, 256, 64, 64)
    # 运行模型并打印输入和输出的形状
    output_tensor = IDC(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("0utput shape:", output_tensor.shape)