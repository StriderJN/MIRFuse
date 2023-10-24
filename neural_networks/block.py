import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_filters: int,
    ):
        '''
        第一层res块提取图像特征
        in_channels: 输入图像通道数
        num_filters: filter数量
        '''
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels, num_filters, kernel_size=1, stride=1)
        self.norm = nn.InstanceNorm2d(num_filters)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm(y)

        y = self.conv2(y)
        x = self.conv3(x)
        return F.leaky_relu_(x + y)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_filter: int,
        kernel_size: int,
        stride: int,
        padding: int
    ):
        '''
         in_channels: 输入通道
         num_filter: 输出通道
         kernel_size:
         stride:
         padding:
        '''
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_filter, kernel_size, stride, padding)
    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu_(x)

class ConvBlock_1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_filter: int,
        kernel_size: int,
        stride: int,
        padding: int
    ):
        '''
         in_channels: 输入通道
         num_filter: 输出通道
         kernel_size:
         stride:
         padding:
        '''
        super(ConvBlock_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_filter, kernel_size, stride, padding)
    def forward(self, x):
        z = self.conv(x)
        return z

class Fully_connected(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(Fully_connected, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.linear(x)
        return x

class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ):
        super(ResBlock, self).__init__()
        num_filters = in_channels
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(num_filters, affine=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm(y)

        y = self.conv2(y)
        return F.leaky_relu_(x + y)

class Deconv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding
    ):
        super(Deconv, self).__init__()
        num_filter = in_channels
        self.deconv = nn.ConvTranspose2d(in_channels, num_filter, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        x = self.deconv(x)
        normalized_shape = x.size()[1:]
        self.Norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        x = self.Norm(x)
        return F.leaky_relu_(x)



if __name__ == "__main__":

    img_size = 128
    x = torch.zeros((64, 1, img_size, img_size))




