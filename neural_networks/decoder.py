import torch
import torch.nn as nn
from neural_networks.block import Deconv, ResBlock, ConvBlock, ConvBlock_1
from neural_networks.encoder import Encoder
import torch.nn.functional as F
# from utils.output_type import EncoderOutputs, DecoderOutputs

class Decoder(nn.Module):
    def __init__(
            self,
            filter:int
    ):
        super(Decoder, self).__init__()
        self.cat_channels = filter * 4 + filter * 4
        self.res = ResBlock(self.cat_channels)
        # self.conv = ConvBlock(self.cat_channels, self.cat_channels, 3, 1, 1)
        # self.deconv1 = Deconv(filter * 8, 3, 2, 1, 1)
        # self.deconv2 = Deconv(filter * 8, 3, 2, 1, 1)
        self.conv1 = ConvBlock(self.cat_channels, filter * 4, 3, 1, 1)
        self.conv2 = ConvBlock(filter * 4 , filter * 2, 3, 1, 1)
        self.conv3 = ConvBlock(filter * 2, filter * 2, 3, 1, 1)
        self.conv4 = ConvBlock_1(filter * 2, 1, 3, 1, 1)

        # self.conv2 = ConvBlock_g(filter, 1, 3, 1, 1)

    def forward(self, sh, ex):
        x = torch.cat([sh, ex], dim=1)
        x = self.res(x)
        # x = self.deconv1(x)
        # x = self.deconv2(x)
        x = self.conv1(x)
        # x = torch.cat([feature_map, x], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # x1 = self.deconv1(x)
        # x1 = self.deconv2(x1)
        # feature_map = torch.cat([x1, y], dim=1)
        # z = self.res(feature_map)
        # z = self.conv1(z)
        # z = self.conv2(z)
        # z = self.conv3(z)
        # z = self.conv4(z)

        return torch.tanh(x)

