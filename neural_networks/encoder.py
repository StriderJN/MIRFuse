import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.block import BasicResBlock, ConvBlock, Fully_connected
from utils.output_type import EncoderOutputs

class Encoder(nn.Module):
    def __init__(
            self,
            channels: int,
            filter: int,
            shared_dim: int,
            exclusive_dim: int
    ):
        '''
                channels: 图像通道数
                shared_dim: shared表示向量的维数
                '''
        super(Encoder, self).__init__()
        # vi
        self.res1 = BasicResBlock(channels, filter)
        # com
        self.conv1cv = ConvBlock(filter, filter, 3, 1, 1)
        self.conv2cv = ConvBlock(filter, filter * 2, 3, 1, 1)
        self.conv3cv = ConvBlock(filter * 2, filter * 4, 3, 1, 1)
        # mod
        self.conv1mv = ConvBlock(filter, filter, 3, 1, 1)
        self.conv2mv = ConvBlock(filter, filter * 2, 3, 1, 1)
        self.conv3mv = ConvBlock(filter * 2, filter * 4, 3, 1, 1)
        # ir
        self.res2 = BasicResBlock(channels, filter)
        # com
        self.conv1ci = ConvBlock(filter, filter, 3, 1, 1)
        self.conv2ci = ConvBlock(filter, filter * 2, 3, 1, 1)
        self.conv3ci = ConvBlock(filter * 2, filter * 4, 3, 1, 1)
        # mod
        self.conv1mi = ConvBlock(filter, filter, 3, 1, 1)
        self.conv2mi = ConvBlock(filter, filter * 2, 3, 1, 1)
        self.conv3mi = ConvBlock(filter * 2, filter * 4, 3, 1, 1)

        self.input_dim = filter * 4
        self.fccv = Fully_connected(self.input_dim, shared_dim)
        self.fcci = Fully_connected(self.input_dim, shared_dim)
        self.fcmv = Fully_connected(self.input_dim, exclusive_dim)
        self.fcmi = Fully_connected(self.input_dim, exclusive_dim)

        # self.norm = nn.InstanceNorm2d(filter * 4, affine=False)

    def forward(self, vi, ir):
        x = self.res1(vi)

        x1 = self.conv1cv(x)
        x1 = self.conv2cv(x1)
        x1 = self.conv3cv(x1)

        x2 = self.conv1mv(x)
        x2 = self.conv2mv(x2)
        x2 = self.conv3mv(x2)
        # x2 = self.norm(x2)


        y = self.res2(ir)

        y1 = self.conv1ci(y)
        y1 = self.conv2ci(y1)
        y1 = self.conv3ci(y1)

        y2 = self.conv1mi(y)
        y2 = self.conv2mi(y2)
        y2 = self.conv3mi(y2)
        # y2 = self.norm(y2)

        zcv = F.adaptive_avg_pool2d(x1, (1, 1))
        zcv = zcv.view(zcv.size(0), -1)
        zmv = F.adaptive_avg_pool2d(x2, (1, 1))
        zmv = zmv.view(zmv.size(0), -1)

        zci = F.adaptive_avg_pool2d(y1, (1, 1))
        zci = zci.view(zci.size(0), -1)
        zmi = F.adaptive_avg_pool2d(y2, (1, 1))
        zmi = zmi.view(zmi.size(0), -1)

        shared_representation_vi = self.fccv(zcv)
        shared_representation_ir = self.fcci(zci)

        exclusive_representation_vi = self.fcmv(zmv)
        exclusive_representation_ir = self.fcmi(zmi)

        return EncoderOutputs(
            sh_feature_map_vi=x1,
            sh_feature_map_ir=y1,
            ex_feature_map_vi=x2,
            ex_feature_map_ir=y2,
            shared_representation_vi=shared_representation_vi,
            shared_representation_ir=shared_representation_ir,
            exclusive_representation_vi=exclusive_representation_vi,
            exclusive_representation_ir=exclusive_representation_ir
        )

if __name__ == "__main__":

    img_size = 128
    x = torch.zeros((64, 1, img_size, img_size))
    y = torch.zeros((64, 1, img_size, img_size))
    E = Encoder(1, 32, 64, 64)
    a = E(x, y)
    print(a.sh_feature_map_vi.shape)

