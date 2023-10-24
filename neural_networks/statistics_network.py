import torch.nn as nn
import torch
from Loss.loss_function import *

def tile_and_concat(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Merge 1D and 2D tensor (use to aggregate feature maps and representation
    and compute local mutual information estimation)

    Args:
        tensor (torch.Tensor): 2D tensor (feature maps)
        vector (torch.Tensor): 1D tensor representation
    融合1D和2D的Tensor(融合representation和feature maps)

    Returns:
        torch.Tensor: Merged tensor (2D)
    """

    B, C, H, W = tensor.size()
    vector = vector.unsqueeze(2).unsqueeze(2)
    expanded_vector = vector.expand((B, vector.size(1), H, W))
    return torch.cat([tensor, expanded_vector], dim=1)

class LocalStatisticsNetwork(nn.Module):
    def __init__(self, img_feature_channels: int):
        """
        Local statistique network
        局部统计网络结构，输入是feature map和表征向量，输出一个Tensor
        输入是feature_map和representation_vector concat后的tensor 输出是[batch_size, 1, feature_map_size, feature_map_size]
        Args:
            img_feature_channels (int): [Number of input channels]
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=img_feature_channels, out_channels=512, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, concat_feature: torch.Tensor) -> torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        local_statistics = self.conv3(x)
        return local_statistics

class GlobalStatisticsNetwork(nn.Module):
    """
    Global statistics network
    全局统计网络结构，输入是图像和表征向量（错误），输出一个1*1的Tensor
    输入的是feature_map和representation_vector
    Args:
        feature_map_size (int): Size of input feature maps
        feature_map_channels (int): Number of channels in the input feature maps
        latent_dim (int): Dimension of the representations
    """

    def __init__(
        self, feature_map_size: int, feature_map_channels: int, latent_dim: int
    ):

        super().__init__()
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            in_features=(feature_map_size ** 2 * feature_map_channels) + latent_dim,
            out_features=512,
        )
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()

    def forward(
        self, feature_map: torch.Tensor, representation: torch.Tensor
    ) -> torch.Tensor:
        feature_map = self.avgpool(feature_map)
        feature_map = self.flatten(feature_map)
        x = torch.cat([feature_map, representation], dim=1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        global_statistics = self.dense3(x)  # 输出一个1*1的tensor

        return global_statistics

# if __name__ == "__main__":
#
#     img_size = 192
#     x = torch.zeros((64, 1, img_size, img_size))
#     enc_sh = BaseEncoder(
#         img_size=img_size, in_channels=1, num_filters=16, kernel_size=1, repr_dim=64
#     )
#     enc_ex = BaseEncoder(
#         img_size=img_size,
#         in_channels=1,
#         num_filters=16,
#         kernel_size=1,
#         repr_dim=64,
#     )
#
#     sh_repr, sh_feature = enc_sh(x)
#     ex_repr, ex_feature = enc_ex(x)
#     merge_repr = torch.cat([sh_repr, ex_repr], dim=1)
#     merge_feature = torch.cat([sh_feature, ex_feature], dim=1)
#     concat_repr = tile_and_concat(tensor=merge_feature, vector=merge_repr)
#     t_loc = LocalStatisticsNetwork(img_feature_channels=concat_repr.size(1))
#     t_glob = GlobalStatisticsNetwork(
#         feature_map_size=merge_feature.size(2),
#         feature_map_channels=merge_feature.size(1),
#         latent_dim=merge_repr.size(1),
#     )
#     print(merge_repr.shape)
#     print(merge_feature.shape)
#     print(concat_repr.shape)
#     print(t_glob(feature_map=merge_feature, representation=merge_repr).shape)
#     print(tile_and_concat(tensor=merge_feature, vector=merge_repr).shape)
#     print(t_loc(concat_feature=concat_repr).shape)
#     # print(b[0])
#     # # print(b[0].shape)
if __name__ == "__main__":
    img = torch.zeros(64, 1, 128, 128)
    feature_map = torch.zeros(64, 192, 32, 32)
    vector = torch.zeros(64,64)
    cat = tile_and_concat(feature_map, vector)
    print(cat.shape)
    print('size(1) of cat =', cat.size(1))
    Local = LocalStatisticsNetwork(img_feature_channels=cat.size(1))
    T1 = Local(cat)
    print(T1.shape)
    Global = GlobalStatisticsNetwork(feature_map_size=32, feature_map_channels=192, latent_dim=64)
    T2 = Global(feature_map, vector)
    print(T2.shape)
    djs_loss = DJSLoss()
    c = djs_loss(img,T2)
    print(c)

