import torch
import torch.nn as nn
from neural_networks.encoder import Encoder
from neural_networks.decoder import Decoder
from neural_networks.statistics_network import tile_and_concat, LocalStatisticsNetwork, GlobalStatisticsNetwork
from utils.output_type import MIRFOutputs

class MIRFuse(nn.Module):
    def __init__(
            self,
            channels: int,
            filter: int,
            shared_dim: int,
            exclusive_dim: int
    ):
        super(MIRFuse, self).__init__()
        self.filter = filter
        self.channels = channels
        self.shared_dim = shared_dim
        self.exclusive_dim = exclusive_dim

        self.encoder = Encoder(channels=channels, filter=filter, shared_dim=shared_dim, exclusive_dim=exclusive_dim)
        self.decoder = Decoder(filter=filter)

        self.feature_channels = filter * 4
        self.feature_size = 32

        # local statistics network
        self.local_stat_VI = LocalStatisticsNetwork(img_feature_channels=self.feature_channels + self.shared_dim)
        self.local_stat_IR = LocalStatisticsNetwork(img_feature_channels=self.feature_channels + self.shared_dim)

        # global statistics network
        self.global_stat_VI = GlobalStatisticsNetwork(
            feature_map_size=self.feature_size, feature_map_channels=self.feature_channels, latent_dim=self.shared_dim)
        self.global_stat_IR = GlobalStatisticsNetwork(
            feature_map_size=self.feature_size, feature_map_channels=self.feature_channels, latent_dim=self.shared_dim)


    def forward(self, vi: torch.Tensor, ir: torch.Tensor):
        out = self.encoder(vi, ir)

        M_X = out.sh_feature_map_vi
        M_Y = out.sh_feature_map_ir
        ex_M_x = out.ex_feature_map_vi
        ex_M_y = out.ex_feature_map_ir
        shared_X = out.shared_representation_vi
        shared_Y = out.shared_representation_ir
        exclusive_X = out.exclusive_representation_vi
        exclusive_Y = out.exclusive_representation_ir

        # Shuffle M to create M*_X M*_Y 为了产生独立分布
        M_X_prime = torch.cat([M_X[1:], M_X[0].unsqueeze(0)], dim=0)
        M_Y_prime = torch.cat([M_Y[1:], M_Y[0].unsqueeze(0)], dim=0)

        # Global mutual information estimation
        global_mutual_M_R_x = self.global_stat_VI(M_X, shared_Y)
        global_mutual_M_R_x_prime = self.global_stat_VI(M_X_prime, shared_Y)

        global_mutual_M_R_y = self.global_stat_IR(M_Y, shared_X)
        global_mutual_M_R_y_prime = self.global_stat_IR(M_Y_prime, shared_X)

        # Merge the feature map with the shared representation

        concat_M_R_x = tile_and_concat(tensor=M_X, vector=shared_Y)
        concat_M_R_x_prime = tile_and_concat(tensor=M_X_prime, vector=shared_Y)

        concat_M_R_y = tile_and_concat(tensor=M_Y, vector=shared_X)
        concat_M_R_y_prime = tile_and_concat(tensor=M_Y_prime, vector=shared_X)

        # Local mutual information estimation

        local_mutual_M_R_x = self.local_stat_VI(concat_M_R_x)
        local_mutual_M_R_x_prime = self.local_stat_VI(concat_M_R_x_prime)
        local_mutual_M_R_y = self.local_stat_VI(concat_M_R_y)
        local_mutual_M_R_y_prime = self.local_stat_VI(concat_M_R_y_prime)

        # Reconstruction
        recons_img_vi = self.decoder(out.sh_feature_map_vi, out.ex_feature_map_vi)
        recons_img_ir = self.decoder(out.sh_feature_map_ir, out.ex_feature_map_ir)

        return MIRFOutputs(
            global_mutual_M_R_x=global_mutual_M_R_x,
            global_mutual_M_R_x_prime=global_mutual_M_R_x_prime,
            global_mutual_M_R_y=global_mutual_M_R_y,
            global_mutual_M_R_y_prime=global_mutual_M_R_y_prime,
            local_mutual_M_R_x=local_mutual_M_R_x,
            local_mutual_M_R_x_prime=local_mutual_M_R_x_prime,
            local_mutual_M_R_y=local_mutual_M_R_y,
            local_mutual_M_R_y_prime=local_mutual_M_R_y_prime,
            shared_x=shared_X,
            shared_y=shared_Y,
            exclusive_x=exclusive_X,
            exclusive_y=exclusive_Y,
            img_recon_vi=recons_img_vi,
            img_recon_ir=recons_img_ir,
            sh_M_x=M_X,
            sh_M_y=M_Y,
            ex_M_x =ex_M_x,
            ex_M_y =ex_M_y
        )