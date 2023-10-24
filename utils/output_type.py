from typing import NamedTuple, Tuple
import torch

class MIRlosses(NamedTuple):
    mim_loss: torch.Tensor
    shared_loss: torch.Tensor
    mse_loss: torch.Tensor
    het_loss: torch.Tensor
    total_loss: torch.Tensor

class EncoderOutputs(NamedTuple):
    sh_feature_map_vi: torch.Tensor
    sh_feature_map_ir: torch.Tensor
    ex_feature_map_vi: torch.Tensor
    ex_feature_map_ir: torch.Tensor
    shared_representation_vi: torch.Tensor
    shared_representation_ir: torch.Tensor
    exclusive_representation_vi: torch.Tensor
    exclusive_representation_ir: torch.Tensor

class MIRFOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    global_mutual_M_R_y: torch.Tensor
    global_mutual_M_R_y_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_y: torch.Tensor
    local_mutual_M_R_y_prime: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor
    exclusive_x: torch.Tensor
    exclusive_y: torch.Tensor
    img_recon_vi: torch.Tensor
    img_recon_ir: torch.Tensor
    sh_M_x: torch.Tensor
    sh_M_y: torch.Tensor
    ex_M_x: torch.Tensor
    ex_M_y: torch.Tensor

class SDIMlosses(NamedTuple):
    total_loss: torch.Tensor
    shared_loss: torch.Tensor

# class sh_encoder_output(NamedTuple):
#     feature_c1: torch.Tensor
#     feature_c2: torch.Tensor
#     sh_feature_map: torch.Tensor
#     # representation: torch.Tensor

class SDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    global_mutual_M_R_y: torch.Tensor
    global_mutual_M_R_y_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_y: torch.Tensor
    local_mutual_M_R_y_prime: torch.Tensor
    # feature_map_vi: torch.Tensor
    # feature_map_ir: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor
    # feature_map_C1_vi:torch.Tensor
    # feature_map_C2_vi:torch.Tensor
    # feature_map_C1_ir:torch.Tensor
    # feature_map_C2_ir:torch.Tensor

class EDIMoutput(NamedTuple):
    img_recon_vi: torch.Tensor
    img_recon_ir: torch.Tensor
    R_y_x: torch.Tensor
    R_x_y: torch.Tensor
    shuffle_x: torch.Tensor
    shuffle_y: torch.Tensor
    fake_x: torch.Tensor
    fake_y: torch.Tensor






class DiscrLosses(NamedTuple):
    gan_loss_d: torch.Tensor

class SharedEncoderOut(NamedTuple):
    feature_map_vi: torch.Tensor
    representation_vi: torch.Tensor
    feature_map_ir: torch.Tensor
    representation_ir: torch.Tensor

class ExclusiveEncoderOut(NamedTuple):
    exclusive_feature_map: torch.Tensor
    # exclusive_representation: torch.Tensor



# class Recons_Losses(NamedTuple):
#     recons_loss: torch.Tensor
#     ssim_loss_VI: torch.Tensor
#     ssim_loss_IR: torch.Tensor
#     mse_loss_VI: torch.Tensor
#     mse_loss_IR: torch.Tensor

# class JOINTOutput(NamedTuple):
#     global_mutual_M_R_x: torch.Tensor
#     global_mutual_M_R_x_prime: torch.Tensor
#     global_mutual_M_R_y: torch.Tensor
#     global_mutual_M_R_y_prime: torch.Tensor
#     local_mutual_M_R_x: torch.Tensor
#     local_mutual_M_R_x_prime: torch.Tensor
#     local_mutual_M_R_y: torch.Tensor
#     local_mutual_M_R_y_prime: torch.Tensor
#     shared_x: torch.Tensor
#     shared_y: torch.Tensor
#     img_recon_vi: torch.Tensor
#     img_recon_ir: torch.Tensor
#     R_y_x: torch.Tensor
#     R_x_y: torch.Tensor
#     shuffle_x: torch.Tensor
#     shuffle_y: torch.Tensor
#     fake_x: torch.Tensor
#     fake_y: torch.Tensor
#
# class DiscriminatorOutputs(NamedTuple):
#     disentangling_information_x: torch.Tensor
#     disentangling_information_x_prime: torch.Tensor
#     disentangling_information_y: torch.Tensor
#     disentangling_information_y_prime: torch.Tensor
