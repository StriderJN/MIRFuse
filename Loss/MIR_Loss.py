import torch
import torch.nn as nn
from kornia.losses import ssim
from Loss.loss_function import DJSLoss, HSICLoss
from utils.output_type import MIRlosses,MIRFOutputs
import torch.nn.functional as F

class MIRLoss(nn.Module):
    def __init__(
            self,
            local_mutual_loss_coeff: float,
            global_mutual_loss_coeff: float,
            shared_loss_coeff: float,
            window_size: int,
            gamma
    ):
        super(MIRLoss, self).__init__()
        # Consistency Constraint
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.shared_loss_coeff = shared_loss_coeff
        self.djs_loss = DJSLoss()
        self.l1_loss = nn.MSELoss()

        # Heterogeneity constraint
        self.HSIC_loss = HSICLoss()

        # Reconstruction Constraint
        self.ssim_loss = ssim.SSIMLoss(window_size=window_size, reduction='mean')
        self.MSELoss = nn.MSELoss()
        self.gamma = gamma

    def forward(
            self,
            img_vi : torch.Tensor,
            img_ir : torch.Tensor,
            outputs : MIRFOutputs,
    ):
        # Compute Global mutual loss
        global_mutual_loss_x = self.djs_loss(
            T=outputs.global_mutual_M_R_x,
            T_prime=outputs.global_mutual_M_R_x_prime,
        )
        global_mutual_loss_y = self.djs_loss(
            T=outputs.global_mutual_M_R_y,
            T_prime=outputs.global_mutual_M_R_y_prime,
        )
        global_mutual_loss = (global_mutual_loss_x + global_mutual_loss_y) * self.global_mutual_loss_coeff

        # Compute Local mutual loss

        local_mutual_loss_x = self.djs_loss(
            T=outputs.local_mutual_M_R_x,
            T_prime=outputs.local_mutual_M_R_x_prime,
        )
        local_mutual_loss_y = self.djs_loss(
            T=outputs.local_mutual_M_R_y,
            T_prime=outputs.local_mutual_M_R_y_prime,
        )
        local_mutual_loss = (local_mutual_loss_x + local_mutual_loss_y) * self.local_mutual_loss_coeff

        # Compute total mutual loss
        mim_loss = global_mutual_loss + local_mutual_loss

        # Compute L1 on shared features
        # shared_x = F.adaptive_avg_pool3d(outputs.sh_M_x, (1, 128, 128))
        # shared_y = F.adaptive_avg_pool3d(outputs.sh_M_y, (1, 128, 128))
        shared_loss = self.l1_loss(outputs.shared_x, outputs.shared_y)
        shared_loss = shared_loss * self.shared_loss_coeff

        # Compute total disparity loss
        het_loss_vi = self.HSIC_loss(outputs.sh_M_x, outputs.ex_M_x)
        het_loss_ir = self.HSIC_loss(outputs.sh_M_y, outputs.ex_M_y)
        het_loss = het_loss_vi + het_loss_ir

        # Compute reconstruction loss, include l1 & ssim
        mse_loss_VI = self.gamma * self.ssim_loss(img_vi, outputs.img_recon_vi) + self.l1_loss(img_vi, outputs.img_recon_vi)
        mse_loss_IR = self.gamma * self.ssim_loss(img_ir, outputs.img_recon_ir) + self.l1_loss(img_ir, outputs.img_recon_ir)
        mse_loss = mse_loss_VI + mse_loss_IR

        total_loss = 0 * mim_loss + 0 * shared_loss  + 0 * het_loss + 40 * mse_loss

        return MIRlosses(
            mim_loss=mim_loss,
            shared_loss=shared_loss,
            mse_loss=mse_loss,
            het_loss=het_loss,
            total_loss=total_loss
        )