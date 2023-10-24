from Loss.loss_function import *
from utils.output_type import SDIMlosses
import torch.nn.functional as F

class SDIMLoss(nn.Module):
    """
    Loss function to extract shared information from the image, see paper equation (5)

    Args:
        local_mutual_loss_coeff (float): Coefficient of the local Jensen Shannon loss
        局部JS损失系数α_sh
        global_mutual_loss_coeff (float): Coefficient of the global Jensen Shannon loss
        全局JS损失系数β_sh
        shared_loss_coeff (float): Coefficient of L1 loss, see paper equation (6)
        共享表征损失系数
    """

    def __init__(
        self,
        local_mutual_loss_coeff: float,
        global_mutual_loss_coeff: float,
        shared_loss_coeff: float,
    ):

        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.shared_loss_coeff = shared_loss_coeff

        self.djs_loss = DJSLoss()
        self.l1_loss = nn.MSELoss()  # see equation (6)

    def __call__(self, sdim_outputs):
        """
        Compute all the loss functions needed to extract the shared part

        Args:
            sdim_outputs (SDIMOutputs): Output of the forward pass of the shared information model

        Returns:
            SDIMLosses: Shared information losses
            共享信息损失
        """

        # Compute Global mutual loss
        global_mutual_loss_x = self.djs_loss(
            T=sdim_outputs.global_mutual_M_R_x,
            T_prime=sdim_outputs.global_mutual_M_R_x_prime,
        )
        global_mutual_loss_y = self.djs_loss(
            T=sdim_outputs.global_mutual_M_R_y,
            T_prime=sdim_outputs.global_mutual_M_R_y_prime,
        )
        global_mutual_loss = (
            global_mutual_loss_x + global_mutual_loss_y
        ) * self.global_mutual_loss_coeff

        # Compute Local mutual loss

        local_mutual_loss_x = self.djs_loss(
            T=sdim_outputs.local_mutual_M_R_x,
            T_prime=sdim_outputs.local_mutual_M_R_x_prime,
        )
        local_mutual_loss_y = self.djs_loss(
            T=sdim_outputs.local_mutual_M_R_y,
            T_prime=sdim_outputs.local_mutual_M_R_y_prime,
        )
        local_mutual_loss = (
            local_mutual_loss_x + local_mutual_loss_y
        ) * self.local_mutual_loss_coeff

        # Compute L1 on shared features
        # shared_loss = self.l1_loss(sdim_outputs.shared_x, sdim_outputs.shared_y)
        shared_loss = torch.mean(F.pairwise_distance(sdim_outputs.shared_x, sdim_outputs.shared_y))
        shared_loss = shared_loss * self.shared_loss_coeff

        encoder_loss = global_mutual_loss + local_mutual_loss + shared_loss

        total_losses = (
            global_mutual_loss
            + local_mutual_loss
            + shared_loss
        )

        return SDIMlosses(
            total_loss=total_losses,
            shared_loss = shared_loss
        )