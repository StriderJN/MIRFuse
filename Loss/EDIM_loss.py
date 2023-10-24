import torch
import torch.nn as nn
from kornia.losses import ssim
from Model.EDIM import *
from Loss.loss_function import *
from utils.output_type import EDIMoutput, Recons_Losses, DiscrLosses,  DiscriminatorOutput  # JOINTOutput,


class EDIMLoss(nn.Module):
    def __init__(
            self,
            window_size:int,
            disentangling_loss_coeff: float,
            gamma
    ):
        super(EDIMLoss, self).__init__()
        self.ssimloss = ssim.SSIMLoss(window_size=window_size, reduction='mean')
        self.MSELoss = nn.MSELoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()
        self.gamma = gamma
        self.disentangling_loss_coeff = disentangling_loss_coeff

    def reconstruction_loss(
    # def __call__(
            self,
            img_vi,
            img_ir,
            outputs : EDIMoutput,
            # outputs : JOINTOutput
    ):
        # ssim_loss_VI = self.ssimloss(img_vi, outputs.img_recon_vi)
        # ssim_loss_IR = self.ssimloss(img_ir, outputs.img_recon_ir)
        mse_loss_VI = self.gamma * self.ssimloss(img_vi, outputs.img_recon_vi) + self.MSELoss(img_vi, outputs.img_recon_vi)
        mse_loss_IR = self.gamma * self.ssimloss(img_ir, outputs.img_recon_ir) + self.MSELoss(img_ir, outputs.img_recon_ir)
        recons_loss = mse_loss_VI + mse_loss_IR

        gan_loss_x_g = self.generator_loss(fake_logits=outputs.fake_x)
        gan_loss_y_g = self.generator_loss(fake_logits=outputs.fake_y)

        gan_loss_g = (gan_loss_x_g + gan_loss_y_g) * self.disentangling_loss_coeff

        total_loss = gan_loss_g + recons_loss
        return Recons_Losses(
            total_loss =  total_loss,
            recons_loss=recons_loss,
            gan_loss_g=gan_loss_g,
        )
        # return Recons_Losses(
        #     recons_loss=recons_loss,
        #     ssim_loss_VI=ssim_loss_VI,
        #     ssim_loss_IR=ssim_loss_IR,
        #     mse_loss_VI=mse_loss_VI,
        #     mse_loss_IR=mse_loss_IR
        # )

    def compute_discriminator_loss(
            self,
            discr_outputs: DiscriminatorOutput
            # discr_outputs: DiscriminatorOutputs
    ):
        """Discriminator loss see paper equation (9)

               Args:
                   discr_outputs (DiscriminatorOutputs): Output of the forward pass of the discriminators model

               Returns:
                   DiscrLosses: Discriminator losses
               """
        gan_loss_x_d = self.discriminator_loss(
            real_logits=discr_outputs.disentangling_information_x_prime,
            fake_logits=discr_outputs.disentangling_information_x,
        )
        gan_loss_y_d = self.discriminator_loss(
            real_logits=discr_outputs.disentangling_information_y_prime,
            fake_logits=discr_outputs.disentangling_information_y,
        )

        gan_loss_d = (gan_loss_x_d + gan_loss_y_d) * self.disentangling_loss_coeff

        return DiscrLosses(gan_loss_d=gan_loss_d)




if __name__ == "__main__":
    img_size = 192
    vi = torch.zeros(64, 1, img_size, img_size).to('cuda:2')
    ir = torch.zeros(64, 1, img_size, img_size).to('cuda:2')
    dir = '/data/zsl/SDIM/weight'
    trained_sh_encoder_VI = torch.load(dir + '/sh_encoder_VI.pth')
    trained_sh_encoder_IR = torch.load(dir + '/sh_encoder_IR.pth')
    model = EDIM(channels=1, filter=32, trained_encoder_vi=trained_sh_encoder_VI,
                 trained_encoder_ir=trained_sh_encoder_IR).to('cuda:2')
    out = model(vi, ir)

    recon = reconstruction_loss(11)
    a = recon(vi, ir, out, 100)
    print(a)
