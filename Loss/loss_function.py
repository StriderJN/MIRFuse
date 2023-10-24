import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DJSLoss(nn.Module):
    """
    JS散度loss
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        """Estimator of the Jensen Shannon Divergence see paper equation (2)
        论文公式(2)JS散度估计器，T：输入两分布乘积，T_prime：联合分布
        Args:
            T (torch.Tensor): Statistique network estimation from the marginal distribution P(x)P(z)
            T_prime (torch.Tensor): Statistique network estimation from the joint distribution P(xz)

        Returns:
            float: DJS estimation value
        """
        joint_expectation = (-F.softplus(-T)).mean()    # 联合期望
        marginal_expectation = F.softplus(T_prime).mean()   # 乘积期望
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info

class HSICLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x_feat: torch.Tensor, y_feat: torch.Tensor):
        # x_prime = (x_repr.unsqueeze(1) - x_repr.unsqueeze(2)).cpu().detach().numpy()
        # y_prime = (y_repr.unsqueeze(1) - y_repr.unsqueeze(2)).cpu().detach().numpy()
        # hsic = 0
        # for i in range(x_prime.shape[0]):
        #     Kx = x_prime[i]
        #     Ky = y_prime[i]
        #     Kx = np.exp(- Kx ** 2)  # 计算核矩阵
        #     Ky = np.exp(- Ky ** 2)  # 计算核矩阵
        #     Kxy = np.dot(Kx, Ky)
        #     n = Kxy.shape[0]
        #     h = np.trace(Kxy) / n ** 2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
        #     hsic +=  h * n ** 2 / (n - 1) ** 2
        # return abs(hsic)
        x = F.adaptive_avg_pool3d(x_feat, (1, 128, 128))
        y = F.adaptive_avg_pool3d(y_feat, (1, 128, 128))
        x_prime = (x - torch.transpose(x, 3, 2))
        y_prime = (y - torch.transpose(y, 3, 2))
        hsic = 0
        for i in range(x_prime.shape[0]):
            for j in range(x_prime.shape[1]):
                Kx = x_prime[i][j]
                Ky = y_prime[i][j]
                Kx = torch.exp(- Kx ** 2)  # 计算核矩阵
                Ky = torch.exp(- Ky ** 2)  # 计算核矩阵
                Kxy = Kx.mm(Ky)
                n = Kxy.shape[0]
                h = Kxy.trace() / n ** 2 + torch.mean(Kx) * torch.mean(Ky) - 2 * torch.mean(Kxy) / n
                hsic += h * n ** 2 / (n - 1) ** 2

        return abs(hsic)


if __name__ == "__main__":
    x_feat = torch.rand(64, 128, 128, 128)
    x = F.adaptive_avg_pool3d(x_feat, (1,128, 128))
    print(x.shape)
    y_feat = torch.rand(64, 128, 128, 128)
    HSIC = HSICLoss()
    a = HSIC(x_feat, y_feat)
    print(a)