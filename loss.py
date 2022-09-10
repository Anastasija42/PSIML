import torch
from torch import nn
import numpy as np


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))

        return 1 - torch.mean(prod / (fake_norm * real_norm))


def imgrad(img):
    N, C, _, _ = img.size()

    img = torch.mean(img, dim=1, keepdim=True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)


depth_criterion = nn.MSELoss()
grad_criterion = GradLoss()
normal_criterion = NormalLoss()
