import torch.nn as nn

class AppearanceAwarePGD(nn.Module):
    def __init__(self, pgd, color_module):
        super().__init__()
        self.pgd = pgd
        self.color_module = color_module

    def forward(self, noisy_xyz, rgb):
        denoised_xyz = self.pgd_denoise(noisy_xyz)
        correction = self.color_module(noisy_xyz, rgb, denoised_xyz)
        return denoised_xyz + correction