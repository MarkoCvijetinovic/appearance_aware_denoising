import torch
from torch import nn


class AppearancePGD(nn.Module):
    def __init__(self, pgd):
        super().__init__()

        self.pgd = pgd

        self.adapter = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, noisy_xyz, noisy_rgb):
        with torch.no_grad():
            base_xyz = run_pgd(self.pgd, noisy_xyz)

        features = torch.cat([noisy_xyz, noisy_rgb, base_xyz - noisy_xyz], dim=-1)

        correction = self.adapter(features)

        return base_xyz + correction