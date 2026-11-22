import torch
from torch import nn

def run_pgd(pgd, noisy_xyz):
    with torch.no_grad():
        prediction = noisy_xyz

        for _ in range(1000):
            prediction = pgd.patch_based_denoise(
                pcl_noisy=prediction,
                patch_size=1000,
                seed_k=6,
                seed_k_alpha=10,
            )

    return prediction

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