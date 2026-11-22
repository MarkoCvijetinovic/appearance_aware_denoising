from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class GSODataset(Dataset):
    def __init__(self, root, split, xyz_noise=0.02, rgb_noise=0.05):
        self.samples = sorted(
            (Path(root) / split).glob(f"*/noise_xyz_{xyz_noise:g}_rgb_{rgb_noise:g}.npz")
        )

        if not self.samples:
            raise ValueError(f"No samples found for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        noisy_path = self.samples[index]
        clean_path = noisy_path.parent / "clean.npz"

        with np.load(noisy_path) as data:
            noisy_xyz = data["xyz"].astype(np.float32)
            noisy_rgb = data["rgb"].astype(np.float32)

        with np.load(clean_path) as data:
            clean_xyz = data["xyz"].astype(np.float32)

        return {
            "noisy_xyz": torch.from_numpy(noisy_xyz),
            "noisy_rgb": torch.from_numpy(noisy_rgb),
            "clean_xyz": torch.from_numpy(clean_xyz),
        }