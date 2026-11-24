import random

import torch
from torch.utils.data import Dataset

from src.models.torch_dataset import GSODataset


def normalise_unit_sphere(clean, *others):
    centre = (clean.max(dim=0).values + clean.min(dim=0).values) / 2
    scale = (clean - centre).norm(dim=1).max()
    return [(c - centre) / scale for c in (clean, *others)]


def random_rotation(generator):
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class GSOPatchDataset(Dataset):
    def __init__(self, root, split, xyz_noise=(0.01, 0.02, 0.025), rgb_noise=(0.0, 0.05), patch_size=1000,
                 patches_per_object=100, rotate=True, seed=None):
        self.levels = [GSODataset(root, split, xyz_noise=x, rgb_noise=r) for x in xyz_noise for r in rgb_noise]
        objects = [[p.parent for p in level.samples] for level in self.levels]
        if any(o != objects[0] for o in objects):
            raise ValueError("every object needs a file for every requested noise combination")
        self.n_objects = len(objects[0])
        self.patch_size = patch_size
        self.patches_per_object = patches_per_object
        self.rotate = rotate
        self.seed = seed

    def __len__(self):
        return self.n_objects * self.patches_per_object

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        generator = torch.Generator()
        generator.manual_seed(self.seed + index if self.seed is not None else random.getrandbits(63))
        level = self.levels[int(torch.randint(len(self.levels), (1,), generator=generator))]
        sample = level[index % self.n_objects]

        clean, noisy = normalise_unit_sphere(sample["clean_xyz"], sample["noisy_xyz"])
        if self.rotate:
            rotation = random_rotation(generator)
            clean, noisy = clean @ rotation.T, noisy @ rotation.T

        n_points = noisy.shape[0]
        seed_idx = int(torch.randint(n_points, (1,), generator=generator))
        size = min(self.patch_size, n_points)
        idx = (noisy - noisy[seed_idx]).norm(dim=1).topk(size, largest=False).indices
        seed_point = noisy[seed_idx]
        return {
            "noisy_xyz": noisy[idx] - seed_point,
            "clean_xyz": clean[idx] - seed_point,
            "noisy_rgb": sample["noisy_rgb"][idx],
        }
