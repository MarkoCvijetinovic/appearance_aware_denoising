import torch

from .loader import add_pgd_to_path

add_pgd_to_path()
from models.utils import farthest_point_sampling


@torch.no_grad()
def patch_based_denoise(model, xyz, rgb, patch_size=1000, seed_k=6, seed_k_alpha=10, patches_per_pass=None):
    was_training = model.training
    model.eval()
    try:
        return _patch_based_denoise(model, xyz, rgb, patch_size, seed_k, seed_k_alpha, patches_per_pass)
    finally:
        model.train(was_training)


def _patch_based_denoise(model, xyz, rgb, patch_size, seed_k, seed_k_alpha, patches_per_pass=None):
    # the only difference to the original PGD is that we now also pass the colour to the network
    n_points = xyz.shape[0]
    num_patches = int(seed_k * n_points / patch_size)
    seeds, _ = farthest_point_sampling(xyz.unsqueeze(0), num_patches)
    dists = torch.cdist(seeds[0], xyz)
    patch_dists, patch_idx = dists.topk(patch_size, dim=1, largest=False)

    relative = patch_dists / patch_dists[:, -1:]
    closeness = torch.full((num_patches, n_points), float("inf"), device=xyz.device)
    closeness.scatter_(1, patch_idx, relative)
    best_patch = closeness.argmin(dim=0)

    patches = xyz[patch_idx] - seeds[0].unsqueeze(1)
    patch_rgb = rgb[patch_idx]
    # hier adden wir nun die Farbe zu den Patches 
    step = patches_per_pass or max(1, int(n_points / (seed_k_alpha * patch_size)))
    denoised = torch.empty_like(patches)
    for start in range(0, num_patches, step):
        end = min(start + step, num_patches)
        displacement, _ = model(patches[start:end], patch_rgb[start:end])
        denoised[start:end] = patches[start:end] + displacement
    denoised = denoised + seeds[0].unsqueeze(1)

    position = (patch_idx[best_patch] == torch.arange(n_points, device=xyz.device).unsqueeze(1)).float().argmax(dim=1)
    return denoised[best_patch, position]


@torch.no_grad()
def denoise(model, xyz, rgb, niters=2, **patch_kwargs):
    """Denoise a whole cloud as PGD's test.py does, with colours kept next to their points.

    Normalise to the unit sphere, run patch_based_denoise niters times, undo the normalisation.
    xyz and rgb have shape (N, 3); returns the denoised xyz (N, 3).
    """
    centre = (xyz.max(dim=0).values + xyz.min(dim=0).values) / 2
    scale = (xyz - centre).norm(dim=1).max()
    current = (xyz - centre) / scale
    for _ in range(niters):
        current = patch_based_denoise(model, current, rgb, **patch_kwargs)
    return current * scale + centre
