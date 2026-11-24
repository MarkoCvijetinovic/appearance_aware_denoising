import torch

from src.data.gso_patches import GSOPatchDataset


def test_patches_have_aligned_points_and_colours(gso_root):
    dataset = GSOPatchDataset(gso_root, "train", xyz_noise=(0.01,), rgb_noise=(0.0,), patch_size=300,
                              patches_per_object=4, rotate=False, seed=0)
    assert len(dataset) == 2 * 4
    for patch in dataset:
        noisy, clean, rgb = patch["noisy_xyz"], patch["clean_xyz"], patch["noisy_rgb"]
        assert noisy.shape == clean.shape == rgb.shape == (300, 3)
        assert (noisy - clean).norm(dim=1).max() < 0.1
        lower = clean[:, 2] < clean[:, 2].min() + 1e-4
        red = rgb[:, 0] > rgb[:, 2]
        assert torch.equal(lower, red)


def test_fixed_seed_gives_the_same_patches(gso_root):
    a = GSOPatchDataset(gso_root, "val", patch_size=300, patches_per_object=2, seed=0, xyz_noise=(0.01, 0.02))
    b = GSOPatchDataset(gso_root, "val", patch_size=300, patches_per_object=2, seed=0, xyz_noise=(0.01, 0.02))
    assert all(torch.equal(a[i]["noisy_xyz"], b[i]["noisy_xyz"]) for i in range(len(a)))
