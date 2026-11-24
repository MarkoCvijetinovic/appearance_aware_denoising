import functools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "cpu_shims"))
sys.path.insert(1, str(ROOT))

if not torch.cuda.is_available():
    _tensor = torch.tensor

    @functools.wraps(_tensor)
    def _tensor_cpu(*args, device=None, **kwargs):
        if device is not None and str(device).startswith("cuda"):
            device = "cpu"
        return _tensor(*args, device=device, **kwargs)

    torch.tensor = _tensor_cpu

    torch.Tensor.cuda = lambda self, *args, **kwargs: self

import numpy as np
import pytest

XYZ_NOISE = (0.01, 0.02)
RGB_NOISE = (0.0, 0.05)


def two_plates(n_points, gap, rng):
    half = n_points // 2
    plate = rng.uniform(-0.5, 0.5, size=(n_points, 2))
    xyz = np.zeros((n_points, 3), dtype=np.float32)
    xyz[:, :2] = plate
    xyz[half:, 2] = gap
    rgb = np.zeros((n_points, 3), dtype=np.float32)
    rgb[:half] = (0.8, 0.1, 0.1)
    rgb[half:] = (0.1, 0.1, 0.8)
    return xyz, rgb


@pytest.fixture(scope="session")
def gso_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("gso")
    rng = np.random.default_rng(0)
    for split, n_objects in (("train", 2), ("val", 1), ("test", 1)):
        for i in range(n_objects):
            folder = root / split / f"{split}_{i}"
            folder.mkdir(parents=True)
            xyz, rgb = two_plates(2000, gap=0.05, rng=rng)
            np.savez_compressed(folder / "clean.npz", xyz=xyz, rgb=rgb)
            for x in XYZ_NOISE:
                for r in RGB_NOISE:
                    noisy_xyz = (xyz + rng.normal(0, x, xyz.shape)).astype(np.float32)
                    noisy_rgb = np.clip(rgb + rng.normal(0, r, rgb.shape), 0, 1).astype(np.float32)
                    np.savez_compressed(folder / f"noise_xyz_{x:g}_rgb_{r:g}.npz", xyz=noisy_xyz, rgb=noisy_rgb)
    return root
