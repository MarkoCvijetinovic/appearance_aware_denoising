import importlib
import sys
import types

import torch

from src.models.colour_weighted_pgd import build_colour_weighted_pgd
from src.pgd.denoise import denoise, patch_based_denoise
from src.pgd.loader import PGD_ROOT, load_pgd_network, get_frozen_pgd

SMALL = dict(patch_size=200, seed_k=5, seed_k_alpha=10)


def pgd_reference(network):
    hidden = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "datasets" or k.startswith("datasets.")}
    stub = types.ModuleType("datasets")
    stub.__path__ = [str(PGD_ROOT / "datasets")]
    sys.modules["datasets"] = stub
    sys.path.insert(0, str(PGD_ROOT))
    try:
        pgd = importlib.import_module("models.pgd")
    finally:
        sys.path.remove(str(PGD_ROOT))
        for k in [k for k in sys.modules if k == "datasets" or k.startswith("datasets.")]:
            del sys.modules[k]
        sys.modules.update(hidden)
    model = pgd.PGDModel.__new__(pgd.PGDModel)
    torch.nn.Module.__init__(model)
    model.feature_nets = network.eval()
    return model


def test_same_result_as_pgd_patch_based_denoise():
    g = torch.Generator().manual_seed(0)
    xyz = torch.rand(2000, 3, generator=g) - 0.5
    rgb = torch.rand(2000, 3, generator=g)
    expected = pgd_reference(load_pgd_network()).patch_based_denoise(xyz, **SMALL)
    for model in (get_frozen_pgd(), build_colour_weighted_pgd("gated")):
        ours = patch_based_denoise(model, xyz, rgb, **SMALL)
        assert torch.allclose(ours, expected, atol=1e-6)


class ColourShift(torch.nn.Module):
    def forward(self, xyz, rgb):
        return 0.01 * (rgb - 0.5), torch.tensor(0.0)


def test_colours_travel_with_their_points():
    g = torch.Generator().manual_seed(1)
    xyz = torch.rand(2000, 3, generator=g) - 0.5
    rgb = torch.rand(2000, 3, generator=g)
    out = patch_based_denoise(ColourShift(), xyz, rgb, **SMALL)
    assert torch.allclose(out, xyz + 0.01 * (rgb - 0.5), atol=1e-6)


def test_pgd_on_cpu_denoises_a_noisy_plane():
    g = torch.Generator().manual_seed(2)
    n = 5000
    plane = torch.cat([torch.rand(n, 2, generator=g) - 0.5, torch.zeros(n, 1)], dim=1)
    noisy = plane + torch.randn(n, 3, generator=g) * 0.01
    out = denoise(get_frozen_pgd(), noisy, torch.full_like(noisy, 0.5), niters=1,
                  patch_size=1000, seed_k=5, seed_k_alpha=5)
    assert out[:, 2].std() < 0.7 * noisy[:, 2].std()


def test_patches_per_pass_changes_nothing():
    g = torch.Generator().manual_seed(3)
    xyz = torch.rand(2000, 3, generator=g) - 0.5
    rgb = torch.rand(2000, 3, generator=g)
    model = build_colour_weighted_pgd("gated")
    torch.nn.init.normal_(model.colour_mre.rfe_1.colour_conv.weight, std=0.1)
    one = patch_based_denoise(model, xyz, rgb, **SMALL)
    many = patch_based_denoise(model, xyz, rgb, patches_per_pass=7, **SMALL)
    assert torch.allclose(one, many, atol=1e-5)
