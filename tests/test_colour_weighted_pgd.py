import numpy as np
import pytest
import torch
from skimage.color import rgb2lab

from src.models.colour_weighted_pgd import (VARIANTS, ColourCueSelector, compute_gate_inputs,
                                           build_colour_weighted_pgd, compute_neighbour_colour_differences, convert_rgb_to_lab)
from src.pgd.loader import load_pgd_network, get_frozen_pgd


def test_rgb_to_lab_matches_skimage():
    rgb = np.random.default_rng(0).uniform(size=(500, 3))
    ours = convert_rgb_to_lab(torch.tensor(rgb, dtype=torch.float64)).numpy()
    assert np.allclose(ours, rgb2lab(rgb), atol=1e-2)


def test_differences_separate_brightness_from_hue():
    rgb = torch.tensor([[0.8, 0.1, 0.1], [0.5, 0.06, 0.06], [0.3, 0.3, 1.0]])
    lab = convert_rgb_to_lab(rgb)
    idx = torch.tensor([0, 1, 2, 0, 0, 0, 0, 0, 0])
    diff = compute_neighbour_colour_differences(lab, idx, 3)[0]
    assert diff[0].abs().max() == 0
    d_dark, d_blue = diff[1], diff[2]
    assert d_dark[0] > d_blue[0]
    assert d_blue[1] > d_dark[1]


def test_surface_variation_is_zero_on_a_plane_and_high_for_a_blob():
    g = torch.Generator().manual_seed(0)
    plane = torch.cat([torch.rand(1, 16, 2, generator=g), torch.zeros(1, 16, 1)], dim=2)
    blob = torch.randn(1, 16, 3, generator=g)
    zeros = torch.zeros(1, 16, 2)
    assert compute_gate_inputs(plane, zeros)[0, 0] < 1e-5
    assert compute_gate_inputs(blob, zeros)[0, 0] > 0.2


def _cues():
    g = torch.Generator().manual_seed(3)
    return torch.randn(50, 16, 3, generator=g) * 0.05, torch.rand(50, 16, 2, generator=g)


def test_selector_passes_on_what_each_variant_may_use():
    relative, cues = _cues()
    assert torch.equal(ColourCueSelector("geometry_only")(relative, cues), torch.zeros_like(cues))
    gray = ColourCueSelector("grayscale")(relative, cues)
    assert torch.equal(gray[..., 0], cues[..., 0]) and torch.equal(gray[..., 1], torch.zeros_like(cues[..., 1]))
    assert torch.equal(ColourCueSelector("rgb_fusion")(relative, cues), cues)
    gated = ColourCueSelector("gated")
    out = gated(relative, cues)
    g = gated.gate(compute_gate_inputs(relative, cues))
    assert g.shape == (50, 1) and (g > 0).all() and (g < 1).all()
    assert torch.allclose(out, cues * g.unsqueeze(1))


def test_grayscale_is_the_brightness_of_a_grey_image():
    g = torch.Generator().manual_seed(4)
    rgb = torch.rand(200, 3, generator=g)
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    y = linear @ torch.tensor([0.2126729, 0.7151522, 0.0721750])
    grey = torch.where(y > 0.0031308, 1.055 * y.clamp(min=1e-12) ** (1 / 2.4) - 0.055, 12.92 * y)
    lab, lab_grey = convert_rgb_to_lab(rgb), convert_rgb_to_lab(grey.unsqueeze(1).expand(-1, 3))
    assert torch.allclose(lab[:, 0], lab_grey[:, 0], atol=1e-3)
    assert lab_grey[:, 1:].abs().max() < 1e-2


def test_gate_inputs_work_under_mixed_precision():
    relative, cues = _cues()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inputs = compute_gate_inputs(relative, cues)
        out = ColourCueSelector("gated")(relative, cues)
    assert torch.allclose(inputs, compute_gate_inputs(relative, cues), atol=1e-6) and torch.isfinite(out).all()


@pytest.fixture(scope="module")
def patches():
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 1000, 3, generator=g) * 0.1, torch.rand(2, 1000, 3, generator=g)


@pytest.fixture(scope="module")
def original(patches):
    network = load_pgd_network().eval()
    offset = torch.tensor([1000, 2000], dtype=torch.int32)
    with torch.no_grad():
        return network(patches[0], None, offset)[0]


def run(model, xyz, rgb):
    with torch.no_grad():
        return model.eval()(xyz, rgb)[0]


def activate_colour(model):
    torch.nn.init.normal_(model.colour_mre.rfe_1.colour_conv.weight, std=1.0)
    return model


@pytest.mark.parametrize("variant", VARIANTS)
def test_fresh_model_is_exactly_pgd(variant, patches, original):
    assert torch.equal(run(build_colour_weighted_pgd(variant), *patches), original)


def test_all_variants_have_the_same_parameters():
    counts = {v: (sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad))
              for v in VARIANTS for m in [build_colour_weighted_pgd(v)]}
    assert len(set(counts.values())) == 1, counts


def test_geometry_only_ignores_colour(patches, original):
    model = activate_colour(build_colour_weighted_pgd("geometry_only"))
    assert torch.equal(run(model, patches[0], torch.rand_like(patches[1])), original)


@pytest.mark.parametrize("variant", ["grayscale", "rgb_fusion", "gated"])
def test_constant_colour_falls_back_to_geometry(variant, patches, original):
    model = activate_colour(build_colour_weighted_pgd(variant))
    assert torch.allclose(run(model, patches[0], torch.full_like(patches[1], 0.5)), original, atol=1e-6)


@pytest.mark.parametrize("variant", ["grayscale", "rgb_fusion", "gated"])
def test_colour_changes_the_result_once_trained(variant, patches, original):
    model = activate_colour(build_colour_weighted_pgd(variant))
    assert not torch.allclose(run(model, *patches), original, atol=1e-6)


def test_training_mode_keeps_frozen_parts_fixed():
    model = build_colour_weighted_pgd("gated").train()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("network.encoder_blocks.1.") for n in trainable)
    assert model.network.encoder_blocks[1].training
    assert not model.network.encoder_blocks[2].training
    assert not any(cb.training for cb in model.network.codebooks if cb is not None)


def test_first_block_has_the_documented_size():
    model = build_colour_weighted_pgd("gated")
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 15801


def test_batch_norm_statistics_stay_fixed_during_training(patches):
    model = build_colour_weighted_pgd("geometry_only").train()
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert norms and not any(m.training for m in norms)
    before = {k: v.clone() for k, v in model.state_dict().items() if "running" in k}
    model(*patches)
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())
    assert any(p.requires_grad for m in norms for p in m.parameters())


def test_gradient_predicts_the_change_of_the_loss():
    g = torch.Generator().manual_seed(8)
    plane = torch.cat([torch.rand(2, 1000, 2, generator=g) - 0.5, torch.zeros(2, 1000, 1)], dim=2) * 0.2
    noisy = plane + torch.randn(2, 1000, 3, generator=g) * 0.004
    rgb = torch.rand(2, 1000, 3, generator=g)
    model = build_colour_weighted_pgd("rgb_fusion").train()
    params = [p for p in model.parameters() if p.requires_grad]

    def loss():
        displacement, _ = model(noisy, rgb)
        return ((noisy + displacement - plane) ** 2).sum(dim=-1).sqrt().mean()

    value = loss()
    grads = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(torch.autograd.grad(value, params, allow_unused=True), params)]
    norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    eps = 1e-4
    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= eps * g / norm
        actual = loss() - value
    assert 0.5 < (actual / (-eps * norm)).item() < 1.5


def test_get_frozen_pgd_is_the_network_and_ignores_colour(patches, original):
    model = get_frozen_pgd()
    with torch.no_grad():
        assert torch.equal(model(*patches)[0], original)
        assert torch.equal(model(patches[0], torch.rand_like(patches[1]))[0], original)
    assert not any(p.requires_grad for p in model.parameters())


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="variant"):
        build_colour_weighted_pgd("colourful")
