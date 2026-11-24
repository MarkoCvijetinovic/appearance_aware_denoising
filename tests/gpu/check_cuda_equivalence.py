import argparse
import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--stand-ins", action="store_true", help="use the CPU stand-ins instead of CUDA (on the CPU)")
    args = parser.parse_args()

    if args.stand_ins:
        sys.path.insert(0, str(ROOT / "tests" / "cpu_shims"))
    sys.path.insert(1, str(ROOT))
    import torch
    if args.stand_ins:
        import tests.conftest
    from src.models.colour_weighted_pgd import VARIANTS, build_colour_weighted_pgd
    from src.pgd.denoise import patch_based_denoise
    from src.pgd.loader import PGD_ROOT, load_pgd_network, get_frozen_pgd
    from src.models.torch_dataset import GSODataset
    from src.data.gso_patches import normalise_unit_sphere

    device = "cpu" if args.stand_ins else "cuda"
    dataset = GSODataset(ROOT / "data" / "gso", "test", xyz_noise=0.02, rgb_noise=0.05)
    sample = dataset[0]
    name = dataset.samples[0].parent.name
    xyz = sample["noisy_xyz"].to(device)
    rgb = sample["noisy_rgb"].to(device)
    result = {"object": name, "device": device}

    hidden = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "datasets" or k.startswith("datasets.")}
    stub = types.ModuleType("datasets")
    stub.__path__ = [str(PGD_ROOT / "datasets")]
    sys.modules["datasets"] = stub
    sys.path.insert(0, str(PGD_ROOT))
    pgd_module = importlib.import_module("models.pgd")
    sys.path.remove(str(PGD_ROOT))
    sys.modules.update(hidden)
    reference = pgd_module.PGDModel.__new__(pgd_module.PGDModel)
    torch.nn.Module.__init__(reference)
    reference.feature_nets = load_pgd_network().to(device).eval()

    normalised = normalise_unit_sphere(xyz)[0]
    with torch.no_grad():
        expected = reference.patch_based_denoise(normalised, patch_size=1000, seed_k=6, seed_k_alpha=10)
    diffs = {"get_frozen_pgd": (patch_based_denoise(get_frozen_pgd().to(device), normalised, rgb) - expected).abs().max().item()}
    for variant in VARIANTS:
        model = build_colour_weighted_pgd(variant).to(device)
        ours = patch_based_denoise(model, normalised, rgb)
        diffs[variant] = (ours - expected).abs().max().item()
    result["max_abs_diff_to_pgd_patch_based_denoise"] = diffs

    model = build_colour_weighted_pgd("gated").to(device)
    for module in model.modules():
        if hasattr(module, "colour_conv"):
            torch.nn.init.normal_(module.colour_conv.weight, std=0.1)
    single = patch_based_denoise(model, normalised, rgb)
    batched = patch_based_denoise(model, normalised, rgb, patches_per_pass=20)
    result["max_abs_diff_patches_per_pass_20"] = (single - batched).abs().max().item()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
