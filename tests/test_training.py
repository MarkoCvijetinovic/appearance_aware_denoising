import csv
import json

import pytest
import torch

from src.models.colour_weighted_pgd import build_colour_weighted_pgd
from src.pgd.loader import PROJECT_ROOT
from src.training.train import apply_overrides, count_parameters, load_config, load_trained, make_loaders, train

CONFIG = PROJECT_ROOT / "configs" / "task4" / "colour_weighted_pgd.yaml"


def trainable_prefixes():
    model = build_colour_weighted_pgd("gated")
    return tuple(f"{name}." for name, module in model.named_modules() if any(module is r for r in model.get_trainable_roots()))


def tiny(config, gso_root, **overrides):
    config["data"].update(root=str(gso_root), xyz_noise=[0.01, 0.02], patch_size=200, patches_per_object=2,
                          val_patches_per_object=1)
    config["training"].update(steps=4, batch_size=2, val_every=2, num_workers=0, amp=False, lr=1e-2)
    return apply_overrides(config, **overrides)


def test_patch_stream_depends_only_on_the_seed(gso_root):
    def first_batch(variant, seed):
        config = tiny(load_config(CONFIG), gso_root, variant=variant, seed=seed)
        build_colour_weighted_pgd(config["variant"])
        return next(iter(make_loaders(config)[0]))["noisy_xyz"]
    reference = first_batch("gated", 1)
    assert torch.equal(first_batch("geometry_only", 1), reference)
    assert not torch.equal(first_batch("gated", 2), reference)


@pytest.fixture(scope="module")
def tiny_run(gso_root, tmp_path_factory):
    config = tiny(load_config(CONFIG), gso_root, variant="gated", seed=3)
    return train(config, output_root=tmp_path_factory.mktemp("runs"), device="cpu")


def test_run_folder_has_config_log_summary_and_weights(tiny_run):
    assert {p.name for p in tiny_run.iterdir()} == {"config.yaml", "log.csv", "summary.json", "trained.pt"}
    rows = list(csv.DictReader((tiny_run / "log.csv").open()))
    assert [int(r["step"]) for r in rows] == [0, 1, 2, 3, 4]
    assert all(r["val_loss"] for r in rows if int(r["step"]) % 2 == 0)
    summary = json.loads((tiny_run / "summary.json").read_text())
    assert summary["seed"] == 3 and summary["trainable_parameters"] == count_parameters(build_colour_weighted_pgd("gated"))[1]
    assert summary["initial_val_loss"] == pytest.approx(float(rows[0]["val_loss"]), rel=1e-5)
    assert tiny_run.name == "gated_seed3"


def test_only_the_trainable_part_changes(tiny_run):
    fresh = build_colour_weighted_pgd("gated").state_dict()
    trained = load_trained(tiny_run).state_dict()
    changed = {k for k in fresh if not torch.equal(fresh[k], trained[k])}
    assert changed, "training changed nothing"
    assert all(k.startswith(trainable_prefixes()) for k in changed), sorted(changed)[:5]
