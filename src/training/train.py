import argparse
import csv
import json
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.gso_patches import GSOPatchDataset
from src.models.colour_weighted_pgd import build_colour_weighted_pgd
from src.pgd.loader import PROJECT_ROOT, add_pgd_to_path

add_pgd_to_path()
from models.InfoCD import calc_cd_like_InfoV2


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def count_parameters(model):
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def patch_loss(model, batch, device):
    noisy = batch["noisy_xyz"].to(device)
    displacement, commitment = model(noisy, batch["noisy_rgb"].to(device))
    return calc_cd_like_InfoV2(noisy + displacement.float(), batch["clean_xyz"].to(device)) + commitment


def trained_state(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    for root in model.get_trainable_roots():
        prefix = next(n for n, m in model.named_modules() if m is root)
        names |= {f"{prefix}.{n}" for n, _ in root.named_buffers()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def load_trained(run_dir, map_location="cpu"):
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = build_colour_weighted_pgd(config["variant"], PROJECT_ROOT / config["checkpoint"], map_location)
    missing, unexpected = model.load_state_dict(torch.load(run_dir / "trained.pt", map_location=map_location),
                                                strict=False)
    if unexpected:
        raise ValueError(f"unexpected keys in {run_dir}: {unexpected}")
    return model.eval()


def make_loaders(config):
    cfg_data, cfg_train = config["data"], config["training"]
    common = dict(root=PROJECT_ROOT / cfg_data["root"], xyz_noise=cfg_data["xyz_noise"],
                  rgb_noise=cfg_data["rgb_noise"], patch_size=cfg_data["patch_size"])
    train_set = GSOPatchDataset(split="train", patches_per_object=cfg_data["patches_per_object"], **common)
    val_set = GSOPatchDataset(split="val", patches_per_object=cfg_data["val_patches_per_object"], rotate=False,
                              seed=0, **common)
    seed_everything(cfg_train["seed"])
    train_loader = DataLoader(train_set, batch_size=cfg_train["batch_size"], shuffle=True,
                              num_workers=cfg_train["num_workers"], drop_last=True,
                              generator=torch.Generator().manual_seed(cfg_train["seed"]))
    val_loader = DataLoader(val_set, batch_size=cfg_train["batch_size"], num_workers=cfg_train["num_workers"])
    return train_loader, val_loader


def train_step(model, batch, optimiser, scaler, use_amp, device):
    with torch.autocast(device.type, enabled=use_amp):
        loss = patch_loss(model, batch, device)
    optimiser.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimiser)
    scaler.update()
    return loss.item()


def train(config, output_root=None, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg_train = config["training"]
    seed_everything(cfg_train["seed"])

    model = build_colour_weighted_pgd(config["variant"], PROJECT_ROOT / config["checkpoint"]).to(device)
    train_loader, val_loader = make_loaders(config)

    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg_train["lr"])
    use_amp = cfg_train["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    name = config.get("run_name") or f"{config['variant']}_seed{cfg_train['seed']}"
    run_dir = Path(output_root or PROJECT_ROOT / config["output_root"]) / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    def validate():
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
            losses = [patch_loss(model, batch, device).item() for batch in val_loader]
        model.train()
        return sum(losses) / len(losses)

    best, start = float("inf"), time.time()
    batches = itertools.chain.from_iterable(itertools.repeat(train_loader))
    model.train()
    with (run_dir / "log.csv").open("w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["step", "train_loss", "val_loss", "seconds"])
        initial = validate()
        log.writerow([0, "", f"{initial:.6f}", "0.0"])
        for step, batch in zip(range(1, cfg_train["steps"] + 1), batches):
            loss, val_loss = train_step(model, batch, optimiser, scaler, use_amp, device), None
            if step % cfg_train["val_every"] == 0 or step == cfg_train["steps"]:
                val_loss = validate()
                if val_loss < best:
                    best = val_loss
                    torch.save(trained_state(model), run_dir / "trained.pt")
            log.writerow([step, f"{loss:.6f}", "" if val_loss is None else f"{val_loss:.6f}",
                          f"{time.time() - start:.1f}"])
            log_file.flush()
    total, trainable = count_parameters(model)
    (run_dir / "summary.json").write_text(json.dumps({
        "variant": config["variant"], "seed": cfg_train["seed"],
        "initial_val_loss": initial, "best_val_loss": best, "steps": cfg_train["steps"], "seconds": round(time.time() - start, 1),
        "parameters": total, "trainable_parameters": trainable}, indent=2))
    return run_dir


def apply_overrides(config, variant=None, seed=None, lr=None, steps=None, run_name=None):
    if variant is not None:
        config["variant"] = variant
    for key, value in (("seed", seed), ("lr", lr), ("steps", steps)):
        if value is not None:
            config["training"][key] = value
    if run_name is not None:
        config["run_name"] = run_name
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None, help="overrides output_root of the config")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.variant, args.seed, args.lr, args.steps, args.run_name)
    print("run folder:", train(config, args.output_root, args.device))


if __name__ == "__main__":
    main()
