"""Train matched geometry, RGB and gated RGB residual heads on frozen PGD."""
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

MODES = ("geometry", "plain_rgb", "gated_rgb")
XYZ_LEVELS = (0.01, 0.02, 0.025)
RGB_LEVELS = (0.0, 0.05)
PATCH_SIZE = 1000
PATCH_OVERSAMPLING = 6
NEIGHBORS = 16


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))


class ResidualHead(nn.Module):
    """Same trainable layers in all modes. Only the inputs and gate pooling differ."""

    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown model: {mode}")
        self.mode = mode
        self.geometry = mlp(9, 3)
        self.fusion = mlp(15, 4)
        for head in (self.geometry, self.fusion):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, base, noisy, indices, rgb):
        relative = base[indices] - base[:, None]
        local_geometry = torch.cat(
            [relative.mean(dim=1), relative.abs().mean(dim=1)], dim=-1
        )
        geometry = torch.cat([base - noisy, local_geometry], dim=-1)
        if self.mode == "geometry":
            context = local_geometry
        else:
            relative_rgb = rgb[indices] - rgb[:, None]
            context = torch.cat(
                [relative_rgb.mean(dim=1), relative_rgb.abs().mean(dim=1)], dim=-1
            )

        mixed = self.fusion(torch.cat([geometry, context], dim=-1))
        reference = self.fusion(torch.cat([geometry, torch.zeros_like(context)], dim=-1))
        # Isolate the change caused by context. Uniform RGB gives zero colour correction.
        contextual_delta = mixed[:, :3] - reference[:, :3]
        logits = mixed[:, 3:]
        if self.mode != "gated_rgb":
            logits = logits.mean(dim=0, keepdim=True).expand_as(logits)
        gain = logits.sigmoid()
        return base + self.geometry(geometry) + gain * contextual_delta, gain


@dataclass
class Sample:
    oid: str
    split: str
    group: str
    noise: float
    xyz: np.ndarray
    clean: np.ndarray
    colors: dict
    metric_scale: float


def load_cloud(path):
    with np.load(path, allow_pickle=False) as arrays:
        xyz, rgb = (arrays[key].astype(np.float32) for key in ("xyz", "rgb"))
    if (xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[1] != 3 or
            not np.isfinite(xyz).all() or not np.isfinite(rgb).all() or
            np.any(rgb < 0) or np.any(rgb > 1)):
        raise ValueError(f"Invalid XYZRGB arrays in {path}")
    return xyz, rgb


def load_samples(root, smoke):
    assignments = json.loads((root / "selection.json").read_text())
    split_ids = {split: sorted(oid for oid, info in assignments.items() if info["split"] == split)
                 for split in ("train", "val", "test")}
    if set().union(*map(set, split_ids.values())) != set(assignments):
        raise ValueError("Unexpected split label in selection.json")
    samples = {}
    for split, names in split_ids.items():
        minimum = 2 if split == "train" else 1
        if len(names) < minimum:
            raise ValueError(f"{split} requires at least {minimum} distinct objects")
        samples[split] = []
        for oid in names[:minimum] if smoke else names:
            if Path(oid).name != oid or oid in {".", ".."}:
                raise ValueError(f"Invalid object ID: {oid}")
            folder = root / split / oid
            clean, _ = load_cloud(folder / "clean.npz")
            clean_center = (clean.max(0) + clean.min(0)) / 2
            clean_radius = np.linalg.norm(clean - clean_center, axis=1).max()
            if len(clean) < PATCH_SIZE or clean_radius <= 0:
                raise ValueError(f"Invalid clean point cloud for {oid}")
            for noise in (0.02,) if smoke else XYZ_LEVELS:
                xyz, original = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.npz")
                other_xyz, jitter = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.05.npz")
                if xyz.shape != clean.shape or not np.array_equal(xyz, other_xyz):
                    raise ValueError(f"Point count or paired geometry mismatch for {oid} at {noise}")
                center = (xyz.max(0) + xyz.min(0)) / 2
                radius = np.linalg.norm(xyz - center, axis=1).max()
                if radius <= 0:
                    raise ValueError(f"Degenerate input cloud for {oid}")
                samples[split].append(Sample(
                    oid, split, assignments[oid]["group"], noise,
                    (xyz - center) / radius, (clean - center) / radius,
                    {"original": original, "jitter": jitter}, float(radius / clean_radius),
                ))
    return samples


def pgd_patches(model, points, knn, fps, iterations):
    """Stitch predictions at their original point indices so each RGB stays aligned."""
    center_count = max(1, PATCH_OVERSAMPLING * len(points) // PATCH_SIZE)
    best = points.new_full((len(points),), float("inf"))
    for _ in range(iterations):
        centers, _ = fps(points[None], center_count)
        result = torch.empty_like(points)
        best.fill_(float("inf"))

        def apply_patches(patch_centers):
            distances, indices, neighborhoods = knn(
                patch_centers[None], points[None], K=PATCH_SIZE, return_nn=True,
            )
            for center, dists, ids, patch in zip(
                patch_centers, distances[0], indices[0], neighborhoods[0]
            ):
                relative = dists / dists[-1].clamp_min(1e-12)
                predicted = model.denoise_langevin_dynamics((patch - center)[None])[0] + center
                take = relative < best[ids]
                result[ids[take]] = predicted[take]
                best[ids[take]] = relative[take]

        apply_patches(centers[0])
        # FPS patches can miss a few rows. Process centred patches for any such rows.
        for idx in torch.where(~best.isfinite())[0].tolist():
            if not torch.isfinite(best[idx]):
                apply_patches(points[idx:idx + 1])
        if not best.isfinite().all() or not result.isfinite().all():
            raise RuntimeError("PGD stitching failed to reconstruct every input row")
        points = result
    return points


def digest_file(path):
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def precompute(samples, pgd_root, checkpoint, iterations, cache, seed, device, knn):
    sys.path.insert(0, str(pgd_root))
    from models.pgd import PGDModel
    from models.utils import farthest_point_sampling

    revision = subprocess.check_output(["git", "-C", str(pgd_root), "rev-parse", "HEAD"], text=True).strip()
    pgd_config = dict(revision=revision, weights=digest_file(checkpoint), iterations=iterations,
                      patch_size=PATCH_SIZE, oversampling=PATCH_OVERSAMPLING, seed=seed,
                      stitch="indexed_v1", device=torch.cuda.get_device_name(device),
                      torch=torch.__version__)
    model = None
    cache.mkdir(parents=True, exist_ok=True)
    prepared = {}
    started = time.monotonic()
    all_samples = [sample for split in samples.values() for sample in split]
    for number, sample in enumerate(all_samples, 1):
        key = hashlib.sha256(json.dumps(pgd_config, sort_keys=True).encode() + sample.xyz.tobytes())
        filename = cache / f"{key.hexdigest()}.npy"
        if filename.is_file():
            baseline = np.load(filename, allow_pickle=False)
        else:
            if model is None:
                model = PGDModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
                model = model.to(device).requires_grad_(False).eval()
            # Stable result for each cache entry independently of processing order.
            torch.manual_seed(seed)
            with torch.no_grad():
                baseline = pgd_patches(model, torch.from_numpy(sample.xyz).to(device), knn,
                                       farthest_point_sampling, iterations).cpu().numpy()
            partial = filename.with_suffix(".part")
            with partial.open("wb") as stream:
                np.save(stream, baseline)
            partial.replace(filename)
        if baseline.shape != sample.xyz.shape or not np.isfinite(baseline).all():
            raise ValueError(f"Invalid frozen prediction for {sample.oid}")
        xyz = torch.from_numpy(sample.xyz.copy()).to(device)
        with torch.no_grad():
            ids = knn(xyz[None], xyz[None], K=NEIGHBORS + 1).idx[0]
            self_index = torch.arange(len(xyz), device=device)[:, None]
            order = (ids == self_index).to(torch.int32).argsort(dim=1, stable=True)
            neighbors = ids.gather(1, order[:, :NEIGHBORS]).cpu()
        prepared[(sample.oid, sample.noise)] = (torch.from_numpy(baseline), neighbors)
        print(f"PGD {number}/{len(all_samples)} · {sample.split}/{sample.oid} · {sample.noise:g}", flush=True)
    print(f"PGD preparation took {time.monotonic() - started:.1f} s", flush=True)
    return prepared, pgd_config


def batch(sample, prepared, condition, device):
    base, neighbors = prepared[(sample.oid, sample.noise)]
    return (base.to(device), torch.from_numpy(sample.xyz).to(device),
            neighbors.to(device), torch.from_numpy(sample.colors[condition]).to(device),
            torch.from_numpy(sample.clean).to(device))


def chamfer(prediction, target, knn):
    a = knn(prediction[None], target[None], K=1).dists
    b = knn(target[None], prediction[None], K=1).dists
    return a.mean() + b.mean()


def fit(mode, samples, prepared, epochs, seed, device, knn, directory):
    torch.manual_seed(seed)
    model = ResidualHead(mode).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    color_conditions = ("original", "jitter")
    best_score, best_state = math.inf, None
    history = []
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        train_losses = []
        for index in rng.permutation(len(samples["train"])):
            sample = samples["train"][index]
            condition = color_conditions[(epoch + index) % len(color_conditions)]
            base, xyz, neighbors, rgb, target = batch(sample, prepared, condition, device)
            optim.zero_grad(set_to_none=True)
            predicted, _ = model(base, xyz, neighbors, rgb)
            loss = chamfer(predicted, target, knn)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss for {mode} on {sample.oid}")
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        model.eval()
        val_scores = []
        with torch.no_grad():
            for sample in samples["val"]:
                for condition in color_conditions:
                    base, xyz, neighbors, rgb, target = batch(sample, prepared, condition, device)
                    predicted, _ = model(base, xyz, neighbors, rgb)
                    val_scores.append(chamfer(predicted, target, knn).item() * sample.metric_scale**2)
        score = float(np.mean(val_scores))
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite validation score for {mode}")
        history.append(dict(epoch=epoch + 1, train_cd=float(np.mean(train_losses)), validation_cd=score))
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        seconds = time.monotonic() - started
        print(f"{mode} · epoch {epoch + 1}/{epochs} · validation CD {score:.6g} · {seconds:.1f} s", flush=True)
    torch.save(best_state, directory / f"{mode}.pt")
    model.load_state_dict(best_state)
    return model.eval(), history


@torch.no_grad()
def evaluate(samples, heads, prepared, device, knn):
    rows = []
    for sample in samples:
        for condition in ("original", "jitter"):
            base, xyz, neighbors, rgb, target = batch(sample, prepared, condition, device)
            predictions = {"noisy": (xyz, None), "frozen_pgd": (base, None)}
            for mode, head in heads.items():
                predictions[mode] = head(base, xyz, neighbors, rgb)
            for method, (predicted, gate) in predictions.items():
                score = chamfer(predicted, target, knn).item() * sample.metric_scale**2
                if not math.isfinite(score):
                    raise RuntimeError(f"Non-finite test score for {method} on {sample.oid}")
                rows.append(dict(object_id=sample.oid, group=sample.group, xyz_noise=sample.noise,
                                 rgb=condition, method=method, cd_squared=score,
                                 gate_mean=gate.mean().item() if method == "gated_rgb" else ""))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    torch.manual_seed(42)
    xyz = torch.randn(32, 3)
    base = xyz * 0.99
    rgb = torch.rand_like(xyz)
    distances = torch.cdist(xyz, xyz)
    distances.fill_diagonal_(float("inf"))
    indices = distances.argsort(dim=1)[:, :NEIGHBORS]
    counts = []
    for mode in MODES:
        head = ResidualHead(mode)
        counts.append(sum(p.numel() for p in head.parameters()))
        prediction, _ = head(base, xyz, indices, rgb)
        assert torch.equal(prediction, base)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        target = xyz * 0.97 + rgb * 0.05
        for _ in range(4):
            optimizer.zero_grad()
            prediction, _ = head(base, xyz, indices, rgb)
            (prediction - target).square().mean().backward()
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
            optimizer.step()
        prediction, gate = head(base, xyz, indices, rgb)
        other, _ = head(base, xyz, indices, 1 - rgb)
        assert prediction.shape == xyz.shape and torch.isfinite(prediction).all()
        assert (0 < gate).all() and (gate < 1).all()
        if mode == "geometry":
            assert torch.equal(prediction, other)
        else:
            assert not torch.equal(prediction, other)
            constant_a, _ = head(base, xyz, indices, torch.ones_like(rgb))
            constant_b, _ = head(base, xyz, indices, torch.zeros_like(rgb))
            assert torch.equal(constant_a, constant_b)
        if mode == "gated_rgb":
            assert gate.std() > 0
        else:
            assert torch.equal(gate[0].expand_as(gate), gate)
    assert len(set(counts)) == 1
    print(f"PASS · all three heads trained · {counts[0]} trainable parameters each")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--pgd", type=Path, default=Path("/content/PGD"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.data is None or args.out is None or args.epochs < 1 or args.iterations < 1:
        parser.error("A training run requires --data, --out and positive training settings")
    if not torch.cuda.is_available():
        parser.error("The published PGD operators require a CUDA runtime")
    from pytorch3d.ops import knn_points
    device = torch.device("cuda")
    checkpoint = args.pgd / "pretrained" / "PGD.ckpt"
    samples = load_samples(args.data, args.smoke)
    args.out.mkdir(parents=True, exist_ok=False)
    cache = args.cache or args.out / "cache"
    prepared, config = precompute(samples, args.pgd.resolve(), checkpoint, args.iterations, cache,
                                  args.seed, device, knn_points)
    epochs = min(args.epochs, 2) if args.smoke else args.epochs
    heads, history = {}, {}
    for mode in MODES:
        heads[mode], history[mode] = fit(
            mode, samples, prepared, epochs, args.seed, device, knn_points, args.out,
        )
    rows = evaluate(samples["test"], heads, prepared, device, knn_points)
    write_csv(args.out / "per_object.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        for group in ("all", row["group"]):
            key = (group, row["xyz_noise"], row["rgb"], row["method"])
            groups[key].append(row["cd_squared"])
    write_csv(args.out / "summary.csv", [
        dict(group=group, xyz_noise=noise, rgb=rgb, method=method, n_objects=len(values),
             mean_cd_squared=float(np.mean(values)))
        for (group, noise, rgb, method), values in sorted(groups.items())
    ])
    metadata = dict(pgd=config, epochs=epochs, seed=args.seed, smoke=args.smoke,
                    selection_sha256=digest_file(args.data / "selection.json"), history=history,
                    objects={split: sorted({s.oid for s in subset}) for split, subset in samples.items()})
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"COMPLETE · {'smoke' if args.smoke else 'full'} · {len(rows)} test rows · {args.out}")


if __name__ == "__main__":
    main()
