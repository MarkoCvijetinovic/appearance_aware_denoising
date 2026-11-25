"""Fit a mesh-supervised, multiscale residual on frozen PGD point clouds."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import torch
from scipy.spatial import cKDTree
from torch import nn

from evaluate_surface import find_shards, normalize, original_meshes, point_to_surface
from train_rgb_pgd import XYZ_LEVELS, digest_file, load_cloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task5"))
from evaluate_gso import geometry_regions, intervention_colors, measured_output, read_mesh

SCALES = (8, 16, 32)
MAX_STEP = 0.035
SEED = 20261124


@dataclass
class Case:
    oid: str
    noise: float
    scale: float
    base: np.ndarray
    edges: torch.Tensor
    point: torch.Tensor
    rgb_edges: dict
    projection: torch.Tensor
    center: np.ndarray


def local_geometry(base, noisy):
    _, ids = cKDTree(base).query(base, k=SCALES[-1] + 1)
    indices = np.arange(len(base))
    # Duplicated coordinates can put the query index anywhere in the result.
    ids = ids[np.arange(len(base))[:, None],
              np.argsort(ids == indices[:, None], axis=1, kind="stable")][:, :SCALES[-1]]
    relative = base[ids] - base[:, None]
    centered = relative - relative.mean(axis=1, keepdims=True)
    eigenvalues, frame = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered / len(ids[0]))
    # Coordinate signs come from neighbours, so rotating an object preserves the features.
    projected = relative @ frame
    sign = np.sign(np.take_along_axis(projected, np.abs(projected).argmax(axis=1)[:, None], axis=1))
    frame = frame * np.where(sign == 0, 1, sign)
    noisy_relative = (noisy[ids] - noisy[:, None]) @ frame
    edges = np.concatenate((relative @ frame, noisy_relative,
                            np.linalg.norm(relative, axis=-1, keepdims=True)), axis=-1)
    offset = np.einsum("ni,nij->nj", noisy - base, frame)
    point = np.concatenate((offset, np.sqrt(np.maximum(eigenvalues, 0))), axis=1)
    half = lambda array: torch.from_numpy(array.astype(np.float16))
    return ids, frame, half(edges * 20), half(point * 20)


def block(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.SiLU(),
                         nn.Linear(width, outputs), nn.SiLU())


def pool(encoded):
    return torch.cat([stat
                      for k in SCALES
                      for stat in (encoded[:, :k].mean(1), encoded[:, :k].amax(1))], dim=1)


class ShapeResidual(nn.Module):
    def __init__(self, use_rgb, learned_gate=True):
        super().__init__()
        self.use_rgb = use_rgb
        self.learned_gate = learned_gate
        self.geometry = block(7, 64, 48)
        self.context = block(6 * 48 + 6, 192, 128)
        self.colour = block(7 + 3, 64, 48) if use_rgb else None
        self.gate = block(6 * 48 + 128, 192, 96) if use_rgb else None
        self.correction = nn.Linear(128, 3)
        self.colour_correction = nn.Linear(96, 4) if use_rgb else None
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        if self.colour_correction is not None:
            nn.init.zeros_(self.colour_correction.weight)
            nn.init.zeros_(self.colour_correction.bias)

    def forward(self, edges, point, rgb=None):
        geometry = self.context(torch.cat((pool(self.geometry(edges)), point), dim=-1))
        correction = self.correction(geometry)
        if self.use_rgb and rgb is not None:
            branch = self.colour(torch.cat((edges, rgb), dim=-1))
            mix = self.colour_correction(self.gate(torch.cat((pool(branch), geometry), dim=-1)))
            contrast = (2 * rgb.abs().amax(dim=(1, 2), keepdim=False)).clamp(max=1)
            gain = mix[:, 3:].sigmoid() if self.learned_gate else 0.5
            correction = correction + mix[:, :3] * gain * contrast[:, None]
        return MAX_STEP * correction.tanh()


def load_group(oid, split, args, mesh_dir, prefix, conditions, need_targets):
    data = args.extra if oid in args.extra_ids else args.data
    folder = data / split / oid
    clean, colour = load_cloud(folder / "clean.npz")
    center, radius = normalize(clean)
    norm_clean = (clean.astype(np.float64) - center) / radius
    mesh, topology = read_mesh(mesh_dir / f"{oid}.glb", center, radius)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    clean_rms = np.sqrt(point_to_surface(norm_clean, vertices, faces).mean())
    if not np.isfinite(clean_rms) or clean_rms > 1e-5:
        raise ValueError(f"Mesh alignment failed for {oid}: {clean_rms:g}")
    cases = []
    for noise in XYZ_LEVELS:
        noisy, original = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.npz")
        other, jitter = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.05.npz")
        if noisy.shape != clean.shape or not np.array_equal(noisy, other):
            raise ValueError(f"Unpaired cloud for {oid}, {noise}")
        work_center, work_radius = normalize(noisy)
        xyz = (noisy - work_center) / work_radius
        key = hashlib.sha256(prefix + xyz.tobytes()).hexdigest()
        base = np.load(args.pgd_cache / f"{key}.npy", allow_pickle=False)
        if base.shape != xyz.shape or not np.isfinite(base).all():
            raise ValueError(f"Invalid PGD output for {oid}, {noise}")
        ids, frame, edges, point = local_geometry(base, xyz)
        colours = intervention_colors(original, jitter, oid)
        rgb_edges = {name: torch.from_numpy((colours[name][ids] - colours[name][:, None]).astype(np.float16))
                     for name in conditions if name != "constant"}
        projection = None
        if need_targets:
            pgd = (base.astype(np.float64) * work_radius + work_center - center) / radius
            _, fid, bc = pcu.closest_points_on_mesh(pgd, vertices, faces)
            if np.any(fid < 0) or not np.isfinite(bc).all():
                raise ValueError(f"Invalid mesh projection for {oid}, {noise}")
            closest = pcu.interpolate_barycentric_coords(faces, fid, bc, vertices)
            offset = (closest - pgd) * radius / work_radius
            projection = torch.from_numpy(np.einsum("ni,nij->nj", offset, frame).astype(np.float32))
        # Frame is needed only when reconstructing an output cloud.
        if need_targets:
            frame = None
        cases.append((Case(oid, noise, float(work_radius / radius), base, edges, point, rgb_edges,
                           projection, (work_center.astype(np.float64) - center) / radius), frame))
    return cases, norm_clean, colour, vertices, faces, topology


def batch(case, indices, condition, device):
    rgb = case.rgb_edges.get(condition)
    rgb = rgb[indices].to(device, dtype=torch.float32) if rgb is not None else None
    return case.edges[indices].to(device, dtype=torch.float32), case.point[indices].to(
        device, dtype=torch.float32), rgb


@torch.inference_mode()
def predict(model, case, frame, condition, device):
    n = len(case.base)
    local = torch.cat([model(*batch(case, slice(i, i + 768), condition, device)).cpu()
                       for i in range(0, n, 768)]).numpy()
    return np.einsum("nij,nj->ni", frame, local)


def physical(case, correction):
    return (case.base.astype(np.float64) + correction) * case.scale + case.center


def p2s_scales(model, validation, device):
    model.eval()
    scores = {scale: [] for scale in (0.0, 0.5, 1.0)}
    for cases, _, _, vertices, faces, _ in validation:
        projections = []
        for case, frame in cases:
            conds = ("original", "gaussian_0.05") if model.use_rgb else (None,)
            deltas = [predict(model, case, frame, condition, device) for condition in conds]
            for scale in scores:
                projections.extend(physical(case, scale * delta) for delta in deltas)
        n = len(cases[0][0].base)
        squared = point_to_surface(np.concatenate(projections), vertices, faces).reshape(
            len(cases), len(scores), len(conds), n
        )
        for index, scale in enumerate(scores):
            scores[scale].extend(squared[:, index].mean(axis=(1, 2)))
    return {scale: float(np.mean(values)) for scale, values in scores.items()}


def fit(mode, training, validation, args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ShapeResidual(mode != "geometry", learned_gate=mode != "plain_rgb").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00005)
    weights = args.out / f"{mode}.pt"
    state = args.out / f"{mode}_state.pt"
    records = []
    start, best, stale = 1, float("inf"), 0
    if state.is_file():
        saved = torch.load(state, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["random"]
        start, best, stale = saved["epoch"] + 1, saved["best"], saved["stale"]
        records = saved["records"]
    per_epoch = args.cases_per_epoch
    if len(training) % per_epoch or per_epoch % 3:
        raise ValueError("The training schedule requires complete batches and equally sized case rotations")
    rotation = len(training) // per_epoch
    for epoch in range(start, args.epochs + 1):
        if stale >= 4 and not args.extra:
            break
        tick = time.monotonic()
        model.train()
        losses = []
        if rotation == 1:
            order = rng.permutation(len(training))
        else:
            cycle, part = divmod(epoch - 1, rotation)
            cycle_rng = np.random.default_rng(args.seed + cycle)
            order = cycle_rng.permutation(len(training))[part * per_epoch:(part + 1) * per_epoch]
        for group in np.array_split(order, per_epoch // 3):
            samples, targets = [], []
            for index in group:
                case = training[index]
                rows = rng.choice(len(case.base), size=args.points, replace=False)
                colour = rng.choice(("original", "gaussian_0.05", "gaussian_0.15", "shuffled", "constant"),
                                    p=(0.48, 0.25, 0.08, 0.10, 0.09)) if model.use_rgb else None
                edges, point, rgb = batch(case, rows, colour, device)
                samples.append((edges, point, rgb if rgb is not None else torch.zeros_like(edges[:, :, :3])))
                targets.append(case.projection[rows].to(device) * 100)
            prediction = model(*(torch.cat(values, dim=0) for values in zip(*samples))) * 100
            target = torch.cat(targets)
            loss = nn.functional.smooth_l1_loss(prediction, target, beta=0.30)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        loss_mean = torch.stack(losses).mean().item()
        record = dict(epoch=epoch, loss=loss_mean, updates=len(losses), train_seconds=time.monotonic() - tick)
        print(f"{mode} epoch {epoch}/{args.epochs} · loss {loss_mean:.5f} · {record['train_seconds']:.1f}s",
              flush=True)
        if epoch == 1 or epoch % args.validate_every == 0 or epoch == args.epochs:
            scores = p2s_scales(model, validation, device)
            # Keep the best learned candidate even when validation favours raw PGD.
            scale = min((0.5, 1.0), key=scores.get)
            record.update(validation={str(k): v for k, v in scores.items()}, validation_seconds=time.monotonic()-tick-record["train_seconds"])
            if scores[scale] < best:
                save_atomic(weights, dict(model=model.state_dict(), scale=scale, epoch=epoch,
                                          validation=scores[scale], pgd_validation=scores[0.0]))
                best, stale = scores[scale], 0
            else:
                stale += 1
            print(f"{mode} validation · P2S² × 1e6 "
                  + " / ".join(f"{s:g}: {v*1e6:.3f}" for s, v in scores.items()), flush=True)
        records.append(record)
        save_atomic(state, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                               scheduler=scheduler.state_dict(), epoch=epoch, random=rng.bit_generator.state,
                               best=best, stale=stale, records=records))
    saved = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(saved.pop("model"))
    return model.eval(), saved, records


def save_atomic(path, value):
    partial = path.with_suffix(".part")
    torch.save(value, partial)
    partial.replace(path)


def evaluate(groups, fits, selection, args, mesh_dir, prefix, device):
    thresholds = json.loads((args.run / "surface/texture_regions/mask_metadata.json").read_text())
    rows = {seed: [] for seed in fits}
    conditions = ("original", "gaussian_0.05", "gaussian_0.15", "shuffled", "constant")
    for oid in groups["test"]:
        cases, clean, rgb, vertices, faces, mesh = load_group(
            oid, "test", args, mesh_dir, prefix, conditions, False
        )
        clean_tree = cKDTree(clean)
        regions = geometry_regions(clean, rgb, mesh, thresholds["geometry_p50"], thresholds["colour_p90"])[:6]
        for case, frame in cases:
            baseline = physical(case, 0.0)
            pgd_scores = measured_output(
                baseline, point_to_surface(baseline, vertices, faces), *regions, clean, clean_tree
            )
            shared = dict(object_id=oid, group=selection[oid]["group"], xyz_noise=case.noise)
            for seed, (_, models) in fits.items():
                rows[seed].append(dict(**shared, method="frozen_pgd", colour="constant", **pgd_scores))
                cases_for_seed = [
                    (mode, cond, physical(case, chosen["scale"] * predict(model, case, frame, cond, device)))
                    for mode, (model, chosen, _) in models.items()
                    for cond in (conditions if model.use_rgb else ("constant",))
                ]
                points = np.concatenate([pred for _, _, pred in cases_for_seed])
                squared = point_to_surface(points, vertices, faces).reshape(len(cases_for_seed), -1)
                for (method, condition, pred), distances in zip(cases_for_seed, squared, strict=True):
                    scores = measured_output(pred, distances, *regions, clean, clean_tree)
                    rows[seed].append(dict(**shared, method=method, colour=condition, **scores))
        print(f"Test · {oid}", flush=True)
    for seed, (out, _) in fits.items():
        result = pd.DataFrame(rows[seed])
        result.to_csv(out / "per_object.csv", index=False)
        columns = list(result.select_dtypes(include="number").columns.drop("xyz_noise"))
        summary = result.groupby(["method", "colour"])[columns].mean()
        summary.to_csv(out / "summary.csv")
        result.groupby(["xyz_noise", "method", "colour"])[columns].mean().to_csv(out / "by_noise.csv")
        print(f"Test means · seed {seed}")
        print(summary[["cd_squared", "p2surface_squared", "sharp_p2surface_squared",
                       "texture_p2surface_squared"]].mul(1e6).to_string())


def main(args):
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    args.mesh_cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime for the full training experiment")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    meta = json.loads((args.run / "run.json").read_text())
    if meta["smoke"] or digest_file(args.data / "selection.json") != meta["selection_sha256"]:
        raise ValueError("Task 4 PGD cache does not match this dataset split")
    selection = json.loads((args.data / "selection.json").read_text())
    groups = meta["objects"]
    ids = set(selection)
    if any(sorted(oid for oid, row in selection.items() if row["split"] == split) != sorted(groups[split])
           for split in ("train", "val", "test")):
        raise ValueError("Inconsistent object splits")
    args.extra_ids = set()
    if args.extra:
        extra = json.loads((args.extra / "selection.json").read_text())
        expansion = json.loads((args.extra / "expansion.json").read_text())
        if (ids & extra.keys() or expansion["base_selection_sha256"] != meta["selection_sha256"]
                or expansion["additional_selection_sha256"] != digest_file(args.extra / "selection.json")):
            raise ValueError("The expansion does not match the original split")
        fresh = {split: sorted(oid for oid, row in extra.items() if row["split"] == split)
                 for split in ("train", "val", "test")}
        if tuple(map(len, fresh.values())) != (84, 24, 42) or args.train_objects not in (42, 126):
            raise ValueError("Expected 84 extra training, 24 fresh validation and 42 fresh test objects")
        groups = dict(train=groups["train"] + fresh["train"][:args.train_objects - 42],
                      val=fresh["val"], test=fresh["test"])
        selection.update(extra)
        args.extra_ids = set(extra)
        ids = set().union(*groups.values())
    args.cases_per_epoch = 42 * len(XYZ_LEVELS) if args.extra else len(groups["train"]) * len(XYZ_LEVELS)
    manifest = json.loads((args.data / "manifest.json").read_text())
    mesh_dir = args.mesh_cache / manifest["revision"]
    protocol = dict(epochs=args.epochs, points=args.points, validate_every=args.validate_every,
                    seed=SEED, source_run=args.run.name, selection_sha256=meta["selection_sha256"],
                    code_sha256=digest_file(Path(__file__)))
    if args.extra:
        protocol.update(objects=groups, expansion_sha256=digest_file(args.extra / "selection.json"),
                        cases_per_epoch=args.cases_per_epoch, updates_per_epoch=args.cases_per_epoch // 3)
    root = args.out
    seeds = (SEED, *args.repeat_seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each training seed must be unique")
    train_plain = args.plain_rgb or (args.extra and args.train_objects == 126)
    modes = ("geometry", "rgb", "plain_rgb") if train_plain else ("geometry", "rgb")
    runs = {}
    for seed in seeds:
        out = root if seed == SEED else root / "repeats" / str(seed)
        out.mkdir(parents=True, exist_ok=True)
        details = dict(protocol, seed=seed, modes=list(modes))
        protocol_path = out / "protocol.json"
        if protocol_path.is_file() and json.loads(protocol_path.read_text()) != details:
            raise ValueError(f"Choose a new output folder for a changed experiment: {out}")
        protocol_path.write_text(json.dumps(details, indent=2))
        runs[seed] = out
    original_meshes(manifest["source"], manifest["revision"], find_shards(ids, args.candidates), mesh_dir)
    prefix = json.dumps(meta["pgd"], sort_keys=True).encode()
    training = []
    for i, oid in enumerate(groups["train"], start=1):
        cases, *_ = load_group(oid, "train", args, mesh_dir, prefix,
                               ("original", "gaussian_0.05", "gaussian_0.15", "shuffled"), True)
        training.extend(case for case, _ in cases)
        if i % 10 == 0 or i == len(groups["train"]):
            print(f"Training projections · {i}/{len(groups['train'])}", flush=True)
    validation = [load_group(oid, "val", args, mesh_dir, prefix, ("original", "gaussian_0.05"), False)
                  for oid in groups["val"]]
    print(f"Preparation · {time.monotonic()-started:.1f}s · {torch.cuda.get_device_name()}", flush=True)
    fits = {}
    for seed, out in runs.items():
        args.seed, args.out = seed, out
        print(f"Training · {len(groups['train'])} objects · seed {seed}", flush=True)
        fits[seed] = out, {mode: fit(mode, training, validation, args, device) for mode in modes}
    evaluate(groups, fits, selection, args, mesh_dir, prefix, device)
    seconds = time.monotonic() - started
    for seed, (out, models) in fits.items():
        result = dict(seconds=seconds, device=torch.cuda.get_device_name(),
                      models={mode: dict(checkpoint=chosen, history=history)
                              for mode, (_, chosen, history) in models.items()})
        (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"COMPLETE · {seconds/60:.1f} minutes · {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "pgd-cache", "candidates", "mesh-cache", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--extra", type=Path, help="Supplemental GSO split from expand_gso.py")
    parser.add_argument("--train-objects", type=int, choices=(42, 126), default=126)
    parser.add_argument("--repeat-seed", type=int, action="append", default=[])
    parser.add_argument("--plain-rgb", action="store_true", help="Train an identical RGB head with a fixed gate")
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--points", type=int, default=640)
    parser.add_argument("--validate-every", type=int, default=6)
    main(parser.parse_args())
