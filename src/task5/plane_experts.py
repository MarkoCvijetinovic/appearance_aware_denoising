"""Compare RGB-guided neighbourhood selection with a geometry-only plane expert."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit

from evaluate_gso import (COLOURS, geometry_regions, intervention_colors, load_cloud, measured_output,
                          normalize, original_meshes, find_shards, point_to_surface, read_mesh, seed_for,
                          XYZ_LEVELS, digest_file)

K = 32
RGB_BANDWIDTH = 0.20
TRAIN_POINTS = 320
METHODS = ("frozen_pgd", "geometry_plane", "plain_rgb_plane", "gated_rgb_plane")


def neighbours(points, k=K):
    distances, indices = cKDTree(points).query(points, k=k + 1)
    keep = indices != np.arange(len(points))[:, None]
    column = np.argsort(~keep, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(distances, column, axis=1), np.take_along_axis(indices, column, axis=1)


def local_plane(points, nearby, weights):
    weights = weights / weights.sum(axis=1, keepdims=True)
    center = np.einsum("nk,nkj->nj", weights, nearby)
    relative = nearby - center[:, None]
    covariance = np.einsum("nk,nki,nkj->nij", weights, relative, relative, optimize=True)
    eigen, basis = np.linalg.eigh(covariance)
    normal = basis[:, :, 0]
    variation = np.maximum(eigen[:, 0], 0) / np.maximum(eigen.sum(axis=1), 1e-12)
    offset = np.einsum("ni,ni->n", center - points, normal)
    return np.clip(offset, -0.04, 0.04)[:, None] * normal, normal, variation, 1 / np.sum(weights**2, axis=1)


def geometric_features(pgd, noisy):
    distance, ids = neighbours(pgd)
    nearby = pgd[ids]
    bandwidth = np.maximum(np.median(distance, axis=1), 1e-6)
    weights = np.exp(-0.5 * (distance / bandwidth[:, None])**2)
    delta, normal, variation, support = local_plane(pgd, nearby, weights)
    features = np.column_stack((np.log1p(100 * variation), np.log1p(100 * np.linalg.norm(delta, axis=1)),
                                np.log1p(100 * bandwidth), np.log1p(100 * np.linalg.norm(pgd - noisy, axis=1))))
    return features, delta, normal, variation, (nearby, ids, weights, support)


def colour_features(pgd, rgb, geom, point_rgb):
    features, delta, normal, variation, (nearby, ids, weights, geometry_support) = geom
    gap = np.linalg.norm(rgb[ids] - point_rgb[:, None, :], axis=-1)
    affinity = np.exp(-0.5 * (gap / RGB_BANDWIDTH)**2)
    weighted = np.maximum(affinity, 1e-8) * weights
    color_delta, color_normal, color_variation, color_support = local_plane(pgd, nearby, weighted)
    reliability = np.minimum(color_support / 8, 1)
    color_delta = delta + reliability[:, None] * (color_delta - delta)
    extra = np.column_stack((gap.mean(axis=1), color_support / geometry_support,
                             np.log1p(100 * color_variation) - np.log1p(100 * variation),
                             1 - np.abs(np.einsum("ni,ni->n", normal, color_normal)),
                             np.log1p(100 * np.linalg.norm(color_delta - delta, axis=1))))
    return np.concatenate([features, extra], axis=1), color_delta


def projected(v, n):
    return np.einsum("ij,ij->i", v, n)


def learn(features, error, delta):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    standardized = np.clip((features - center) / scale, -8, 8)
    matrix = np.column_stack((np.ones(len(features)), standardized))
    count = len(matrix)

    def objective(params):
        gain = expit(matrix @ params)
        residual = error + gain * delta
        risk = np.mean(residual**2) * 1e6 + 0.03 * np.sum(params[1:]**2)
        gradient = (2e6 / count) * matrix.T @ (residual * delta * gain * (1 - gain))
        gradient[1:] += 0.06 * params[1:]
        return risk, gradient

    initial = np.zeros(matrix.shape[1])
    initial[0] = -1
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(-9, 9)] * len(initial), options={"maxiter": 100})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(result.message)
    return dict(center=center.tolist(), scale=scale.tolist(), params=result.x.tolist())


def gain(model, features):
    standardized = np.clip((features - model["center"]) / model["scale"], -8, 8)
    params = np.asarray(model["params"])
    return expit(params[0] + standardized @ params[1:])


def load_samples(data, run, meta, split, shape):
    prefix = json.dumps(meta["pgd"], sort_keys=True).encode()
    path = data / split / shape
    clean, clean_rgb = load_cloud(path / "clean.npz")
    clean_center, clean_radius = normalize(clean)
    norm_clean = (clean.astype(np.float64) - clean_center) / clean_radius
    for noise in XYZ_LEVELS:
        noisy, rgb = load_cloud(path / f"noise_xyz_{noise:g}_rgb_0.npz")
        other, jitter = load_cloud(path / f"noise_xyz_{noise:g}_rgb_0.05.npz")
        assert np.array_equal(noisy, other) and np.array_equal(rgb, clean_rgb)
        center, radius = normalize(noisy)
        xyz = (noisy - center) / radius
        key = hashlib.sha256(prefix + xyz.tobytes()).hexdigest()
        baseline = np.load(run.parent / "pgd_cache" / f"{key}.npy", allow_pickle=False)
        pgd = (baseline.astype(np.float64) * radius + center - clean_center) / clean_radius
        work_noisy = (noisy.astype(np.float64) - clean_center) / clean_radius
        yield noise, norm_clean, pgd, work_noisy, intervention_colors(rgb, jitter, shape)


def training_data(data, run, meta):
    samples = {key: [] for key in ("geometry", "colour", "error", "delta_geo", "delta_rgb")}
    for i, shape in enumerate(sorted(meta["objects"]["train"]), 1):
        normals = None
        for noise, clean, pgd, noisy, colors in load_samples(data, run, meta, "train", shape):
            if normals is None:
                _, clean_ids = neighbours(clean, k=24)
                _, normals, _, _ = local_plane(clean, clean[clean_ids], np.ones(clean_ids.shape))
            context = geometric_features(pgd, noisy)
            rng = np.random.default_rng(seed_for(shape, f"fit-{noise:g}"))
            keep = rng.choice(len(pgd), min(TRAIN_POINTS, len(pgd)), replace=False)
            sampled = tuple(array[keep] for array in context[:4]) + (tuple(array[keep] for array in context[4]),)
            features, geometry_delta = sampled[:2]
            normal = normals[keep]
            error = projected((pgd - clean)[keep], normal)
            geo = projected(geometry_delta, normal)
            for rgb in colors.values():
                colour, rgb_delta = colour_features(pgd[keep], rgb, sampled, rgb[keep])
                for key, value in (("geometry", features), ("colour", colour), ("error", error),
                                   ("delta_geo", geo), ("delta_rgb", projected(rgb_delta, normal))):
                    samples[key].append(value)
        if i % 7 == 0:
            print(f"Plane experts: prepared {i}/42 training objects", flush=True)
    return {key: np.concatenate(value) for key, value in samples.items()}


def fit(args, meta):
    x = training_data(args.data, args.run, meta)
    geom = learn(x["geometry"], x["error"], x["delta_geo"])
    strength = gain(geom, x["geometry"])
    residual = x["error"] + strength * x["delta_geo"]
    opt = strength * (x["delta_rgb"] - x["delta_geo"])
    global_mix = float(np.clip(-np.dot(residual, opt) / max(np.dot(opt, opt), 1e-12), 0, 1))
    gate = learn(x["colour"], residual, opt)
    return dict(geometry=geom, plain_rgb_weight=global_mix, rgb_gate=gate, neighbours=K,
                rgb_bandwidth=RGB_BANDWIDTH, train_points_per_xyz=TRAIN_POINTS,
                loss="squared clean-sample normal residual", train_ids=meta["objects"]["train"])


def measure(args, meta, model):
    manifest = json.loads((args.data / "manifest.json").read_text())
    selection = json.loads((args.data / "selection.json").read_text())
    thresholds = json.loads((args.run / "surface/texture_regions/mask_metadata.json").read_text())
    ids = set(meta["objects"]["test"])
    original_meshes(manifest["source"], manifest["revision"], find_shards(ids, args.candidates),
                    args.mesh_cache / manifest["revision"])
    rows, val_rows = [], []
    for split, ids in (("val", meta["objects"]["val"]), ("test", meta["objects"]["test"])):
        for shape in sorted(ids):
            if split == "test":
                ref, rgb = load_cloud(args.data / "test" / shape / "clean.npz")
                center_ref, radius_ref = normalize(ref)
                mesh, topology = read_mesh(args.mesh_cache / manifest["revision"] / f"{shape}.glb", center_ref, radius_ref)
                vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
                faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
                sharp, texture, left, right, normal, clean_step, _ = geometry_regions(
                    (ref.astype(np.float64) - center_ref) / radius_ref, rgb, topology,
                    thresholds["geometry_p50"], thresholds["colour_p90"])
            for noise, clean, pgd, noisy, colors in load_samples(args.data, args.run, meta, split, shape):
                clean_tree = cKDTree(clean)
                context = geometric_features(pgd, noisy)
                geo_features, geo_delta = context[:2]
                strength = gain(model["geometry"], geo_features)[:, None]
                geometry = pgd + strength * geo_delta
                for name, rgb in colors.items():
                    color_features, color_delta = colour_features(pgd, rgb, context, rgb)
                    difference = strength * (color_delta - geo_delta)
                    color_gate = gain(model["rgb_gate"], color_features)
                    pred = [pgd, geometry, geometry + model["plain_rgb_weight"] * difference,
                            geometry + color_gate[:, None] * difference]
                    if split == "test":
                        sq = point_to_surface(np.concatenate(pred), vertices, faces).reshape(len(pred), -1)
                    for j, (method, value) in enumerate(zip(METHODS, pred, strict=True)):
                        row = dict(object_id=shape, group=selection[shape]["group"], xyz_noise=noise,
                                   colour=name, method=method)
                        if split == "test":
                            row.update(measured_output(value, sq[j], sharp, texture, left, right, normal,
                                                       clean_step, clean, clean_tree))
                            if method == "gated_rgb_plane":
                                row["mean_rgb_gate"] = float(color_gate.mean())
                            rows.append(row)
                        else:
                            forward = clean_tree.query(value)[0]
                            backward = cKDTree(value).query(clean)[0]
                            row["cd_squared"] = float(np.mean(forward**2) + np.mean(backward**2))
                            val_rows.append(row)
            print(f"Plane expert {split}: {shape}", flush=True)
    scores = pd.DataFrame(rows)
    assert len(scores) == 12 * len(XYZ_LEVELS) * len(COLOURS) * len(METHODS)
    scores.to_csv(args.out / "test_scores.csv", index=False)
    pd.DataFrame(val_rows).to_csv(args.out / "validation_scores.csv", index=False)
    print(f"PLANE EXPERTS COMPLETE · {len(scores)} test rows · {args.out}", flush=True)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.run / "run.json").read_text())
    assert digest_file(args.data / "selection.json") == meta["selection_sha256"]
    assert {key: len(meta["objects"][key]) for key in ("train", "val", "test")} == {"train": 42, "val": 6, "test": 12}
    config_path = args.out / "fit.json"
    model = fit(args, meta)
    config_path.write_text(json.dumps(model, indent=2))
    print(f"Fitted plane experts: plain RGB weight {model['plain_rgb_weight']:.4f}", flush=True)
    measure(args, meta, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "run", "candidates", "mesh-cache", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    main(parser.parse_args())
