"""Add a colour boundary to a known flat GSO patch, with paired geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from evaluate_gso import (NEIGHBORS, XYZ_LEVELS, ResidualHead, geometry_regions, load_cloud,
                          normalize, read_mesh, seed_for)
from plane_experts import geometric_features, colour_features, gain

RADIUS = 0.24
NORMAL_ANGLE = np.cos(np.deg2rad(8))
CONTRASTS = (0.0, 0.40, 0.80)


def select_patch(xyz, normals, shape):
    tree = cKDTree(xyz)
    rng = np.random.default_rng(seed_for(shape, "paint_probe"))
    pool = rng.choice(len(xyz), min(400, len(xyz)), replace=False)
    # Select by true geometry and spatial coverage only, before model outputs.
    best = None
    for index, group in zip(pool, tree.query_ball_point(xyz[pool], RADIUS), strict=True):
        n = normals[index]
        local = xyz[group] - xyz[index]
        clean_plane = (np.abs(local @ n) <= 0.002) & ((normals[group] @ n) >= NORMAL_ANGLE)
        if clean_plane.sum() < 55:
            continue
        group = np.asarray(group)[clean_plane]
        local = xyz[group] - xyz[index]
        axis = np.eye(3)[np.argmin(np.abs(n))]
        u = np.cross(n, axis)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        coordinates = np.column_stack((local @ u, local @ v))
        s, t = coordinates.T
        left = (s < 0) & (s >= -0.055) & (np.abs(t) <= 0.13)
        right = (s >= 0) & (s <= 0.055) & (np.abs(t) <= 0.13)
        if min(left.sum(), right.sum()) < 11:
            continue
        gap, nearest = cKDTree(coordinates[right]).query(coordinates[left])
        nearby = gap <= 0.067
        pairs = np.column_stack((group[left][nearby], group[right][nearest[nearby]]))
        if len(pairs) < 11 or len(np.unique(pairs[:, 1])) < 8:
            continue
        priority = (min(len(np.unique(pairs[:, 0])), len(np.unique(pairs[:, 1]))), len(pairs))
        if best is None or priority > best[0]:
            best = (priority, group, coordinates, pairs, n, int(index))
    return None if best is None else best[1:]


def run(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    meta = json.loads((args.run / "run.json").read_text())
    threshold = json.loads((args.run / "surface/texture_regions/mask_metadata.json").read_text())
    prefix = json.dumps(meta["pgd"], sort_keys=True).encode()
    plane_model = json.loads(args.plane_model.read_text()) if args.plane_model.is_file() else None
    torch.set_num_threads(2)
    heads = {name: ResidualHead(name) for name in ("plain_rgb", "gated_rgb")}
    for name, network in heads.items():
        network.load_state_dict(torch.load(args.run / f"{name}.pt", map_location="cpu", weights_only=True))
        network.eval()
    rows = []
    cases = args.out / "paint_cases"
    cases.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for shape in sorted(meta["objects"]["test"]):
            folder = args.data / "test" / shape
            xyz, rgb = load_cloud(folder / "clean.npz")
            center, radius = normalize(xyz)
            clean = (xyz.astype(np.float64) - center) / radius
            _, topology = read_mesh(args.mesh_cache / manifest["revision"] / f"{shape}.glb", center, radius)
            normals = geometry_regions(clean, rgb, topology, threshold["geometry_p50"], threshold["colour_p90"])[-1]
            selected = select_patch(clean, normals, shape)
            if selected is None:
                print(f"Paint probe: no qualifying flat patch for {shape}", flush=True)
                continue
            area, local, pairs, normal, anchor = selected
            left, right = pairs.T
            clean_step = (clean[right] - clean[left]) @ normal
            cross = np.unique(pairs)
            for noise in XYZ_LEVELS:
                noisy, _ = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.npz")
                noise_center, noise_radius = normalize(noisy)
                norm_noisy = (noisy - noise_center) / noise_radius
                pgd_key = hashlib.sha256(prefix + norm_noisy.tobytes()).hexdigest()
                pgd = np.load(args.run.parent / "pgd_cache" / f"{pgd_key}.npy", allow_pickle=False)
                _, ids = cKDTree(norm_noisy).query(norm_noisy, k=NEIGHBORS + 1)
                ids = torch.from_numpy(np.asarray([row[row != i][:NEIGHBORS] for i, row in enumerate(ids)]))
                features = (torch.from_numpy(pgd), torch.from_numpy(norm_noisy), ids)

                def to_clean(points):
                    return (points.astype(np.float64) * noise_radius + noise_center - center) / radius

                base = to_clean(pgd)
                if plane_model is not None:
                    geom = geometric_features(base, to_clean(norm_noisy))
                    geo_delta = geom[1]
                    strength = gain(plane_model["geometry"], geom[0])[:, None]
                    geo = base + strength * geo_delta
                constants, heights = {}, {}
                for contrast in CONTRASTS:
                    colors = np.full(rgb.shape, 0.5, dtype=np.float32)
                    colors[area] += (0.5 * contrast * np.where(local[:, 0] >= 0, 1, -1))[:, None]
                    pred = {"frozen_pgd": base}
                    for name, network in heads.items():
                        pred[name] = to_clean(network(*features, torch.from_numpy(colors))[0].numpy())
                    if plane_model is not None:
                        cues, color_delta = colour_features(base, colors, geom, colors)
                        delta = strength * (color_delta - geo_delta)
                        pred["geometry_plane"] = geo
                        pred["plain_rgb_plane"] = geo + plane_model["plain_rgb_weight"] * delta
                        pred["gated_rgb_plane"] = geo + gain(plane_model["rgb_gate"], cues)[:, None] * delta
                    for method, values in pred.items():
                        step = (values[right] - values[left]) @ normal - clean_step
                        height = (values[cross] - clean[cross]) @ normal
                        if contrast == 0:
                            constants[method] = (values.copy(), step.copy())
                        ref, ref_step = constants[method]
                        effects = (values[cross] - ref[cross]) @ normal
                        rows.append(dict(object_id=shape, clean_anchor=anchor, xyz_noise=noise, paint_contrast=contrast,
                                         method=method, n_patch_points=len(area), n_links=len(pairs),
                                         normal_height_rms=float(np.sqrt(np.mean(height**2))),
                                         step_rms=float(np.sqrt(np.mean(step**2))),
                                         step_rms_change=float(np.sqrt(np.mean(step**2)) - np.sqrt(np.mean(ref_step**2))),
                                         induced_normal_shift_rms=float(np.sqrt(np.mean(effects**2))),
                                         induced_step_rms=float(np.sqrt(np.mean((step - ref_step)**2)))))
                        if noise == 0.02:
                            heights[f"{method}_{contrast:g}"] = (values[area] - clean[area]) @ normal
                if noise == 0.02:
                    np.savez_compressed(cases / f"{shape}.npz", local=local, **heights)
            print(f"Paint probe {shape} · {len(area)} planar samples · {len(pairs)} cross-boundary links", flush=True)
    result = pd.DataFrame(rows)
    assert result.object_id.nunique() >= 4, "Too few validated flat patches"
    assert not result.duplicated(["object_id", "xyz_noise", "paint_contrast", "method"]).any()
    invariant = result[result.method.isin(["frozen_pgd", "geometry_plane"])]
    assert (invariant.induced_step_rms == 0).all() and (invariant.induced_normal_shift_rms == 0).all()
    result.to_csv(args.out / "paint_probe_scores.csv", index=False)
    (args.out / "paint_protocol.json").write_text(json.dumps(dict(plane_radius=RADIUS, clean_height_tolerance=0.002,
        clean_normal_tolerance_degrees=8, contrasts=CONTRASTS, test_objects=sorted(result.object_id.unique()),
        baseline="same model and XYZ, uniform grey; treatment paints two grey levels on the clean planar patch"), indent=2))
    print(f"PAINT PROBE COMPLETE · {result.object_id.nunique()} objects · {len(result)} scores", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "run", "mesh-cache", "plane-model", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    run(parser.parse_args())
