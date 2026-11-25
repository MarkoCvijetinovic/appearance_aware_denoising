"""Evaluate the same RGB checkpoints with aligned and displaced paint boundaries."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task4"))
from evaluate_surface import digest_file, find_shards, original_meshes, point_to_surface
from surface_residual import Case, ShapeResidual, local_geometry, physical, predict
from train_rgb_pgd import XYZ_LEVELS, load_cloud
from evaluate_gso import normalize, read_mesh, seed_for
from summarise import bootstrap
from paint_probe import select_patch

CONTRASTS = (0.4, 0.8)
SHIFTS = (0.10, 0.13)
HALF_LENGTH = 0.14
TRUE_WIDTH = 0.045
FALSE_WIDTH = 0.045
PLANE_TOL = 0.003
NORMAL_LIMIT = np.cos(np.deg2rad(12))


@dataclass
class Probe:
    across: np.ndarray
    flat_across: np.ndarray
    true: np.ndarray
    pairs: np.ndarray
    flat_normal: np.ndarray
    shift: float
    angle: float
    placement: str


def locate_probe(clean, normals, topology, oid, allow_separate=False):
    angles = topology.face_adjacency_angles
    segments = topology.vertices[topology.face_adjacency_edges]
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    eligible = np.flatnonzero(
        (angles >= np.deg2rad(30)) & (angles <= np.deg2rad(120)) & (lengths >= 0.003)
    )
    if not len(eligible):
        return None
    longest = eligible[np.argsort(-lengths[eligible], kind="stable")[:64]]
    remaining = np.setdiff1d(eligible, longest)
    rng = np.random.default_rng(seed_for(oid, "aligned_colour_probe"))
    sampled = rng.choice(remaining, size=min(128, len(remaining)), replace=False)
    centers = topology.triangles_center
    mesh_normals = topology.face_normals
    tree = cKDTree(clean)
    best = best_crease = None
    for edge in np.r_[longest, sampled]:
        a, b = segments[edge]
        anchor = (a + b) * 0.5
        tangent = (b - a) / lengths[edge]
        faces = topology.face_adjacency[edge]
        rays = centers[faces] - anchor
        rays -= (rays @ tangent)[:, None] * tangent
        norms = np.linalg.norm(rays, axis=1)
        if min(norms) < 1e-9:
            continue
        rays /= norms[:, None]
        across = rays[0] - rays[1]
        across /= np.linalg.norm(across)
        group = np.asarray(tree.query_ball_point(anchor, 0.40), dtype=int)
        if len(group) < 30:
            continue
        relative = clean[group] - anchor
        x, y = relative @ across, relative @ tangent
        strip = np.abs(y) <= HALF_LENGTH
        base_planes = [
            (np.abs(relative @ normal) <= PLANE_TOL) &
            ((normals[group] @ normal) >= NORMAL_LIMIT) & (sign * x >= 0)
            for normal, sign in zip(mesh_normals[faces], (1, -1), strict=True)
        ]
        if allow_separate:
            broader = (np.abs(x) <= 0.06) & (np.abs(y) <= 0.20)
            sides = [group[side & broader] for side in base_planes]
            priority = (min(map(len, sides)), sum(map(len, sides)))
            if priority[0] >= 7 and (best_crease is None or priority > best_crease[0]):
                best_crease = (priority, anchor, across, np.concatenate(sides), float(angles[edge]))
        planes = [side & strip for side in base_planes]
        edge_near = np.abs(x) <= TRUE_WIDTH
        valid_edge = [group[side & edge_near] for side in planes]
        if min(map(len, valid_edge)) < 7:
            continue
        for face_ix, sign in enumerate((1, -1)):
            flat = planes[face_ix]
            for offset in SHIFTS:
                shift = sign * offset
                left = flat & (x < shift) & (x >= shift - FALSE_WIDTH)
                right = flat & (x >= shift) & (x <= shift + FALSE_WIDTH)
                if min(left.sum(), right.sum()) < 8:
                    continue
                gap, match = cKDTree(clean[group[right]]).query(clean[group[left]])
                good = gap <= 0.080
                pairs = np.column_stack((group[left][good], group[right][match[good]]))
                n_right = len(np.unique(pairs[:, 1]))
                if len(pairs) < 10 or n_right < 7:
                    continue
                score = (min(len(pairs), n_right * 2, *map(len, valid_edge)), len(pairs),
                         min(map(len, valid_edge)))
                if best is None or score > best[0]:
                    best = (score, anchor, across, np.concatenate(valid_edge), pairs,
                            mesh_normals[faces[face_ix]], shift, float(angles[edge]))
    if best is not None:
        _, anchor, across, true, pairs, normal, shift, angle = best
        position = (clean - anchor) @ across
        return Probe(position, position - shift, true, pairs, normal, shift, angle, "adjacent")
    if best_crease is None:
        return None
    flat = select_patch(clean, normals, oid)
    if flat is None:
        return None
    _, _, pairs, normal, index = flat
    _, anchor, across, true, angle = best_crease
    u = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    u /= np.linalg.norm(u)
    return Probe((clean - anchor) @ across, (clean - clean[index]) @ u, true, pairs,
                 normal, 0.0, angle, "separate")


def prepare_case(folder, noise, prefix, cache, center, radius):
    noisy, _ = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.npz")
    work_center, work_radius = normalize(noisy)
    xyz = (noisy - work_center) / work_radius
    key = hashlib.sha256(prefix + xyz.tobytes()).hexdigest()
    base = np.load(cache / f"{key}.npy", allow_pickle=False)
    if base.shape != xyz.shape or not np.isfinite(base).all():
        raise ValueError(f"Invalid cached PGD output in {folder}")
    ids, frame, edges, point = local_geometry(base, xyz)
    case = Case(folder.name, noise, float(work_radius / radius), base, edges, point, {},
                None, (work_center.astype(np.float64) - center) / radius)
    return case, frame, ids


def score_points(pred, clean, probe, verts, faces, ref):
    left, right = probe.pairs.T
    selected = np.unique(probe.pairs)
    true_sq = point_to_surface(pred[probe.true], verts, faces)
    coverage = cKDTree(pred).query(clean[probe.true])[0]
    flat_step_error = ((pred[right] - pred[left]) - (clean[right] - clean[left])) @ probe.flat_normal
    colour_step = ((pred[right] - pred[left]) - (ref[right] - ref[left])) @ probe.flat_normal
    height = (pred[selected] - ref[selected]) @ probe.flat_normal
    return dict(true_surface_sq=float(true_sq.mean()),
                true_coverage_sq=float(np.mean(coverage**2)),
                flat_step_rms=float(np.sqrt(np.mean(flat_step_error**2))),
                induced_flat_step_rms=float(np.sqrt(np.mean(colour_step**2))),
                induced_flat_normal_rms=float(np.sqrt(np.mean(height**2))))


def summarize(result, out):
    columns = ["true_surface_sq", "true_coverage_sq", "flat_step_rms",
               "induced_flat_step_rms", "induced_flat_normal_rms"]
    result.groupby(["method", "paint", "contrast"])[columns].mean().to_csv(out / "means.csv")

    def at(method, paint, contrast, metric):
        use = result[(result.method == method) & (result.paint == paint) & (result.contrast == contrast)]
        return use.set_index(["object_id", "xyz_noise"], verify_integrity=True)[metric].sort_index()

    contrasts = []
    for strength in CONTRASTS:
        for metric, paint, reference in (
            ("true_surface_sq", "aligned", "uniform"),
            ("true_surface_sq", "aligned", "shifted"),
            ("true_coverage_sq", "aligned", "uniform"),
            ("true_coverage_sq", "aligned", "shifted"),
            ("flat_step_rms", "shifted", "uniform"),
            ("flat_step_rms", "shifted", "aligned"),
            ("induced_flat_step_rms", "shifted", "uniform"),
        ):
            factor = 1e6 if metric.endswith("_sq") else 1e3
            ref_level = 0.0 if reference == "uniform" else strength
            effects = {}
            for method in ("learned", "fixed"):
                treatment = at(method, paint, strength, metric)
                control = at(method, reference, ref_level, metric)
                if not treatment.index.equals(control.index):
                    raise ValueError("Incomplete paired paint scores")
                effects[method] = (treatment - control).groupby(level="object_id").mean()
            effects["learned_minus_fixed"] = effects["learned"] - effects["fixed"]
            for method, values in effects.items():
                value, lo, hi = bootstrap(values.to_numpy() * factor)
                contrasts.append(dict(contrast=strength, method=method, metric=metric, treatment=paint,
                                      reference=reference, mean=value, ci_low=lo, ci_high=hi,
                                      n_objects=len(values), n_lower=int((values < 0).sum()), scale=factor))
    stats = pd.DataFrame(contrasts)
    stats.to_csv(out / "paired_effects.csv", index=False)
    print(stats[stats.contrast == 0.8].to_string(index=False), flush=True)


def run(args):
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    protocol = json.loads((args.model / "protocol.json").read_text())
    meta = json.loads((args.run / "run.json").read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    selection = json.loads((args.extra / "selection.json").read_text())
    original = json.loads((args.data / "selection.json").read_text())
    ids = sorted(oid for oid, entry in selection.items() if entry["split"] == "test")
    if (len(ids) != 42 or set(original) & set(ids) or ids != sorted(protocol["objects"]["test"])
            or digest_file(args.data / "selection.json") != protocol["selection_sha256"]
            or digest_file(args.extra / "selection.json") != protocol["expansion_sha256"]
            or digest_file(args.data / "selection.json") != meta["selection_sha256"]
            or len(protocol["objects"]["train"]) != 126):
        raise ValueError("Expected the exact saved 126/24/42 GSO experiment")
    models = {}
    weights = {}
    for method, filename in (("learned", "rgb.pt"), ("fixed", "plain_rgb.pt")):
        path = args.model / filename
        saved = torch.load(path, map_location=device, weights_only=True)
        net = ShapeResidual(use_rgb=True, learned_gate=method == "learned").to(device).eval()
        net.load_state_dict(saved["model"])
        models[method] = (net, saved["scale"])
        weights[method] = dict(file=filename, sha256=digest_file(path), epoch=saved["epoch"], scale=saved["scale"])
    print(f"Loaded both checkpoints on {device}", flush=True)
    meshes = args.mesh_cache / manifest["revision"]
    original_meshes(manifest["source"], manifest["revision"],
                    find_shards(set(ids), args.candidates), meshes)
    prefix = json.dumps(meta["pgd"], sort_keys=True).encode()

    # Determine object eligibility using the clean mesh before running either model.
    anchors, usable = [], {}
    for index, oid in enumerate(ids, start=1):
        folder = args.extra / "test" / oid
        xyz, _ = load_cloud(folder / "clean.npz")
        center, radius = normalize(xyz)
        clean = (xyz.astype(np.float64) - center) / radius
        mesh, topology = read_mesh(meshes / f"{oid}.glb", center, radius)
        verts = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
        distance, fid, _ = pcu.closest_points_on_mesh(clean, verts, faces)
        if np.any(fid < 0) or not np.isfinite(distance).all() or np.max(np.abs(distance)) > 1e-5:
            raise ValueError(f"Invalid clean mesh alignment for {oid}")
        probe = locate_probe(clean, mesh.face_normals[fid], topology, oid, args.allow_separate)
        if probe is not None:
            usable[oid] = (folder, center, radius, clean, verts, faces, probe)
        anchors.append(dict(object_id=oid, group=selection[oid]["group"], eligible=probe is not None,
                            placement=None if probe is None else probe.placement,
                            true_points=0 if probe is None else len(probe.true),
                            false_pairs=0 if probe is None else len(probe.pairs),
                            shift=None if probe is None else probe.shift,
                            edge_angle_degrees=None if probe is None else np.rad2deg(probe.angle)))
        if index % 7 == 0:
            print(f"Mesh screening {index}/{len(ids)} · {len(usable)} qualifying pairs", flush=True)
    pd.DataFrame(anchors).to_csv(args.out / "eligibility.csv", index=False)
    if not usable:
        raise RuntimeError("No object has both a qualifying crease and a neighbouring flat surface")

    rows = []
    for index, (oid, values) in enumerate(usable.items(), start=1):
        folder, center, radius, clean, verts, faces, probe = values
        for noise in XYZ_LEVELS:
            case, frame, neighbors = prepare_case(folder, noise, prefix, args.pgd_cache, center, radius)
            for paint, coordinates in (("aligned", probe.across), ("shifted", probe.flat_across)):
                for contrast in CONTRASTS:
                    gray = (0.5 + 0.5 * contrast * np.where(coordinates >= 0, 1, -1)).astype(np.float32)
                    channel = gray[neighbors] - gray[:, None]
                    case.rgb_edges[paint, contrast] = torch.from_numpy(
                        np.repeat(channel[:, :, None], 3, axis=2).astype(np.float16)
                    )
            conditions = [("uniform", 0.0), *case.rgb_edges]
            for method, (net, scale) in models.items():
                ref = physical(case, scale * predict(net, case, frame, conditions[0], device))
                for paint, contrast in conditions:
                    points = ref if contrast == 0 else physical(
                        case, scale * predict(net, case, frame, (paint, contrast), device)
                    )
                    rows.append(dict(object_id=oid, group=selection[oid]["group"], xyz_noise=noise,
                                     method=method, paint=paint, contrast=contrast, placement=probe.placement,
                                     **score_points(points, clean, probe, verts, faces, ref)))
        if index == 1 or index % 5 == 0 or index == len(usable):
            print(f"Inference {index}/{len(usable)} · {oid}", flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == len(usable) * len(XYZ_LEVELS) * len(models) * (1 + 2 * len(CONTRASTS))
    assert not result.duplicated(["object_id", "xyz_noise", "method", "paint", "contrast"]).any()
    assert result.loc[result.paint == "uniform", "induced_flat_step_rms"].eq(0).all()
    result.to_csv(args.out / "per_object.csv", index=False)
    summarize(result, args.out)
    (args.out / "protocol.json").write_text(json.dumps(dict(
        test_count=len(ids), eligible_objects=list(usable), geometry_angle_range_degrees=[30, 120],
        crease_band=TRUE_WIDTH, flat_band=FALSE_WIDTH, half_length=HALF_LENGTH,
        plane_tolerance=PLANE_TOL, max_normal_angle_degrees=12, shift_candidates=SHIFTS,
        allow_separate=args.allow_separate, broader_crease_band=0.06, broader_crease_half_length=0.20,
        placement_counts=pd.Series([row[-1].placement for row in usable.values()]).value_counts().to_dict(),
        colour_contrasts=CONTRASTS, noise_levels=XYZ_LEVELS, weights=weights,
        inference_device=str(device), seconds=time.monotonic()-started), indent=2))
    print(f"COLOUR_PROBE_COMPLETE · {len(usable)}/{len(ids)} objects · {(time.monotonic()-started)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "extra", "run", "model", "pgd-cache", "candidates", "mesh-cache", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--allow-separate", action="store_true",
                        help="Allow a second geometry-verified patch on the same object")
    run(parser.parse_args())
