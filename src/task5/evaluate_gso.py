"""Task 5 paired RGB interventions on the held-out GSO benchmark."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task4"))
from evaluate_surface import find_shards, normalize, original_meshes, point_to_surface
from train_rgb_pgd import MODES, NEIGHBORS, XYZ_LEVELS, ResidualHead, digest_file, load_cloud

COLOURS = ("original", "constant", "shuffled", "gaussian_0.05", "gaussian_0.15")
METHODS = ("noisy", "frozen_pgd", *MODES)
BAND_RADIUS = 0.03
MIN_REGION_POINTS = 30
MIN_TEXTURE_LINKS = 30
SEED = 20261123


def seed_for(oid, purpose):
    return int.from_bytes(hashlib.sha256(f"{SEED}:{oid}:{purpose}".encode()).digest()[:8], "little")


def intervention_colors(rgb, jitter, oid):
    perm = np.random.default_rng(seed_for(oid, "shuffle")).permutation(len(rgb))
    noise = np.random.default_rng(seed_for(oid, "gaussian")).normal(0, 0.15, rgb.shape)
    return {
        "original": rgb,
        "constant": np.broadcast_to(rgb.mean(axis=0), rgb.shape).copy(),
        "shuffled": rgb[perm],
        "gaussian_0.05": jitter,
        "gaussian_0.15": np.clip(rgb + noise, 0, 1).astype(np.float32),
    }


def geometry_regions(clean, rgb, mesh, g_median, c_p90):
    xyz = np.ascontiguousarray(clean, dtype=np.float64)
    edges = mesh.face_adjacency_edges
    angles = mesh.face_adjacency_angles
    segments = mesh.vertices[edges[(angles >= np.deg2rad(30)) & (angles <= np.deg2rad(150))]]
    if len(segments):
        n_samples = np.ceil(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) / 0.005).astype(int)
        divisions = np.maximum(n_samples, 1)
        segment_id = np.repeat(np.arange(len(segments)), divisions + 1)
        first_index = np.repeat(np.cumsum(divisions + 1) - divisions - 1, divisions + 1)
        steps = (np.arange(len(segment_id)) - first_index) / divisions[segment_id]
        edge_samples = segments[segment_id, 0] + steps[:, None] * (segments[segment_id, 1] - segments[segment_id, 0])
        sharp = cKDTree(edge_samples).query(xyz)[0] <= BAND_RADIUS
    else:
        sharp = np.zeros(len(xyz), dtype=bool)

    knn = NearestNeighbors(n_neighbors=25, algorithm="kd_tree").fit(xyz)
    indices = knn.kneighbors(xyz, return_distance=False)[:, 1:]
    neighbours = xyz[indices]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    eigen = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", centered, centered) / indices.shape[1])
    g = eigen[:, 0] / (eigen.sum(axis=1) + 1e-12)
    c = np.linalg.norm(rgb[indices] - rgb[:, None, :], axis=2).mean(axis=1)
    texture = (g <= g_median) & (c >= c_p90)

    import point_cloud_utils as pcu

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    _, face_ids, _ = pcu.closest_points_on_mesh(xyz, vertices, faces)
    normals = mesh.face_normals[face_ids]
    left = np.repeat(np.arange(len(xyz)), indices.shape[1])
    right = indices.ravel()
    potential = (texture[left] | texture[right]) & ~sharp[left] & ~sharp[right]
    pairs = np.unique(np.sort(np.column_stack((left[potential], right[potential])), axis=1), axis=0)
    a, b = pairs.T
    vector = xyz[b] - xyz[a]
    pair_normal = normals[a] + normals[b]
    pair_normal /= np.maximum(np.linalg.norm(pair_normal, axis=1, keepdims=True), 1e-12)
    clean_step = np.einsum("ij,ij->i", vector, pair_normal)
    across_texture = (
        (np.linalg.norm(rgb[b] - rgb[a], axis=1) >= c_p90)
        & (np.einsum("ij,ij->i", normals[a], normals[b]) >= np.cos(np.deg2rad(15)))
        & (np.linalg.norm(vector, axis=1) <= 0.06)
        & (np.abs(clean_step) <= 0.003)
    )
    a, b = a[across_texture], b[across_texture]
    return sharp, texture, a, b, pair_normal[across_texture], clean_step[across_texture], normals


def read_mesh(path, center, radius):
    scene = trimesh.load(io.BytesIO(path.read_bytes()), file_type="glb", force="scene")
    original = trimesh.util.concatenate([part for part in scene.dump() if isinstance(part, trimesh.Trimesh)])
    raw = trimesh.Trimesh(vertices=(original.vertices - center) / radius, faces=original.faces, process=False)
    topology = raw.copy()
    topology.merge_vertices()
    return raw, topology


def measured_output(pred, surface_sq, sharp, texture, left, right, pair_normal, clean_step, clean, clean_tree):
    pred_tree = cKDTree(pred)
    forward = clean_tree.query(pred)[0]
    backward = pred_tree.query(clean)[0]
    stats = {
        "cd_squared": float(np.mean(forward**2) + np.mean(backward**2)),
        "p2surface_squared": float(surface_sq.mean()),
        "p2surface_mean": float(np.sqrt(surface_sq).mean()),
    }
    if sharp.sum() >= MIN_REGION_POINTS:
        stats["sharp_p2surface_squared"] = float(surface_sq[sharp].mean())
        stats["sharp_coverage_squared"] = float(np.mean(backward[sharp]**2))
        stats["sharp_coverage_p95"] = float(np.quantile(backward[sharp], 0.95))
    if texture.sum() >= MIN_REGION_POINTS:
        stats["texture_p2surface_squared"] = float(surface_sq[texture].mean())
    if len(left) >= MIN_TEXTURE_LINKS:
        output_step = np.einsum("ij,ij->i", pred[right] - pred[left], pair_normal)
        stats["texture_normal_jump_rms"] = float(np.sqrt(np.mean((output_step - clean_step)**2)))
    return stats


def measure(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    meta = json.loads((args.run / "run.json").read_text())
    selection = json.loads((args.data / "selection.json").read_text())
    assert digest_file(args.data / "selection.json") == meta["selection_sha256"]
    ids = sorted(meta["objects"]["test"])
    assert len(ids) == 12 and set(ids) == {oid for oid, entry in selection.items() if entry["split"] == "test"}
    thresholds = json.loads((args.run / "surface/texture_regions/mask_metadata.json").read_text())
    assert thresholds["revision"] == manifest["revision"] and thresholds["candidate_objects"] == 1030
    reference = pd.read_csv(args.run / "surface/per_object_surface.csv").set_index(
        ["object_id", "xyz_noise", "rgb", "method"], verify_integrity=True
    )
    mesh_dir = args.mesh_cache / manifest["revision"]
    original_meshes(manifest["source"], manifest["revision"], find_shards(set(ids), args.candidates), mesh_dir)
    prefix = json.dumps(meta["pgd"], sort_keys=True).encode()
    torch.set_num_threads(2)
    heads = {}
    for mode in MODES:
        head = ResidualHead(mode)
        head.load_state_dict(torch.load(args.run / f"{mode}.pt", map_location="cpu", weights_only=True))
        heads[mode] = head.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    picture_dir = args.out / "cases"
    picture_dir.mkdir(exist_ok=True)
    results, region_rows = [], []
    with torch.inference_mode():
        for oid in ids:
            folder = args.data / "test" / oid
            clean, clean_rgb = load_cloud(folder / "clean.npz")
            center, radius = normalize(clean)
            norm_clean = (clean.astype(np.float64) - center) / radius
            tree = cKDTree(norm_clean)
            raw_mesh, topology = read_mesh(mesh_dir / f"{oid}.glb", center, radius)
            sharp, texture, left, right, pair_normal, clean_step, normals = geometry_regions(
                norm_clean, clean_rgb, topology, thresholds["geometry_p50"], thresholds["colour_p90"]
            )
            vertices = np.ascontiguousarray(raw_mesh.vertices, dtype=np.float64)
            faces = np.ascontiguousarray(raw_mesh.faces, dtype=np.int32)
            clean_rms = float(np.sqrt(point_to_surface(norm_clean, vertices, faces).mean()))
            assert clean_rms < 1e-5, (oid, clean_rms)
            group = selection[oid]["group"]
            region_rows.append(dict(object_id=oid, group=group, clean_rms=clean_rms,
                                    sharp_points=int(sharp.sum()), texture_points=int(texture.sum()), texture_links=len(left)))
            for noise in XYZ_LEVELS:
                noisy, rgb = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.npz")
                same_noisy, jitter = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.05.npz")
                assert np.array_equal(noisy, same_noisy) and np.array_equal(rgb, clean_rgb)
                work_center, work_radius = normalize(noisy)
                xyz = (noisy - work_center) / work_radius
                key = hashlib.sha256(prefix + xyz.tobytes()).hexdigest()
                baseline = np.load(args.pgd_cache / f"{key}.npy", allow_pickle=False)
                assert baseline.shape == xyz.shape and np.isfinite(baseline).all()
                ids_knn = cKDTree(xyz).query(xyz, k=NEIGHBORS + 1)[1]
                knn = torch.from_numpy(np.asarray([row[row != i][:NEIGHBORS] for i, row in enumerate(ids_knn)]))
                features = (torch.from_numpy(baseline), torch.from_numpy(xyz), knn)
                colors = intervention_colors(rgb, jitter, oid)
                invariant = {"noisy": xyz, "frozen_pgd": baseline, "geometry": heads["geometry"](*features, torch.from_numpy(rgb))[0].numpy()}
                preds = {("", method): pred for method, pred in invariant.items()}
                for condition, color in colors.items():
                    for mode in ("plain_rgb", "gated_rgb"):
                        preds[(condition, mode)] = heads[mode](*features, torch.from_numpy(color))[0].numpy()
                normalized = {(rgb_name, method): (pred.astype(np.float64) * work_radius + work_center - center) / radius
                              for (rgb_name, method), pred in preds.items()}
                distances = point_to_surface(np.concatenate(list(normalized.values())), vertices, faces).reshape(len(preds), -1)
                scores = {key: measured_output(pred, surface_sq, sharp, texture, left, right, pair_normal,
                                               clean_step, norm_clean, tree)
                          for (key, pred), surface_sq in zip(normalized.items(), distances, strict=True)}
                for condition in COLOURS:
                    for method in METHODS:
                        source = ("", method) if method in invariant else (condition, method)
                        row = scores[source]
                        if condition in ("original", "gaussian_0.05"):
                            color_key = "original" if condition == "original" else "jitter"
                            previous = reference.loc[(oid, noise, color_key, method)]
                            for metric in ("cd_squared", "p2surface_squared"):
                                assert np.isclose(row[metric], previous[metric], rtol=1e-4, atol=1e-8), (oid, method, metric)
                        results.append(dict(object_id=oid, group=group, xyz_noise=noise, colour=condition,
                                            method=method, **row))
                if group == "texture_only" and noise == 0.02:
                    np.savez_compressed(picture_dir / f"{oid}.npz", clean=norm_clean, rgb=clean_rgb, normals=normals,
                                        sharp=sharp, texture=texture, left=left, right=right, pair_normal=pair_normal,
                                        clean_step=clean_step,
                                        **{f"{color}_{model}": pred for (color, model), pred in normalized.items()})
            print(f"Task 5 {len(region_rows)}/{len(ids)} · {oid} · {int(sharp.sum())} sharp samples · {len(left)} texture links", flush=True)
    scores = pd.DataFrame(results)
    assert len(scores) == 12 * 3 * len(COLOURS) * len(METHODS)
    assert not scores.duplicated(["object_id", "xyz_noise", "colour", "method"]).any()
    scores.to_csv(args.out / "scores.csv", index=False)
    pd.DataFrame(region_rows).to_csv(args.out / "region_sizes.csv", index=False)
    (args.out / "protocol.json").write_text(json.dumps(dict(seed=SEED, colour_conditions=COLOURS, methods=METHODS,
        sharp_dihedral_degrees=[30, 150], sharp_band=BAND_RADIUS, edge_sample_spacing=0.005,
        min_region_points=MIN_REGION_POINTS, min_texture_links=MIN_TEXTURE_LINKS,
        max_texture_neighbour_distance=0.06, max_clean_normal_jump=0.003,
        max_texture_normal_angle_degrees=15, mask_thresholds=thresholds, test_ids=ids,
        run_selection_sha256=meta["selection_sha256"]), indent=2))
    print(f"TASK 5 SCORES SAVED · {len(scores)} rows · {args.out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "run", "pgd-cache", "candidates", "mesh-cache", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    measure(parser.parse_args())
