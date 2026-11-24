"""Measure the completed GSO experiment against its original triangle surfaces."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import requests
import torch
import trimesh
from scipy.spatial import cKDTree

from train_rgb_pgd import MODES, XYZ_LEVELS, NEIGHBORS, ResidualHead, digest_file, load_cloud


def find_shards(ids, candidates):
    mapping = defaultdict(set)
    found = set()
    for archive in sorted(candidates.glob("*.npz")):
        with np.load(archive, allow_pickle=False) as data:
            matching = ids.intersection(data["ids"].tolist())
        if found.intersection(matching):
            raise ValueError("Duplicate test IDs in the candidate archives")
        if matching:
            mapping[int(archive.stem)].update(matching)
            found.update(matching)
        if found == ids:
            return mapping
    raise FileNotFoundError(f"Missing source shard for {sorted(ids - found)}")


def original_meshes(source, revision, mapping, directory):
    directory.mkdir(parents=True, exist_ok=True)

    def retrieve(item):
        index, objects = item
        pending = {oid for oid in objects if not (directory / f"{oid}.glb").is_file()}
        if not pending:
            return
        archive = f"gso-train-{index:05d}.tar"
        url = f"https://huggingface.co/datasets/{source}/resolve/{revision}/data/{archive}"
        for attempt in range(3):
            by_stem = defaultdict(dict)
            try:
                with requests.get(url, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    with tarfile.open(fileobj=response.raw, mode="r|*") as tar:
                        for member in tar:
                            stem, _, suffix = Path(member.name).name.partition(".")
                            if not member.isfile() or suffix not in {"glb", "json"}:
                                continue
                            if stem not in pending:
                                continue
                            content = tar.extractfile(member).read()
                            entry = by_stem[stem]
                            entry[suffix] = content
                            if len(entry) == 2:
                                oid = str(json.loads(entry["json"])["object_id"])
                                if oid != stem:
                                    raise ValueError(f"{archive}: metadata and archive key differ for {stem}")
                                path = directory / f"{oid}.glb"
                                partial = path.with_suffix(".part")
                                partial.write_bytes(entry["glb"])
                                partial.replace(path)
                                pending.remove(oid)
                                del by_stem[stem]
                                if not pending:
                                    break
                if pending:
                    raise ValueError(f"{archive}: did not find meshes for {sorted(pending)}")
                print(f"{archive} · {len(objects)} reference mesh(es) ready", flush=True)
                return
            except (requests.RequestException, OSError, tarfile.ReadError):
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(retrieve, sorted(mapping.items())))


def normalize(xyz):
    center = (xyz.max(0) + xyz.min(0)) / 2
    radius = np.linalg.norm(xyz - center, axis=1).max()
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("Degenerate cloud")
    return center, radius


def point_to_surface(points, vertices, faces):
    _, fid, bc = pcu.closest_points_on_mesh(np.ascontiguousarray(points), vertices, faces)
    if np.any(fid < 0) or not np.isfinite(bc).all():
        raise ValueError("Failed closest-point query")
    closest = pcu.interpolate_barycentric_coords(faces, fid, bc, vertices)
    return np.sum((points - closest)**2, axis=-1)


def measure(args):
    meta = json.loads((args.run / "run.json").read_text())
    if meta["smoke"] or digest_file(args.data / "selection.json") != meta["selection_sha256"]:
        raise ValueError("This full run does not match the supplied Task 3 split")
    manifest = json.loads((args.data / "manifest.json").read_text())
    selections = json.loads((args.data / "selection.json").read_text())
    ids = set(meta["objects"]["test"])
    if ids != {oid for oid, info in selections.items() if info["split"] == "test"} or len(ids) != 12:
        raise ValueError("Run and dataset disagree about held-out objects")
    if any(Path(oid).name != oid or oid in {".", ".."} for oid in ids):
        raise ValueError("Invalid object ID")
    reference = pd.read_csv(args.run / "per_object.csv").set_index(
        ["object_id", "xyz_noise", "rgb", "method"], verify_integrity=True
    )
    if len(reference) != 360:
        raise ValueError("Full test CSV must have 360 rows")
    masks = {}
    if args.point_masks is not None:
        with np.load(args.point_masks, allow_pickle=False) as mask_archive:
            masks = {oid: mask_archive[oid] for oid in mask_archive.files}
        if not masks or not set(masks) <= ids:
            raise ValueError("Point masks must name held-out objects")
        ids = set(masks)
    mesh_dir = args.mesh_cache / manifest["revision"]
    original_meshes(manifest["source"], manifest["revision"], find_shards(ids, args.candidates), mesh_dir)

    torch.set_num_threads(2)
    heads = {}
    for mode in MODES:
        head = ResidualHead(mode)
        head.load_state_dict(torch.load(args.run / f"{mode}.pt", map_location="cpu", weights_only=True))
        heads[mode] = head.eval()
    cache_prefix = json.dumps(meta["pgd"], sort_keys=True).encode()
    rows, mesh_checks = [], []
    with torch.no_grad():
        for oid in sorted(ids):
            folder = args.data / "test" / oid
            clean, _ = load_cloud(folder / "clean.npz")
            clean_center, clean_radius = normalize(clean)
            scene = trimesh.load(io.BytesIO((mesh_dir / f"{oid}.glb").read_bytes()), file_type="glb", force="scene")
            mesh = trimesh.util.concatenate([m for m in scene.dump() if isinstance(m, trimesh.Trimesh)])
            vertices = np.ascontiguousarray((mesh.vertices - clean_center) / clean_radius, dtype=np.float64)
            faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
            points_per_cloud = len(clean)
            mask = masks.get(oid)
            if mask is not None:
                if mask.dtype != np.bool_ or mask.shape != (len(clean),) or not mask.any():
                    raise ValueError(f"Invalid clean-point mask for {oid}")
                points_per_cloud = int(mask.sum())
            all_points = [(clean.astype(np.float64) - clean_center) / clean_radius]
            object_rows = []
            for noise in XYZ_LEVELS:
                noisy, original = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.npz")
                same_noisy, jitter = load_cloud(folder / f"noise_xyz_{noise:g}_rgb_0.05.npz")
                if not np.array_equal(noisy, same_noisy):
                    raise ValueError(f"XYZ differs between RGB settings for {oid}")
                center, radius = normalize(noisy)
                xyz = (noisy - center) / radius
                key = hashlib.sha256(cache_prefix + xyz.tobytes()).hexdigest()
                pgd = np.load(args.pgd_cache / f"{key}.npy", allow_pickle=False)
                x = torch.from_numpy(xyz)
                base = torch.from_numpy(pgd)
                target = (clean - center) / radius
                target_tree = cKDTree(target)
                _, neighbor_ids = cKDTree(xyz).query(xyz, k=NEIGHBORS + 1)
                neighbors = torch.from_numpy(np.asarray(
                    [row[row != i][:NEIGHBORS] for i, row in enumerate(neighbor_ids)]
                ))
                for condition, rgb in [("original", original), ("jitter", jitter)]:
                    colors = torch.from_numpy(rgb)
                    predictions = {"noisy": x, "frozen_pgd": base}
                    predictions.update({mode: head(base, x, neighbors, colors)[0] for mode, head in heads.items()})
                    for method, prediction in predictions.items():
                        saved = reference.loc[(oid, noise, condition, method)]
                        predicted = prediction.numpy()
                        a, _ = target_tree.query(predicted)
                        b, _ = cKDTree(predicted).query(target)
                        score = (np.mean(a**2) + np.mean(b**2)) * float(radius / clean_radius)**2
                        if not np.isclose(score, saved.cd_squared, rtol=1e-4, atol=1e-8):
                            raise ValueError(
                                f"Recomputed Chamfer mismatch for {oid}/{noise}/{condition}/{method}: "
                                f"{score:g} vs {saved.cd_squared:g}"
                            )
                        row = dict(object_id=oid, group=saved.group, xyz_noise=noise, rgb=condition, method=method)
                        if mask is None:
                            row["cd_squared"] = saved.cd_squared
                        else:
                            row["n_points"] = points_per_cloud
                        normalized = (predicted.astype(np.float64) * radius + center - clean_center) / clean_radius
                        all_points.append(normalized)
                        object_rows.append(row)
            if mask is not None:
                all_points = [points[mask] for points in all_points]
            squared = point_to_surface(np.concatenate(all_points), vertices, faces)
            squared = squared.reshape(len(all_points), points_per_cloud)
            aligned = float(np.sqrt(squared[0].mean()))
            if aligned > 1e-5 or not np.isfinite(squared).all():
                raise ValueError(f"Reference mesh misalignment for {oid}: clean RMS = {aligned:g}")
            mesh_checks.append(dict(object_id=oid, n_faces=len(faces), clean_rms=aligned))
            for row, distances in zip(object_rows, squared[1:]):
                row["p2surface_mean"] = float(np.sqrt(distances).mean())
                row["p2surface_squared"] = float(distances.mean())
            rows.extend(object_rows)
            print(f"{len(mesh_checks)}/{len(ids)} · {oid} · clean reference RMS {aligned:.3g}", flush=True)
    result = pd.DataFrame(rows)
    expected = reference.loc[reference.index.get_level_values("object_id").isin(ids)]
    actual = result.set_index(reference.index.names, verify_integrity=True)
    if not actual.index.sort_values().equals(expected.index.sort_values()):
        raise ValueError("Surface rows do not cover the selected test cases")
    metrics = ["p2surface_mean", "p2surface_squared"]
    if not masks:
        metrics.insert(0, "cd_squared")
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "per_object_surface.csv", index=False)
    pd.DataFrame(mesh_checks).to_csv(args.out / "reference_mesh_checks.csv", index=False)
    result.groupby(["xyz_noise", "rgb", "method"])[metrics].mean().to_csv(args.out / "surface_by_condition.csv")
    result.groupby(["group", "method"])[metrics].mean().to_csv(args.out / "surface_by_group.csv")
    print("Overall means:\n" + result.groupby("method")[metrics].mean().to_string(float_format="%.9g"))
    print("Group means, squared P2S × 1e6:\n" +
          (result.groupby(["group", "method"]).p2surface_squared.mean().unstack()*1e6).to_string(float_format="%.6g"))
    print(f"COMPLETE · {len(result)} surface scores · {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "run", "pgd-cache", "candidates", "mesh-cache", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--point-masks", type=Path, help="NPZ mapping held-out object IDs to clean-point masks")
    measure(parser.parse_args())
