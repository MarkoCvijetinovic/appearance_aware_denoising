import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from datasets import load_dataset
from trimesh.visual.color import uv_to_interpolated_color

from corruption import add_geometry_noise, add_colour_noise

DATASET = "suvadityamuk/google-scanned-objects"


def parse_levels(value):
    return [float(x) for x in value.split(",")]


def stable_seed(seed, object_id, label):
    value = f"{seed}:{object_id}:{label}".encode()
    digest = hashlib.sha256(value).digest()
    return int.from_bytes(digest[:8], "little")


def load_glb(asset):
    if isinstance(asset, (trimesh.Scene, trimesh.Trimesh)):
        return asset

    if isinstance(asset, dict):
        if asset.get("bytes") is not None:
            asset = asset["bytes"]
        else:
            return trimesh.load(asset["path"], file_type="glb", force="scene")

    return trimesh.load(io.BytesIO(asset), file_type="glb", force="scene")



def sample_coloured_cloud(asset, count, rng):
    scene = load_glb(asset)

    if isinstance(scene, trimesh.Scene):
        meshes = [
            m for m in scene.dump()
            if isinstance(m, trimesh.Trimesh)
            and len(m.faces) > 0
            and m.area > 0
        ]
    else:
        meshes = [scene]

    if not meshes:
        raise ValueError("No valid mesh geometry")

    areas = np.asarray([m.area for m in meshes])
    counts = rng.multinomial(count, areas / areas.sum())

    xyz_parts = []
    rgb_parts = []

    for mesh, n in zip(meshes, counts):
        if n == 0:
            continue

        points, face_indices, bary = (
            trimesh.sample.sample_surface(
                mesh,
                n,
                return_barycentric=True,
                seed=int(rng.integers(0, 2**32)),
            )
        )

        visual = mesh.visual

        if isinstance(visual, trimesh.visual.TextureVisuals):
            if visual.uv is None:
                raise ValueError("Texture mesh has no UV coordinates")

            # Three UV coordinates per sampled triangle.
            triangle_uv = visual.uv[mesh.faces[face_indices]]

            # Interpolate UVs at the sampled barycentric positions.
            uv = np.einsum("nij,ni->nj", triangle_uv, bary)

            material = visual.material

            if isinstance(material, trimesh.visual.material.PBRMaterial):
                texture = material.baseColorTexture
                factor = material.baseColorFactor
            elif isinstance(material, trimesh.visual.material.SimpleMaterial):
                texture = material.image
                factor = None
            else:
                raise ValueError(
                    f"Unsupported material type: {type(material)}"
                )

            if texture is None:
                raise ValueError("Material has no base-colour texture")

            rgba = uv_to_interpolated_color(uv, texture)
            rgb = rgba[:, :3].astype(np.float32) / 255.0

            # glTF baseColorFactor modulates the sampled texture.
            if factor is not None:
                rgb *= np.asarray(factor[:3], dtype=np.float32) / 255.0

        elif visual.kind == "vertex":
            vertex_rgb = visual.vertex_colors[
                mesh.faces[face_indices], :3
            ].astype(np.float32)

            rgb = np.einsum("nij,ni->nj", vertex_rgb, bary) / 255.0

        elif visual.kind == "face":
            rgb = (
                visual.face_colors[face_indices, :3]
                .astype(np.float32) / 255.0
            )

        else:
            raise ValueError(
                f"Unsupported visual type: {type(visual)}"
            )

        xyz_parts.append(points.astype(np.float32))
        rgb_parts.append(np.clip(rgb, 0.0, 1.0))

    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(rgb_parts)

    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise ValueError("Non-finite coordinates or colours")

    return xyz, rgb


def preview(path, clean, noisy, clean_rgb, noisy_rgb):
    fig = plt.figure(figsize=(10, 5))

    for j, (xyz, rgb, title) in enumerate([
        (clean, clean_rgb, "Clean XYZRGB"),
        (noisy, noisy_rgb, "Noisy XYZRGB"),
    ]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=rgb, s=0.2, rasterized=True,
        )
        ax.set_title(title)
        ax.set_box_aspect(np.maximum(np.ptp(clean, axis=0), 1e-6))
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    with open(args.catalogue, newline="") as f:
        catalogue = list(csv.DictReader(f))

    rng_selection = np.random.default_rng(args.seed)

    if args.selection:
        with open(args.selection) as f:
            selected = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    else:
        ids = [r["object_id"] for r in catalogue]
        if args.amount > len(ids):
            raise ValueError("Requested more objects than available")
        selected = list(
            rng_selection.choice(
                ids, size=args.amount, replace=False
            )
        )

    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate object IDs in selection")

    # Assign whole objects to splits before generating samples.
    selected = sorted(selected)
    rng_selection.shuffle(selected)

    n = len(selected)
    n_train = int(n * args.train_fraction)
    n_val = int(n * args.val_fraction)

    assignments = {}
    for i, object_id in enumerate(selected):
        split = (
            "train" if i < n_train else
            "val" if i < n_train + n_val else
            "test"
        )
        assignments[object_id] = split

    # Save the exact selection and assignments.
    (root / "selection.json").write_text(
        json.dumps(assignments, indent=2)
    )

    ds = load_dataset(DATASET, split="train", streaming=True)
    pending = set(selected)
    results = []

    for sample in ds:
        meta = sample["json"]
        object_id = str(meta["object_id"])

        if object_id not in pending:
            continue

        split = assignments[object_id]
        folder = root / split / object_id
        folder.mkdir(parents=True, exist_ok=True)

        print(f"{split}: {object_id} ({meta.get('name')})")

        try:
            rng_points = np.random.default_rng(stable_seed(args.seed, object_id, "surface"))

            clean, rgb = sample_coloured_cloud(sample["glb"], args.points, rng_points)

            # Preserve the GLB coordinate system. Estimate scale
            # from this sampled cloud, matching our benchmark checks.
            center = (clean.max(axis=0) + clean.min(axis=0)) / 2
            radius = np.linalg.norm(clean - center, axis=1).max()

            if radius <= 0:
                raise ValueError("Degenerate point cloud")

            np.savez_compressed(
                folder / "clean.npz",
                xyz=clean,
                rgb=rgb,
            )

            for xyz_level in args.xyz_noise:
                for rgb_level in args.rgb_noise:
                    label = (f"noise_xyz_{xyz_level:g}_" f"rgb_{rgb_level:g}")

                    rng_noise = np.random.default_rng(stable_seed(args.seed, object_id, label))

                    noisy_xyz = add_geometry_noise(clean, xyz_level, rng_noise)
                    noisy_rgb = add_colour_noise(rgb, rgb_level, rng_noise)

                    np.savez_compressed(
                        folder / f"{label}.npz",
                        xyz=noisy_xyz,
                        rgb=noisy_rgb,
                    )

                    if args.preview:
                        preview(
                            folder / f"{label}.png",
                            clean, noisy_xyz, rgb, noisy_rgb
                        )

            results.append({
                "object_id": object_id,
                "name": meta.get("name"),
                "category": meta.get("category"),
                "split": split,
                "n_points": len(clean),
                "radius": float(radius),
                "status": "ok",
            })

        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({
                "object_id": object_id,
                "split": split,
                "status": "failed",
                "error": str(exc),
            })

        pending.remove(object_id)

        if not pending:
            break

    if pending:
        print("Objects not found:", sorted(pending))

    manifest = {
        "dataset": DATASET,
        "seed": args.seed,
        "xyz_noise": args.xyz_noise,
        "rgb_noise": args.rgb_noise,
        "objects": results,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"Generated {sum(r['status'] == 'ok' for r in results)} "
        f"of {len(selected)} selected objects"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue",
                        default="/data/catalogue/objects.csv")
    parser.add_argument("--selection", default=None)
    parser.add_argument("--amount", type=int, default=50)
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--xyz-noise", type=parse_levels,
                        default=[0.01, 0.02, 0.025])
    parser.add_argument("--rgb-noise", type=parse_levels,
                        default=[0.0, 0.05])
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", default="/data/gso")
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    if not (0 <= args.train_fraction <= 1
            and 0 <= args.val_fraction <= 1
            and args.train_fraction + args.val_fraction <= 1):
        parser.error("Invalid train/validation split fractions")

    main(args)
