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
from datasets import load_dataset

from sampling import load_glb, sample_coloured_cloud
from corruption import add_geometry_noise, add_colour_noise
from visualization import save_comparison

DATASET = "suvadityamuk/google-scanned-objects"


def parse_levels(value):
    return [float(x) for x in value.split(",")]


def stable_seed(seed, object_id, label):
    value = f"{seed}:{object_id}:{label}".encode()
    digest = hashlib.sha256(value).digest()
    return int.from_bytes(digest[:8], "little")

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

            meshes = load_glb(sample["glb"])
            clean, rgb = sample_coloured_cloud(meshes, args.points, rng_points)

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
                        save_comparison(folder / f"{label}.png", clean, noisy_xyz, rgb, noisy_rgb)

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
                        default="data/catalogue/objects.csv")
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
    parser.add_argument("--output", default="data/gso")
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    if not (0 <= args.train_fraction <= 1
            and 0 <= args.val_fraction <= 1
            and args.train_fraction + args.val_fraction <= 1):
        parser.error("Invalid train/validation split fractions")

    main(args)
