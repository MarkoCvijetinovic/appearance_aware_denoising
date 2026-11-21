import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from sampling import load_glb, sample_coloured_cloud
from corruption import add_geometry_noise, add_colour_noise
from visualization import save_comparison
from selection import create_assignments

DATASET = "suvadityamuk/google-scanned-objects"


def parse_levels(value):
    return [float(x) for x in value.split(",")]


def stable_seed(seed, object_id, label):
    value = f"{seed}:{object_id}:{label}".encode()
    digest = hashlib.sha256(value).digest()
    return int.from_bytes(digest[:8], "little")

def generate_object(sample, folder, args):
    object_id = str(sample["json"]["object_id"])

    rng_points = np.random.default_rng(stable_seed(args.seed, object_id, "surface"))

    meshes = load_glb(sample["glb"])
    clean, rgb = sample_coloured_cloud(meshes, args.points, rng_points)

    np.savez_compressed(folder / "clean.npz", xyz=clean, rgb=rgb)

    for xyz_level in args.xyz_noise:
        for rgb_level in args.rgb_noise:
            label = f"noise_xyz_{xyz_level:g}_rgb_{rgb_level:g}"

            rng_xyz = np.random.default_rng(
                stable_seed(args.seed, object_id, f"xyz_{xyz_level:g}")
            )

            rng_rgb = np.random.default_rng(
                stable_seed(args.seed, object_id, f"rgb_{rgb_level:g}")
            )

            noisy_xyz = add_geometry_noise(clean, xyz_level, rng_xyz)
            noisy_rgb = add_colour_noise(rgb, rgb_level, rng_rgb)

            np.savez_compressed(
                folder / f"{label}.npz",
                xyz=noisy_xyz,
                rgb=noisy_rgb,
            )

            if args.preview:
                save_comparison(folder / f"{label}.png", clean, noisy_xyz, rgb, noisy_rgb)

    return clean

def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    assignments = create_assignments(args)

    ds = load_dataset(DATASET, split="train", streaming=True)

    pending = set(assignments)
    results = []

    for sample in ds:
        meta = sample["json"]
        object_id = str(meta["object_id"])

        if object_id not in pending:
            continue

        split = assignments[object_id]
        folder = root / split / object_id
        folder.mkdir(parents=True, exist_ok=True)

        print(f"{split}: {object_id}")

        try:
            clean = generate_object(sample, folder, args)

            results.append({
                "object_id": object_id,
                "name": meta.get("name"),
                "category": meta.get("category"),
                "split": split,
                "n_points": len(clean),
                "status": "ok",
            })

        except Exception as exc:
            print(f"FAILED: {object_id}: {exc}")

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
