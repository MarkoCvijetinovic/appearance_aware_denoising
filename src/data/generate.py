import argparse
from pathlib import Path
import numpy as np
import json

from loading import prepare_candidates, stable_seed
from corruption import add_geometry_noise, add_colour_noise
from visualization import save_comparison
from ranking import select_objects, create_splits, rank_candidates


def parse_levels(value):
    return [float(x) for x in value.split(",")]

def generate_object(candidate, info, root, args):
    xyz, rgb, meta = candidate
    object_id = str(meta["object_id"])

    folder = root / info["split"] / object_id
    folder.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(folder / "clean.npz", xyz=xyz, rgb=rgb)

    for xyz_level in args.xyz_noise:
        for rgb_level in args.rgb_noise:
            label = f"noise_xyz_{xyz_level:g}_rgb_{rgb_level:g}"

            rng_xyz = np.random.default_rng(stable_seed(args.seed, object_id, f"xyz_{xyz_level:g}"))
            rng_rgb = np.random.default_rng(stable_seed(args.seed, object_id, f"rgb_{rgb_level:g}"))

            noisy_xyz = add_geometry_noise(xyz, xyz_level, rng_xyz)
            noisy_rgb = add_colour_noise(rgb, rgb_level, rng_rgb)

            np.savez_compressed(folder / f"{label}.npz", xyz=noisy_xyz, rgb=noisy_rgb)

            if args.preview:
                save_comparison(folder / f"{label}.png", xyz, noisy_xyz, rgb, noisy_rgb)

def generate_objects(assignments, candidates, args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        object_id = str(candidate[2]["object_id"])

        if object_id in assignments:
            generate_object(candidate, assignments[object_id], root, args)

def save_selection(assignments, args):
    selection_path = Path(args.output) / "selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)

    if selection_path.exists():
        existing = json.loads(selection_path.read_text())
        if existing != assignments:
            raise RuntimeError(
                "Existing dataset uses different object assignments. "
                "Choose another --output directory."
            )
    else:
        selection_path.write_text(json.dumps(assignments, indent=2))

def main(args):
    # 1. Generate/load clean candidate clouds.
    candidates = prepare_candidates(count=args.points, seed=args.seed)

    # 2. Rank and select objects.
    ranking = rank_candidates(candidates)
    groups = select_objects(ranking, seed=args.seed)
    assignments = create_splits(groups, seed=args.seed)

    # 3. Generate corrupted data for selected objects.
    save_selection(assignments, args)
    generate_objects(assignments, candidates, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", default="data/catalogue/objects.csv")
    parser.add_argument("--selection", default=None)
    parser.add_argument("--amount", type=int, default=50)
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--xyz-noise", type=parse_levels, default=[0.01, 0.02, 0.025])
    parser.add_argument("--rgb-noise", type=parse_levels, default=[0.0, 0.05])
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
