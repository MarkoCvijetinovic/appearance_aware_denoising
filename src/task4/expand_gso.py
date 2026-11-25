"""Build an object-disjoint GSO expansion and reuse the frozen PGD cache."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from train_rgb_pgd import RGB_LEVELS, XYZ_LEVELS, digest_file, load_samples, precompute

SEED = 20261124
COUNTS = {"train": (28, 14, 14, 28), "val": (8, 4, 4, 8), "test": (14, 7, 7, 14)}


def family(oid):
    name, _, suffix = oid.rpartition("_")
    if name and len(suffix) == 11 and suffix.isalnum() and any(c.isdigit() for c in suffix):
        oid = name
    return re.sub(r"[^a-z0-9]", "", oid.lower())


def prepare(base, candidates, output):
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from corruption import add_colour_noise, add_geometry_noise
    from ranking import rank_candidates, select_objects

    original = json.loads((base / "selection.json").read_text())
    manifest = json.loads((base / "manifest.json").read_text())
    if len(original) != 60 or manifest["skipped"]:
        raise ValueError("Expected the original 60-object GSO experiment")
    archives = sorted(candidates.glob("*.npz"))
    if len(archives) != 43:
        raise ValueError("Expected the 43 cached GSO archives")
    output.mkdir(parents=True, exist_ok=True)
    ranking_path = output / "candidate_ranking.csv"
    source = {}
    for path in archives:
        with np.load(path, allow_pickle=False) as data:
            for index, oid in enumerate(data["ids"].tolist()):
                if oid in source or Path(oid).name != oid or oid in {".", ".."}:
                    raise ValueError(f"Invalid or duplicate object ID: {oid}")
                source[oid] = path, index
    if len(source) != 1030 or not set(original) <= source.keys():
        raise ValueError("Incomplete candidate source")

    def all_clouds():
        for i, path in enumerate(archives, 1):
            with np.load(path, allow_pickle=False) as data:
                for oid, xyz, rgb in zip(data["ids"], data["xyz"], data["rgb"], strict=True):
                    yield xyz, rgb, {"object_id": str(oid)}
            if i % 7 == 0 or i == len(archives):
                print(f"Ranked {i}/{len(archives)} GSO shards", flush=True)

    if ranking_path.is_file():
        ranking = pd.read_csv(ranking_path, dtype={"object_id": str})
    else:
        ranking = rank_candidates(all_clouds())
        partial = ranking_path.with_suffix(".part")
        ranking.to_csv(partial, index=False)
        partial.replace(ranking_path)
    if set(ranking.object_id) != set(source) or ranking.object_id.duplicated().any():
        raise ValueError("GSO ranking and candidate IDs differ")
    seen = {family(oid) for oid in original}
    eligible = []
    for oid in sorted(set(source) - set(original)):
        signature = family(oid)
        if signature not in seen:
            eligible.append(oid)
            seen.add(signature)
    composition = tuple(map(sum, zip(*COUNTS.values(), strict=True)))
    groups = select_objects(ranking[ranking.object_id.isin(eligible)], seed=SEED, composition=composition)
    rng = np.random.default_rng(SEED)
    extras = {}
    names = ("agreement", "geometry_only", "texture_only", "random")
    for group in names:
        ordered = rng.permutation(groups[group]).tolist()
        first = 0
        for split, counts in COUNTS.items():
            end = first + counts[names.index(group)]
            extras.update({oid: {"split": split, "group": group} for oid in ordered[first:end]})
            first = end
    assert len(extras) == 150 and not set(extras) & set(original)

    selection = output / "selection.json"
    if selection.exists() and json.loads(selection.read_text()) != extras:
        raise ValueError("A different selection already exists at this output path")
    selection.write_text(json.dumps(extras, indent=2) + "\n")
    protocol = {
        "source": manifest["source"], "revision": manifest["revision"], "seed": SEED,
        "base_selection_sha256": digest_file(base / "selection.json"),
        "additional_objects_by_split": {split: sum(counts) for split, counts in COUNTS.items()},
        "groups_by_split": {split: dict(zip(names, counts, strict=True)) for split, counts in COUNTS.items()},
        "additional_selection_sha256": digest_file(selection),
    }
    (output / "expansion.json").write_text(json.dumps(protocol, indent=2) + "\n")

    selected = {}
    for oid in extras:
        path, index = source[oid]
        selected.setdefault(path, []).append((oid, index))

    def random_for(oid, tag):
        key = hashlib.sha256(f"{manifest['seed']}:{oid}:{tag}".encode()).digest()[:8]
        return np.random.default_rng(int.from_bytes(key, "little"))

    def save(path, xyz, rgb):
        with path.with_suffix(".part").open("wb") as stream:
            np.savez_compressed(stream, xyz=xyz, rgb=rgb)
        path.with_suffix(".part").replace(path)

    completed = 0
    for archive, items in sorted(selected.items()):
        with np.load(archive, allow_pickle=False) as data:
            all_xyz, all_rgb = data["xyz"], data["rgb"]
        for oid, idx in items:
            xyz, rgb = all_xyz[idx], all_rgb[idx]
            if xyz.shape != (manifest["points"], 3) or rgb.shape != xyz.shape:
                raise ValueError(f"Bad candidate shape for {oid}")
            folder = output / extras[oid]["split"] / oid
            if not (folder / ".complete").is_file():
                folder.mkdir(parents=True, exist_ok=True)
                save(folder / "clean.npz", xyz, rgb)
                for sigma in XYZ_LEVELS:
                    noisy = add_geometry_noise(xyz, sigma, random_for(oid, f"xyz_{sigma:g}"))
                    for colour in RGB_LEVELS:
                        colours = add_colour_noise(rgb, colour, random_for(oid, f"rgb_{colour:g}"))
                        save(folder / f"noise_xyz_{sigma:g}_rgb_{colour:g}.npz", noisy, colours)
                (folder / ".complete").touch()
            completed += 1
        print(f"Generated {completed}/{len(extras)} additional GSO objects", flush=True)


def pgd(args):
    import torch
    from pytorch3d.ops import knn_points

    meta = json.loads((args.run / "run.json").read_text())
    original = meta["pgd"]
    repo = args.pgd.resolve()
    checkpoint = repo / "pretrained" / "PGD.ckpt"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    actual = dict(revision=revision, weights=digest_file(checkpoint),
                  torch=torch.__version__, device=torch.cuda.get_device_name())
    if any(actual[k] != original[k] for k in actual):
        raise ValueError(f"PGD environment differs from original: {actual}")
    samples = load_samples(args.extra, smoke=False)
    _, config = precompute(samples, repo, checkpoint, original["iterations"], args.pgd_cache,
                           original["seed"], torch.device("cuda"), knn_points)
    if config != original:
        raise ValueError("PGD configuration changed")
    print("EXPANDED_PGD_READY", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prep = subparsers.add_parser("prepare")
    baseline = subparsers.add_parser("pgd")
    for flag in ("data", "candidates", "extra"):
        prep.add_argument(f"--{flag}", required=True, type=Path)
    for flag in ("extra", "run", "pgd", "pgd-cache"):
        baseline.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.data, args.candidates, args.extra)
    else:
        pgd(args)
