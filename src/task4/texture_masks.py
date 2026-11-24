"""Recreate the original GSO texture-only masks from the Task 3 candidates."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from ranking import compute_scores, compute_thresholds

CANDIDATE_COUNT = 1030
SHARD_COUNT = 43
NEIGHBOURS = 24


def generate(data, candidates, out):
    manifest = json.loads((data / "manifest.json").read_text())
    selection = json.loads((data / "selection.json").read_text())
    targets = {
        oid for oid, item in selection.items()
        if item["split"] == "test" and item["group"] == "texture_only"
    }
    if len(targets) != 2 or manifest["skipped"]:
        raise ValueError("Expected the original complete GSO run and its two texture-only test shapes")
    shards = sorted(candidates.glob("*.npz"))
    if len(shards) != SHARD_COUNT:
        raise ValueError(f"Expected {SHARD_COUNT} Task 3 candidate shards in {candidates}")

    scores = {}
    with threadpool_limits(limits=2):
        for index, shard in enumerate(shards, 1):
            with np.load(shard, allow_pickle=False) as archive:
                if len(archive["skipped"]):
                    raise ValueError(f"Missing candidate objects in {shard}")
                for name, xyz, rgb in zip(archive["ids"], archive["xyz"], archive["rgb"], strict=True):
                    oid = str(name)
                    if oid in scores:
                        raise ValueError(f"Duplicate candidate object ID {oid}")
                    g, c = compute_scores(xyz, rgb, k=NEIGHBOURS)
                    if not (np.isfinite(g).all() and np.isfinite(c).all()):
                        raise ValueError(f"Invalid candidate scores for {oid}")
                    scores[oid] = (g, c)
                    if oid in targets:
                        with np.load(data / "test" / oid / "clean.npz", allow_pickle=False) as clean:
                            if not (np.array_equal(clean["xyz"], xyz) and np.array_equal(clean["rgb"], rgb)):
                                raise ValueError(f"Candidate and clean test cloud differ for {oid}")
            if index % 7 == 0 or index == len(shards):
                print(f"{index}/{len(shards)} candidate archives · {len(scores)} objects", flush=True)

    if len(scores) != CANDIDATE_COUNT or not targets <= scores.keys():
        raise ValueError("Incomplete Task 3 candidate cache")
    thresholds = compute_thresholds(scores)
    geometry_p50 = float(thresholds["g_low"])
    colour_p90 = float(thresholds["c_high"])
    masks = {oid: (scores[oid][0] <= geometry_p50) & (scores[oid][1] >= colour_p90)
             for oid in sorted(targets)}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("Texture-only mask is empty")

    metadata = dict(revision=manifest["revision"], candidate_objects=len(scores), neighbours=NEIGHBOURS,
                    geometry_p50=geometry_p50, colour_p90=colour_p90,
                    selected_points={oid: int(mask.sum()) for oid, mask in masks.items()},
                    numpy=version("numpy"), scikit_learn=version("scikit-learn"))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "clean_point_masks.npz", **masks)
    (out / "mask_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "candidates", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    generate(args.data, args.candidates, args.out)
