"""Compare the 42- and 126-training-object models on the same fresh GSO objects."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task5"))
from summarise import KEYS, SCALES, bootstrap

MODELS = {
    "PGD": ("frozen_pgd", "constant"),
    "Geometry": ("geometry", "constant"),
    "RGB": ("rgb", "original"),
    "RGB constant": ("rgb", "constant"),
    "RGB shuffled": ("rgb", "shuffled"),
    "RGB Gaussian 0.05": ("rgb", "gaussian_0.05"),
    "RGB Gaussian 0.15": ("rgb", "gaussian_0.15"),
}


def read_run(root, size, seed):
    directory = root / f"train_{size}"
    if seed != 20261124:
        directory = directory / "repeats" / str(seed)
    frame = pd.read_csv(directory / "per_object.csv")
    protocol = json.loads((directory / "protocol.json").read_text())
    report = json.loads((directory / "result.json").read_text())
    objects = protocol["objects"]
    models = dict(MODELS)
    if size == 126:
        for label, (mode, colour) in MODELS.items():
            if mode == "rgb":
                models[label.replace("RGB", "Plain RGB")] = "plain_rgb", colour
    if len(frame) != 42 * 3 * len(models) or set(objects["test"]) != set(frame.object_id):
        raise ValueError(f"Incomplete evaluation for {size} training objects")
    if tuple(len(objects[split]) for split in ("train", "val", "test")) != (size, 24, 42):
        raise ValueError("Unexpected split sizes")
    for mode in protocol["modes"]:
        if sum(row["updates"] for row in report["models"][mode]["history"]) != 3024:
            raise ValueError(f"Unexpected optimizer step count for {mode} at {size}")
    scores = {}
    for label, (model, condition) in models.items():
        subset = frame[(frame.method == model) & (frame.colour == condition)]
        paired = subset.set_index(KEYS, verify_integrity=True).sort_index()
        if len(paired) != 42 * 3:
            raise ValueError(f"Incomplete {label} at {size} training objects")
        scores[label] = paired[list(SCALES)]
    return scores, protocol, report


def paired_summary(label, treatment, reference):
    for metric, scale in SCALES.items():
        paired = pd.concat({"treatment": treatment[metric], "reference": reference[metric]}, axis=1)
        objects = paired.dropna().groupby(level="object_id").mean() * scale
        delta = objects.treatment - objects.reference
        mean, low, high = bootstrap(delta)
        ref_mean = float(objects.reference.mean())
        yield dict(comparison=label, metric=metric, scale=scale, n_objects=len(objects),
                   wins=int((delta < 0).sum()), treatment=float(objects.treatment.mean()),
                   reference=ref_mean, delta=mean, ci_low=low, ci_high=high,
                   relative_pct=100 * mean / ref_mean if ref_mean else np.nan)


def compare(root):
    repeat_sets = [
        {int(path.name) for path in (root / f"train_{count}" / "repeats").glob("*")
         if path.name.isdecimal() and (path / "result.json").is_file()}
        for count in (42, 126)
    ]
    if repeat_sets[0] != repeat_sets[1]:
        raise ValueError("Each completed training seed must be present in both runs")
    seeds = (20261124, *sorted(repeat_sets[0]))
    inputs = {(count, seed): read_run(root, count, seed) for count in (42, 126) for seed in seeds}
    reference, p_small, _ = inputs[42, seeds[0]]
    for (count, _), (scores, protocol, _) in inputs.items():
        for split in ("val", "test"):
            if p_small["objects"][split] != protocol["objects"][split]:
                raise ValueError(f"Mismatched held-out {split} set")
        a, b = set(p_small["objects"]["train"]), set(protocol["objects"]["train"])
        if (count == 42 and a != b) or (count == 126 and not a < b):
            raise ValueError("The training expansion is not nested")
        if any(not frame.index.equals(reference["PGD"].index) for frame in scores.values()):
            raise ValueError("Evaluation inputs are not paired")
        pd.testing.assert_frame_equal(reference["PGD"], scores["PGD"], check_exact=False,
                                      rtol=1e-9, atol=1e-12)

    def requests(small, large):
        pairs = {f"{label}: 126 vs 42": (large[label], small[label])
                 for label in MODELS if label != "PGD"}
        for count, scores in ((42, small), (126, large)):
            pairs[f"{count}: geometry vs PGD"] = scores["Geometry"], scores["PGD"]
            pairs[f"{count}: original RGB vs PGD"] = scores["RGB"], scores["PGD"]
            pairs[f"{count}: original vs constant RGB"] = scores["RGB"], scores["RGB constant"]
            pairs[f"{count}: original vs shuffled RGB"] = scores["RGB"], scores["RGB shuffled"]
        for condition in ("", " constant", " shuffled", " Gaussian 0.05", " Gaussian 0.15"):
            pairs[f"126: learned vs fixed gate{condition}"] = (
                large[f"RGB{condition}"], large[f"Plain RGB{condition}"]
            )
        return pairs

    runs = {seed: requests(inputs[42, seed][0], inputs[126, seed][0]) for seed in seeds}
    by_seed = pd.DataFrame(
        dict(seed=seed, **row)
        for seed, pairs in runs.items()
        for name, (treatment, reference_scores) in pairs.items()
        for row in paired_summary(name, treatment, reference_scores)
    )
    by_seed.to_csv(root / "paired_comparison_by_seed.csv", index=False)
    contrasts = pd.DataFrame(
        row for name in runs[seeds[0]]
        for row in paired_summary(
            name, sum(runs[seed][name][0] for seed in seeds)/len(seeds),
            sum(runs[seed][name][1] for seed in seeds)/len(seeds),
        )
    )
    contrasts.to_csv(root / "paired_scale_comparison.csv", index=False)
    summary = pd.DataFrame([
        dict(train_objects=count, seed=seed, method=model, **(values.mean() * pd.Series(SCALES)).to_dict())
        for (count, seed), (scores, _, _) in inputs.items() for model, values in scores.items()
    ])
    summary.to_csv(root / "test_means_scaled.csv", index=False)
    test_groups = pd.read_csv(root / "train_42" / "per_object.csv", usecols=["object_id", "group"])
    test_groups = test_groups.drop_duplicates().set_index("object_id", verify_integrity=True)["group"]
    slice_tables = []
    for (count, seed), (scores, _, _) in inputs.items():
        for label, values in scores.items():
            metrics = values.mul(pd.Series(SCALES)).reset_index()
            metrics["group"] = metrics.object_id.map(test_groups)
            view = metrics.groupby(["group", "xyz_noise"])[list(SCALES)].mean().reset_index()
            view.insert(0, "method", label)
            view.insert(0, "seed", seed)
            view.insert(0, "train_objects", count)
            slice_tables.append(view)
    pd.concat(slice_tables, ignore_index=True).to_csv(root / "test_by_group_and_noise.csv", index=False)
    selection = {
        f"{size}/{seed}": {
            "seconds_for_all_seeds": report["seconds"],
            "checkpoints": {mode: row["checkpoint"] for mode, row in report["models"].items()},
        }
        for (size, seed), (_, _, report) in inputs.items()
    }
    (root / "scale_run_summary.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(contrasts[contrasts.metric == "p2surface_squared"][
          ["comparison", "relative_pct", "ci_low", "ci_high", "wins", "n_objects"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    compare(parser.parse_args().root)
