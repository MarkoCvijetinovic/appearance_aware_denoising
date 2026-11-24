"""Produce paired object-level contrasts and figures for the Task 5 report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEYS = ["object_id", "xyz_noise"]
SCALES = {
    "cd_squared": 1e6,
    "p2surface_squared": 1e6,
    "sharp_p2surface_squared": 1e6,
    "sharp_coverage_squared": 1e6,
    "texture_p2surface_squared": 1e6,
    "texture_normal_jump_rms": 1e3,
}
COLOURS = ("original", "constant", "shuffled", "gaussian_0.05", "gaussian_0.15")
PLOT_COLOURS = {"plain_rgb": "#C24C30", "gated_rgb": "#24658A",
                "plain_rgb_plane": "#8655A0", "gated_rgb_plane": "#2D8078"}
LABELS = {"plain_rgb": "Plain RGB residual", "gated_rgb": "Gated RGB residual",
          "plain_rgb_plane": "Plain RGB plane", "gated_rgb_plane": "Gated RGB plane"}


def bootstrap(values):
    values = np.asarray(values)
    rng = np.random.default_rng(1623)
    if not len(values):
        return np.nan, np.nan, np.nan
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return float(values.mean()), *map(float, np.quantile(means, [0.025, 0.975]))


def paired(frame, method_a, colour_a, method_b, colour_b, metric):
    a = frame[(frame.method == method_a) & (frame.colour == colour_a)].set_index(KEYS)[metric]
    b = frame[(frame.method == method_b) & (frame.colour == colour_b)].set_index(KEYS)[metric]
    assert a.index.is_unique and b.index.is_unique and a.index.sort_values().equals(b.index.sort_values())
    return pd.concat({"treatment": a, "reference": b}, axis=1).dropna().groupby(level=0).mean()


def record_contrast(frame, label, method_a, colour_a, method_b, colour_b, metric):
    objects = paired(frame, method_a, colour_a, method_b, colour_b, metric)
    delta = objects.treatment - objects.reference
    mu, lo, hi = bootstrap(delta * SCALES[metric])
    valid = objects.reference.abs() > 1e-12
    pct, pct_lo, pct_hi = bootstrap(100 * delta[valid] / objects.reference[valid])
    return dict(contrast=label, metric=metric, treatment=method_a, treatment_colour=colour_a,
                reference=method_b, reference_colour=colour_b, n_objects=len(delta), wins=int((delta < 0).sum()),
                equal=int((delta == 0).sum()), mean_treatment=float(objects.treatment.mean() * SCALES[metric]),
                mean_reference=float(objects.reference.mean() * SCALES[metric]), mean_difference=mu,
                ci_low=lo, ci_high=hi, object_mean_percent=pct, pct_ci_low=pct_lo, pct_ci_high=pct_hi)


def contrasts(main, planes):
    requests = []
    for method in ("plain_rgb", "gated_rgb"):
        requests.append((main, f"{method}: original vs constant", method, "original", method, "constant"))
        for colour in ("shuffled", "gaussian_0.05", "gaussian_0.15"):
            requests.append((main, f"{method}: {colour} vs original", method, colour, method, "original"))
    for colour in COLOURS:
        requests.append((main, f"residual gate vs plain: {colour}", "gated_rgb", colour, "plain_rgb", colour))
        requests.append((planes, f"plane gate vs geometry: {colour}", "gated_rgb_plane", colour, "geometry_plane", colour))
        requests.append((planes, f"plane gate vs plain: {colour}", "gated_rgb_plane", colour, "plain_rgb_plane", colour))
    for method in ("geometry_plane", "plain_rgb_plane", "gated_rgb_plane"):
        requests.append((planes, f"{method}: original vs frozen PGD", method, "original", "frozen_pgd", "original"))
    return pd.DataFrame([record_contrast(*entry, metric) for entry in requests for metric in SCALES])


def plot_ablation(contrast, out):
    metrics = [("cd_squared", "Chamfer"), ("p2surface_squared", "Surface error"),
               ("sharp_coverage_squared", "Sharp edge coverage"), ("texture_p2surface_squared", "Texture surface error")]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 3.6), layout="constrained")
    for axis, (key, title) in zip(axes.flat, metrics, strict=True):
        rows = contrast[contrast.metric.eq(key) & contrast.contrast.str.endswith("original vs constant")]
        axis.axvline(0, color="#888888", linewidth=0.8)
        for y, row in enumerate(rows.itertuples()):
            lo, hi, mid = row.pct_ci_low, row.pct_ci_high, row.object_mean_percent
            axis.plot([lo, hi], [y, y], lw=2, color=PLOT_COLOURS[row.treatment])
            axis.scatter(mid, y, color=PLOT_COLOURS[row.treatment], s=31, zorder=3)
        axis.set(yticks=[0, 1], yticklabels=["Plain RGB", "Gated RGB"], ylim=(-0.55, 1.55), title=title)
        axis.set_xlabel("Change from constant RGB (%)", fontsize=8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
    fig.savefig(out / "rgb_ablation.png", dpi=240)
    plt.close(fig)


def summarise_paint(paint, paint_dir, out):
    fields = ["induced_normal_shift_rms", "induced_step_rms", "step_rms_change"]
    per_object = paint.groupby(["method", "paint_contrast", "object_id"])[fields].mean()
    summary = []
    for (method, contrast), rows in per_object.groupby(level=[0, 1]):
        for metric in fields:
            mean, lo, hi = bootstrap(rows[metric] * 1e3)
            summary.append(dict(method=method, contrast=contrast, metric=metric, n_objects=len(rows),
                                mean=mean, ci_low=lo, ci_high=hi,
                                positive_objects=int((rows[metric] > 0).sum())))
    result = pd.DataFrame(summary)
    result.to_csv(out / "paint_summary.csv", index=False)
    focus = result[result.metric.eq("induced_step_rms")]
    fig, axis = plt.subplots(figsize=(7.5, 3.0), layout="constrained")
    for method, label in LABELS.items():
        rows = focus[focus.method.eq(method)].sort_values("contrast")
        if rows.empty:
            continue
        color = PLOT_COLOURS[method]
        axis.plot(rows.contrast, rows["mean"], marker="o", color=color, label=label)
        axis.fill_between(rows.contrast, rows.ci_low, rows.ci_high, color=color, alpha=0.12)
    axis.set(xlabel="Paint contrast per channel (0–1 RGB scale)",
             ylabel="Induced normal step RMS × 1,000", xticks=[0, 0.4, 0.8])
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out / "paint_intervention.png", dpi=240)
    plt.close(fig)

    strongest = paint[(paint.method == "plain_rgb") & (paint.xyz_noise == 0.02) & (paint.paint_contrast == 0.8)]
    case = strongest.loc[strongest.step_rms_change.idxmax()]
    with np.load(paint_dir / "paint_cases" / f"{case.object_id}.npz", allow_pickle=False) as saved:
        local = saved["local"]
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), layout="constrained")
        axes[0].scatter(local[:, 0], local[:, 1], s=10, c=np.where(local[:, 0] >= 0, "#A9B2BE", "#354C66"))
        axes[0].axvline(0, color="#444444", lw=0.6)
        axes[0].set(title="Paint applied to a flat patch", xlabel="Across the colour boundary",
                    ylabel="Along the colour boundary", aspect="equal")
        for method in PLOT_COLOURS:
            key = f"{method}_0.8"
            if key not in saved:
                continue
            displacement = (saved[key] - saved[f"{method}_0"]) * 1e3
            axis_distance = local[:, 0]
            use = np.abs(local[:, 1]) <= 0.07
            bins = np.linspace(-0.10, 0.10, 17)
            centers = 0.5 * (bins[:-1] + bins[1:])
            values = []
            for a, b in zip(bins[:-1], bins[1:], strict=True):
                selected = displacement[use & (axis_distance >= a) & (axis_distance < b)]
                values.append(selected.mean() if len(selected) >= 3 else np.nan)
            axes[1].plot(centers, values, marker=".", color=PLOT_COLOURS[method], label=LABELS[method])
        axes[1].axvline(0, color="#888888", lw=0.6)
        axes[1].axhline(0, color="#888888", lw=0.6)
        axes[1].set(title="Output change caused by paint", xlabel="Across the colour boundary",
                    ylabel="Signed normal displacement × 1,000")
        axes[1].legend(frameon=False, fontsize=7)
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
        fig.savefig(out / "paint_failure_case.png", dpi=240)
        plt.close(fig)
    return result, dict(object_id=str(case.object_id), xyz_noise=float(case.xyz_noise),
                        contrast=float(case.paint_contrast), plain_step_rms_change=float(case.step_rms_change * 1e3),
                        n_links=int(case.n_links))


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(args.gso / "scores.csv")
    planes = pd.read_csv(args.planes / "test_scores.csv")
    paint = pd.read_csv(args.paint / "paint_probe_scores.csv")
    contrast = contrasts(main, planes)
    contrast.to_csv(args.out / "paired_contrasts.csv", index=False)
    for name, frame in (("task4", main), ("plane", planes)):
        means = frame.groupby(["method", "colour"])[list(SCALES)].mean() * pd.Series(SCALES)
        means.to_csv(args.out / f"{name}_summary.csv")
    plot_ablation(contrast, args.out)
    paint_summary, case = summarise_paint(paint, args.paint, args.out)
    protocol = dict(bootstrap="10,000 paired resamples of whole test objects after averaging XYZ settings",
                    failure_case=case,
                    n_test_objects=int(main.object_id.nunique()),
                    n_paint_objects=int(paint.object_id.nunique()))
    (args.out / "summary_metadata.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps(protocol), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("gso", "planes", "paint", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    main(parser.parse_args())
