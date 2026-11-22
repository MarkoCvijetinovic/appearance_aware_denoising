"""Derive the Task 2 summary tables from the per-shape common evaluation.

Input:  results/task2/common_protocol/per_shape_common_eval.csv (notebook step 3, one row per method, setting, resolution, noise and shape)
Output: results/task2/common_protocol/common_protocol_means.csv          means over the 20 shapes (Table 4 of the report)
        results/task2/common_protocol/common_protocol_per_shape_summary.csv  per-shape CD ratios, averaged over the six settings
Usage:  python src/evaluation/summarise_common_protocol.py [results_dir]
"""
import csv
import statistics
import sys
from pathlib import Path

results = Path(sys.argv[1] if len(sys.argv) > 1 else "results/task2/common_protocol")
rows = list(csv.DictReader(open(results / "per_shape_common_eval.csv")))
group = lambda r: (r["method"], r["setting"], r["points"], r["noise"])

# means over the shapes, in the order the rows appear
groups = {}
for r in rows:
    groups.setdefault(group(r), []).append(r)
order = {"Noisy": 0, "PGD": 1, "P2P-Bridge": 2}
with open(results / "common_protocol_means.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["method", "setting", "points", "noise", "cd_x1e-4", "p2m_area0_x1e-4", "p2m_area0.005_x1e-4"])
    for key in sorted(groups, key=lambda k: (order[k[0]], k[1], int(k[2]), float(k[3]))):
        rs = groups[key]
        w.writerow(list(key) + ["%.4f" % statistics.mean(float(r[c]) for r in rs)
                                for c in ("cd_x1e-4", "p2m_area0_x1e-4", "p2m_area0.005_x1e-4")])

# per-shape CD ratios over the six settings (PGD with --niters 2 at 1 and 2 %, --niters 3 at 2.5 %, as in Table 4)
cd = {group(r) + (r["shape"],): float(r["cd_x1e-4"]) for r in rows}
settings = [(p, n) for p in ("10000", "50000") for n in ("0.01", "0.02", "0.025")]
pgd = lambda n: "niters_3" if n == "0.025" else "niters_2"
summary = []
for shape in sorted({r["shape"] for r in rows}):
    ratios = [(cd[("P2P-Bridge", "steps_5", p, n, shape)] / cd[("PGD", pgd(n), p, n, shape)],
               cd[("PGD", pgd(n), p, n, shape)] / cd[("Noisy", "input", p, n, shape)],
               cd[("P2P-Bridge", "steps_5", p, n, shape)] / cd[("Noisy", "input", p, n, shape)]) for p, n in settings]
    summary.append([shape] + [statistics.mean(x[i] for x in ratios) for i in range(3)])
summary.sort(key=lambda s: s[1])
with open(results / "common_protocol_per_shape_summary.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["shape", "cd_ratio_p2pb_over_pgd", "cd_ratio_pgd_over_noisy", "cd_ratio_p2pb_over_noisy"])
    w.writerows([s[0]] + ["%.2f" % x for x in s[1:]] for s in summary)
print("wrote", results / "common_protocol_means.csv", "and", results / "common_protocol_per_shape_summary.csv")
