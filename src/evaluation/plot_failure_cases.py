"""Render the failure-case figure of the Task 2 report (chair and Octahedron at 50K points and 2.5 % noise).

Runs in Colab with the project Drive mounted at /content/drive. It reads the clean clouds from the cached
data.zip, the noisy inputs from notebook step 2a and the outputs of steps 2b (PGD, --niters 3) and 2c
(P2P-Bridge, 5 steps), normalises everything with the ground truth's bounding sphere and writes
results/task2/figures/failure_cases_50k_2.5.png. The CD labels (x 1e-4, three decimals as in Tables 4 and 5) are read from
the per-shape CSV of notebook step 3 (results/task2/common_protocol/per_shape_common_eval.csv in the repo).
"""
import glob, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/content/drive/MyDrive/L335_MiniProject"
import zipfile
gt_dir = "/content/gt50k"; os.makedirs(gt_dir, exist_ok=True)
with zipfile.ZipFile(D + "/cache/data.zip") as z:
    for m in z.namelist():
        if "pointclouds/test/50000_poisson/" in m and m.split("/")[-1] in ("chair.xyz", "Octahedron.xyz"):
            open(gt_dir + "/" + m.split("/")[-1], "wb").write(z.read(m)); print("extracted", m)
roots = [D + "/results/task2", D + "/results"]
def find(pattern):
    for r in roots:
        h = sorted(glob.glob(r + "/**/" + pattern, recursive=True))
        if h: return h[0]
noisy_dir = D + "/results/task2/inputs/PUNet_50000_poisson_0.025"
src = {"Clean": gt_dir,
       "Noisy": noisy_dir,
       "PGD": find("PUNet_Ours3x__50000_poisson_0.025"),
       "P2P-Bridge": find("P2P-Bridge_steps_5_50000_0.025/pcl")}
print(src)
# CD labels (x 1e-4) read from the common evaluation of notebook step 3
import csv
run = {"Noisy": ("Noisy", "input"), "PGD": ("PGD", "niters_3"), "P2P-Bridge": ("P2P-Bridge", "steps_5")}
rows = list(csv.DictReader(open(D + "/results/task2/per_shape_common_eval.csv")))
cd = {shape: {k: float(next(r["cd_x1e-4"] for r in rows if (r["method"], r["setting"]) == m
                            and r["points"] == "50000" and r["noise"] == "0.025" and r["shape"] == shape))
              for k, m in run.items()} for shape in ("chair", "Octahedron")}
print(cd)
view = {"chair": (15, -50), "Octahedron": (15, -30)}
fig = plt.figure(figsize=(12, 6.2))
for i, shape in enumerate(["chair", "Octahedron"]):
    gt = np.loadtxt(src["Clean"] + f"/{shape}.xyz")[:, :3]
    c = (gt.max(0) + gt.min(0)) / 2; s = np.linalg.norm(gt - c, axis=1).max()
    for j, k in enumerate(src):
        p = (np.loadtxt(src[k] + f"/{shape}.xyz")[:, :3] - c) / s
        ax = fig.add_subplot(2, 4, 4 * i + j + 1, projection="3d")
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=0.25, c="#2f6db3", alpha=0.6, linewidths=0, depthshade=False)
        ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7); ax.set_zlim(-0.7, 0.7)
        ax.set_box_aspect((1, 1, 1)); ax.set_proj_type("ortho"); ax.view_init(*view[shape]); ax.set_axis_off()
        t = k if k == "Clean" else f"{k}\nCD {cd[shape][k]:.3f}"
        ax.set_title(t, fontsize=12, y=1.0)
plt.subplots_adjust(left=0, right=1, top=0.92, bottom=0, wspace=0, hspace=0.15)
os.makedirs("/content/results/task2/figures", exist_ok=True)
out = "/content/results/task2/figures/failure_cases_50k_2.5.png"
fig.savefig(out, dpi=200)
os.makedirs(D + "/results/task2/figures", exist_ok=True)
import shutil; shutil.copy(out, D + "/results/task2/figures/")
print("FIG_DONE", out)
