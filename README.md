# L335 MiniProject: Appearance-Aware Geometric Denoising of Coloured Point Clouds

Can colour help a network recover the geometry of a noisy point cloud, and when does it mislead it, for example at texture edges that are not geometric edges? The project reproduces two denoising baselines, builds a coloured point cloud dataset, adds a small module that learns when to trust colour, and evaluates when colour helps or harms.

## Tasks

1. **Literature and baselines:** PGD and P2P-Bridge as main baselines; Score-Based Denoising and PD-Flow as background; SuperPC as a recent alternative.
2. **Reproduce two baselines:** PGD and the geometry-only object model of P2P-Bridge, with their released checkpoints on the PU-Net test data. First with each method's documented settings, then under one common protocol. Analyse failure cases before changing anything.
3. **Coloured dataset:** points and texture colours sampled from Google Scanned Objects, with separate geometric and colour corruption and object-disjoint splits.
4. **Appearance-aware module:** extend one pretrained baseline; compare geometry only, plain RGB fusion and a learned gate.
5. **Evaluation:** Chamfer distance, point-to-surface distance, thin structures and sharp features; ablations with constant, shuffled and corrupted colours.

## Baselines

| Method | Code | Checkpoint | Notes |
|---|---|---|---|
| PGD (AAAI 2026) | [git-guocc/PGD](https://github.com/git-guocc/PGD) → `external/PGD` | `pretrained/PGD.ckpt`, included in the repo | `test.py` applies the model 2× for 1 % and 2 % noise, 3× for 2.5 % (`--niters`) |
| P2P-Bridge (ECCV 2024) | [matvogel/P2P-Bridge](https://github.com/matvogel/P2P-Bridge) → `external/P2P-Bridge` | [Google Drive](https://drive.google.com/drive/folders/1hkd_gTU2EAMFJmgUzHmifviKDVunb6aK?usp=sharing), folder `PVDS_PUNet` | object model, xyz only; the `PVDL_*` models are for indoor scenes |

Both baselines are git submodules and stay unchanged. Before modifying one (Task 4), fork it and point the submodule to the fork.

## Data

PU-Net and PC-Net test data come from [ScoreDenoise](https://github.com/luost26/score-denoise) ([Google Drive](https://drive.google.com/drive/folders/1--MvLnP7dsBgBZiu46H0S32Y1eBa_j6P?usp=sharing), `data.zip`). Data, checkpoints and raw outputs are not tracked by git. Each baseline expects the data in its own folder layout, see their READMEs.

## GSO experiments for Tasks 3 to 5

1. Run the [Task 3 dataset notebook](notebooks/Task3_GSO_Team_Dataset_Colab.ipynb) in a CPU Colab runtime or reuse its Drive output. The dataset has 42 training, six validation and 12 test objects, with 10,000 points per object. Keep its `selection.json`, `manifest.json` and sibling candidate cache.
2. Run the [Task 4 training notebook](notebooks/Task4_RGB_PGD_Colab.ipynb) on a T4. It wraps the pinned PGD checkpoint in three matched residual models and saves their weights and per-object scores. The `frozen_pgd` row is PGD alone. The `geometry` row adds our trained XYZ-only residual. See the [model description](src/task4/README.md).
3. Use the [surface notebook](notebooks/Task4_Surface_Eval_Colab.ipynb) for mesh-based whole-object and texture-region scores. The [Task 5 notebook](notebooks/Task5_Paired_Evidence_Colab.ipynb) can generate its own prerequisite surface metrics and masks. It then runs the colour ablations, sharp-feature analysis, local plane experiment and flat paint intervention. Both use a CPU runtime. Set `PROJECT` and `TASK4_RUN` in the first code cell to select the Drive project and a specific full training run.

For a matched comparison, use the same test selection, identical noisy XYZ at 1%, 2% and 2.5%, and the original and Gaussian RGB conditions from Task 3. In Task 4, pair rows by `object_id`, `xyz_noise` and `rgb`. In Task 5, pair by `object_id`, `xyz_noise` and `colour`. Squared Chamfer uses both directions against the clean sampled cloud. One-way point-to-surface uses the clean triangle mesh. Both use the clean cloud radius as the unit of distance. The indexed PGD patch stitching in Task 4 preserves the RGB correspondence.

[Task 4 results](results/task4/gso_test_means.csv) average the original and 0.05 RGB-noise conditions. The texture column is restricted to the two texture-only test objects and their selected points. The [Task 5 summary tables](results/task5) cover the original, constant, shuffled and corrupted RGB conditions, the plane models with original RGB, and the high-contrast paint intervention. These are rounded summaries of the measured Colab runs. Full score CSVs and diagnostic figures are written to Drive. The replay notebook saves Task 5 results under `task5/<full run folder>`.

Reports are available for [Task 4](reports/task4/pgd_colour_denoising.docx) and [Task 5](reports/task5/colour_reliability.docx).

## Layout

```
L335_MiniProject/
├─ external/
│  ├─ PGD/            baseline 1 (submodule)
│  └─ P2P-Bridge/     baseline 2 (submodule)
├─ configs/           common evaluation protocol, experiment settings
├─ src/               our code: data preparation, metrics, appearance-aware module
├─ notebooks/         analysis and figures
├─ reports/           written findings per task
├─ results/           small result tables (CSV), one subfolder per task (task2/, …); raw outputs stay out of git
└─ data/              local only, not tracked
```

## Setup

```bash
git clone --recurse-submodules https://github.com/milanagm/L335_MiniProject.git
```

- Both baselines compile custom CUDA code and were tested with CUDA 11.8. They need an NVIDIA GPU (e.g. Colab) and do not run on Apple Silicon.
### Getting started on Colab

1. Open `notebooks/L335MiniProject.ipynb` in Colab (File → Open notebook → GitHub, tick "Include private repos" and allow access once).
2. Runtime → Change runtime type → T4 GPU. Keep the T4: the cache is built for its architecture.
3. Check the Settings cell (default: a folder `L335_MiniProject` in your own Drive), then Runtime → Run all and confirm the Drive dialog right away.
4. The first run compiles everything (about 45 min) and Step 5 stores a cache in your Drive; later runs take about 17 min. Your results go to `DRIVE_ROOT/results`.

- `notebooks/L335MiniProject.ipynb` sets up both baselines on a Colab T4 and runs the Task 2 reproductions (copy of the Colab notebook, with outputs). Compiled extensions, data and checkpoints are cached in Google Drive (`MyDrive/L335_MiniProject/cache`), so a new session takes about 17 min.
- The repository is private: cloning it on Colab needs a GitHub access token.

## Common evaluation protocol (Task 2)

For the comparison both methods get the same test shapes, point counts, noise realisations, normalisation and metric code:

- Test data: the 20 PU-Net test shapes from ScoreDenoise at 10K and 50K points.
- Noise: *x* % means Gaussian noise with a standard deviation of *x* % of the shape's radius, measured from the centre of its bounding box to its farthest point (the ScoreDenoise definition that both papers use). PGD reports 1 %, 2 % and 2.5 %; P2P-Bridge reports 1 %, 2 % and 3 %. The ScoreDenoise test data only contains noisy inputs at 1 %, 2 % and 3 %, so we use the public inputs at 1 % and 2 % and generate the 2.5 % inputs ourselves with the same definition (`src/data/corruption.py`, seed 2025).
- Settings: PGD with `--niters 2` at 1 % and 2 % and `--niters 3` at 2.5 %, as its README prescribes; P2P-Bridge with its default of 5 steps.
- P2M: for P2P-Bridge at 10K points and 1 % noise, its own paper reports 0.39, the PGD paper 0.686 (both ×10⁻⁴), while the Chamfer distances agree (2.28 vs. 2.284). The cause is one argument of PyTorch3D's `point_mesh_face_distance`: P2P-Bridge passes `min_triangle_area=0.0`, PGD keeps the default `0.005` (PyTorch3D 0.7.8), which treats small mesh triangles differently. Evaluating the same P2P-Bridge outputs (10K, 1 %) gives P2M 0.393 with `0.0` and 0.685 with `0.005`. Chamfer distance is not affected.
- Both repositories normalise prediction and ground truth to the unit sphere of the ground truth (ScoreDenoise procedure) and sum both directions of CD and P2M. The common evaluation uses this normalisation for all outputs and reports P2M with `min_triangle_area` 0 and 0.005; the ranking of the mean values is the same with both.

## Reproduction with documented settings (Task 2, step 1)

Run on a Colab T4 with the released checkpoints and the ScoreDenoise PU-Net test data (20 shapes per setting). Each method is evaluated with its own metric code.

- **PGD** (`results/task2/reproduction/pgd_documented_settings.csv`): `test.py` with `--niters 2` at 1 % and 2 % noise. Matches PGD's Table 1 to three decimals except at 50K / 1 % (CD 0.477 vs. 0.479, P2M 0.297 vs. 0.286). The README lists `--resolutions` and `--noise_lvls` separated by spaces, but `test.py` expects commas.
- **P2P-Bridge** (`results/task2/reproduction/p2pb_documented_settings.csv`): the README command `evaluate_objects.py --model_path ./pretrained/PVDS_PUNet/latest.pth --dataset PUNet` (default `--steps 5`). Its paper text says Table 1 used three denoising steps, so `--steps 3` was run as well. With `--steps 5` all 12 values match Table 1 within 0.01 (10 of them exactly when rounded); with `--steps 3` most values differ by 0.01 to 0.03. The published table therefore corresponds to the README setting. Runtime on a T4: about 25 min for `--steps 5` and 22 min for `--steps 3` (all six settings).
- **Per shape** (`results/task2/reproduction/per_shape_documented_settings.csv`): CD and P2M for each of the 20 test shapes in every setting above, taken from the evaluation logs, for the failure analysis. P2P-Bridge's log prints the raw values with six decimals, so its per-shape values have a resolution of 0.01 (×10⁻⁴); their means agree with the logged means within 0.002.
- **Raw outputs** (denoised point clouds as `.xyz`, an estimated 300 MB, plus the full logs) are in Google Drive under `MyDrive/L335_MiniProject/results/task2/` and are not tracked by git; the notebook regenerates them.

## Comparison under a common protocol (Task 2, step 2)

Run in the same Colab notebook (steps 2a to 3), evaluated with one metric code for both methods.

- **2.5 % inputs** (notebook step 2a): generated from the clean test clouds with the definition above. P2P-Bridge on these inputs matches its 2.5 % values in PGD's paper to within 2 %, so they are comparable to the paper's (unpublished) inputs.
- **Runs at 2.5 %** (steps 2b and 2c): PGD with `--niters 3` and, for comparison, `--niters 2`; P2P-Bridge with 5 steps (its `evaluate_objects.sample()` is called with noise 0.025, because the script fixes the levels to 1/2/3 %).
- **Per shape** (`results/task2/common_protocol/per_shape_common_eval.csv`, step 3): CD and P2M (both `min_triangle_area` values) of the noisy inputs and of all outputs at 1 %, 2 % and 2.5 %, one row per method, setting, resolution, noise level and shape (400 rows).
- **Summary** (`src/evaluation/summarise_common_protocol.py`, runs locally without a GPU): writes `common_protocol_means.csv` (means over the 20 shapes, Tables 4 and 5 of the report) and `common_protocol_per_shape_summary.csv` (per-shape CD ratios for the failure analysis) next to the per-shape file.
- **Results**: PGD has the lower mean CD and P2M in all six settings and the lower CD in 114 of 120 combinations of shape and setting; P2P-Bridge only wins on compact shapes at 50K points and 2.5 %. At 50K points and 2.5 % PGD is 5 to 6 % worse than its paper while P2P-Bridge on the same inputs agrees, which matches PGD's P2M deviation at 50K / 1 % on the public inputs, so the generated inputs do not explain it.
- **Figure** (`src/evaluation/plot_failure_cases.py`, runs in Colab with the Drive mounted): chair and Octahedron at 50K points and 2.5 %, saved as `reports/task2/figures/failure_cases_50k_2.5.png`.
- **Raw outputs** (2.5 % inputs, denoised point clouds and logs) are in Google Drive under `MyDrive/L335_MiniProject/results/task2/` and are not tracked by git.
