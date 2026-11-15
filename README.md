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

## Layout

```
L335_MiniProject/
├─ external/
│  ├─ PGD/            baseline 1 (submodule)
│  └─ P2P-Bridge/     baseline 2 (submodule)
├─ configs/           common evaluation protocol, experiment settings
├─ src/               our code: data preparation, metrics, appearance-aware module
├─ notebooks/         analysis and figures
├─ results/           small result tables (CSV); raw outputs stay out of git
└─ data/              local only, not tracked
```

## Setup

```bash
git clone --recurse-submodules https://github.com/milanagm/L335_MiniProject.git
```

- Both baselines compile custom CUDA code and were tested with CUDA 11.8. They need an NVIDIA GPU (e.g. Colab) and do not run on Apple Silicon.
- `notebooks/L335MiniProject.ipynb` sets up both baselines on a Colab T4 and runs the Task 2 reproductions (copy of the Colab notebook, with outputs). Compiled extensions, data and checkpoints are cached in Google Drive (`MyDrive/L335_MiniProject/cache`), so a new session takes about 17 min.
- The repository is private: cloning it on Colab needs a GitHub access token.

## Common evaluation protocol (Task 2)

To be fixed before comparing: test shapes, point counts, noise realisations, normalisation and metric definitions. Known differences between the two papers so far:

- Noise levels: PGD reports 1 %, 2 % and 2.5 %; P2P-Bridge reports 1 %, 2 % and 3 %. The ScoreDenoise test data only contains noisy inputs at 1 %, 2 % and 3 %, so PGD's 2.5 % inputs are not public.
- P2M: for P2P-Bridge at 10K points and 1 % noise, its own paper reports 0.39, the PGD paper 0.686 (both ×10⁻⁴), while the Chamfer distances agree (2.28 vs. 2.284). The cause is one argument of PyTorch3D's `point_mesh_face_distance`: P2P-Bridge passes `min_triangle_area=0.0`, PGD keeps the default `0.005` (PyTorch3D 0.7.8), which treats small mesh triangles differently. Evaluating the same P2P-Bridge outputs (10K, 1 %) gives P2M 0.393 with `0.0` and 0.685 with `0.005`. Chamfer distance is not affected.
- Both repositories normalise prediction and ground truth to the unit sphere of the ground truth (ScoreDenoise procedure) and sum both directions of CD and P2M.

## Reproduction with documented settings (Task 2, step 1)

Run on a Colab T4 with the released checkpoints and the ScoreDenoise PU-Net test data (20 shapes per setting). Each method is evaluated with its own metric code.

- **PGD** (`results/pgd_documented_settings.csv`): `test.py` with `--niters 2` at 1 % and 2 % noise. Matches PGD's Table 1 to three decimals except at 50K / 1 % (CD 0.477 vs. 0.479, P2M 0.297 vs. 0.286). The README lists `--resolutions` and `--noise_lvls` separated by spaces, but `test.py` expects commas.
- **P2P-Bridge** (`results/p2pb_documented_settings.csv`): the README command `evaluate_objects.py --model_path ./pretrained/PVDS_PUNet/latest.pth --dataset PUNet` (default `--steps 5`). Its paper text says Table 1 used three denoising steps, so `--steps 3` was run as well. With `--steps 5` all 12 values match Table 1 within 0.01 (10 of them exactly when rounded); with `--steps 3` most values differ by 0.01 to 0.03. The published table therefore corresponds to the README setting. Runtime on a T4: about 25 min for `--steps 5` and 22 min for `--steps 3` (all six settings).
- **Per shape** (`results/per_shape_documented_settings.csv`): CD and P2M for each of the 20 test shapes in every setting above, taken from the evaluation logs, for the failure analysis. P2P-Bridge's log prints the raw values with six decimals, so its per-shape values have a resolution of 0.01 (×10⁻⁴); their means agree with the logged means within 0.002.
- **Raw outputs** (denoised point clouds as `.xyz`, an estimated 300 MB, plus the full logs) are in Google Drive under `MyDrive/L335_MiniProject/results/` and are not tracked by git; the notebook regenerates them.
