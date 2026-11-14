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
- The repository is private: cloning it on Colab needs a GitHub access token.

## Common evaluation protocol (Task 2)

To be fixed before comparing: test shapes, point counts, noise realisations, normalisation and metric definitions. Known differences between the two papers so far:

- Noise levels: PGD reports 1 %, 2 % and 2.5 %; P2P-Bridge reports 1 %, 2 % and 3 %.
- P2M: for P2P-Bridge at 10K points and 1 % noise, its own paper reports 0.39, the PGD paper 0.686 (both ×10⁻⁴), while the Chamfer distances agree (2.28 vs. 2.284).
