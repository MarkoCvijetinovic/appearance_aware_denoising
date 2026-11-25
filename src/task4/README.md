# Task 4 RGB denoising with frozen PGD

Open `notebooks/Task4_RGB_PGD_Colab.ipynb` on a Colab T4 and run all cells.
The notebook first checks two training objects, one validation object and one
test object at 2% XYZ noise. It then runs the full 42/6/12 split and all three
XYZ noise levels. Set `RUN_FULL = False` to stop after the short check. The dataset path points to the generated
Task 3 folder and can be changed in the first cell. Use a Drive account with
that folder and space for cached PGD predictions and checkpoints.

The notebook uses the Task 2 PGD revision and released checkpoint. It reuses
the team's CUDA environment or builds the PGD environment alone. The compiled
wheels and the exact environment package list are cached in Drive.

## Experiment

PGD runs with frozen weights in evaluation mode. It processes 1,000-point
patches and stitches the predictions back to the original point indices.
This keeps each output paired with the same RGB row. The patch stitching is
specific to this experiment, so compare models against the frozen PGD scores
from the same run. All variants use the same cached PGD predictions, nearest
neighbours, network width, trainable parameter count, random seed and budget.

- Geometry uses only XYZ and local geometric context.
- Plain RGB uses RGB differences between neighbours and one learned fusion
  weight per cloud.
- Gated RGB learns a different fusion weight for each point using its local
  geometry and colour context.

The heads start from the PGD prediction and learn three-coordinate
corrections. Training alternates the two RGB conditions. Validation chooses
the best checkpoint. Test scores report squared symmetric Chamfer distance
normalised by the radius of the clean cloud. Summaries aggregate by noise and
by the object groups in `selection.json`. The generated dataset provides
sampled clean points, so these outputs report Chamfer to the clean samples.

Each Drive run contains the three saved heads, `per_object.csv`, `summary.csv`
and `run.json` with the settings, selection checksum and training history.
The smoke run checks the pipeline on real data before fitting all objects.

## Mesh-supervised residual and GSO training-set comparison

The larger residual uses centred 8-, 16- and 32-neighbour features around
each frozen PGD prediction. Its training target is the displacement to the
closest point on the corresponding clean triangle mesh. The training target
uses the mesh; inference takes only the noisy point cloud, its frozen PGD
prediction and RGB when enabled. The original small, matched-capacity
Task 4 experiment is described above.

The 42- and 126-object training runs share a fresh 24-object validation set
and a fresh 42-object test set. Neither held-out set contains IDs from the
original 60-object split. The expanded training set includes the original 42
training objects. Both runs use 72 epochs of 42 optimizer updates, with
640 points from each of three clouds per update. In the expanded run, a
complete rotation over the training clouds takes three epochs. The default
training seed is 20261124. Add `--repeat-seed 20261125` to both training
commands to check sensitivity to initialisation and sampling. The 126-object runs also fit an RGB
control that holds the learned gate at 0.5. This control uses the same RGB
features, contrast multiplier and residual network as the learned-gate arm.

Set these shell variables to the original GSO export, its candidate cache,
the new supplemental dataset folder, the original full PGD run, the
existing PGD prediction cache, and the new result folder. Use the T4 and
the original PGD environment for the PGD step. The PGD source revision,
weights, Torch version and device must match the original run metadata.

```bash
python src/task4/expand_gso.py prepare \
  --data "$DATA" --candidates "$CANDIDATES" --extra "$EXTRA"
"$PGD_PYTHON" src/task4/expand_gso.py pgd \
  --extra "$EXTRA" --run "$PGD_RUN" --pgd "$PGD_REPO" --pgd-cache "$PGD_CACHE"
for count in 42 126; do
  python src/task4/surface_residual.py --data "$DATA" --extra "$EXTRA" \
    --train-objects "$count" --run "$PGD_RUN" --pgd-cache "$PGD_CACHE" \
    --candidates "$CANDIDATES" --mesh-cache "$MESH_CACHE" --out "$RESULTS/train_$count"
done
python src/task4/compare_gso_scale.py "$RESULTS"
```

Set `PGD_PYTHON` to the interpreter in the restored PGD environment.
The surface runtime also requires NumPy, PyTorch, pandas, SciPy, scikit-learn,
trimesh, point-cloud-utils and matplotlib. It downloads the original mesh
references from the revision in the Task 3 manifest. The object-level scores retain the split group for each test object.
Checkpoints and raw scores are
stored per seed beside each training protocol. The paired bootstrap averages
the three XYZ conditions and the available seeds per test object before
resampling the objects.
