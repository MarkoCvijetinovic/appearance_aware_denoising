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
