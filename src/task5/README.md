# Colour alignment on the multiscale residual

Use the 126-object checkpoints and frozen PGD predictions from the [Task 4 replay](../task4/README.md). On the same held-out objects, compare uniform grey with a two-tone RGB boundary aligned to a real mesh crease or placed on a flat patch. Evaluate contrasts of 0.4 and 0.8 at all three XYZ noise levels. The learned and fixed gates use their existing checkpoints without retuning. The clean mesh locates and scores the test regions.

Run from the repository root with the Task 4 surface runtime. Use the same input paths and set `PROBE_OUT` to an output directory.

```bash
python src/task5/aligned_colour_probe.py \
  --data "$DATA" --extra "$EXTRA" --run "$PGD_RUN" \
  --model "$RESULTS/train_126" --pgd-cache "$PGD_CACHE" \
  --candidates "$CANDIDATES" --mesh-cache "$MESH_CACHE" \
  --out "$PROBE_OUT" --allow-separate
```

With `--allow-separate`, the probe can use a verified flat patch elsewhere on the same object. Both regions qualified on 24 of the 42 test objects in this run. The script writes `eligibility.csv`, `per_object.csv`, `means.csv`, `paired_effects.csv` and `protocol.json`.

At contrast 0.8, aligned colour lowered local squared surface error by 3.2% for the learned gate and 3.7% for the fixed weight, relative to uniform grey. On flat patches, the artificial boundary changed the predicted height difference across it. Mean step error against the clean surface stayed level or fell, and the paired results gave no clear advantage to the learned gate. The [rounded results](../../results/task5/gso_aligned_colour_probe_high_contrast.csv) report the induced change separately from error against the clean surface.
