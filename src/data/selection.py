from pathlib import Path
import csv
import json
import numpy as np

def create_assignments(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    with open(args.catalogue, newline="") as f:
        catalogue = list(csv.DictReader(f))

    rng_selection = np.random.default_rng(args.seed)

    if args.selection:
        with open(args.selection) as f:
            selected = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    else:
        ids = [r["object_id"] for r in catalogue]
        if args.amount > len(ids):
            raise ValueError("Requested more objects than available")
        selected = list(rng_selection.choice(ids, size=args.amount, replace=False))

    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate object IDs in selection")

    # Assign whole objects to splits before generating samples.
    selected = sorted(selected)
    rng_selection.shuffle(selected)

    n = len(selected)
    n_train = int(n * args.train_fraction)
    n_val = int(n * args.val_fraction)

    assignments = {}
    for i, object_id in enumerate(selected):
        split = (
            "train" if i < n_train else
            "val" if i < n_train + n_val else
            "test"
        )
        assignments[object_id] = split

    # Save the exact selection and assignments.
    (root / "selection.json").write_text(json.dumps(assignments, indent=2))

    return assignments