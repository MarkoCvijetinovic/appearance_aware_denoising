
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_scores(xyz, rgb, k=24):
    # Find neighbours in clean 3D geometry.
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xyz)

    indices = knn.kneighbors(xyz, return_distance=False)[:, 1:]

    neighbours = xyz[indices]  # (N, k, 3)

    # Local geometric variation via PCA.
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)

    covariance = (np.einsum("nki,nkj->nij", centered, centered) / k)

    eigenvalues = np.linalg.eigvalsh(covariance)

    geometry_score = (eigenvalues[:, 0] / (eigenvalues.sum(axis=1) + 1e-12))

    # Local RGB variation.
    color_distances = np.linalg.norm(rgb[indices] - rgb[:, None, :], axis=2)

    color_score = color_distances.mean(axis=1)

    return geometry_score, color_score


def main(root, k):
    root = Path(root)
    objects = []
    all_g = []
    all_c = []

    for path in sorted(root.glob("*/**/clean.npz")):
        data = np.load(path)

        xyz = data["xyz"].astype(np.float64)
        rgb = data["rgb"].astype(np.float64)

        if len(xyz) <= k:
            continue

        g, c = compute_scores(xyz, rgb, k)

        objects.append((path.parent.name, g, c))
        all_g.append(g)
        all_c.append(c)

    if not objects:
        raise RuntimeError(f"No clean.npz files found in {root}")

    # Global thresholds, shared across all objects.
    # High = upper 10%; low = lower 50%.
    global_g = np.concatenate(all_g)
    global_c = np.concatenate(all_c)

    g_high = np.quantile(global_g, 0.90)
    c_high = np.quantile(global_c, 0.90)

    g_low = np.quantile(global_g, 0.50)
    c_low = np.quantile(global_c, 0.50)

    rows = []

    for object_id, g, c in objects:
        texture_only = np.mean((c >= c_high) & (g <= g_low))
        geometry_only = np.mean((g >= g_high) & (c <= c_low))
        aligned = np.mean((g >= g_high) & (c >= c_high))

        rows.append({
            "object_id": object_id,
            "texture_only": texture_only,
            "geometry_only": geometry_only,
            "aligned": aligned,
            "disagreement": texture_only + geometry_only,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("disagreement", ascending=False)

    output = root / "boundary_ranking.csv"
    df.to_csv(output, index=False)

    print(df.head(30).to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/gso")
    parser.add_argument("--k", type=int, default=24)
    args = parser.parse_args()

    main(args.root, args.k)
