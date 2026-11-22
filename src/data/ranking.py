
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

DATASET_COMPOSITION = (20, 10, 10, 20) # agreement, disagreement (geometry correct), disagreement (texture correct), random samples

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

def compute_thresholds(objects):
    """
    objects: dict mapping object_id -> (geometry_scores, colour_scores)
    """
    all_g = np.concatenate([g for g, c in objects.values()])
    all_c = np.concatenate([c for g, c in objects.values()])

    return {
        "g_high": np.quantile(all_g, 0.90),
        "g_low": np.quantile(all_g, 0.50),
        "c_high": np.quantile(all_c, 0.90),
        "c_low": np.quantile(all_c, 0.50),
    }


def summarize_object(g, c, thresholds):
    gh = thresholds["g_high"]
    gl = thresholds["g_low"]
    ch = thresholds["c_high"]
    cl = thresholds["c_low"]

    texture_only = np.mean((c >= ch) & (g <= gl))
    geometry_only = np.mean((g >= gh) & (c <= cl))
    aligned = np.mean((g >= gh) & (c >= ch))

    return {
        "texture_only": float(texture_only),
        "geometry_only": float(geometry_only),
        "aligned": float(aligned),
        "disagreement": float(texture_only + geometry_only),
    }

def select_objects(ranking, seed=2025, composition=DATASET_COMPOSITION):
    selected = set()
    groups = {}

    agreement, geometry_only, texture_only, random_samples = composition

    criteria = [
        ("agreement", "aligned", agreement),
        ("texture_only", "texture_only", texture_only),
        ("geometry_only", "geometry_only", geometry_only),
    ]

    for group, column, count in criteria:
        ranked = ranking.sort_values([column, "object_id"], ascending=[False, True])
        choices = [obj for obj in ranked["object_id"] if obj not in selected][:count]

        if len(choices) != count:
            raise ValueError(f"Not enough objects for {group}")

        groups[group] = choices
        selected.update(choices)

    remaining = ranking.loc[~ranking["object_id"].isin(selected), "object_id", ].to_numpy()

    rng = np.random.default_rng(seed)

    groups["random"] = list(rng.choice(remaining, size=random_samples, replace=False))

    return groups

def create_splits(groups, seed=2025):
    rng = np.random.default_rng(seed)
    assignments = {}

    for group, ids in groups.items():
        ids = list(ids)
        rng.shuffle(ids)

        n_train = int(len(ids) * 0.7)
        n_val = int(len(ids) * 0.1)

        for i, object_id in enumerate(ids):
            split = (
                "train" if i < n_train
                else "val" if i < n_train + n_val
                else "test"
            )

            assignments[object_id] = {"split": split, "group": group}

    return assignments

def rank_candidates(candidates):
    objects = {}

    for candidate in candidates:
        points, colors, meta = candidate
        object_id = str(meta["object_id"])

        objects[object_id] = compute_scores(points, colors)

    thresholds = compute_thresholds(objects)

    rows = [
            {"object_id": object_id,
             **summarize_object(g, c, thresholds)}
            for object_id, (g, c) in objects.items()
        ]
    
    ranking = pd.DataFrame(rows)
    return ranking