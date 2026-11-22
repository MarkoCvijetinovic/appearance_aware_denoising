import numpy as np
from pathlib import Path

def bounding_radius(xyz):
    center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    return np.linalg.norm(xyz - center, axis=1).max()


def add_geometry_noise(xyz, level, rng):
    sigma = level * bounding_radius(xyz)
    return (xyz + rng.normal(0, sigma, size=xyz.shape)).astype(np.float32)


def add_colour_noise(rgb, level, rng):
    return np.clip(rgb + rng.normal(0, level, size=rgb.shape), 0, 1).astype(np.float32)


# Running the script generates 2.5% noise point clouds that were missing from data.zip
if __name__ == "main":
    rng = np.random.default_rng(2025)

    root = Path("data")
    noise_level = 0.025

    for resolution in ["10000_poisson", "50000_poisson"]:

        clean_dir = (root / "PUNet" / "pointclouds" / "test" / resolution)
        output_dir = (root / "examples" / f"PUNet_{resolution}_{noise_level}")

        # Preserve the previous experiment.
        if output_dir.exists():
            backup_dir = output_dir.with_name(output_dir.name + "_fixed_std_backup")

            if backup_dir.exists():
                raise FileExistsError(backup_dir)

            output_dir.rename(backup_dir)

        output_dir.mkdir(parents=True)

        for path in sorted(clean_dir.glob("*.xyz")):
            clean = np.loadtxt(path, dtype=np.float32)
            noisy = add_geometry_noise(clean, level=noise_level, rng=rng)
            np.savetxt(output_dir / path.name, noisy, fmt="%.8f")

    print("Generated radius-scaled 2.5% noisy clouds.")