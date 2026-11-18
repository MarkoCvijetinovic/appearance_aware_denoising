from pathlib import Path
import numpy as np

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

        center = (clean.max(axis=0) + clean.min(axis=0)) / 2
        radius = np.linalg.norm(clean - center, axis=1).max()

        sigma = noise_level * radius

        noisy = clean + rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)

        np.savetxt(output_dir / path.name, noisy, fmt="%.8f")

        print(f"{resolution:15s} {path.stem:15s} " f"radius={radius:.4f}, sigma={sigma:.5f}")

print("Generated radius-scaled 2.5% noisy clouds.")