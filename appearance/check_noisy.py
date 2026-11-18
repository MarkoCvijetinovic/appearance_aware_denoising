import numpy as np
from pathlib import Path
root = Path("data")

clean = np.loadtxt("data/PUNet/pointclouds/test/10000_poisson/Icosahedron.xyz")
noisy = np.loadtxt("data/examples/PUNet_10000_poisson_0.02/Icosahedron.xyz")

center = (clean.max(axis=0) + clean.min(axis=0)) / 2
radius = np.linalg.norm(clean - center, axis=1).max()

print("Clean cloud bounding radius:", radius)
print("Empirical noise standard deviation:", np.std(noisy - clean))
print("Noise std / radius:", np.std(noisy - clean) / radius)

for res in ["10000_poisson", "50000_poisson"]:
    clean_dir = root / "PUNet/pointclouds/test" / res

    print(f"\nResolution: {res}")

    for noise in ["0.02", "0.025"]:
        noisy_dir = root / "examples" / f"PUNet_{res}_{noise}"

        stds = []
        shape_stds = []

        for path in sorted(clean_dir.glob("*.xyz")):
            noisy_path = noisy_dir / path.name
            if not noisy_path.exists():
                continue

            clean = np.loadtxt(path)
            noisy = np.loadtxt(noisy_path)

            if clean.shape != noisy.shape:
                print("Shape mismatch:", path.name)
                continue

            residual = noisy - clean
            stds.append(residual.std())
            shape_stds.append((path.stem, residual.std()))

        if stds:
            print(
                f"Noise {noise}: "
                f"mean std={np.mean(stds):.6f}, "
                f"min={np.min(stds):.6f}, "
                f"max={np.max(stds):.6f}"
            )

            print("Largest std:", sorted(shape_stds, key=lambda x: -x[1])[:3])

res = "50000_poisson"

for path in sorted((root / "PUNet/pointclouds/test" / res).glob("*.xyz")):
    clean = np.loadtxt(path)
    noisy = np.loadtxt(root / "examples" / f"PUNet_{res}_0.02" / path.name)

    center = (clean.max(axis=0) + clean.min(axis=0)) / 2
    radius = np.linalg.norm(clean - center, axis=1).max()
    std = np.std(noisy - clean)

    print(
        f"{path.stem:15s} "
        f"radius={radius:.4f} "
        f"noise_std={std:.5f} "
        f"ratio={std / radius:.5f}"
    )