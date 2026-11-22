
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def show_mesh(mesh):
    """Interactively display a textured mesh."""
    mesh.show()


def show_cloud(xyz, rgb, title="Point cloud"):
    """Interactively display XYZRGB points using Open3D."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([pcd], window_name=title)


def save_comparison(
    path, clean_xyz, noisy_xyz, clean_rgb, noisy_rgb
):
    """Save side-by-side clean and noisy point-cloud previews."""
    fig = plt.figure(figsize=(10, 5))

    clouds = [
        (clean_xyz, clean_rgb, "Clean XYZRGB"),
        (noisy_xyz, noisy_rgb, "Noisy XYZRGB"),
    ]

    for i, (xyz, rgb, title) in enumerate(clouds):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=0.2, rasterized=True)
        ax.set_title(title)
        ax.set_box_aspect(np.maximum(np.ptp(clean_xyz, axis=0), 1e-6))
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)