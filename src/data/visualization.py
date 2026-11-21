from datasets import load_dataset
import matplotlib.pyplot as plt
import io
import trimesh
import open3d as o3d
import numpy as np

from corruption import add_colour_noise, add_geometry_noise

ds = load_dataset("suvadityamuk/google-scanned-objects", split="train", streaming=True)

sample = next(iter(ds))

print(sample["json"]["name"])
print(sample["json"]["category"])
print(sample.keys())

asset = sample["glb"]

if isinstance(asset, (trimesh.Trimesh, trimesh.Scene)):
    mesh = asset
elif isinstance(asset, dict):
    if asset.get("bytes") is not None:
        mesh = trimesh.load(io.BytesIO(asset["bytes"]), file_type="glb", force="scene")
    else:
        mesh = trimesh.load(asset["path"], force="scene")
else:
    mesh = trimesh.load(io.BytesIO(asset), file_type="glb", force="scene")

mesh.show()

from trimesh.visual.color import uv_to_interpolated_color

# GLB may contain several separate mesh components.
if isinstance(mesh, trimesh.Scene):
    meshes = [g for g in mesh.dump() if isinstance(g, trimesh.Trimesh)]
else:
    meshes = [mesh]

rng = np.random.default_rng(2025)
n_points = 10000

# Distribute sampled points according to each mesh's surface area.
areas = np.array([m.area for m in meshes])
counts = rng.multinomial(n_points, areas / areas.sum())

points_list = []
colors_list = []

for geom, count in zip(meshes, counts):
    if count == 0:
        continue

    # Sample XYZ positions and corresponding barycentric coordinates.
    points, face_indices, bary = trimesh.sample.sample_surface(
        geom, count, return_barycentric=True, seed=int(rng.integers(0, 2**31)),
    )

    visual = geom.visual

    if isinstance(visual, trimesh.visual.TextureVisuals):
        # Interpolate the UV coordinates of each sampled point.
        uv_vertices = visual.uv[geom.faces[face_indices]]
        uv = np.einsum("nij,ni->nj", uv_vertices, bary)

        material = visual.material

        # Support both traditional and GLB/PBR texture materials.
        texture = getattr(material, "image", None)
        if texture is None:
            texture = getattr(material, "baseColorTexture", None)

        if texture is None:
            raise ValueError("No colour texture found for mesh component")

        colors = uv_to_interpolated_color(uv, texture)[:, :3]

    elif visual.kind == "vertex":
        vertex_colors = visual.vertex_colors[geom.faces[face_indices], :3]
        colors = np.einsum("nij,ni->nj", vertex_colors.astype(float), bary)

    elif visual.kind == "face":
        colors = visual.face_colors[face_indices, :3]

    else:
        raise ValueError("Mesh component has no usable colour data")

    points_list.append(points)
    colors_list.append(colors)

points = np.concatenate(points_list)
colors = np.concatenate(colors_list).astype(np.float64) / 255.0

print("Points:", points.shape)
print("Colours:", colors.shape)
print("RGB range:", colors.min(), colors.max())

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Clean coloured point cloud"
)

noise_level_points = 0.02
noise_level_colors = 0.1

noisy_colors = add_colour_noise(colors, noise_level_colors, rng)
noisy_points = add_geometry_noise(points, noise_level_points, rng)

pcd_noisy = o3d.geometry.PointCloud()
pcd_noisy.points = o3d.utility.Vector3dVector(noisy_points)
pcd_noisy.colors = o3d.utility.Vector3dVector(noisy_colors)

o3d.visualization.draw_geometries([pcd_noisy], window_name="Noisy geometry & noisy colours")
