from datasets import load_dataset
import io
import trimesh
import hashlib
import numpy as np
from trimesh.visual.color import uv_to_interpolated_color

from corruption import add_colour_noise, add_geometry_noise
from visualization import show_mesh, show_cloud

DATASET = "suvadityamuk/google-scanned-objects"

def stable_seed(seed, object_id, label):
    value = f"{seed}:{object_id}:{label}".encode()
    digest = hashlib.sha256(value).digest()
    return int.from_bytes(digest[:8], "little")

def load_glb(asset):
    if isinstance(asset, (trimesh.Trimesh, trimesh.Scene)):
        mesh = asset
    elif isinstance(asset, dict):
        if asset.get("bytes") is not None:
            mesh = trimesh.load(io.BytesIO(asset["bytes"]), file_type="glb", force="scene")
        else:
            mesh = trimesh.load(asset["path"], force="scene")
    else:
        mesh = trimesh.load(io.BytesIO(asset), file_type="glb", force="scene")

    # GLB may contain several separate mesh components.
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.dump() if isinstance(g, trimesh.Trimesh)]
    else:
        meshes = [mesh]

    return meshes


def sample_coloured_cloud(meshes, count, rng):
    areas = np.array([m.area for m in meshes])
    counts = rng.multinomial(count, areas / areas.sum())

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
    colors = np.concatenate(colors_list).astype(np.float32) / 255.0

    return points, colors

def prepare_candidates(count=10000, seed=2025):
    candidates = []

    ds = load_dataset(DATASET, split="train", streaming=True)

    for sample in ds:
        meta = sample["json"]
        object_id = str(meta["object_id"])

        rng = np.random.default_rng(stable_seed(seed, object_id, "surface"))

        meshes = load_glb(sample["glb"])
        xyz, rgb = sample_coloured_cloud(meshes, count, rng)

        candidates.append((xyz, rgb, meta))

    return candidates

# Running the code shows one sample of google scanned objects
if __name__ == "__main__":
    ds = load_dataset(DATASET, split="train", streaming=True)

    sample = next(iter(ds))
    print(sample["json"]["name"])

    meshes = load_glb(sample["glb"])
    show_mesh(meshes[0])

    rng = np.random.default_rng(2025)

    points, colors = sample_coloured_cloud(meshes, count=10000, rng=rng)

    show_cloud(points, colors, "Clean coloured point cloud")

    noisy_points = add_geometry_noise(points, 0.02, rng)
    noisy_colors = add_colour_noise(colors, 0.1, rng)

    show_cloud(
        noisy_points,
        noisy_colors,
        "Noisy geometry & noisy colours",
    )