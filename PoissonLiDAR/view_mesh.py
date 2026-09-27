import open3d as o3d
import numpy as np
from pathlib import Path


MESH_PATH = Path("output/meshes/000009_poisson.ply")

# SemanticKITTI LiDAR / ego origin
EGO = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def make_ego_marker():
    geometries = []

    # Red sphere at the LiDAR origin.
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.translate(EGO)
    geometries.append(sphere)

    # Coordinate axes centered at ego.
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0,
        origin=EGO,
    )
    geometries.append(axes)

    # Forward direction: +X, matching the Godot view_yaw = 0 convention.
    forward = o3d.geometry.LineSet()
    forward.points = o3d.utility.Vector3dVector(
        np.array(
            [
                EGO,
                [15.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    forward.lines = o3d.utility.Vector2iVector([[0, 1]])
    forward.colors = o3d.utility.Vector3dVector(
        [[1.0, 0.0, 0.0]]
    )
    geometries.append(forward)

    return geometries


print("Loading:", MESH_PATH)

mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))

if mesh.is_empty():
    print("ERROR: Mesh is empty.")
    raise SystemExit(1)

print("Vertices :", len(mesh.vertices))
print("Triangles:", len(mesh.triangles))
print("Bounds   :", mesh.get_min_bound(), "to", mesh.get_max_bound())

mesh.compute_vertex_normals()

ego_geometries = make_ego_marker()

print()
print("EGO / LiDAR position : (0, 0, 0)")
print("Red sphere            : ego position")
print("XYZ axes              : ego coordinate frame")
print("Red line              : +X forward direction")
print()
print("Close the viewer window when finished.")

o3d.visualization.draw_geometries(
    [mesh] + ego_geometries,
    window_name="SemanticKITTI Poisson Mesh + Ego Position",
    width=1400,
    height=900,
    mesh_show_back_face=True,
)
