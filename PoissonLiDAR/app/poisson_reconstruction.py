from pathlib import Path
import struct

import numpy as np
import open3d as o3d


# ============================================================
# CONFIG
# ============================================================

BIN_PATH = Path("data/000009.bin")
LABEL_PATH = Path("data/000009.label")

OUTPUT_MESHBIN = Path("output/meshes/000009_poisson.meshbin")
OUTPUT_PLY = Path("output/meshes/000009_poisson.ply")

REDUCTION_CELL = 0.25

# ------------------------------------------------------------
# Local Poisson reconstruction
# ------------------------------------------------------------
# The LiDAR frame is reconstructed in overlapping XY chunks
# instead of one global Poisson surface.
CHUNK_SIZE = 20.0
CHUNK_OVERLAP = 3.0
MIN_POINTS_PER_CHUNK = 120

POISSON_DEPTH = 7
NORMAL_RADIUS = 1.0
NORMAL_MAX_NN = 30

# Geometry support constraints.
MAX_VERTEX_LIDAR_DISTANCE = 1.5
MAX_TRIANGLE_EDGE = 3.0
MAX_TRIANGLE_AREA = 8.0

# Remove only the lowest-density reconstructed vertices.
DENSITY_QUANTILE = 0.03


# ============================================================
# LOAD SEMANTICKITTI
# ============================================================

def load_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError("Invalid SemanticKITTI BIN file.")

    return raw.reshape(-1, 4)[:, :3]


def load_labels(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32)

    # SemanticKITTI semantic class = lower 16 bits.
    return (raw & 0xFFFF).astype(np.int32)


# ============================================================
# SPATIAL REDUCTION + SEMANTIC MAJORITY LABEL
# ============================================================

def reduce_points(
    points: np.ndarray,
    labels: np.ndarray,
    cell_size: float,
):
    cells = {}

    cells_xy = np.floor(points[:, :2] / cell_size).astype(np.int64)

    for i, (cx, cy) in enumerate(cells_xy):
        key = (int(cx), int(cy))

        if key not in cells:
            cells[key] = []

        cells[key].append(i)

    reduced_points = []
    reduced_labels = []

    for indices in cells.values():
        p = points[indices]

        reduced_points.append(p.mean(axis=0))

        unique, counts = np.unique(labels[indices], return_counts=True)
        reduced_labels.append(int(unique[np.argmax(counts)]))

    return (
        np.asarray(reduced_points, dtype=np.float32),
        np.asarray(reduced_labels, dtype=np.int32),
    )


# ============================================================
# SINGLE LOCAL POISSON RECONSTRUCTION
# ============================================================

def build_local_poisson(points: np.ndarray):
    if len(points) < MIN_POINTS_PER_CHUNK:
        return None

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN,
        )
    )

    try:
        cloud.orient_normals_consistent_tangent_plane(20)
    except RuntimeError:
        print(
            "Warning: local normal orientation failed; "
            "using estimated normals."
        )

    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud,
            depth=POISSON_DEPTH,
            width=0,
            scale=1.1,
            linear_fit=True,
        )
    )

    densities = np.asarray(densities)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None

    threshold = float(np.quantile(densities, DENSITY_QUANTILE))
    keep_vertex = densities >= threshold

    mesh.remove_vertices_by_mask(~keep_vertex)
    mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None

    return mesh


# ============================================================
# FILTER LOCAL POISSON GEOMETRY AGAINST LOCAL REAL LIDAR
# ============================================================

def constrain_local_mesh(
    mesh: o3d.geometry.TriangleMesh,
    lidar_points: np.ndarray,
):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(vertices) == 0 or len(triangles) == 0:
        return mesh

    lidar_cloud = o3d.geometry.PointCloud()
    lidar_cloud.points = o3d.utility.Vector3dVector(
        lidar_points.astype(np.float64)
    )

    kdtree = o3d.geometry.KDTreeFlann(lidar_cloud)

    distances = np.empty(len(vertices), dtype=np.float32)

    for i, vertex in enumerate(vertices):
        count, _, sq_distances = kdtree.search_knn_vector_3d(
            vertex,
            1,
        )

        if count == 0:
            distances[i] = np.inf
        else:
            distances[i] = np.sqrt(sq_distances[0])

    supported_vertex = distances <= MAX_VERTEX_LIDAR_DISTANCE

    keep_triangles = np.zeros(len(triangles), dtype=bool)

    for i, tri in enumerate(triangles):
        a, b, c = tri

        if not supported_vertex[a]:
            continue
        if not supported_vertex[b]:
            continue
        if not supported_vertex[c]:
            continue

        p0 = vertices[a]
        p1 = vertices[b]
        p2 = vertices[c]

        e0 = np.linalg.norm(p0 - p1)
        e1 = np.linalg.norm(p1 - p2)
        e2 = np.linalg.norm(p2 - p0)

        if e0 > MAX_TRIANGLE_EDGE:
            continue
        if e1 > MAX_TRIANGLE_EDGE:
            continue
        if e2 > MAX_TRIANGLE_EDGE:
            continue

        cross = np.cross(p1 - p0, p2 - p0)
        area = 0.5 * np.linalg.norm(cross)

        if area > MAX_TRIANGLE_AREA:
            continue

        keep_triangles[i] = True

    filtered = o3d.geometry.TriangleMesh()
    filtered.vertices = o3d.utility.Vector3dVector(vertices)
    filtered.triangles = o3d.utility.Vector3iVector(
        triangles[keep_triangles]
    )
    filtered.remove_unreferenced_vertices()

    return filtered


# ============================================================
# TRANSFER SEMANTIC LABELS
# ============================================================

def transfer_semantics(
    mesh: o3d.geometry.TriangleMesh,
    source_points: np.ndarray,
    source_labels: np.ndarray,
):
    vertices = np.asarray(mesh.vertices)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(
        source_points.astype(np.float64)
    )

    tree = o3d.geometry.KDTreeFlann(source)

    vertex_labels = np.zeros(len(vertices), dtype=np.int32)

    for i, vertex in enumerate(vertices):
        count, indices, _ = tree.search_knn_vector_3d(
            vertex,
            3,
        )

        if count == 0:
            continue

        nearby = source_labels[
            np.asarray(indices, dtype=np.int64)
        ]

        unique, counts = np.unique(nearby, return_counts=True)
        vertex_labels[i] = int(unique[np.argmax(counts)])

    return vertex_labels


# ============================================================
# SAVE P2DM V2
# ============================================================

def save_meshbin(
    path: Path,
    vertices: np.ndarray,
    labels: np.ndarray,
    triangles: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.write(b"P2DM")
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<I", len(vertices)))
        f.write(struct.pack("<I", len(triangles)))

        for vertex, label in zip(vertices, labels):
            f.write(
                struct.pack(
                    "<fffI",
                    float(vertex[0]),
                    float(vertex[1]),
                    float(vertex[2]),
                    int(label),
                )
            )

        for tri in triangles:
            f.write(
                struct.pack(
                    "<III",
                    int(tri[0]),
                    int(tri[1]),
                    int(tri[2]),
                )
            )


# ============================================================
# LOCAL CHUNK GENERATION
# ============================================================

def build_chunk_bounds(points: np.ndarray):
    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)

    start_x = np.floor(min_xy[0] / CHUNK_SIZE) * CHUNK_SIZE
    start_y = np.floor(min_xy[1] / CHUNK_SIZE) * CHUNK_SIZE

    end_x = np.ceil(max_xy[0] / CHUNK_SIZE) * CHUNK_SIZE
    end_y = np.ceil(max_xy[1] / CHUNK_SIZE) * CHUNK_SIZE

    x_values = np.arange(
        start_x,
        end_x + CHUNK_SIZE * 0.5,
        CHUNK_SIZE,
    )
    y_values = np.arange(
        start_y,
        end_y + CHUNK_SIZE * 0.5,
        CHUNK_SIZE,
    )

    return x_values, y_values


def reconstruct_local_chunks(
    reduced_points: np.ndarray,
    reduced_labels: np.ndarray,
):
    x_values, y_values = build_chunk_bounds(reduced_points)

    all_vertices = []
    all_triangles = []
    all_labels = []

    total_initial_triangles = 0
    total_after_density_triangles = 0
    total_final_triangles = 0

    chunk_number = 0
    usable_chunks = 0

    print()
    print("==================================================")
    print("LOCAL POISSON CHUNKS")
    print("==================================================")
    print("Chunk size   :", CHUNK_SIZE, "m")
    print("Overlap      :", CHUNK_OVERLAP, "m")

    for x0 in x_values:
        for y0 in y_values:
            x1 = x0 + CHUNK_SIZE
            y1 = y0 + CHUNK_SIZE

            # Expanded region provides context for the local Poisson
            # reconstruction.
            ex0 = x0 - CHUNK_OVERLAP
            ey0 = y0 - CHUNK_OVERLAP
            ex1 = x1 + CHUNK_OVERLAP
            ey1 = y1 + CHUNK_OVERLAP

            context_mask = (
                (reduced_points[:, 0] >= ex0)
                & (reduced_points[:, 0] < ex1)
                & (reduced_points[:, 1] >= ey0)
                & (reduced_points[:, 1] < ey1)
            )

            context_indices = np.flatnonzero(context_mask)

            chunk_number += 1

            if len(context_indices) < MIN_POINTS_PER_CHUNK:
                continue

            context_points = reduced_points[context_indices]
            context_labels = reduced_labels[context_indices]

            mesh = build_local_poisson(context_points)

            if mesh is None:
                continue

            initial_triangles = len(mesh.triangles)
            total_initial_triangles += initial_triangles

            # The density filter is performed inside build_local_poisson.
            total_after_density_triangles += initial_triangles

            mesh = constrain_local_mesh(
                mesh,
                context_points,
            )

            if len(mesh.triangles) == 0:
                continue

            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            # Keep only triangles whose centroid lies inside the core
            # chunk. This removes overlap duplicates while preserving
            # context at chunk boundaries.
            centroids = (
                vertices[triangles[:, 0]]
                + vertices[triangles[:, 1]]
                + vertices[triangles[:, 2]]
            ) / 3.0

            core_mask = (
                (centroids[:, 0] >= x0)
                & (centroids[:, 0] < x1)
                & (centroids[:, 1] >= y0)
                & (centroids[:, 1] < y1)
            )

            triangles = triangles[core_mask]

            if len(triangles) == 0:
                continue

            # Remove vertices that are no longer referenced after the
            # core-triangle selection.
            used = np.unique(triangles.reshape(-1))
            remap = -np.ones(len(vertices), dtype=np.int64)
            remap[used] = np.arange(len(used), dtype=np.int64)

            vertices = vertices[used]
            triangles = remap[triangles]

            local_mesh = o3d.geometry.TriangleMesh()
            local_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            local_mesh.triangles = o3d.utility.Vector3iVector(
                triangles.astype(np.int32)
            )

            local_labels = transfer_semantics(
                local_mesh,
                context_points,
                context_labels,
            )

            base = sum(len(v) for v in all_vertices)

            all_vertices.append(vertices.astype(np.float32))
            all_labels.append(local_labels.astype(np.int32))
            all_triangles.append(
                triangles.astype(np.int64) + base
            )

            total_final_triangles += len(triangles)
            usable_chunks += 1

            print(
                f"Chunk {chunk_number:02d}: "
                f"points={len(context_points):5d}, "
                f"triangles={len(triangles):5d}"
            )

    if not all_vertices:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 3), dtype=np.int64),
            total_initial_triangles,
            total_after_density_triangles,
            0,
            usable_chunks,
        )

    vertices = np.concatenate(all_vertices, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    triangles = np.concatenate(all_triangles, axis=0)

    return (
        vertices,
        labels,
        triangles,
        total_initial_triangles,
        total_after_density_triangles,
        total_final_triangles,
        usable_chunks,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("==================================================")
    print("LOCAL / CHUNKED POISSON LiDAR MESH")
    print("==================================================")

    points = load_bin(BIN_PATH)
    labels = load_labels(LABEL_PATH)

    if len(points) != len(labels):
        raise ValueError(
            f"Point/label mismatch: {len(points)} vs {len(labels)}"
        )

    print("Raw points :", len(points))
    print("Raw labels :", len(labels))

    reduced_points, reduced_labels = reduce_points(
        points,
        labels,
        REDUCTION_CELL,
    )

    print("Reduced points :", len(reduced_points))

    (
        vertices,
        vertex_labels,
        triangles,
        initial_triangles,
        density_triangles,
        final_triangles,
        usable_chunks,
    ) = reconstruct_local_chunks(
        reduced_points,
        reduced_labels,
    )

    if len(vertices) == 0 or len(triangles) == 0:
        raise RuntimeError(
            "No usable local Poisson geometry was generated."
        )

    print()
    print("==================================================")
    print("LOCAL POISSON MESH GENERATED")
    print("==================================================")
    print("Usable chunks          :", usable_chunks)
    print("Initial chunk triangles:", initial_triangles)
    print("After density          :", density_triangles)
    print("Final triangles        :", final_triangles)
    print("Final vertices         :", len(vertices))
    print("Semantic labels        :", len(vertex_labels))
    print("==================================================")

    # Build one Open3D mesh only for PLY inspection.
    output_mesh = o3d.geometry.TriangleMesh()
    output_mesh.vertices = o3d.utility.Vector3dVector(
        vertices.astype(np.float64)
    )
    output_mesh.triangles = o3d.utility.Vector3iVector(
        triangles.astype(np.int32)
    )
    output_mesh.compute_vertex_normals()

    OUTPUT_PLY.parent.mkdir(parents=True, exist_ok=True)

    o3d.io.write_triangle_mesh(
        str(OUTPUT_PLY),
        output_mesh,
        write_ascii=False,
    )

    save_meshbin(
        OUTPUT_MESHBIN,
        vertices,
        vertex_labels,
        triangles,
    )

    print()
    print("==================================================")
    print("OUTPUT")
    print("==================================================")
    print("PLY     :", OUTPUT_PLY)
    print("MESHBIN :", OUTPUT_MESHBIN)
    print("Format  : P2DM v2")
    print("==================================================")


if __name__ == "__main__":
    main()
