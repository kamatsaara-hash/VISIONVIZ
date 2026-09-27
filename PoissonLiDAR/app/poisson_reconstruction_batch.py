from pathlib import Path
import struct

import numpy as np
import open3d as o3d


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
# Your screenshot shows frames directly inside PoissonLiDAR/data/
# (not inside data/00/). Support both layouts automatically.
DATA_ROOT = ROOT / "data"
SEQUENCE_DIR = DATA_ROOT / "00"

def count_matching_frames(folder: Path) -> int:
    """Count frames that have both a .bin and a matching .label file."""
    if not folder.is_dir():
        return 0
    return sum(1 for bin_file in folder.glob("*.bin")
               if bin_file.with_suffix(".label").is_file())

# Choose the folder that actually contains matching input pairs.
# This avoids selecting an existing-but-empty data/00 folder.
ROOT_FRAME_COUNT = count_matching_frames(DATA_ROOT)
SEQUENCE_FRAME_COUNT = count_matching_frames(SEQUENCE_DIR)

if SEQUENCE_FRAME_COUNT > ROOT_FRAME_COUNT:
    DATA_DIR = SEQUENCE_DIR
else:
    DATA_DIR = DATA_ROOT

OUTPUT_DIR = DATA_DIR / "poisson_meshes"

START_FRAME = 0
END_FRAME = 500

REDUCTION_CELL = 0.25
CHUNK_SIZE = 20.0
CHUNK_OVERLAP = 3.0
MIN_POINTS_PER_CHUNK = 120

POISSON_DEPTH = 7
NORMAL_RADIUS = 1.0
NORMAL_MAX_NN = 30

MAX_VERTEX_LIDAR_DISTANCE = 0.45
MAX_TRIANGLE_EDGE = 1.25
MAX_TRIANGLE_AREA = 0.75
DENSITY_QUANTILE = 0.08


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
    """Reduce in XYZ voxels, not XY columns, preserving vertical structure."""
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    voxel_keys = np.floor(points / cell_size).astype(np.int64)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)

    voxel_count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)

    reduced_points = np.column_stack([
        np.bincount(inverse, weights=points[:, axis], minlength=voxel_count)
        / counts
        for axis in range(3)
    ]).astype(np.float32)

    reduced_labels = np.zeros(voxel_count, dtype=np.int32)
    for voxel_id in range(voxel_count):
        voxel_labels = labels[inverse == voxel_id]
        unique, label_counts = np.unique(voxel_labels, return_counts=True)
        reduced_labels[voxel_id] = int(unique[np.argmax(label_counts)])

    return reduced_points, reduced_labels


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

            chunk_triangles = 0
            chunk_classes = 0

            # Reconstruct each semantic class independently. A single Poisson
            # surface over mixed classes tends to bridge road, vehicles,
            # vegetation, and buildings.
            class_ids, class_counts = np.unique(
                context_labels, return_counts=True
            )

            for semantic_id, class_count in zip(class_ids, class_counts):
                # Skip tiny class fragments; they cannot support a stable
                # local surface. Keep the threshold low for small objects.
                if int(class_count) < MIN_POINTS_PER_CHUNK:
                    continue

                class_mask = context_labels == semantic_id
                class_points = context_points[class_mask]
                if len(class_points) < MIN_POINTS_PER_CHUNK:
                    continue

                mesh = build_local_poisson(class_points)
                if mesh is None:
                    continue

                initial_triangles = len(mesh.triangles)
                total_initial_triangles += initial_triangles
                total_after_density_triangles += initial_triangles

                mesh = constrain_local_mesh(mesh, class_points)
                if len(mesh.triangles) == 0:
                    continue

                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)

                # Keep triangle centroids inside the non-overlapping core.
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

                used = np.unique(triangles.reshape(-1))
                remap = -np.ones(len(vertices), dtype=np.int64)
                remap[used] = np.arange(len(used), dtype=np.int64)
                vertices = vertices[used]
                triangles = remap[triangles]

                base = sum(len(v) for v in all_vertices)
                all_vertices.append(vertices.astype(np.float32))
                all_labels.append(
                    np.full(len(vertices), int(semantic_id), dtype=np.int32)
                )
                all_triangles.append(triangles.astype(np.int64) + base)

                chunk_triangles += len(triangles)
                total_final_triangles += len(triangles)
                chunk_classes += 1

            if chunk_triangles > 0:
                usable_chunks += 1
                print(
                    f"Chunk {chunk_number:02d}: "
                    f"points={len(context_points):5d}, "
                    f"classes={chunk_classes:2d}, "
                    f"triangles={chunk_triangles:5d}"
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
    print("=" * 60)
    print("BATCH SEMANTIC-AWARE LOCAL POISSON LiDAR RECONSTRUCTION")
    print(f"Frames: {START_FRAME:06d} through {END_FRAME:06d}")
    print(f"Input:  {DATA_DIR}")
    print(f"Matching .bin/.label pairs found: {count_matching_frames(DATA_DIR)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for frame_number in range(START_FRAME, END_FRAME + 1):
        # Accept common SemanticKITTI six-digit names and the five-digit
        # names shown in your current VS Code Explorer.
        candidates = [
            f"{frame_number:06d}",
            f"{frame_number:05d}",
            str(frame_number),
        ]
        input_stem = next(
            (
                stem for stem in candidates
                if (DATA_DIR / f"{stem}.bin").is_file()
                and (DATA_DIR / f"{stem}.label").is_file()
            ),
            None,
        )

        # Always write six-digit output names so the Godot playback script
        # can request frames such as 000009_poisson.meshbin.
        output_name = f"{frame_number:06d}"
        if input_stem is None:
            print(
                f"SKIP {output_name}: no matching .bin + .label pair "
                f"in {DATA_DIR} (tried {', '.join(candidates)})."
            )
            continue

        bin_path = DATA_DIR / f"{input_stem}.bin"
        label_path = DATA_DIR / f"{input_stem}.label"
        output_meshbin = OUTPUT_DIR / f"{output_name}_poisson.meshbin"
        output_ply = OUTPUT_DIR / f"{output_name}_poisson.ply"

        print("\n" + "=" * 60)
        print(f"PROCESSING FRAME {output_name}")
        print("=" * 60)

        try:
            points = load_bin(bin_path)
            labels = load_labels(label_path)
            if len(points) != len(labels):
                raise ValueError(
                    f"Point/label mismatch: {len(points)} vs {len(labels)}"
                )

            reduced_points, reduced_labels = reduce_points(
                points, labels, REDUCTION_CELL
            )
            print("Raw points:", len(points))
            print("Reduced points:", len(reduced_points))

            (
                vertices,
                vertex_labels,
                triangles,
                initial_triangles,
                density_triangles,
                final_triangles,
                usable_chunks,
            ) = reconstruct_local_chunks(reduced_points, reduced_labels)

            if len(vertices) == 0 or len(triangles) == 0:
                print(f"SKIP {output_name}: no usable Poisson geometry.")
                continue

            output_mesh = o3d.geometry.TriangleMesh()
            output_mesh.vertices = o3d.utility.Vector3dVector(
                vertices.astype(np.float64)
            )
            output_mesh.triangles = o3d.utility.Vector3iVector(
                triangles.astype(np.int32)
            )
            output_mesh.compute_vertex_normals()

            o3d.io.write_triangle_mesh(
                str(output_ply), output_mesh, write_ascii=False
            )
            save_meshbin(
                output_meshbin, vertices, vertex_labels, triangles
            )

            print(
                f"COMPLETE {output_name}: "
                f"vertices={len(vertices)}, triangles={len(triangles)}, "
                f"chunks={usable_chunks}"
            )
        except Exception as exc:
            print(f"ERROR {output_name}: {exc}")

    print("\nBatch Poisson reconstruction finished.")


if __name__ == "__main__":
    main()
