from pathlib import Path
import struct

import numpy as np
import open3d as o3d


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PLY_PATH = (
    ROOT
    / "output"
    / "meshes"
    / "000009_poisson.ply"
)

OUTPUT_PATH = (
    ROOT
    / "output"
    / "meshes"
    / "000009_poisson.meshbin"
)


# ============================================================
# BINARY FORMAT
# ============================================================

# 4-byte identifier.
MAGIC = b"P2DM"

# Format version.
VERSION = 1


# ============================================================
# LOAD PLY
# ============================================================

def load_mesh(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"PLY file not found:\n{path}"
        )

    print()
    print("=" * 60)
    print("LOADING POISSON MESH")
    print("=" * 60)

    print(
        f"Input: {path}"
    )

    mesh = o3d.io.read_triangle_mesh(
        str(path)
    )

    if mesh.is_empty():
        raise RuntimeError(
            "The PLY mesh is empty."
        )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float32
    )

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.uint32
    )

    print(
        f"Vertices : {len(vertices)}"
    )

    print(
        f"Triangles: {len(triangles)}"
    )

    return vertices, triangles


# ============================================================
# WRITE GODOT BINARY
# ============================================================

def write_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    output_path: Path
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("WRITING GODOT MESH")
    print("=" * 60)

    print(
        f"Output: {output_path}"
    )

    with open(
        output_path,
        "wb"
    ) as file:

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        file.write(
            MAGIC
        )

        file.write(
            struct.pack(
                "<I",
                VERSION
            )
        )

        file.write(
            struct.pack(
                "<I",
                len(vertices)
            )
        )

        file.write(
            struct.pack(
                "<I",
                len(triangles)
            )
        )

        # ----------------------------------------------------
        # Vertices
        #
        # Each vertex:
        #
        # float32 X
        # float32 Y
        # float32 Z
        #
        # 12 bytes
        # ----------------------------------------------------

        vertices = np.ascontiguousarray(
            vertices,
            dtype=np.float32
        )

        file.write(
            vertices.tobytes()
        )

        # ----------------------------------------------------
        # Triangle indices
        #
        # Each triangle:
        #
        # uint32 A
        # uint32 B
        # uint32 C
        #
        # 12 bytes
        # ----------------------------------------------------

        triangles = np.ascontiguousarray(
            triangles,
            dtype=np.uint32
        )

        file.write(
            triangles.tobytes()
        )

    print()
    print(
        "Mesh binary created successfully."
    )

    print(
        f"Vertices : {len(vertices)}"
    )

    print(
        f"Triangles: {len(triangles)}"
    )

    print(
        f"File size: "
        f"{output_path.stat().st_size / 1024:.2f} KB"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PLY → GODOT MESH CONVERTER")
    print("=" * 60)

    vertices, triangles = load_mesh(
        PLY_PATH
    )

    write_mesh(
        vertices,
        triangles,
        OUTPUT_PATH
    )

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()