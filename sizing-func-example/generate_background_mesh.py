from pathlib import Path

import pyvista as pv


def get_bounds(polyfile):
    """Get the bounds of a .poly file (tetgen format)."""
    bounds = [(float("inf"), -float("inf")) for _ in range(3)]
    vertex_num = None
    vertex_counter = 0

    with open(polyfile, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            components = line.split()
            if not vertex_num:
                assert len(components) == 2
                assert components[1] == "3"
                vertex_num = int(components[0])
                assert vertex_num > 0
                continue

            if vertex_counter == vertex_num:
                break

            assert len(components) == 4
            x, y, z = map(float, components[1:])
            bounds[0] = (min(bounds[0][0], x), max(bounds[0][1], x))
            bounds[1] = (min(bounds[1][0], y), max(bounds[1][1], y))
            bounds[2] = (min(bounds[2][0], z), max(bounds[2][1], z))
            vertex_counter += 1

    return bounds


def generate_background_mesh(polyfile, resolution=20):
    """Generate a new mesh with the same bounds as the passed .poly file.

    We will use this as a background mesh for the sizing function.
    """
    bounds = get_bounds(polyfile)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    eps = 1e-6
    new_vertices = np.meshgrid(
        *[
            np.linspace(xmin - eps, xmax + eps, resolution),
            np.linspace(ymin - eps, ymax + eps, resolution),
            np.linspace(zmin - eps, zmax + eps, resolution),
        ],
        indexing="ij",
    )

    # tetgen supports only tetrahedral meshes
    new_mesh = pv.StructuredGrid(*new_vertices).triangulate()
    return new_mesh


def write_background_mesh(polyfile, sizing_function):
    """Write a background mesh to a file.

    This writes the mesh in tetgen format (X.b.node, X.b.ele) a X.b.mtr file
    containing the target size for each note in the background mesh.

    Args:
        polyfile (str): Path to the .poly file (tetgen format) to use as a reference.
            Only used to get the bounds of the mesh.
        sizing_function (callable): Function that takes a point (x, y, z) and returns
            the target size for the background mesh at that point.
    """
    outfile = Path(polyfile).with_suffix(".b.node")
    mtr_file = Path(polyfile).with_suffix(".b.mtr")

    mesh = generate_background_mesh(polyfile)

    mtr_content = [f"{mesh.n_points} 1"]
    for point in mesh.points:
        mtr_content.append(f"{sizing_function(point)}")

    pv.save_meshio(outfile, mesh)

    with open(mtr_file, "w") as f:
        f.write("\n".join(mtr_content))


if __name__ == "__main__":
    import sys
    import numpy as np

    def sizing_function(point):
        """Return the target size at a given point.

        This is just an example. You can use any function you want.
        """
        x, y, z = point
        return np.where(x < 0.5, 0.5, 0.1)

    polyfile = sys.argv[1]
    write_background_mesh(polyfile, sizing_function)
