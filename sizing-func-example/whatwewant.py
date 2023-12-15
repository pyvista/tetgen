from pathlib import Path
import tempfile
import numpy as np
import pyvista as pv
import tetgen

def sizing_function(points):
    """Return the target size at a given point.

    This is just an example. You can use any function you want.
    """
    x, y, z = points.T
    return np.where(x < 0, 0.5, 0.1)


def generate_background_mesh(boundary_mesh, resolution=20):
    """Generate a new mesh with the same bounds as the passed .poly file.

    We will use this as a background mesh for the sizing function.
    """
    bounds = boundary_mesh.bounds
    (xmin, xmax, ymin, ymax, zmin, zmax) = bounds
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


def write_background_mesh(background_mesh, outfile):
    """Write a background mesh to a file.

    This writes the mesh in tetgen format (X.b.node, X.b.ele) and a X.b.mtr file
    containing the target size for each node in the background mesh.
    """
    mtr_file = Path(outfile).with_suffix(".mtr")

    mtr_content = [f"{background_mesh.n_points} 1"]
    target_size = background_mesh.point_data["target_size"]
    for i in range(background_mesh.n_points):
        mtr_content.append(f"{target_size[i]}")

    pv.save_meshio(outfile, background_mesh)

    with open(mtr_file, "w") as f:
        f.write("\n".join(mtr_content))


def test_vanilla():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
    tet = tetgen.TetGen(sphere)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid

    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)

    # plot this
    plotter = pv.Plotter()
    plotter.set_background("w")
    plotter.add_mesh(subgrid, "lightgrey", lighting=True)
    plotter.add_mesh(grid, "r", "wireframe")
    plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
    plotter.show()


def test_sizing_function_file():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)

    # NEW: generate a background mesh and write it to a file
    bgmesh = generate_background_mesh(sphere)
    bgmesh.point_data["target_size"] = sizing_function(bgmesh.points)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir).joinpath("bgmesh.b.node")
        write_background_mesh(bgmesh, tmpfile)

        # NEW: pass the background mesh file to tetgen
        tet = tetgen.TetGen(sphere)
        tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5, background_mesh_file=str(tmpfile))
        grid = tet.grid

    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)

    # plot this
    plotter = pv.Plotter()
    plotter.set_background("w")
    plotter.add_mesh(subgrid, "lightgrey", lighting=True)
    plotter.add_mesh(grid, "r", "wireframe")
    plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
    plotter.show()


def test_sizing_function_direct():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)

    # NEW: generate a background mesh as a pyvista mesh
    bgmesh = generate_background_mesh(sphere)
    bgmesh.point_data["target_size"] = sizing_function(bgmesh.points)

    # NEW: pass the background mesh to tetgen
    tet = tetgen.TetGen(sphere)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5, background_mesh=bgmesh)
    grid = tet.grid

    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)

    # plot this
    plotter = pv.Plotter()
    plotter.set_background("w")
    plotter.add_mesh(subgrid, "lightgrey", lighting=True)
    plotter.add_mesh(grid, "r", "wireframe")
    plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
    plotter.show()


if __name__ == "__main__":
    test_vanilla()
    test_sizing_function_file()
    test_sizing_function_direct()