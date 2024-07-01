"""Test the mesh resizing feature of tetgen with sizing function."""

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


def generate_background_mesh(boundary_mesh, resolution=20, eps=1e-6):
    """Generate a new mesh with the same bounds as the boundary meshy.

    We will use this as a background mesh for the sizing function.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = boundary_mesh.bounds
    new_vertices = np.meshgrid(
        np.linspace(xmin - eps, xmax + eps, resolution),
        np.linspace(ymin - eps, ymax + eps, resolution),
        np.linspace(zmin - eps, zmax + eps, resolution),
        indexing="ij",
    )

    # tetgen supports only tetrahedral meshes
    return pv.StructuredGrid(*new_vertices).triangulate()


def write_background_mesh(background_mesh, out_stem):
    """Write a background mesh to a file.

    This writes the mesh in tetgen format (X.b.node, X.b.ele) and a X.b.mtr file
    containing the target size for each node in the background mesh.
    """
    mtr_content = [f"{background_mesh.n_points} 1"]
    target_size = background_mesh.point_data["target_size"]
    for i in range(background_mesh.n_points):
        mtr_content.append(f"{target_size[i]:.8f}")

    pv.save_meshio(f"{out_stem}.node", background_mesh)
    mtr_file = f"{out_stem}.mtr"

    with open(mtr_file, "w") as f:
        f.write("\n".join(mtr_content))


def mesh_resizing_with_bgmeshfilename(mesh, bgmesh, **kwargs):
    """Performs mesh resizing with a background mesh file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir).joinpath("bgmesh.b")
        write_background_mesh(bgmesh, tmpfile)

        # Pass the background mesh file to tetgen
        tet = tetgen.TetGen(mesh)
        tet.tetrahedralize(bgmeshfilename=str(tmpfile), metric=1, **kwargs)
        grid = tet.grid

    # extract cells below the 0 xy plane
    # cell_center = grid.cell_centers().points
    # subgrid = grid.extract_cells(cell_center[:, 2] < 0)

    # debug: plot this
    # plotter = pv.Plotter()
    # plotter.set_background("w")
    # plotter.add_mesh(subgrid, "lightgrey", lighting=True)
    # plotter.add_mesh(grid, "r", "wireframe")
    # plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
    # plotter.show() # Uncomment for visualisation of resized mesh

    return grid


def mesh_resizing_with_pyvista_bgmesh(mesh, bgmesh, **kwargs):
    """Performs mesh resizing with a pyvista bgmesh."""
    # Pass the background mesh to tetgen
    tet = tetgen.TetGen(mesh)
    tet.tetrahedralize(bgmesh=bgmesh, metric=1, **kwargs)
    grid = tet.grid

    # Extract cells below the 0 xy plane
    # cell_center = grid.cell_centers().points
    # subgrid = grid.extract_cells(cell_center[:, 2] < 0)

    # Debug: uncomment for visualisation of resized mesh
    # plotter = pv.Plotter()
    # plotter.set_background("w")
    # plotter.add_mesh(subgrid, "lightgrey", lighting=True)
    # plotter.add_mesh(grid, "r", "wireframe")
    # plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
    # plotter.show()
    return grid


def test_mesh_resizing():
    """Test the mesh resizing feature of tetgen with sizing function."""
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
    tet_kwargs = dict(order=1, mindihedral=20, minratio=1.5)

    # Vanilla tetgen for reference
    tet = tetgen.TetGen(sphere)
    tet.tetrahedralize(**tet_kwargs)
    grid = tet.grid

    # Generate background mesh
    bgmesh = generate_background_mesh(sphere)
    bgmesh.point_data["target_size"] = sizing_function(bgmesh.points)

    resized_grid_file = mesh_resizing_with_bgmeshfilename(sphere, bgmesh, **tet_kwargs)
    assert resized_grid_file.n_points >= grid.n_points

    resized_grid_direct = mesh_resizing_with_pyvista_bgmesh(sphere, bgmesh, **tet_kwargs)
    assert resized_grid_direct.n_points > grid.n_points

    assert resized_grid_file == resized_grid_direct
