import os
from pathlib import Path
import tempfile

import numpy as np
import pyvista as pv
import tetgen

path = os.path.dirname(os.path.abspath(__file__))


def test_load_arrays():
    sphere = pv.Sphere()
    v = sphere.points
    f = sphere.faces.reshape(-1, 4)[:, 1:]
    tet = tetgen.TetGen(v, f)


def test_vtk_tetrahedralize():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
    tet = tetgen.TetGen(sphere)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    assert grid.n_cells
    assert grid.n_points


def test_tetrahedralize_swithces():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
    tet = tetgen.TetGen(sphere)
    tet.tetrahedralize(switches="pq1.1/10YQ")
    grid = tet.grid
    assert grid.n_cells
    assert grid.n_points


def test_numpy_tetrahedralize(tmpdir):
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    f = np.vstack(
        [
            [0, 1, 2],
            [2, 3, 0],
            [0, 1, 5],
            [5, 4, 0],
            [1, 2, 6],
            [6, 5, 1],
            [2, 3, 7],
            [7, 6, 2],
            [3, 0, 4],
            [4, 7, 3],
            [4, 5, 6],
            [6, 7, 4],
        ]
    )

    tgen = tetgen.TetGen(v, f)

    nodes, elems = tgen.tetrahedralize()
    assert np.any(nodes)
    assert np.any(elems)

    # test save as well
    filename = str(tmpdir.mkdir("tmpdir").join("test_mesh.vtk"))
    tgen.write(filename)


def test_mesh_repair():
    cowfile = os.path.join(path, "cow.ply")
    tet = tetgen.TetGen(cowfile)
    tet.make_manifold()
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)


def functional_tet_example():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
    tet = tetgen.TetGen(sphere)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    assert grid.n_cells
    assert grid.n_points

    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)

    # plot this
    subgrid.plot(scalars=subgrid.quality, stitle="quality", cmap="bwr", flip_scalars=True)

    # advanced plotting
    plotter = pv.Plotter()
    plotter.set_background("w")
    plotter.add_mesh(subgrid, "lightgrey", lighting=True)
    plotter.add_mesh(grid, "r", "wireframe")
    plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
    plotter.show()

    plotter = pv.Plotter()
    plotter.set_background("w")
    plotter.add_mesh(grid, "r", "wireframe")
    plotter.plot(auto_close=False, interactive_update=True)
    for i in range(500):
        single_cell = grid.extract_cells([i])
        plotter.add_mesh(single_cell)
        plotter.update()
    plotter.close()


# Test the mesh resizing feature of tetgen with sizing function.


def sizing_function(points):
    """Return the target size at a given point.

    This is just an example. You can use any function you want.
    """
    x, y, z = points.T
    return np.where(x < 0, 0.5, 0.1)


def generate_background_mesh(boundary_mesh, resolution=20):
    """Generate a new mesh with the same bounds as the boundary meshy.

    We will use this as a background mesh for the sizing function.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = boundary_mesh.bounds
    eps = 1e-6
    new_vertices = np.meshgrid(
        np.linspace(xmin - eps, xmax + eps, resolution),
        np.linspace(ymin - eps, ymax + eps, resolution),
        np.linspace(zmin - eps, zmax + eps, resolution),
        indexing="ij",
    )

    # tetgen supports only tetrahedral meshes
    new_mesh = pv.StructuredGrid(*new_vertices).triangulate()
    return new_mesh


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
