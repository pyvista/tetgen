import os

import pyvista as pv
import tetgen
import numpy as np

try:
    __file__
except:  # local testing
    __file__ = '/home/alex/afrl/python/source/tetgen/tests/test_tetgen.py'

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


def test_mesh_repair():
    cowfile = os.path.join(path, 'cow.ply')
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
    subgrid.plot(scalars=subgrid.quality, stitle='quality', cmap='bwr',
                 flip_scalars=True)

    # advanced plotting
    plotter = pv.Plotter()
    plotter.set_background('w')
    plotter.add_mesh(subgrid, 'lightgrey', lighting=True)
    plotter.add_mesh(grid, 'r', 'wireframe')
    plotter.add_legend([[' Input Mesh ', 'r'],
                       [' Tesselated Mesh ', 'black']])
    plotter.show()

    plotter = pv.Plotter()
    plotter.set_background('w')
    plotter.add_mesh(grid, 'r', 'wireframe')
    plotter.plot(auto_close=False, interactive_update=True)
    for i in range(500):
        single_cell = grid.extract_cells([i])
        plotter.add_mesh(single_cell)
        plotter.update()
    plotter.close()


