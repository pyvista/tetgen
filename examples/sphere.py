"""
Sphere
~~~~~~

Tetrahedralize a sphere
"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import tetgen
import numpy as np

###############################################################################
sphere = pv.Sphere()
tet = tetgen.TetGen(sphere)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid
grid.plot(show_edges=True)


###############################################################################

# get cell centroids
cells = grid.cells.reshape(-1, 5)[:, 1:]
cell_center = grid.points[cells].mean(1)

# extract cells below the 0 xy plane
mask = cell_center[:, 2] < 0
cell_ind = mask.nonzero()[0]
subgrid = grid.extract_cells(cell_ind)

# advanced plotting
plotter = pv.Plotter()
plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
plotter.add_mesh(sphere, 'r', 'wireframe')
plotter.add_legend([[' Input Mesh ', 'r'],
                    [' Tesselated Mesh ', 'black']])
plotter.show()

###############################################################################
# Use pyansys to compute cell quality

import pyansys
cell_qual = pyansys.quality(subgrid)

# plot quality
subgrid.plot(scalars=cell_qual, stitle='Quality', cmap='bwr', clim=[0, 1],
             flip_scalars=True, show_edges=True,)
