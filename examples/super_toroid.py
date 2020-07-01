"""
Super Toroid
~~~~~~~~~~~~

Tetrahedralize a super toroid surface
"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import tetgen
import numpy as np

###############################################################################
toroid = pv.ParametricSuperToroid()
tet = tetgen.TetGen(toroid)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid
grid.plot()


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
plotter.add_mesh(subgrid, color='lightgrey', lighting=True, show_edges=True)
plotter.add_mesh(toroid, color='r', style='wireframe')
plotter.add_legend([[' Input Mesh ', 'r'],
                    [' Tesselated Mesh ', 'black']])
plotter.plot()

###############################################################################
# Cell quality using pyansys
import pyansys
cell_qual = pyansys.quality(subgrid)

# plot quality
subgrid.plot(scalars=cell_qual, stitle='quality', cmap='bwr',  clim=[0,1],
             flip_scalars=True, show_edges=True)
