"""
Super Toroid
~~~~~~~~~~~~

Tetrahedralize a super toroid surface.

"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import tetgen
import numpy as np


###############################################################################
# Create and tetrahedralize a super torid.
#
# We merge the points here to make sure that the surface is manifold.

toroid = pv.ParametricSuperToroid(u_res=50, v_res=50, w_res=50).clean(tolerance=1E-9)
tet = tetgen.TetGen(toroid)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid
grid.plot()


###############################################################################
# Plot the tesselated mesh.

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
                    [' Tessellated Mesh ', 'black']])
plotter.show()

###############################################################################
# Show the cell quality

cell_qual = subgrid.compute_cell_quality()['CellQuality']
subgrid.plot(scalars=cell_qual, scalar_bar_args={'title': 'Cell Quality'},
             cmap='bwr',  clim=[0, 1],
             flip_scalars=True, show_edges=True)
