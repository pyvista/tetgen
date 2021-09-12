"""
Using external modules to create a mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tetrahedralize a sphere
"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import tetgen
import numpy as np

###############################################################################
# Using PyVista
# ~~~~~~~~~~~~~
# Create a surface mesh using ``pyvista`` and then tetrahedralize it.
sphere = pv.Sphere()
tet = tetgen.TetGen(sphere)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid
grid.plot(show_edges=True)


###############################################################################
# Use pyvista to plot

# get cell centroids
cell_center = grid.cell_centers().points

# extract cells below the 0 xy plane
mask = cell_center[:, 2] < 0
cell_ind = mask.nonzero()[0]
subgrid = grid.extract_cells(cell_ind)

# advanced plotting
plotter = pv.Plotter()
plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
plotter.add_mesh(sphere, 'r', 'wireframe')
plotter.add_legend([[' Input Mesh ', 'r'],
                    [' Tessellated Mesh ', 'black']])
plotter.show()

###############################################################################
# Using ``ansys-mapdl-reader``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Use pyansys's `legacy reader <https://github.com/pyansys/reader>`_
# library to compute cell quality.  This is the minimum scaled
# jacobian of each cell.

from ansys.mapdl.reader import quality
cell_qual = quality(subgrid)
print(f'Mean cell quality: {cell_qual.mean():.3}')

# plot quality
subgrid.plot(scalars=cell_qual, scalar_bar_args={'title': 'Quality'},
             cmap='bwr', clim=[0, 1], flip_scalars=True,
             show_edges=True,)


###############################################################################
# Using pyacvd
# ~~~~~~~~~~~~
#
# We can use `pyacvd <https://github.com/pyvista/pyacvd>`_ to create a
# more uniform mesh using the mesh generated from ``pyvista``.  We can
# use the ``pyacvd`` module to generate a more uniform surface mesh
# and then tetrahedralize that.
#
# Here we re-run the above example, except using a more uniform surface

import pyacvd

n_surf = 1000

clustered = pyacvd.Clustering(sphere)
clustered.subdivide(2)
clustered.cluster(n_surf)
uniform_surf = clustered.create_mesh()

# generate interior mesh
tet = tetgen.TetGen(uniform_surf)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
uniform_grid = tet.grid
uniform_grid.plot(show_edges=True)

cell_center = uniform_grid.cell_centers().points

# extract cells below the 0 xy plane
mask = cell_center[:, 2] < 0
cell_ind = mask.nonzero()[0]
subgrid = uniform_grid.extract_cells(cell_ind)

cell_qual = quality(subgrid)
print(f'Mean cell quality: {cell_qual.mean():.3}')

# plot quality
subgrid.plot(scalars=cell_qual, scalar_bar_args={'title': 'Quality'},
             cmap='bwr', clim=[0, 1], flip_scalars=True,
             show_edges=True,)
