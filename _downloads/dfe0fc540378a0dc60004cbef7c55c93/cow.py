"""
Cow
~~~

Tetrahedralize a cow mesh
"""
# sphinx_gallery_thumbnail_number = 3
import numpy as np
import pyvista as pv
from pyvista import examples
import tetgen
import pymeshfix

###############################################################################

cow_mesh = examples.download_cow().tri_filter()

cpos = [(13., 7.6, -13.85),
 (0.44, -0.4, -0.37),
 (-0.28, 0.9, 0.3)]

cow_mesh.plot(cpos=cpos)

###############################################################################

tet = tetgen.TetGen(cow_mesh)
tet.make_manifold()
tet.tetrahedralize()

cow_grid = tet.grid

###############################################################################

# plot half the cow
mask = np.logical_or(cow_grid.points[:, 0] < 0, cow_grid.points[:, 0] > 4)
half_cow = cow_grid.extract_selection_points(mask)

###############################################################################

plotter = pv.Plotter()
plotter.add_mesh(half_cow, color='w', show_edges=True)
plotter.add_mesh(cow_grid, color='r', style='wireframe', opacity=0.2)
plotter.camera_position = cpos
plotter.show()

###############################################################################
# Construct silly spinning cow animation

plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
plotter.open_gif('cow.gif')
plotter.add_mesh(half_cow, color='w', show_edges=True)
plotter.add_mesh(cow_grid, color='r', style='wireframe', opacity=0.2)
plotter.camera_position = cpos
plotter.write_frame()
nframe = 36
deg = 360./(nframe + 1)
for i in range(nframe):
    half_cow.rotate_y(deg)
    cow_grid.rotate_y(deg)
    plotter.update()
    plotter.write_frame()
plotter.close()
