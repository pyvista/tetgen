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

cow_mesh = examples.download_cow().triangulate()

cpos = [(13., 7.6, -13.85),
        (0.44, -0.4, -0.37),
        (-0.28, 0.9, 0.3)]

cpos = [(15.87144235049248, 4.879216382405231, -12.14248864876951),
        (1.1623113035352375, -0.7609060338348953, 0.3192320579894903),
        (-0.19477922834083672, 0.9593375398915212, 0.20428542963665386)]

cow_mesh.plot(cpos=cpos)

###############################################################################

tet = tetgen.TetGen(cow_mesh)
tet.make_manifold()
tet.tetrahedralize()

cow_grid = tet.grid

###############################################################################

# plot half the cow
mask = np.logical_or(cow_grid.points[:, 0] < 0, cow_grid.points[:, 0] > 4)
half_cow = cow_grid.extract_points(mask)

###############################################################################

plotter = pv.Plotter()
plotter.add_mesh(half_cow, color='w', show_edges=True)
plotter.add_mesh(cow_grid, color='r', style='wireframe', opacity=0.2)
plotter.camera_position = cpos
plotter.show()

###############################################################################
# Make animation of the mesh construction

plotter = pv.Plotter(off_screen=True, window_size=[1000, 1000])
plotter.open_gif('cow.gif')
plotter.add_mesh(cow_grid, color='r', style='wireframe', opacity=0.2)
plotter.camera_position = cpos
plotter.write_frame()

nframe = 36
xb = np.array(cow_grid.bounds[0:2])
step = xb.ptp() / nframe
for val in np.arange(xb[0]+step, xb[1]+step, step):
    mask = np.argwhere(cow_grid.cell_centers().points[:, 0] < val)
    half_cow = cow_grid.extract_cells(mask)
    plotter.add_mesh(half_cow, color='w', show_edges=True, name='building')
    plotter.update()
    plotter.write_frame()

plotter.close()
