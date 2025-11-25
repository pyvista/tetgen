"""
Holes
~~~~~~~~~~~~

Example of using TetGen to create a tetrahedral mesh with holes.

"""

import pyvista as pv
import tetgen

# create a cube with two spheres inside it
cube = pv.Cube(x_length=2).triangulate()
sphere1 = pv.Sphere(center=[-0.5, 0.0, 0.0], theta_resolution=16, phi_resolution=16, radius=0.25)
sphere2 = pv.Sphere(center=[0.5, 0.0, 0.0], theta_resolution=16, phi_resolution=16, radius=0.1)

# merge the geometries
mesh = pv.merge([cube, sphere1, sphere2])

tgen = tetgen.TetGen(mesh)

# add the center of the spheres as holes
tgen.add_hole([-0.5, 0.0, 0.0])
tgen.add_hole([0.5, 0.0, 0.0])

# tetrahedralize the mesh using TetGen
nodes, elem, _, _ = tgen.tetrahedralize(switches="pzq1.4")
grid = tgen.grid

# plot a slice to see the holes
grid.clip(normal="z").plot(show_edges=True)
