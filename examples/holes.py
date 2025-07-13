# """
# Holes
# ~~~~~~~~~~~~
# Example of using TetGen to create a tetrahedral mesh with holes.
# """

import pyvista as pv
import tetgen

cube = pv.Cube(x_length=2).triangulate()
sphere1 = pv.Sphere(center=[-0.5, 0.0, 0.0], theta_resolution=16, phi_resolution=16, radius=0.25)
sphere2 = pv.Sphere(center=[0.5, 0.0, 0.0], theta_resolution=16, phi_resolution=16, radius=0.1)

mesh = pv.merge([cube, sphere1, sphere2])

tgen = tetgen.TetGen(mesh)

# add the center of the spheres as holes
tgen.add_hole([-0.5, 0.0, 0.0]) 
tgen.add_hole([0.5, 0.0, 0.0]) 

nodes, elem = tgen.tetrahedralize(switches="pzq1.4")
grid = tgen.grid

grid.slice(normal='z').plot(show_edges=True, cpos="xy")
        
