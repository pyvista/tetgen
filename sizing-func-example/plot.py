import pyvista as pv

mesh = pv.read("out/bar3.1.node")
mesh.plot(show_edges=True)
