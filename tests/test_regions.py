import numpy as np
import pyvista as pv
import tetgen

r = 0.1  # 100mm
L = 0.5  # 500mm, distance of centers
# result: 6.99pF in air


airbox = pv.Cube(center=[0, 0, 0], x_length=3 * L, y_length=L, z_length=L).triangulate()
sphere1 = pv.Sphere(theta_resolution=16, phi_resolution=16, center=[-L / 2, 0, 0], radius=r)
sphere2 = pv.Sphere(theta_resolution=16, phi_resolution=16, center=[L / 2, 0, 0], radius=r)
mesh = pv.merge([sphere1, sphere2, airbox])

plotter = pv.Plotter(shape=(1, 2))

plotter.add_mesh(mesh, show_edges=True, opacity=0.5, label="Input Mesh")
plotter.add_text("Input Mesh", font_size=10)
plotter.subplot(0, 1)


plotter.add_text("Tetrahedralized Mesh", font_size=10)

tgen = tetgen.TetGen(mesh)

V_sphere = 4 / 3 * np.pi * r**3

tgen.addRegion(100, [-L / 2, 0, 0.0], V_sphere / 1000)
tgen.addRegion(200, [L / 2, 0, 0.0], V_sphere / 1000)
tgen.addRegion(300, [L * 1.5, 0, 0.0], V_sphere / 100)

nodes, elem, attrib = tgen.tetrahedralize(switches="pzq1.4Aa")

grid = tgen.grid

regions = np.unique(attrib[:, 0])

for reg_id in regions:
    name = f"Region {reg_id}"
    reg_mask = attrib[:, 0] == reg_id
    subgrid = grid.extract_cells(reg_mask)

    if reg_id == 100:
        color = "red"
    elif reg_id == 200:
        color = "blue"
    else:
        color = "white"

    if reg_id < 300:
        opacity = 1.0
    else:
        opacity = 0.25

    plotter.add_mesh(subgrid, show_edges=True, color=color, opacity=opacity, label=name)

plotter.add_legend()


plotter.link_views()
plotter.show()

exit()


grid.plot(show_edges=True, scalars=colors, opacity=0.5)

# get cell centroids
cells = grid.cells.reshape(-1, 5)[:, 1:]
cell_center = grid.points[cells].mean(1)

# extract cells below the 0 xy plane
mask = cell_center[:, 2] < 0
cell_ind = mask.nonzero()[0]
subgrid = grid.extract_cells(cell_ind)

cell_qual = subgrid.compute_cell_quality()["CellQuality"]

# advanced plotting
plotter = pv.Plotter()
plotter.add_mesh(
    subgrid,
    lighting=True,
    show_edges=True,
    scalars=cell_qual,
    cmap="bwr",
    clim=[0, 1],
    flip_scalars=True,
)

plotter.add_mesh(mesh, "r", "wireframe")
plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
plotter.show()

# subgrid.plot(scalars=cell_qual, cmap='bwr', clim=[0, 1], flip_scalars=True, show_edges=True)

# slices = tgen.grid.slice_orthogonal()

# slices.plot(show_edges=True)
