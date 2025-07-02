import numpy as np
import pyvista as pv
import tetgen


def test_tetrahedralize_regions():

    airbox = pv.Cube(center=[0, 0, 0], x_length=1.5, y_length=0.5, z_length=0.5).triangulate()
    sphere1 = pv.Sphere(theta_resolution=16, phi_resolution=16, center=[-0.25, 0, 0], radius=0.1)
    sphere2 = pv.Sphere(theta_resolution=16, phi_resolution=16, center=[0.25, 0, 0], radius=0.1)
    mesh = pv.merge([sphere1, sphere2, airbox])

    tgen = tetgen.TetGen(mesh)

    V_sphere = 4 / 3 * 3.1415 * 0.1**3
    tgen.addRegion(100, [-0.25, 0, 0], V_sphere / 5000)  # sphere 1
    tgen.addRegion(200, [0.25 / 2, 0, 0], V_sphere / 5000)  # sphere 2
    tgen.addRegion(300, [1.5, 0, 0], V_sphere / 100)  # airbox

    _, elem, attrib = tgen.tetrahedralize(switches="pzq1.4Aa")

    assert attrib is not None
    assert len(attrib) == len(elem)

    regions = np.unique(attrib[:, 0])

    assert len(regions) == 3
