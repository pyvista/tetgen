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
    tgen.add_region(100, [-0.25, 0, 0], V_sphere / 500)  # sphere 1
    tgen.add_region(200, [0.25 / 2, 0, 0], V_sphere / 500)  # sphere 2
    tgen.add_region(300, [1.5, 0, 0], V_sphere / 10)  # airbox

    ret = tgen.tetrahedralize(switches="pzq1.4Aa")

    assert ret is not None
    assert len(ret) == 3

    attrib = ret[2]

    # get all the region IDs assigned from tetgen
    regions = np.unique(attrib[:, 0])

    assert len(regions) == 3
