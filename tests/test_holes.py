import pyvista as pv
import tetgen


def test_tetrahedralize_holes():
    cube = pv.Cube().triangulate()
    sphere = pv.Sphere(center=[0.0, 0.0, 0.0], theta_resolution=16, phi_resolution=16, radius=0.49)

    mesh = pv.merge([cube, sphere])

    tgen_full = tetgen.TetGen(mesh)
    nodes_full, elem_full = tgen_full.tetrahedralize(switches="pzq1.4")

    tgen_holes = tetgen.TetGen(mesh)
    tgen_holes.add_hole([0.0, 0.0, 0.0])

    nodes_holes, elem_holes = tgen_holes.tetrahedralize(switches="pzq1.4")

    assert nodes_full.shape > nodes_holes.shape
