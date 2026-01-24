"""Test the nanobind extension."""

import re
import os
import sys
import gc

from tetgen._tetgen import PyTetgen
from tetgen.pytetgen import _to_ugrid, VTK_TETRA, VTK_QUADRATIC_TETRA
import numpy as np
import pytest
import pyvista.core as pv
from pyvista.core.pointset import PolyData


@pytest.fixture
def sphere() -> PolyData:
    """Low density geodesic polyhedron."""
    mesh = pv.Icosphere(nsub=2)
    mesh.points = mesh.points.astype(np.float64)
    return mesh


def test_init(sphere: PolyData) -> None:
    tgen = PyTetgen()
    assert not tgen.n_cells
    points_out = tgen.return_nodes()
    assert points_out.shape == (0, 3)
    assert not tgen.n_faces

    with pytest.raises(RuntimeError, match="Facet marker count does not match"):
        facet_markers_in = np.ones(sphere.n_cells, dtype=np.int32)
        tgen.load_facet_markers(facet_markers_in)

    faces = sphere._connectivity_array.reshape(-1, 3).astype(np.int32)
    tgen.load_mesh(sphere.points, faces)
    assert tgen.n_faces == sphere.n_cells
    tgen.load_facet_markers(np.ones(sphere.n_cells, dtype=np.int32))

    facet_markers = tgen.return_facet_markers()
    assert (facet_markers == 1).all()


def test_tetrahedralize_switches(sphere: PolyData, capfd: pytest.CaptureFixture[str]) -> None:
    faces = sphere._connectivity_array.reshape(-1, 3).astype(np.int32)
    tgen = PyTetgen()
    tgen.load_mesh(sphere.points, faces)

    with pytest.raises(RuntimeError, match="Unable to parse switches"):
        tgen.tetrahedralize(switches_str="npa;slkdjf;aldsjfwp")

    out, err = capfd.readouterr()
    assert err

    tgen.tetrahedralize(switches_str="p")
    out, err = capfd.readouterr()
    assert not err

    matches = re.search(r"Mesh points:\s+(\d+)", out)
    assert matches
    n_points = int(matches.group(1))

    matches = re.search(r"Mesh tetrahedra:\s+(\d+)", out)
    assert matches
    n_cells = int(matches.group(1))

    assert tgen.n_cells == n_cells
    points = tgen.return_nodes()
    assert np.allclose(sphere.points, points)
    assert points.shape == (n_points, 3)
    tets = tgen.return_tets()
    assert tets.shape == (n_cells, 4)


def test_tetrahedralize(sphere: PolyData, capfd: pytest.CaptureFixture[str]) -> None:
    faces = sphere._connectivity_array.reshape(-1, 3).astype(np.int32)
    tgen = PyTetgen()
    tgen.load_mesh(sphere.points, faces)
    assert np.allclose(tgen.return_input_points(), sphere.points)
    assert np.allclose(tgen.return_input_faces(), faces)

    assert tgen.n_trifaces == 0

    tgen.tetrahedralize()
    tets = tgen.return_tets()
    assert tets.shape == (tgen.n_cells, 4)

    ugrid = _to_ugrid(tgen.return_nodes(), tets)
    qual = ugrid.cell_quality()
    assert qual["scaled_jacobian"].mean() > 0.1
    assert (ugrid.celltypes == VTK_TETRA).all()

    # test trifaces
    assert tgen.n_trifaces == ugrid.extract_surface().n_cells

    trifaces = tgen.return_trifaces()
    assert trifaces.shape == (tgen.n_trifaces, 3)
    assert trifaces.min() >= 0
    assert trifaces.max() < tgen.n_nodes

    # quadratic order appears to be broken in tetgen
    # tgen = PyTetgen()
    # tgen.load_mesh(sphere.points, faces)
    # tgen.tetrahedralize(minratio=1.5, mindihedral=60.0, order=2)  # segfault here
    # points = tgen.return_nodes()
    # tets = tgen.return_tets()
    # ugrid = _to_ugrid(points, tets)
    # qual = ugrid.cell_quality()
    # assert qual["scaled_jacobian"].mean() > 0.2
    # assert (ugrid.celltypes == VTK_QUADRATIC_TETRA).all()


def test_load_region(sphere: PolyData):
    faces = sphere._connectivity_array.reshape(-1, 3).astype(np.int32)

    tgen = PyTetgen()
    assert tgen.n_regions == 0

    with pytest.raises(RuntimeError, match="Region must be of size 5"):
        tgen.load_region([])
    assert tgen.n_regions == 0

    tgen.load_region([2.0, 0.0, 0.0, 0.0, 0.0])
    assert tgen.n_regions == 1

    tgen.load_region([3.0, 0.0, 0.0, 0.0, 0.0])
    assert tgen.n_regions == 2

    tgen.load_mesh(sphere.points, faces)
    tgen.tetrahedralize()
    assert not tgen.n_cell_attr
    assert tgen.return_tetrahedron_attributes().shape == (0, 0)
    tgen.tetrahedralize(regionattrib=True)
    assert tgen.n_cell_attr

    # expect single region
    attr = tgen.return_tetrahedron_attributes()
    assert attr.shape == (tgen.n_cells, tgen.n_cell_attr)
    assert np.allclose(attr, 1)


def test_load_hole(sphere: PolyData) -> None:
    tgen = PyTetgen()
    faces = sphere._connectivity_array.reshape(-1, 3).astype(np.int32)
    tgen.load_mesh(sphere.points, faces)

    with pytest.raises(RuntimeError, match="Hole must be of size 3"):
        tgen.load_hole([])

    tgen.load_hole([0.0, 0.0, 0.0])
    assert tgen.n_holes == 1

    tgen.tetrahedralize()

    # expecting no cells as the entire mesh is a "hole"
    assert not tgen.n_cells
