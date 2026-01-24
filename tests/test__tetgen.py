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
    mesh = pv.Icosphere(nsub=1)
    mesh.points = mesh.points.astype(np.float64)
    return mesh


def test_init() -> None:
    tgen = PyTetgen()
    assert not tgen.n_cells
    points_out = tgen.return_nodes()

    # ensure no memory errors
    del tgen
    gc.collect()
    assert points_out.shape == (0, 3)


def test_tetrahedralize_switches(
    sphere: PolyData, capfd: pytest.CaptureFixture[str]
) -> None:
    faces = sphere._connectivity_array.reshape(-1, 3).astype(np.int32)
    tgen = PyTetgen()
    tgen.load_mesh(sphere.points, faces)

    tgen.tetrahedralize_switches("p")
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

    tgen.tetrahedralize(minratio=1.5, mindihedral=60.0, order=1)
    ugrid = _to_ugrid(tgen.return_nodes(), tgen.return_tets())
    qual = ugrid.cell_quality()
    assert qual["scaled_jacobian"].mean() > 0.2
    assert (ugrid.celltypes == VTK_TETRA).all()

    # tgen = PyTetgen()
    # tgen.load_mesh(sphere.points, faces)
    # tgen.tetrahedralize(minratio=1.5, mindihedral=60.0, order=2)  # segfault here
    # points = tgen.return_nodes()
    # tets = tgen.return_tets()
    # ugrid = _to_ugrid(points, tets)
    # qual = ugrid.cell_quality()
    # assert qual["scaled_jacobian"].mean() > 0.2
    # assert (ugrid.celltypes == VTK_QUADRATIC_TETRA).all()
