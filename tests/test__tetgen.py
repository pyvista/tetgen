"""Test the nanobind extension."""

import os
import sys
import gc
from contextlib import contextmanager

from tetgen._tetgen import PyTetgen
import numpy as np
import pytest
import pyvista
from pyvista.core.pointset import PolyData

import os
import sys
from contextlib import contextmanager


import os
import sys
import ctypes
from contextlib import contextmanager


import os
import sys
import ctypes
from contextlib import contextmanager


@pytest.fixture
def sphere() -> PolyData:
    """Low density geodesic polyhedron."""
    mesh = pyvista.Icosphere(nsub=1)
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


def test_load_mesh(sphere: PolyData) -> None:
    faces = sphere._connectivity_array.reshape(-1, 3).astype(np.int32)
    tgen = PyTetgen()
    tgen.load_mesh(sphere.points, faces)
    points_out = tgen.return_nodes()
    assert np.allclose(points_out, sphere.points)

    # Should be different arrays
    sphere.points[:] += 1
    assert np.allclose(points_out + 1, sphere.points)

    # ensure no memory errors
    del tgen
    gc.collect()
    assert np.allclose(points_out + 1, sphere.points)


def test_tetrahedralize_switches(sphere: PolyData, capsys) -> None:
    faces = sphere._connectivity_array.reshape(-1, 3).astype(np.int32)
    tgen = PyTetgen()
    tgen.load_mesh(sphere.points, faces)

    tgen.tetrahedralize_switches("p")

    points = tgen.return_nodes()
    assert tgen.n_cells

    capsys.readouterr()
    captured = capsys.readouterr()
