"""Tests for the ``.tetgen`` accessor registered on :class:`pyvista.PolyData`."""

from __future__ import annotations

import pytest
import pyvista as pv

import tetgen
import tetgen._accessor  # noqa: F401 — registers the ``.tetgen`` accessor

pytestmark = pytest.mark.skipif(
    not tetgen._accessor.HAS_ACCESSOR_REGISTRY,
    reason="requires pyvista >= 0.48 dataset accessor registry",
)


def test_accessor_attached_on_polydata():
    assert hasattr(pv.Sphere(), "tetgen")


def test_accessor_not_attached_on_non_polydata():
    # Grid/ImageData types should not expose a surface-mesh accessor.
    assert not hasattr(pv.ImageData(), "tetgen")
    assert not hasattr(pv.UnstructuredGrid(), "tetgen")


def test_accessor_cached_per_instance():
    sphere = pv.Sphere()
    assert sphere.tetgen is sphere.tetgen


def test_tetrahedralize_returns_unstructured_grid():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
    grid = sphere.tetgen.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells > 0
    assert grid.n_points > 0


def test_tetrahedralize_matches_direct_api():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
    direct = tetgen.TetGen(sphere)
    direct.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    direct_grid = direct.grid

    accessor_grid = sphere.tetgen.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    assert accessor_grid.n_cells == direct_grid.n_cells
    assert accessor_grid.n_points == direct_grid.n_points


def test_instance_lazy_and_cached():
    sphere = pv.Sphere()
    accessor = sphere.tetgen
    first = accessor.instance
    second = accessor.instance
    assert first is second
    assert isinstance(first, tetgen.TetGen)


def test_chains_with_core_filters():
    """Accessor result chains cleanly with a core PyVista filter."""
    grid = (
        pv.Sphere(theta_resolution=10, phi_resolution=10)
        .tetgen.tetrahedralize(order=1)
        .extract_cells([0, 1, 2])
    )
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells == 3


def test_registered_record_reports_tetgen_as_source():
    records = [r for r in pv.registered_accessors() if r.name == "tetgen"]
    assert len(records) == 1
    record = records[0]
    assert record.target is pv.PolyData
    assert record.source.startswith("tetgen._accessor")
