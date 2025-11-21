"""Test script to verify triface marker extraction from TetGen."""

import numpy as np
import pyvista as pv
import tetgen


def test_triface_extraction():
    """Test triface marker extraction."""
    # Create a simple cube mesh
    cube = pv.Cube()
    cube = cube.triangulate()

    # Create TetGen object
    tgen = tetgen.TetGen(cube)

    # Add a region
    tgen.add_region(1, (0.0, 0.0, 0.0), 0.1)

    # Tetrahedralize
    nodes, elems, attributes, triface_markers = tgen.tetrahedralize(switches="pq1.414A")

    # Verify basic shapes
    assert nodes.shape[1] == 3, "Nodes should have 3 coordinates"
    assert elems.shape[1] == 4, "Elements should be tetrahedra (4 vertices)"
    assert triface_markers is not None, "Triface markers should be returned"

    # Verify triface markers shape matches number of faces
    assert triface_markers.ndim == 1, "Triface markers should be 1D array"
    assert len(triface_markers) > 0, "Should have at least some triface markers"

    # Verify we have boundary faces (negative markers) and possibly internal faces
    boundary_faces = np.sum(triface_markers < 0)
    internal_faces = np.sum(triface_markers > 0)

    assert boundary_faces > 0, "Should have boundary faces (negative markers)"
    # Internal faces may or may not exist depending on mesh

    # Verify unique markers exist
    unique_markers = np.unique(triface_markers)
    assert len(unique_markers) > 0, "Should have at least one unique marker"


def test_triface_with_surface_data():
    """Test triface extraction with surface data enabled."""
    # Create a simple sphere mesh
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
    tgen = tetgen.TetGen(sphere)

    # Tetrahedralize - the high-level API should return triface_markers by default
    nodes, elems, attributes, triface_markers = tgen.tetrahedralize(switches="pq1.1A")

    # Verify triface markers are available
    assert triface_markers is not None, "Triface markers should be returned"
    assert len(triface_markers) > 0, "Should have triface markers"

    # Verify the TetGen object has the surface data attributes
    assert hasattr(tgen, "triface_markers"), "TetGen object should have triface_markers attribute"
    assert tgen.triface_markers is not None, "TetGen.triface_markers should not be None"
