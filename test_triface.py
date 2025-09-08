#!/usr/bin/env python3
"""
Test script to verify triface marker extraction from TetGen
"""

import numpy as np
import tetgen
import pyvista as pv

def test_triface_extraction():
    """Test triface marker extraction"""

    # Create a simple cube mesh
    cube = pv.Cube()
    cube = cube.triangulate()

    # Create TetGen object
    tgen = tetgen.TetGen(cube)

    # Add a region
    tgen.add_region(1, (0.0, 0.0, 0.0), 0.1)

    # Tetrahedralize
    nodes, elems, attributes, triface_markers = tgen.tetrahedralize(switches="pq1.414A")

    print(f"Nodes shape: {nodes.shape}")
    print(f"Elements shape: {elems.shape}")
    print(f"Attributes shape: {attributes.shape if attributes is not None else 'None'}")
    print(f"Triface markers shape: {triface_markers.shape if triface_markers is not None else 'None'}")

    if triface_markers is not None:
        print(f"Triface markers: {triface_markers}")
        print(f"Unique markers: {np.unique(triface_markers)}")

        # Count boundary faces (negative markers)
        boundary_faces = np.sum(triface_markers < 0)
        print(f"Boundary faces: {boundary_faces}")
        print(f"Internal faces: {np.sum(triface_markers > 0)}")

    return nodes, elems, attributes, triface_markers

if __name__ == "__main__":
    test_triface_extraction()
