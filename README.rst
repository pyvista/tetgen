tetgen
======
.. image:: https://travis-ci.org/akaszynski/tetgen.svg?branch=master
    :target: https://travis-ci.org/akaszynski/tetgen

This python module is a python interface to Hang Si's `tetgen <https://github.com/ufz/tetgen>`__ C++ software. This module combines speed of C++ with the portability and ease of installation of Python along integration with the visualization toolkit  `VTK <https://www.vtk.org/>`__ .  See the `tetgen <https://github.com/ufz/tetgen>`__ GitHub page for more details on the creator.

The last update to the original C++ software was on 19 January 2011, but the software remains relevant today.  Brief description from `Weierstrass Institute Software <http://wias-berlin.de/software/index.jsp?id=TetGen&lang=1>`__:

TetGen is a program to generate tetrahedral meshes of any 3D polyhedral domains. TetGen generates exact constrained Delaunay tetrahedralization, boundary conforming Delaunay meshes, and Voronoi partitions.

TetGen provides various features to generate good quality and adaptive tetrahedral meshes suitable for numerical methods, such as finite element or finite volume methods. For more information of TetGen, please take a look at a list of `features <http://wias-berlin.de/software/tetgen/features.html>`__.


Installation
------------

From `PyPI <https://pypi.python.org/pypi/tetgen>`__

.. code:: bash

    pip install tetgen --user

From source at `GitHub <https://github.com/akaszynski/tetgen>`__

.. code:: bash

    git clone https://github.com/akaszynski/tetgen
    cd tetgen
    pip install . --user


Basic Example
-------------
The features of the C++ TetGen software implemented in this module are primarily focused on the tetrahedralization a manifold triangular surface.  This basic example demonstrates how to tetrahedralize a manifold surface and plot part of the mesh.

.. code:: python

    import pyvista as pv
    import tetgen
    import numpy as np

    sphere = pv.Sphere()
    tet = tetgen.TetGen(sphere)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    grid.plot()

.. figure:: https://github.com/akaszynski/tetgen/raw/master/docs/images/sphere.png
    :width: 300pt

    Tetrahedralized Sphere

Extract a portion of the sphere's tetrahedral mesh below the xy plane and plot the mesh quality.

.. code:: python

    # get cell centroids
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)

    # advanced plotting
    plotter = pv.Plotter()
    plotter.set_background('w')
    plotter.add_mesh(subgrid, 'lightgrey', lighting=True)
    plotter.add_mesh(sphere, 'r', 'wireframe')
    plotter.add_legend([[' Input Mesh ', 'r'],
                        [' Tesselated Mesh ', 'black']])
    plotter.plot()

.. image:: https://github.com/akaszynski/tetgen/raw/master/docs/images/sphere_subgrid.png

Cell quality scalars can be obtained and plotted with:

.. code:: python

    cell_qual = subgrid.quality

    # plot quality
    subgrid.plot(scalars=cell_qual, stitle='quality', cmap='bwr', flip_scalars=True)

.. image:: https://github.com/akaszynski/tetgen/raw/master/docs/images/sphere_qual.png


Acknowledgments
---------------
Software was originally created by Hang Si based on work published in `TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator <https://dl.acm.org/citation.cfm?doid=2629697>`__.  
