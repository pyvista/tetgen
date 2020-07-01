"""Python module to interface with wrapped TetGen C++ code
"""
import logging
import ctypes

import numpy as np
import pyvista as pv
import vtk

from tetgen import _tetgen

LOG = logging.getLogger(__name__)
LOG.setLevel('CRITICAL')


VTK9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9


invalid_input = TypeError('Invalid input.  Must be either a pyvista.PolyData\n' +
                          'object or vertex and face arrays')


class TetGen(object):
    """Input, clean, and tetrahedralize surface meshes using
    TetGen

    Parameters
    ----------
    args : str, :class:`pyvista.PolyData` or (``np.ndarray``, ``np.ndarray``)
        Either a pyvista surface mesh or a ``n x 3`` vertex array and ``n x 3`` face
        array.

    Examples
    --------
    Tetrahedralize a sphere using pyvista

    >>> import pyvista
    >>> import tetgen
    >>> sphere = pyvista.Sphere(theta_resolution=10, phi_resolution=10)
    >>> tgen = tetgen.TetGen(sphere)
    >>> nodes, elem = tgen.tetrahedralize()
    >>> tgen.grid.plot(show_edges=True)

    Tetrahedralize a cube using numpy arrays

    >>> import numpy as np
    >>> import tetgen
    >>> v = np.array([[0, 0, 0], [1, 0, 0],
                      [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1],
                      [1, 1, 1], [0, 1, 1],])
    >>> f = np.vstack([[0, 1, 2], [2, 3, 0],
                       [0, 1, 5], [5, 4, 0],
                       [1, 2, 6], [6, 5, 1],
                       [2, 3, 7], [7, 6, 2],
                       [3, 0, 4], [4, 7, 3],
                       [4, 5, 6], [6, 7, 4]])
    >>> tgen = tetgen.TetGen(v, f)
    >>> nodes, elems = tgen.tetrahedralize()
    """
    _updated = None

    def __init__(self, *args):
        """ initializes MeshFix using a mesh """
        self.v = None
        self.f = None
        self.node = None
        self.elem = None
        self._grid = None

        def parse_mesh(mesh):
            if not mesh.is_all_triangles:
                raise RuntimeError('Invalid mesh.  Must be an all triangular mesh')

            self.v = mesh.points
            faces = mesh.faces
            self.f = np.ascontiguousarray(faces.reshape(-1, 4)[:, 1:])

        if not args:
            raise invalid_input
        elif isinstance(args[0], pv.PolyData):
            parse_mesh(args[0])
        elif isinstance(args[0], (np.ndarray, list)):
            self._load_arrays(args[0], args[1])
        elif isinstance(args[0], str):
            mesh = pv.read(args[0])
            parse_mesh(mesh)
        else:
            raise invalid_input

    def _load_arrays(self, v, f):
        """Loads triangular mesh from vertex and face arrays

        Face arrays/lists are v and f.  Both vertex and face arrays
        should be 2D arrays with each vertex containing XYZ data and
        each face containing three points.
        """
        # Check inputs
        if not isinstance(v, np.ndarray):
            try:
                v = np.asarray(v, np.float)
                if v.ndim != 2 and v.shape[1] != 3:
                    raise Exception(
                        'Invalid vertex format.  Shape should be (npoints, 3)')
            except BaseException:
                raise Exception(
                    'Unable to convert vertex input to valid numpy array')

        if not isinstance(f, np.ndarray):
            try:
                f = np.asarray(f, ctypes.c_int)
                if f.ndim != 2 and f.shape[1] != 3:
                    raise Exception(
                        'Invalid face format.  Shape should be (nfaces, 3)')
            except BaseException:
                raise Exception(
                    'Unable to convert face input to valid numpy array')

        # Store to self
        self.v = v
        self.f = f

    def make_manifold(self, verbose=False):
        """Reconstruct a manifold clean surface from input mesh.
        Updates mesh in-place.

        Requires pymeshfix

        Parameters
        ----------
        verbose : bool, optional
            Controls output printing.  Default False.
        """
        try:
            import pymeshfix
        except ImportError:
            raise ImportError('pymeshfix not installed.  Please run:\n'
                              'pip install pymeshfix')

        # Run meshfix
        meshfix = pymeshfix.MeshFix(self.v, self.f)
        meshfix.repair(verbose)

        # overwrite this object with cleaned mesh
        self.v = meshfix.v
        self.f = meshfix.f

    def plot(self, **kwargs):
        """Displays input mesh

        See help(:func:`pyvista.plot`) for available arguments.
        """
        self.mesh.plot(**kwargs)

    @property
    def mesh(self):
        """Return the surface mesh"""
        triangles = np.empty((self.f.shape[0], 4))
        triangles[:, -3:] = self.f
        triangles[:, 0] = 3
        return pv.PolyData(self.v, triangles, deep=False)

    def tetrahedralize(self,
                       psc=0,
                       refine=0,
                       quality=1,
                       nobisect=True,
                       coarsen=0,
                       metric=0,
                       weighted=0,
                       brio_hilbert=1,
                       incrflip=0,
                       flipinsert=0,
                       varvolume=0,
                       fixedvolume=0,
                       noexact=0,
                       nostaticfilter=0,
                       insertaddpoints=0,
                       regionattrib=0,
                       cdtrefine=0,
                       diagnose=0,
                       convex=0,
                       zeroindex=0,
                       facesout=0,
                       edgesout=0,
                       neighout=0,
                       voroout=0,
                       meditview=0,
                       vtkview=0,
                       nobound=0,
                       nonodewritten=1,
                       noelewritten=1,
                       nofacewritten=1,
                       noiterationnum=0,
                       nomergefacet=0,
                       nomergevertex=0,
                       nojettison=0,
                       docheck=0,
                       quiet=0,
                       verbose=0,
                       vertexperblock=4092,
                       tetrahedraperblock=8188,
                       shellfaceperblock=4092,
                       nobisect_nomerge=1,
                       supsteiner_level=2,
                       addsteiner_algo=1,
                       coarsen_param=0,
                       weighted_param=0,
                       fliplinklevel=-1,
                       flipstarsize=-1,
                       fliplinklevelinc=1,
                       reflevel=3,
                       optscheme=7,
                       optlevel=2,
                       delmaxfliplevel=1,
                       order=2,
                       reversetetori=0,
                       steinerleft=10000,
                       no_sort=0,
                       hilbert_order=52,
                       hilbert_limit=8,
                       brio_threshold=64,
                       brio_ratio=0.125,
                       facet_separate_ang_tol=179.9,
                       facet_overlap_ang_tol=0.001,
                       facet_small_ang_tol=15.0,
                       maxvolume=-1.0,
                       minratio=2.0,
                       mindihedral=0.0,
                       optmaxdihedral=165.0,
                       optminsmtdihed=179.0,
                       optminslidihed=179.0,
                       epsilon=1.0e-8,
                       coarsen_percent=1.0,
                       switches=None):
        """Generates tetrahedrals interior to the surface mesh
        described by the vertex and face arrays already loaded.
        Returns nodes and elements belonging to the all tetrahedral
        mesh.

        The tetrahedral generator uses the C++ library TetGen and can
        be configured by either using a string of ``switches`` or by
        changing the underlying behavior using optional inputs.

        Should the user desire more control over the mesh
        tetrahedralization or wish to control the tetrahedralization
        in a more pythonic manner, use the optional inputs rather than
        inputting switches.

        Parameters
        ----------
        facet_overlap_ang_tol : double, optional
            Threshold angle at which TetGen will consider to faces
            overlapping.  Raising this will require a higher quality
            mesh input and may cause tetrahedralize to fail.  Default
            0.001.

        quality : bool, optional
            Enables/disables mesh improvement.  Enabled by default.
            Disable this to speed up mesh generation while sacrificing
            quality.  Default True.

        minratio : double, optional.
            Maximum allowable radius-edge ratio.  Must be greater than
            1.0 the closer to 1.0, the higher the quality of the mesh.
            Be sure to raise steinerleft to allow for the addition of
            points to improve the quality of the mesh.  Avoid overly
            restrictive requirements, otherwise, meshing will appear
            to hang.  Default 2.0

            Testing has showed that 1.1 is a reasonable input for a
            high quality mesh.

        mindihedral : double, optional
            Minimum allowable dihedral angle.  The larger this number,
            the higher the quality of the resulting mesh.  Be sure to
            raise steinerleft to allow for the addition of points to
            improve the quality of the mesh.  Avoid overly restrictive
            requirements, otherwise, meshing will appear to hang.
            Default 0.0

            Testing has shown that 10 is a reasonable input

        verbose : int, optional
            Controls the underlying TetGen library to output text to
            console.  Users using iPython will not see this output.
            Setting to 1 enables some information about the mesh
            generation while setting verbose to 2 enables more debug
            output.  Default 0, or no output.

        nobisect : bool, optional
            Controls if Steiner points are added to the input surface
            mesh.  When enabled, the surface mesh will be modified.
            Default False.

            Testing has shown that if your input surface mesh is
            already well shaped, disabling this setting will improve
            meshing speed and mesh quality.

        steinerleft : int, optional
            Steiner points are points added to the original surface
            mesh to create a valid tetrahedral mesh.  Settings this to
            -1 will allow tetgen to create an unlimited number of
            steiner points, but the program will likely hang if this
            is used in combination with narrow quality requirements.
            Default 100000.

            The first type of Steiner points are used in creating an
            initial tetrahedralization of PLC. These Steiner points
            are mandatory in order to create a valid
            tetrahedralization

            The second type of Steiner points are used in creating
            quality tetra- hedral meshes of PLCs. These Steiner points
            are optional, while they may be necessary in order to
            improve the mesh quality or to conform the size of mesh
            elements.

        double : optmaxdihedral, optional
            Setting unreachable using switches.  Controls the optimial
            maximum dihedral.  Settings closer, but not exceeding, 180
            degrees results in a lower quality mesh.  Should be
            between 135 and 180 degrees.  Default 165.0

        order : int optional
            Controls whether TetGen creates linear tetrahedrals or
            quadradic tetrahedrals.  Set order to 2 to output
            quadradic tetrahedrals.  Default 2.

        Examples
        --------
        The following switches "pq1.1/10Y" would be:

        >>> node, elem = tgen.tetrahedralize(nobisect=True, quality=True,
                                             minratio=1.1, mindihedral=10)

        Using the switches option:

        >>> node, elem = tgen.tetrahedralize(switches="pq1.1/10Y")

        Notes
        -----
        There are many other options and the TetGen documentation
        contains descritpions only for the switches of the original
        C++ program.  This is the relationship between tetgen switches
        and python optinal inputs:

        - ``psc`` --> ``'-s'``
        - ``refine`` --> ``'-r'``
        - ``quality`` --> ``'-q'``
        - ``nobisect`` --> ``'-Y'``
        - ``coarsen`` --> ``'-R'``
        - ``weighted`` --> ``'-w'``
        - ``brio_hilbert`` --> ``'-b'``
        - ``incrflip`` --> ``'-l'``
        - ``flipinsert`` --> ``'-L'``
        - ``metric`` --> ``'-m'``
        - ``varvolume`` --> ``'-a'``
        - ``fixedvolume`` --> ``'-a'``
        - ``regionattrib`` --> ``'-A'``
        - ``cdtrefine`` --> ``'-D'``
        - ``insertaddpoints`` --> ``'-i'``
        - ``diagnose`` --> ``'-d'``
        - ``convex`` --> ``'-c'``
        - ``nomergefacet`` --> ``'-M'``
        - ``nomergevertex`` --> ``'-M'``
        - ``noexact`` --> ``'-X'``
        - ``nostaticfilter`` --> ``'-X'``
        - ``zeroindex`` --> ``'-z'``
        - ``voroout`` --> ``'-v'``
        - ``meditview`` --> ``'-g'``
        - ``vtkview`` --> ``'-k'``
        - ``nobound`` --> ``'-B'``
        - ``noiterationnum`` --> ``'-I'``
        - ``nojettison`` --> ``'-J'``
        - ``docheck`` --> ``'-C'``
        - ``quiet`` --> ``'-Q'``
        - ``verbose`` --> ``'-V'``
        - ``vertexperblock`` --> ``'-x', 4092.``
        - ``tetrahedraperblock`` --> ``'-x', 8188.``
        - ``shellfaceperblock`` --> ``'-x', 2044.``
        - ``nobisect_nomerge`` --> ``'-Y', 1.``
        - ``supsteiner_level`` --> ``'-Y/', 2.``
        - ``addsteiner_algo`` --> ``'-Y//', 1.``
        - ``coarsen_param`` --> ``'-R', 0.``
        - ``weighted_param`` --> ``'-w', 0.``
        - ``reflevel`` --> ``'-D', 3.``
        - ``optlevel`` --> ``'-O', 2.``
        - ``optscheme`` --> ``'-O', 7.``
        - ``order`` --> ``'-o', 1.``
        - ``reversetetori`` --> ``'-o/', 0.``
        - ``steinerleft`` --> ``'-S', 0.``
        - ``hilbert_order`` --> ``'-b///', 52.``
        - ``hilbert_limit`` --> ``'-b//'  8.``
        - ``brio_threshold`` --> ``'-b' 64.``
        - ``brio_ratio`` --> ``'-b/' 0.125.``
        - ``facet_separate_ang_tol`` --> ``'-p', 179.9.``
        - ``facet_overlap_ang_tol`` --> ``'-p/',  0.1.``
        - ``facet_small_ang_tol`` --> ``'-p//', 15.0.``
        - ``maxvolume`` --> ``'-a', -1.0.``
        - ``minratio`` --> ``'-q', 0.0.``
        - ``mindihedral`` --> ``'-q', 5.0.``
        - ``optmaxdihedral`` --> ``165.0.``
        - ``optminsmtdihed`` --> ``179.0.``
        - ``optminslidihed`` --> ``179.0.``
        - ``epsilon`` --> ``'-T', 1.0e-8.``
        - ``coarsen_percent`` --> ``-R1/#, 1.0.``
        """

        # format switches
        if switches is None:
            switches_str = b''
        else:
            switches_str = bytes(switches, 'utf-8')

        # check verbose switch
        if verbose == 0:
            quiet = 1

        # Call libary
        plc = True  # always true
        try:
            self.node, self.elem = _tetgen.Tetrahedralize(self.v,
                                                          self.f,
                                                          switches_str,
                                                          plc,
                                                          psc,
                                                          refine,
                                                          quality,
                                                          nobisect,
                                                          coarsen,
                                                          metric,
                                                          weighted,
                                                          brio_hilbert,
                                                          incrflip,
                                                          flipinsert,
                                                          varvolume,
                                                          fixedvolume,
                                                          noexact,
                                                          nostaticfilter,
                                                          insertaddpoints,
                                                          regionattrib,
                                                          cdtrefine,
                                                          diagnose,
                                                          convex,
                                                          zeroindex,
                                                          facesout,
                                                          edgesout,
                                                          neighout,
                                                          voroout,
                                                          meditview,
                                                          vtkview,
                                                          nobound,
                                                          nonodewritten,
                                                          noelewritten,
                                                          nofacewritten,
                                                          noiterationnum,
                                                          nomergefacet,
                                                          nomergevertex,
                                                          nojettison,
                                                          docheck,
                                                          quiet,
                                                          verbose,
                                                          vertexperblock,
                                                          tetrahedraperblock,
                                                          shellfaceperblock,
                                                          nobisect_nomerge,
                                                          supsteiner_level,
                                                          addsteiner_algo,
                                                          coarsen_param,
                                                          weighted_param,
                                                          fliplinklevel,
                                                          flipstarsize,
                                                          fliplinklevelinc,
                                                          reflevel,
                                                          optscheme,
                                                          optlevel,
                                                          delmaxfliplevel,
                                                          order,
                                                          reversetetori,
                                                          steinerleft,
                                                          no_sort,
                                                          hilbert_order,
                                                          hilbert_limit,
                                                          brio_threshold,
                                                          brio_ratio,
                                                          facet_separate_ang_tol,
                                                          facet_overlap_ang_tol,
                                                          facet_small_ang_tol,
                                                          maxvolume,
                                                          minratio,
                                                          mindihedral,
                                                          optmaxdihedral,
                                                          optminsmtdihed,
                                                          optminslidihed,
                                                          epsilon,
                                                          coarsen_percent)
        except RuntimeError as e:
            raise RuntimeError('Failed to tetrahedralize.\n' +
                               'May need to repair surface by making it manifold:\n' +
                               str(e))

        # check if a mesh was generated
        if not np.any(self.node):
            raise RuntimeError('Failed to tetrahedralize.\n' +
                               'May need to repair surface by making it manifold')

        # Return nodes and elements
        LOG.info('Generated mesh with %d nodes and %d elements', self.node.shape[0],
                 self.elem.shape[0])
        self._updated = True

        return self.node, self.elem

    @property
    def grid(self):
        """ Returns a :class:`pyvista.UnstructuredGrid` """
        if self.node is None:
            raise RuntimeError('Run Tetrahedralize first')

        if self._grid is not None and not self._updated:
            return self._grid

        buf = np.empty((self.elem.shape[0], 1), pv.ID_TYPE)
        cell_type = np.empty(self.elem.shape[0], dtype='uint8')
        if self.elem.shape[1] == 4:  # linear
            buf[:] = 4
            cell_type[:] = 10
        elif self.elem.shape[1] == 10:  # quadradic
            buf[:] = 10
            cell_type[:] = 24
        else:
            raise Exception('Invalid element array shape %s' % str(self.elem.shape))

        cells = np.hstack((buf, self.elem))
        if VTK9:
            self._grid = pv.UnstructuredGrid(cells, cell_type, self.node)
        else:
            offset = np.cumsum(buf + 1) - (buf[0] + 1)
            self._grid = pv.UnstructuredGrid(offset, cells, cell_type, self.node)

        self._updated = False
        return self._grid

    def write(self, filename, binary=False):
        """Writes an unstructured grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written.  The file extension will
            select the type of writer to use.

            - ``".vtk"`` will use the vtk legacy writer
            - ``".vtu"`` will select the VTK XML writer
            - ``".cdb"`` will write an ANSYS APDL archive file

        binary : bool, optional
            Writes as a binary file by default.  Set to False to write
            ASCII.  Ignored when output is a cdb.

        Examples
        --------
        >>> tgen.write('grid.vtk', binary=True)

        Notes
        -----
        Binary files write much faster than ASCII, but binary files
        written on one system may not be readable on other systems.
        Binary can be used only with the legacy writer.
        """
        self.grid.save(filename, binary)
