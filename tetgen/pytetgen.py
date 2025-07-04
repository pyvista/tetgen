"""Python module to interface with wrapped TetGen C++ code."""

import ctypes
import logging
from pathlib import Path

import numpy as np
import pyvista as pv
from tetgen import _tetgen

LOG = logging.getLogger(__name__)
LOG.setLevel("CRITICAL")

MTR_POINTDATA_KEY = "target_size"

invalid_input = TypeError(
    "Invalid input. Must be either a pyvista.PolyData object or vertex and face arrays."
)


class TetGen:
    """Input, clean, and tetrahedralize surface meshes using TetGen.

    Parameters
    ----------
    args : str | pyvista.PolyData | numpy.ndarray
        Either a pyvista surface mesh or a ``n x 3`` vertex array and ``n x 3`` face
        array.

    Examples
    --------
    Tetrahedralize a sphere using pyvista.

    >>> import pyvista
    >>> import tetgen
    >>> sphere = pyvista.Sphere(theta_resolution=10, phi_resolution=10)
    >>> tgen = tetgen.TetGen(sphere)
    >>> nodes, elem = tgen.tetrahedralize()
    >>> tgen.grid.plot(show_edges=True)

    Tetrahedralize a cube using numpy arrays.

    >>> import numpy as np
    >>> import tetgen
    >>> v = np.array(
    ...     [
    ...         [0, 0, 0],
    ...         [1, 0, 0],
    ...         [1, 1, 0],
    ...         [0, 1, 0],
    ...         [0, 0, 1],
    ...         [1, 0, 1],
    ...         [1, 1, 1],
    ...         [0, 1, 1],
    ...     ]
    ... )
    >>> f = np.vstack(
    ...     [
    ...         [0, 1, 2],
    ...         [2, 3, 0],
    ...         [0, 1, 5],
    ...         [5, 4, 0],
    ...         [1, 2, 6],
    ...         [6, 5, 1],
    ...         [2, 3, 7],
    ...         [7, 6, 2],
    ...         [3, 0, 4],
    ...         [4, 7, 3],
    ...         [4, 5, 6],
    ...         [6, 7, 4],
    ...     ]
    ... )
    >>> tgen = tetgen.TetGen(v, f)
    >>> nodes, elems = tgen.tetrahedralize()
    """

    _updated = None

    def __init__(self, *args):
        """Initialize MeshFix using a mesh or arrays."""
        self.v = None
        self.f = None
        self.node = None
        self.elem = None
        self.attributes = None
        self._grid = None

        self.regions = {}

        def parse_mesh(mesh):
            if not mesh.is_all_triangles:
                raise RuntimeError("Invalid mesh. Must be an all triangular mesh")

            self.v = mesh.points
            faces = mesh.faces
            self.f = np.ascontiguousarray(faces.reshape(-1, 4)[:, 1:])

        if not args:
            raise invalid_input
        elif isinstance(args[0], pv.PolyData):
            parse_mesh(args[0])
        elif isinstance(args[0], (np.ndarray, list)):
            self._load_arrays(args[0], args[1])
        elif isinstance(args[0], (str, Path)):
            mesh = pv.read(args[0])
            parse_mesh(mesh)
        else:
            raise invalid_input

    def _load_arrays(self, v, f):
        """Load triangular mesh from vertex and face arrays.

        Face arrays/lists are v and f. Both vertex and face arrays
        should be 2D arrays with each vertex containing XYZ data and
        each face containing three points.
        """
        # Check inputs
        if not isinstance(v, np.ndarray):
            try:
                v = np.asarray(v, np.float)
                if v.ndim != 2 and v.shape[1] != 3:
                    raise Exception("Invalid vertex format. Shape should be (npoints, 3)")
            except BaseException:
                raise Exception("Unable to convert vertex input to valid numpy array")

        if not isinstance(f, np.ndarray):
            try:
                f = np.asarray(f, ctypes.c_int)
                if f.ndim != 2 and f.shape[1] != 3:
                    raise Exception("Invalid face format. Shape should be (nfaces, 3)")
            except BaseException:
                raise Exception("Unable to convert face input to valid numpy array")

        if v.shape[0] < 4:
            raise ValueError(
                f"The vertex array should contain at least 4 points. Found {v.shape[0]}."
            )

        # Store to self
        self.v = v
        self.f = f

    def add_region(
        self, id: int, point_in_region: tuple[float, float, float], max_vol: float = 0.0
    ):
        """Add a region to the mesh.

        Parameters
        ----------
        id : int
            Unique identifier for the region.
        point_in_region : tuple, list, np.array of float
            A single point inside the region, specified as (x, y, z).
        max_vol : float, default: 0.0
            Maximum volume for the region.

        Examples
        --------
        Create a sphere in a cube in PyVista and mesh the region in the sphere
        at a higher density.

        >>> import pyvista as pv
        >>> import tetgen
        >>> cube = pv.Cube().triangulate()
        >>> sphere = pv.Sphere(theta_resolution=16, phi_resolution=16, radius=0.25)
        >>> mesh = pv.merge([sphere, cube])
        >>> tgen = tetgen.TetGen(mesh)
        >>> tgen.add_region(1, (0.0, 0.0, 0.0), 8e-6)  # sphere
        >>> tgen.add_region(2, [0.99, 0.0, 0.0], 4e-4)  # cube
        >>> nodes, elem, attrib = tgen.tetrahedralize(switches="pzq1.4Aa")
        >>> grid = tgen.grid
        >>> grid
        UnstructuredGrid (0x7cc05412bb80)
          N Cells:    23768
          N Points:   3964
          X Bounds:   -5.000e-01, 5.000e-01
          Y Bounds:   -5.000e-01, 5.000e-01
          Z Bounds:   -5.000e-01, 5.000e-01
          N Arrays:   1

        Supply the region info to the grid and then plot a slice through the
        grid.

        >>> grid["regions"] = attrib.ravel()
        >>> grid.slice().plot(show_edges=True, cpos="zy")

        """
        point_in_region_arr = np.asarray(point_in_region, dtype=float)
        self.regions[id] = (*point_in_region_arr, max_vol)

    def make_manifold(self, verbose=False):
        """Reconstruct a manifold clean surface from input mesh.

        Updates mesh in-place.

        Requires `pymeshfix <https://pypi.org/project/pymeshfix/>`_.

        Parameters
        ----------
        verbose : bool, default: False
            Enable verbose output.

        Examples
        --------
        Create a mesh and ensure it's manfold.

        >>> import pyvista
        >>> import tetgen
        >>> sphere = pyvista.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tgen = tetgen.TetGen(sphere)
        >>> tgen.make_manifold()

        """
        try:
            import pymeshfix
        except ImportError:
            raise ImportError("pymeshfix not installed. Please run:\npip install pymeshfix")

        # Run meshfix
        meshfix = pymeshfix.MeshFix(self.v, self.f)
        meshfix.repair(verbose)

        # overwrite this object with cleaned mesh
        self.v = meshfix.v
        self.f = meshfix.f

    def plot(self, **kwargs):
        """Display the input mesh.

        See :func:`pyvista.plot` for available arguments.

        Examples
        --------
        Plot the input mesh.

        >>> import pyvista
        >>> import tetgen
        >>> sphere = pyvista.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tgen = tetgen.TetGen(sphere)
        >>> tgen.plot()

        """
        self.mesh.plot(**kwargs)

    @property
    def mesh(self):
        """Return the surface mesh.

        Returns
        -------
        pyvista.PolyData
            Input surface mesh.

        Examples
        --------
        Generate a :class:`tetgen.TetGen` and return a :class:`pyvista.PolyData`.

        >>> import pyvista
        >>> import tetgen
        >>> sphere = pyvista.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tgen = tetgen.TetGen(sphere)
        >>> tgen.mesh
        PolyData (0x7fa3c97138e0)
          N Cells:    160
          N Points:   82
          N Strips:   0
          X Bounds:   -4.924e-01, 4.924e-01
          Y Bounds:   -4.683e-01, 4.683e-01
          Z Bounds:   -5.000e-01, 5.000e-01
          N Arrays:   0


        """
        triangles = np.empty((self.f.shape[0], 4), dtype="int")
        triangles[:, -3:] = self.f
        triangles[:, 0] = 3
        return pv.PolyData(self.v, triangles, deep=False)

    def tetrahedralize(
        self,
        plc=True,
        psc=0.0,
        refine=0.0,
        quality=True,
        nobisect=False,
        cdt=0.0,
        cdtrefine=7.0,
        coarsen=0.0,
        weighted=0.0,
        brio_hilbert=1.0,
        flipinsert=0.0,
        metric=0.0,
        varvolume=0.0,
        fixedvolume=0.0,
        regionattrib=0.0,
        insertaddpoints=0.0,
        diagnose=0.0,
        convex=0.0,
        nomergefacet=0.0,
        nomergevertex=0.0,
        noexact=0.0,
        nostaticfilter=0.0,
        zeroindex=0.0,
        facesout=0.0,
        edgesout=0.0,
        neighout=0.0,
        voroout=0.0,
        meditview=0.0,
        vtkview=0.0,
        vtksurfview=0.0,
        nobound=0.0,
        nonodewritten=0.0,
        noelewritten=0.0,
        nofacewritten=0.0,
        noiterationnum=0.0,
        nojettison=0.0,
        docheck=0.0,
        quiet=0.0,
        nowarning=0.0,
        verbose=0.0,
        vertexperblock=4092.0,
        tetrahedraperblock=8188.0,
        shellfaceperblock=2044.0,
        supsteiner_level=2.0,
        addsteiner_algo=1.0,
        coarsen_param=0.0,
        weighted_param=0.0,
        fliplinklevel=-1.0,
        flipstarsize=-1.0,
        fliplinklevelinc=1.0,
        opt_max_flip_level=3.0,
        opt_scheme=7.0,
        opt_iterations=3.0,
        smooth_cirterion=1.0,
        smooth_maxiter=7.0,
        delmaxfliplevel=1.0,
        order=1.0,
        reversetetori=0.0,
        steinerleft=100000.0,
        unflip_queue_limit=1000.0,
        no_sort=0.0,
        hilbert_order=52.0,
        hilbert_limit=8.0,
        brio_threshold=64.0,
        brio_ratio=0.125,
        epsilon=1.0e-8,
        facet_separate_ang_tol=179.9,
        collinear_ang_tol=179.9,
        facet_small_ang_tol=15.0,
        maxvolume=-1.0,
        maxvolume_length=-1.0,
        minratio=2.0,
        opt_max_asp_ratio=1000.0,
        opt_max_edge_ratio=100.0,
        mindihedral=0.0,
        optmaxdihedral=177.0,
        metric_scale=1.0,
        smooth_alpha=0.3,
        coarsen_percent=1.0,
        elem_growth_ratio=0.0,
        refine_progress_ratio=0.333,
        switches=None,
        bgmeshfilename="",
        bgmesh=None,
    ):
        """Generate tetrahedrals interior to the surface mesh.

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
        quality : bool, optional
            Enables/disables mesh improvement. Enabled by default.
            Disable this to speed up mesh generation while sacrificing
            quality. Default True.

        minratio : double, default: 2.0
            Maximum allowable radius-edge ratio.  Must be greater than
            1.0 the closer to 1.0, the higher the quality of the mesh.
            Be sure to raise ``steinerleft`` to allow for the addition of
            points to improve the quality of the mesh. Avoid overly
            restrictive requirements, otherwise, meshing will appear
            to hang.

            Testing has showed that 1.1 is a reasonable input for a
            high quality mesh.

        mindihedral : double, default: 0.0
            Minimum allowable dihedral angle. The larger this number,
            the higher the quality of the resulting mesh. Be sure to
            raise ``steinerleft`` to allow for the addition of points to
            improve the quality of the mesh. Avoid overly restrictive
            requirements, otherwise, meshing will appear to hang.

            Testing has shown that 10 is a reasonable input

        verbose : int, default: 0
            Controls the underlying TetGen library to output text to
            console. Users using iPython will not see this output.
            Setting to 1 enables some information about the mesh
            generation while setting verbose to 2 enables more debug
            output. Default is no output

        nobisect : bool, default: False
            Controls if Steiner points are added to the input surface
            mesh. When enabled, the surface mesh will be modified.

            Testing has shown that if your input surface mesh is
            already well shaped, disabling this setting will improve
            meshing speed and mesh quality.

        steinerleft : int, default: 100000
            Steiner points are points added to the original surface
            mesh to create a valid tetrahedral mesh. Settings this to
            -1 will allow tetgen to create an unlimited number of
            steiner points, but the program will likely hang if this
            is used in combination with narrow quality requirements.

            The first type of Steiner points are used in creating an
            initial tetrahedralization of PLC. These Steiner points
            are mandatory in order to create a valid
            tetrahedralization

            The second type of Steiner points are used in creating
            quality tetra- hedral meshes of PLCs. These Steiner points
            are optional, while they may be necessary in order to
            improve the mesh quality or to conform the size of mesh
            elements.

        order : int, default: 2
            Controls whether TetGen creates linear tetrahedrals or
            quadradic tetrahedrals. Set order to 2 to output
            quadradic tetrahedrals.

        bgmeshfilename : str, optional
            Filename of the background mesh.

        bgmesh : pv.UnstructuredGrid
            Background mesh to be processed. Must be composed of only linear
            tetra. Cannot specify both ``bgmeshfilename`` and ``bgmesh``.

        Returns
        -------
        nodes : numpy.ndarray
            Array of nodes representing the tetrahedral mesh.

        elems : numpy.ndarray
            Array of elements representing the tetrahedral mesh.

        Examples
        --------
        The following switches "pq1.1/10Y" would be:

        >>> nodes, elems = tgen.tetrahedralize(
        ...     nobisect=True, quality=True, minratio=1.1, mindihedral=10
        ... )

        Using the switches option:

        >>> nodes, elems = tgen.tetrahedralize(switches="pq1.1/10Y")

        Notes
        -----
        There are many other options and the TetGen documentation
        contains descriptions only for the switches of the original
        C++ program.  This is the relationship between tetgen switches
        and python optional inputs:

        Switches of TetGen.

        +---------------------------+---------------+---------+
        | Option                    | Switch        | Default |
        +---------------------------+---------------+---------+
        | plc                       | ``'-p'``      | 0.      |
        +---------------------------+---------------+---------+
        | psc                       | ``'-s'``      | 0.      |
        +---------------------------+---------------+---------+
        | refine                    | ``'-r'``      | 0.      |
        +---------------------------+---------------+---------+
        | quality                   | ``'-q'``      | 0.      |
        +---------------------------+---------------+---------+
        | nobisect                  | ``'-Y'``      | 0.      |
        +---------------------------+---------------+---------+
        | cdt                       | ``'-D'``      | 0.      |
        +---------------------------+---------------+---------+
        | cdtrefine                 | ``'-D#'``     | 7.      |
        +---------------------------+---------------+---------+
        | coarsen                   | ``'-R'``      | 0.      |
        +---------------------------+---------------+---------+
        | weighted                  | ``'-w'``      | 0.      |
        +---------------------------+---------------+---------+
        | brio_hilbert              | ``'-b'``      | 1.      |
        +---------------------------+---------------+---------+
        | flipinsert                | ``'-L'``      | 0.      |
        +---------------------------+---------------+---------+
        | metric                    | ``'-m'``      | 0.      |
        +---------------------------+---------------+---------+
        | varvolume                 | ``'-a'``      | 0.      |
        +---------------------------+---------------+---------+
        | fixedvolume               | ``'-a'``      | 0.      |
        +---------------------------+---------------+---------+
        | regionattrib              | ``'-A'``      | 0.      |
        +---------------------------+---------------+---------+
        | insertaddpoints           | ``'-i'``      | 0.      |
        +---------------------------+---------------+---------+
        | diagnose                  | ``'-d'``      | 0.      |
        +---------------------------+---------------+---------+
        | convex                    | ``'-c'``      | 0.      |
        +---------------------------+---------------+---------+
        | nomergefacet              | ``'-M'``      | 0.      |
        +---------------------------+---------------+---------+
        | nomergevertex             | ``'-M'``      | 0.      |
        +---------------------------+---------------+---------+
        | noexact                   | ``'-X'``      | 0.      |
        +---------------------------+---------------+---------+
        | nostaticfilter            | ``'-X'``      | 0.      |
        +---------------------------+---------------+---------+
        | zeroindex                 | ``'-z'``      | 0.      |
        +---------------------------+---------------+---------+
        | facesout                  | ``'-f'``      | 0.      |
        +---------------------------+---------------+---------+
        | edgesout                  | ``'-e'``      | 0.      |
        +---------------------------+---------------+---------+
        | neighout                  | ``'-n'``      | 0.      |
        +---------------------------+---------------+---------+
        | voroout                   | ``'-v'``      | 0.      |
        +---------------------------+---------------+---------+
        | meditview                 | ``'-g'``      | 0.      |
        +---------------------------+---------------+---------+
        | vtkview                   | ``'-k'``      | 0.      |
        +---------------------------+---------------+---------+
        | vtksurfview               | ``'-k'``      | 0.      |
        +---------------------------+---------------+---------+
        | nobound                   | ``'-B'``      | 0.      |
        +---------------------------+---------------+---------+
        | nonodewritten             | ``'-N'``      | 0.      |
        +---------------------------+---------------+---------+
        | noelewritten              | ``'-E'``      | 0.      |
        +---------------------------+---------------+---------+
        | nofacewritten             | ``'-F'``      | 0.      |
        +---------------------------+---------------+---------+
        | noiterationnum            | ``'-I'``      | 0.      |
        +---------------------------+---------------+---------+
        | nojettison                | ``'-J'``      | 0.      |
        +---------------------------+---------------+---------+
        | docheck                   | ``'-C'``      | 0.      |
        +---------------------------+---------------+---------+
        | quiet                     | ``'-Q'``      | 0.      |
        +---------------------------+---------------+---------+
        | nowarning                 | ``'-W'``      | 0.      |
        +---------------------------+---------------+---------+
        | verbose                   | ``'-V'``      | 0.      |
        +---------------------------+---------------+---------+

        Parameters of TetGen.

        +---------------------------+---------------+---------+
        | Option                    | Switch        | Default |
        +---------------------------+---------------+---------+
        | vertexperblock            | ``'-x'``      | 4092.   |
        +---------------------------+---------------+---------+
        | tetrahedraperblock        | ``'-x'``      | 8188.   |
        +---------------------------+---------------+---------+
        | shellfaceperblock         | ``'-x'``      | 2044.   |
        +---------------------------+---------------+---------+
        | supsteiner_level          | ``'-Y/'``     | 2.      |
        +---------------------------+---------------+---------+
        | addsteiner_algo           | ``'-Y//'``    | 1.      |
        +---------------------------+---------------+---------+
        | coarsen_param             | ``'-R'``      | 0.      |
        +---------------------------+---------------+---------+
        | weighted_param            | ``'-w'``      | 0.      |
        +---------------------------+---------------+---------+
        | opt_max_flip_level        | ``'-O'``      | 3.      |
        +---------------------------+---------------+---------+
        | opt_scheme                | ``'-O/#'``    | 7.      |
        +---------------------------+---------------+---------+
        | opt_iterations            | ``'-O//#'``   | 3.      |
        +---------------------------+---------------+---------+
        | smooth_cirterion          | ``'-s'``      | 1.      |
        +---------------------------+---------------+---------+
        | smooth_maxiter            | ``'-s'``      | 7.      |
        +---------------------------+---------------+---------+
        | order                     | ``'-o'``      | 1.      |
        +---------------------------+---------------+---------+
        | reversetetori             | ``'-o/'``     | 0.      |
        +---------------------------+---------------+---------+
        | steinerleft               | ``'-S'``      | 0.      |
        +---------------------------+---------------+---------+
        | unflip_queue_limit        | ``'-U#'``     | 1000.   |
        +---------------------------+---------------+---------+
        | hilbert_order             | ``'-b///'``   | 52.     |
        +---------------------------+---------------+---------+
        | hilbert_limit             | ``'-b//'``    |  8.     |
        +---------------------------+---------------+---------+
        | brio_threshold            | ``'-b'``      | 64.     |
        +---------------------------+---------------+---------+
        | brio_ratio                | ``'-b/'``     |0.125.   |
        +---------------------------+---------------+---------+
        | epsilon                   | ``'-T'``      | 1.0e-8. |
        +---------------------------+---------------+---------+
        | facet_separate_ang_tol    | ``'-p'``      | 179.9.  |
        +---------------------------+---------------+---------+
        | collinear_ang_tol         | ``'-p/'``     | 179.9.  |
        +---------------------------+---------------+---------+
        | facet_small_ang_tol       | ``'-p//'``    | 15.0.   |
        +---------------------------+---------------+---------+
        | maxvolume                 | ``'-a'``      | -1.0.   |
        +---------------------------+---------------+---------+
        | maxvolume_length          | ``'-a'``      | -1.0.   |
        +---------------------------+---------------+---------+
        | minratio                  | ``'-q'``      | 0.0.    |
        +---------------------------+---------------+---------+
        | mindihedral               | ``'-q'``      | 5.0.    |
        +---------------------------+---------------+---------+
        | optmaxdihedral            | ``'-o/#'``    | 177.0.  |
        +---------------------------+---------------+---------+
        | metric_scale              | ``'-m#'``     | 1.0.    |
        +---------------------------+---------------+---------+
        | smooth_alpha              | ``'-s'``      | 0.3.    |
        +---------------------------+---------------+---------+
        | coarsen_percent           | ``'-R1/#'``   | 1.0.    |
        +---------------------------+---------------+---------+
        | elem_growth_ratio         | ``'-r#'``     | 0.0.    |
        +---------------------------+---------------+---------+
        | refine_progress_ratio     | ``'-r/#'``    | 0.333.  |
        +---------------------------+---------------+---------+

        """
        # format switches
        if switches is None:
            switches_str = b""
        else:
            switches_str = bytes(switches, "utf-8")

        # check verbose switch
        if verbose == 0:
            quiet = 1

        # Validate background mesh parameters
        if bgmesh and bgmeshfilename:
            raise ValueError("Cannot specify both bgmesh and bgmeshfilename")
        if bgmesh or bgmeshfilename:
            # Passing a background mesh only makes sense with metric set to true
            # (will be silently ignored otherwise)
            metric = 1
        if bgmesh:
            bgmesh_v, bgmesh_tet, bgmesh_mtr = self._process_bgmesh(bgmesh)
        else:
            bgmesh_v, bgmesh_tet, bgmesh_mtr = None, None, None

        # Call library
        plc = True  # always true

        # Convert regions to the format expected by TetGen
        # regions should be a list of lists, each containing [x, y, z, id, maxVol]
        regions = [[*values[0:3], id, values[3]] for id, values in self.regions.items()]

        try:
            self.node, self.elem, self.attributes = _tetgen.Tetrahedralize(
                self.v,
                self.f,
                regions,
                switches_str,
                plc,
                psc,
                refine,
                quality,
                nobisect,
                cdt,
                cdtrefine,
                coarsen,
                weighted,
                brio_hilbert,
                flipinsert,
                metric,
                varvolume,
                fixedvolume,
                regionattrib,
                insertaddpoints,
                diagnose,
                convex,
                nomergefacet,
                nomergevertex,
                noexact,
                nostaticfilter,
                zeroindex,
                facesout,
                edgesout,
                neighout,
                voroout,
                meditview,
                vtkview,
                vtksurfview,
                nobound,
                nonodewritten,
                noelewritten,
                nofacewritten,
                noiterationnum,
                nojettison,
                docheck,
                quiet,
                nowarning,
                verbose,
                vertexperblock,
                tetrahedraperblock,
                shellfaceperblock,
                supsteiner_level,
                addsteiner_algo,
                coarsen_param,
                weighted_param,
                fliplinklevel,
                flipstarsize,
                fliplinklevelinc,
                opt_max_flip_level,
                opt_scheme,
                opt_iterations,
                smooth_cirterion,
                smooth_maxiter,
                delmaxfliplevel,
                order,
                reversetetori,
                steinerleft,
                unflip_queue_limit,
                no_sort,
                hilbert_order,
                hilbert_limit,
                brio_threshold,
                brio_ratio,
                epsilon,
                facet_separate_ang_tol,
                collinear_ang_tol,
                facet_small_ang_tol,
                maxvolume,
                maxvolume_length,
                minratio,
                opt_max_asp_ratio,
                opt_max_edge_ratio,
                mindihedral,
                optmaxdihedral,
                metric_scale,
                smooth_alpha,
                coarsen_percent,
                elem_growth_ratio,
                refine_progress_ratio,
                bgmeshfilename,
                bgmesh_v,
                bgmesh_tet,
                bgmesh_mtr,
            )
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to tetrahedralize.\n"
                + "May need to repair surface by making it manifold:\n"
                + str(e)
            )

        # check if a mesh was generated
        if not np.any(self.node):
            raise RuntimeError(
                "Failed to tetrahedralize.\nMay need to repair surface by making it manifold"
            )

        # Return nodes and elements
        LOG.info(
            "Generated mesh with %d nodes and %d elements",
            self.node.shape[0],
            self.elem.shape[0],
        )
        self._updated = True

        # return with attributes if they exist
        if self.attributes is not None:
            return self.node, self.elem, self.attributes

        return self.node, self.elem

    @property
    def grid(self):
        """Return a :class:`pyvista.UnstructuredGrid`."""
        if self.node is None:
            raise RuntimeError("Run Tetrahedralize first")

        if self._grid is not None and not self._updated:
            return self._grid

        buf = np.empty((self.elem.shape[0], 1), pv.ID_TYPE)
        cell_type = np.empty(self.elem.shape[0], dtype="uint8")
        if self.elem.shape[1] == 4:  # linear
            buf[:] = 4
            cell_type[:] = 10
        elif self.elem.shape[1] == 10:  # quadradic
            buf[:] = 10
            cell_type[:] = 24
        else:
            raise Exception("Invalid element array shape %s" % str(self.elem.shape))

        cells = np.hstack((buf, self.elem))
        self._grid = pv.UnstructuredGrid(cells, cell_type, self.node)

        self._updated = False
        return self._grid

    def write(self, filename, binary=False):
        """Write an unstructured grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written. The file extension will
            select the type of writer to use.

            - ``".vtk"`` will use the vtk legacy writer
            - ``".vtu"`` will select the VTK XML writer

        binary : bool, default: False
            Writes as a binary file by default. Set to ``False`` to write
            ASCII.

        Examples
        --------
        Write to a VTK file.

        >>> tgen.write("grid.vtk", binary=True)

        Notes
        -----
        Binary files write much faster than ASCII, but binary files
        written on one system may not be readable on other systems.
        Binary can be used only with the legacy writer.

        You can use utilities like `meshio <https://github.com/nschloe/meshio>`_
        to convert to other formats in order to import into FEA software.
        """
        self.grid.save(filename, binary)

    @staticmethod
    def _process_bgmesh(mesh):
        """Process a background mesh.

        Parameters
        ----------
        bgmesh : pyvista.UnstructuredGrid
            Background mesh to be processed. Must be composed of only linear tetra,

        Returns
        -------
        bgmesh_v: numpy.ndarray
            Vertex array of the background mesh.
        bgmesh_tet: numpy.ndarray
            Tet array of the background mesh.
        bgmesh_mtr: numpy.ndarray
            Target size array of the background mesh.
        """
        if MTR_POINTDATA_KEY not in mesh.point_data:
            raise ValueError("Background mesh does not have target size information")

        # Celltype check
        if not (mesh.celltypes == pv.CellType.TETRA).all():
            raise ValueError("`mesh` must contain only tetrahedrons")

        bgmesh_v = mesh.points.astype(np.float64, copy=True).ravel()
        bgmesh_tet = mesh.cell_connectivity.astype(np.int32, copy=True)
        bgmesh_mtr = mesh.point_data[MTR_POINTDATA_KEY].astype(np.float64, copy=True).ravel()
        return bgmesh_v, bgmesh_tet, bgmesh_mtr
