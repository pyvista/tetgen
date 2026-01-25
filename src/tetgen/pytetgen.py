"""Python module to interface with wrapped TetGen C++ code."""

from importlib.util import find_spec
import ctypes
import logging
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from tetgen import _tetgen

if TYPE_CHECKING:
    from pyvista.core.pointset import PolyData, UnstructuredGrid

VTK_UNSIGNED_CHAR = 3
VTK_TETRA = 10
VTK_QUADRATIC_TETRA = 24

LOG = logging.getLogger(__name__)
LOG.setLevel("CRITICAL")

MTR_POINTDATA_KEY = "target_size"

invalid_input = TypeError(
    "Invalid input. First argument must be either a pyvista.PolyData object or vertex array, followed by a face arrays and optionally a face marker array."
)


def _polydata_from_faces(points: NDArray[np.float64], faces: NDArray[np.int32]) -> "PolyData":
    """
    Generate a polydata from a faces array containing no padding and all triangles.

    Parameters
    ----------
    points : np.ndarray
        Points array.
    faces : np.ndarray
        ``(n, 3)`` faces array.

    Returns
    -------
    PolyData
        New mesh.

    """
    if find_spec("pyvista.core") is None:
        raise ModuleNotFoundError(
            "To use this feature install pyvista with:\n\n    `pip install pyvista"
        )

    from pyvista.core.pointset import PolyData
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkCellArray
    from vtkmodules.vtkCommonCore import vtkTypeInt32Array

    if faces.ndim != 2:
        raise ValueError("Expected a two dimensional face array.")

    pdata = PolyData()
    pdata.points = points

    # convert to vtk arrays without copying
    vtk_dtype = vtkTypeInt32Array().GetDataType()

    offset = np.arange(0, faces.size + 1, faces.shape[1], dtype=np.int32)
    offset_vtk = numpy_to_vtk(offset, deep=False, array_type=vtk_dtype)
    faces_vtk = numpy_to_vtk(faces.ravel(), deep=False, array_type=vtk_dtype)

    carr = vtkCellArray()
    carr.SetData(offset_vtk, faces_vtk)

    pdata.SetPolys(carr)
    return pdata


def _to_ugrid(points: NDArray[np.float64], cells: NDArray[np.int32]) -> "UnstructuredGrid":
    """No copy unstructured grid creation."""
    if find_spec("pyvista.core") is None:
        raise ModuleNotFoundError(
            "To use this feature install pyvista with:\n\n    `pip install pyvista"
        )

    from pyvista.core.pointset import UnstructuredGrid
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkCellArray
    from vtkmodules.vtkCommonCore import vtkTypeInt32Array

    n_cells, node_per_cell = cells.shape
    cell_type = VTK_TETRA if node_per_cell == 4 else VTK_QUADRATIC_TETRA
    vtk_dtype = vtkTypeInt32Array().GetDataType()
    offsets = np.arange(0, node_per_cell * (n_cells + 1), node_per_cell, dtype=np.int32)

    offsets_vtk = numpy_to_vtk(offsets, deep=False, array_type=vtk_dtype)
    conn_vtk = numpy_to_vtk(cells.ravel(), deep=False, array_type=vtk_dtype)

    cell_array = vtkCellArray()
    cell_array.SetData(offsets_vtk, conn_vtk)

    cell_types_vtk = numpy_to_vtk(
        np.full(n_cells, cell_type), deep=False, array_type=VTK_UNSIGNED_CHAR
    )
    ugrid = UnstructuredGrid()
    ugrid.SetCells(cell_types_vtk, cell_array)
    ugrid.points = points
    return ugrid


class MeshNotTetrahedralizedError(RuntimeError):
    """RuntimeError raise raised when :class:`tetgen.Tetgen` has not been tetrahedralized."""

    def __init__(self, msg: str = "Tetrahedralize the surface mesh first with `tetrahedralize`."):
        """Initialize the error."""
        super().__init__(msg)


class TetGen:
    """
    Input, clean, and tetrahedralize surface meshes using TetGen.

    Parameters
    ----------
    args : str | pyvista.PolyData | numpy.ndarray
        Either a pyvista surface mesh or a ``(n, 3)`` vertex array and ``(m,
        3)`` face array.

    Examples
    --------
    Tetrahedralize a sphere using pyvista.

    >>> import pyvista
    >>> import tetgen
    >>> sphere = pyvista.Sphere(theta_resolution=10, phi_resolution=10)
    >>> tgen = tetgen.TetGen(sphere)
    >>> nodes, elem, attr, triface_markers = tgen.tetrahedralize()
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
    >>> nodes, elems, attr, triface_markers = tgen.tetrahedralize()

    """

    def __init__(
        self,
        arg0: "PolyData | NDArray[np.float64] | str | Path",
        arg1: NDArray[np.int32] | None = None,
        arg2: NDArray[np.int32] | None = None,
    ) -> None:
        """Initialize MeshFix using a mesh or arrays."""
        self._tetgen = _tetgen.PyTetgen()

        # self._edges: None | NDArray[np.int32] = None
        # self._edge_markers: None | NDArray[np.int32] = None
        self._grid: UnstructuredGrid | None = None

        def store_mesh(mesh: "PolyData") -> None:
            if not mesh.is_all_triangles:
                raise RuntimeError("Invalid mesh. Must be an all triangular mesh.")

            points = mesh.points.astype(np.float64, copy=False)
            faces = mesh._connectivity_array.reshape(-1, 3).astype(np.int32, copy=False)
            self._tetgen.load_mesh(points, faces)

        if "PolyData" in str(type(arg0)):  # check without importing
            from pyvista.core.pointset import PolyData

            if not isinstance(arg0, PolyData):
                raise TypeError(f"Unknown type {type(arg0)}. Expected a `pyvista.PolyData`.")
            mesh: PolyData = arg0

            store_mesh(mesh)
        elif isinstance(arg0, (np.ndarray, list)):
            points = np.asarray(arg0, dtype=np.float64)
            if isinstance(arg1, (np.ndarray, list)) and isinstance(arg2, (np.ndarray, list)):
                self._load_arrays(points, arg1, arg2)
            elif isinstance(arg1, (np.ndarray, list)):
                self._load_arrays(points, arg1)
            else:
                raise invalid_input
        elif isinstance(arg0, (str, Path)):
            try:
                import pyvista.core as pv
                from pyvista.core.pointset import PolyData
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "To load from a filename install pyvista with:\n\n    `pip install pyvista"
                )

            mesh = pv.read(arg0)
            if not isinstance(mesh, PolyData):
                raise RuntimeError(
                    "Loaded surface is not readable by pyvista as a "
                    f"surface. Expected `pyvista.PolyData`, got `{type(mesh)}`"
                )
            store_mesh(mesh)
        else:
            raise invalid_input

        # nasty segfault on tetgen if there are no input faces
        if not self._tetgen.n_faces:
            raise RuntimeError("Failed to load input faces.")

    def _load_arrays(
        self,
        v: NDArray[np.float64],
        f: NDArray[np.int32],
        fmarkers: NDArray[np.int32] | None = None,
    ):
        """
        Load triangular mesh from vertex and face arrays.

        Face arrays/lists are v and f. Both vertex and face arrays
        should be 2D arrays with each vertex containing XYZ data and
        each face containing three points.

        Optionally include faces markers.

        """
        # Check inputs
        if not isinstance(v, np.ndarray):
            try:
                v = np.asarray(v, np.float64)
                if v.ndim != 2 and v.shape[1] != 3:
                    raise Exception("Invalid vertex format. Shape should be `(npoints, 3)`")
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

        self._tetgen.load_mesh(
            v.astype(np.float64, copy=False),
            f.astype(np.int32, copy=False),
        )

        # Optional face markers
        if fmarkers is not None:
            fmarkers = np.asarray(fmarkers, dtype=np.int32)
            if fmarkers.ndim != 1 or fmarkers.shape[0] != f.shape[0]:
                raise ValueError(
                    f"Invalid face marker array. Shape should match the number of faces {f.shape[0]}"
                )
            self._tetgen.load_facet_markers(fmarkers)

    def add_region(self, id: int, point_in_region: Sequence[float], max_vol: float = 0.0):
        """
        Add a region to the mesh.

        Parameters
        ----------
        id : int
            Unique identifier for the region.
        point_in_region : tuple[float, float, float], list, np.array[np.float64]
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
        pt = pt = [float(item) for item in point_in_region]
        if len(pt) != 3:
            raise ValueError("Expected point to be a sequence of three floats")

        # regions must contain [x, y, z, id, maxVol]
        region = pt + [float(id)] + [max_vol]
        self._tetgen.load_region(region)

    def add_hole(self, point_in_hole: Sequence[float]) -> None:
        """
        Add a hole to the mesh.

        Parameters
        ----------
        point_in_hole : tuple, list, np.array of float
            A single point inside the hole, specified as (x, y, z).

        Examples
        --------
        Create a sphere as a hole in a cube in PyVista

        >>> import pyvista as pv
        >>> import tetgen
        >>> cube = pv.Cube().triangulate()
        >>> sphere = pv.Sphere(theta_resolution=16, phi_resolution=16, radius=0.25)
        >>> mesh = pv.merge([sphere, cube])
        >>> tgen = tetgen.TetGen(mesh)
        >>> tgen.add_hole([0.0, 0.0, 0.0])
        >>> nodes, elem = tgen.tetrahedralize(switches="pzq1.4")
        >>> grid = tgen.grid
        >>> grid
        UnstructuredGrid (0x28233f6d420)
        N Cells:    3533
        N Points:   781
        X Bounds:   -5.000e-01, 5.000e-01
        Y Bounds:   -5.000e-01, 5.000e-01
        Z Bounds:   -5.000e-01, 5.000e-01
        N Arrays:   0
        >>> grid.slice(normal="z").plot(show_edges=True, cpos="xy")

        """
        pt = [float(item) for item in point_in_hole]
        if len(pt) != 3:
            raise ValueError("Expected point to be a sequence of three floats")
        self._tetgen.load_hole(pt)

    def make_manifold(self, verbose: bool = False) -> None:
        """
        Reconstruct a manifold clean surface from input mesh.

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
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "`pymeshfix` is not installed. Install it with:\npip install pymeshfix"
            )

        # Run meshfix
        meshfix = pymeshfix.MeshFix(
            self._tetgen.return_input_points(), self._tetgen.return_input_faces()
        )
        meshfix.repair(verbose)

        # overwrite the loaded arrays object with the cleaned mesh
        self._tetgen.load_mesh(meshfix.v, meshfix.f)

    def plot(self, **kwargs: Any) -> Any:
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
        return self.mesh.plot(**kwargs)

    @property
    def mesh(self) -> "PolyData":
        """
        Return the input surface mesh.

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
        return _polydata_from_faces(
            self._tetgen.return_input_points(), self._tetgen.return_input_faces()
        )

    def tetrahedralize(
        self,
        plc: bool = True,
        psc: bool = False,
        refine: bool = False,
        quality: bool = True,
        nobisect: bool = False,
        cdt: bool = False,
        cdtrefine: int = 7,
        coarsen: bool = False,
        weighted: bool = False,
        brio_hilbert: bool = True,
        flipinsert: bool = False,
        metric: bool = False,
        varvolume: bool = False,
        fixedvolume: bool = False,
        regionattrib: bool = False,
        insertaddpoints: bool = False,
        diagnose: bool = False,
        convex: bool = False,
        nomergefacet: bool = False,
        nomergevertex: bool = False,
        noexact: bool = False,
        nostaticfilter: bool = False,
        zeroindex: bool = False,
        facesout: bool = False,
        edgesout: bool = False,
        neighout: bool = False,
        voroout: bool = False,
        meditview: bool = False,
        vtkview: bool = False,
        vtksurfview: bool = False,
        nobound: bool = False,
        nonodewritten: bool = False,
        noelewritten: bool = False,
        nofacewritten: bool = False,
        noiterationnum: bool = False,
        nojettison: bool = False,
        docheck: bool = False,
        quiet: bool = True,
        nowarning: bool = False,
        verbose: int = 0,
        vertexperblock: int = 4092,
        tetrahedraperblock: int = 8188,
        shellfaceperblock: int = 2044,
        supsteiner_level: int = 2,
        addsteiner_algo: int = 1,
        coarsen_param: int = 0,
        weighted_param: int = 0,
        fliplinklevel: int = -1,
        flipstarsize: int = -1,
        fliplinklevelinc: int = 1,
        opt_max_flip_level: int = 3,
        opt_scheme: int = 7,
        opt_iterations: int = 3,
        smooth_cirterion: int = 1,
        smooth_maxiter: int = 7,
        delmaxfliplevel: int = 1,
        order: int = 1,
        reversetetori: int = 0,
        steinerleft: int = 100000,
        unflip_queue_limit: int = 1000,
        no_sort: bool = False,
        hilbert_order: int = 52,
        hilbert_limit: int = 8,
        brio_threshold: int = 64,
        brio_ratio: float = 0.125,
        epsilon: float = 1.0e-8,
        facet_separate_ang_tol: float = 179.9,
        collinear_ang_tol: float = 179.9,
        facet_small_ang_tol: float = 15.0,
        maxvolume: float = -1.0,
        maxvolume_length: float = -1.0,
        minratio: float = 2.0,
        opt_max_asp_ratio: float = 1000.0,
        opt_max_edge_ratio: float = 100.0,
        mindihedral: float = 0.0,
        optmaxdihedral: float = 177.0,
        metric_scale: float = 1.0,
        smooth_alpha: float = 0.3,
        coarsen_percent: float = 1.0,
        elem_growth_ratio: float = 0.0,
        refine_progress_ratio: float = 0.333,
        switches: str = "",
        bgmeshfilename: str = "",
        bgmesh: Optional["UnstructuredGrid"] = None,
    ):
        """
        Generate tetrahedrals interior to the surface mesh.

        Returns nodes and elements belonging to the all tetrahedral mesh.

        The tetrahedral generator uses the C++ library TetGen and can be
        configured by either using a string of ``switches`` or by changing the
        underlying behavior using optional inputs.

        Should the user desire more control over the mesh tetrahedralization or
        wish to control the tetrahedralization in a more pythonic manner, use
        the optional inputs rather than inputting switches.

        Parameters
        ----------
        quality : bool, optional
            Enables/disables mesh improvement. Enabled by default. Disable
            this to speed up mesh generation while sacrificing quality. Default
            True.

        minratio : double, default: 2.0
            Maximum allowable radius-edge ratio. Must be greater than 1.0 the
            closer to 1.0, the higher the quality of the mesh. Be sure to
            raise ``steinerleft`` to allow for the addition of points to
            improve the quality of the mesh. Avoid overly restrictive
            requirements, otherwise, meshing will appear to hang.

            Testing has showed that 1.1 is a reasonable input for a high
            quality mesh.

        mindihedral : double, default: 0.0
            Minimum allowable dihedral angle. The larger this number, the
            higher the quality of the resulting mesh. Be sure to raise
            ``steinerleft`` to allow for the addition of points to improve the
            quality of the mesh. Avoid overly restrictive requirements,
            otherwise, meshing will appear to hang.

            Testing has shown that 10.0 is a reasonable input.

        quiet : bool, default: True
            Generate no output to stdout.

        verbose : int, default: 0
            Controls the underlying TetGen library to output text to
            console. Users using ``ipython`` may not see this output. Setting
            to 1 enables some information about the mesh generation while
            setting verbose to 2 enables more debug output. Default (``0``) is
            minimal output.

        nobisect : bool, default: False
            Controls if Steiner points are added to the input surface
            mesh. When enabled, the surface mesh will be modified.

            Testing has shown that if your input surface mesh is already well
            shaped, disabling this setting will improve meshing speed and mesh
            quality.

        steinerleft : int, default: 100000
            Steiner points are points added to the original surface mesh to
            create a valid tetrahedral mesh. Settings this to -1 will allow
            tetgen to create an unlimited number of steiner points, but the
            program will likely hang if this is used in combination with narrow
            quality requirements.

            The first type of Steiner points are used in creating an initial
            tetrahedralization of PLC. These Steiner points are mandatory in
            order to create a valid tetrahedralization

            The second type of Steiner points are used in creating quality
            tetra- hedral meshes of PLCs. These Steiner points are optional,
            while they may be necessary in order to improve the mesh quality or
            to conform the size of mesh elements.

        facesout : bool, default: False
            Build the faces to edge array.

        order : int, default: 1
            Controls whether TetGen creates linear tetrahedrals or quadradic
            tetrahedrals. Set order to 2 to output quadradic tetrahedrals.

        bgmeshfilename : str, default: ""
            Filename of the background mesh with the target size associated
            with the nodes. Cannot specify both ``bgmeshfilename`` and
            ``bgmesh``.

        regionattrib : bool, default: False
            Return region attributes.

        bgmesh : pyvista.UnstructuredGrid
            Background mesh to be processed. Must be composed of only linear
            tetra with the sizing contained in the `point_data` of the mesh
            within the `'target_size'` key. Cannot specify both
            ``bgmeshfilename`` and ``bgmesh``.

        switches : str, default: ""
            String of switches. When passed, overrides all keyword arguments.

        Returns
        -------
        nodes : np.ndarray[np.float64]
            Array of nodes representing the tetrahedral mesh.
        elems : np.ndarray[np.int32]
            Array of elements representing the tetrahedral mesh.
        attr : np.ndarray[np.float64]
            Region attributes. Empty unless ``regionattrib=True`` or the tetgen
            flag ``"A"`` is passed.
        triface_markers : np.ndarray[np.int32]
            Marker for each face in :attr:`TetGen.triface_list`.

        Examples
        --------
        The following switches ``"pq1.1/10Y"`` would be:

        >>> nodes, elems = tgen.tetrahedralize(
        ...     nobisect=True, quality=True, minratio=1.1, mindihedral=10
        ... )

        Using the switches input:

        >>> nodes, elems = tgen.tetrahedralize(switches="pq1.1/10Y")

        Notes
        -----
        There are many other options and the TetGen documentation contains
        descriptions only for the switches of the original C++ program. This is
        the relationship between tetgen switches and python optional inputs:

        Switches of TetGen:

        +---------------------------+---------------+---------+
        | Option                    | Switch        | Default |
        +---------------------------+---------------+---------+
        | plc                       | ``'-p'``      | False   |
        +---------------------------+---------------+---------+
        | psc                       | ``'-s'``      | False   |
        +---------------------------+---------------+---------+
        | refine                    | ``'-r'``      | False   |
        +---------------------------+---------------+---------+
        | quality                   | ``'-q'``      | False   |
        +---------------------------+---------------+---------+
        | nobisect                  | ``'-Y'``      | False   |
        +---------------------------+---------------+---------+
        | cdt                       | ``'-D'``      | False   |
        +---------------------------+---------------+---------+
        | coarsen                   | ``'-R'``      | False   |
        +---------------------------+---------------+---------+
        | weighted                  | ``'-w'``      | False   |
        +---------------------------+---------------+---------+
        | brio_hilbert              | ``'-b'``      | True    |
        +---------------------------+---------------+---------+
        | flipinsert                | ``'-L'``      | False   |
        +---------------------------+---------------+---------+
        | metric                    | ``'-m'``      | False   |
        +---------------------------+---------------+---------+
        | varvolume                 | ``'-a'``      | False   |
        +---------------------------+---------------+---------+
        | fixedvolume               | ``'-a'``      | False   |
        +---------------------------+---------------+---------+
        | regionattrib              | ``'-A'``      | False   |
        +---------------------------+---------------+---------+
        | insertaddpoints           | ``'-i'``      | False   |
        +---------------------------+---------------+---------+
        | diagnose                  | ``'-d'``      | False   |
        +---------------------------+---------------+---------+
        | convex                    | ``'-c'``      | False   |
        +---------------------------+---------------+---------+
        | nomergefacet              | ``'-M'``      | False   |
        +---------------------------+---------------+---------+
        | nomergevertex             | ``'-M'``      | False   |
        +---------------------------+---------------+---------+
        | noexact                   | ``'-X'``      | False   |
        +---------------------------+---------------+---------+
        | nostaticfilter            | ``'-X'``      | False   |
        +---------------------------+---------------+---------+
        | zeroindex                 | ``'-z'``      | False   |
        +---------------------------+---------------+---------+
        | facesout                  | ``'-f'``      | False   |
        +---------------------------+---------------+---------+
        | edgesout                  | ``'-e'``      | False   |
        +---------------------------+---------------+---------+
        | neighout                  | ``'-n'``      | False   |
        +---------------------------+---------------+---------+
        | voroout                   | ``'-v'``      | False   |
        +---------------------------+---------------+---------+
        | meditview                 | ``'-g'``      | False   |
        +---------------------------+---------------+---------+
        | vtkview                   | ``'-k'``      | False   |
        +---------------------------+---------------+---------+
        | vtksurfview               | ``'-k'``      | False   |
        +---------------------------+---------------+---------+
        | nobound                   | ``'-B'``      | False   |
        +---------------------------+---------------+---------+
        | nonodewritten             | ``'-N'``      | False   |
        +---------------------------+---------------+---------+
        | noelewritten              | ``'-E'``      | False   |
        +---------------------------+---------------+---------+
        | nofacewritten             | ``'-F'``      | False   |
        +---------------------------+---------------+---------+
        | noiterationnum            | ``'-I'``      | False   |
        +---------------------------+---------------+---------+
        | nojettison                | ``'-J'``      | False   |
        +---------------------------+---------------+---------+
        | docheck                   | ``'-C'``      | False   |
        +---------------------------+---------------+---------+
        | quiet                     | ``'-Q'``      | False   |
        +---------------------------+---------------+---------+
        | nowarning                 | ``'-W'``      | False   |
        +---------------------------+---------------+---------+
        | verbose                   | ``'-V'``      | False   |
        +---------------------------+---------------+---------+

        Parameters of TetGen:

        +---------------------------+---------------+---------+
        | Option                    | Switch        | Default |
        +---------------------------+---------------+---------+
        | cdtrefine                 | ``'-D#'``     | 7       |
        +---------------------------+---------------+---------+
        | vertexperblock            | ``'-x'``      | 4092    |
        +---------------------------+---------------+---------+
        | tetrahedraperblock        | ``'-x'``      | 8188    |
        +---------------------------+---------------+---------+
        | shellfaceperblock         | ``'-x'``      | 2044    |
        +---------------------------+---------------+---------+
        | supsteiner_level          | ``'-Y/'``     | 2       |
        +---------------------------+---------------+---------+
        | addsteiner_algo           | ``'-Y//'``    | 1       |
        +---------------------------+---------------+---------+
        | coarsen_param             | ``'-R'``      | 0       |
        +---------------------------+---------------+---------+
        | weighted_param            | ``'-w'``      | 0       |
        +---------------------------+---------------+---------+
        | opt_max_flip_level        | ``'-O'``      | 3       |
        +---------------------------+---------------+---------+
        | opt_scheme                | ``'-O/#'``    | 7       |
        +---------------------------+---------------+---------+
        | opt_iterations            | ``'-O//#'``   | 3       |
        +---------------------------+---------------+---------+
        | smooth_cirterion          | ``'-s'``      | 1       |
        +---------------------------+---------------+---------+
        | smooth_maxiter            | ``'-s'``      | 7       |
        +---------------------------+---------------+---------+
        | order                     | ``'-o'``      | 1       |
        +---------------------------+---------------+---------+
        | reversetetori             | ``'-o/'``     | 0       |
        +---------------------------+---------------+---------+
        | steinerleft               | ``'-S'``      | 100000  |
        +---------------------------+---------------+---------+
        | unflip_queue_limit        | ``'-U#'``     | 1000    |
        +---------------------------+---------------+---------+
        | hilbert_order             | ``'-b///'``   | 52      |
        +---------------------------+---------------+---------+
        | hilbert_limit             | ``'-b//'``    | 8       |
        +---------------------------+---------------+---------+
        | brio_threshold            | ``'-b'``      | 64      |
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
        plc = True  # must be always true

        # # Validate background mesh parameters
        if bgmesh and bgmeshfilename:
            raise ValueError("Cannot specify both `bgmesh` and `bgmeshfilename`")
        if bgmesh or bgmeshfilename:
            # Passing a background mesh only makes sense with metric set to true
            # (will be silently ignored otherwise)
            if switches and "m" not in switches:
                switches += "m"
            else:
                metric = True

        if bgmesh:
            self._process_bgmesh(bgmesh)
        elif bgmeshfilename:
            self._tetgen.load_bgmesh_from_filename(bgmeshfilename)

        # else:
        #     bgmesh_v, bgmesh_tet, bgmesh_mtr = None, None, None

        # self.fmarkers if hasattr(self, "fmarkers") else None,
        # regions,
        # self.holes,
        # bgmesh_v,
        # bgmesh_tet,
        # bgmesh_mtr,
        # return_surface_data=True,
        # return_edge_data=True,

        self._tetgen.tetrahedralize(
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
            switches,
        )
        # except RuntimeError as e:
        #     raise RuntimeError(
        #         "Failed to tetrahedralize.\n"
        #         f"May need to repair surface by making it manifold:\n{str(e)}"
        #     )

        # # Unpack results (backwards compatible)
        # if len(result) >= 8:
        #     (
        #         self._node,
        #         self._elem,
        #         self._attributes,
        #         self._triface_markers,
        #         self._trifaces,
        #         self._face2tet,
        #         self._edges,
        #         self._edge_markers,
        #     ) = result
        # else:
        #     self._node, self._elem, self._attributes, self._triface_markers = result

        # # check if a mesh was generated
        # if not np.any(self._node):
        #     raise RuntimeError(
        #         "Failed to tetrahedralize.\nMay need to repair surface by making it manifold"
        #     )

        # LOG.info(
        #     "Generated mesh with %d nodes and %d elements",
        #     self._node.shape[0],
        #     self._elem.shape[0],
        # )

        self._grid = None  # reset the cached grid
        return self._node, self._elem, self._attributes, self._triface_markers

    @property
    def _triface_markers(self) -> NDArray[np.int32]:
        return self._tetgen.return_triface_markers()

    @property
    def _attributes(self) -> NDArray[np.float64]:
        return self._tetgen.return_tetrahedron_attributes()

    @property
    def _node(self) -> NDArray[np.float64]:
        return self._tetgen.return_nodes()

    @property
    def _elem(self) -> NDArray[np.int32]:
        return self._tetgen.return_tets()

    @property
    def _trifaces(self) -> NDArray[np.int32]:
        return self._tetgen.return_trifaces()

    @property
    def face2tet(self) -> NDArray:
        """Return the mapping between each triface and the tetrahedral elements."""
        return self._face2tet

    @property
    def edges(self) -> NDArray[np.int32]:
        """
        Return the ``(n, 2)`` array of edges composing the tetrahedralized grid.

        This attribute is only available after running
        :meth:`TetGen.tetrahedralize`.

        Examples
        --------
        Tetrahedralize a sphere and the first 10 edges composing the tetrahedralized
        grid.

        >>> import tetgen
        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tet = tetgen.TetGen(sphere)
        >>> tet.tetrahedralize(switches="pq1.1/10YQ")
        >>> tet.edges[:10]
        array([[46, 47],
               [46, 40],
               [38, 40],
               [38, 47],
               [38, 46],
               [40, 47],
               [11,  4],
               [11,  2],
               [ 3,  2],
               [ 3,  4]], dtype=int32)

        """
        if self._edges is None:
            raise MeshNotTetrahedralizedError
        return self._edges

    @property
    def edge_markers(self) -> NDArray[np.int32]:
        """
        Return the ``(n, )`` array of edge markers denoting if an edge is internal.

        Interior edges are 0 and exterior edges are -1.

        This attribute is only available after running
        :meth:`TetGen.tetrahedralize`.

        Examples
        --------
        Tetrahedralize a sphere and create a mask of internal edges, ``is_internal``.

        >>> import tetgen
        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tet = tetgen.TetGen(sphere)
        >>> tet.tetrahedralize(switches="pq1.1/10YQ")
        >>> is_internal = tet.edge_markers == 0
        >>> is_internal[:10]
        array([ True,  True, False, False,  True, False,  True,  True, False,
               False])

        """
        if self._edge_markers is None:
            raise MeshNotTetrahedralizedError
        return self._edge_markers

    @property
    def triface_markers(self) -> NDArray[np.int32]:
        """
        Return the ``(n, )`` array of markers for each triangular face in :attr:`TetGen.trifaces`.

        Interior faces are denoted with 0 and exterior faces are marked as -1.

        This attribute is only available after running
        :meth:`TetGen.tetrahedralize`.

        Examples
        --------
        Tetrahedralize a sphere and return the array of the triangular faces
        that compose the tetrahedralized grid.

        >>> import tetgen
        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tet = tetgen.TetGen(sphere)
        >>> tet.tetrahedralize(switches="pq1.1/10YQ")
        >>> is_interior = tet.triface_markers == 0
        >>> is_interior[:10]
        array([ True,  True,  True, False,  True,  True,  True, False,  True,
                True])

        """
        if not self._tetgen.n_cells:
            raise MeshNotTetrahedralizedError
        return self._triface_markers

    @property
    def trifaces(self) -> NDArray[np.int32]:
        """
        Return the ``(n, 3)`` array of triangular faces composing the tetrahedral mesh.

        The indices of these faces correspond to a node in :attr:`TetGen.node`.

        This attribute is only available after running
        :meth:`TetGen.tetrahedralize`.

        Examples
        --------
        Tetrahedralize a sphere and return the array of the triangular faces
        that compose the tetrahedralized grid.

        >>> import tetgen
        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tet = tetgen.TetGen(sphere)
        >>> tet.tetrahedralize(switches="pq1.1/10YQ")
        >>> tet.trifaces
        array([[107,   1,   9],
               [107,  81,   1],
               [  9,  81, 107],
               ...,
               [ 15,   6, 109],
               [ 15,   7,   6],
               [ 15,   6,  14]], shape=(814, 3), dtype=int32)

        """
        if self._trifaces is None:
            raise MeshNotTetrahedralizedError
        return self._trifaces

    @property
    def node(self) -> NDArray[np.float64]:
        """
        Return an ``(n, 3)`` array of nodes composing the tetrahedralized surface.

        This attribute is only available after running
        :meth:`TetGen.tetrahedralize`.

        Examples
        --------
        Tetrahedralize a sphere and return the first three nodes.

        >>> import tetgen
        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tet = tetgen.TetGen(sphere)
        >>> tet.tetrahedralize(switches="pq1.1/10YQ")
        >>> tet.node[:3]
        array([[ 0.        ,  0.        ,  0.5       ],
               [ 0.        ,  0.        , -0.5       ],
               [ 0.17101008,  0.        ,  0.46984631]])

        """
        if self._node is None:
            raise MeshNotTetrahedralizedError
        return self._node

    @property
    def elem(self) -> NDArray[np.int32]:
        """
        Return the ``(n, 4)`` or ``(n, 10)`` array of elements composing the grid.

        This attribute is only available after running
        :meth:`TetGen.tetrahedralize`.

        Examples
        --------
        Tetrahedralize a sphere and return the first five elements.

        >>> import tetgen
        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tet = tetgen.TetGen(sphere)
        >>> tet.tetrahedralize(switches="pq1.1/10YQ")
        >>> tet.elem[:5]
        array([[ 81,   9,   1, 107],
               [ 58,  50,   0,  98],
               [  0, 104,  58,  98],
               [ 98, 105,  90,  87],
               [ 82,  91,  92, 102]], dtype=int32)

        """
        if self._elem is None:
            raise MeshNotTetrahedralizedError
        return self._elem

    @property
    def grid(self) -> "UnstructuredGrid":
        """
        Return a :class:`pyvista.UnstructuredGrid` of the tetrahedralized surface.

        This attribute is only available after running
        :meth:`TetGen.tetrahedralize`.

        Examples
        --------
        Tetrahedralize a ``pyvista.PolyData`` surface mesh into a
        ``pyvista.UnstructuredGrid``.

        >>> import tetgen
        >>> import pyvista as pv
        >>> sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)
        >>> tet = tetgen.TetGen(sphere)
        >>> tet.tetrahedralize(switches="pq1.1/10YQ")
        >>> grid = tet.grid
        >>> grid
        UnstructuredGrid (...)
          N Cells:    367
          N Points:   110
          X Bounds:   -4.924e-01, 4.924e-01
          Y Bounds:   -4.683e-01, 4.683e-01
          Z Bounds:   -5.000e-01, 5.000e-01
          N Arrays:   0

        """
        if not self._tetgen.n_nodes or not self._tetgen.n_cells:
            raise MeshNotTetrahedralizedError

        if self._grid is None:
            self._grid = _to_ugrid(self._node, self._elem)
        return self._grid

    def write(self, filename: str | Path, binary: bool = True) -> None:
        """
        Write an unstructured grid to disk.

        Parameters
        ----------
        filename : str | pathlib.Path
            Filename of grid to be written. The file extension will
            select the type of writer to use.

            - ``".vtk"`` will use the vtk legacy writer
            - ``".vtu"`` will select the VTK XML writer

        binary : bool, default: True
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

    def _process_bgmesh(self, mesh: "UnstructuredGrid") -> None:
        """
        Process a background mesh.

        Parameters
        ----------
        bgmesh : pyvista.UnstructuredGrid
            Background mesh to be processed. Must be composed of only linear
            tetrahedra.

        """
        import pyvista.core as pv

        if MTR_POINTDATA_KEY not in mesh.point_data:
            raise ValueError(
                "Background mesh does not have target size information in "
                f"key '{MTR_POINTDATA_KEY}' in the point data"
            )

        # Celltype check
        if not (mesh.celltypes == pv.CellType.TETRA).all():
            raise ValueError("Background mesh must contain only tetrahedrons.")

        # Vertex array of the background mesh.
        bgmesh_v = mesh.points.astype(np.float64, copy=False)

        # Tet array of the background mesh.
        bgmesh_tet = mesh.cell_connectivity.astype(np.int32, copy=False).reshape(-1, 4)

        # Target size array of the background mesh.
        bgmesh_mtr = mesh.point_data[MTR_POINTDATA_KEY].astype(np.float64, copy=False)
        if bgmesh_mtr.ndim != 1:
            raise ValueError(
                f"Expected 1 dimensional background mesh sizing, got {bgmesh_mtr.shape}"
            )

        self._tetgen.load_bgmesh_from_arrays(bgmesh_v, bgmesh_tet, bgmesh_mtr)
