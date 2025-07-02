# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython nonecheck=False

import numpy as np

cimport numpy as np

import ctypes

from cython cimport view
from libc.string cimport strcpy

# # Numpy must be initialized. When using numpy from C or Cython you must
# # _always_ do that, or you will have segfaults
# np.import_array()

""" Wrapped tetgen class """
cdef extern from "tetgen_wrap.h":
    cdef cppclass tetgenio_wrap:
        tetgenio_wrap()
        void initialize()

        # Point arrays
        int numberofpoints
        double* pointlist

        # Tetrahedron arrays
        int numberoftetrahedra
        int* tetrahedronlist

        double *tetrahedronattributelist;
        int numberoftetrahedronattributes;

        # Loads Arrays directly to tetgenio object
        void LoadArray(int, double*, int, int*)
        # Loads MTR Arrays directly to tetgenio object
        void LoadMTRArray(int, double*, int, int*, double*)
        # Loads tetmesh from file
        bint LoadTetMesh(char*, int)
        # Loads Regions directly to tetgenio object
        void LoadRegions(int, double*)

cdef extern from "tetgen.h":
    cdef cppclass tetrahedralize:
        int tetrahedralize(char*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*)
        int tetrahedralize(tetgenbehavior*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*)

    cdef cppclass tetgenbehavior:
        void tetgenbehavior()

        # Switches of TetGen.
        int plc
        int psc
        int refine
        int quality
        int nobisect
        int cdt
        int cdtrefine
        int coarsen
        int weighted
        int brio_hilbert
        int flipinsert
        int metric
        int varvolume
        int fixedvolume
        int regionattrib
        int insertaddpoints
        int diagnose
        int convex
        int nomergefacet
        int nomergevertex
        int noexact
        int nostaticfilter
        int zeroindex
        int facesout
        int edgesout
        int neighout
        int voroout
        int meditview
        int vtkview
        int vtksurfview
        int nobound
        int nonodewritten
        int noelewritten
        int nofacewritten
        int noiterationnum
        int nojettison
        int docheck
        int quiet
        int nowarning
        int verbose

        # Parameters of TetGen.
        int vertexperblock
        int tetrahedraperblock
        int shellfaceperblock
        int supsteiner_level
        int addsteiner_algo
        int coarsen_param
        int weighted_param
        int fliplinklevel
        int flipstarsize
        int fliplinklevelinc
        int opt_max_flip_level
        int opt_scheme
        int opt_iterations
        int smooth_cirterion
        int smooth_maxiter
        int delmaxfliplevel
        int order
        int reversetetori
        int steinerleft
        int unflip_queue_limit
        int no_sort
        int hilbert_order
        int hilbert_limit
        int brio_threshold
        double brio_ratio
        double epsilon
        double facet_separate_ang_tol
        double collinear_ang_tol
        double facet_small_ang_tol
        double maxvolume
        double maxvolume_length
        double minratio
        double opt_max_asp_ratio
        double opt_max_edge_ratio
        double mindihedral
        double optmaxdihedral
        double metric_scale
        double smooth_alpha
        double coarsen_percent
        double elem_growth_ratio
        double refine_progress_ratio
        char bgmeshfilename[1024]



    # Different calls depending on using settings input
    void tetrahedralize_with_switches "tetrahedralize" (char*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*) except +
    void tetrahedralize_with_behavior "tetrahedralize" (tetgenbehavior*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*) except +

    # cdef void tetrahedralize(tetgenbehavior*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*) except +
    # cdef void tetrahedralize(char*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*, tetgenio_wrap*) except +


cdef extern from "tetgen.h" namespace "tetgenbehavior":
    cdef enum objecttype: NODES, POLY, OFF, PLY, STL, MEDIT, VTK, MESH, NEU_MESH


cdef class PyBehavior:
    """ Python interface to tetgen behavior class """
    cdef tetgenbehavior c_behavior      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.c_behavior = tetgenbehavior()


cdef class PyTetgenio:
    """ Python interface to tetgenio """
    cdef tetgenio_wrap c_tetio      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.c_tetio = tetgenio_wrap()
        self.c_tetio.initialize()


    def ReturnNodes(self):
        """ Returns nodes from tetgen """

        # Create python copy of array
        cdef int npoints  = self.c_tetio.numberofpoints*3
        cdef double [::1] nodes = np.empty(npoints)

        cdef int i
        for i in range(npoints):
            nodes[i] = self.c_tetio.pointlist[i]

        return np.asarray(nodes).reshape((-1, 3))

    def ReturnTetrahedronAttributes(self):
        arrsz = self.c_tetio.numberoftetrahedra * self.c_tetio.numberoftetrahedronattributes
        # Create python copy of tetrahedral attributes array
        cdef double [::1] attrib = np.empty(arrsz)

        cdef int i
        cdef int j
        cdef int arrsz_c = arrsz

        if arrsz_c < 1:
            return None

        for i in range(arrsz_c):
                attrib[i] = self.c_tetio.tetrahedronattributelist[i]

        return np.asarray(attrib).astype(int).reshape((-1, self.c_tetio.numberoftetrahedronattributes))

    def ReturnTetrahedrals(self, order):
        """ Returns tetrahedrals from tetgen """

        # Determine
        if order == 1:
            arrsz = self.c_tetio.numberoftetrahedra*4
        else:
            arrsz = self.c_tetio.numberoftetrahedra*10

        # Create python copy of tetrahedral array
        cdef int [::1] tets = np.empty(arrsz, ctypes.c_int)

        cdef int i
        cdef int arrsz_c = arrsz
        for i in range(arrsz_c):
            tets[i] = self.c_tetio.tetrahedronlist[i]

        # Return properly shaped array based on cell order (quadradic or not)
        if order == 1:
            return np.asarray(tets).reshape((-1, 4))
        else:
            # Rearrange to match vtk cell format
            tetarr = np.asarray(tets).reshape((-1, 10))
            tetcopy = np.empty((tetarr.shape[0], 10), ctypes.c_int)
            tetcopy[:, :4] = tetarr[:, :4]
            """
            vtkQuadraticTetra
            The ordering of the ten points defining the cell is point ids
            (0-3, 4-9) where ids 0-3 are the four tetra vertices,
            and point ids 4-9 are the midedge nodes between:
            (0,1), (1,2), (2,0), (0,3), (1,3), and (2,3)
            """
            tetcopy[:, 4] = tetarr[:, 6]
            tetcopy[:, 5] = tetarr[:, 7]
            tetcopy[:, 6] = tetarr[:, 9]
            tetcopy[:, 7] = tetarr[:, 5]
            tetcopy[:, 8] = tetarr[:, 8]
            tetcopy[:, 9] = tetarr[:, 4]

            return tetcopy


    def LoadMesh(self, double [::1] points, int [::1] faces):
        """ Loads points and faces into TetGen """
        npoints = points.size/3
        nfaces = faces.size/3
        self.c_tetio.LoadArray(npoints, &points[0], nfaces, &faces[0])

    def LoadRegions(self, double [::1] regions):
        nregions = regions.size / 5
        self.c_tetio.LoadRegions(nregions, &regions[0])

    def LoadMTRMesh(self, double [::1] points, int [::1] tets, double [::1] mtr):
        """ Loads points and tets into TetGen """
        npoints = points.size/3
        ntets = tets.size/4
        self.c_tetio.LoadMTRArray(npoints, &points[0], ntets, &tets[0], &mtr[0])


    def LoadTetMesh(self, char* filename, int order):
        """ Loads tetmesh from file """
        return self.c_tetio.LoadTetMesh(filename, order)


def Tetrahedralize(
        v,
        f,
        regions=None,
        switches='',

        # Switches of TetGen
        plc=0.,
        psc=0.,
        refine=0.,
        quality=0.,
        nobisect=0.,
        cdt=0.,
        cdtrefine=7.,
        coarsen=0.,
        weighted=0.,
        brio_hilbert=1.,
        flipinsert=0.,
        metric=0.,
        varvolume=0.,
        fixedvolume=0.,
        regionattrib=0.,
        insertaddpoints=0.,
        diagnose=0.,
        convex=0.,
        nomergefacet=0.,
        nomergevertex=0.,
        noexact=0.,
        nostaticfilter=0.,
        zeroindex=0.,
        facesout=0.,
        edgesout=0.,
        neighout=0.,
        voroout=0.,
        meditview=0.,
        vtkview=0.,
        vtksurfview=0.,
        nobound=0.,
        nonodewritten=0.,
        noelewritten=0.,
        nofacewritten=0.,
        noiterationnum=0.,
        nojettison=0.,
        docheck=0.,
        quiet=0.,
        nowarning=0.,
        verbose=0.,

        # Parameters of TetGen.
        vertexperblock=4092.,
        tetrahedraperblock=8188.,
        shellfaceperblock=2044.,
        supsteiner_level=2.,
        addsteiner_algo=1.,
        coarsen_param=0.,
        weighted_param=0.,
        fliplinklevel=-1.,
        flipstarsize=-1.,
        fliplinklevelinc=1.,
        opt_max_flip_level=3.,
        opt_scheme=7.,
        opt_iterations=3.,
        smooth_cirterion=1.,
        smooth_maxiter=7.,
        delmaxfliplevel=1.,
        order=1.,
        reversetetori=0.,
        steinerleft=0.,
        unflip_queue_limit=1000.,
        no_sort=0.,
        hilbert_order=52.,
        hilbert_limit=8.,
        brio_threshold=64.,
        brio_ratio=0.125,
        epsilon=1.0e-8,
        facet_separate_ang_tol=179.9,
        collinear_ang_tol=179.9,
        facet_small_ang_tol=15.0,
        maxvolume=-1.0,
        maxvolume_length=-1.0,
        minratio=0.0,
        opt_max_asp_ratio=1000.0,
        opt_max_edge_ratio=100.0,
        mindihedral=5.0,
        optmaxdihedral=177.0,
        metric_scale=1.0,
        smooth_alpha=0.3,
        coarsen_percent=1.0,
        elem_growth_ratio=0.0,
        refine_progress_ratio=0.333,
        bgmeshfilename='',
        bgmesh_v=None,
        bgmesh_tet=None,
        bgmesh_mtr=None,
    ):

    """Tetgen function to interface with TetGen C++ program."""
    # convert switches to c object
    cdef char *cstring = switches

    # convert bgmeshfilename to c object
    cdef char bgmeshfilename_c[1024]
    cdef bytes py_bgmeshfilename = bgmeshfilename.encode('utf-8')
    cdef char* bgmeshfilename_py = py_bgmeshfilename
    strcpy(bgmeshfilename_c, bgmeshfilename_py)

    # Check that inputs are valid
    def cast_to_cint(x):
        return np.ascontiguousarray(x, dtype=ctypes.c_int)

    def cast_to_cdouble(x):
        return np.ascontiguousarray(x, dtype=np.float64)

    v = cast_to_cdouble(v)
    f = cast_to_cint(f)

    if regions is not None:
        regions = cast_to_cdouble(regions)

    if bgmesh_v is not None:
        bgmesh_v = cast_to_cdouble(bgmesh_v)
    if bgmesh_tet is not None:
        bgmesh_tet = cast_to_cint(bgmesh_tet)
    if bgmesh_mtr is not None:
        bgmesh_mtr = cast_to_cdouble(bgmesh_mtr)

    # Create input class
    tetgenio_in = PyTetgenio()
    tetgenio_in.LoadMesh(v.ravel(), f.ravel())

    if regions is not None:
        tetgenio_in.LoadRegions(regions.ravel())

    # Create output class
    tetgenio_out = PyTetgenio()

    tetgenio_bg = PyTetgenio()
    if bgmesh_mtr is not None:
        tetgenio_bg.LoadMTRMesh(bgmesh_v, bgmesh_tet, bgmesh_mtr)
    if bgmeshfilename:
        tetgenio_bg.LoadTetMesh(bgmeshfilename_c, <int>objecttype.NODES)

    if switches:
        tetrahedralize_with_switches(cstring, &tetgenio_in.c_tetio, &tetgenio_out.c_tetio, NULL, &tetgenio_bg.c_tetio)
        if b'o2' in switches:
            order = 2

    else: # set defaults or user input settings
        # Enable plc if checking for self intersections
        if diagnose:
            plc = 1

        # Ensure user has input order properly
        if order != 1 and order != 2:
            raise Exception('Settings error: Order should be 1 or 2')

        # Set behavior
        behavior = PyBehavior()
        behavior.c_behavior.plc = plc
        behavior.c_behavior.psc = psc
        behavior.c_behavior.refine = refine
        behavior.c_behavior.quality = quality
        behavior.c_behavior.nobisect = nobisect
        behavior.c_behavior.cdt = cdt
        behavior.c_behavior.cdtrefine = cdtrefine
        behavior.c_behavior.coarsen = coarsen
        behavior.c_behavior.weighted = weighted
        behavior.c_behavior.brio_hilbert = brio_hilbert
        behavior.c_behavior.flipinsert = flipinsert
        behavior.c_behavior.metric = metric
        behavior.c_behavior.varvolume = varvolume
        behavior.c_behavior.fixedvolume = fixedvolume
        behavior.c_behavior.regionattrib = regionattrib
        behavior.c_behavior.insertaddpoints = insertaddpoints
        behavior.c_behavior.diagnose = diagnose
        behavior.c_behavior.convex = convex
        behavior.c_behavior.nomergefacet = nomergefacet
        behavior.c_behavior.nomergevertex = nomergevertex
        behavior.c_behavior.noexact = noexact
        behavior.c_behavior.nostaticfilter = nostaticfilter
        behavior.c_behavior.zeroindex = zeroindex
        behavior.c_behavior.facesout = facesout
        behavior.c_behavior.edgesout = edgesout
        behavior.c_behavior.neighout = neighout
        behavior.c_behavior.voroout = voroout
        behavior.c_behavior.meditview = meditview
        behavior.c_behavior.vtkview = vtkview
        behavior.c_behavior.vtksurfview = vtksurfview
        behavior.c_behavior.nobound = nobound
        behavior.c_behavior.nonodewritten = nonodewritten
        behavior.c_behavior.noelewritten = noelewritten
        behavior.c_behavior.nofacewritten = nofacewritten
        behavior.c_behavior.noiterationnum = noiterationnum
        behavior.c_behavior.nojettison = nojettison
        behavior.c_behavior.docheck = docheck
        behavior.c_behavior.quiet = quiet
        behavior.c_behavior.nowarning = nowarning
        behavior.c_behavior.verbose = verbose
        behavior.c_behavior.vertexperblock = vertexperblock
        behavior.c_behavior.tetrahedraperblock = tetrahedraperblock
        behavior.c_behavior.shellfaceperblock = shellfaceperblock
        behavior.c_behavior.supsteiner_level = supsteiner_level
        behavior.c_behavior.addsteiner_algo = addsteiner_algo
        behavior.c_behavior.coarsen_param = coarsen_param
        behavior.c_behavior.weighted_param = weighted_param
        behavior.c_behavior.fliplinklevel = fliplinklevel
        behavior.c_behavior.flipstarsize = flipstarsize
        behavior.c_behavior.fliplinklevelinc = fliplinklevelinc
        behavior.c_behavior.opt_max_flip_level = opt_max_flip_level
        behavior.c_behavior.opt_scheme = opt_scheme
        behavior.c_behavior.opt_iterations = opt_iterations
        behavior.c_behavior.smooth_cirterion = smooth_cirterion
        behavior.c_behavior.smooth_maxiter = smooth_maxiter
        behavior.c_behavior.delmaxfliplevel = delmaxfliplevel
        behavior.c_behavior.order = order
        behavior.c_behavior.reversetetori = reversetetori
        behavior.c_behavior.steinerleft = steinerleft
        behavior.c_behavior.unflip_queue_limit = unflip_queue_limit
        behavior.c_behavior.no_sort = no_sort
        behavior.c_behavior.hilbert_order = hilbert_order
        behavior.c_behavior.hilbert_limit = hilbert_limit
        behavior.c_behavior.brio_threshold = brio_threshold
        behavior.c_behavior.brio_ratio = brio_ratio
        behavior.c_behavior.epsilon = epsilon
        behavior.c_behavior.facet_separate_ang_tol = facet_separate_ang_tol
        behavior.c_behavior.collinear_ang_tol = collinear_ang_tol
        behavior.c_behavior.facet_small_ang_tol = facet_small_ang_tol
        behavior.c_behavior.maxvolume = maxvolume
        behavior.c_behavior.maxvolume_length = maxvolume_length
        behavior.c_behavior.minratio = minratio
        behavior.c_behavior.opt_max_asp_ratio = opt_max_asp_ratio
        behavior.c_behavior.opt_max_edge_ratio = opt_max_edge_ratio
        behavior.c_behavior.mindihedral = mindihedral
        behavior.c_behavior.optmaxdihedral = optmaxdihedral
        behavior.c_behavior.metric_scale = metric_scale
        behavior.c_behavior.smooth_alpha = smooth_alpha
        behavior.c_behavior.coarsen_percent = coarsen_percent
        behavior.c_behavior.elem_growth_ratio = elem_growth_ratio
        behavior.c_behavior.refine_progress_ratio = refine_progress_ratio

        # Process from C++ side using behavior object
        tetrahedralize_with_behavior(&behavior.c_behavior, &tetgenio_in.c_tetio,
                       &tetgenio_out.c_tetio, NULL, &tetgenio_bg.c_tetio)


    # Returns vertices and tetrahedrals of new mesh
    nodes = tetgenio_out.ReturnNodes()
    tets = tetgenio_out.ReturnTetrahedrals(order)
    attributes = tetgenio_out.ReturnTetrahedronAttributes()

    return nodes, tets, attributes
