# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython nonecheck=False

from libcpp cimport bool
from cpython.string cimport PyString_AsString

import numpy as np
cimport numpy as np

import ctypes
from cython cimport view

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

    
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

        # Loads Arrays directly to tetgenio object
        void LoadArray(int, double*, int, int*)
        

cdef extern from "tetgen.h":

    cdef cppclass tetrahedralize:
        int tetrahedralize(char*, tetgenio_wrap*, tetgenio_wrap*)

    cdef cppclass tetgenbehavior:
        void tetgenbehavior()
        
        # Switches of TetGen
        int plc;                                                         # '-p', 0.
        int psc;                                                         # '-s', 0.
        int refine;                                                      # '-r', 0.
        int quality;                                                     # '-q', 0.
        int nobisect;                                                    # '-Y', 0.
        int coarsen;                                                     # '-R', 0.
        int weighted;                                                    # '-w', 0.
        int brio_hilbert;                                                # '-b', 1.
        int incrflip;                                                    # '-l', 0.
        int flipinsert;                                                  # '-L', 0.
        int metric;                                                      # '-m', 0.
        int varvolume;                                                   # '-a', 0.
        int fixedvolume;                                                 # '-a', 0.
        int regionattrib;                                                # '-A', 0.
        int cdtrefine;                                                   # '-D', 0.
        int insertaddpoints;                                             # '-i', 0.
        int diagnose;                                                    # '-d', 0.
        int convex;                                                      # '-c', 0.
        int nomergefacet;                                                # '-M', 0.
        int nomergevertex;                                               # '-M', 0.
        int noexact;                                                     # '-X', 0.
        int nostaticfilter;                                              # '-X', 0.
        int zeroindex;                                                   # '-z', 0.
        int facesout;                                                    # '-f', 0.
        int edgesout;                                                    # '-e', 0.
        int neighout;                                                    # '-n', 0.
        int voroout;                                                     # '-v', 0.
        int meditview;                                                   # '-g', 0.
        int vtkview;                                                     # '-k', 0.
        int nobound;                                                     # '-B', 0.
        int nonodewritten;                                               # '-N', 0.
        int noelewritten;                                                # '-E', 0.
        int nofacewritten;                                               # '-F', 0.
        int noiterationnum;                                              # '-I', 0.
        int nojettison;                                                  # '-J', 0.
        int docheck;                                                     # '-C', 0.
        int quiet;                                                       # '-Q', 0.
        int verbose;                                                     # '-V', 0.

        # Parameters of TetGen. 
        int vertexperblock;                                           # '-x', 4092.
        int tetrahedraperblock;                                       # '-x', 8188.
        int shellfaceperblock;                                        # '-x', 2044.
        int nobisect_nomerge;                                            # '-Y', 1.
        int supsteiner_level;                                           # '-Y/', 2.
        int addsteiner_algo;                                           # '-Y#', 1.
        int coarsen_param;                                               # '-R', 0.
        int weighted_param;                                              # '-w', 0.
        int fliplinklevel;                                                    # -1.
        int flipstarsize;                                                     # -1.
        int fliplinklevelinc;                                                 #  1.
        int reflevel;                                                    # '-D', 3.
        int optlevel;                                                    # '-O', 2.
        int optscheme;                                                   # '-O', 7.
        int delmaxfliplevel;                                                   # 1.
        int order;                                                       # '-o', 1.
        int reversetetori;                                              # '-o/', 0.
        int steinerleft;                                                 # '-S', 0.
        int no_sort;                                                           # 0.
        int hilbert_order;                                           # '-b#/', 52.
        int hilbert_limit;                                             # '-b#'  8.
        int brio_threshold;                                              # '-b' 64.
        double brio_ratio;                                             # '-b/' 0.125.
        double facet_separate_ang_tol;                                 # '-p', 179.9.
        double facet_overlap_ang_tol;                                  # '-p/',  0.1.
        double facet_small_ang_tol;                                   # '-p#', 15.0.
        double maxvolume;                                               # '-a', -1.0.
        double minratio;                                                 # '-q', 0.0.
        double mindihedral;                                              # '-q', 5.0.
        double optmaxdihedral;                                               # 165.0.
        double optminsmtdihed;                                               # 179.0.
        double optminslidihed;                                               # 179.0.  
        double epsilon;                                               # '-T', 1.0e-8.
        double coarsen_percent;                                         # -R1/#, 1.0.

    # Different calls depending on using settings input
    cdef void tetrahedralize(tetgenbehavior*, tetgenio_wrap*, tetgenio_wrap*) except +
    cdef void tetrahedralize(char*, tetgenio_wrap*, tetgenio_wrap*) except +


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
    
    
def Tetrahedralize(v, f, switches='', 
    plc=0,
    psc=0,
    refine=0,
    quality=0,
    nobisect=0,
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
    facesout =0,
    edgesout =0,
    neighout =0,
    voroout =0,
    meditview =0,
    vtkview=0,
    nobound=0,
    nonodewritten=1,
    noelewritten=1,
    nofacewritten=1,
    noiterationnum =0,
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
    order=1,
    reversetetori=0,
    steinerleft=10000, # default is -1, but this often leads to the program hanging
    no_sort=0,
    hilbert_order=52,
    hilbert_limit=8,
    brio_threshold=64,
    brio_ratio=0.125,
    facet_separate_ang_tol=179.9,
    facet_overlap_ang_tol=0.1,
    facet_small_ang_tol=15.0,
    maxvolume=-1.0,
    minratio=2.0,
    mindihedral=0.0,
    optmaxdihedral=165.0,
    optminsmtdihed=179.0,
    optminslidihed=179.0,
    epsilon=1.0e-8,
    coarsen_percent=1.0):
    """
    Tetgen function to interface with TetGen C++ program
    
    """
    # convert switches to c object
    # cdef char *cstring = PyString_AsString(switches)
    cdef char *cstring = switches

    # Check that inputs are valid
    if not v.flags['C_CONTIGUOUS']:
        if v.dtype != np.float:
            v = np.ascontiguousarray(v, dtype=np.float)
        else:
            v = np.ascontiguousarray(v)

    elif v.dtype != np.float:
        v = v.astype(np.float)

    # Ensure inputs are of the right type
    if not f.flags['C_CONTIGUOUS']:
        if f.dtype != ctypes.c_int:
            f = np.ascontiguousarray(f, dtype=ctypes.c_int)
        else:
            f = np.ascontiguousarray(f)

    elif f.dtype != ctypes.c_int:
        f = f.astype(ctypes.c_int)
    
    # Create input class
    tetgenio_in = PyTetgenio()
    tetgenio_in.LoadMesh(v.ravel(), f.ravel())
        
    # Create output class
    tetgenio_out = PyTetgenio()        
    
    if switches:
        tetrahedralize(cstring, &tetgenio_in.c_tetio, &tetgenio_out.c_tetio)    

        if 'o2' in switches:
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
        behavior.c_behavior.coarsen = coarsen
        behavior.c_behavior.metric = metric
        behavior.c_behavior.weighted = weighted
        behavior.c_behavior.brio_hilbert = brio_hilbert
        behavior.c_behavior.incrflip = incrflip
        behavior.c_behavior.flipinsert = flipinsert
        behavior.c_behavior.varvolume = varvolume
        behavior.c_behavior.fixedvolume = fixedvolume
        behavior.c_behavior.noexact = noexact
        behavior.c_behavior.nostaticfilter = nostaticfilter
        behavior.c_behavior.insertaddpoints = insertaddpoints
        behavior.c_behavior.regionattrib = regionattrib
        behavior.c_behavior.cdtrefine = cdtrefine
        behavior.c_behavior.diagnose = diagnose
        behavior.c_behavior.convex = convex
        behavior.c_behavior.zeroindex = zeroindex
        behavior.c_behavior.facesout = facesout
        behavior.c_behavior.edgesout = edgesout
        behavior.c_behavior.neighout = neighout
        behavior.c_behavior.voroout = voroout
        behavior.c_behavior.meditview = meditview
        behavior.c_behavior.vtkview = vtkview
        behavior.c_behavior.nobound = nobound
    #    behavior.c_behavior.nonodewritten = nonodewritten
    #    behavior.c_behavior.noelewritten = noelewritten
    #    behavior.c_behavior.nofacewritten = nofacewritten,
        behavior.c_behavior.noiterationnum = noiterationnum
        behavior.c_behavior.nomergefacet = nomergefacet
        behavior.c_behavior.nomergevertex = nomergevertex
        behavior.c_behavior.nojettison = nojettison
        behavior.c_behavior.docheck = docheck
        behavior.c_behavior.quiet = quiet
        behavior.c_behavior.verbose = verbose
    #    behavior.c_behavior.vertexperblock = 4092,
    #    behavior.c_behavior.tetrahedraperblock = 8188,
    #    behavior.c_behavior.shellfaceperblock = 4092,
        behavior.c_behavior.nobisect_nomerge = nobisect_nomerge
        behavior.c_behavior.supsteiner_level = supsteiner_level
        behavior.c_behavior.addsteiner_algo = addsteiner_algo
        behavior.c_behavior.coarsen_param = coarsen_param
        behavior.c_behavior.weighted_param = weighted_param
    #    behavior.c_behavior.fliplinklevel = -1, 
    #    behavior.c_behavior.flipstarsize = -1,  
    #    behavior.c_behavior.fliplinklevelinc = 1,
        behavior.c_behavior.reflevel = reflevel
        behavior.c_behavior.optscheme = optscheme
        behavior.c_behavior.optlevel = optlevel
        behavior.c_behavior.delmaxfliplevel = delmaxfliplevel
        behavior.c_behavior.order = order
        behavior.c_behavior.reversetetori = reversetetori
        behavior.c_behavior.steinerleft = steinerleft
        behavior.c_behavior.no_sort = no_sort
        behavior.c_behavior.hilbert_order = hilbert_order
        behavior.c_behavior.hilbert_limit = hilbert_limit
        behavior.c_behavior.brio_threshold = brio_threshold
        behavior.c_behavior.brio_ratio = brio_ratio
        behavior.c_behavior.facet_separate_ang_tol = facet_separate_ang_tol
        behavior.c_behavior.facet_overlap_ang_tol = facet_overlap_ang_tol
        behavior.c_behavior.facet_small_ang_tol = facet_small_ang_tol
        behavior.c_behavior.maxvolume = maxvolume
        behavior.c_behavior.minratio = minratio
        behavior.c_behavior.mindihedral = mindihedral
        behavior.c_behavior.optmaxdihedral = optmaxdihedral
        behavior.c_behavior.optminsmtdihed = optminsmtdihed
        behavior.c_behavior.optminslidihed = optminslidihed
        behavior.c_behavior.epsilon = epsilon
        behavior.c_behavior.coarsen_percent = coarsen_percent
    
        # Process from C++ side using behavior object
        tetrahedralize(&behavior.c_behavior, &tetgenio_in.c_tetio, 
                       &tetgenio_out.c_tetio)

    # Returns verticies and tetrahedrals of new mesh
    nodes = tetgenio_out.ReturnNodes()
    tets = tetgenio_out.ReturnTetrahedrals(order)

    return nodes, tets  

