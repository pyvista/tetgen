# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef inline double TripleProduct(double [3] ab, double [3] bc, double [3] cd):
    return ab[0] * (bc[1] * cd[2] - bc[2] * cd[1]) -\
           bc[0] * (ab[1] * cd[2] - ab[2] * cd[1]) +\
           cd[0] * (ab[1] * bc[2] - ab[2] * bc[1])


cdef inline double NormCalc(double [3] vec):
    return sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)


# Define tetrahedral edges  
cdef int [4][3] tet_edges
tet_edges[0][0] = 1
tet_edges[0][1] = 2
tet_edges[0][2] = 3

tet_edges[1][0] = 2
tet_edges[1][1] = 0
tet_edges[1][2] = 3

tet_edges[2][0] = 0
tet_edges[2][1] = 1
tet_edges[2][2] = 3

tet_edges[3][0] = 0
tet_edges[3][1] = 2
tet_edges[3][2] = 1


def CompScJac(int [:, :] cellarr, double [:, ::1] pts):
    """
    Returns the minimum scaled jacobian for each cell given a square cell
    array from the Cython function "SquareCells"
    
    """
    
    cdef double [3] e0
    cdef double [3] e1
    cdef double [3] e2

    cdef int i, j, indS, ind0, ind1, ind2, cnum
    cdef int ncells = cellarr.shape[0]
    cdef double [::1] jacs = np.empty(ncells)
    cdef double jac, normjac, tnorm
    for cnum in range(ncells):
        jac = 1.1

        # If tetrahedral
        for i in range(4):
            indS = cellarr[cnum, i]
            ind0 = cellarr[cnum, tet_edges[i][0]]
            ind1 = cellarr[cnum, tet_edges[i][1]]
            ind2 = cellarr[cnum, tet_edges[i][2]]
            
            for j in range(3):
                e0[j] = pts[ind0, j] - pts[indS, j]
                e1[j] = pts[ind1, j] - pts[indS, j]
                e2[j] = pts[ind2, j] - pts[indS, j]
               
            # normalize the determinant of the jacobian
            tnorm = NormCalc(e0)*NormCalc(e1)*NormCalc(e2)
            normjac = TripleProduct(e1, e2, e0)/tnorm

            # Track minimum jacobian
            if normjac < jac:
                jac = normjac
                    
        jacs[cnum] = jac*1.4142135623730951
            
    return np.asarray(jacs)
    
