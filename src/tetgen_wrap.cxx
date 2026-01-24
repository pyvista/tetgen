#include "tetgen_wrap.h"
#include <cstring>
#include <stdio.h>

// Object
tetgenio_wrap::tetgenio_wrap() {}

void tetgenio_wrap::LoadHoles(int nholes, double *holes) {
    int i;

    // Allocate memory for holes and store them
    numberofholes = nholes;
    holelist = new double[nholes * 3];

    for (i = 0; i < nholes * 3; i++) {
        holelist[i] = holes[i];
    }
}

void tetgenio_wrap::LoadMTRArray(
    int npoints, double *points, int ntets, int *tetarr, double *mtrpoints) {
    int i, j;
    int count = 0;

    // Allocate memory for points and store them
    numberofpoints = npoints;
    pointlist = new double[npoints * 3];

    for (i = 0; i < npoints * 3; i++) {
        pointlist[i] = points[i];
    }

    // Populate pointmtrlist
    numberofpointmtrs = 1;
    pointmtrlist = new double[npoints];
    for (i = 0; i < npoints; i++) {
        pointmtrlist[i] = mtrpoints[i];
    }

    // Load tets (assumes 4 nodes per tetrahedron)
    numberoftetrahedra = ntets;
    numberofcorners = 4;
    tetrahedronlist = new int[ntets * numberofcorners];
    numberoftetrahedronattributes = 0;
    for (i = 0; i < ntets * numberofcorners; i++) {
        tetrahedronlist[i] = tetarr[i];
    }
}

// Wrapper around loadtetmesh function for filename support
bool tetgenio_wrap::LoadTetMesh(char *filename, int object) {
    return load_tetmesh(filename, object);
}

// unnecessary as tetgen internally handles collecting
tetgenio_wrap::~tetgenio_wrap() {
    // delete[] pointlist;
    // delete[] regionlist;
    // delete[] holelist;
    // delete[] facetlist;
    // delete[] facetmarkerlist;
    // delete[] pointmtrlist;
    // delete[] tetrahedronlist;

    // pointlist = nullptr;
    // regionlist = nullptr;
    // holelist = nullptr;
    // facetlist = nullptr;
    // facetmarkerlist = nullptr;
    // pointmtrlist = nullptr;
    // tetrahedronlist = nullptr;
}
