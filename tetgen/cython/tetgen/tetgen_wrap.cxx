#include <stdio.h>
#include "tetgen_wrap.h"

// Object
tetgenio_wrap::tetgenio_wrap(){}


void tetgenio_wrap::LoadArray(int npoints, double* points, int nfaces,
                             int* facearr)
{
  facet *f;
  polygon *p;
  int i, j;
  int count = 0;

  // Allocate memory for points and store them
  numberofpoints = npoints;
  pointlist = new double[npoints*3];

  for(i = 0; i < npoints*3; i++) {
    pointlist[i] = points[i];
  }

  // Store the number of faces and allocate memory
  numberoffacets = nfaces;
  facetlist = new tetgenio::facet[nfaces];

  // Load in faces as facets
  for (i = 0; i < nfaces; i++) {
    // Initialize a face
    f = &facetlist[i];
    init(f);
    
    // Each facet has one polygon, no hole, and each polygon has a three\
    //vertices
    f->numberofpolygons = 1;
    f->polygonlist = new tetgenio::polygon[1];

    p = &f->polygonlist[0];
    init(p);
    p->numberofvertices = 3;
    p->vertexlist = new int[3];
    for (j = 0; j < 3; j++) {
      p->vertexlist[j] = facearr[count];
      count++;
    }
  }
}

