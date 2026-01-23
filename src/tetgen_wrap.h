#include "tetgen.h"

// Adds additional functionality to original tetgen object
class tetgenio_wrap : public tetgenio {
  public:
    // constructor
    tetgenio_wrap();

    facet *f;
    polygon *p;

    void LoadArray(int, double *, int, int *);
    void LoadArrayWithMarkers(int, double *, int, int *, int *);
    void LoadMTRArray(int, double *, int, int *, double *);
    bool LoadTetMesh(char *, int);
    void LoadRegions(int nregions, double *regions);
    void LoadHoles(int nholes, double *holes);

    // destructor (We're missing this...
    ~tetgenio_wrap();
};
