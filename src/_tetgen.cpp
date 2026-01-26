// Python interface to tetgen via nanobind.
#include <stdio.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "array_support.h"

#include "tetgen.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

// Class to be exposed to python
struct PyTetgen {
    tetgenio io;
    tetgenio io_out;
    tetgenio io_bg;
    std::vector<std::vector<double>> regions;
    std::vector<std::vector<double>> holes;

    PyTetgen() {
        io = tetgenio(); // input
        io.initialize();
        io_out = tetgenio(); // output
        io_bg = tetgenio();  // background mesh
    }

    NDArray<double, 2> return_nodes() {
        int n_points = io_out.numberofpoints;

        // create the array using numpy to allow Python to take control of gc
        // otherwise, when tetgen goes out of scope the array is deleted within tetgen
        NDArray<double, 2> points_arr = MakeNDArray<double, 2>({n_points, 3});
        memcpy(points_arr.data(), io_out.pointlist, sizeof(double) * n_points * 3);

        return points_arr;
    }

    NDArray<int, 2> return_tets() {
        int n_nodes = io_out.numberofcorners;
        int n_cells = io_out.numberoftetrahedra;
        auto arr = MakeNDArray<int, 2>({n_cells, n_nodes});
        std::memcpy(arr.data(), io_out.tetrahedronlist, sizeof(int) * n_cells * n_nodes);

        // ensure it's 1 based regardless of internal indexing
        if (io_out.firstnumber == 1) {
            int *p = arr.data();
            int total = n_cells * n_nodes;
            for (int i = 0; i < total; ++i) {
                --p[i];
            }
        }

        return arr;
    }

    // Tetrahedralize using keyword args
    void tetrahedralize_wrapper(
        // Switches of TetGen.
        int plc,
        int psc,
        int refine,
        int quality,
        int nobisect,
        int cdt,
        int cdtrefine,
        int coarsen,
        int weighted,
        int brio_hilbert,
        int flipinsert,
        int metric,
        int varvolume,
        int fixedvolume,
        int regionattrib,
        int insertaddpoints,
        int diagnose,
        int convex,
        int nomergefacet,
        int nomergevertex,
        int noexact,
        int nostaticfilter,
        int zeroindex,
        int facesout,
        int edgesout,
        int neighout,
        int voroout,
        int meditview,
        int vtkview,
        int vtksurfview,
        int nobound,
        int nonodewritten,
        int noelewritten,
        int nofacewritten,
        int noiterationnum,
        int nojettison,
        int docheck,
        int quiet,
        int nowarning,
        int verbose,
        int vertexperblock,
        int tetrahedraperblock,
        int shellfaceperblock,
        int supsteiner_level,
        int addsteiner_algo,
        int coarsen_param,
        int weighted_param,
        int fliplinklevel,
        int flipstarsize,
        int fliplinklevelinc,
        int opt_max_flip_level,
        int opt_scheme,
        int opt_iterations,
        int smooth_cirterion,
        int smooth_maxiter,
        int delmaxfliplevel,
        int order,
        int reversetetori,
        int steinerleft,
        int unflip_queue_limit,
        int no_sort,
        int hilbert_order,
        int hilbert_limit,
        int brio_threshold,
        double brio_ratio,
        double epsilon,
        double facet_separate_ang_tol,
        double collinear_ang_tol,
        double facet_small_ang_tol,
        double maxvolume,
        double maxvolume_length,
        double minratio,
        double opt_max_asp_ratio,
        double opt_max_edge_ratio,
        double mindihedral,
        double optmaxdihedral,
        double metric_scale,
        double smooth_alpha,
        double coarsen_percent,
        double elem_growth_ratio,
        double refine_progress_ratio,
        std::string switches_str) {

        tetgenio io_unused = tetgenio();

        tetgenbehavior behavior = tetgenbehavior();
        if (!switches_str.empty()) {
            std::cout << "parsing switches" << std::endl;
            // populate behavior from switches string
            char switches[1024];
            std::strncpy(switches, switches_str.c_str(), sizeof(switches) - 1);
            switches[sizeof(switches) - 1] = '\0'; // Ensure null termination
            bool success = behavior.parse_commandline(switches);
            if (!success) {
                throw std::runtime_error("Unable to parse switches");
            }
            std::cout << "success" << std::endl;

        } else {
            behavior.plc = plc;
            behavior.psc = psc;
            behavior.refine = refine;
            behavior.quality = quality;
            behavior.nobisect = nobisect;
            behavior.cdt = cdt;
            behavior.cdtrefine = cdtrefine;
            behavior.coarsen = coarsen;
            behavior.weighted = weighted;
            behavior.brio_hilbert = brio_hilbert;
            behavior.flipinsert = flipinsert;
            behavior.metric = metric;
            behavior.varvolume = varvolume;
            behavior.fixedvolume = fixedvolume;
            behavior.regionattrib = regionattrib;
            behavior.insertaddpoints = insertaddpoints;
            behavior.diagnose = diagnose;
            behavior.convex = convex;
            behavior.nomergefacet = nomergefacet;
            behavior.nomergevertex = nomergevertex;
            behavior.noexact = noexact;
            behavior.nostaticfilter = nostaticfilter;
            behavior.zeroindex = zeroindex;
            behavior.facesout = facesout;
            behavior.edgesout = edgesout;
            behavior.neighout = (neighout != 0) ? 2 : 0; // weird, needs neighout to be > 1
            behavior.voroout = voroout;
            behavior.meditview = meditview;
            behavior.vtkview = vtkview;
            behavior.vtksurfview = vtksurfview;
            behavior.nobound = nobound;
            behavior.nonodewritten = nonodewritten;
            behavior.noelewritten = noelewritten;
            behavior.nofacewritten = nofacewritten;
            behavior.noiterationnum = noiterationnum;
            behavior.nojettison = nojettison;
            behavior.docheck = docheck;
            behavior.quiet = quiet;
            behavior.nowarning = nowarning;
            behavior.verbose = verbose;
            // Parameters of TetGen.
            behavior.vertexperblock = vertexperblock;
            behavior.tetrahedraperblock = tetrahedraperblock;
            behavior.shellfaceperblock = shellfaceperblock;
            behavior.supsteiner_level = supsteiner_level;
            behavior.addsteiner_algo = addsteiner_algo;
            behavior.coarsen_param = coarsen_param;
            behavior.weighted_param = weighted_param;
            behavior.fliplinklevel = fliplinklevel;
            behavior.flipstarsize = flipstarsize;
            behavior.fliplinklevelinc = fliplinklevelinc;
            behavior.opt_max_flip_level = opt_max_flip_level;
            behavior.opt_scheme = opt_scheme;
            behavior.opt_iterations = opt_iterations;
            behavior.smooth_cirterion = smooth_cirterion;
            behavior.smooth_maxiter = smooth_maxiter;
            behavior.delmaxfliplevel = delmaxfliplevel;
            behavior.order = order;
            behavior.reversetetori = reversetetori;
            behavior.steinerleft = steinerleft;
            behavior.unflip_queue_limit = unflip_queue_limit;
            behavior.no_sort = no_sort;
            behavior.hilbert_order = hilbert_order;
            behavior.hilbert_limit = hilbert_limit;
            behavior.brio_threshold = brio_threshold;
            behavior.brio_ratio = brio_ratio;
            behavior.epsilon = epsilon;
            behavior.facet_separate_ang_tol = facet_separate_ang_tol;
            behavior.collinear_ang_tol = collinear_ang_tol;
            behavior.facet_small_ang_tol = facet_small_ang_tol;
            behavior.maxvolume = maxvolume;
            behavior.maxvolume_length = maxvolume_length;
            behavior.minratio = minratio;
            behavior.opt_max_asp_ratio = opt_max_asp_ratio;
            behavior.opt_max_edge_ratio = opt_max_edge_ratio;
            behavior.mindihedral = mindihedral;
            behavior.optmaxdihedral = optmaxdihedral;
            behavior.metric_scale = metric_scale;
            behavior.smooth_alpha = smooth_alpha;
            behavior.coarsen_percent = coarsen_percent;
            behavior.elem_growth_ratio = elem_growth_ratio;
            behavior.refine_progress_ratio = refine_progress_ratio;
        }

        _finalize_regions();
        _finalize_holes();
        tetrahedralize(&behavior, &io, &io_out, &io_unused, &io_bg);
    }

    // inputs
    int n_faces() { return io.numberoffacets; }
    int n_regions() { return regions.size(); }; // stored in this struct
    int n_holes() { return holes.size(); };     // stored in this struct

    // background mesh
    int n_bg_nodes() { return io_bg.numberofpoints; };

    // outputs
    int n_cells() { return io_out.numberoftetrahedra; }
    int n_cell_attr() { return io_out.numberoftetrahedronattributes; }
    int n_trifaces() { return io_out.numberoftrifaces; }
    int n_nodes() { return io_out.numberofpoints; }

    NDArray<int, 2> return_trifaces() {
        // Returns triangular face connectivity (N x 3) from output.

        int n_faces = io_out.numberoftrifaces;
        NDArray<int, 2> faces_arr = MakeNDArray<int, 2>({n_faces, 3});
        int *faces = faces_arr.data();
        std::memcpy(faces, io_out.trifacelist, sizeof(int) * 3 * n_faces);

        // ensure it's 1 based regardless of internal indexing
        if (io_out.firstnumber == 1) {
            int total = n_faces * 3;
            for (int i = 0; i < total; ++i) {
                --faces[i];
            }
        }

        return faces_arr;
    }

    void load_facet_markers(const NDArray<const int, 1> facetmarkers_arr) {
        int n_faces = facetmarkers_arr.shape(0);
        if (n_faces != io.numberoffacets) {
            throw std::runtime_error(
                "Facet marker count does not match number of faces stored in the mesh");
        }
        io.facetmarkerlist = new int[n_faces];
        std::memcpy(io.facetmarkerlist, facetmarkers_arr.data(), sizeof(int) * n_faces);
    }

    NDArray<int, 1> return_facet_markers() {
        NDArray<int, 1> facetmarkers_arr = MakeNDArray<int, 1>({io.numberoffacets});
        std::memcpy(
            facetmarkers_arr.data(), io.facetmarkerlist, sizeof(int) * io.numberoffacets);
        return facetmarkers_arr;
    }

    void load_hole(std::vector<double> hole) {
        if (hole.size() != 3) {
            throw std::runtime_error("Hole must be of size 3.");
        }
        holes.push_back(hole);
    }

    void _finalize_holes() {
        io.numberofholes = holes.size();
        io.holelist = new double[io.numberofholes * 3];

        for (int i = 0; i < io.numberofholes; ++i) {
            const std::vector<double> &hole = holes[i];
            io.holelist[i * 3 + 0] = hole[0];
            io.holelist[i * 3 + 1] = hole[1];
            io.holelist[i * 3 + 2] = hole[2];
        }
    }

    void load_region(std::vector<double> region) {
        if (region.size() != 5) {
            throw std::runtime_error("Region must be of size 5.");
        }
        regions.push_back(region);
    }

    void _finalize_regions() {
        // Allocate memory for regions and store them to tetgenio
        io.numberofregions = regions.size();
        io.regionlist = new double[io.numberofregions * 5];

        for (int i = 0; i < io.numberofregions; i++) {
            std::vector<double> region = regions[i];
            for (int j = 0; j < 5; j++) {
                io.regionlist[i * 5 + j] = region[j];
            }
        }
    }

    // Returns triangular face markers from tetgen
    NDArray<int, 1> return_triface_markers() {
        // Create python copy of trifacemarkerlist array
        int n_faces = io_out.numberoftrifaces;
        NDArray<int, 1> triface_markers_arr = MakeNDArray<int, 1>({n_faces});
        memcpy(triface_markers_arr.data(), io_out.trifacemarkerlist, sizeof(int) * n_faces);
        return triface_markers_arr;
    }

    NDArray<double, 2> return_tetrahedron_attributes() {
        int nt = io_out.numberoftetrahedra;
        int na = io_out.numberoftetrahedronattributes;

        if (nt < 1 || na < 1 || io_out.tetrahedronattributelist == nullptr) {
            return MakeNDArray<double, 2>({0, 0});
        }

        NDArray<double, 2> arr = MakeNDArray<double, 2>({nt, na});
        std::memcpy(arr.data(), io_out.tetrahedronattributelist, sizeof(double) * nt * na);

        return arr;
    }

    void load_mesh(
        const NDArray<const double, 2> point_arr, const NDArray<const int, 2> face_arr) {
        tetgenio::facet *f;
        tetgenio::polygon *p;

        // Allocate memory for points and store them
        io.numberofpoints = point_arr.shape(0);

        // perform a copy to the tetgen array
        io.pointlist = new double[io.numberofpoints * 3];
        memcpy(io.pointlist, point_arr.data(), sizeof(double) * io.numberofpoints * 3);

        // Store the number of faces and allocate memory
        io.numberoffacets = face_arr.shape(0);
        io.facetlist = new tetgenio::facet[io.numberoffacets];
        const int *faces = face_arr.data();

        // Load in faces as facets
        for (int i = 0; i < io.numberoffacets; i++) {
            // Initialize a face
            f = &io.facetlist[i];
            io.init(f);

            // Each facet has one polygon, no hole, and each polygon has a three vertices
            f->numberofpolygons = 1;
            f->polygonlist = new tetgenio::polygon[1];

            p = &f->polygonlist[0];
            io.init(p);
            p->numberofvertices = 3;
            p->vertexlist = new int[3];
            for (int j = 0; j < 3; j++) {
                p->vertexlist[j] = faces[i * 3 + j];
            }
        }

    } // load_mesh

    NDArray<double, 2> return_input_points() {
        int n_points = io.numberofpoints;
        NDArray<double, 2> arr = MakeNDArray<double, 2>({n_points, 3});
        std::memcpy(arr.data(), io.pointlist, sizeof(double) * n_points * 3);
        return arr;
    }

    NDArray<int, 2> return_input_faces() {
        int n_faces = io.numberoffacets;
        NDArray<int, 2> arr = MakeNDArray<int, 2>({n_faces, 3});
        int *out = arr.data();

        for (int i = 0; i < n_faces; ++i) {
            tetgenio::facet &f = io.facetlist[i];
            tetgenio::polygon &p = f.polygonlist[0];
            out[i * 3 + 0] = p.vertexlist[0];
            out[i * 3 + 1] = p.vertexlist[1];
            out[i * 3 + 2] = p.vertexlist[2];
        }

        return arr;
    }

    // load a background mesh from a filename
    void load_bgmesh_from_filename(std::string bgmesh_filename) {
        char filename[1024];
        std::strncpy(filename, bgmesh_filename.c_str(), sizeof(filename) - 1);
        filename[sizeof(filename) - 1] = '\0';
        io_bg.load_tetmesh(filename, tetgenbehavior::NODES);
    }

    void load_bgmesh_from_arrays(
        const NDArray<const double, 2> v,
        const NDArray<const int, 2> tet,
        const NDArray<const double, 1> mtr) {

        // store points
        if (v.shape(1) != 3) {
            throw std::runtime_error(
                "Invalid point array shape[1] is not 3. Supports 3D only.");
        }
        io_bg.numberofpoints = v.shape(0);
        io_bg.pointlist = new double[io_bg.numberofpoints * 3];
        std::memcpy(io_bg.pointlist, v.data(), sizeof(double) * io_bg.numberofpoints * 3);

        // Load tets (assumes 4 nodes per tetrahedron)
        io_bg.numberoftetrahedra = tet.shape(0);
        if (tet.shape(1) != 4) {
            throw std::runtime_error(
                "Invalid tetrahedral array shape[1] is not 4. Supports linear tetrahedrals "
                "only.");
        }
        io_bg.numberofcorners = 4;
        io_bg.tetrahedronlist = new int[io_bg.numberoftetrahedra * 4];
        io_bg.numberoftetrahedronattributes = 0;
        std::memcpy(
            io_bg.tetrahedronlist, tet.data(), sizeof(int) * io_bg.numberoftetrahedra * 4);

        // Populate pointmtrlist
        io_bg.numberofpointmtrs = 1;
        io_bg.pointmtrlist = new double[io_bg.numberofpoints];
        std::memcpy(io_bg.pointmtrlist, mtr.data(), sizeof(double) * io_bg.numberofpoints);
    }

    // Returns face-to-tetrahedra mapping (N x 2).
    NDArray<int, 2> return_face2tet() {
        if (io_out.face2tetlist == nullptr) {
            return MakeNDArray<int, 2>({0, 2});
        }
        int n_faces = io_out.numberoftrifaces;
        NDArray<int, 2> f2t_arr = MakeNDArray<int, 2>({n_faces, 2});
        std::memcpy(f2t_arr.data(), io_out.face2tetlist, sizeof(int) * n_faces * 2);
        return f2t_arr;
    }

    // Returns edge connectivity (E x 2).
    NDArray<int, 2> return_edges() {
        int n_edges = io_out.numberofedges;
        if (n_edges <= 0 || io_out.edgelist == nullptr) {
            return MakeNDArray<int, 2>({0, 2});
        }

        NDArray<int, 2> edges_arr = MakeNDArray<int, 2>({n_edges, 2});
        std::memcpy(edges_arr.data(), io_out.edgelist, sizeof(int) * n_edges * 2);

        return edges_arr;
    }

    // Returns edge markers (E,).
    NDArray<int, 1> return_edge_markers() {
        int n_edges = io_out.numberofedges;
        if (n_edges <= 0 || io_out.edgemarkerlist == nullptr) {
            return MakeNDArray<int, 1>({0});
        }

        NDArray<int, 1> markers_arr = MakeNDArray<int, 1>({n_edges});
        std::memcpy(markers_arr.data(), io_out.edgemarkerlist, sizeof(int) * n_edges);

        return markers_arr;
    }

}; // PyTetgen

NB_MODULE(_tetgen, m) { // "_tetgen" must match library name from CMakeLists.txt
    nb::class_<PyTetgen>(m, "PyTetgen")
        .def(nb::init<>())
        .def("load_facet_markers", &PyTetgen::load_facet_markers)
        .def("load_mesh", &PyTetgen::load_mesh)
        .def("load_region", &PyTetgen::load_region)
        .def("load_hole", &PyTetgen::load_hole)
        .def("load_bgmesh_from_filename", &PyTetgen::load_bgmesh_from_filename)
        .def("load_bgmesh_from_arrays", &PyTetgen::load_bgmesh_from_arrays)
        .def("return_edges", &PyTetgen::return_edges)
        .def("return_edge_markers", &PyTetgen::return_edge_markers)
        .def("return_face2tet", &PyTetgen::return_face2tet)
        .def("return_facet_markers", &PyTetgen::return_facet_markers)
        .def("return_input_faces", &PyTetgen::return_input_faces)
        .def("return_input_points", &PyTetgen::return_input_points)
        .def("return_nodes", &PyTetgen::return_nodes)
        .def("return_tetrahedron_attributes", &PyTetgen::return_tetrahedron_attributes)
        .def("return_tets", &PyTetgen::return_tets)
        .def("return_triface_markers", &PyTetgen::return_triface_markers)
        .def("return_trifaces", &PyTetgen::return_trifaces)
        .def_prop_ro("n_bg_nodes", &PyTetgen::n_bg_nodes)
        .def_prop_ro("n_cell_attr", &PyTetgen::n_cell_attr)
        .def_prop_ro("n_cells", &PyTetgen::n_cells)
        .def_prop_ro("n_faces", &PyTetgen::n_faces)
        .def_prop_ro("n_holes", &PyTetgen::n_holes)
        .def_prop_ro("n_nodes", &PyTetgen::n_nodes)
        .def_prop_ro("n_regions", &PyTetgen::n_regions)
        .def_prop_ro("n_trifaces", &PyTetgen::n_trifaces)
        .def(
            "tetrahedralize",
            &PyTetgen::tetrahedralize_wrapper,
            nb::arg("plc") = true,
            nb::arg("psc") = false,
            nb::arg("refine") = false,
            nb::arg("quality") = true,
            nb::arg("nobisect") = false,
            nb::arg("cdt") = false,
            nb::arg("cdtrefine") = 7,
            nb::arg("coarsen") = false,
            nb::arg("weighted") = false,
            nb::arg("brio_hilbert") = true,
            nb::arg("flipinsert") = false,
            nb::arg("metric") = false,
            nb::arg("varvolume") = false,
            nb::arg("fixedvolume") = false,
            nb::arg("regionattrib") = false,
            nb::arg("insertaddpoints") = false,
            nb::arg("diagnose") = false,
            nb::arg("convex") = false,
            nb::arg("nomergefacet") = false,
            nb::arg("nomergevertex") = false,
            nb::arg("noexact") = false,
            nb::arg("nostaticfilter") = false,
            nb::arg("zeroindex") = false,
            nb::arg("facesout") = false,
            nb::arg("edgesout") = false,
            nb::arg("neighout") = false,
            nb::arg("voroout") = false,
            nb::arg("meditview") = false,
            nb::arg("vtkview") = false,
            nb::arg("vtksurfview") = false,
            nb::arg("nobound") = false,
            nb::arg("nonodewritten") = false,
            nb::arg("noelewritten") = false,
            nb::arg("nofacewritten") = false,
            nb::arg("noiterationnum") = false,
            nb::arg("nojettison") = false,
            nb::arg("docheck") = false,
            nb::arg("quiet") = false,
            nb::arg("nowarning") = false,
            nb::arg("verbose") = 0,
            nb::arg("vertexperblock") = 4092,
            nb::arg("tetrahedraperblock") = 8188,
            nb::arg("shellfaceperblock") = 2044,
            nb::arg("supsteiner_level") = 2,
            nb::arg("addsteiner_algo") = 1,
            nb::arg("coarsen_param") = 0,
            nb::arg("weighted_param") = 0,
            nb::arg("fliplinklevel") = -1,
            nb::arg("flipstarsize") = -1,
            nb::arg("fliplinklevelinc") = 1,
            nb::arg("opt_max_flip_level") = 3,
            nb::arg("opt_scheme") = 7,
            nb::arg("opt_iterations") = 3,
            nb::arg("smooth_cirterion") = 1,
            nb::arg("smooth_maxiter") = 7,
            nb::arg("delmaxfliplevel") = 1,
            nb::arg("order") = 1,
            nb::arg("reversetetori") = 0,
            nb::arg("steinerleft") = 100000,
            nb::arg("unflip_queue_limit") = 1000,
            nb::arg("no_sort") = false,
            nb::arg("hilbert_order") = 52,
            nb::arg("hilbert_limit") = 8,
            nb::arg("brio_threshold") = 64,
            nb::arg("brio_ratio") = 0.125,
            nb::arg("epsilon") = 1.0e-8,
            nb::arg("facet_separate_ang_tol") = 179.9,
            nb::arg("collinear_ang_tol") = 179.9,
            nb::arg("facet_small_ang_tol") = 15.0,
            nb::arg("maxvolume") = -1.0,
            nb::arg("maxvolume_length") = -1.0,
            nb::arg("minratio") = 2.0,
            nb::arg("opt_max_asp_ratio") = 1000.0,
            nb::arg("opt_max_edge_ratio") = 100.0,
            nb::arg("mindihedral") = 0.0,
            nb::arg("optmaxdihedral") = 177.0,
            nb::arg("metric_scale") = 1.0,
            nb::arg("smooth_alpha") = 0.3,
            nb::arg("coarsen_percent") = 1.0,
            nb::arg("elem_growth_ratio") = 0.0,
            nb::arg("refine_progress_ratio") = 0.333,
            nb::arg("switches_str") = "");
}
