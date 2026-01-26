import numpy as np
from numpy.typing import NDArray

class PyTetgen:
    def __init__(self) -> None: ...
    def load_mesh(self, point_arr: NDArray[np.float64], face_arr: NDArray[np.int32]) -> None: ...
    def tetrahedralize(
        self,
        plc: int = ...,
        psc: int = ...,
        refine: int = ...,
        quality: int = ...,
        nobisect: int = ...,
        cdt: int = ...,
        cdtrefine: int = ...,
        coarsen: int = ...,
        weighted: int = ...,
        brio_hilbert: int = ...,
        flipinsert: int = ...,
        metric: int = ...,
        varvolume: int = ...,
        fixedvolume: int = ...,
        regionattrib: int = ...,
        insertaddpoints: int = ...,
        diagnose: int = ...,
        convex: int = ...,
        nomergefacet: int = ...,
        nomergevertex: int = ...,
        noexact: int = ...,
        nostaticfilter: int = ...,
        zeroindex: int = ...,
        facesout: int = ...,
        edgesout: int = ...,
        neighout: int = ...,
        voroout: int = ...,
        meditview: int = ...,
        vtkview: int = ...,
        vtksurfview: int = ...,
        nobound: int = ...,
        nonodewritten: int = ...,
        noelewritten: int = ...,
        nofacewritten: int = ...,
        noiterationnum: int = ...,
        nojettison: int = ...,
        docheck: int = ...,
        quiet: int = ...,
        nowarning: int = ...,
        verbose: int = ...,
        vertexperblock: int = ...,
        tetrahedraperblock: int = ...,
        shellfaceperblock: int = ...,
        supsteiner_level: int = ...,
        addsteiner_algo: int = ...,
        coarsen_param: int = ...,
        weighted_param: int = ...,
        fliplinklevel: int = ...,
        flipstarsize: int = ...,
        fliplinklevelinc: int = ...,
        opt_max_flip_level: int = ...,
        opt_scheme: int = ...,
        opt_iterations: int = ...,
        smooth_cirterion: int = ...,
        smooth_maxiter: int = ...,
        delmaxfliplevel: int = ...,
        order: int = ...,
        reversetetori: int = ...,
        steinerleft: int = ...,
        unflip_queue_limit: int = ...,
        no_sort: int = ...,
        hilbert_order: int = ...,
        hilbert_limit: int = ...,
        brio_threshold: int = ...,
        brio_ratio: float = ...,
        epsilon: float = ...,
        facet_separate_ang_tol: float = ...,
        collinear_ang_tol: float = ...,
        facet_small_ang_tol: float = ...,
        maxvolume: float = ...,
        maxvolume_length: float = ...,
        minratio: float = ...,
        opt_max_asp_ratio: float = ...,
        opt_max_edge_ratio: float = ...,
        mindihedral: float = ...,
        optmaxdihedral: float = ...,
        metric_scale: float = ...,
        smooth_alpha: float = ...,
        coarsen_percent: float = ...,
        elem_growth_ratio: float = ...,
        refine_progress_ratio: float = ...,
        switches_str: str = "",
    ) -> None: ...
    def return_nodes(self) -> NDArray[np.float64]: ...
    def return_tets(self, vtk_indexing: bool = False) -> NDArray[np.int32]: ...
    @property
    def n_cells(self) -> int: ...
    @property
    def n_cell_attr(self) -> int: ...
    @property
    def n_faces(self) -> int: ...
    @property
    def n_regions(self) -> int: ...
    @property
    def n_holes(self) -> int: ...
    @property
    def n_trifaces(self) -> int: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_bg_nodes(self) -> int: ...
    def load_facet_markers(self, facet_markers: NDArray[np.int32]) -> None: ...
    def load_region(self, regions: list[float]) -> None: ...
    def return_facet_markers(self) -> NDArray[np.int32]: ...
    def return_triface_markers(self) -> NDArray[np.int32]: ...
    def return_trifaces(self) -> NDArray[np.int32]: ...
    def return_tetrahedron_attributes(self) -> NDArray[np.float64]: ...
    def return_input_points(self) -> NDArray[np.float64]: ...
    def return_input_faces(self) -> NDArray[np.int32]: ...
    def load_hole(self, hole: list[float]) -> None: ...
    def load_bgmesh_from_filename(self, bgmeshfilename: str) -> None: ...
    def load_bgmesh_from_arrays(
        self,
        bgmesh_v: NDArray[np.float64],
        bgmesh_tet: NDArray[np.int32],
        bgmesh_mtr: NDArray[np.float64],
    ) -> None: ...
    def return_face2tet(self) -> NDArray[np.int32]: ...
    def return_edges(self) -> NDArray[np.int32]: ...
    def return_edge_markers(self) -> NDArray[np.int32]: ...
