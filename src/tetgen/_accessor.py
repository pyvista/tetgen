"""Register a ``.tetgen`` accessor on :class:`pyvista.PolyData`.

Importing this module (which :mod:`tetgen` does on package import)
attaches a :class:`TetGenAccessor` so every :class:`~pyvista.PolyData`
instance exposes the ``.tetgen`` namespace.

Requires PyVista >= 0.48, which introduced ``register_dataset_accessor``.
On older versions importing this module is a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pyvista as pv

if TYPE_CHECKING:
    from tetgen.pytetgen import TetGen


HAS_ACCESSOR_REGISTRY = hasattr(pv, "register_dataset_accessor")


def _register(cls):
    if HAS_ACCESSOR_REGISTRY:
        return pv.register_dataset_accessor("tetgen", pv.PolyData)(cls)
    return cls


@_register
class TetGenAccessor:
    """Tetrahedralization accessor for surface meshes.

    Wraps the :class:`tetgen.TetGen` tetrahedralizer so a PyVista user
    can call it directly on any :class:`~pyvista.PolyData` surface::

        import pyvista as pv
        import tetgen  # noqa: F401 — registers the ``.tetgen`` accessor

        grid = pv.Sphere().tetgen.tetrahedralize(order=1, mindihedral=20)

    The underlying :class:`~tetgen.TetGen` instance is constructed
    lazily on first use and cached. Access it via :attr:`instance`
    when you need the lower-level API (raw arrays, markers, edges).
    """

    def __init__(self, mesh: pv.PolyData) -> None:
        """Bind the accessor to its parent :class:`~pyvista.PolyData`."""
        self._mesh = mesh
        self._tetgen: TetGen | None = None

    @property
    def instance(self) -> TetGen:
        """Return the underlying :class:`tetgen.TetGen` object.

        Constructed lazily on first access. Use this for the raw array
        outputs (``node``, ``elem``, ``triface_markers``) or the
        ``.make_manifold()`` preprocessor.
        """
        if self._tetgen is None:
            from tetgen import TetGen  # noqa: PLC0415

            self._tetgen = TetGen(self._mesh)
        return self._tetgen

    def tetrahedralize(self, **kwargs: Any) -> pv.UnstructuredGrid:
        """Tetrahedralize the surface mesh and return the volume grid.

        Forwards all keyword arguments to
        :meth:`tetgen.TetGen.tetrahedralize`. Returns the resulting
        :class:`~pyvista.UnstructuredGrid` directly so the call chains
        cleanly with PyVista filters::

            pv.Sphere().tetgen.tetrahedralize(order=1).extract_cells([0, 1, 2])

        See :meth:`tetgen.TetGen.tetrahedralize` for the full parameter
        list.
        """
        self.instance.tetrahedralize(**kwargs)
        return self.instance.grid

    def make_manifold(self, **kwargs: Any) -> pv.PolyData:
        """Repair non-manifold input and return the cleaned surface.

        Forwards to :meth:`tetgen.TetGen.make_manifold`. See that
        method's docstring for parameters.
        """
        self.instance.make_manifold(**kwargs)
        return self.instance.mesh
