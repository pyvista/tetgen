from vtkInterface import examples
import vtkInterface as vtki
import tetgen
import numpy as np


def test_tetrahedralize():
    sphere = vtki.PolyData(examples.spherefile)
    tet = tetgen.TetGen(sphere)
    tet.Tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    assert grid.GetNumberOfCells()
    assert grid.GetNumberOfPoints()
    assert np.all(grid.quality > 0)
    return grid


def functional_tet():
    grid = test_tetrahedralize()
    # get cell centroids
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.ExtractSelectionCells(cell_ind)

    # plot this
    subgrid.Plot(scalars=subgrid.quality, stitle='quality', colormap='bwr',
                 flipscalars=True)

    # advanced plotting
    plotter = vtki.PlotClass()
    plotter.SetBackground('w')
    plotter.AddMesh(subgrid, 'lightgrey', lighting=True)
    plotter.AddMesh(grid, 'r', 'wireframe')
    plotter.AddLegend([[' Input Mesh ', 'r'],
                       [' Tesselated Mesh ', 'black']])
    plotter.Plot()

    plotter = vtki.PlotClass()
    plotter.SetBackground('w')
    plotter.AddMesh(grid, 'r', 'wireframe')
    plotter.Plot(autoclose=False)
    plotter.Plot(autoclose=False, interactive_update=True)
    for i in range(500):
        single_cell = grid.ExtractSelectionCells([i])
        plotter.AddMesh(single_cell)
        plotter.Update()
    plotter.Close()

if __name__ == '__main__':
    # functional_tet()
    test_tetrahedralize()
    print('PASS')
