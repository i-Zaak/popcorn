import numpy as np
import pandas as pd
import nibabel.freesurfer.io as fio
import os

try:
    import pyvista as pv
except:
    print('Warning: pyvista not installed, some plots will not work')

try:
    import datashader as ds
except:
    print('Warning: datashader not installed, some plots will not work')

def plot_sparse(sp_mat):
    sp_coo = sp_mat.tocoo()
    cvs = ds.Canvas(plot_width=500, plot_height=500)
    agg = cvs.points(
        pd.DataFrame(dict(col=sp_coo.col, row=sp_coo.row, data=sp_coo.data)),
        x='col', y='row'
    )
    img = ds.tf.shade(agg, how='log')
    return img


def load_mesh(path, lh='lh.inflated_ico5', rh='rh.inflated_ico5'):
    v, t = fio.read_geometry(os.path.join(path,lh))
    v[:,0] -= v[:,0].max()
    faces = np.full((t.shape[0], 4), 3 )
    faces[:,1:] = t
    mesh_lh = pv.PolyData(v, faces)

    v, t = fio.read_geometry(os.path.join(path,rh))
    v[:,0] -= v[:,0].min()

    faces = np.full((t.shape[0], 4), 3 )
    faces[:,1:] = t
    mesh_rh = pv.PolyData(v, faces)

    return mesh_lh, mesh_rh


def plot_brain( mesh, values=None, plotter=None, show=True):
    assert len(mesh) == 2, 'provide meshes for exactly two hemispheres'
    mesh_lh, mesh_rh = mesh
    if plotter is None:
        plotter = pv.Plotter()
    if values is not None:
        assert mesh_lh.n_points == mesh_lh.n_points, "expect same number of vertices for both hemispheres"
        assert len(values) == 2 * mesh_lh.n_points, "mismatched number of values and mesh vertices"
        n_verts = mesh_lh.n_points
        mesh_lh['ROI'] = values[:n_verts]
        mesh_rh['ROI'] = values[n_verts:]
        plotter.add_mesh(mesh_lh, scalars="ROI", cmap="plasma")
        plotter.add_mesh(mesh_rh, scalars="ROI", cmap="plasma")
    else:
        plotter.add_mesh(mesh_lh)
        plotter.add_mesh(mesh_rh)

    if show:
        plotter.show()
    return plotter

def plot_brain_multiview(mesh, values=None, clip=0.008):
    plotter = pv.Plotter(shape=(2, 2))
    if clip is not None:
        values = values.copy()
        values[values>clip] = 0.008

    plotter.subplot(0, 0)
    plot_brain(mesh, values, show=False, plotter=plotter)
    plotter.view_yz(negative=True)

    plotter.subplot(1, 0)
    plot_brain(mesh, values, show=False, plotter=plotter)
    plotter.view_yz(negative=False)

    plotter.subplot(0, 1)
    plot_brain(mesh, values, show=False, plotter=plotter)
    plotter.view_xz(negative=False)

    plotter.subplot(1, 1)
    plot_brain(mesh, values, show=False, plotter=plotter)
    plotter.view_xy(negative=True)

    plotter.show()
    return plotter


