import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs

from src.config import load_paths


def test_averages_noread(fn_grid_low, var, figpath, label):
    gridfil = xr.open_dataset(fn_grid_low)
    clon_vertices = np.rad2deg(gridfil.clon_vertices.values)
    clat_vertices = np.rad2deg(gridfil.clat_vertices.values)

    var = var.flatten()

    ncells, nv = clon_vertices.shape[0], clon_vertices.shape[1]
    if len(var) != ncells:
        raise ValueError(f"Mismatch: var has {len(var)} values but grid has {ncells} cells")

    varMin = np.nanmin(var)
    varMax = np.nanmax(var)

    varNum = 32
    dx = (varMax - varMin) / varNum
    levels = np.linspace(varMin - dx, varMax + dx, varNum + 1)
    nlevs = levels.size

    projection = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=projection))
    ax.set_global()
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='dimgray', alpha=0.4, zorder=2)

    cmap = plt.get_cmap('Reds', nlevs)
    colors = np.ones([ncells, 4], np.float32) * 0.9

    for m in range(1, nlevs - 1):
        mask = (var >= levels[m]) & (var < levels[m + 1])
        colors[mask, :] = cmap(m / nlevs)

    mask_low = var < varMin
    colors[mask_low, :] = [1, 1, 1, 1]

    clon_vertices = np.where(clon_vertices < -180., clon_vertices + 360., clon_vertices)
    clon_vertices = np.where(clon_vertices > 180., clon_vertices - 360., clon_vertices)

    triangles = np.zeros((ncells, nv, 2), np.float32)
    for i in range(ncells):
        triangles[i, :, 0] = clon_vertices[i, :]
        triangles[i, :, 1] = clat_vertices[i, :]

    coll = PolyCollection(triangles, array=None, facecolors=colors, edgecolors='none',
                           linewidth=0.0, transform=ccrs.Geodetic(), zorder=0)
    ax.add_collection(coll)

    cb = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=varMin, vmax=varMax))
    cb.set_array([])

    cbar = plt.colorbar(cb, ax=ax, orientation='horizontal', format='%.2e',
                         shrink=0.8, pad=0.04, aspect=30)

    cbar.ax.tick_params(axis='x', labelrotation=30)
    cbar.set_label(label, fontsize=12)

    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close()


def global_map_energy(path_grid_low, var, figpath, label):
    test_averages_noread(path_grid_low, var, figpath, label)
    return


def save_grad(grad, grad_mag, label, resolved_flow_dir):
    ds = xr.Dataset(
        data_vars=dict(
            gradian=(["ntri", "time", "heigts", "del_x_y"], grad),
            gradian_mag=(["ntri", "time", "height"], grad_mag),
        ),
        attrs=dict(description="Gradian"),
    )
    fname = "gradian_" + label + "_2016.nc"
    ds.to_netcdf(resolved_flow_dir + fname)


def save_vorticity(vorticity, resolved_flow_dir):
    ds = xr.Dataset(
        data_vars=dict(gradian=(["ntri", "time", "heigts"], vorticity)),
        attrs=dict(description="voorticity_2016"),
    )
    fname = "vorticity_2016.nc"
    ds.to_netcdf(resolved_flow_dir + fname)


def Gree_gauss(path_grid, avarge_path, variable, plots_dir):
    ds_grid = xr.open_dataset(path_grid)
    cell_area = ds_grid.cell_area.data
    nx = ds_grid.zonal_normal_primal_edge.data
    ny = ds_grid.meridional_normal_primal_edge.data
    n_vec = np.stack([nx, ny], axis=-1)
    edge_of_cell = ds_grid.edge_of_cell.data - 1
    adj_cell = ds_grid.adjacent_cell_of_edge.data - 1
    edge_len = ds_grid.edge_length.data
    edge_dist = ds_grid.edge_cell_distance.data

    ds_var = xr.open_dataset(avarge_path + "average_" + variable + "-sphere_2016.nc")
    var = ds_var["var"].data
    nt, nz, nCell = var.shape
    grad = np.zeros((nCell, nt, nz, 2))

    for c in range(nCell):
        grad_sum = np.zeros((nt, nz, 2))
        for i in range(3):
            e = edge_of_cell[i, c]
            cells_e = adj_cell[:, e]
            n_edge = n_vec[e, :]

            if cells_e[0] == c:
                n_local = n_edge
                dp = edge_dist[0, e]
                df = edge_dist[1, e]
                c_other = cells_e[1]
            else:
                n_local = -n_edge
                dp = edge_dist[1, e]
                df = edge_dist[0, e]
                c_other = cells_e[0]

            w_p = df / (dp + df)
            w_f = dp / (dp + df)
            u_edge = w_p * var[:, :, c] + w_f * var[:, :, c_other]

            grad_sum[..., 0] += u_edge * edge_len[e] * n_local[0]
            grad_sum[..., 1] += u_edge * edge_len[e] * n_local[1]

        grad[c, ...] = grad_sum / cell_area[c]

    grad_mag = np.sqrt(grad[..., 0]**2 + grad[..., 1]**2)
    save_grad(grad=grad, grad_mag=grad_mag, label=variable, resolved_flow_dir=avarge_path)
    global_map_energy(path_grid_low=path_grid, var=grad_mag[:, 0, 0],
                       figpath=plots_dir + "gradian_" + variable,
                       label=fr"$\|\nabla {variable}\|$")


def compute_vorticity(path_grid, resolved_flow_dir, plots_dir):
    ds_u = xr.open_dataset(resolved_flow_dir + "gradian_u_2016.nc")
    ds_v = xr.open_dataset(resolved_flow_dir + "gradian_v_2016.nc")
    du = ds_u.gradian.data
    dv = ds_v.gradian.data
    vorticity = dv[:, :, :, 0] - du[:, :, :, 1]
    save_vorticity(vorticity, resolved_flow_dir)
    global_map_energy(path_grid_low=path_grid, var=vorticity[:, 0, 0],
                       figpath=plots_dir + "vorticity", label=fr"$\omega_z$")


if __name__ == "__main__":
    paths = load_paths()
    path_grid    = paths["grid_file_low"]
    avarge_path  = paths["resolved_flow_dir"]
    plots_dir    = paths["plots_dir"]

    for variable in ["u", "v", "w", "p", "T"]:
        print(f"computing variable {variable}")
        Gree_gauss(path_grid, avarge_path=avarge_path, variable=variable, plots_dir=plots_dir)
    compute_vorticity(path_grid=path_grid, resolved_flow_dir=avarge_path, plots_dir=plots_dir)
