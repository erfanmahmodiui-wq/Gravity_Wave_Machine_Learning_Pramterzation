import os

import cartopy.crs as ccrs
import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xarray as xr
from matplotlib.collections import PolyCollection
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import binned_statistic
import os
import sys
sys.path.append(os.path.abspath(".."))
from model.model import read_scale_data
from src.config import load_paths


def plot_global_map(var, save_dir, grid_file, filename="global_flux_map", title=r'$\sum_i (N)$',
                     cmap='Reds', vmin=None, vmax=None):
    gridfil = xr.open_dataset(grid_file)
    clon_vertices = np.rad2deg(gridfil.clon_vertices.values)
    clat_vertices = np.rad2deg(gridfil.clat_vertices.values)

    print("=" * 50)
    print("VARIABLE DISTRIBUTION SUMMARY")
    print("=" * 50)
    print(f"Mean   : {np.nanmean(var):.4f}")
    print(f"Median : {np.nanmedian(var):.4f}")
    print(f"Std    : {np.nanstd(var):.4f}")
    print(f"Min    : {np.nanmin(var):.4f}")
    print(f"Max    : {np.nanmax(var):.4f}")
    print("=" * 50)

    if vmin is None:
        vmin = np.nanmin(var)
    if vmax is None:
        vmax = np.nanmax(var)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(var))

    clon_vertices = np.where(clon_vertices < -180., clon_vertices + 360., clon_vertices)
    clon_vertices = np.where(clon_vertices > 180., clon_vertices - 360., clon_vertices)
    triangles = np.stack([clon_vertices, clat_vertices], axis=-1)

    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_global()
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='dimgray', alpha=0.4, zorder=2)

    coll = PolyCollection(triangles, array=None, fc=colors, edgecolors='none',
                           linewidth=0.05, transform=ccrs.Geodetic(), zorder=0)
    ax.add_collection(coll)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.8, pad=0.04, aspect=30)
    cbar.set_label(title, fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"--> saved to {outpath}")


def plot_global_map_new(var, save_dir, grid_file, filename="global_flux_map", title="Global Flux",
                         cmap='RdYlBu_r', vmin=None, vmax=None,
                         percentile_clip=2.0, symm=True):
    gridfil = xr.open_dataset(grid_file)
    clon_vertices = np.rad2deg(gridfil.clon_vertices.values)
    clat_vertices = np.rad2deg(gridfil.clat_vertices.values)

    print("=" * 50)
    print("VARIABLE DISTRIBUTION SUMMARY")
    print("=" * 50)
    print(f"Mean   : {np.nanmean(var):.4f}")
    print(f"Median : {np.nanmedian(var):.4f}")
    print(f"Std    : {np.nanstd(var):.4f}")
    print(f"Min    : {np.nanmin(var):.4f}")
    print(f"Max    : {np.nanmax(var):.4f}")
    print(f"p{percentile_clip:.0f}   : {np.nanpercentile(var, percentile_clip):.4f}")
    print(f"p{100-percentile_clip:.0f}  : {np.nanpercentile(var, 100-percentile_clip):.4f}")
    print("=" * 50)

    if vmin is None or vmax is None:
        vmin_pc = np.nanpercentile(var, percentile_clip)
        vmax_pc = np.nanpercentile(var, 100 - percentile_clip)
        if symm:
            vlim = max(abs(vmin_pc), abs(vmax_pc))
            vmin, vmax = -vlim, vlim
        else:
            vmin, vmax = vmin_pc, vmax_pc

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(var))

    clon_vertices = np.where(clon_vertices < -180., clon_vertices + 360., clon_vertices)
    clon_vertices = np.where(clon_vertices > 180., clon_vertices - 360., clon_vertices)
    triangles = np.stack([clon_vertices, clat_vertices], axis=-1)

    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_global()
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='dimgray', alpha=0.4, zorder=2)

    coll = PolyCollection(triangles, array=None, fc=colors, edgecolors='none',
                           linewidth=0.05, transform=ccrs.Geodetic(), zorder=0)
    ax.add_collection(coll)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.8, pad=0.04, aspect=30, extend='both')
    cbar.set_label(title, fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    ax.text(0.01, 0.01,
            f"Colorbar clipped to [{vmin:.2f}, {vmax:.2f}] mPa  "
            f"(p{percentile_clip:.0f}\u2013p{100-percentile_clip:.0f})",
            transform=ax.transAxes, fontsize=7, color='gray', va='bottom')

    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"--> saved to {outpath}")


def compute_intrinsic_feq_cgz(data_base_dir_jan, data_base_dir_aug, coarse_action_dir_jan):
    ntime = 44
    intrinsic_freq = np.zeros((20480, ntime, 144))
    cgz = np.zeros((20480, ntime, 144))

    for tag in ["ocean", "land"]:
        with xr.open_dataset(coarse_action_dir_jan + "coarse_gw_spectrum_group_1_fileslide_0.nc") as ds:
            k_lunch = ds.k.data
            l_lunch = ds.l.data
            m_lunch = ds.m.data

        fname = "corsed_action_land.pkl" if tag == "land" else "corsed_action_ocean.pkl"
        Data_Base_jan = joblib.load(data_base_dir_jan + fname)
        cgz_jan = Data_Base_jan["cgz_test"]
        omega_hat_jan = Data_Base_jan["omega_hat_test"]
        ntri          = Data_Base_jan["ntri"]


        Data_Base_aug = joblib.load(data_base_dir_aug + fname)
        cgz_aug = Data_Base_aug["cgz_test"]
        omega_hat_aug = Data_Base_aug["omega_hat_test"]

        omega_hat = np.concatenate([omega_hat_jan, omega_hat_aug], axis=1)
        c_gz       = np.concatenate([cgz_jan, cgz_aug], axis=1)
        
        intrinsic_freq[ntri, :, ] = omega_hat.reshape(len(ntri), ntime, 144)
        cgz[ntri, ...] = c_gz.reshape(len(ntri), ntime, 144)

    return cgz, intrinsic_freq, k_lunch, l_lunch, m_lunch


def filename_decomposed(sign, grnr, var="u", decomposed=True):
    if decomposed:
        return "decomposed_" + var + "_gw_" + sign + "_group-" + str(grnr) + ".nc"
    else:
        return "sp3D-" + var + "-ny4_window_group-" + str(grnr) + ".nc"


def read_decomposition(datadir, filename, orig=False):
    ds = xr.open_dataset(datadir + filename)
    ntri = ds["ntri"].values
    k = ds["k"].values
    l = ds["l"].values
    if orig:
        l = l / 1000
        k = k / 1000
    omega = ds['omega'].values
    maxind = ds["max_ind"][:, :].values
    alt = ds["z"].values
    return k, l, omega, maxind, ntri, alt


def parsavel_avarge_cofficents(decomposed_dir, tri_list_dir):
    fn_mask = "triangles_list-sphere"
    tri_list = np.load(tri_list_dir + fn_mask + ".npy", allow_pickle=True)
    prs_coff = np.zeros((20480))

    for gri in range(160):
        fname = filename_decomposed(sign='plus', grnr=gri, decomposed=True)
        k_all, l_all, omega, maxind, ntri, alt = read_decomposition(decomposed_dir, fname)
        for tri_i in range(len(ntri)):
            tri = ntri[tri_i]
            [maxik, maxil] = maxind[tri_i, :]
            coff = len(k_all[tri_i, :maxik]) * len(l_all[tri_i, :maxil]) / 144
            prs_coff[tri] = coff / (len(tri_list[tri]) * 144)
    return prs_coff[:, np.newaxis]


def compute_wave_action_density(ml_model_dir, save_dir):
    ntime = 44
    global_N_model = np.zeros((20480, ntime, 144))
    global_N_test = np.zeros((20480, ntime, 144))
    for tag in ["land", "ocean"]:
        fname_model = f"cnn_coarsed_{tag}.keras"
        (_, _, _, _, X_test, y_test, _, ntri, _, _, _, 
         _, _, _, _, _, _) = read_scale_data(
            model_save_dir=save_dir,
            land_ocean_tag=tag
        )
        model = tf.keras.models.load_model(
            ml_model_dir + fname_model,
            custom_objects=False, compile=False)
        y_pred_test = model.predict(X_test)

        y_test_real = np.exp(y_test) - 1
        y_test_real = y_test_real.reshape((len(ntri), ntime, 144))
        y_pred_test_real = np.exp(y_pred_test) - 1
        y_pred_test_real = y_pred_test_real.reshape((len(ntri), ntime, 144))
        global_N_test[ntri, :, :] = y_test_real
        global_N_model[ntri, :, :] = y_pred_test_real
    return global_N_model, global_N_test


def compute_energy_flux(global_N_model, global_N_test, cgz, intrinsic_freq, parsavel_coff, k_launch, l_launch):
    ntime = global_N_model.shape[1]
    cgz_bin = cgz.reshape(20480, ntime, 6, 6, 4)
    Energy_test = np.sum(global_N_test * intrinsic_freq, axis=-1)
    Energy_model = np.sum(global_N_model * intrinsic_freq, axis=-1)
    k3D = k_launch[None, None, :, None, None]
    l3D = l_launch[None, None, None, :, None]
    N_global_test_bin = global_N_test.reshape(20480, ntime, 6, 6, 4)
    N_global_model_bin = global_N_model.reshape(20480, ntime, 6, 6, 4)
    uw_test = np.sum(N_global_test_bin * k3D * cgz_bin, axis=(2, 3, 4)) * parsavel_coff
    uw_model = np.sum(N_global_model_bin * k3D * cgz_bin, axis=(2, 3, 4)) * parsavel_coff
    vw_test = np.sum(N_global_test_bin * l3D * cgz_bin, axis=(2, 3, 4)) * parsavel_coff
    vw_model = np.sum(N_global_model_bin * l3D * cgz_bin, axis=(2, 3, 4)) * parsavel_coff
    return Energy_test, Energy_model, uw_test, uw_model, vw_test, vw_model


def zonally_avarge(var, grid_file):
    gridfil = xr.open_dataset(grid_file)
    clat = np.rad2deg(gridfil.clat.values)
    lat_bins = np.linspace(clat.min(), clat.max(), 101)
    zonal_mean = np.zeros((100, var.shape[1]))

    for k in range(var.shape[1]):
        zonal_mean[:, k], _, _ = binned_statistic(
            clat, var[:, k], statistic='mean', bins=lat_bins)
    return zonal_mean, lat_bins


def plot_zonal_mean_avg(var_test, var_model, lat_bins, save_dir):
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(lat_centers, var_test * 1000, lw=2.5, label='ICON')
    ax.plot(lat_centers, var_model * 1000, lw=2.5, ls='--', label='ML model')

    ax.set_xlabel('Latitude (\u00b0)')
    ax.set_ylabel('Momentum flux (mPa)')
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.legend(frameon=False, loc='best')

    plt.tight_layout()
    plt.savefig(save_dir + "zonal_flux.png")


def physical_values():
    paths = load_paths()

    cgz, intrinsic_freq, k, l, m = compute_intrinsic_feq_cgz(
        data_base_dir_jan=paths["data_base_dir_jan"],
        data_base_dir_aug=paths["data_base_dir_aug"],
        coarse_action_dir_jan=paths["coarse_action_dir_jan"],
    )
    parsavel_coff = parsavel_avarge_cofficents(
        decomposed_dir=paths["decomposed_dir"], tri_list_dir=paths["tri_list_dir"])
    global_N_model, global_N_test = compute_wave_action_density(
        ml_model_dir=paths["ml_model_dir"], save_dir=paths["save_dir"])
    Energy_test, Energy_model, uw_test, uw_model, vw_test, vw_model = compute_energy_flux(
        global_N_model, global_N_test, cgz, intrinsic_freq, parsavel_coff, k, l)
    uw_model_zonal_mean, _ = zonally_avarge(var=uw_model, grid_file=paths["grid_file_low"])
    uw_test_zonal_mean, lat_bins = zonally_avarge(var=uw_test, grid_file=paths["grid_file_low"])
    vw_model_zonal_mean, _ = zonally_avarge(var=vw_model, grid_file=paths["grid_file_low"])
    vw_test_zonal_mean, lat_bins = zonally_avarge(var=vw_test, grid_file=paths["grid_file_low"])
    return (Energy_test, Energy_model, uw_test, uw_model, vw_test, vw_model,
            uw_model_zonal_mean, uw_test_zonal_mean, vw_model_zonal_mean, vw_test_zonal_mean, lat_bins)


if __name__ == "__main__":
    paths = load_paths()

    (Energy_test, Energy_model, uw_test, uw_model, vw_test, vw_model,
    uw_model_zonal_mean, uw_test_zonal_mean, vw_model_zonal_mean,
    vw_test_zonal_mean, lat_bins) = physical_values()

    metrics_dir = paths["flux_energy_metrics_dir"]
    np.save(metrics_dir + "Energy_test.npy", Energy_test)
    np.save(metrics_dir + "Energy_model.npy", Energy_model)
    np.save(metrics_dir + "flux_test.npy", uw_test)
    np.save(metrics_dir + "flux_model.npy", uw_model)
    np.save(metrics_dir + "vw_test.npy", vw_test)
    np.save(metrics_dir + "vw_model.npy", vw_model)
    np.save(metrics_dir + "uw_model_zonal_mean.npy", uw_model_zonal_mean)
    np.save(metrics_dir + "uw_test_zonal_mean.npy", uw_test_zonal_mean)
    np.save(metrics_dir + "vw_model_zonal_mean.npy", vw_model_zonal_mean)
    np.save(metrics_dir + "vw_test_zonal_mean.npy", vw_test_zonal_mean)
    np.save(metrics_dir + "lat_bins.npy", lat_bins)
