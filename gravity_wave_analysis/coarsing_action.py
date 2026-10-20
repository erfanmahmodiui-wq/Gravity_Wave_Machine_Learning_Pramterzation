import numpy as np
import xarray as xr
import gc
import constants
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import scipy.interpolate as interpolate
import multiprocessing as mp

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import numpy as np
import xarray as xr

import numpy as np
import xarray as xr


def load_action_spectrum(wave_action_dir, groupnr, file_slide):
    fname = f"interp-alt0-fileslide_{file_slide}-regroup-{groupnr}.nc"
    print(f"coars graining wave action density file slide {fname}")
    with xr.open_dataset(wave_action_dir + fname) as ds:
        print(f"  dims: {ds.data.dims}, shape: {ds.data.shape}")
        N = ds.data.data.astype(np.float64)
        k = ds.k.data
        l = ds.l.data
        m = ds.m.data
        ntri = ds.ntri.data
        time = ds.time.data
        m_negative = m[:12]
        print(f"max m : {np.max(m_negative)}, min m {np.min(m_negative)}")
        ds.close()
    return N, k, l, m, ntri, time

def bin_wave_action_spectrum(N, k_indices, l_indices, m_indices):
    """
    Bin-average wave action density spectrum N over coarse k, l, m bins.

    Parameters
    ----------
    N         : np.ndarray, shape (time, triangle, l, k, m)
    k_indices : list of 4 arrays — fine k indices falling in each coarse k bin
    l_indices : list of 4 arrays — fine l indices falling in each coarse l bin
    m_indices : list of 4 arrays — fine m indices falling in each coarse m bin

    Returns
    -------
    N_coarse  : np.ndarray, shape (time, triangle, n_l, n_k, n_m)
                NaN where a bin is empty.
    """
    n_time, n_tri, n_l_fine, n_k_fine, n_m_fine = N.shape

    n_k = len(k_indices)   # 4
    n_l = len(l_indices)   # 4
    n_m = len(m_indices)   # 4

    N_coarse = np.full((n_time, n_tri, n_l, n_k, n_m), np.nan)

    for ik, k_idx in enumerate(k_indices):
        for il, l_idx in enumerate(l_indices):
            for im, m_idx in enumerate(m_indices):

                # skip empty bins
                if len(k_idx) == 0 or len(l_idx) == 0 or len(m_idx) == 0:
                    continue

                # extract the sub-block: shape (time, triangle, n_l_bin, n_k_bin, n_m_bin)
                sub = N[
                    :,          # time
                    :,          # triangle
                    l_idx[:, None, None],   # l bin  → broadcast
                    k_idx[None, :, None],   # k bin  → broadcast
                    m_idx[None, None, :]    # m bin  → broadcast
                ]
                # sub.shape = (time, triangle, len(l_idx), len(k_idx), len(m_idx))

                # average over the 3 fine dims (axes -3, -2, -1)
                N_coarse[:, :, il, ik, im] = sub.mean(axis=(-3, -2, -1))

    return N_coarse


def build_launch_grid(k, l, m):
    
    # --- wavelength limits ---
    lambda_min = 2e3   # 2 km
    lambda_max = 18e3    # 18 km

    m_min_abs = 2 * np.pi / lambda_max   # smallest |m| (longest λ)
    m_max_abs = 2 * np.pi / lambda_min   # largest |m| (shortest λ)

    print(f"  |m| range for λz in [2,18] km: [{m_min_abs:.4e}, {m_max_abs:.4e}]")

    # --- filter valid m (negative only, inside band) ---
    m_neg_mask = (m < 0) & (np.abs(m) >= m_min_abs) & (np.abs(m) <= m_max_abs)
    m_neg = m[m_neg_mask]

    print(f"  m values in band: {m_neg}")
    print(f"  λz values: {2*np.pi/np.abs(m_neg)/1000} km")

    # --- launch grid (k,l unchanged) ---
    k_launch = np.array([-9.23802555e-05, -4.61901277e-05, +4.61901277e-05, +9.23802555e-05])
    l_launch = np.array([-9.23802555e-05, -4.61901277e-05, +4.61901277e-05, +9.23802555e-05])

    # --- m launch (log spaced inside band) ---
    m_launch = np.sort(-np.logspace(np.log10(m_min_abs), np.log10(m_max_abs), 3))


    print(f"  m_launch targets: {m_launch}")
    print(f"  m_launch λz: {2*np.pi/np.abs(m_launch)/1000} km")

    # --- helper to make bin edges ---
    def make_edges(x_launch):
        return np.concatenate([
            [2*x_launch[0]  - x_launch[1]],
            (x_launch[1:] + x_launch[:-1]) / 2,
            [2*x_launch[-1] - x_launch[-2]]
        ])

    k_edges = make_edges(k_launch)
    l_edges = make_edges(l_launch)
    m_edges = make_edges(m_launch)

    print(f"  k_edges: {k_edges}")
    print(f"  l_edges: {l_edges}")
    print(f"  m_edges: {m_edges}")

    # --- assign k bins ---
    k_indices = [
        np.where((k >= k_edges[ik]) & (k < k_edges[ik+1]))[0]
        for ik in range(len(k_launch))
    ]
    k_indices[-1] = np.where((k >= k_edges[-2]) & (k <= k_edges[-1]))[0]

    # --- assign l bins ---
    l_indices = [
        np.where((l >= l_edges[il]) & (l < l_edges[il+1]))[0]
        for il in range(len(l_launch))
    ]
    l_indices[-1] = np.where((l >= l_edges[-2]) & (l <= l_edges[-1]))[0]

    # --- assign m bins (ONLY inside valid band) ---
    m_indices = [
        np.where((m >= m_edges[im]) & (m < m_edges[im+1]) & m_neg_mask)[0]
        for im in range(len(m_launch))
    ]
    m_indices[-1] = np.where((m >= m_edges[-2]) & (m <= m_edges[-1]) & m_neg_mask)[0]

    k_indices = [np.array(idx) for idx in k_indices]
    l_indices = [np.array(idx) for idx in l_indices]
    m_indices = [np.array(idx) for idx in m_indices]

    # --- coarse centers (mean of fine points) ---
    #k_coarse = np.array([
    #    k[idx][k[idx] !=0 ].mean() if len(idx) > 0 else k_launch[ik]
    #    for ik, idx in enumerate(k_indices)
    #])
    
    #l_coarse = np.array([
    #    l[idx][l[idx] != 0].mean() if len(idx) > 0 else l_launch[il]
    #    for il, idx in enumerate(l_indices)
    #])
    
    k_coarse = k_launch.copy()
    l_coarse = l_launch.copy()
    m_coarse = m_launch.copy()
    #m_coarse = np.array([
    #    m[idx].mean() if len(idx) > 0 else m_launch[im]
    #    for im, idx in enumerate(m_indices)
    #])

    # --- weights / widths ---
    # REPLACE with this:
    dk = np.mean(np.diff(k))
    dl = np.mean(np.diff(l))
    dm_const = (m_max_abs - m_min_abs) / 4   # constant, uniform m bins

    dk_new = np.array([len(idx) * dk  for idx in k_indices])
    dl_new = np.array([len(idx) * dl  for idx in l_indices])
    dm_new = np.full(4, dm_const)             # same width for all 4 m bins

    # --- debug prints ---
    print(f"  dk_new: {dk_new}")
    print(f"  dl_new: {dl_new}")
    print(f"  dm_new: {dm_new}")
    print(f"  volume check (sum dk*dl*dm): {np.sum(dk_new) * np.sum(dl_new) * np.sum(dm_new):.4e}")
    print(f"  original k span: {np.sum(np.ones(len(k))*dk):.4e}, l span: {np.sum(np.ones(len(l))*dl):.4e}")

    print(f"  k bin counts: {[len(x) for x in k_indices]}, sum={sum(len(x) for x in k_indices)}/{len(k)}")
    print(f"  l bin counts: {[len(x) for x in l_indices]}, sum={sum(len(x) for x in l_indices)}/{len(l)}")
    print(f"  m bin counts: {[len(x) for x in m_indices]}, sum={sum(len(x) for x in m_indices)}/{len(m_neg)}")

    print(f"  k_coarse: {k_coarse}")
    print(f"  l_coarse: {l_coarse}")
    print(f"  m_coarse: {m_coarse}")
    print(f"  m_coarse λz: {2*np.pi/np.abs(m_coarse)/1000} km")

    return k_coarse, l_coarse, m_coarse, k_indices, l_indices, m_indices, dk_new, dl_new, dm_new


def save_interpolated(save_path,intdata, k, l, m_new, tri, varname, time, file_slide, groupnr):
    ds=xr.Dataset(data_vars=dict(
            data=(["ntime","ntri","k","l","m"], intdata),
            k=(["nk"], k),
            l=(["nl"], l),
            m=(["nm"], m_new),
        ),
        coords=dict(
            ntri=np.array(tri), #Indices of the original triangles
            time = time[:]
        ),
        attrs=dict(description="Interpolated data for the group",
                    varname=varname,),
        )
    ds.to_netcdf(save_path+f"interp-alt0-fileslide_{file_slide}-regroup-{groupnr}.nc")
    ds.close()

# --- main ---
if __name__ == "__main__":
    wave_action_dir = "/work/bb1097/b383521/dummy/interp_wave_action_density/"
    save_path       = "/work/bb1097/b383521/dummy/coarsed_wave_action_density/"
    

    for groupnr in range(10):
        for file_slide in range(73):
            print(f"coars graining wave action density file slide {file_slide}")
            N, k, l, m, ntri, time = load_action_spectrum(wave_action_dir, groupnr, file_slide)
            k_coarse, l_coarse, m_coarse, k_indices, l_indices, m_indices, dk_new, dl_new, dm_new = build_launch_grid(k, l, m)
            N_lunch = bin_wave_action_spectrum(N, k_indices=k_indices, l_indices=l_indices, m_indices=m_indices)
            print(f"ration of nan : {np.sum(np.isnan(N_lunch)/np.sum(~np.isnan(N_lunch)))}")
            mask = np.isnan(N_lunch)
            N_lunch[mask] = 0.0
            save_interpolated(save_path, N_lunch, k_coarse, l_coarse, m_coarse, tri=ntri, varname="Wave Action Density Spetrum", time=time, file_slide=file_slide, groupnr=groupnr)

