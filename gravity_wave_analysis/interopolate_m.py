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
from coarsing_action import save_interpolated

def find_max_min_m(groupnr, data_dir):
    m_max = []
    m_min = []
    m_at_zer_list = []
    for file_slide in range(73):
        fname = f"interp-alt0-fileslide_{file_slide}-regroup-{groupnr}.nc"
        with xr.open_dataset(data_dir + fname) as ds:
            m_file = ds.m.data
            m_max.append(np.max(m_file))
            m_min.append(np.min(m_file))
            # also track closest to zero
            m_minus = m_file[m_file < 0]
            m_plus  = m_file[m_file > 0]
            m_at_zer_list.append(max(np.abs(m_minus.max()), np.abs(m_plus.min())))
        print(f"  file_slide={file_slide}: m=[{m_file.min():.4e}, {m_file.max():.4e}]")

    global_min_m  = np.min(m_min)
    global_max_m  = np.max(m_max)
    global_at_zer = np.max(m_at_zer_list)  # smallest "near zero" across all slides

    print(f"\n  global m range: [{global_min_m:.4e}, {global_max_m:.4e}]")
    print(f"  global m_at_zero: {global_at_zer:.4e}")

    return global_min_m, global_max_m, global_at_zer


def build_common_m(global_min_m, global_max_m, global_at_zer, Nm_new=24):
    """Build a single common m grid to use across all file slides."""
    n_half = Nm_new // 2

    m_common_minus = -np.logspace(
        np.log10(np.abs(global_min_m)),
        np.log10(np.abs(global_at_zer)),
        n_half
    )
    m_common_plus = +np.logspace(
        np.log10(np.abs(global_at_zer)),
        np.log10(np.abs(global_max_m)),
        n_half
    )

    m_common = np.concatenate([np.sort(m_common_minus), np.sort(m_common_plus)])
    dm_common = np.abs(np.gradient(m_common))

    print(f"  m_common n={len(m_common)}, range=[{m_common.min():.4e}, {m_common.max():.4e}]")
    print(f"  m_common: {m_common}")

    return m_common, dm_common


def interpolate_m_only(N, k, l, m, dm, m_common, dm_common):
    """
    Interpolate N from per-file m grid onto common m grid.
    N shape: (ntime, ntri, l, k, m)
    Only interpolates in m, keeps k and l unchanged.
    """
    dk = np.mean(np.diff(k))
    dl = np.mean(np.diff(l))

    # to continuous
    N_continuous = N / (dk * dl * dm[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :])

    n_time, n_tri, nl, nk, nm = N_continuous.shape
    Nm_new = len(m_common)

    print(f"  interpolating m: {nm} -> {Nm_new}")
    print(f"  m_file:   [{m.min():.4e}, {m.max():.4e}]")
    print(f"  m_common: [{m_common.min():.4e}, {m_common.max():.4e}]")

    N_out = np.zeros((n_time, n_tri, nl, nk, Nm_new))

    for ti in range(n_time):
        for tri_i in range(n_tri):
            if tri_i % 256 == 0:
                print(f"  interp time={ti}, triangle={tri_i}/{n_tri}")

            for il in range(nl):
                for ik in range(nk):
                    N_col = N_continuous[ti, tri_i, il, ik, :]  # (m,)

                    valid = np.isfinite(N_col)
                    if valid.sum() < 2:
                        N_out[ti, tri_i, il, ik, :] = 0.0
                        continue

                    interp = interpolate.interp1d(
                        m[valid], N_col[valid],
                        kind='linear',
                        bounds_error=False,
                        fill_value=0.0
                    )
                    N_out[ti, tri_i, il, ik, :] = interp(m_common)


    # back to bin-integrated
    N_out = N_out * dk * dl * dm_common[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    # conservation check
    total_before = np.nansum(N[N > 1e-10])
    total_after  = np.nansum(N_out)
    print(f"  Total wave action before: {total_before:.6e}")
    print(f"  Total wave action after:  {total_after:.6e}")
    print(f"  Conservation ratio:       {total_after/total_before:.6f}")

    return N_out

def load_action_spectrum(wave_action_dir, groupnr, file_slide):
    fname = f"interp-alt0-fileslide_{file_slide}-regroup-{groupnr}.nc"
    print(f"file loaded {fname}")
    with xr.open_dataset(wave_action_dir + fname) as ds:
        print(f"  dims: {ds.data.dims}, shape: {ds.data.shape}")
        N = ds.data.data.astype(np.float64)
        k = ds.k.data
        l = ds.l.data
        m = ds.m.data
        ntri = ds.ntri.data
        time = ds.time.data



    print(f"  N_raw min positive: {N[N > 0].min():.4e}")
    print(f"  N_raw histogram: {np.histogram(np.log10(N[N > 0]), bins=10)}")
    # mask fill values
    N[N <= 1e-10] = 0

    dk = np.mean(np.diff(k))
    dl = np.mean(np.diff(l))
    dm = np.abs(np.gradient(m))

    N_continuous = N / (dk * dl * dm[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :])

    print(f"  k: len={len(k)}, range=[{k.min():.4e}, {k.max():.4e}], dk={dk:.4e}")
    print(f"  l: len={len(l)}, range=[{l.min():.4e}, {l.max():.4e}], dl={dl:.4e}")
    print(f"  m: len={len(m)}, range=[{m.min():.4e}, {m.max():.4e}]")
    print(f"  dm: min={dm.min():.4e}, max={dm.max():.4e}, sum={dm.sum():.4e}")
    print(f"  N valid points: {np.sum(np.isfinite(N))} / {N.size}")
    print(f"  N_continuous: min={np.nanmin(N_continuous):.4e}, max={np.nanmax(N_continuous):.4e}")
    print(f"  N_intgrate: min={np.nanmin(N):.4e}, max={np.nanmax(N):.4e}")
    print(f"  Total wave action (binned): {np.nansum(N):.6e}")

    return dk, dl, dm, N_continuous, k, l, m, ntri, time, N


# --- main usage ---
if __name__ ==  "__main__":
    groupnr = 8
    wave_action_dir = "/work/bb1097/b383521/dummy/interp_wave_action_density/"
    save_path       = "/work/bb1097/b383521/dummy/interp_wave_action_density_unim/"
    global_min_m, global_max_m, global_at_zer = find_max_min_m(groupnr, wave_action_dir)
    m_common, dm_common = build_common_m(global_min_m, global_max_m, global_at_zer, Nm_new=24)

    for file_slide in range(73):
        print(f"files_slide : {file_slide}")
        dk, dl, dm, N_continuous, k, l, m, ntri, time, N_raw = load_action_spectrum(wave_action_dir, groupnr, file_slide)
        N_interp = interpolate_m_only(N_raw, k, l, m, dm, m_common, dm_common)
        save_interpolated( save_path=save_path, intdata=N_interp, k=k, l=l, m_new=m_common, tri=ntri, varname="Wave action density", time=time, file_slide=file_slide, groupnr=groupnr)