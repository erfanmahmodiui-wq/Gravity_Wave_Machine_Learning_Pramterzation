import numpy as np
import multiprocessing as mp
import xarray as xr
import os
from datetime import datetime
import math
from functools import partial
import time
from pathlib import Path
from pyjokes import get_joke
from timespectrum import run_all_variables_parallel
from regroup_for_PCA import regroup
from interp_kl import interpolate_kl
from decomposition import decompose
from m_inter import vertical_interp
from EOF import compute_EOFs, compute_avrgae_save, read_interp_data
from anaylisi_EOF import list_sort_interp_action
import pickle
from wave_action import action_coarse
import shutil

def ntri_orgraghy_nonorgrghy_index(resolved_flow_dirctory , action_dir, group_id):
    ## read ntri index of the group_id, and then find the index of orography and non-orography
    with xr.open_dataset(action_dir + f"interp-alt0-fileslide_0-regroup-{group_id}.nc", engine="h5netcdf") as ds:
        ntri = ds.ntri.data
    
    with xr.open_dataset(resolved_flow_dirctory + "orography.nc", engine="h5netcdf") as ds:
        values = ds.Hspec.data[ntri, :]
        oro_sorted = -np.sort(-values, axis=1)
        ntri_orography = ntri[np.where(oro_sorted[:, 0] > 0)[0]]
        ntri_nonorography = ntri[np.where(oro_sorted[:, 0] == 0)[0]]
        ds.close()
    return ntri_orography, ntri_nonorography

def read_avrage():
    ds = xr.open_dataset("/work/bb1097/b383521/slided_window_avrage/EOF/pca_new_eigen_group_4.nc")
    avg = ds.avrage.data
    return avg

def compute_avrage_EOF_all():
    variables = ["u", "v", "w", "p", "b"]
    # Process each file_slide
    
    for fileslide in range(0, 1):
        print(f"\n{'='*60}")
        print(f"Processing file_slide {0}")
        print(f"{'='*60}")
        print("\n[STEP 1/2] Computing 3D spectra...")
        start_spectral = time.time()
        run_all_variables_parallel(file_slide=fileslide)
        end_spectral = time.time()
        print(f"Spectral computation took {end_spectral - start_spectral:.2f}s")


        #Step 2: Regroup files for all variables IN PARALLEL
        print("\n[STEP 2/2] Regrouping files in parallel...")
        start_regroup = time.time()
        
        with mp.Pool(processes=5) as pool:  # 5 processes for 5 variables
            pool.map(regroup, variables)
        
        end_regroup = time.time()
        print(f"Regrouping took {end_regroup - start_regroup:.2f}s")

        # delete 160 groups not neccsaary anymore
        print("deleting old groups")
        for var in variables:
            directory = Path("/work/bb1097/b383521/slided_window_avrage/3D_spr_old_group/" + var + "/")
            for file in directory.iterdir():
                if file.is_file():
                    file.unlink()  # unlink() deletes the file
                    print(f"Deleted: {file.name}")

        for var in ["u", "v", "w", "p", "b"]:
            regroup(var)
    
        groupid = 4
        for var in ["b"]:
            print(f"horizental interpolation of variable {var}")
            interpolate_kl(groupid, var)
        
        
        # delete 10 groups not neccsaary anymore
        print("deleting old groups")
        for var in variables:
            directory = Path("/work/bb1097/b383521/slided_window_avrage/regroup/" + var + "/")
            for file in directory.iterdir():
                if file.is_file():
                    file.unlink()  # unlink() deletes the file
                    print(f"Deleted: {file.name}")


        decompose(groupnr=4, file_slide=fileslide)


        vertical_interp(group_id=groupid, file_slide=fileslide)

        compute_avrgae_save(groupnr=groupid, file_slide=fileslide)

def save_mean(group_id, l, k, m, mean_f, name_oro):

    savedir = "/work/bb1097/b383521/dummy/avg_full_new/"
    os.makedirs(savedir, exist_ok=True)

    # Clean dtype
    mean_f = np.asarray(mean_f, dtype=np.float32)

    # Build dataset
    ds = xr.Dataset(
        data_vars={
            "avrage": ("spectral_index", mean_f),
        },
        coords={
            "spectral_index": np.arange(len(mean_f)),
            "k": ("k", k),
            "l": ("l", l),
            "m": ("m", m),
        },
        attrs={
            "group_id": group_id,
            "name_oro": name_oro,
            "original_shape": f"({len(l)}, {len(k)}, {len(m)})",
        },
    )

    output_path = (
        savedir
        + f"avarage-action-log-{name_oro}-{group_id}.nc"
    )

    # Remove old file if exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Save (no HDF5 = no crash)
    ds.to_netcdf(output_path, engine="scipy")

    ds.close()

    print("Saved:", output_path)

    return output_path
                 
def avrage_all(path, group_id, ntri_orography, ntri_nonorography):
    print(f"computing and saving avrage of group {group_id}")
    avg_list_oro = []
    avg_list_nonoro = []
    for file_slide in range(0, 54):
        print(f"file slide {file_slide}")
        file_name = "interp-alt0-fileslide_" + str(file_slide) + "-regroup-" + str(group_id) + ".nc"
        with xr.open_dataset(path + file_name, engine="h5netcdf") as ds:
            k = ds.k.data
            l = ds.l.data
            m = ds.m.data
            nt = ds.time.data
            ntri = ds.ntri.data
            index_orography = np.where(np.isin(ntri, ntri_orography))[0]
            index_nonorography = np.where(np.isin(ntri, ntri_nonorography))[0]
            N = ds.data.data
            N_oro = np.log(N[:, index_orography, ...] + 1)
            N_nonoro = np.log(N[:, index_nonorography, ...] + 1)
            print(f"shape of N_oro in file {file_name} is {N_oro.shape}")
            print(f"shape of N_nonoro in file {file_name} is {N_nonoro.shape}")
            N_reshape_oro = N_oro.reshape(len(nt)*len(index_orography), len(l)*len(k)*len(m))
            N_reshape_nonoro = N_nonoro.reshape(len(nt)*len(index_nonorography), len(l)*len(k)*len(m))
            N_avg_oro = np.mean(N_reshape_oro, axis=0)
            N_avg_nonoro = np.mean(N_reshape_nonoro, axis=0)
            avg_list_oro.append(N_avg_oro)
            avg_list_nonoro.append(N_avg_nonoro)

    avrage_full_oro = np.zeros((len(avg_list_oro), len(avg_list_oro[0])))
    avrage_full_nonoro = np.zeros((len(avg_list_nonoro), len(avg_list_nonoro[0])))
    for i in range(len(avg_list_oro)):
        avrage_full_oro[i, :] = avg_list_oro[i]
        avrage_full_nonoro[i, :] = avg_list_nonoro[i]

    avrage_full_oro = np.mean(avrage_full_oro, axis=0)
    avrage_full_nonoro = np.mean(avrage_full_nonoro, axis=0)

    save_mean(group_id=group_id, l=l, k=k, m=m, mean_f=avrage_full_oro, name_oro="orography")
    save_mean(group_id=group_id, l=l, k=k, m=m, mean_f=avrage_full_nonoro, name_oro="nonorography")

def eigen_decomposition(cov_matrix):
    eigvals, eigvects = np.linalg.eig(cov_matrix)
    indx_eigvals = np.argsort(eigvals)[::-1]
    eigvals = eigvals[indx_eigvals]
    eigvects = eigvects[:, indx_eigvals]
    return eigvals.real, eigvects.real

def save_EOF(eigvals, eigvects, l, k, m, EOF_save_dir, groupnr, name_oro):

    os.makedirs(EOF_save_dir, exist_ok=True)

    # Clean dtypes
    eigvals = np.asarray(eigvals, dtype=np.float32)
    eigvects = np.asarray(eigvects, dtype=np.float32)

    neig = len(eigvals)
    lkm = eigvects.shape[1]

    # Build dataset
    ds = xr.Dataset(
        data_vars={
            "eigenvalues": ("neig", eigvals),
            "eigenvectors": (["lkm", "neig"], eigvects),
        },
        coords={
            "neig": np.arange(neig),
            "lkm": np.arange(lkm),
            "k": ("k", k),
            "l": ("l", l),
            "m": ("m", m),
        },
        attrs={
            "description": "Eigenvalues and eigenvectors from the PCA",
            "groupnr": groupnr,
            "name_oro": name_oro,
            "original_shape": f"({len(l)}, {len(k)}, {len(m)})",
        },
    )

    output_path = (
        EOF_save_dir
        + f"EOFs_logscale_{name_oro}_{groupnr}.nc"
    )

    # Remove old file if exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Save (stable backend)
    ds.to_netcdf(output_path, engine="scipy")

    ds.close()

    print("Saved:", output_path)


def cov_all(avg_dir, action_dir, save_EOF_dir, groupid, ntri_orography, ntri_nonorography):
    avg_fname_oro = "avarage-action-log-orography-" + str(groupid) + ".nc"
    avg_fname_nonoro = "avarage-action-log-nonorography-" + str(groupid) + ".nc"
    with xr.open_dataset(avg_dir+avg_fname_oro, engine="scipy") as ds_average:
        avarage_oro = ds_average.avrage.data
        k = ds_average.k.data
        l = ds_average.l.data
        m = ds_average.m.data
        ds_average.close()

    with xr.open_dataset(avg_dir+avg_fname_nonoro, engine="scipy") as ds_average:
        avarage_nonoro = ds_average.avrage.data
        ds_average.close()

    cov_full_oro = np.zeros((len(l)*len(k)*len(m), len(l)*len(k)*len(m)))
    cov_full_nonoro = np.zeros((len(l)*len(k)*len(m), len(l)*len(k)*len(m)))
    #lis = list_sort_interp_action(interp_dir=action_dir, group_id=groupid)
    for file_slide in range(54):
        fname = "interp-alt0-fileslide_" + str(file_slide) + "-regroup-" + str(groupid) + ".nc"
        with xr.open_dataset(action_dir+fname, engine="h5netcdf") as ds_interp_action:
            print(f"computing covariance for file {fname}")
            N = ds_interp_action.data.data
            ntri = ds_interp_action.ntri.data
            index_orography = np.where(np.isin(ntri, ntri_orography))[0]
            index_nonorography = np.where(np.isin(ntri, ntri_nonorography))[0]
            N_oro = np.log(N[:, index_orography, ...] + 1)
            N_nonoro = np.log(N[:, index_nonorography, ...] + 1)
            k = ds_interp_action.k.data
            l = ds_interp_action.l.data
            m = ds_interp_action.m.data
            time = ds_interp_action.time.data
        N_reshape_oro = np.reshape(N_oro, (len(time)*len(ntri_orography), len(l)*len(k)*len(m)))
        N_reshape_nonoro = np.reshape(N_nonoro, (len(time)*len(ntri_nonorography), len(l)*len(k)*len(m)))
        print(f"shape of interpolated wave action density in file : {fname} with this reshape form of {N_reshape_oro.shape}")
        N_avg_subtracted_oro = N_reshape_oro - avarage_oro
        N_avg_subtracted_nonoro = N_reshape_nonoro - avarage_nonoro
        cov_oro = np.cov(N_avg_subtracted_oro, rowvar=False)
        cov_full_oro = cov_full_oro + cov_oro
        cov_full_nonoro = cov_full_nonoro + np.cov(N_avg_subtracted_nonoro, rowvar=False)
        print(f"shape of covariance {cov_oro.shape}")
        print(f"shape of full covariance {cov_full_nonoro.shape}")

    eigenvalues_oro, eigenvectors_oro = eigen_decomposition(cov_full_oro)
    save_EOF(eigvals=eigenvalues_oro, eigvects=eigenvectors_oro, l=l, k=k, m=m, EOF_save_dir=save_EOF_dir, groupnr=groupid, name_oro="orography")
    eigenvalues_nonoro, eigenvectors_nonoro = eigen_decomposition(cov_full_nonoro)
    save_EOF(eigvals=eigenvalues_nonoro, eigvects=eigenvectors_nonoro, l=l, k=k, m=m, EOF_save_dir=save_EOF_dir, groupnr=groupid, name_oro="nonorography")

def time_coff(avg_dir, action_dir, EOF_dir, time_coff_dir, num_components, group_id, ntri_orography, ntri_nonorography):
    num_component_oro = num_components[0]
    num_component_nonoro = num_components[1]
    avg_fname_oro = "avarage-action-log-orography-" + str(group_id) + ".nc"
    avg_fname_nonoro = "avarage-action-log-nonorography-" + str(group_id) + ".nc"
    with xr.open_dataset(avg_dir+avg_fname_oro) as ds_average:
        avarage_oro = ds_average.avrage.data
        k = ds_average.k.data
        l = ds_average.l.data
        m = ds_average.m.data
        ds_average.close()

    with xr.open_dataset(avg_dir+avg_fname_nonoro) as ds_average:
        avarage_nonoro = ds_average.avrage.data
        ds_average.close()

    EOF_oro_name = "EOFs_logscale_orography_" + str(group_id) + ".nc"
    EOF_nonoro_name = "EOFs_logscale_nonorography_" + str(group_id) + ".nc"
    with xr.open_dataset(EOF_dir+EOF_oro_name) as ds_EOF:
        EOFs_oro = ds_EOF.eigenvectors.data
        ds_EOF.close()
    with xr.open_dataset(EOF_dir+EOF_nonoro_name) as ds_EOF:
        EOFs_nonoro = ds_EOF.eigenvectors.data
        ds_EOF.close()
    
    #lis = list_sort_interp_action(interp_dir=action_dir, group_id=group_id)
    pca_coff_oro = np.zeros((865, len(ntri_orography), num_component_oro))
    pca_coff_nonoro = np.zeros((865, len(ntri_nonorography), num_component_nonoro))
    Time = np.zeros((865))
    for file_slide in range(73):
        fname = "interp-alt0-fileslide_" + str(file_slide) + "-regroup-" + str(group_id) + ".nc"
        print(f"PCA COFF file {fname}")
        with xr.open_dataset(action_dir+fname, engine="h5netcdf") as ds_interp_action:
            N = ds_interp_action.data.data
            k = ds_interp_action.k.data
            l = ds_interp_action.l.data
            m = ds_interp_action.m.data
            time = ds_interp_action.time.data
            ntri = ds_interp_action.ntri.data
            ds_interp_action.close()
        index_orography = np.where(np.isin(ntri, ntri_orography))[0]
        index_nonorography = np.where(np.isin(ntri, ntri_nonorography))[0]
        N_oro = np.log(1 + N[:, index_orography, ...])
        N_nonoro = np.log(1 + N[:, index_nonorography, ...])
        N_oro_reshape = np.reshape(N_oro, (len(time)*len(index_orography), len(l)*len(k)*len(m)))
        N_nonoro_reshape = np.reshape(N_nonoro, (len(time)*len(index_nonorography), len(l)*len(k)*len(m)))
        print(f"shape of interpolated wave action density in file : {fname} with this reshape form of {N_oro_reshape.shape}")
        N_subtract_oro = N_oro_reshape - avarage_oro
        N_subtract_nonoro = N_nonoro_reshape - avarage_nonoro
        N_reshape_subtract_oro = np.reshape(N_subtract_oro, (len(time), len(index_orography), len(l)*len(k)*len(m)))
        N_reshape_subtract_nonoro = np.reshape(N_subtract_nonoro, (len(time), len(index_nonorography), len(l)*len(k)*len(m)))
        at_oro = N_reshape_subtract_oro @ EOFs_oro  [:, :num_component_oro]
        at_nonoro = N_reshape_subtract_nonoro @ EOFs_nonoro  [:, :num_component_nonoro]
        at_oro_reshape = np.reshape(at_oro, (len(time), len(index_orography), num_component_oro))
        at_nonoro_reshape = np.reshape(at_nonoro, (len(time), len(index_nonorography), num_component_nonoro))
        start_idx = file_slide*12
        end_idx   = start_idx + len(time)
        Time[start_idx:end_idx] = time
        pca_coff_oro[start_idx:end_idx, :, :] = at_oro_reshape
        pca_coff_nonoro[start_idx:end_idx, :, :] = at_nonoro_reshape

    save_time_coeff(time_coeff=pca_coff_oro, k=k, l=l, m=m, time=np.array(Time), ntri=ntri_orography, file_slide=file_slide, save_dir=time_coff_dir, group_id=group_id, name_oro="orography")
    save_time_coeff(time_coeff=pca_coff_nonoro, k=k, l=l, m=m, time=np.array(Time), ntri=ntri_nonorography, file_slide=file_slide, save_dir=time_coff_dir, group_id=group_id, name_oro="nonorography")

def compute_variance(group_id, EOF_path):
    fname_oro = "EOFs_logscale_orography_" + str(group_id) + ".nc"
    with xr.open_dataset(EOF_path+fname_oro, engine="scipy") as ds:
        variance_oro = ds.eigenvalues.data
        portion_oro = variance_oro / np.sum(variance_oro)
        ds.close()
    
    fname_nonoro = "EOFs_logscale_nonorography_" + str(group_id) + ".nc"
    with xr.open_dataset(EOF_path+fname_nonoro, engine="scipy") as ds:
        variance_nonoro = ds.eigenvalues.data
        portion_nonoro = variance_nonoro / np.sum(variance_nonoro)
        ds.close()
    
    cumulative_oro = np.cumsum(portion_oro)
    cumulative_nonoro = np.cumsum(portion_nonoro)
    n_total_or = len(cumulative_oro)
    n_total_nonoro = len(cumulative_nonoro)
    n_pcs_60_oro = np.argmax(cumulative_oro >= 0.60) + 1
    n_pcs_60_nonoro = np.argmax(cumulative_nonoro >= 0.60) + 1
    n_pcs_70_oro = np.argmax(cumulative_oro >= 0.70) + 1
    n_pcs_90_oro = np.argmax(cumulative_oro >= 0.90) + 1
    n_pcs_70_nonoro = np.argmax(cumulative_nonoro >= 0.70) + 1
    n_pcs_90_nonoro = np.argmax(cumulative_nonoro >= 0.90) + 1
    
    return group_id,n_pcs_60_oro, n_pcs_60_nonoro, n_pcs_70_oro, n_pcs_90_oro, n_pcs_70_nonoro, n_pcs_90_nonoro, portion_oro, portion_nonoro, n_total_or, n_total_nonoro


def save_time_coeff(time_coeff, k, l, m, time, ntri, file_slide, save_dir, group_id, name_oro):

    os.makedirs(save_dir, exist_ok=True)

    # Ensure clean dtype
    time_coeff = np.asarray(time_coeff, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    l = np.asarray(l, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    ntri = np.asarray(ntri, dtype=np.int32)

    # Build dataset
    ds = xr.Dataset(
        data_vars={
            "PCA_coff": (["time", "ntri", "n_components"], time_coeff),
            "l": ("l", l),
            "k": ("k", k),
            "m": ("m", m)
        },
        coords={
            "time": time,
            "ntri": ntri,
        },
        attrs={
            "description": "time coefficients",
            "group_id": group_id,
            "name_oro": name_oro,
        }
    )

    output_path = os.path.join(
        save_dir,
        f"{name_oro}-logscale-time-coff_{group_id}.nc"
    )

    # Remove old file if exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Save (stable, cluster-safe)
    ds.to_netcdf(output_path, engine="scipy")

    ds.close()

    print("Saved:", output_path)

def main3(fileslide):
    
    variables = ["u", "v", "w", "p", "b"]
    
    print(f"\n{'='*60}")
    print(f"Processing file_slide {fileslide}")
    print(f"{'='*60}")

    print("\n[STEP 1/3] Computing 3D spectra...")
    run_all_variables_parallel(file_slide=fileslide)

    print("\n[STEP 2/3] Decomposing all groups...")
    for group_id in range(160): # change this
        print(f"  decomposition group {group_id}...")
        decompose(groupnr=group_id, file_slide=fileslide)
    
    print("\n[STEP 3/3] Coarse graining...")
    #action_coarse(file_slide=fileslide)

    # ── cleanup ───────────────────────────────────────────────
    print("\nCleaning up old_group files...")
    shutil.rmtree(f"/work/bb1097/b383521/GW_Data/3D_spr_old_group/slide_{fileslide}/")

    print("Cleaning up decomposition files...")
    #shutil.rmtree(f"/work/bb1097/b383521/GW_Data/decomposition/slide_{fileslide}/")


def main2():

    variables = ["u", "v", "w", "p", "b"]
    # Process each file_slide
    avg_dir = "/work/bb1097/b383521/dummy/avarage_of_all/"
    action_dir = "/work/bb1097/b383521/dummy/inter_wave_action_density_filled_log/"
    for fileslide in [72]:
        print(f"\n{'='*60}")
        print(f"Processing file_slide {fileslide}")
        print(f"{'='*60}")
        print("\n[STEP 1/2] Computing 3D spectra...")
        run_all_variables_parallel(file_slide=fileslide)
        
        #Step 2: Regroup files for all variables IN PARALLEL
        print("\n[STEP 2/2] Regrouping files in parallel...") 
        with mp.Pool(processes=5) as pool:  # 5 processes for 5 variables
            pool.map(regroup, variables)
        
    
    
        print("deleting old groups")
        for var in variables:
            directory = Path("/work/bb1097/b383521/dummy/3D_spr_old_group/" + var + "/")
            for file in directory.iterdir():
                if file.is_file():
                    file.unlink()  # unlink() deletes the file
                    print(f"Deleted: {file.name}")


        for groupid in range(10):

            for var in variables:
                print(f"horizental interpolation of variable {var}")
                interpolate_kl(groupid, var)
            
            
            decompose(groupnr=groupid, file_slide=fileslide)

            vertical_interp(group_id=groupid, file_slide=fileslide)

            directory = Path("/work/bb1097/b383521/dummy/decomposition/")
            for file in directory.iterdir():
                if file.is_file():
                    file.unlink()  # unlink() deletes the file
                    print(f"Deleted: {file.name}")
        
            compute_avrgae_save(groupnr=groupid, file_slide=fileslide)
        
        for var in variables:
            directory = Path("/work/bb1097/b383521/dummy/regroup/" + var + "/")
            for file in directory.iterdir():
                if file.is_file():
                    file.unlink()  # unlink() deletes the file
                    print(f"Deleted: {file.name}")


import sys

if __name__ == "__main__":
    file_slide = int(sys.argv[1])
    main3(file_slide)   # pass it in