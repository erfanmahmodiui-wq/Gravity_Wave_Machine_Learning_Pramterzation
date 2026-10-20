import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from plot_functions import plot_3D_intersections
import os


def read_EOFs_at(time_coeffi_dir, EOF_dir, avg_dir):
    ds_at = xr.open_dataset(time_coeffi_dir)
    at = ds_at.time_coffi.data
    time = ds_at.time.data
    ds_EOF = xr.open_dataset(EOF_dir)
    EOFs = ds_EOF.eigenvectors.data
    EOF = EOFs[:, :at.shape[2]]
    ds_avg = xr.open_dataset(avg_dir)
    avarage = ds_avg.avrage.data
    return at, EOF, avarage, time

def list_sort_interp_action(interp_dir, group_id):
    lis = os.listdir(interp_dir)
    files = []
    for file in lis:
        if file.split("-")[-1][0] == str(group_id):
            files.append(file)
            
    file_num = [int(fi.split("-")[1]) for fi in files]
    sort_indx = np.argsort(file_num)
    list_files = np.array(files)
    sorted_list = list_files[sort_indx]
    return sorted_list

def Recunstruction_error(interp_dir, lis, at, avarage, EOF):
    RMSE = []
    Error = []
    for file_slid in range(len(lis)):
        print(f"read action data file : {lis[file_slid]}")
        ds_actuall = xr.open_dataset(interp_dir+lis[file_slid])
        N_actual = ds_actuall.data.data
        l = ds_actuall.l.data
        k = ds_actuall.k.data
        m = ds_actuall.m.data
        ntri = ds_actuall.ntri.data
        time = ds_actuall.time.data
        for ti in  range(len(time)):
            N_actual_1time = N_actual[ti, ...]
            N_EOF = at[file_slid*12+ti, :, :]@EOF.T + avarage
            N_EOF = np.reshape(N_EOF, (len(ntri), len(l), len(k), len(m)))
            rmse = np.sqrt(np.mean((N_actual - N_EOF)**2))
            RMSE.append(rmse)
            error = rmse/(np.max(N_actual_1time) - np.min(N_actual_1time))
            Error.append(error)
    return RMSE, Error, ntri, k, l, m

def plot_time_coff(at, plot_dir):
    max_tau = 867
    t_index = np.arange(0, 865, 1)
    np.random.seed(42)
    random_tri = np.random.randint(0, 2048, size=3)
    components = [0, 33, 49]

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(42, 12))

    ax[0].plot(t_index, at[:, random_tri, components[0]])
    ax[0].set_ylabel(f'$a_{{{components[0]}}}(t)$', fontsize=14)
    ax[0].set_xticks(np.arange(0, max_tau + 1, 50))
    ax[0].set_xlim(0, 867)

    ax[1].plot(t_index, at[:, random_tri, components[1]])
    ax[1].set_ylabel(f'$a_{{{components[1]}}}(t)$', fontsize=14)
    ax[1].set_xticks(np.arange(0, max_tau + 1, 50))
    ax[1].set_xlim(0, 867)

    for tri in random_tri:
        ax[2].plot(t_index, at[:, tri, components[2]], label=f'Triangle {tri}')
    ax[2].set_ylabel(f'$a_{{{components[2]}}}(t)$', fontsize=14)
    ax[2].set_xlabel(r'$\tau$', fontsize=14)
    ax[2].legend(fontsize=12)
    ax[2].set_xlim(0, 867)

    # Set x-axis ticks every 50 starting from 0
    ax[2].set_xticks(np.arange(0, max_tau + 1, 50))

    plt.tight_layout()
    plt.show()
    fig.savefig(plot_dir+"trajectory.png")



if __name__ == "__main__":
    time_coeffi_dir = "/work/bb1097/b383521/slided_window_avrage/time_coff_all/full-time-coff_4.nc"
    EOF_dir = "/work/bb1097/b383521/slided_window_avrage/EOF/EOFs_slided_window4.nc"
    avg_dir = "/work/bb1097/b383521/slided_window_avrage/avrage_of_all/avarage-action-slided_window-full-4.nc"
    interp_dir = "/work/bb1097/b383521/slided_window_avrage/inter_wave_action_density_filled_log/"
    plot_dir = "/work/bb1097/b383521/Results/"
    at, EOF, avarage, time = read_EOFs_at(time_coeffi_dir, EOF_dir, avg_dir)
    lis = list_sort_interp_action(interp_dir)
    #RMSE, Error, ntri, k, l, m = Recunstruction_error(interp_dir, lis, at, avarage, EOF)
    plot_time_coff(at, plot_dir)
