import numpy as np
import xarray as xr
import joblib
from netCDF4 import Dataset
from src.config import load_paths


def save_data_partion_time(X_input, y_encoded, Data_Base_Dir, ntri, group_id):
    total_time = X_input.shape[1]  # 145

    train_end = int(total_time * 0.75)
    val_end   = int(total_time * 0.85)

    X_train = X_input[:, :train_end, :]
    X_val   = X_input[:, train_end:val_end, :]
    X_test  = X_input[:, val_end:, :]
    y_train = y_encoded[:, :train_end, :]
    y_val   = y_encoded[:, train_end:val_end, :]
    y_test  = y_encoded[:, val_end:, :]

    Data_base = {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "ntri":    ntri
    }

    fname = Data_Base_Dir + f"corsed_action_{group_id}.pkl"
    joblib.dump(Data_base, fname)


def orographic_features(resolved_flow_dir):
    ds_oro = Dataset(resolved_flow_dir + "icon_etopo_global_merged.nc")

    oro_f = np.full((20480, 146, 10), np.nan)

    for i in range(20480):
        group_1 = ds_oro.groups[str(i)]
        oro = group_1['is_land'][:].data

        if oro == 1:
            H_spec  = group_1['H_spec'][:].data
            kks     = group_1['kks'][:].data
            lls     = group_1['lls'][:].data

            abs_lls = np.abs(lls)
            k_valid = kks > 0
            l_valid = abs_lls > 0

            H_sum = np.sum(H_spec)
            variance = H_sum

            mean_k = np.sum(H_spec * kks) / H_sum
            mean_l = np.sum(H_spec * lls) / H_sum
            mean_abs_l = np.sum(H_spec * abs_lls) / H_sum

            ridge_oro_k = np.sum(H_spec[k_valid] * (1.0 / kks[k_valid]))     / H_sum
            ridge_oro_l = np.sum(H_spec[l_valid] * (1.0 / abs_lls[l_valid])) / H_sum

            spectral_width_k = np.sum(H_spec * (kks     - mean_k)**2) / H_sum
            spectral_width_l = np.sum(H_spec * (abs_lls - mean_abs_l)**2) / H_sum

            denom_aniso = np.sum(H_spec * abs_lls)
            anisotropy  = np.sum(H_spec * kks) / denom_aniso if denom_aniso != 0 else 0.0

            orientation = np.arctan2(mean_l, mean_k)

            oro_f[i, :, 0] = variance
            oro_f[i, :, 1] = mean_k
            oro_f[i, :, 2] = mean_l
            oro_f[i, :, 3] = ridge_oro_k
            oro_f[i, :, 4] = ridge_oro_l
            oro_f[i, :, 5] = spectral_width_k
            oro_f[i, :, 6] = spectral_width_l
            oro_f[i, :, 7] = anisotropy
            oro_f[i, :, 8] = orientation
            oro_f[i, :, 9] = 1.0
    return oro_f


def form_X(resolved_flow_dirctory, ntri, orography, step=6, tag_year="_2019"):
    if tag_year == "_2019":
        print("it is extra test")
        nt = 1
        time_indices = [0]
        n_windows = 1
    elif tag_year == "_2016":
        print("it is not out sample")
        nt = 875
        time_indices = np.arange(0, nt, step)
        n_windows = len(time_indices)
    else:
        print("it is not out sample")
        nt = 865
        time_indices = np.arange(0, nt, step)
        n_windows = len(time_indices)

    data_base = {}
    for var in ["u", "v", "w", "p", "T", "N", "b", "Tw"]:
        with xr.open_dataset(resolved_flow_dirctory + f"average_{var}-sphere{tag_year}.nc", engine="h5netcdf") as ds:
            values = ds["var"].values
            values_reshaped = np.transpose(values, (2, 0, 1))
            data_base[var] = values_reshaped[ntri, :nt, :][:, time_indices, :]

    for var in ["u", "v", "T", "p"]:
        with xr.open_dataset(resolved_flow_dirctory + f"Hgradian_{var}{tag_year}.nc", engine="h5netcdf") as ds:
            values = ds.gradian.data
            data_base["dx" + var] = values[ntri, :nt, :, 0][:, time_indices, :]
            data_base["dy" + var] = values[ntri, :nt, :, 1][:, time_indices, :]

    with xr.open_dataset(resolved_flow_dirctory + f"vorticity{tag_year}.nc", engine="h5netcdf") as ds:
        values = ds.gradian.data
        data_base["vorticity"] = values[ntri, :nt, :][:, time_indices, :]

    keys = ["u", "v", "w", "p", "T", "N", "b", "Tw",
            "dxu", "dyu", "dxv", "dyv", "dxT", "dyT", "dxp", "dyp",
            "vorticity"]
    dynamic_f = len(keys) * 6
    n_features = dynamic_f + 9 + 1
    inputs = np.zeros((len(ntri), n_windows, n_features))
    i = 0
    for var in keys:
        inputs[:, :, i:i+6] = data_base[var]
        i += 6

    inputs[:, :, dynamic_f:n_features] = orography[ntri, :n_windows, :]

    return inputs



def ntri_orgraghy_nonorgrghy_index(resolved_flow_dirctory, action_dir, group_id):
    with xr.open_dataset(action_dir + f"coarse_gw_spectrum_group_{group_id}_fileslide_0.nc", engine="h5netcdf") as ds:
        ntri = ds.triangle.data

    with xr.open_dataset(resolved_flow_dirctory + "orography.nc", engine="h5netcdf") as ds:
        values = ds.Hspec.data[ntri, :]
        oro_sorted = -np.sort(-values, axis=1)
        ntri_orography = ntri[np.where(oro_sorted[:, 0] > 0)[0]]
        ntri_nonorography = ntri[np.where(oro_sorted[:, 0] == 0)[0]]
    return ntri_orography, ntri_nonorography, ntri


def form_Y_coarseN(coarse_action_dir, groupnr, tag_year="_2019"):
    if tag_year == "_2019":
        action = np.zeros((1, 128, 6, 6, 4))
        slide = 1
    elif tag_year == "_2016":
        action = np.zeros((146, 128, 6, 6, 4))
        slide = 73
    else:
        action = np.zeros((145, 128, 6, 6, 4))
        slide = 73

    for i in range(slide):
        fname = f"coarse_gw_spectrum_group_{groupnr}_fileslide_{i}_2.nc"
        with xr.open_dataset(coarse_action_dir + fname, engine="h5netcdf") as ds:
            N = ds.N_action_coarse.data
            time = ds.time.data
            start_idx = 2 * i
            end_idx   = start_idx + len(time)
            action[start_idx:end_idx, ...] = N

    action = np.transpose(action, (1, 0, 2, 3, 4))
    return action

def compute_cgz_intrinsic_paramters(X_input, coarse_action_dir_jan, path_grid):
    N2 = X_input[:, :, 30]
    
    with xr.open_dataset(coarse_action_dir_jan + "coarse_gw_spectrum_group_1_fileslide_0_2.nc") as ds:
        k_lunch = ds.k.data
        l_lunch = ds.l.data
        m_lunch = -ds.m.data

    Om = 7.2921e-5  # 2pi/24h in rad/s
    clat = xr.open_dataset(path_grid).clat.data[ntri]
    f = 2 * Om * np.sin(clat)

    k3D = k_lunch[None, None, :, None, None]
    l3D = l_lunch[None, None, None, :, None]
    m3D = m_lunch[None, None, None, None, :]
    
    N2_3D = N2[:, :, None, None, None]
    f_3D = f[:, None, None, None, None]
    f2_3D = f_3D ** 2
    
    kh2 = k3D ** 2 + l3D ** 2
    kappa2 = kh2 + m3D ** 2
    
    omega_hat = np.sqrt((N2_3D * kh2 + f2_3D * m3D ** 2) / kappa2)
    c_gz = -(N2_3D - f2_3D) * kh2 * m3D / (omega_hat * kappa2 ** 2)
    return c_gz, omega_hat, k_lunch, l_lunch, m_lunch


def split_time_and_landocean(X_input, y_encoded, ntri, cgz, omega_hat, k_lunch, l_lunch, m_lunch):
    total_time = X_input.shape[1]
    train_end = int(total_time * 0.75)
    val_end   = int(total_time * 0.85)

    X_train = X_input[:, :train_end, :]
    X_val   = X_input[:, train_end:val_end, :]
    X_test  = X_input[:, val_end:, :]
    cgz_train = cgz[:, :train_end, :]
    cgz_val   = cgz[:, train_end:val_end, :]
    cgz_test  = cgz[:, val_end:, :]
    omega_hat_train = omega_hat[:, :train_end, :]
    omega_hat_val   = omega_hat[:, train_end:val_end, :]
    omega_hat_test  = omega_hat[:, val_end:, :]
    y_train = y_encoded[:, :train_end, :]
    y_val   = y_encoded[:, train_end:val_end, :]
    y_test  = y_encoded[:, val_end:, :]
    

    dynamic_field = 17 * 6

    land_mask  = X_train[:, 0, -1] == 1
    ocean_mask = np.isnan(X_train[:, 0, -1])

    land = {
        "X_train": X_train[land_mask, :, :-1],
        "X_val":   X_val[land_mask, :, :-1],
        "X_test":  X_test[land_mask, :, :-1],
        "cgz_train": cgz_train[land_mask, :, :],
        "cgz_val":   cgz_val[land_mask, :, :],
        "cgz_test":  cgz_test[land_mask, :, :],
        "omega_hat_train": omega_hat_train[land_mask, :, :],
        "omega_hat_val":   omega_hat_val[land_mask, :, :],
        "omega_hat_test":  omega_hat_test[land_mask, :, :],
        "y_train": y_train[land_mask, ...],
        "y_val":   y_val[land_mask, ...],
        "y_test":  y_test[land_mask, ...],
        "ntri":    ntri[land_mask],
        "k_lunch": k_lunch,
        "l_lunch": l_lunch,
        "m_lunch": m_lunch
    }

    ocean = {
        "X_train": X_train[ocean_mask, :, :dynamic_field],
        "X_val":   X_val[ocean_mask, :, :dynamic_field],
        "X_test":  X_test[ocean_mask, :, :dynamic_field],
        "cgz_train": cgz_train[ocean_mask, :, :],
        "cgz_val":   cgz_val[ocean_mask, :, :],
        "cgz_test":  cgz_test[ocean_mask, :, :],
        "omega_hat_train": omega_hat_train[ocean_mask, :, :],
        "omega_hat_val":   omega_hat_val[ocean_mask, :, :],
        "omega_hat_test":  omega_hat_test[ocean_mask, :, :],
        "y_train": y_train[ocean_mask, ...],
        "y_val":   y_val[ocean_mask, ...],
        "y_test":  y_test[ocean_mask, ...],
        "ntri":    ntri[ocean_mask],
        "k_lunch": k_lunch,
        "l_lunch":l_lunch,
        "m_lunch":m_lunch

    }

    return land, ocean

def form_ds_outsample(X_input, y_encoded, ntri, cgz, omega_hat, k_lunch, l_lunch, m_lunch):
    dynamic_field = 17 * 6
    
    land_mask  = X_input[:, 0, -1] == 1
    ocean_mask = np.isnan(X_input[:, 0, -1])

    land = {
        "X" : X_input[land_mask, :, :-1],
        "Y" : y_encoded[land_mask, ...],
        "cgz" : cgz[land_mask, ...],
        "omega_hat" : omega_hat[land_mask, ...],
        "ntri" : ntri[land_mask],
        "k" : k_lunch,
        "l" : l_lunch,
        "m" : m_lunch
    }

    ocean = {
        "X" : X_input[ocean_mask, :, :dynamic_field],
        "Y" : y_encoded[ocean_mask, ...],
        "cgz" : cgz[ocean_mask, ...],
        "omega_hat" : omega_hat[ocean_mask, ...],
        "ntri" : ntri[ocean_mask],
        "k" : k_lunch,
        "l" : l_lunch,
        "m" : m_lunch
    }
    return land, ocean


if __name__ == "__main__":
    paths = load_paths()
    tag_season = "Nov"
    resolved_flow_dirctory = paths["resolved_flow_dir"]
    coarse_action_dir      = paths[f"coarse_action_dir_{tag_season}"]
    Data_Base_Dir           = paths[f"data_base_dir_{tag_season}"]
    grid_path               = paths[f"grid_file_low"]

    orography = orographic_features(resolved_flow_dir=resolved_flow_dirctory)

    if tag_season == "Nov":
        keys = ["X", "Y", "cgz", "omega_hat", "ntri", "k", "l", "m"]
    elif tag_season == "jan" or tag_season == "aug":
        keys = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "ntri",
             "cgz_train", "cgz_val", "cgz_test","omega_hat_train", "omega_hat_val",
               "omega_hat_test", "k_lunch", "l_lunch", "m_lunch"]

    land_acc  = {k: [] for k in keys}
    ocean_acc = {k: [] for k in keys}

    for group_id in range(160):
        ntri_orography, ntri_nonorography, ntri = ntri_orgraghy_nonorgrghy_index(
            resolved_flow_dirctory, coarse_action_dir, group_id=group_id
        )
        input_orgraphy   = form_X(resolved_flow_dirctory, ntri=ntri,
                                   orography=orography, tag_year="_2019")
        print(input_orgraphy[0, 0, 0])
        spectrum_coarsed = form_Y_coarseN(coarse_action_dir=coarse_action_dir,
                                           groupnr=group_id, tag_year="_2019")
        cgz, omega_hat, k_lunch, l_lunch, m_lunch = compute_cgz_intrinsic_paramters(
            input_orgraphy, coarse_action_dir, grid_path)

        if tag_season == "jan" or tag_season=="aug":
            land, ocean = split_time_and_landocean(input_orgraphy, spectrum_coarsed, 
                                                   ntri, cgz, omega_hat, k_lunch, l_lunch,
                                                     m_lunch)
        elif tag_season == "Nov":
            land, ocean = form_ds_outsample(X_input=input_orgraphy, y_encoded=spectrum_coarsed,
                                            ntri=ntri, cgz=cgz, k_lunch=k_lunch,
                                            omega_hat=omega_hat, m_lunch=m_lunch,
                                            l_lunch=l_lunch)

        for k in keys:
            land_acc[k].append(land[k])
            ocean_acc[k].append(ocean[k])

        print(f"group {group_id}: land n={land['ntri'].shape[0]}, ocean n={ocean['ntri'].shape[0]}")

    Data_based_land = {k: np.concatenate(v, axis=0) for k, v in land_acc.items()}
    Data_based_ocean = {k: np.concatenate(v, axis=0) for k, v in ocean_acc.items()}
    joblib.dump(Data_based_land, Data_Base_Dir + "corsed_action_land_2.pkl")
    joblib.dump(Data_based_ocean, Data_Base_Dir + "corsed_action_ocean_2.pkl")