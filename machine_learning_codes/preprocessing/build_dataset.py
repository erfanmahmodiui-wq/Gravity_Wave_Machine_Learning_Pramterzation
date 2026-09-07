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


def form_X(resolved_flow_dirctory, ntri, orography, nt=876, step=6, tag_year="_2016"):
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

    inputs[:, :, dynamic_f:n_features] = orography[ntri, :, :]

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


def form_Y_coarseN(coarse_action_dir, groupnr):
    action = np.zeros((146, 128, 6, 6, 4))

    for i in range(73):
        fname = f"coarse_gw_spectrum_group_{groupnr}_fileslide_{i}.nc"
        with xr.open_dataset(coarse_action_dir + fname, engine="h5netcdf") as ds:
            N = ds.N_action_coarse.data
            time = ds.time.data
            start_idx = 2 * i
            end_idx   = start_idx + len(time)
            action[start_idx:end_idx, ...] = N

    action = np.transpose(action, (1, 0, 2, 3, 4))
    return action


def split_time_and_landocean(X_input, y_encoded, ntri):
    total_time = X_input.shape[1]
    train_end = int(total_time * 0.75)
    val_end   = int(total_time * 0.85)

    X_train = X_input[:, :train_end, :]
    X_val   = X_input[:, train_end:val_end, :]
    X_test  = X_input[:, val_end:, :]

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
        "y_train": y_train[land_mask, ...],
        "y_val":   y_val[land_mask, ...],
        "y_test":  y_test[land_mask, ...],
        "ntri":    ntri[land_mask],
    }

    ocean = {
        "X_train": X_train[ocean_mask, :, :dynamic_field],
        "X_val":   X_val[ocean_mask, :, :dynamic_field],
        "X_test":  X_test[ocean_mask, :, :dynamic_field],
        "y_train": y_train[ocean_mask, ...],
        "y_val":   y_val[ocean_mask, ...],
        "y_test":  y_test[ocean_mask, ...],
        "ntri":    ntri[ocean_mask],
    }

    return land, ocean


if __name__ == "__main__":
    paths = load_paths()
    resolved_flow_dirctory = paths["resolved_flow_dir"]
    coarse_action_dir      = paths["coarse_action_dir"]
    Data_Base_Dir           = paths["data_base_dir_aug"]

    orography = orographic_features(resolved_flow_dir=resolved_flow_dirctory)

    keys = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "ntri"]
    land_acc  = {k: [] for k in keys}
    ocean_acc = {k: [] for k in keys}

    for group_id in range(160):
        ntri_orography, ntri_nonorography, ntri = ntri_orgraghy_nonorgrghy_index(
            resolved_flow_dirctory, coarse_action_dir, group_id=group_id
        )
        input_orgraphy   = form_X(resolved_flow_dirctory, ntri=ntri, orography=orography)
        spectrum_coarsed = form_Y_coarseN(coarse_action_dir=coarse_action_dir, groupnr=group_id)

        land, ocean = split_time_and_landocean(input_orgraphy, spectrum_coarsed, ntri)

        for k in keys:
            land_acc[k].append(land[k])
            ocean_acc[k].append(ocean[k])

        print(f"group {group_id}: land n={land['ntri'].shape[0]}, ocean n={ocean['ntri'].shape[0]}")

    Data_based_land = {k: np.concatenate(v, axis=0) for k, v in land_acc.items()}
    Data_based_ocean = {k: np.concatenate(v, axis=0) for k, v in ocean_acc.items()}

    joblib.dump(Data_based_land, Data_Base_Dir + "corsed_action_land.pkl")
    joblib.dump(Data_based_ocean, Data_Base_Dir + "corsed_action_ocean.pkl")
