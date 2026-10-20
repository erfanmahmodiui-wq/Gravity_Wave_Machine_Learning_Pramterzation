import numpy as np
import xarray as xr
import joblib
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(".."))
from model.model import read_scale_data
from evaluation.flux_energy import physical_values
from src.config import load_paths


def inference(X_train, X_val, X_test, ML_dir, fname_model):
    model = tf.keras.models.load_model(
        ML_dir + fname_model,
        custom_objects=False, compile=False)
    
    y_pred_test  = model.predict(X_test)
    print(f"max of predicted test {(np.max(y_pred_test))}")
    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    #y_pred_test  = model.predict(X_test)
    #print(f"max of predicted test {(np.max(y_pred_test))}")

    return y_pred_train, y_pred_val, y_pred_test


def compute_mse_R2_logspace(y_train_true, y_train_pred, y_val_true, y_val_pred, y_test_true, y_test_pred):
    mse_train = np.mean((y_train_true - y_train_pred)**2, axis=0)
    mse_val   = np.mean((y_val_true - y_val_pred)**2, axis=0)
    mse_test  = np.mean((y_test_true - y_test_pred)**2, axis=0)

    train_var = np.var(y_train_true, axis=0)
    val_var   = np.var(y_val_true, axis=0)
    test_var  = np.var(y_test_true, axis=0)

    R2_train = 1 - mse_train / train_var
    R2_val   = 1 - mse_val / val_var
    R2_test  = 1 - mse_test / test_var

    print(f"R2_train={R2_train.mean():.3f}  R2_val={R2_val.mean():.3f}  R2_test={R2_test.mean():.3f}")

    return mse_train, mse_val, mse_test, R2_train, R2_val, R2_test


def compute_mae_in_real_space(y_train_true, y_train_pred,
                               y_val_true,   y_val_pred,
                               y_test_true,  y_test_pred):

    y_train_real      = np.expm1(y_train_true)
    y_val_real        = np.expm1(y_val_true)
    y_test_real       = np.expm1(y_test_true)
    y_train_pred_real = np.expm1(y_train_pred)
    y_val_pred_real   = np.expm1(y_val_pred)
    y_test_pred_real  = np.expm1(y_test_pred)

    std_train = np.std(y_train_real, axis=0)
    base_line = np.mean(y_train_real, axis=0)

    mae_train = np.mean(np.abs(y_train_real - y_train_pred_real), axis=0)
    mae_val   = np.mean(np.abs(y_val_real - y_val_pred_real), axis=0)
    mae_test  = np.mean(np.abs(y_test_real - y_test_pred_real), axis=0)

    mae_train_norm_std = mae_train / std_train
    mae_val_norm_std   = mae_val / std_train
    mae_test_norm_std  = mae_test / std_train

    skill_train = 1 - mae_train / np.mean(np.abs(y_train_real - base_line), axis=0)
    skill_val   = 1 - mae_val / np.mean(np.abs(y_val_real - base_line), axis=0)
    skill_test  = 1 - mae_test / np.mean(np.abs(y_test_real - base_line), axis=0)

    print(f"MAE_train={mae_train.mean():.3f}  MAE_val={mae_val.mean():.3f}  MAE_test={mae_test.mean():.3f}")
    print(f"MAE_train_norm_std={mae_train_norm_std.mean():.3f}  MAE_val_norm_std={mae_val_norm_std.mean():.3f}  MAE_test_norm_std={mae_test_norm_std.mean():.3f}")
    print(f"Skill_train={skill_train.mean():.3f}  Skill_val={skill_val.mean():.3f}  Skill_test={skill_test.mean():.3f}")

    return mae_train, mae_val, mae_test, mae_train_norm_std, mae_val_norm_std, mae_test_norm_std, skill_train, skill_val, skill_test


def regime_analysis_integrated_energy(
    y_train_log,
    y_test_log,
    y_pred_log,
    percentiles=(0, 50, 90, 99, 100),
    eps=1e-12,
):
    """
    Regime analysis based on integrated true wave action density.
    y_*_log shape: (n_samples, n_outputs). Assumes data are in log1p space.
    """
    y_train = np.expm1(y_train_log)
    y_true  = np.expm1(y_test_log)
    y_pred  = np.expm1(y_pred_log)

    y_pred = np.maximum(y_pred, 0.0)

    clim = np.mean(y_train, axis=0)
    E_true = np.sum(y_true, axis=1)
    edges = np.percentile(E_true, percentiles)

    print(
        f"{'Regime':<12}{'n':>10}{'E range':>28}{'MAE':>14}"
        f"{'MAE_clim':>14}{'Skill':>10}{'Bias':>14}{'RelBias':>12}"
    )

    results = []

    for i in range(len(edges) - 1):
        p_lo = percentiles[i]
        p_hi = percentiles[i + 1]

        if i < len(edges) - 2:
            mask = (E_true >= edges[i]) & (E_true < edges[i + 1])
        else:
            mask = (E_true >= edges[i]) & (E_true <= edges[i + 1])

        n = mask.sum()
        if n == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        err_model = np.abs(yt - yp)
        err_clim  = np.abs(yt - clim[None, :])

        mae = np.mean(err_model)
        mae_clim = np.mean(err_clim)
        skill = 1.0 - mae / mae_clim if mae_clim > eps else np.nan

        bias = np.mean(yp - yt)
        mean_true = np.mean(yt)
        rel_bias = bias / mean_true if mean_true > eps else np.nan

        label = f"P{p_lo}-P{p_hi}"
        e_range = f"[{edges[i]:.3e}, {edges[i+1]:.3e}]"

        print(
            f"{label:<12}{n:>10}{e_range:>28}{mae:>14.3f}"
            f"{mae_clim:>14.3f}{skill:>10.3f}{bias:>14.3f}{rel_bias:>12.3f}"
        )

        results.append({
            "regime": label,
            "n": int(n),
            "E_min": edges[i],
            "E_max": edges[i + 1],
            "MAE": mae,
            "MAE_clim": mae_clim,
            "Skill": skill,
            "Bias": bias,
            "RelBias": rel_bias,
        })

    return results


def compute_skill_by_regime(y_train_real, y_test_real, y_test_pred_real, n_bins=4):
    y_train_real     = np.expm1(y_train_real)
    y_test_real      = np.expm1(y_test_real)
    y_test_pred_real = np.expm1(y_test_pred_real)

    base_line = np.mean(y_train_real, axis=0)
    mad_train = np.mean(np.abs(y_train_real - base_line), axis=0)

    rel_flat = (np.abs(y_test_real - base_line) / mad_train).flatten()

    y_flat    = y_test_real.flatten()
    pred_flat = y_test_pred_real.flatten()
    base_flat = np.broadcast_to(base_line, y_test_real.shape).flatten()

    err_model    = np.abs(y_flat - pred_flat)
    err_baseline = np.abs(y_flat - base_flat)

    bin_edges = np.quantile(rel_flat, np.linspace(0, 1, n_bins + 1))
    bin_idx   = np.digitize(rel_flat, bin_edges[1:-1])

    print(f"{'Regime':<10}{'n':>8}{'rel range':>18}{'MAE':>10}{'MAE_base':>10}{'Skill':>10}")
    for b in range(n_bins):
        mask = bin_idx == b
        n = mask.sum()
        if n == 0:
            continue
        lo, hi = rel_flat[mask].min(), rel_flat[mask].max()
        mae_b      = err_model[mask].mean()
        mae_base_b = err_baseline[mask].mean()
        skill_b    = 1 - mae_b / mae_base_b
        print(f"{b:<10}{n:>8}{f'[{lo:.2f}, {hi:.2f}]':>18}{mae_b:>10.3f}{mae_base_b:>10.3f}{skill_b:>10.3f}")


def separate_season(y, ntri):
    spatial_temp = y.shape[0]
    ntime = int(spatial_temp / len(ntri))
    ntime_jan = int(np.floor(ntime / 2))
    y_new = y.reshape(len(ntri), ntime, y.shape[-1])
    y_jan = y_new[:, :ntime_jan, :]
    y_aug = y_new[:, ntime_jan:, :]
    return y_jan.reshape(-1, y.shape[-1]), y_aug.reshape(-1, y.shape[-1])


if __name__ == "__main__":
    paths = load_paths()
    Data_base_Dir = paths["data_base_dir_jan"]
    save_dir      = paths["save_dir"]
    ML_dir        = paths["ml_model_dir"]

    results = {}
    regims = {}

    for tag in ["ocean", "land"]:
        fname_model = f"cnn_coarsed_{tag}_2.keras"

        (X_train, X_val, y_train, y_val, X_test, y_test, X_scaler, ntri, cgz_train, cgz_val, _, 
         omega_hat_train, omega_hat_val, _, k, l, m) = read_scale_data(
            model_save_dir=save_dir,
            coarse_action_dir_jan=paths["coarse_action_dir_jan"],
            land_ocean_tag=tag
        )

        y_pred_train, y_pred_val, y_pred_test = inference(
            X_train=X_train, X_val=X_val, X_test=X_test,
            ML_dir=ML_dir, fname_model=fname_model)

        y_train_jan, y_train_aug = separate_season(y_train, ntri)
        y_val_jan, y_val_aug = separate_season(y_val, ntri)
        y_test_jan, y_test_aug = separate_season(y_test, ntri)

        y_pred_train_jan, y_pred_train_aug = separate_season(y_pred_train, ntri)
        y_pred_val_jan, y_pred_val_aug = separate_season(y_pred_val, ntri)
        y_pred_test_jan, y_pred_test_aug = separate_season(y_pred_test, ntri)

        season_data = {
            "jan": {
                "y_train": y_train_jan, "y_val": y_val_jan, "y_test": y_test_jan,
                "y_pred_train": y_pred_train_jan, "y_pred_val": y_pred_val_jan, "y_pred_test": y_pred_test_jan,
            },
            "aug": {
                "y_train": y_train_aug, "y_val": y_val_aug, "y_test": y_test_aug,
                "y_pred_train": y_pred_train_aug, "y_pred_val": y_pred_val_aug, "y_pred_test": y_pred_test_aug,
            },
            "combined": {
                "y_train": y_train, "y_val": y_val, "y_test": y_test,
                "y_pred_train": y_pred_train, "y_pred_val": y_pred_val, "y_pred_test": y_pred_test,
            },
        }

        results.setdefault(tag, {})
        regims.setdefault(tag, {})

        for season, data in season_data.items():
            (mse_train, mse_val, mse_test, r2_train, r2_val, r2_test) = compute_mse_R2_logspace(
                y_train_true=data["y_train"], y_train_pred=data["y_pred_train"],
                y_val_true=data["y_val"], y_val_pred=data["y_pred_val"],
                y_test_true=data["y_test"], y_test_pred=data["y_pred_test"],
            )

            (mae_train, mae_val, mae_test, mae_train_norm_std, mae_val_norm_std, mae_test_norm_std,
             skill_train, skill_val, skill_test) = compute_mae_in_real_space(
                y_train_true=data["y_train"], y_train_pred=data["y_pred_train"],
                y_val_true=data["y_val"], y_val_pred=data["y_pred_val"],
                y_test_true=data["y_test"], y_test_pred=data["y_pred_test"],
            )

            regime_results = regime_analysis_integrated_energy(
                y_train_log=data["y_train"], y_test_log=data["y_test"], y_pred_log=data["y_pred_test"],
                percentiles=(0, 50, 90, 99, 100),
            )

            results[tag][season] = {
                "mse_train": mse_train, "mse_val": mse_val, "mse_test": mse_test,
                "R2_train": r2_train, "R2_val": r2_val, "R2_test": r2_test,
                "mae_train": mae_train, "mae_val": mae_val, "mae_test": mae_test,
                "mae_train_norm_std": mae_train_norm_std, "mae_val_norm_std": mae_val_norm_std,
                "mae_test_norm_std": mae_test_norm_std,
                "skill_train": skill_train, "skill_val": skill_val, "skill_test": skill_test,
                "n_train": data["y_train"].shape[0], "n_val": data["y_val"].shape[0], "n_test": data["y_test"].shape[0],
                "y_test": data["y_test"], "y_pred_test": data["y_pred_test"],
            }

            regims[tag][season] = regime_results

        del (X_train, X_val, y_train, y_val, X_test, y_test, X_scaler, ntri, cgz_train, cgz_val,
         omega_hat_train, omega_hat_val, _, k, l, m, y_pred_train, y_pred_val,
           y_pred_test,  y_train_jan, y_train_aug, y_val_jan, y_val_aug, y_pred_train_jan,
           y_pred_val_jan, y_pred_val_aug)

    (Energy_test, Energy_model, uw_test, uw_model, vw_test, vw_model,
    uw_model_zonal_mean, uw_test_zonal_mean, vw_model_zonal_mean,
    vw_test_zonal_mean, lat_bins) = physical_values()

    results["Energy_test"] = Energy_test
    results["Energy_model"] = Energy_model
    results["uw_test"] = uw_test
    results["uw_model"] = uw_model
    results["vw_test"] = vw_test
    results["vw_model"] = vw_model
    results["uw_model_zonal_mean"] = uw_model_zonal_mean
    results["uw_test_zonal_mean"] = uw_test_zonal_mean
    results["vw_model_zonal_mean"] = vw_model_zonal_mean
    results["vw_test_zonal_mean"] = vw_test_zonal_mean
    results["lat_bins"] = lat_bins

    joblib.dump(results, save_dir + "metrics/results_cnn_coarsed_grained_2.pkl")
    joblib.dump(regims,  save_dir + "metrics/regimes_cnn_coarsed_grained_2.pkl")