import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(".."))
from model.model import read_scale_data
from src.config import load_paths


def compute_shap_values_best_worse(tag, save_dir, ml_model_dir, shap_dir):

    X_train, X_val, y_train, y_val, X_test, _, _, _ = read_scale_data(
        model_save_dir=save_dir,
        land_ocean_tag=tag
    )

    fname_model = f"cnn_coarsed_{tag}.keras"

    model = tf.keras.models.load_model(
        ml_model_dir + fname_model,
        custom_objects=False, compile=False)

    core_model = tf.keras.Model(inputs=model.input, outputs=model.output)

    n_background = 100

    base_vars = ["u", "v", "w", "p", "T", "N", "b", "Tw",
                 "dxu", "dyu", "dxv", "dyv", "dxT", "dyT", "vorticity"]
    n_levels = 6

    feat_labels = []
    for var in base_vars:
        for lev in range(1, n_levels + 1):
            feat_labels.append(f"{var}_L{lev}")
    for i in range(1, 11):
        feat_labels.append(f"oro_{i}")
    feat_labels.append("mask")
    feat_labels = np.array(feat_labels)

    metrics_dir = save_dir + "metrics/"
    fname = f"results_{tag}.pkl"
    metrics = joblib.load(metrics_dir + fname)
    ntri = metrics["ntri"]
    spatial_mae = metrics["log_space_spatial"]["Test"]
    n_best = 100
    n_worst = 100

    flat_mae = spatial_mae.flatten()

    best_flat = np.argsort(flat_mae)[:n_best]
    worst_flat = np.argsort(flat_mae)[-n_worst:][::-1]

    best_tri, best_time = np.unravel_index(best_flat, spatial_mae.shape)
    worst_tri, worst_time = np.unravel_index(worst_flat, spatial_mae.shape)

    print("BEST  samples  (tri, time, mae):")
    for tri, t, idx in zip(best_tri, best_time, best_flat):
        print(f"  tri={tri:5d}  time={t:5d}  mae={flat_mae[idx]:.6f}")

    print("\nWORST samples  (tri, time, mae):")
    for tri, t, idx in zip(worst_tri, worst_time, worst_flat):
        print(f"  tri={tri:5d}  time={t:5d}  mae={flat_mae[idx]:.6f}")

    best_indices_per_model = best_flat[np.newaxis, :]
    worst_indices_per_model = worst_flat[np.newaxis, :]

    rng = np.random.default_rng(100)
    bg_idx = rng.choice(X_train.shape[0], size=n_background, replace=False)
    background = X_train[bg_idx, :]

    X_best = X_test[best_indices_per_model[0, :], :]
    X_worst = X_test[worst_indices_per_model[0, :], :]

    dic = {}
    dic["tag"] = tag
    dic["best_time"] = best_time
    dic["worst_time"] = worst_time
    dic["best_tri"] = best_tri
    dic["worst_tri"] = worst_tri
    dic["mae_best"] = best_flat
    dic["mea_worst"] = worst_flat
    joblib.dump(dic, shap_dir + f"shape_info_{tag}.pkl")

    explainer = shap.KernelExplainer(core_model.predict, background)

    print("Computing SHAP for best samples \u2026")
    shap_best = explainer(X_best)
    print("Computing SHAP for worst samples \u2026")
    shap_worst = explainer(X_worst)

    sv_best = shap_best.values
    sv_worst = shap_worst.values

    np.save(shap_dir + f"svbest_{tag}.npy", sv_best)
    np.save(shap_dir + f"svworst_{tag}.npy", sv_worst)
    return sv_best, sv_worst, base_vars


def wrap_model_for_energy(model):
    """Appends expm1 + sum-over-outputs as a Lambda layer, producing scalar energy output."""
    energy_output = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(tf.math.expm1(x), axis=1, keepdims=True)
    )(model.output)
    return tf.keras.Model(inputs=model.input, outputs=energy_output)


def choose_samples_by_regime(tag, save_dir, n_samples=100):
    X_train, _, _, _, X_test, y_test, _, ntri = read_scale_data(
        model_save_dir=save_dir,
        land_ocean_tag=tag)
    N_test = np.expm1(y_test)
    energy_test = np.sum(N_test, axis=1)
    P90 = np.percentile(energy_test, 90)
    P99 = np.percentile(energy_test, 99)
    low_energy_indices = np.where(energy_test < P90)[0]
    high_energy_indices = np.where(energy_test > P99)[0]
    rng = np.random.default_rng(100)
    low_energy_samples = rng.choice(low_energy_indices, size=n_samples, replace=False)
    high_energy_samples = rng.choice(high_energy_indices, size=n_samples, replace=False)
    return X_test[low_energy_samples], X_test[high_energy_samples], X_train


def compute_shap_values_by_regime(tag, save_dir, ml_model_dir, shap_dir, n_samples=100):
    X_test_low, X_test_high, X_train = choose_samples_by_regime(tag, save_dir, n_samples=n_samples)
    fname_model = f"cnn_coarsed_{tag}.keras"
    model = tf.keras.models.load_model(
        ml_model_dir + fname_model,
        custom_objects=False, compile=False)

    energy_wraper = wrap_model_for_energy(model)

    n_background = 200
    rng = np.random.default_rng(100)
    bg_idx = rng.choice(X_train.shape[0], size=n_background, replace=False)
    background = X_train[bg_idx, :]

    explainer_low = shap.KernelExplainer(energy_wraper, background)
    print("Computing SHAP for low energy samples \u2026")
    shap_low = explainer_low(X_test_low)
    sv_low = shap_low.values
    np.save(shap_dir + f"svlow_{tag}.npy", sv_low)

    explainer_high = shap.KernelExplainer(energy_wraper, background)
    print("Computing SHAP for high energy samples \u2026")
    shap_high = explainer_high(X_test_high)
    sv_high = shap_high.values
    np.save(shap_dir + f"svhigh_{tag}.npy", sv_high)


def compute_shap_values_global(tag, save_dir, ml_model_dir, shap_dir):

    (X_train, _, _, _, X_test, _, _, _, _, _, _, 
         _, _, _, _, _, _) = read_scale_data(
            model_save_dir=save_dir,
            coarse_action_dir_jan=paths["coarse_action_dir_jan"],
            land_ocean_tag=tag
        )

    fname_model = f"cnn_coarsed_{tag}_2.keras"

    model = tf.keras.models.load_model(
        ml_model_dir + fname_model,
        custom_objects=False, compile=False)

    core_model = tf.keras.Model(inputs=model.input, outputs=model.output)

    n_background = 100

    base_vars = ["u", "v", "w", "p", "T", "N", "b", "Tw",
                 "dxu", "dyu", "dxv", "dyv", "dxT", "dyT", "dxp", "dyp", "vorticity"]
    n_levels = 6

    feat_labels = []
    for var in base_vars:
        for lev in range(1, n_levels + 1):
            feat_labels.append(f"{var}_L{lev}")
    for i in range(1, 11):
        feat_labels.append(f"oro_{i}")
    feat_labels.append("mask")
    feat_labels = np.array(feat_labels)

    # NOTE: rng must be defined before use — the original file used it before
    # definition here, which would raise NameError.
    rng = np.random.default_rng(100)
    idx = rng.choice(X_test.shape[0], size=200, replace=False)   # 1D index array
    X_shap = X_test[idx]
    print(X_shap.shape)  

    bg_idx = rng.choice(X_train.shape[0], size=n_background, replace=False)
    background = X_train[bg_idx, :]


    explainer = shap.KernelExplainer(core_model.predict, background)

    print("Computing SHAP for best samples \u2026")
    shap_val = explainer(X_shap)

    sv_global = shap_val.values

    np.save(shap_dir + f"svglobal_{tag}_2.npy", sv_global)


def plot_shaply_values(sv_best, sv_worst, base_vars, tag, shap_plots_dir, n_levels=6):
    levels_labels = ["15.1km", "13km", "11km", "9km", "7km", "5km"]
    levels_display = levels_labels[::-1]

    print(f"shape of best shap : {sv_best.shape}")

    if sv_best.ndim == 3:
        sv_best = np.mean(np.abs(sv_best), axis=2)
        sv_worst = np.mean(np.abs(sv_worst), axis=2)
    else:
        sv_best = np.abs(sv_best)
        sv_worst = np.abs(sv_worst)

    mean_best = np.mean(sv_best, axis=0)
    mean_worst = np.mean(sv_worst, axis=0)
    print(f"shape of best mean : {mean_best.shape}")

    shap_matrix_best = np.zeros((len(base_vars), n_levels))
    shap_matrix_worst = np.zeros((len(base_vars), n_levels))

    for i in range(len(base_vars)):
        sl = slice(i * n_levels, i * n_levels + n_levels)
        shap_matrix_best[i, :] = mean_best[sl]
        shap_matrix_worst[i, :] = mean_worst[sl]

    if tag == "land":
        orographic_feature_names = [
            "Orographic Variance",
            "Mean Zonal Wavenumber",
            "Mean Meridional Wavenumber",
            "Zonal Ridge Scale",
            "Meridional Ridge Scale",
            "Zonal Spectral Width",
            "Meridional Spectral Width",
            "Anisotropy",
            "Ridge Orientation",
        ]
        group_names = np.array(base_vars + orographic_feature_names)
    else:
        group_names = np.array(base_vars)

    group_best = []
    group_worst = []
    for i in range(len(base_vars)):
        sl = slice(i * n_levels, i * n_levels + n_levels)
        group_best.append(mean_best[sl].mean())
        group_worst.append(mean_worst[sl].mean())

    if tag == "land":
        for i in range(len(orographic_feature_names)):
            group_best.append(mean_best[90 + i])
            group_worst.append(mean_worst[90 + i])

    group_best = np.array(group_best)
    group_worst = np.array(group_worst)
    sort_idx = np.argsort(group_best + group_worst)[::-1]

    print(f"shape of group best : {group_best.shape}")
    print(f"shape of sort idx {sort_idx}")

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(group_names))
    w = 0.35

    ax.bar(x - w / 2, group_best[sort_idx], width=w, label="Best 200", color="#2196F3", alpha=0.85)
    ax.bar(x + w / 2, group_worst[sort_idx], width=w, label="Worst 200", color="#F44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names[sort_idx], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean |SHAP| (avg over levels & outputs)", fontsize=11)
    ax.set_title(f"Variable Importance \u2013 Best vs Worst | tag {tag}", fontsize=13)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(shap_plots_dir + f"shap_{tag}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved \u2192 shap_{tag}.png")

    mat_best_flip = shap_matrix_best[:, ::-1]
    mat_worst_flip = shap_matrix_worst[:, ::-1]

    vmin = min(mat_best_flip.min(), mat_worst_flip.min())
    vmax = max(mat_best_flip.max(), mat_worst_flip.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for i, (ax, mat, title) in enumerate(zip(
            axes,
            [mat_best_flip, mat_worst_flip],
            ["Best 200", "Worst 200"])):

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_levels))
        ax.set_xticklabels(levels_display, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(base_vars)))
        ax.set_yticklabels(base_vars, fontsize=9)
        ax.set_xlabel("Vertical level (ascending)")
        ax.set_title(title, fontsize=12)

        if i == 1:
            fig.colorbar(im, ax=ax, label="Mean |SHAP|", shrink=0.8, pad=0.02)

    fig.suptitle(f"SHAP Heatmap: Variable \u00d7 Level | tag {tag}", fontsize=13)
    plt.tight_layout()
    plt.savefig(shap_plots_dir + f"shap_heatmap{tag}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved \u2192 shap_heatmap_{tag}.png")

    top5_idx = np.argsort(group_best[:-2] + group_worst[:-2])[::-1][:5]
    top5_vars = np.array(base_vars)[top5_idx]

    fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
    fig.suptitle(f"SHAP Vertical Profile (top 5 vars) | tag {tag}", fontsize=13)

    for ax, var in zip(axes, top5_vars):
        vi = base_vars.index(var)
        sl = slice(vi * n_levels, vi * n_levels + n_levels)
        b_lev = mean_best[sl][::-1]
        w_lev = mean_worst[sl][::-1]

        ax.plot(b_lev, range(n_levels), "o-", color="#2196F3", label="Best", linewidth=2)
        ax.plot(w_lev, range(n_levels), "s-", color="#F44336", label="Worst", linewidth=2)
        ax.set_title(var, fontsize=11, fontweight="bold")
        ax.set_xlabel("|SHAP|", fontsize=9)
        ax.set_yticks(range(n_levels))
        ax.set_yticklabels(levels_display, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Altitude")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(shap_plots_dir + f"shap_vertical_profile_{tag}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved \u2192 shap_vertical_profile_{tag}.png")

    delta_matrix = (shap_matrix_worst - shap_matrix_best)[:, ::-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    vabs = np.abs(delta_matrix).max()
    im = ax.imshow(delta_matrix, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels(levels_display, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(base_vars)))
    ax.set_yticklabels(base_vars, fontsize=9)
    ax.set_xlabel("Vertical level (ascending)")
    ax.set_title(f"SHAP \u0394 (Worst \u2212 Best) | tag {tag}\nRed = more important in worst samples", fontsize=12)
    fig.colorbar(im, ax=ax, label="\u0394 Mean |SHAP|", pad=0.02)
    plt.tight_layout()
    plt.savefig(shap_plots_dir + f"shap_delta_heatmap_tag{tag}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved \u2192 shap_delta_heatmap_group{tag}.png")


if __name__ == "__main__":
    paths = load_paths()
    save_dir     = paths["save_dir"]
    ml_model_dir = paths["ml_model_dir"]
    shap_dir     = paths["shap_dir"]

    for tag in ["ocean"]:
        compute_shap_values_global(tag, save_dir=save_dir, ml_model_dir=ml_model_dir, shap_dir=shap_dir)
