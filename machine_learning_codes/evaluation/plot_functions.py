import matplotlib.pyplot as plt
import numpy as np

def plot_zonal_mean_pair(
    lat,
    test_jan,
    model_jan,
    test_aug,
    model_aug,
    ylabel=r"$\rho\overline{u'w'}$ [mPa]",
    savename="zonal_mean_uw",
    fontsize=11,  # Standard paper font size (9–11 pt)
):
    # Standard 2-column paper figure dimensions
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.2), sharey=True, constrained_layout=True
    )

    datasets = [
        ("January", test_jan, model_jan, axes[0]),
        ("August", test_aug, model_aug, axes[1]),
    ]

    handles = []

    for name, true_vals, pred_vals, ax in datasets:
        ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.7)

        (l1,) = ax.plot(lat, true_vals, "-", color="black", lw=1.8, label="True")
        (l2,) = ax.plot(
            lat, pred_vals, "--", color="firebrick", lw=1.8, label="Predicted"
        )

        ax.set_title(name, fontsize=fontsize, pad=6)
        ax.set_xlim(lat.min(), lat.max())

        # Clean degree-formatted latitude ticks
        xticks = np.arange(-90, 91, 30)
        xticklabels = [
            f"{abs(x)}°S" if x < 0 else (f"{x}°N" if x > 0 else "0°")
            for x in xticks
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=fontsize - 1)

        ax.tick_params(axis="both", labelsize=fontsize - 1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=fontsize)
            handles = [l1, l2]

    # Centered figure-level x-axis label
    fig.supxlabel("Latitude", fontsize=fontsize)

    # Clean upper legend
    fig.legend(
        handles=handles,
        labels=["True", "Predicted"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=fontsize - 1,
        frameon=False,
    )

    plt.savefig(
        f"/work/bb1097/b383521/plots/paper_plots/{savename}.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.savefig(
        f"/work/bb1097/b383521/plots/paper_plots/{savename}.pdf",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()

def plot_zonal_mean_pair_extra(
    lat,
    uw_test,
    uw_pred,
    vw_test,
    vw_pred,
    savename="zonal_mean_uw_vw",
    fontsize=11,  # Standard paper font size (9–11 pt)
):
    # Standard 2-column paper figure dimensions
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.2), sharey=False, constrained_layout=True
    )

    datasets = [
        (r"$\rho\overline{u'w'}$ [mPa]", uw_test, uw_pred, axes[0]),
        (r"$\rho\overline{v'w'}$ [mPa]", vw_test, vw_pred, axes[1]),
    ]

    handles = []

    for ylabel, true_vals, pred_vals, ax in datasets:
        ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.7)

        (l1,) = ax.plot(lat, true_vals, "-", color="black", lw=1.8, label="True")
        (l2,) = ax.plot(
            lat, pred_vals, "--", color="firebrick", lw=1.8, label="Predicted"
        )

        ax.set_xlim(lat.min(), lat.max())

        # Clean degree-formatted latitude ticks
        xticks = np.arange(-90, 91, 30)
        xticklabels = [
            f"{abs(x)}°S" if x < 0 else (f"{x}°N" if x > 0 else "0°")
            for x in xticks
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=fontsize - 1)

        ax.tick_params(axis="both", labelsize=fontsize - 1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_ylabel(ylabel, fontsize=fontsize)

        if ax == axes[0]:
            handles = [l1, l2]

    # Centered figure-level x-axis label
    fig.supxlabel("Latitude", fontsize=fontsize)

    # Clean upper legend
    fig.legend(
        handles=handles,
        labels=["True", "Predicted"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=fontsize - 1,
        frameon=False,
    )

    plt.savefig(
        f"/work/bb1097/b383521/plots/paper_plots/{savename}.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.savefig(
        f"/work/bb1097/b383521/plots/paper_plots/{savename}.pdf",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()

import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm


def plot_global_map_flux_pair(
    var_top,
    var_bottom,
    save_dir,
    filename="global_momentum_flux_map_pair",
    title_top="True",
    title_bottom="Predicted",
    cbar_label=r"$\rho\overline{u'w'}$ [mPa]",
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    percentile_clip=True,
    p_low=2,
    p_high=98,
    symmetric=True,
    fontsize=11,
):
    var_top = np.asarray(var_top, dtype=float)
    var_bottom = np.asarray(var_bottom, dtype=float)

    gridfil = xr.open_dataset(
        "/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    )
    clon_vertices = np.rad2deg(gridfil.clon_vertices.values)
    clat_vertices = np.rad2deg(gridfil.clat_vertices.values)
    gridfil.close()

    clon_vertices = np.where(
        clon_vertices < -180.0, clon_vertices + 360.0, clon_vertices
    )
    clon_vertices = np.where(
        clon_vertices > 180.0, clon_vertices - 360.0, clon_vertices
    )
    triangles = np.stack([clon_vertices, clat_vertices], axis=-1)

    # ---------------------------
    # Shared color range across BOTH panels
    # ---------------------------
    if vmin is None or vmax is None:
        combined = np.concatenate(
            [var_top[np.isfinite(var_top)], var_bottom[np.isfinite(var_bottom)]]
        )
        if percentile_clip:
            p_lo_val, p_hi_val = np.nanpercentile(combined, [p_low, p_high])
            if symmetric:
                vmax_abs = max(abs(p_lo_val), abs(p_hi_val))
                vmin, vmax = -vmax_abs, vmax_abs
            else:
                vmin, vmax = p_lo_val, p_hi_val
        else:
            vmin, vmax = np.nanmin(combined), np.nanmax(combined)

    print(f"Using shared color range: vmin={vmin:.4f}, vmax={vmax:.4f}")

    cmap_obj = plt.get_cmap(cmap)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # 2 rows, 1 column vertical stack layout (Standard paper width 7.0 in)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.8),
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        constrained_layout=True,
    )

    datasets = [
        (var_top, title_top, axes[0]),
        (var_bottom, title_bottom, axes[1]),
    ]

    for var, subtitle, ax in datasets:
        colors = cmap_obj(norm(var))
        colors[~np.isfinite(var)] = [1, 1, 1, 0]

        ax.set_global()
        ax.coastlines(linewidth=0.6, zorder=2)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            color="dimgray",
            alpha=0.45,
            zorder=2,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": fontsize - 2}
        gl.ylabel_style = {"size": fontsize - 2}

        coll = PolyCollection(
            triangles,
            fc=colors,
            edgecolors="none",
            linewidth=0.05,
            transform=ccrs.Geodetic(),
            zorder=0,
        )
        ax.add_collection(coll)

        # Unbolded subpanel title
        ax.set_title(subtitle, fontsize=fontsize, pad=4)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])

    # Vertical colorbar on the right with compact width and padding
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="vertical",
        fraction=0.03,
        pad=0.03,
        shrink=0.85,
        extend="both",
    )
    cbar.set_label(cbar_label, fontsize=fontsize, fontweight="normal")
    cbar.ax.tick_params(labelsize=fontsize - 1)

    os.makedirs(save_dir, exist_ok=True)
    outpath_png = os.path.join(save_dir, f"{filename}.png")
    outpath_pdf = os.path.join(save_dir, f"{filename}.pdf")

    plt.savefig(outpath_png, dpi=400, bbox_inches="tight")
    plt.savefig(outpath_pdf, dpi=400, bbox_inches="tight")
    plt.show()
    print(f"--> saved to {outpath_png} and {outpath_pdf}")
