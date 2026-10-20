import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plot_quadrants_norm(ax, array, fixed_coord, cmap, norm, x, y, z):
    nx, ny, nz = array.shape
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]

    plane_data = array[index]
    n0, n1 = plane_data.shape

    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]

    if fixed_coord == 'x':
        y_quads = [y[:ny // 2 + 1], y[ny // 2:]]
        z_quads = [z[:nz // 2 + 1], z[nz // 2:]]
        mid_x = x[nx // 2]
        for i, quadrant in enumerate(quadrants):
            facecolors = cmap(norm(quadrant))
            y_vals = y_quads[i // 2]
            z_vals = z_quads[i % 2]
            Y, Z = np.meshgrid(y_vals, z_vals, indexing='ij')
            X = np.full_like(Y, mid_x)
            ax.plot_surface(X, Y, Z, facecolors=facecolors, shade=False)
    elif fixed_coord == 'y':
        x_quads = [x[:nx // 2 + 1], x[nx // 2:]]
        z_quads = [z[:nz // 2 + 1], z[nz // 2:]]
        mid_y = y[ny // 2]
        for i, quadrant in enumerate(quadrants):
            facecolors = cmap(norm(quadrant))
            x_vals = x_quads[i // 2]
            z_vals = z_quads[i % 2]
            X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
            Y = np.full_like(X, mid_y)
            ax.plot_surface(X, Y, Z, facecolors=facecolors, shade=False)
    elif fixed_coord == 'z':
        x_quads = [x[:nx // 2 + 1], x[nx // 2:]]
        y_quads = [y[:ny // 2 + 1], y[ny // 2:]]
        mid_z = z[nz // 2]
        for i, quadrant in enumerate(quadrants):
            facecolors = cmap(norm(quadrant))
            x_vals = x_quads[i // 2]
            y_vals = y_quads[i % 2]
            X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
            Z = np.full_like(X, mid_z)
            ax.plot_surface(X, Y, Z, facecolors=facecolors, shade=False)

def figure_3D_array_slices(array,x,y,z, eigvecnr, label, cmap=None):
    """Plot a 3d array using three intersecting centered planes."""
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(array.shape)
    norm = mcolors.Normalize(vmin=array.min(), vmax=array.max())
    cmap = plt.get_cmap(cmap)
    cmap = cmap.reversed()
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    plot_quadrants_norm(ax, array, 'x', cmap,norm,x,y,z)
    plot_quadrants_norm(ax, array, 'y', cmap,norm,x,y,z)
    plot_quadrants_norm(ax, array, 'z', cmap,norm,x,y,z)
    # Set axis limits explicitly to cover the full coordinate ranges
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_box_aspect((np.ptp(x)*15, np.ptp(y)*15, np.ptp(z)))
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, orientation='vertical')
    if label=="EOF":
        cbar.set_label(f"EOF{eigvecnr}", fontsize=14, labelpad=15)
    else:
        cbar.set_label(r"$\langle \log_{10} N \rangle$", fontsize=14, labelpad=15)
    #nr=3 #Doesn't really help
    #ticksx = np.arange(x.min(), x.max(),(x.max()- x.min())/nr)
    #ticksy = np.arange(y.min(), y.max(),(y.max()- y.min())/nr)
    #ticksz = np.arange(z.min(), z.max(),(z.max()- z.min())/nr)
    #ax.set_yticks(ticksy)
    #ax.set_xticks(ticksx)
    #ax.set_zticks(ticksz)
    return fig, ax

def plot_3D_intersections(data_3D, k, l, m, eigvecnr, label, save_dir, groupnr):

    fig,ax=figure_3D_array_slices(data_3D, k*1000,l*1000, m*1000, eigvecnr,cmap='viridis_r', label=label)

    #average=ds.average.data
    #k=ds.average.k.data
    #l=ds.average.l.data
    #m=ds.average.m.data
    #fig,ax=figure_3D_array_slices(average[...],k*1000,l*1000,m*1000,cmap='viridis_r')
    ax.set_xlabel(r"$k = 2\pi/\lambda$ [km$^{-1}$]", fontsize=15, labelpad=15)
    ax.set_ylabel(r"$l = 2\pi/\lambda$ [km$^{-1}$]", fontsize=15, labelpad=15)
    ax.set_zlabel(r"$m = 2\pi/\lambda$ [km$^{-1}$]", fontsize=15, labelpad=15)

    if label == "EOF":
        fname = f"3D_eigvec-group-{groupnr}-EOF-{eigvecnr}.png" if groupnr is not None else f"3D_eigvec-EOF-{eigvecnr}.png"
    if label=="avg":
        fname = f"3D_avg-group-{groupnr}.svg" if groupnr is not None else "3D_avg.png"

    fig.savefig(os.path.join(save_dir, fname), bbox_inches="tight", dpi=200)
    plt.show()

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

def ploteigenvalues(values,plotdir, groupnr):
    r = np.cumsum(values)
    total = np.sum(values)

    fig, ax = plt.subplots()  # <--- makes sure it's 2D
    ax.plot(np.arange(1, len(values)+1), r / total * 100, marker="o")

    ax.set_xlabel("Number of eigenvalues", fontsize=12)
    ax.set_ylabel("Cumulative variance explained [%]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    outname = "variance" + "-group-" + str(groupnr) + ".png"
    plt.savefig(plotdir + outname, bbox_inches='tight')
    plt.show()
    plt.close(fig)

