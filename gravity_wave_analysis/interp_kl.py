import numpy as np
import xarray as xr
import gc
import constants
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import scipy.interpolate as interpolate
import multiprocessing as mp
import os

def read_3dspr(varname, dir_dec, spr_name_beg, groupid):
    path = dir_dec + varname + "/" + spr_name_beg + str(groupid) + ".nc"
    ds = xr.open_dataset(path)
    ampls_real = ds.ampls_real.data
    ampls_im = ds.ampls_imag.data
    k = ds.k.data
    l = ds.l.data
    maxind = ds.maxind.data

    # mask invalid wave numbers
    maxl = maxind[:, 1]
    mask = np.arange(l.shape[1])[None, :] >= np.array(maxl)[:, None]
    l[mask] = np.nan
    maxk = maxind[:, 0]
    mask = np.arange(k.shape[1])[None, :] >= np.array(maxk)[:, None]
    k[mask] = np.nan

    omega = ds.omega.data
    ntri = ds.ntri.data
    time = ds.time.data
    ds.close()

    return ampls_real, ampls_im, maxind, k, l, omega, ntri,time

def new_kl(ampls_real, ampls_im, k, l):
    minsk = np.nanmin(k, axis=1)
    maxsk = np.nanmax(k, axis=1)
    minsl = np.nanmin(l, axis=1)
    maxsl = np.nanmax(l, axis=1)
    min_l, max_l = np.nanmax(minsl), np.nanmin(maxsl)
    min_k, max_k = np.nanmax(minsk), np.nanmin(maxsk)
    #min_k = -0.32
    #max_k = 0.32
    #min_l = -0.31
    #max_l = 0.31
    ntri = k.shape[0]
    nk = k.shape[1]
    nl = l.shape[1]
    k_new = np.linspace(min_k, max_k, nk//2+2)[1:-1]
    l_new = np.linspace(min_l, max_l, nl//2+2)[1:-1]

    l_new_2D, k_new_2D = np.meshgrid(l_new, k_new, indexing="ij")
    #intdata = np.zeros((nz, ntri, nk//2, nl//2, len(m_new)), np.float64)
    k1d = np.repeat(k[:, np.newaxis, :], repeats=nl, axis=1)
    l1d = np.repeat(l[:, :, np.newaxis], repeats=nk, axis=2)
    k1d = k1d.reshape(*k1d.shape[:-2], -1)
    l1d = l1d.reshape(*l1d.shape[:-2], -1)
    ampls_real_1d = ampls_real.reshape(*ampls_real.shape[:-2], -1)
    ampls_im_1d = ampls_im.reshape(*ampls_im.shape[:-2], -1)
    return l_new_2D, k_new_2D, k1d, l1d, ampls_real_1d, ampls_im_1d, k_new, l_new

def createctypearrs_boolean(varlist):
    #Copy variables from varlist to new arrays that can be shared among multiprocessing processes
    result=[]
    for var in varlist:
        X_shape=var.shape
        X = mp.RawArray('b', var.size)
        X_np = np.frombuffer(X, dtype=bool).reshape(X_shape)
        np.copyto(X_np, var)
        result.append([X,X_shape])
    return result

def createctypearrs_float64(varlist):
    #Copy variables from varlist to new arrays that can be shared among multiprocessing processes
    result=[]
    for var in varlist:
        X_shape=var.shape
        X = mp.RawArray('d', var.size)
        X_np = np.frombuffer(X, dtype=np.float64).reshape(X_shape)
        np.copyto(X_np, var)
        result.append([X,X_shape])
    return result

def createctypearrs_int32(varlist):
    # Copy int32 variables (like maxind) to new RawArray for multiprocessing
    result = []
    for var in varlist:
        X_shape = var.shape
        X = mp.RawArray('i', int(np.prod(X_shape)))  # 'i' = int (usually 32-bit)
        X_np = np.frombuffer(X, dtype=np.int32).reshape(X_shape)
        np.copyto(X_np, var.astype(np.int32, copy=False))
        result.append([X, X_shape])
    return result

def create_var_dict(var_dict, varname, var):
    var_dict[varname] = var[0]
    var_dict[varname+'_sh'] = var[1]

def init_map(k, l, maxind,ampls_1d, k_new_2D, l_new_2D):
    if not 'var_dict' in globals():
        global var_dict
        var_dict = {}
        create_var_dict(var_dict=var_dict, varname="k", var=k)
        create_var_dict(var_dict=var_dict, varname="l", var=l)
        create_var_dict(var_dict=var_dict, varname="maxind", var=maxind)
        create_var_dict(var_dict=var_dict, varname="ampls", var=ampls_1d)
        create_var_dict(var_dict=var_dict, varname="k_new_2D", var=k_new_2D)
        create_var_dict(var_dict=var_dict, varname="l_new_2D", var=l_new_2D)

def debuffer_boolean(vartext):
    return[np.frombuffer(var_dict[text], dtype=bool).reshape(var_dict[text+'_sh']) for text in vartext]

def debuffer_float64(vartext):
    return[np.frombuffer(var_dict[text], dtype=np.float64).reshape(var_dict[text+'_sh']) for text in vartext]

def debuffer_int32(vartext):
    return [np.frombuffer(var_dict[text], dtype=np.int32).reshape(var_dict[text+'_sh']) for text in vartext]

def interpolate_for_triangle_structure_old(tri, k, l, maxind, ampls, k_new_2D, l_new_2D):
    n_time = ampls.shape[0]
    n_omega = ampls.shape[2]
    result_for_tri = np.empty((n_time, n_omega, k_new_2D.shape[0], k_new_2D.shape[1]))
    
    print(f"interpolating tri {tri} to kl")
    
    
    k_val = k[tri, :maxind[tri, 0]]
    l_val = l[tri, :maxind[tri, 1]]
    points_new = np.array([l_new_2D.ravel(), k_new_2D.ravel()]).T
    
    output_shape = k_new_2D.shape
    
    for t_i in range(n_time):
            for omega_i in range(n_omega):
                values = ampls[t_i, tri, omega_i, :maxind[tri, 1], :maxind[tri, 0]]
                interp = interpolate.RegularGridInterpolator((l_val, k_val), values)
                result_for_tri[t_i, omega_i, :, :] = interp(points_new).reshape(k_new_2D.shape)
    
    return tri, result_for_tri

def interpolate_for_triangle_structure(tri, k, l, maxind, ampls, k_new_2D, l_new_2D):
    n_time = ampls.shape[0]
    n_omega = ampls.shape[2]
    
    print(f"interpolating tri {tri} to kl")
    
    k_val = k[tri, :maxind[tri, 0]]
    l_val = l[tri, :maxind[tri, 1]]
    
    # original grid sizes for this triangle
    Nx_old = maxind[tri, 0]
    Ny_old = maxind[tri, 1]
    Nl_new, Nk_new = k_new_2D.shape
    
    # rescaling factor from Eq (16) — beta cancels, only N ratio remains
    rescale = (Nx_old / Nk_new) * (Ny_old / Nl_new)
    
    points_new = np.array([l_new_2D.ravel(), k_new_2D.ravel()]).T
    result_for_tri = np.empty((n_time, n_omega, k_new_2D.shape[0], k_new_2D.shape[1]))
    
    for t_i in range(n_time):
        for omega_i in range(n_omega):
            values = ampls[t_i, tri, omega_i, :maxind[tri, 1], :maxind[tri, 0]]
            interp = interpolate.RegularGridInterpolator((l_val, k_val), values)
            result_for_tri[t_i, omega_i, :, :] = interp(points_new).reshape(k_new_2D.shape)
    
    # ── Eq (16): rescale to account for new grid spacing ──────────────────────
    result_for_tri *= rescale

    return tri, result_for_tri

def interpolate_for_triangle_noarg_mp(tri):
    [k, l, ampls] = debuffer_float64(["k", "l","ampls"])
    [maxind] = debuffer_int32(["maxind"])
    [k_new_2D, l_new_2D] = debuffer_float64(["k_new_2D", "l_new_2D"])
    return interpolate_for_triangle_structure(tri=tri, k=k, l=l, maxind=maxind, ampls=ampls, k_new_2D=k_new_2D, l_new_2D=l_new_2D)

def multiprocess_intropolation(l_new_2D, k_new_2D, k, l, maxind,ampls, ntri, processes=None):
    """
    Fixed version with:
    1. Automatic process count based on CPU count (not 128!)
    2. Proper pool cleanup
    """
    # FIX 1: Use reasonable number of processes
    if processes is None:
        processes = min(32, os.cpu_count() or 1)  # Default to 32 max
    else:
        processes = min(processes, os.cpu_count() or 1)  # Don't exceed CPU count
    
    print(f"Using {processes} processes for interpolation")
    
    num_time = ampls.shape[0]
    num_tri = ampls.shape[1]
    num_omega = ampls.shape[2]
    initdata = np.zeros((num_time, num_tri, num_omega, k_new_2D.shape[0], k_new_2D.shape[1]))
    
    # Print memory usage for debugging
    ampls_size_gb = ampls.nbytes / 1e9
    print(f"Creating shared memory for ampls: {ampls_size_gb:.2f} GB")
    
    [_k,_l, _ampls]=createctypearrs_float64([k, l, ampls])
    [_maxind]       = createctypearrs_int32([maxind])
    [_k_new_2D,_l_new_2D]=createctypearrs_float64([k_new_2D,l_new_2D])
    
    # FIX 2: Properly close pool
    pool = None
    try:
        pool = mp.Pool(processes=processes, initializer=init_map, 
                      initargs=(_k, _l, _maxind, _ampls, _k_new_2D, _l_new_2D))
        print("Arguments created, starting interpolation...")
        
        ran = np.arange(len(ntri))
        res = pool.map(interpolate_for_triangle_noarg_mp, ran)
        print("Pooling finished, reordering...")
        
        int_tri = np.empty(len(ntri))
        res_tri = np.empty(len(ntri))
        for ir,r in enumerate(res):
            initdata[:, ir, :, :, :] = r[1]
            int_tri[ir] = r[0]

        int_tri = np.array(int_tri, dtype=int)
        res_tri = ntri[int_tri]
        if ((res_tri[1:]-res_tri[:-1])==1).all():
            print("Order of triangles is correct.")
        else:
            print("Wrong order of triangles, need to correct result ntri.")
    finally:
        # FIX 3: Always cleanup pool
        if pool is not None:
            pool.close()
            pool.join()
            del pool
        
        # FIX 4: Force garbage collection to free shared memory
        gc.collect()
    
    return initdata, res_tri

def save_interpolated(save_path,intdata_real, intdata_im, omega,k_new, l_new, tri, varname, groupid, time):
    ds=xr.Dataset(data_vars=dict(
            ampls_real=(["time","ntri","omega","l","k"], intdata_real),
            ampls_im=(["time","ntri","omega","l","k"], intdata_im),
            omega = (["omega"], omega),
            k=(["nk"], k_new),
            l=(["nl"], l_new),
        ),
        coords=dict(
            ntri=np.array(tri), #Indices of the original triangles
            time=time[:],
        ),
        attrs=dict(description="Interpolated data for the group",
                    varname=varname,),
        )
    file_name = "sp3D-"+ varname +"-ny4_window_group-" + str(groupid) + ".nc"
    ds.to_netcdf(save_path+file_name)
    ds.close()

def interpolate_kl(groupid, varname):
    dir_dec = "/work/bb1097/b383521/dummy/regroup/"
    spr_name_beg = "sp3D-"+ varname +"-ny4_window_group-"
    save_path = "/work/bb1097/b383521/dummy/regroup_H_interpolation/" + varname + "/"
    ampls_real, ampls_im, maxind, k, l, omega, ntri,time = read_3dspr(varname=varname, dir_dec=dir_dec, spr_name_beg=spr_name_beg, groupid=groupid)
    l_new_2D, k_new_2D, k1d, l1d, ampls_real_1d, ampls_im_1d, k_new, l_new = new_kl(ampls_real=ampls_real, ampls_im=ampls_im, k=k, l=l)

    
    # FIX 5: Use reasonable process count (32 instead of 128)
    intdata_real, res_tri = multiprocess_intropolation(l_new_2D=l_new_2D, k_new_2D=k_new_2D, k=k, l=l, maxind=maxind,ampls=ampls_real, ntri=ntri, processes=32)
    
    # FIX 6: Force cleanup between calls
    gc.collect()
    
    intdata_im, res_tri = multiprocess_intropolation(l_new_2D=l_new_2D, k_new_2D=k_new_2D, k=k, l=l, maxind=maxind,ampls=ampls_im, ntri=ntri, processes=32)
    
    save_interpolated(save_path,intdata_real, intdata_im, omega,k_new, l_new, res_tri, varname, groupid, time)
    del  intdata_im, intdata_real, k1d, ampls_real_1d, ampls_im_1d,  ampls_real, ampls_im
    gc.collect()


if __name__ == "__main__":
    varname = "p"
    dir_dec = "/work/bb1097/b383521/3D_spectrum_regrouping_daily/"
    spr_name_beg = "sp3D-"+ varname +"-ny4_window_group-"
    groupid = 4
    save_path = "/work/bb1097/b383521/3D_spc_negr_lintrkl_daily/" + varname + "/"
    ## read data
    ampls_real, ampls_im, maxind, k, l, omega, ntri,time = read_3dspr(varname=varname, dir_dec=dir_dec, spr_name_beg=spr_name_beg, groupid=groupid)
    l_new_2D, k_new_2D, k1d, l1d, ampls_real_1d, ampls_im_1d, k_new, l_new = new_kl(ampls_real=ampls_real, ampls_im=ampls_im, k=k, l=l)
    intdata_real, res_tri = multiprocess_intropolation(l_new_2D=l_new_2D, k_new_2D=k_new_2D, k=k, l=l, maxind=maxind, ampls=ampls_real, ntri=ntri, processes=32)
    print(f"shape of intdata real : {intdata_real.shape}")
    intdata_im, res_tri = multiprocess_intropolation(l_new_2D=l_new_2D, k_new_2D=k_new_2D, k=k, l=l, maxind=maxind,ampls=ampls_im, ntri=ntri, processes=32)
    save_interpolated(save_path,intdata_real, intdata_im, omega,k_new, l_new, res_tri, varname, groupid, time)