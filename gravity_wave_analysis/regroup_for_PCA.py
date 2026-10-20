import numpy as np
import xarray as xr
import gc
import pandas as pd
import time as time_time

def timespfname(group_id, varname):
    return "sp3D-"+varname+"-ny4_window_group-"+str(group_id)+".nc"

def regroup(var):
    print(f"regrouping var : {var}")
    varname=var
    spectra_dir= "/work/bb1097/b383521/dummy/3D_spr_old_group/" + varname + "/"
    #decomposition_dir="/work/bb1097/b381585/data/decomposed-correction/"

    newgr = 10
    newnr = 2048
    oldgr = 160
    oldnr = 128
    fn_grid_low = "/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    grid_l = xr.open_dataset(fn_grid_low)
    clat = grid_l.clat.data

    so = np.argsort(clat)[::-1] # from high lat to low lat
    indicies = so.reshape((newgr, newnr))
    filenr = indicies//oldnr # each entry indicate that which old .nc file should be opened 
    file_po = indicies%oldnr # at associate group which tri angles is for this lat

    filesh = xr.open_dataset(spectra_dir+timespfname(56, varname))
    shape = filesh.ampls_real.data.shape #(time, tri, omega, l, k)

    for nfi in range(10):
        start_time = time_time.time()
        # for each lat old file number and position
        ofs_ind = filenr[nfi, :]
        file_ind = file_po[nfi, :]
        # list to keep new tri
        outlist_real = []
        outlist_img = []
        ks = []
        ls = []
        tri_out = []
        maxind = np.zeros((newnr, 2), dtype=np.int64)

        # index for newnr
        i = 0
        ofs_ind_un, ofs_counts = np.unique(ofs_ind, return_counts=True)
        for ofi in range(0, len(ofs_ind_un)):
            print("open the file number "+str(ofs_ind_un[ofi])+" file.")
            filesh = xr.open_dataset(spectra_dir+timespfname(ofs_ind_un[ofi], varname))
            in_k = filesh.k.data
            in_l = filesh.l.data
            in_tri = filesh.ntri.data
            in_maxind = filesh.max_ind.data
            inarray_real = filesh.ampls_real.data
            inarray_imag = filesh.ampls_imag.data

            wh = np.where(ofs_ind == ofs_ind_un[ofi])
            pos_in_file = file_ind[wh]
            for trii in range(0, ofs_counts[ofi]):
                ptrii = pos_in_file[trii]
                outlist_real.append(inarray_real[:, [ptrii], :, :, :])
                outlist_img.append(inarray_imag[:, [ptrii], :, :, :])
                ks.append(in_k[ptrii, :])
                ls.append(in_l[ptrii, :])
                tri_out.append(in_tri[ptrii])
                maxind[i, :] = in_maxind[ptrii, :]
                i=i+1

        # Regroup lists into arrays now
        maxk = np.amax(maxind[:, 0])
        maxl = np.amax(maxind[:, 1])
        k = np.zeros((newnr, maxk))
        l = np.zeros((newnr, maxl))
        shape_list = list(shape) 
        shape_list[1] = newnr
        shape_list[-2] = maxl
        shape_list[-1] = maxk
        outarray_re = np.zeros((tuple(shape_list)))
        outarray_im = np.zeros((tuple(shape_list)))

        for i in range(0, len(ks)):
            maxl_min = maxind[i, 1]
            maxk_min = maxind[i, 0]
            k[i, :maxk_min] = ks[i][:maxk_min] # fill only zero values
            l[i, :maxl_min] = ls[i][:maxl_min] # fill only zero values
            minl = outlist_real[i][:, 0, :,:maxl_min, :maxk_min].shape[-2]
            mink = outlist_real[i][:, 0, :,:maxl_min, :maxk_min].shape[-1]
            assert (minl==maxl_min) and (mink==maxk_min), print("The assigning above is not so strange.")
            outarray_re[:, i, :, :minl, :mink] = outlist_real[i][:, 0, :, :maxl_min, :maxk_min]
            outarray_im[:, i, :, :minl, :mink] = outlist_img[i][:, 0, :, :maxl_min, :maxk_min]
        # read additional paramters from file
        
        del outlist_real,outlist_img,ks,ls
        gc.collect()
        time = filesh.time.data
        tri = np.array(tri_out)
        print("Preparing for saving.")

        save_path="/work/bb1097/b383521/dummy/regroup/" +varname+ "/"+"sp3D-"+varname+"-ny4_window_group-"+str(nfi)+".nc"
        ampls_real = outarray_re
        ampls_imag = outarray_im
        ntime = time.shape[0]
        time = filesh.time.data
        omega = filesh.omega.data
        ds=xr.Dataset(data_vars=dict(
                                ampls_real = (["time", "ntri", "omega", "l", "k"], ampls_real),
                                ampls_imag = (["time", "ntri", "omega", "l", "k"], ampls_imag),
                                k          = (["tri", "nk"], k),
                                l          = (["tri", "nl"], l),
                                omega      = (["no"], omega),
                                maxind     = (["ntri", "dim"], maxind)),
                    coords=dict(
                                ntri = np.array(tri),
                                time = np.array(time[:ntime]),
                                ),
                    attrs = dict(description = "Fouriere Amplitudes", varname=varname))
        ds.to_netcdf(save_path)
        end_time = time_time.time()
        print(f"File {nfi} for variable {var} finished.")
        print(f"execuation take {start_time - end_time}")
        del ampls_real, ampls_imag, inarray_real, inarray_imag