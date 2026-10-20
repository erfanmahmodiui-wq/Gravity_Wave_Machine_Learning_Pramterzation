import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse.linalg as sl
from sklearn.decomposition import PCA
#from PCA_plot import plot_cov

def read_interp_data(intrp_path):
    ds = xr.open_dataset(intrp_path)
    k = ds.k.data
    l = ds.l.data
    m = ds.m.data
    intdata = ds.data.data
    time = ds.time.data
    ntri = ds.ntri.data
    return k, l, m, intdata, time, ntri

def reshape_data(intdata):
    #intdata_rot = np.transpose(intdata,(0, 4, 1, 2, 3)) # reshape into (nz, nm, ntri, nl, nk)
    #data = intdata[:, 0, ...] # (ntime, ntri, l, k, m)
    sh = intdata.shape
    sh_new = sh[:2]+(sh[2]*sh[3]*sh[4],)
    intdata_res = np.reshape(intdata, sh_new)
    sh2 = intdata_res.shape
    n_smaples = sh2[0]*sh2[1]
    data_res = intdata_res.reshape(n_smaples, sh2[2])
    return data_res, sh

def save_eigvals_vects(eigvals, eigvects, avrage,k, l, m, savedir, groupnr):
    eigvecs_resh=np.transpose(eigvects,(1,2,3,0))

    ds=xr.Dataset(
        data_vars=dict(
            eigenvalues=(["neig","m"], eigvals.T),
            eigenvectors=(["neig","l","k","m"],eigvecs_resh),
            avrage = (["l","k", "m"], avrage)
        ),
        coords=dict(
            neig=np.arange(eigvals.T.shape[0]), #Indices of the original triangles
            k=k[:],
            l=l[:],
            m=m[:],
        ),
        attrs=dict(description="Eigenvalues and eigenvectors from the PCA"),
        )
    ds.to_netcdf(savedir+"pca_new_eigen_group_"+str(groupnr)+".nc")

def save_eigvals_vects_daily(eigvals, eigvects, avrage,k, l, m, savedir, groupnr):
    print(m)
    ds=xr.Dataset(
        data_vars=dict(
            eigenvalues=(["neig"], eigvals),
            eigenvectors=(["neig","l","k","m"],eigvects),
            avrage = (["l","k", "m"], avrage)
        ),
        coords=dict(
            neig=np.arange(len(eigvals)), #Indices of the original triangles
            k=k[:],
            l=l[:],
            m=m[:],
        ),
        attrs=dict(description="Eigenvalues and eigenvectors from the PCA"),
        )
    ds.to_netcdf(savedir+"pca_new_eigen_group_"+str(groupnr)+".nc")


def fill_nan(intdata_res):
    av_sp = np.nanmean(intdata_res, axis=2, keepdims=True)
    for zi in range(0, len(alt)):
        data_masked = np.ma.masked_invalid(intdata_res[zi, ...])
        for mi in range(0, data_masked.shape[0]):
            for kli in range(0, data_masked.shape[-1]):
                whinvalid = data_masked[mi, :, kli].mask
                av = data_masked[mi, :, kli].mean()
                if not np.ma.is_masked(av):
                    data_masked[mi, whinvalid, kli].mask = False
                    data_masked[mi, whinvalid, kli] = av
                else:
                    print("Whole group of nans.")

    data = data_masked.filled(np.nan)
    return data #(mi, ntri, k*l)


def fill_nan_daily(data):
    av_time_space = np.nanmean(data, axis=0)
    data_masked = np.ma.masked_invalid(data)
    for klmi in range(data_masked.shape[1]):
        whinvalid = data_masked[:, klmi].mask
        av = data_masked[:, klmi].mean()
        if not np.ma.is_masked(av):
            data_masked[whinvalid, klmi].mask = False
            data_masked[whinvalid, klmi] = av
        else:
            print("Whole group of nans.")
    
    data = data_masked.filled(np.nan)
    return data

def PCA_fun(data, m, sh):
    eigvals = np.empty((len(m),data.shape[-1])) #(vertical_number, number of eigenvalues)
    eigvecs = np.empty((len(m),data.shape[-1],sh[3],sh[4])) #(m, number of eigenvectors, k, l)
    avrage = np.empty((len(m), sh[3], sh[4]))
    pca = PCA()
    for mi in range(0, len(m)):
        pca.fit(data[mi, ...])
        avrage[mi, :, :] = pca.mean_.reshape(sh[3], sh[4])
        eigenvectors = pca.components_
        for num_eigvect in range(0, data.shape[2]):
            eigvecs[mi, num_eigvect, :, :] = eigenvectors[num_eigvect, ...].reshape((sh[3], sh[4]))


        eigvals[mi, :] = pca.explained_variance_
    return eigvecs, eigvals, avrage

def PCA_fun_daily(data, shape):
    # shape expected like (_, _, 10, 11, 24) so that shape[2]*shape[3]*shape[4] == n_features
    n_features = data.shape[1]
    k, l, m = shape[2], shape[3], shape[4]
    assert k * l * m == n_features, "Feature count must equal k*l*m"

    # Fit PCA
    pca = PCA()  # or PCA(n_components=..., svd_solver='randomized')
    pca.fit(data)

    # Mean reshaped back
    avrage = pca.mean_.reshape(k, l, m)

    # Eigenvalues
    eigvals = pca.explained_variance_

    # Number of components actually produced
    n_components = pca.components_.shape[0]

    # Allocate eigenvectors (components reshaped)
    eigvecs = np.empty((n_components, k, l, m), dtype=pca.components_.dtype)

    # Fill per-component
    for i in range(n_components):
        eigvecs[i] = pca.components_[i].reshape(k, l, m)

    return avrage, eigvecs, eigvals


"""""
def Compute_and_Save_PCA(intdata_res, sh, k, l, m, alt,Neig,savedir, groupnr):
    nrdata=np.sum(np.isfinite(intdata_res),axis=2)[0,...]

    #Compute the averages and center the data
    #av_sp = np.nanmean(intdata_res, axis=2, keepdims=True) #Averages intdata_res from z,m,tri,kl to z,m,1,kl
    #intdata_res=intdata_res-av_sp #Centering the values
    for zi in range(0,len(alt)):
            
            data_masked=np.ma.masked_invalid(intdata_res[zi,...]) #result is (nm,tri,nk*nl)
            #Fill with mean if nan in the second dimension
            for mi in range(0,data_masked.shape[0]):
                for kli in range(0,data_masked.shape[-1]):
                    whinvalid=data_masked[mi,:,kli].mask
                    av=data_masked[mi,:,kli].mean()
                    if not np.ma.is_masked(av):
                        data_masked[mi,whinvalid,kli].mask=False
                        data_masked[mi,whinvalid,kli]=av
                    else:
                        print("Whole group of nans.")
            
            eigvals=np.empty((len(m),data_masked.shape[-1])) #Shape (m,number of eigenvalue)
            eigvecs=np.empty((len(m),data_masked.shape[-1],sh[3],sh[4])) #Shape (m, number of eigenvector,k,l)
            print("Minimum",np.amin(2*3.14/m[:]))
            print("Maximum",np.amax(2*3.14/m[:]))
            for mi in range(0,len(m)):
            
                print("M is "+str(m[mi]))
                print("Wavelength is "+str(2*3.14/m[mi]))

                #Do something
                print("Computing the covariance matrix")
                data_comp1=data_masked[mi,...]
                #Subset rows that are not nans
                all_nan_rows = data_comp1.mask.any(axis=0)
                varmult=data_comp1.shape[0]/nrdata[mi,~all_nan_rows][np.newaxis,:] #Rescale so that real number of data is considered for the covariance
                data_comp1_sub=data_comp1[:,~all_nan_rows]*varmult

                #Get back to the original length:
                cov=np.matmul(data_comp1_sub.data.T,data_comp1_sub.data)

                print("Computing eigenvalues.")
                val_sub,vec_sub=sl.eigs(cov,k=Neig)
                #Add nans to vecs:
                vec=np.empty((data_comp1.shape[1],data_comp1.shape[1]))
                vec[:] = np.nan
                vec[np.ix_(~all_nan_rows, ~all_nan_rows)] = vec_sub
                val=np.empty(data_comp1.shape[1])
                val[:]=np.nan
                val[~all_nan_rows]=val_sub
                #Sort eigenvectors
                idx = val.argsort()[::-1]   
                val = val[idx]
                vec = vec[:,idx]
                i0=np.where(np.isfinite(val))[0][0] #Index of first eigenvalue that is not nan, now usually (always?) 0
                #plot the eigenvalue distribution
                #val[i0:] contains the eigenvalues
                eigvals[mi,:]=val[:]
                #for ivec in range(0,eigvecs.shape[1]):
                #    eigvecs[mi,ivec,:,:]=np.reshape(vec[:,ivec],sh[3:])
                eigvecs[mi,...]=vec.T.reshape(eigvecs.shape[1], *sh[3:])
                #var_plot = True
                #if var_plot == True:
                #    plot_cov(cov=cov, idx=mi)

                
            #Transpose the eigenvalues and eigenvectors to a more reasonable form
            eigvals_resh=np.transpose(eigvals) #Now the shape is (nr. of eigenvalue, m)
            eigvecs_resh=np.transpose(eigvecs,(1,2,3,0)) #Now the shape is (nr. of eigenvector, k,l,m)
            save_eigvals_vects(eigvals=eigvals_resh, eigvects=eigvecs_resh, k=k, l_new=l, m_new=m, savedir=savedir, groupnr=groupnr)
"""

def compute_EOFs(groupnr):
    dir_dec = "/work/bb1097/b383521/slided_window_avrage/interp_wave_action_density/"
    intrpath = dir_dec+"interp-alt0-regroup-"+str(groupnr)+ ".nc"
    savedir = "/work/bb1097/b383521/slided_window_avrage/EOF/"
    k, l, m, intdata, time = read_interp_data(intrp_path=intrpath)
    intdata_res, sh = reshape_data(intdata)
    data = fill_nan_daily(data=intdata_res)
    data = np.log10(data)
    avrage, eigvecs, eigvals = PCA_fun_daily(data=data, shape=sh)
    print(f"shape of mean {avrage.shape}")
    save_eigvals_vects_daily(eigvals=eigvals, eigvects=eigvecs, avrage=avrage, k=k, l=l, m=m, savedir=savedir, groupnr=groupnr)

def save_avrage(avg, k, l, m, savedir, file_slide, time, groupnr):
    ds=xr.Dataset(
        data_vars=dict(
            avrage = (["l","k", "m"], avg)
        ),
        coords=dict(
            k=k[:],
            l=l[:],
            m=m[:],
            time = time[:]
        ),
        attrs=dict(description="Eigenvalues and eigenvectors from the PCA"),
        )
    ds.to_netcdf(savedir+"avarage-action-window-file-slide_" + str(file_slide) + "-group-" + str(groupnr)  + ".nc")

def save_interp_filled_action(time, ntri, l, k, m, data, groupnr, file_slide, savedir):

    data = np.reshape(data, (len(time), len(ntri), len(l), len(k), len(m)))
    ds = xr.Dataset(
        data_vars={
            "data": (["time", "ntri", "l", "k", "m"], data)
        },
        coords={
            "time": time,
            "ntri": ntri,
            "l": l,
            "k": k,
            "m": m,
        },
        attrs={"description": "interpolated wave action density with filled data"},
    )
    ds.to_netcdf(savedir + f"projected_data_fileslide-{file_slide}-group-{groupnr}.nc")

def compute_avrgae_save(groupnr, file_slide):
    dir_dec = "/work/bb1097/b383521/dummy/interp_wave_action_density/"
    intrpath = dir_dec+"interp-alt0-fileslide_" + str(file_slide) +"-regroup-"+str(groupnr)+".nc"
    savedir = "/work/bb1097/b383521/dummy/avrage/"
    save_dir_filled_action = "/work/bb1097/b383521/dummy/inter_wave_action_density_filled_log/"
    k, l, m, intdata, time, ntri = read_interp_data(intrp_path=intrpath)
    intdata_res, sh = reshape_data(intdata)
    data = fill_nan_daily(data=intdata_res)
    data = np.log10(data)
    save_interp_filled_action(time=time, ntri=ntri, l=l, k=k, m=m, data=data, file_slide=file_slide, savedir=save_dir_filled_action, groupnr=groupnr)
    data = data.T
    print(f"shape of data {data.shape}")
    avrage = np.mean(data, axis=1)
    print(f'shape of avargae: {avrage.shape}')
    avg = avrage.reshape(len(l), len(k), len(m))
    save_avrage(avg, k, l, m, savedir, file_slide, time, groupnr)

def ntri_orgraghy_nonorgrghy_index(resolved_flow_dirctory , time_coff_dir , group_id):
    ## read ntri index of the group_id, and then find the index of orography and non-orography
    with xr.open_dataset(time_coff_dir + f"full-time-coff_{group_id}.nc", engine="h5netcdf") as ds:
        ntri = ds.ntri.data
    
    with xr.open_dataset(resolved_flow_dirctory + "orography.nc", engine="h5netcdf") as ds:
        values = ds.Hspec.data[ntri, :]
        oro_sorted = -np.sort(-values, axis=1)
        ntri_orography = ntri[np.where(oro_sorted[:, 0] > 0)[0]]
        ntri_nonorography = ntri[np.where(oro_sorted[:, 0] == 0)[0]]
        ds.close()
    return ntri_orography, ntri_nonorography

if __name__ == "__main__":
    
    groupnr = 4
    dir_dec = "/work/bb1097/b383521/interp_wave_density_klm-daily/"
    intrpath = dir_dec+"interp-alt0-regroup-"+str(groupnr)+ ".nc"
    savedir = "/work/bb1097/b383521/new_PCA_daily/"
    k, l, m, intdata, time = read_interp_data(intrp_path=intrpath)
    intdata_res, sh = reshape_data(intdata)
    #data = fill_nan(intdata_res=intdata_res)
    data = fill_nan_daily(data=intdata_res)
    #data = np.log10(data)
    #eigvecs, eigvals, avrage = PCA_fun(data=data, m=m, sh=sh)
    avrage, eigvecs, eigvals = PCA_fun_daily(data=data, shape=sh)
    print(f"shape of mean {avrage.shape}")
    save_eigvals_vects_daily(eigvals=eigvals, eigvects=eigvecs, avrage=avrage, k=k, l=l, m=m, savedir=savedir, groupnr=groupnr)