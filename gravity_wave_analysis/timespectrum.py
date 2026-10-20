import numpy as np
import multiprocessing as mp
import xarray as xr
import os
from datetime import datetime
import math
from functools import partial
import time
from pyjokes import get_joke
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path



def set_max_size_amplitude(ampls,maxind,tri_i):
    [maxik,maxil]=maxind[tri_i,:]
    amplssub=ampls[:, tri_i,:maxil,:maxik]
    return amplssub

def compute_time_spectrum(ampls):
    sp=np.fft.fftshift(np.fft.fft(ampls,axis=0),axes=0)[::-1,...] #::-1 because of the change in omega definition
    return sp

def compute_omega(time):
    dt0=time[1]-time[0]
    timem=time[1:]-time[:-1]
    assert (timem==dt0).all(), "Data are not equidistant in time"
    dt=dt0.seconds
    fr=-np.fft.fftshift(np.fft.fftfreq(len(time),dt))[::-1]*2*np.pi #Change the sign and the multiplicators to have the convention of k*x+l*y-omega*t
    return fr

def spectra_on_1_triangle(ampls, maxind, tri_i,shape_or):
    ampls=ampls.reshape(shape_or[:-1])
    ampls=ampls.transpose()
    ampls2=set_max_size_amplitude(ampls,maxind,tri_i)
    spr=compute_time_spectrum(ampls2)
    return spr


def compute_time_spectra(ampls,time,ntri,k,l,maxind):
    resspr=[]
    resfr=compute_omega(time)
    for tri_i in ntri:
        spr=spectra_on_1_triangle(ampls, maxind, tri_i)
        resspr.append(spr)
    return resspr,resfr #resspr contains list of spectra for individual triangles, resfr is omega with only the time-frequency dimension

def readfiles_horsp(listfiles,directory):
    lf2=[directory+lf for lf in listfiles]
    ds=xr.open_mfdataset(lf2,concat_dim="time",combine="nested")
    return ds

def readvars_horsp(listfiles,directory):
    ds=readfiles_horsp(listfiles,directory)
    ampls=ds["ampls_real"]+ds["ampls_imag"]*1j #time,z,ntri,l,k
    time=ds["time"]
    ntri=ds["ntri"]
    k=ds["k"]#ntri,nk
    l=ds["l"]
    if ds.max_ind.data.shape[2] == 2:
        maxind=ds["max_ind"][0, ...]
    else:
        maxind = ds["max_ind"][:, :, 0]

    #center=ds["center"][0,:,:]
    alt=ds["z"]
    ds.close() #prevent problems
    return ampls,time,ntri,k,l,maxind,alt #ampls,time,ntri,k,l,maxind,center,alt

def listfiles_horsp(directory,filebeg):
    lis=os.listdir(directory)
    #Expecting file name in the form filebeg+'20200123T000000Z.nc'
    onevarfiles=[b for b in lis if b.startswith(filebeg)]
    sf=sortfilenames(onevarfiles)
    return sf

def sortfilenames(filelist):
    datep=[fi.split("_")[-1][:-4] for fi in filelist]
    #datep=[fi.split("_")[-1][:-15] for fi in filelist]
    dtt=[]
    for datep1 in datep:
        year=datep1[9:11]
        if year=="24":
            day=int(datep1[6:8])
            listdatep1=list(datep1)
            #listdatep1[6:8]=str(day+1)
            listdatep1[9:11]="23"
            listdatep1[11:13]="59"
            datep1=''.join(listdatep1)
        d=datetime.strptime(datep1,"%Y%m%dT%H%M%S")
        dtt.append(d)
    #dtt=[datetime.strptime(datep1,"%Y%m%dT%H%M%S") for datep1 in datep] #e.g.datetime.fromisoformat('20111104T000523') - works in newer version of python only
    order=np.argsort(np.array(dtt))
    sorted=np.array(filelist)[order]
    return sorted.tolist()

def convert_time(time):
    #Convert time from the original format in the files to datetime
    time2=np.empty(len(time),dtype=object)
    for i,t in enumerate(time.values):
        d0=datetime.strptime(str(int(math.floor(t))),"%Y%m%d")
        rest=(t-math.floor(t))
        sec=int(round(24*60*60*rest))
        second=sec % 60
        minute=int(((sec-second)/60) % 60)
        hour=int(math.floor((sec-second)/60/60))
        dt=datetime(d0.year,d0.month,d0.day,hour,minute,second)
        time2[i]=dt
    return time2

def listandread(directory, varname, group, file_slide):
    namebeg="horsp-"+varname+"-ny4_group-"+str(group)+"_"
    nrfiles = 13
    lis=listfiles_horsp(directory,namebeg)[file_slide:nrfiles+file_slide] 
    #ampls,time,ntri,k,l,maxind,center,alt=readvars_horsp(l,directory)
    ampls,time,ntri,k,l,maxind,alt=readvars_horsp(lis,directory)
    time2=convert_time(time)
    #return ampls.values,time2,ntri.values,k.values,l.values,maxind.values,center.values,alt.values
    return ampls.values,time2,ntri.values,k.values,l.values,maxind.values,alt.values

def rolling_window(a, window_size, step=6):
    shape = ((a.shape[0] - window_size)//step + 1, window_size) + a.shape[1:]
    strides = (a.strides[0]*step,) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def compute_time_spectra_window(ampls,time,ntri,k,l,maxind,window_length):
    #Window length: how many timesteps should be used to calculate the individual spectra. If len(time), uses whole data
    #nt_res=len(time)-window_length #Number of timesteps of the resulting spectrum
    resfr=compute_omega(time[0:window_length])#Assumes that the time values are equidistant, so omega will be same for all the data
   
    #Prepare list of windows:
    lin=rolling_window(ampls,window_length) #Shape of ampls is (time, z, ntri, l, k) => rolling over first dimension
    #Reshape so that the field is 2D with the time axis at the last position (necessary because of the apply_along_axis function)
    lin=np.transpose(lin)
    shape_or=lin.shape
    lin_res = lin.reshape(-1, lin.shape[-1])#Merge all but the last axis
    resspr=[]
    for tri_i in range(0,len(ntri)):
        sp1tr_lambda=lambda am: spectra_on_1_triangle(am, maxind, tri_i,shape_or)
        #spr = sp1tr_lambda(ampls)
        spr=np.apply_along_axis(sp1tr_lambda,0,lin_res)
        sh1=tuple(i for i in range(0, len(spr.shape)))
        newshap=(sh1[-1],)+sh1[:-1]
        sprt=np.transpose(spr,newshap)
        resspr.append(sprt)


    return resspr,resfr #resspr contains list of spectra for individual triangles, resfr is omega with only the time-frequency dimension

def save_time_spectra_window_norot(path, varname, group,resultspectrum, k, l, maxind, omega, time, tri):
    save_time_spectra_window_norot_realim(path, varname, group, resultspectrum.real,resultspectrum.imag, k, l, maxind, omega, time, tri)

def save_time_spectra_window_norot_realim_old(path, varname, group,resultspectrum_re,resultspectrum_im, k, l, maxind, omega, time, tri):
    print(f"group : {group}")
    if resultspectrum_re.shape[0] == 2  :
        #time = time[:12]
        if varname == "gw_plus" or varname == "gw_minus":
            time = time[:]
        else:
            time = time[[0, 6]]
    if resultspectrum_re.shape[0] == 1:
        time = time[[0]]
    #ntime=len(time)-len(omega)+1
    print("time length:", len(time))
    print("ampls_real shape:", resultspectrum_re.shape)
    ds=xr.Dataset(data_vars=dict(
            ampls_real=(["time","ntri","omega","l","k"], resultspectrum_re),
            ampls_imag=(["time","ntri","omega","l","k"], resultspectrum_im),
            k=(["ntri", "nk"], k),
            l=(["ntri", "nl"], l),
            omega=(["no"],omega),
            max_ind=(['ntri','dims'],maxind),
        ),
        coords=dict(
            ntri=np.array(tri),
            time=time[:],
        ),
        attrs=dict(description="Fourier amplitudes",
                    varname=varname,),
        )
    
    file_name = "sp3D-" + varname + "-ny4_window_group-" + str(group) + ".nc"
    ds.to_netcdf(path+file_name)
    ds.close()

def save_time_spectra_window_norot_realim(path, varname, group, resultspectrum_re, resultspectrum_im, k, l, maxind, omega, time, tri):
    if resultspectrum_re.shape[0] == 1:
        time = time[[0]]
    print(f"group : {group}")
    print("time length:", time)
    print("ampls_real shape:", resultspectrum_re.shape)
    ds = xr.Dataset(data_vars=dict(
            ampls_real=(["time", "ntri", "omega", "l", "k"], resultspectrum_re),
            ampls_imag=(["time", "ntri", "omega", "l", "k"], resultspectrum_im),
            k=(["ntri", "nk"], k),
            l=(["ntri", "nl"], l),
            omega=(["no"], omega),
            max_ind=(['ntri', 'dims'], maxind),
        ),
        coords=dict(
            ntri=np.array(tri),
            time=time,  # already correct length
        ),
        attrs=dict(description="Fourier amplitudes", varname=varname),
    )
    file_name = "sp3D-" + varname + "-ny4_window_group-" + str(group) + ".nc"
    ds.to_netcdf(path + file_name)
    ds.close()

def save_time_spectra_window(path,varname, group,resultspectrum,k,l,maxind,omega,time,tri):
    #Save the horizontal spectra together with the temporal component
    
    ntri=len(tri)
    ntime=resultspectrum[0].shape[0]
    nom=resultspectrum[0].shape[1]
    nz=resultspectrum[0].shape[2]
    fr_big=np.zeros((ntime, nom, ntri, l.shape[1], k.shape[1]),dtype=complex) #(ntime, omega, ntri, l, k)
    for i in range(0,ntri):
        [maxk,maxl]=maxind[i,:]
        fr_big[:, :, i, :maxl, :maxk]=resultspectrum[i]
    resultspectrum=np.transpose(fr_big,(0,2,1,3,4))
    save_time_spectra_window_norot(path, varname, group,resultspectrum, k, l, maxind, omega, time, tri)

def read_spectrum_write_noav_old(varname, directory, save_path, file_slide, group):
    ampls, time2, tri, k, l, maxind, alt = listandread(directory, varname, group, file_slide)
    alt = alt[0]
    window_length = 24*6
    ampls_f = ampls[:, 0, ...]
    time_f = time2[:]
    #if file_slide == 0:
    #    ampls_f=ampls[:, 0, ...] #(time, height, ntri, k, l)
    #    time_f = time2[:]
    #else:
    #    ampls_f=ampls[1:, 0, ...]
    #    time_f = time2[1:]

    resultspectrum,omega=compute_time_spectra_window(ampls_f, time_f, tri, k, l, maxind, window_length)
    # 3 windows, keep first 2, drop overlapping 3rd
    resultspectrum = [r[:2] for r in resultspectrum]
    save_time_spectra_window(save_path, varname, group, resultspectrum, k, l, maxind, omega, time_f, tri)
    print("Group "+str(group)+" finished.")

def read_spectrum_write_noav(varname, directory, save_path, file_slide, group):
    ampls, time2, tri, k, l, maxind, alt = listandread(directory, varname, group, file_slide)
    alt = alt[0]
    window_length = 24*6
    ampls_f = ampls[:, 0, ...]
    time_f = time2[:]

    resultspectrum, omega = compute_time_spectra_window(ampls_f, time_f, tri, k, l, maxind, window_length)
    
    # Keep first 2 windows, drop overlapping 3rd
    resultspectrum = [r[:2] for r in resultspectrum]
    window_times = time_f[[0, 6]]  # ← pass correct times directly

    save_time_spectra_window(save_path, varname, group, resultspectrum, k, l, maxind, omega, window_times, tri)
    print("Group " + str(group) + " finished.")


def compute_3spr(file_slide, varname):
    directory="/work/bb1097/b381585/data/horizontal-correction/"+varname+"/"
    save_path = "/work/bb1097/b383521/slided_window_avrage/3D_spr_old_group/" + varname + "/"
    processes=3
    proc_group = partial(read_spectrum_write_noav, varname, directory, save_path, file_slide)
    st=time.time()
    p=mp.Pool(processes)
    p.map(proc_group,np.arange(0,160))#160))    ## change this
    p.close()
    p.join()
    #proc_group(0)
    et=time.time()
    print("Took time "+str(et-st)+ " s.")
    print(get_joke())

if __name__ == "__main__":
    varname="u"
    directory="/work/bb1097/b381585/data/horizontal-correction/"+varname+"/"
    save_path = "/work/bb1097/b383521/slided_window_avrage/3D_spr_old_group/" + varname + "/"
    processes=3
    file_slide = 0
    proc_group = partial(read_spectrum_write_noav, varname, directory, save_path, file_slide)
    st=time.time()
    p=mp.Pool(processes)
    p.map(proc_group,np.arange(0,160))#160))    
    p.close()
    p.join()
    #proc_group(0)
    et=time.time()
    print("Took time "+str(et-st)+ " s.")
    print(get_joke())

def compute_3spr_single_group(varname, file_slide, group):
    """Process a single group for a given variable and file_slide"""
    old_group_dir = Path(f"/work/bb1097/b383521/GW_Data_Summer/3D_spr_old_group/slide_{file_slide}/{varname}/")
    old_group_dir.mkdir(parents=True, exist_ok=True)
    directory = f"/work/bb1097/b383521/GW_Data_Summer/data_2D_spectrum/{varname}/"
    save_path = f"/work/bb1097/b383521/GW_Data_Summer/3D_spr_old_group/slide_{file_slide}/{varname}/"
    read_spectrum_write_noav(varname, directory, save_path, file_slide, group)
    return f"Var: {varname}, Slide: {file_slide}, Group: {group} done"

def compute_3spr_for_variable(varname, file_slide,num_groups=160):
    """Process all groups for one variable at one file_slide using ProcessPoolExecutor"""
    st = time.time()
    
    # Create partial function for this variable and file_slide
    process_func = partial(compute_3spr_single_group, varname, file_slide)
    
    # Use ProcessPoolExecutor for groups (inner parallelization)
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_func, range(160)))
    
    et = time.time()
    print(f"Variable {varname}, slide {file_slide} took {et-st:.2f}s")
    return varname

def run_all_variables_parallel(file_slide,variables=["u", "v", "w", "p", "b"]):
    """Run multiple variables in parallel (outer parallelization)"""
    st = time.time()
    
    # Create partial function for this file_slide
    process_var = partial(compute_3spr_for_variable, file_slide=file_slide)
    
    # Process variables in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_var, variables))
    
    et = time.time()
    print(f"File slide {file_slide} completed in {et-st:.2f}s")
    print(f"Processed variables: {results}")