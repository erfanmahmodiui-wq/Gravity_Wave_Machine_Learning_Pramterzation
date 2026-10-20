import os
import numpy as np
from pyjokes import get_joke
import sys
import gc
import time
import numba
import multiprocessing as mp
import xarray as xr
from functools import partial
from datetime import datetime
from datetime import timedelta
from math import floor
import psutil
from joblib import Parallel, delayed


def fillnovalues(tri_list):
    # Get the size of each large cell (number of smaller cells it contains)
    shapes = [len(tr) for tr in tri_list]
    # Create padded array tl2 with uniform shape: (2048, max_cell_size), filled with -1
    tl2    = np.full((len(tri_list), np.amax(shapes)), -1)
    # Fill tl2 with the indices from tri_list, padding the rest with -1
    for ti in range(len(tri_list)): tl2[ti, 0:shapes[ti]] = tri_list[ti]
    return np.array(tl2), np.array(shapes)

def sortfilenames(filelist):
    datep=[fi.split("_")[-1][:-4] for fi in filelist]
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

def varname_filename(varname):
    if varname=="pres":
        r="p"
    elif varname=="temp":
        r="T"    
    else:
        r=varname
    return r

def listfiles(directory,varname):
    lis=os.listdir(directory)
    #Expecting file name in the form DYAMONDwinter2_atm_selmls_10min_'+varname_filename(varname)+'_ML_20200129T240000Z.nc'
    filebeg='DYAMONDwinter2_atm_selmls_10min_'+varname_filename(varname)+'_ML_'
    onevarfiles=[b for b in lis if b.startswith(filebeg)]
    sf=sortfilenames(onevarfiles)
    return sf

def createctypearrs_int(varlist):
    #Copy variables from varlist to new arrays that can be shared among multiprocessing processes
    result=[]
    for var in varlist:
        X_shape=var.shape
        X = mp.RawArray('i', var.size)
        X_np = np.frombuffer(X, dtype=np.int32).reshape(X_shape)
        np.copyto(X_np, var)
        result.append([X,X_shape])
    return result

@numba.jit(nopython=True, parallel=True)
def average1var_flat_trilist_lin_numba(var,trilist,trilen):
    sh=var.shape
    ntri=len(trilist)
    print(sh)
    result=np.empty((var.shape[0],var.shape[1],ntri))
    for ti in numba.prange(sh[0]):
        for zi in range(sh[1]):
            for i in range(ntri):
                subtri=trilist[i][0:trilen[i]]
                h=0
                for si in subtri:
                    if si==-1: print("Taking wrong value to triangle.")
                    h=h+var[ti,zi,si]
                result[ti,zi,i]=h/len(subtri)
    return result

@numba.jit(nopython=True, parallel=True)
def compute_fluctuations(var, trilist, trilen, mean_val, idx):
    sh = var.shape
    ntri = len(trilist)
    
    # Output has same shape as input
    fluctuation = np.empty((sh[0], sh[1], sh[2]))
    
    for ti in numba.prange(sh[0]):
        for zi in range(sh[1]):
            for i in range(ntri):
                subtri = trilist[i][0:trilen[i]]
                # Compute fluctuations for all small triangles in this big triangle
                for si in subtri:
                    if si == -1:
                        print("Taking wrong output")
                    fluctuation[ti, zi, si] = var[ti, zi, si] - mean_val[idx[ti], zi, i]
    
    return fluctuation

def init_mp_small(tri_list,tri_len,shared_event):
    print("Initializing pool")
    if not 'var_dict' in globals():
        global var_dict
        var_dict = {}
        var_dict['tri_list']    = tri_list[0]
        var_dict['tri_list_sh'] = tri_list[1]
        var_dict['tri_len']    = tri_len[0]
        var_dict['tri_len_sh'] = tri_len[1]
    global event
    print("Pool initialised")
    event = shared_event

def read_and_average(data_dir, varname, fil):
    data_dir = data_dir
    print("Processing file " + fil)
    trilist = np.frombuffer(var_dict['tri_list'], dtype=np.int32).reshape(var_dict['tri_list_sh'])
    trilen = np.frombuffer(var_dict['tri_len'], dtype=np.int32).reshape(var_dict['tri_len_sh'])
    try:
        ds = xr.open_dataset(data_dir + fil)
        time = ds["time"][0:].data
        var = ds[varname][0:, 0:, ...].data
        
        if len(time) == 0:          # ← guard against empty files
            print(f"Skipping {fil}: no time steps")
            return None
            
    except Exception as e:
        print(f"Failed in try in read and average: {e}")
        event.set()
        return None
    
    av1 = average1var_flat_trilist_lin_numba(var, trilist, trilen)
    return av1, time

def read_compute_and_save(data_dir, varname, fil):
    print("Processing file " + fil)
    trilist = np.frombuffer(var_dict['tri_list'], dtype=np.int32).reshape(var_dict['tri_list_sh'])
    trilen = np.frombuffer(var_dict['tri_len'], dtype=np.int32).reshape(var_dict['tri_len_sh'])
    with xr.open_dataset("/work/bb1097/b381585/data/averaged/average_w-sphere.nc") as ds:
        mean_val = ds['var'].values
        time_all = ds.time.data
        ds.close()

    try:
        # Read input
        ds = xr.open_dataset(data_dir + fil)
        var = ds[varname][0:, 0:, ...].data
        print(f"[COMPUTE] Starting fluctuations for shape {var.shape}", flush=True)
        time_file = ds.time.data
        idx = np.where(np.isin(time_all, time_file))[0]
        # Compute fluctuations
        fluc = compute_fluctuations(var, trilist, trilen, mean_val, idx)
        
        # Create output filename
        base_name = fil.rsplit('.', 1)[0]
        extension = fil.rsplit('.', 1)[1] if '.' in fil else 'nc'
        output_fil = f"{base_name}_{varname}_fluc.{extension}"
        
        # Create output dataset preserving all structure
        print(f"[SAVE] Saving to {output_fil}")
        ds_out = xr.Dataset(
            {
                f"{varname}_fluc": (ds[varname].dims, fluc, ds[varname].attrs)
                # This preserves: dimensions names (time, height, ncells)
                #                 dimension order
                #                 variable attributes
            },
            coords=ds.coords,  # Preserves all coordinates (time, height, ncells values)
            attrs=ds.attrs     # Preserves global attributes
        )
        
        # Optionally add metadata about what was computed
        ds_out[f"{varname}_fluc"].attrs['description'] = f'Fluctuations from big triangle means for {varname}'
        ds_out[f"{varname}_fluc"].attrs['computation'] = 'value - mean_of_parent_triangle'
        
        # Save
        ds_out.to_netcdf( "/work/bb1097/DYAMOND_winter_ten_minute_output_fluctuation/w/"+ output_fil)
        ds.close()  # Close input file
        
        print(f"✓ Saved {output_fil}")
        return True
        
    except Exception as e:
        print(f"✗ Failed processing {fil}: {str(e)}")
        event.set()
        return False


def process_single_file(i, file_temp, file_w, directory_fluc):
    ds_w = xr.open_dataset(directory_fluc + "w/" + file_w[i], chunks='auto')
    ds_temp = xr.open_dataset(directory_fluc + "temp/" + file_temp[i], chunks='auto')
    
    T_W_prime = ds_w.w_fluc * ds_temp.temp_fluc
    
    ds_output = ds_w.copy()
    ds_output = ds_output.drop_vars('w_fluc')
    ds_output['Tw'] = T_W_prime
    
    output_filename = file_w[i].replace('_w_', '_Tw_')
    output_path = directory_fluc + "Tw/" + output_filename
    ds_output.to_netcdf(output_path)
    
    ds_w.close()
    ds_temp.close()
    
    return output_filename

def compute_w_T_prime(file_temp, file_w, directory_fluc, n_jobs=2):
    num_file = len(file_temp)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(i, file_temp, file_w, directory_fluc) 
        for i in range(num_file)
    )
    
    for filename in results:
        print(f"Saved: {filename}")

def converttodt(timeicon):
    date=str(floor(timeicon))
    format = '%Y%m%d'
    datetime_str = datetime.strptime(date, format)
    minutes=get_minutes(timeicon)
    converted=datetime_str+timedelta(minutes=float(minutes))
    return converted

def get_minutes(datetimenr):
    part=datetimenr-floor(datetimenr)
    return np.float32(part)*24*60

def timesortorder(timearr):
    dt=[converttodt(onetime) for onetime in timearr]
    so=np.argsort(dt)
    return so

def save_averages(average,varname,time,alt,outpath):
    tri=np.arange(0,average.shape[2])
    ntim=len(time)
    nz=len(alt)
    ds=xr.Dataset(data_vars=dict(
            var=(["time","z","ntri"], average),
        ),
        coords=dict(
            ntri=np.array(tri),
            time=time[0:ntim],
            z=alt[0:nz],
        ),

        attrs=dict(description="Average on large triangles",
                    varname=varname,),
        )
    ds.to_netcdf(outpath)

@numba.jit(nopython=True, parallel=True)
def potential_temp(pres,temp):
    p0=100000 #1000 hPa
    Rcp=0.286 #R/c_p
    return temp*(p0/pres)**Rcp

@numba.jit(nopython=True, parallel=True)
def temp_to_potential_temp(pres,temp):
    p0=100000 #1000 hPa
    Rcp=0.286 #R/c_p
    temp[:]=temp*(p0/pres)**Rcp

def Brunt_Vaisala(theta,z):
    g=9.81
    dertheta=(theta[:,1:,:]-theta[:,:-1,:])/(np.array((z[1:]-z[:-1]))[np.newaxis,:,np.newaxis])
    return g/theta*np.concatenate((dertheta[:,:1,:],dertheta),axis=1) #N^2

def averageN2(theta,z,trilist,trilen):
    if event.is_set(): return 
    thetabar=average1var_flat_trilist_lin_numba(theta,trilist,trilen)
    del theta
    gc.collect()
    if event.is_set(): return
    return Brunt_Vaisala(thetabar,z)

def averageb(theta,z,trilist,trilen):
    if event.is_set(): return 
    thetabar=np.mean(theta,axis=-1)[...,None]
    g=9.81
    res=average1var_flat_trilist_lin_numba(-g*(theta-thetabar)/thetabar,trilist,trilen)
    del thetabar
    gc.collect()
    if event.is_set(): return
    return res


def read_and_compute_Nb(data_dir,alt,varnamep,varnameT,varname_or,fil12):
    print("Starting read and compute N function")
    if event.is_set(): return
    trilist = np.frombuffer(var_dict['tri_list'], dtype=np.int32).reshape(var_dict['tri_list_sh'] )
    trilen = np.frombuffer(var_dict['tri_len'], dtype=np.int32).reshape(var_dict['tri_len_sh'] )
    fil1,fil2=fil12
    process = psutil.Process()
    print(process.memory_info().rss / 1024 ** 2," MiB")
    try:
        #print("In first try")
        dsp=xr.open_dataset(data_dir+fil1)
        dsT=xr.open_dataset(data_dir+fil2)
        #print("Datasets opened")
        time=dsp["time"][0:].data
        if len(time) == 0:
            print(f"Skipping {fil1}/{fil2}: no time steps")
            event.set()  # optional, or just return None
            return None
        if varnamep=="p": varnamep="pres"
        if varnameT=="T": varnameT="temp"
        #print(process.memory_info().rss / 1024 ** 2," MiB")
        varT=dsT[varnameT][0:,0:,...].data
        del dsT
        gc.collect()
        varp=dsp[varnamep][0:,0:,...].data
        del dsp
        gc.collect()
    except:
        print('Something bad happened!')
        event.set()
        return
    try:
        #print("In second try")
        #print(process.memory_info().rss / 1024 ** 2," MiB")
        #numba.set_num_threads(2) #Number of threads by numba parallelisation for each process.
        #varT=potential_temp(varp,varT)
        temp_to_potential_temp(varp,varT)
        #print("Theta calculated with memory ",process.memory_info().rss / 1024 ** 2," MiB")
        del varp
        gc.collect()
        #print("Memory after cleaning ",process.memory_info().rss / 1024 ** 2," MiB")
        if varname_or=="N":
            res=averageN2(varT,alt,trilist,trilen)
        elif varname_or=="b":
            res=averageb(varT,alt,trilist,trilen)
    except:
        print('Something bad happened!')
        event.set()
        return
    print("Before end")
    return res,time

def init_mp_wT(tri_list, tri_len, shared_event, mean_w_path, mean_temp_path):
    init_mp_small(tri_list, tri_len, shared_event)
    global mean_w, mean_temp, time_all
    with xr.open_dataset(mean_w_path) as ds:
        mean_w = ds['var'].values
        time_all = ds.time.data
    with xr.open_dataset(mean_temp_path) as ds:
        mean_temp = ds['var'].values

def read_and_compute_wT(data_dir_w, data_dir_temp, fil12):
    if event.is_set(): return
    trilist = np.frombuffer(var_dict['tri_list'], dtype=np.int32).reshape(var_dict['tri_list_sh'])
    trilen  = np.frombuffer(var_dict['tri_len'], dtype=np.int32).reshape(var_dict['tri_len_sh'])

    fil_w, fil_temp = fil12
    try:
        ds_w = xr.open_dataset(data_dir_w + fil_w)
        ds_t = xr.open_dataset(data_dir_temp + fil_temp)
        time_file = ds_w["time"][0:].data
        if len(time_file) == 0:
            print(f"Skipping {fil_w}/{fil_temp}: no time steps")
            return None
        var_w = ds_w["w"][0:, 0:, ...].data
        var_t = ds_t["temp"][0:, 0:, ...].data
        ds_w.close(); ds_t.close()
    except Exception as e:
        print(f"Failed reading {fil_w}/{fil_temp}: {e}")
        event.set()
        return None

    idx = np.where(np.isin(time_all, time_file))[0]

    fluc_w = compute_fluctuations(var_w, trilist, trilen, mean_w, idx)
    del var_w; gc.collect()
    fluc_t = compute_fluctuations(var_t, trilist, trilen, mean_temp, idx)
    del var_t; gc.collect()

    product = fluc_w * fluc_t
    del fluc_w, fluc_t; gc.collect()

    wT_avg = average1var_flat_trilist_lin_numba(product, trilist, trilen)
    del product; gc.collect()
    return wT_avg, time_file


def compute_avg_var(data_dir, fn_mask, savedir, nproc):
    for varname_or in ["u", "v", "w", "N", "temp", "pres", "b"]:  #["w" , "u", "v", "N", "temp", "pres"]
    #Read triangle mask:
        tri_list=np.load(savedir+fn_mask,allow_pickle=True)
        tri_list2,tri_len=fillnovalues(tri_list)
        del tri_list
        gc.collect() 

        if (varname_or=="N") or (varname_or=="b"):
            varname="p"
            varnameT="T"#None

        else:
            varname=varname_or
            varnameT=None
            #print(f"files {fils}")
        
        fils=listfiles(data_dir,varname)
        if not varnameT==None:
            filsT=listfiles(data_dir,varnameT)
            print(f"files {filsT}")

        print(f"varname : {varname}")
        print(f"files {fils}")
        alt=np.array([15113.594, 13021.524, 11021.524, 9021.524, 7021.524, 5033.731])[0:] #altitudes in the data file
        results=[]
        times=[]
        shared_event = mp.Manager().Event()

        if varnameT==None:
            #Run just averages
            [trilist_mp,trilen_mp]=createctypearrs_int([tri_list2,tri_len])
            del tri_list2,tri_len
            gc.collect()
            partf=partial(read_and_average,data_dir,varname)
            print("Before Pool")
            p=mp.Pool(nproc,initializer=init_mp_small,initargs=(trilist_mp,trilen_mp,shared_event))
            print("Start mapping")
            st=time.time()
            allres=p.map(partf,fils[:])
            en=time.time()
            p.close()
            p.join()
            print("map cycle took "+ str(en-st)+" s")
        
        if (varname_or=="N") or (varname_or=="b"):
            #Run the computation of N
            [trilist_mp,trilen_mp]=createctypearrs_int([tri_list2,tri_len])
            del tri_list2,tri_len
            gc.collect()
            fils12=zip(fils[:],filsT[:])
            #fils12=zip(fils[:nproc],filsT[:nproc])
            partf=partial(read_and_compute_Nb,data_dir,alt,varname,varnameT,varname_or)
            p=mp.Pool(nproc,initializer=init_mp_small,initargs=(trilist_mp,trilen_mp,shared_event))
            st=time.time()
            allres=p.map(partf,fils12)
            en=time.time()
            p.close()
            p.join()
            print("map cycle took "+ str(en-st)+" s")
        
        #Reshape and sort the output and save to file
        allres = [r for r in allres if r is not None]  # ← add this line
        times_sub=[allres[i][1][0] for i in range(0,len(allres))] #First time from every time, extract for sorting
        st=timesortorder(times_sub)

        times=[allres[i][1] for i in st] #Divide allres to times and averages and reorder it by time
        results=[allres[i][0] for i in st]
                
        all_tim=np.concatenate(times)
        all_res=np.concatenate(results,axis=0)

        outpath="/work/bb1097/b383521/resolved_var/"+"average_"+varname_filename(varname_or)+"-sphere_2019.nc"
        save_averages(all_res,varname,all_tim,alt,outpath)
        print(get_joke())


if __name__ == "__main__":

    #### -------- Compute fluctuations --------- ####
    
    # becarfull on compute.... file name
    data_dir='/work/bb1097/DYAMOND_winter_ten_minute_output_2019/'
    fn_mask="triangles_list-sphere.npy"
    savedir="/work/bb1097/b383521/resolved_var/" # know when you are computinging fluctuation or avrage of convective
    
    nproc= 2#10 #3 for N, 10 for only averaging, 2 for b
    #compute_avg_var(data_dir, fn_mask, savedir, nproc)
    
    mean_w_path    = savedir + "average_w-sphere_2019.nc"
    mean_temp_path = savedir + "average_T-sphere_2019.nc"
   

    
    alt=np.array([15113.594, 13021.524, 11021.524, 9021.524, 7021.524, 5033.731])[0:]
    fils_w    = listfiles(data_dir, "w")
    fils_temp = listfiles(data_dir, "temp")
    fils12    = list(zip(fils_w, fils_temp))

    tri_list = np.load(savedir+fn_mask, allow_pickle=True)
    tri_list2, tri_len = fillnovalues(tri_list)
    [trilist_mp, trilen_mp] = createctypearrs_int([tri_list2, tri_len])
    del tri_list, tri_list2, tri_len
    gc.collect()

    shared_event = mp.Manager().Event()
    partf = partial(read_and_compute_wT, data_dir, data_dir)
    p = mp.Pool(nproc, initializer=init_mp_wT,
                initargs=(trilist_mp, trilen_mp, shared_event, mean_w_path, mean_temp_path))
    st = time.time()
    allres = p.map(partf, fils12)
    en = time.time()
    p.close(); p.join()
    print("map cycle took " + str(en-st) + " s")

    allres = [r for r in allres if r is not None]
    times_sub = [allres[i][1][0] for i in range(len(allres))]
    so = timesortorder(times_sub)
    times   = [allres[i][1] for i in so]
    results = [allres[i][0] for i in so]

    all_tim = np.concatenate(times)
    all_res = np.concatenate(results, axis=0)
    save_averages(all_res, "wT", all_tim, alt, savedir + "average_wT-sphere_2019.nc")