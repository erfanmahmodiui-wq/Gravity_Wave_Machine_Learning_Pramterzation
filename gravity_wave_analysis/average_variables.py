import os
import numpy as np
from pyjokes import get_joke
#os.environ['OMP_NUM_THREADS']   =  '4' #Makes it slower
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
from save_triangles import createctypearrs_int
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(os.path.join(dname, "../spec_appx/"))
#sys.path.append(os.path.join(dname, "../spec_appx/src/"))
import inputs.dyamond_functions.subsetdatafunctions_d as dysub # type: ignore


def average1var_flat_trilist_lin(var,trilist,trilen=None):
    sh=var.shape
    ntri=len(trilist)
    print(sh)
    result=np.empty((var.shape[0],var.shape[1],ntri))
    for ti in range(0,sh[0]):
        for zi in range(0,sh[1]):
            for i in range(0,ntri):
                if trilen is not None:
                    subtri=trilist[i][0:trilen[i]]
                else:
                    subtri=trilist[i][trilist[i]>=0]
                result[ti,zi,i]=np.average(var[ti,zi,subtri])
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

def fillnovalues(tri_list):
    shapes=[len(tr) for tr in tri_list]
    tl2=np.full((len(tri_list),np.amax(shapes)),-1)
    for ti in range(len(tri_list)): tl2[ti,0:shapes[ti]]=tri_list[ti]
    return np.array(tl2),np.array(shapes)

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


def read_and_average(data_dir,varname,fil):
    #print("read and average function started")
    print("Processing file "+fil)
    trilist = np.frombuffer(var_dict['tri_list'], dtype=np.int32).reshape(var_dict['tri_list_sh'] )
    trilen = np.frombuffer(var_dict['tri_len'], dtype=np.int32).reshape(var_dict['tri_len_sh'] )
    try:
        ds=xr.open_dataset(data_dir+fil)
        time=ds["time"][0:].data
        var=ds[varname][0:,0:,...].data
    except:
        print("Failed in try in read and average")
        event.set()
        return
    av1=average1var_flat_trilist_lin_numba(var,trilist,trilen)
    return av1,time

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

def get_minutes(datetimenr):
    part=datetimenr-floor(datetimenr)
    return np.float32(part)*24*60

def converttodt(timeicon):
    date=str(floor(timeicon))
    format = '%Y%m%d'
    datetime_str = datetime.strptime(date, format)
    minutes=get_minutes(timeicon)
    converted=datetime_str+timedelta(minutes=minutes)
    return converted

def timesortorder(timearr):
    dt=[converttodt(onetime) for onetime in timearr]
    so=np.argsort(dt)
    return so

if __name__ == "__main__":
    data_dir='/work/bb1097/DYAMOND_winter_ten_minute_output_2019/'
    fn_mask="triangles_list-sphere"
    savedir="/work/bb1097/b383521/resolved_var/"
    nproc= 2#10 #3 for N, 10 for only averaging, 2 for b

    #Read triangle mask:
    tri_list=np.load(savedir+fn_mask+".npy",allow_pickle=True)
    tri_list2,tri_len=fillnovalues(tri_list)
    del tri_list
    gc.collect()

    varname_or="u" #u, v, N, temp, pres,w   

    if (varname_or=="N") or (varname_or=="b"):
        varname="p"
        varnameT="T"#None
    else:
        varname=varname_or
        varnameT=None

    fils=dysub.listfiles(data_dir,varname)
    if not varnameT==None:
        filsT=dysub.listfiles(data_dir,varnameT)

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
    times_sub=[allres[i][1][0] for i in range(0,len(allres))] #First time from every time, extract for sorting
    st=timesortorder(times_sub)

    times=[allres[i][1] for i in st] #Divide allres to times and averages and reorder it by time
    results=[allres[i][0] for i in st]
    
    all_tim=np.concatenate(times)
    all_res=np.concatenate(results,axis=0)

    outpath=savedir+"average_"+dysub.varname_filename(varname_or)+"-sphere_2019.nc"
    save_averages(all_res,varname,all_tim,alt,outpath)
    print(get_joke())