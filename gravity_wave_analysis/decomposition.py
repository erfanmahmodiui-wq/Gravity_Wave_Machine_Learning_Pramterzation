import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import constants
import time
from timespectrum import save_time_spectra_window_norot
from pathlib import Path


def path_varname(directory, filename_beg, filename_mid,groupnr,varname):
    return directory+varname+"/"+filename_beg+varname+filename_mid+str(groupnr)+".nc"

def read_var_spectrum(directory, filename_beg, filename_mid,groupnr,varname,coordinates=False,path_f=None):
    #Reads file with the vertical spectrum
    #If coordinates is set to True, reads also the k,l,omega
    if path_f==None:
        path_f=path_varname(directory, filename_beg, filename_mid,groupnr,varname)
    ds=xr.open_dataset(path_f)
    
    var_real=ds.ampls_real.data[:,:,:,:,:] #Subset for testing #time,z,triangles,omega,l,k
    var_imag=ds.ampls_imag.data[:,:,:,:,:]
    #Subset here
    #var_real=[]
    #var_imag=[]
    #for i in range(0,maxind.shape[0]):
    #    var_real.append(ds.ampls_real.data[:,:,i,:,:maxind[i,1],:maxind[i,0]])
    #    var_imag.append(ds.ampls_imag.data[:,:,i,:,:maxind[i,1],:maxind[i,0]])
    #var_real=ds.ampls_real.data[:,:,:,:,:maxind[:,1],:maxind[i,0]] #Suggestion of chatGPT, should be tested, but I do not believe it
    
    var=var_real+1j*var_imag #"time","z","ntri","omega","l","k"
    
    if coordinates:
        maxind=ds.max_ind.data[:,:] #ntri,2
        k=ds.k.data[:,:]/1000 #Convert from 1/km to 1/m
        l=ds.l.data[:,:]/1000
        omega=ds.omega.data
        tri=ds.ntri.data[:]
        time=ds.time.data[:]
        return var,maxind,k,l,omega,tri,time
    else:
        return var
    

def read_averages(directory,fn_beg,fn_end):
    varlist=["u","v","N","T","p"] 
    paths=[directory+fn_beg+v+fn_end for v in varlist]
    vars= [xr.open_dataset(path)['var'].data[:, 0, :] for path in paths] #TODO: Remove 0 after testing
    vars1=vars[0:3]
    vars_mid=vars[3:5]
    alt=xr.open_dataset(paths[0]).z.data[0] #TODO: Remove 0 after testing
    cs2= constants.cp/constants.cv*constants.R*vars_mid[0] # cp/cv*RT
    H=constants.R*vars_mid[0]/constants.g # RT/g
    H_pot=constants.cp/constants.R*H # cp/R*H
    rho=vars_mid[1]/vars_mid[0]/constants.R # T/(pR)
    rho0=rho*np.exp(alt/H) # rho_0 e^{z/H}
    vars2=[cs2,H,H_pot,rho,rho0]
    names=["u","v","N2","cs2","H","H_pot","rho","rho0"]
    di=dict(zip(names,vars1+vars2))
    return di

def average_averages_length(data_av, window_length):
    core=np.ones(window_length)/window_length
    new_dict = {} 
    for key, value in data_av.items():
        new_dict[key]=np.apply_along_axis(lambda m: np.convolve(m,core,mode="valid"),axis=0,arr=value)
    return new_dict

def readalphafile(path_alpha):
    dict={}
    with open(path_alpha) as f:
        mylist = f.read().splitlines() 
        for line in mylist:
            var,mult=line.split(" ")
            dict[var]=float(mult)
    return dict

def read_for_decomposition(directory, filename_beg, filename_mid,alphas,groupnr):
    u=read_var_spectrum(directory, filename_beg, filename_mid,groupnr,"u")
    v=read_var_spectrum(directory, filename_beg, filename_mid,groupnr,"v")
    w=read_var_spectrum(directory, filename_beg, filename_mid,groupnr,"w")
    p=read_var_spectrum(directory, filename_beg, filename_mid,groupnr,"p")
    b,maxind,k,l,omega,tri,time =read_var_spectrum(directory, filename_beg, filename_mid,groupnr,"b",True)
    #Multiply by the factors
    data=[u,v,w,p,b]
    data_names=['u','v','w',"p","b"]
    data_di=dict(zip(data_names,data))
    for key in data_di:
        data_di[key]=data_di[key]/alphas[key]
    freqs=[k,l,omega]
    freqs_names=["k",'l','omega']
    freqs_di=dict(zip(freqs_names,freqs))
    loc=[tri,time]
    loc_names=['tri','time']
    loc_di=dict(zip(loc_names,loc))
    return data_di,maxind,freqs_di,loc_di

def average_averages(directory,filename_beg,filename_mid,groupnr,data_av,path_f=None):
    #Do windowed time averages of the averaged data, with the window length given by the omega of the spectrum file
    print("Averaging averages.")
    #Get window length:
    if path_f==None:
        path_f=path_varname(directory, filename_beg, filename_mid,groupnr,'u')
    ds=xr.open_dataset(path_f)
    window_length=len(ds.omega)
    del ds
    #Average the data
    return average_averages_length(data_av, window_length)

def select_one_one_variable(u,maxind,ti,tri_i):
    return u[ti, tri_i,:,:maxind[tri_i,1],:maxind[tri_i,0]]

def select_one_spectrum(data,maxind,freqs,loc,ti,tri_i):
    #Selects one spectrum from all variables to be analysed
    #u,v,w,b,p=data
    selecty=lambda x: select_one_one_variable(x,maxind,ti,tri_i)
    data_out = {} 
    for key, value in data.items():
        data_out[key]=selecty(value)
    k,l,omega=freqs['k'],freqs['l'],freqs['omega']
    freqs_out={"k":k[tri_i,:maxind[tri_i,0]],"l":l[tri_i,:maxind[tri_i,1]],"omega":omega}
    tri,time=loc['tri'],loc['time']
    loc_out={"tri":tri[tri_i],"time":time[ti]}
    return data_out,freqs_out,loc_out

def select_one_averaged(data, ti, tri_i, file_slide):
    new_dict = {} 
    for key, value in data.items():
        new_dict[key]=value[12*file_slide + ti*6, tri_i] #TODO: if you were supposed to not interpolate horizentally then you could change this ti to ti*6 because you shift the window by 1hour or 6 time steps, but you would need to change the window length in the average_averages function to 6*window_length
    return new_dict

def intrinsic_frequency(freqs,u,v):
    #Intrinsic frequency from extrinsic frequency
    #Assumes that freqs=k,l,omega
    k3d=freqs['k'][np.newaxis,np.newaxis,:]
    l3d=freqs['l'][np.newaxis,:,np.newaxis]
    omega=freqs['omega'][:,np.newaxis,np.newaxis]
    return omega-l3d*v-k3d*u

def coriolis_parameter(lat):
    #Compute coriolis parameter from latitude in radians
    Om=7.2921e-5 #2pi/24h in rad/s
    return 2*Om*np.sin(lat)

def vertical_wavenumber_gw_s(k,l,omega_hat,cs2,N2,f,H):
    #Compute vertical wavenumber from dispersion relation for gravity waves and sound
    #Without approximation used in the individual equations
    kh2=k[np.newaxis,np.newaxis,:]**2+l[np.newaxis,:,np.newaxis]**2
    return kh2*(omega_hat**2-N2)/(f**2-omega_hat**2)-1/(4*H**2)+omega_hat**2/cs2

def vertical_wavenumber_gw_full(k,l,omega_hat,cs2,N2,f,H):
    #Compute vertical wavenumber from dispersion relation for gravity waves without any approximation
    kh2=k[np.newaxis,np.newaxis,:]**2+l[np.newaxis,:,np.newaxis]**2
    #First select the modes corresponding to the gravity waves
    t=1/2*(f**2+cs2*(kh2+1/4/H**2))
    p=cs2*(N2*kh2+f**2/4/H**2)
    omax2=t-np.sqrt(t**2-p)
    del t,p
    #omax2=(N2*kh2+f**2/4/H**2)/(kh2+1/4/H**2)
    omega_hat[np.abs(omega_hat)<np.abs(f)]=np.nan
    omega_hat[np.abs(omega_hat)>np.sqrt(omax2)]=np.nan
    #Finally compute the m
    #return vertical_wavenumber_gw(k,l,omega_hat,N2,f,H)
    return vertical_wavenumber_gw_s(k,l,omega_hat,cs2,N2,f,H)

def scalar_product_energy(data1,data2,N2,cs2,rhobar):
    velocity_part=data1['u']*np.conjugate(data2['u'])+data1['v']*np.conjugate(data2['v'])+data1['w']*np.conjugate(data2['w'])
    return 1/2*(velocity_part+data1['p']*np.conjugate(data2['p'])/cs2/(rhobar**2)+data1['b']*np.conjugate(data2['b'])/N2)

def eigenmode_u(k,l,omega_hat,f,m,cs2,N2,H_pot,H):
    kh2=k*k+l*l
    brac1=k*omega_hat+1j*l*f
    brac2=1/H_pot-1/(2*H)+1j*m
    denom=omega_hat*omega_hat-f*f-cs2*kh2
    mult=cs2/N2
    #if (denom==0).any():
    #    print("Will be problem in eigenmode_u (division by zero)")
    return mult*brac1*brac2/denom

def eigenmode_v(k,l,omega_hat,f,m,cs2,N2,H_pot,H):
    kh2=k*k+l*l
    brac1=l*omega_hat-1j*k*f
    brac2=1/H_pot-1/(2*H)+1j*m
    denom=omega_hat*omega_hat-f*f-cs2*kh2
    mult=cs2/N2
    #if (denom==0).any():
    #    print("Will be problem in eigenmode_v (division by zero)")
    return mult*brac1*brac2/denom

def eigenmode_w(omega_hat,N2):
    return 1j*omega_hat/N2

def eigenmode_p(k,l,omega_hat,f,m,cs2,N2,H_pot,H,rho0):
    kh2=k*k+l*l
    denom=omega_hat*omega_hat-f*f-cs2*kh2
    mult=rho0*cs2/N2
    brac1=1/H_pot-1/(2*H)+1j*m
    brac2=omega_hat*omega_hat-f*f
    return mult*brac1*brac2/denom

def eigenmode_b():
    return 1

def eigenmodes(freqs,omega_hat,m,data_av,f): #shape of the modes should be omega, l, k, as is the shape of the result
    k3d=freqs['k'][np.newaxis,np.newaxis,:]
    l3d=freqs['l'][np.newaxis,:,np.newaxis]
    
    um=eigenmode_u(k3d,l3d,omega_hat,f,m,data_av['cs2'],data_av['N2'],data_av['H_pot'],data_av['H'])
    vm=eigenmode_v(k3d,l3d,omega_hat,f,m,data_av['cs2'],data_av['N2'],data_av['H_pot'],data_av['H'])
    wm=eigenmode_w(omega_hat,data_av['N2'])
    pm=eigenmode_p(k3d,l3d,omega_hat,f,m,data_av['cs2'],data_av['N2'],data_av['H_pot'],data_av['H'],data_av['rho0'])
    bm=eigenmode_b()*np.ones_like(omega_hat)
    varlist=[um,vm,wm,pm,bm]
    names=['u','v','w','p','b']
    di=dict(zip(names,varlist))
    return di

def project(data, data_av, modes):
    norm2=scalar_product_energy(modes,modes,data_av["N2"],data_av["cs2"],data_av["rho"])
    prod=scalar_product_energy(data,modes,data_av["N2"],data_av["cs2"],data_av["rho"])
    projected={}
    for key, value in modes.items():
        projected[key]=value*prod/norm2
    return projected

def decompose_one(data,freqs,loc,data_av,lat,thickness):
    outvar="u"
    #Calculate intrinsic frequency
    omega_hat=intrinsic_frequency(freqs,data_av['u'],data_av['v'])
    #Get latitude from the grid => compute f
    f=coriolis_parameter(lat)
    #Calculate vertical wavenumber for nonzero intrinsic frequency and not omega_hat=f
    
    m2_gw=vertical_wavenumber_gw_full(freqs['k'],freqs['l'],omega_hat,data_av['cs2'],data_av["N2"],f,data_av["H"])
    #if (m2_gw<0).any():
    #    print("Will have problem with negative m2_gw")
    m_gw_plus=np.sqrt(np.array(m2_gw))
    m_gw_plus[m_gw_plus>2*3.14159/(2*thickness)]=np.nan
    modes_gw_plus=eigenmodes(freqs,omega_hat,m_gw_plus,data_av,f)  
    projected_gw_plus = project(data, data_av, modes_gw_plus)[outvar] 
    modes_gw_minus=eigenmodes(freqs,omega_hat,-m_gw_plus,data_av,f)
    projected_gw_minus = project(data, data_av, modes_gw_minus)[outvar]
    return projected_gw_plus,projected_gw_minus

def decompose_file(directory, filename_beg, filename_mid,groupnr,data_av,lat,thickness,alphas,dir_out, file_slide):
    data, maxind, freqs0, loc0=read_for_decomposition(directory, filename_beg, filename_mid,alphas,groupnr)
    print(f'shape of data : {data["u"].shape}')
    (nt,ntri,no,nl,nk)=data['u'].shape
    gw_plus=np.zeros_like(data['u'])
    gw_minus=np.zeros_like(data['u'])
    t0=time.time()

    for ti in range(0,nt):
        print(f"anaylizing time : {ti}")
        for tri in range(0,ntri):
            data_av1=select_one_averaged(data_av, ti, loc0['tri'][tri], file_slide=file_slide)
            data1,freqs,loc=select_one_spectrum(data,maxind,freqs0,loc0,ti,tri)
            lat1=lat[loc['tri']]
            gw_plus0,gw_minus0=decompose_one(data1,freqs,loc,data_av1,lat1, thickness[0])
            gw_plus[ti,tri,:,:gw_plus0.shape[-2],:gw_plus0.shape[-1]] =gw_plus0
            gw_minus[ti,tri,:,:gw_minus0.shape[-2],:gw_minus0.shape[-1]] =gw_minus0
    t1=time.time()
    print("Loop took "+str(t1-t0)+" s")
    path=dir_out #Where to save the output
    save_time_spectra_window_norot(path,"gw_plus",  groupnr,  gw_plus,freqs0['k'],freqs0['l'],maxind,freqs0['omega'],loc0['time'],loc0['tri'])
    save_time_spectra_window_norot(path,"gw_minus", groupnr,  gw_minus,freqs0['k'],freqs0['l'],maxind,freqs0['omega'],loc0['time'],loc0['tri'])


def decompose_files_all(directory,filename_beg,filename_mid,av_dir,av_filename_beg,av_filename_end,path_grid,th_path,path_alpha,dir_out, groupnr, file_slide):
    lat=xr.open_dataset(path_grid).clat.data
    data_av=read_averages(av_dir,av_filename_beg,av_filename_end)
    data_av=average_averages(directory,filename_beg,filename_mid,groupnr=groupnr,data_av=data_av)
    thickness=np.fromfile(th_path,sep=",")
    alphas=readalphafile(path_alpha) #Multiplicators for the individual spectra
    #for groupnr in range(4,5):#160):#(0,160):
    print("working at the group "+str(groupnr))
    t0=time.time()
    decompose_file(directory,filename_beg,filename_mid,groupnr,data_av,lat,thickness,alphas,dir_out, file_slide=file_slide)
    t1=time.time()
    print("Group "+str(groupnr)+" took "+str(t1-t0)+" s.")

def decompose(groupnr, file_slide):
    # path to spectrum file

    directory=f"/work/bb1097/b383521/GW_Data_Summer/3D_spr_old_group/slide_{file_slide}/"
    filename_beg = "sp3D-"
    filename_mid="-ny4_window_group-"

    # Path for avarges
    av_dir = "/work/bb1097/b383521/resolved_var/"
    av_filename_beg = "average_"
    av_filename_end = "-sphere_2016.nc"
    #Path with the grid
    path_grid="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    dir_alpha="/work/bb1097/b381585/data/alphas/scales.txt"
    th_path=av_dir+"thickness.txt"

    decomp_dir = Path(f"/work/bb1097/b383521/GW_Data_Summer/decomposition/slide_{file_slide}/")
    decomp_dir.mkdir(parents=True, exist_ok=True)
    dir_out = f"/work/bb1097/b383521/GW_Data_Summer/decomposition/slide_{file_slide}/"
    decompose_files_all(directory,filename_beg,filename_mid,av_dir,av_filename_beg,
                        av_filename_end,path_grid,th_path,dir_alpha,dir_out, groupnr, file_slide=file_slide)

if __name__ == "__main__":
    varname= "u"
    # path to spectrum file
    directory="/work/bb1097/b383521/dummy/regroup_H_interpolation/"
    filename_beg = "sp3D-"
    filename_mid="-ny4_window_group-"

    # Path for avarges
    av_dir = "/work/bb1097/b381585/data/averaged/"
    av_filename_beg = "average_"
    av_filename_end = "-sphere.nc"
    #Path with the grid
    path_grid="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    dir_alpha="/work/bb1097/b381585/data/alphas/scales.txt"
    th_path=av_dir+"thickness.txt"

    dir_out = "/work/bb1097/b383521/dummy/decomposition/"
    decompose_files_all(directory,filename_beg,filename_mid,av_dir,av_filename_beg,av_filename_end,path_grid,th_path,dir_alpha,dir_out)