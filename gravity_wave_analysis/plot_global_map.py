import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from energy_from_u import uv_from_u,Ek_from_u,v_from_u,b_from_u,p_from_u,N_from_u_lev,flux_from_u_lev,Etot_from_u_lev
from timehorsp_distribution import select_triangle, read_averages_uv,read_decomposition
from decomposition import average_averages_length,coriolis_parameter, intrinsic_frequency,vertical_wavenumber_gw,read_averages
from test_average import test_averages_noread
from pyjokes import get_joke
import multiprocessing as mp

def avarage_subset_time_alt(data_av,ti,zi):
    new_dict = {} 
    for key, value in data_av.items():
        new_dict[key]=value[ti,zi,:]
    return new_dict

def avarage_subset1d_tri(data_av,tri):
    new_dict = {} 
    for key, value in data_av.items():
        new_dict[key]=value[tri]
    return new_dict

def read_for_global(datadir,av_dir,path_grid,sign,grmax,zi,flux=True,averages=False,mults=None):
    decomposed=True
    varname="u"
    #Loop over files with group, read the triangles and evaluate the spectrum f (compute Ek or uv) for each triangle
    #Save it to an array of length corresponding to the number of triangles
    print("Starting!")
    #Read first file:
    fname=filename_decomposed(sign,0,decomposed=decomposed)
    ampls,k_all,l_all,omega,maxind,ntri,alt=read_decomposition(datadir,fname)
    #Prepare averages
    #data_av=read_averages_uv(av_dir,"average_","-sphere.nc")
    data_av=read_averages(av_dir,"average_","-sphere.nc") #For computation of b and m or multiplication by rhobar
    wilen=len(omega)
    data_av=average_averages_length(data_av,wilen)
    data_av=avarage_subset_time_alt(data_av,0,zi)
    if not flux: rhobar=data_av['rho']
    #Precompute the coriolis parameter
    lat=xr.open_dataset(path_grid)['clat'].data
    f=coriolis_parameter(lat)
    #Prepare output array
    results=np.zeros(len(ntri)*(grmax))
    #Process the spectra
    print("Processing the spectra.")
    mink,minl=0,0
    maxk,maxl=1,1
    for gri in range(0,grmax):
        fname=filename_decomposed(sign,gri,decomposed=decomposed)
        ampls,k_all,l_all,omega,maxind,ntri,alt=read_decomposition(datadir,fname)
        #FOR TESTING ONLY
        if not decomposed:
            fname2=filename_decomposed(sign,gri,var="w",decomposed=decomposed)
            ampls2,k_all,l_all,omega,maxind,ntri,alt=read_decomposition("/work/bb1097/b381585/data/sp3d-correction/w/",fname2)
            ampls=ampls/mults[varname] 
            ampls2=ampls2/mults["w"]
            #ampls=np.sqrt(ampls*ampls2)
        
        for tri_i in range(0,len(ntri)): #tri_i is index to the triangle ntri[tri_i] in the file ds
            amplssub,k,l=select_triangle(ampls,k_all,l_all,maxind,tri_i,zi)
            if not decomposed:
                amplssub2,k,l=select_triangle(ampls2,k_all,l_all,maxind,tri_i,zi)
            tri_values=ntri[tri_i]

            ##Evaluate the intrinsic frequency
            if not decomposed:
                k,l=k/1000,l/1000 #Needed for computation from nonfiltered spectrum
            freqs={'k':k,'l':l,'omega':omega}
            omega_hat=intrinsic_frequency(freqs,data_av['u'][tri_values],data_av['v'][tri_values])
            ##m=np.sqrt(vertical_wavenumber_gw(k,l,omega_hat,data_av['N2'][tri_i],f[tri_values],data_av['H'][tri_i]))
            ##Apply the function to the spectrum
            data_av1=avarage_subset1d_tri(data_av,tri_values)
            #r=N_from_u_lev(omega_hat,k,l,f[tri_values],data_av1,np.abs(amplssub))
            
            #if flux:
            #    r=uv_from_u(omega_hat,k,l,f[tri_values],np.abs(amplssub))
            #else:
            #    r=Ek_from_u(omega_hat,k,l,f[tri_values],np.abs(amplssub))*rhobar[tri_values]

            #r=np.abs(amplssub)
            #r=np.abs(v_from_u(omega_hat,k,l,f[tri_values],amplssub))
            #r=np.abs(b_from_u(omega_hat,k,l,m,f[tri_values],data_av['N2'][tri_i],data_av['H'][tri_i],data_av['H_pot'][tri_i],data_av['cs2'][tri_i],amplssub))
            #r=np.abs(p_from_u(omega_hat,k,l,f[tri_values],data_av['rho0'][tri_i],amplssub))
            l3d=l[np.newaxis,:,np.newaxis]*np.ones_like(amplssub,dtype=np.float64)
            k3d=k[np.newaxis,np.newaxis,:]*np.ones_like(amplssub,dtype=np.float64)
            #mask=np.logical_not(np.logical_and(l3d==0,k3d==0))
            #mask=np.logical_not(np.logical_or(l3d==0,k3d==0))
            #mask2=np.logical_or(omega_hat>np.abs(f[tri_values]),omega_hat<-np.abs(f[tri_values]))
            mask3=np.logical_not(np.logical_and(k3d==0,l3d==0))
            #mask=np.logical_and(mask,mask2)
            #mask4=(np.ones_like(omega_hat)*omega[:,np.newaxis,np.newaxis]>0)
            #mask=np.logical_and(mask,mask4)
            #amplssub[np.logical_not(mask)]=np.nan
            #mask=omega_hat>0
            omega_hat[np.logical_and(omega_hat<2*np.abs(f[tri_values]),omega_hat>-np.abs(2*f[tri_values]))]=np.nan
            omega_hat[np.logical_or(k3d==0,l3d==0)]=np.nan
            #r=flux_from_u_lev(omega_hat,k,l,f[tri_values],data_av1,np.abs(amplssub),'zx',sign)
            r=Etot_from_u_lev(omega_hat,k,l,f[tri_values],data_av1,np.abs(amplssub),sign)
            #r=np.real(amplssub*amplssub2.conjugate())
            #r=N_from_u_lev(omega_hat,k,l,f[tri_values],data_av1,np.abs(amplssub),sign) #NEW
            
            pars_mult=len(l)*len(k)/wilen#*16 #16 since computation up to 4 Nyquist only... number of nk and nl smaller 4x than nx and ny
            #s=np.nanmin(np.sqrt(l3d[mask]*l3d[mask]+k3d[mask]*k3d[mask]))
            #mask=np.logical_not(l3d==0)
            #s=np.nanmin(np.abs(l3d[mask]))
            #kh=np.sqrt(k3d*k3d+l3d*l3d)
            #s=len(k)*len(l)
            #s=np.sum(np.isfinite(amplssub[mask2]))/len(k)/len(l)/len(omega)
            #s=np.sum(np.logical_not(np.isfinite(amplssub[mask2])))
            #s=np.sum(np.logical_not(mask2))
            #if s>0: print("s is "+str(s))
            assert np.sum(np.logical_and(k3d==0,l3d==0))==len(omega),"Some problem with k and l = 0"
            #assert np.sum(np.logical_not(np.isfinite(amplssub[mask2])))<len(omega_hat), "Too many infinite values."
            #s=np.amax(np.abs(l3d))
            #s=np.nansum(r[mask])
            #mask4=(np.ones_like(omega_hat)*omega[:,np.newaxis,np.newaxis]>0)
            #print("Number of finite values in a spectrum: ",np.sum(np.isfinite(r[mask3])))
            s=np.nansum(r[mask3]*pars_mult)
            #s=f[tri_values]
            results[tri_values]=s
            #if s>120000: 
            #    print("Large value in group "+str(gri)+ ", triangle index  "+str(tri_i)+", latitude "+str(lat[tri_values]/3.14*180)+".")
            del amplssub,k,l,omega_hat,l3d,k3d,freqs#,mask,mask2
        print("Group "+str(gri)+" processed.")
        del ampls,k_all,l_all,omega,maxind,ntri,alt
    return results,data_av

def read_for_global_two(datadir,av_dir,path_grid,grmax,zi,flux=True,averages=False):
    #Loop over files with group, read the triangles and evaluate the spectrum f (compute Ek or uv) for each triangle
    #Save it to an array of length corresponding to the number of triangles
    print("Starting!")
    #Read first file:
    fname=filename_decomposed("minus",0)
    ampls,k_all,l_all,omega,maxind,ntri,alt=read_decomposition(datadir,fname)
    #Prepare averages
    #data_av=read_averages_uv(av_dir,"average_","-sphere.nc")
    data_av=read_averages(av_dir,"average_","-sphere.nc") #For computation of b and m or multiplication by rhobar
    wilen=len(omega)
    data_av=average_averages_length(data_av,wilen)
    data_av=avarage_subset_time_alt(data_av,0,zi)
    if not flux: rhobar=data_av['rho']
    #Precompute the coriolis parameter
    lat=xr.open_dataset(path_grid)['clat'].data
    f=coriolis_parameter(lat)
    #Prepare output array
    results=np.zeros(len(ntri)*(grmax))
    #Process the spectra
    print("Processing the spectra.")
    mink,minl=0,0
    maxk,maxl=1,1
    for gri in range(0,grmax):
        fnamep=filename_decomposed("plus",gri)
        fnamem=filename_decomposed("minus",gri)
        amplsp,k_all,l_all,omega,maxind,ntri,alt=read_decomposition(datadir,fnamep) 
        amplsm,k_all,l_all,omega,maxind,ntri,alt=read_decomposition(datadir,fnamem)   
        for tri_i in range(0,len(ntri)): #tri_i is index to the triangle ntri[tri_i] in the file ds
            amplssubp,k,l=select_triangle(amplsp,k_all,l_all,maxind,tri_i,zi)
            amplssubm,k,l=select_triangle(amplsm,k_all,l_all,maxind,tri_i,zi)
            tri_values=ntri[tri_i]

            ##Evaluate the intrinsic frequency
            #k,l=k*1000,l*1000
            freqs={'k':k,'l':l,'omega':omega}
            omega_hat=intrinsic_frequency(freqs,data_av['u'][tri_values],data_av['v'][tri_values])
            ##m=np.sqrt(vertical_wavenumber_gw(k,l,omega_hat,data_av['N2'][tri_i],f[tri_values],data_av['H'][tri_i]))
            ##Apply the function to the spectrum
            data_av1=avarage_subset1d_tri(data_av,tri_values)

            l3d=l[np.newaxis,:,np.newaxis]*np.ones_like(amplssubp,dtype=np.float64)
            k3d=k[np.newaxis,np.newaxis,:]*np.ones_like(amplssubp,dtype=np.float64)

            #mask=(np.ones_like(omega_hat)*omega[:,np.newaxis,np.newaxis]>0)
            #mask=omega_hat<0
            rp=flux_from_u_lev(omega_hat,k,l,f[tri_values],data_av1,np.abs(amplssubp),'zx',"plus")
            rm=flux_from_u_lev(omega_hat,k,l,f[tri_values],data_av1,np.abs(amplssubm),'zx',"minus")

            assert np.sum(np.logical_and(k3d==0,l3d==0))==len(omega),"Some problem with k and l = 0"
            pars_mult=len(l)*len(k)/wilen
            sp=np.nansum(rp*pars_mult)
            sm=np.nansum(rm*pars_mult)

            results[tri_values]=sp+sm
   
            del amplssubp,amplssubm,k,l,omega_hat,l3d,k3d,freqs#,mask,mask2
        print("Group "+str(gri)+" processed.")
        del amplsp,amplsm,k_all,l_all,omega,maxind,ntri,alt
    return results,data_av


def global_map_flux(path_grid_low,var,figpath,label):
    #Plot the array read in the read_for_global function. Basically the same as the function in test_average.py
    test_averages_noread(path_grid_low,var,figpath,label)
    return

def global_dependence(spectrumampl,averagevar,xlabel,figpath):
    plt.scatter(averagevar[:len(spectrumampl)],spectrumampl,s=1)
    plt.xlabel(xlabel)
    plt.ylabel("Ekh amplitude")
    plt.yscale("log")
    plt.savefig(figpath)
    plt.close()


def global_dependence_angle(spectrumampl,averageu,averagev,figpath):
    phi=np.arctan2(averagev,averageu)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.scatter(phi,np.log10(spectrumampl),s=1)
    ax.set_title(r"$log(E_{kh})(\phi)$")
    plt.savefig(figpath)
    plt.close()


def filename_decomposed(sign,grnr,var="u",decomposed=True):
    if decomposed:
        return "decomposed_"+var+"_gw_"+sign+"_group-"+str(grnr)+".nc"
    else:
        return "sp3D-"+var+"-ny4_window_group-"+str(grnr)+".nc"

def filename_average(var):
    #NOT USED, struncture of the file name for different vars is needed...
    return "average_"+var+"-sphere.nc"

def readalphafile(path_alpha):
    dict={}
    with open(path_alpha) as f:
        mylist = f.read().splitlines() 
        for line in mylist:
            var,mult=line.split(" ")
            dict[var]=float(mult)
    return dict

if __name__ == "__main__":
    #Define paths to the files, filenames defined by functions
    datadir="/work/bb1097/b381585/data/decomposed-correction/"
    #datadir="/work/bb1097/b381585/data/sp3d-correction/u/"
    #datadir="/work/bb1097/b381585/data/sp3d/v/" #For tests with the original spectrum
    plotdir="/home/b/b381585/work1097/images/decomposition/"
    av_dir="/work/bb1097/b381585/data/averaged/"
    path_grid="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    grmax=160#60#160 #Maximum number of group to be evaluated, max 160
    sign="plus" #Select the gravity wave mode: minus or plus.
    zi=0 #Index of vertical level

    #For the plotting of a time spectrum (without the decomposition)
    path_alpha="/work/bb1097/b381585/data/alphas/scales.txt"
    mults=readalphafile(path_alpha)

    res,avvar=read_for_global(datadir,av_dir,path_grid,sign,grmax,zi,flux=False,averages=True,mults=mults)
    
    #res,avvar=read_for_global_two(datadir,av_dir,path_grid,grmax,zi,flux=False,averages=True)
    #figpath=plotdir+"dependence_sizeu-log.png"
    #figpath=plotdir+"map-MFzx-filter-correction_fromuw-testcs2only.png"
    figpath=plotdir+"map-MFzx-decomposed-corrmult-topres-log-fixax.png"
    #global_dependence(res,np.sqrt(avvar['v']**2+avvar['u']**2),"size of velocity [m/s]",figpath)
    #global_dependence_angle(res,avvar['u'],avvar['v'],figpath)
    #figpath=plotdir+"global_distr_Ek_rho-smallcut-dpi300-red.png"
    ##label="total amplitude of u at 15 km"
    #label=r"total amplitude of $\overline{\rho}\left(u^2+v^2\right)$ at 15 km"
    label=r"log(-MFzx) [log(m$^2$s$^{-2})$]"
    #label="N"
    global_map_flux(path_grid,np.log10(-2*res),figpath,label) #Two times res to account for positive and negative m
    print(get_joke("de"))