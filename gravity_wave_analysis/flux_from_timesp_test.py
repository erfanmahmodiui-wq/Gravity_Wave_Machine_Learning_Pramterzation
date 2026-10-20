import numpy as np
from pyjokes import get_joke
import average_variables as av
import xarray as xr
import os,sys
import test_average as ta
from decomposition import readalphafile
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(os.path.join(dname, "../spec_appx/"))
#sys.path.append(os.path.join(dname, "../spec_appx/src/"))
import inputs.dyamond_functions.subsetdatafunctions_d as dysub # type: ignore

def timespfname(group_id,varname):
    #return "horsp-u-ny4_group-"+str(group_id)+"_20200123T000000Z-orig-sett.nc" #Original settings
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-allkl.nc"
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-lam2.nc" #Double lambda
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-lamhalf.nc" #Half lambda
    return "sp3D-"+varname+"-ny4_window_group-"+str(group_id)+"-double-lam.nc" #Original 3D spectrum

if __name__ == "__main__":
    varname="u"
    #data_dir='/work/bb1097/b381585/data/final/u/'
    data_dir_u='/work/bb1097/b381585/data/sp3d-correction/'+varname+"/"
    data_dir_w='/work/bb1097/b381585/data/sp3d-correction/'+"w"+"/"
    #data_dir='/work/bb1097/b381585/data/horizontal-tests/'+varname+'/'
    fn_mask="triangles_list-sphere"

    path_alpha="/work/bb1097/b381585/data/alphas/scales.txt"
    
    
    savedir="/work/bb1097/b381585/data/averaged/"

    #figpath="/work/bb1097/b381585/images/averages/"+"testuu-horsp-sum-nokl0.png"
    figpath="/work/bb1097/b381585/images/averages/"+"testuw-tempsp-sum.png"
    label=varname+r"$'w'^2$ [m^2 s$^{-2}$]"
    
    av_dir="/work/bb1097/b381585/data/averaged/"
    av_filename_beg="average_"
    av_filename_end="-sphere.nc"

    fn_grid_low="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"

    max_gr=160
    ntri=128
    ti=10
    zi=0
    #ti=0
    #zi=0
    #max_gr=1
    ntr=0 
    ortr=5000

    #Read rescaling factor (To be used with the temporal spectrum only, not the filtered fields
    alphas=readalphafile(path_alpha)

    #Read triangle mask:
    tri_list=np.load(savedir+fn_mask+".npy",allow_pickle=True)
    tri_list2,tri_len=av.fillnovalues(tri_list)
    del tri_list
    tri_list_sel=tri_list2[ortr,:]
    del tri_list2

    #Create field for results
    #filu=horspfname(0)
    filu=timespfname(ortr,varname)
    tv=xr.open_dataset(data_dir_u+filu)["ampls_real"]
    nomega=tv.shape[3]
    nl=tv.shape[4]
    nk=tv.shape[5]
    result=np.zeros((max_gr*ntri))
    #result=0

    for gri in range(0,max_gr):
        #Read u and w from files
        filu=timespfname(ortr,varname)
        ds=xr.open_dataset(data_dir_u+filu)
        varu=ds["ampls_real"]+ds["ampls_imag"]*1j #time, z, ntri,l,k
        varu=varu[ti,zi,ntr,...].data 
        k=ds['k'].data #ntri,nk
        l=ds['l'].data #ntri,nl

        maxind=ds["max_ind"].data #ntri,dims
        #varw=xr.open_dataset(data_dir+filsw[0])["v"][0,0,...].data
        filw=timespfname(ortr,"w")
        ds=xr.open_dataset(data_dir_w+filw)
        varw=ds["ampls_real"]+ds["ampls_imag"]*1j #time, z, ntri,l,k
        varw=varu[ti,zi,ntr,...].data 

        #Read average variables
        #avu=xr.open_dataset(av_dir+av_filename_beg+"u"+av_filename_end)['var'][10,0,...].data
        #avw=xr.open_dataset(av_dir+av_filename_beg+"v"+av_filename_end)['var'][0,0,...].data

        #Average flux on the triangles
        tri=0
        for tri in range(0,ntri):
            [maxik,maxil]=maxind[tri,:]
            k1=k[tri,:maxik]
            l1=l[tri,:maxil]
            kk,ll=np.meshgrid(k1,l1)
            #mask=np.logical_not(np.logical_and(kk==0,ll==0))
            #mask2=np.repeat(mask[np.newaxis,...],repeats=nomega,axis=0)
            #result=np.sum(np.absolute(varu[:,:maxil,:maxik])**2)#[mask2]
            #print("Number of components: ",varu[tri,:maxil,:maxik].shape)
            result[gri*varu.shape[0]+tri]=np.sum(np.absolute(varu[tri,:maxil,:maxik]*np.absolute(varw[tri,:maxil,:maxik])))#+(varw[subtri]-avw[tri])**2)
    
    #Plot the averages:
    pars_mult=nk*nl/nomega
    #print("Sum for triangle 5000 is "+str(result*pars_mult)+" ms^-2.")
    #print("Sum for triangle 5000 times 16 is "+str(result*pars_mult*16)+" ms^-2.")
    ta.test_averages_noread(fn_grid_low,result*pars_mult,figpath,label) #Approximately *144 to have all days
    

