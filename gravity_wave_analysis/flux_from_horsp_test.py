import numpy as np
from pyjokes import get_joke
import average_variables as av
import xarray as xr
import os,sys
import test_average as ta
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(os.path.join(dname, "../spec_appx/"))
#sys.path.append(os.path.join(dname, "../spec_appx/src/"))
import inputs.dyamond_functions.subsetdatafunctions_d as dysub # type: ignore

def horspfname(group_id):
    #return "horsp-u-ny4_group-"+str(group_id)+"_20200123T000000Z-orig-sett.nc" #Original settings
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-allkl.nc"
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-lam2.nc" #Double lambda
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-lamhalf.nc" #Half lambda
    #return "horsp-u-ny4_group-"+str(group_id)+"_20200123T000000Z-taperhalf.nc" #Half taper
    return "horsp-u-ny4_group-"+str(group_id)+"_20200123T000000Z-double-lam.nc" #Half taper

def horspfname_date(group_id,date,varname):
    #return "horsp-u-ny4_group-"+str(group_id)+"_20200123T000000Z-orig-sett.nc" #Original settings
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-allkl.nc"
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-lam2.nc" #Double lambda
    #return "horsp-u-ny1_group-"+str(group_id)+"_20200123T000000Z-lamhalf.nc" #Half lambda
    #return "horsp-u-ny4_group-"+str(group_id)+"_"+date+"-orig-sett2.nc" 
    #return "horsp-u-ny4_group-"+str(group_id)+"_"+date+"-double-lam.nc" 
    return "horsp-"+varname+"-ny4_group-"+str(group_id)+"_"+date+".nc" 


def horizontal_sp_flux(ti=None,varname="u"):
    data_dir='/work/bb1097/b381585/data/horizontal-correction/'+varname+'/'
    #data_dir='/work/bb1097/b381585/data/horizontal-tests/u/'
    fn_mask="triangles_list-sphere"
    
    savedir="/work/bb1097/b381585/data/averaged/"

    #figpath="/work/bb1097/b381585/images/averages/"+"testuu-horsp-sum-nokl0.png"
    label=r"$u'^2$ [m^2 s$^{-2}$]"
    
    av_dir="/work/bb1097/b381585/data/averaged/"
    av_filename_beg="average_"
    av_filename_end="-sphere.nc"

    fn_grid_low="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"

    max_gr=160
    if ti==None:
        ti=10#10
    zi=0
    #ti=0
    #zi=0
    #max_gr=1
    ntr=0 

    dates=["20200123T000000Z","20200123T020000Z","20200123T040000Z","20200123T060000Z","20200123T080000Z","20200123T100000Z","20200123T120000Z",
           "20200123T140000Z","20200123T160000Z","20200123T180000Z","20200123T200000Z","20200123T220000Z"]
    #dates=["20200126T060000Z","20200126T060000Z"]

    #Read triangle mask:
    tri_list=np.load(savedir+fn_mask+".npy",allow_pickle=True)
    tri_list2,tri_len=av.fillnovalues(tri_list)
    del tri_list

    #Create field for results
    #filu=horspfname(0)
    filu=horspfname_date(ntr,dates[0],varname)
    tv=xr.open_dataset(data_dir+filu)["ampls_real"]
    ntri=tv.shape[2]
    nl=tv.shape[3]
    nk=tv.shape[4]
    result=np.zeros((max_gr*ntri))
    totsum=0
    for gri in range(0,max_gr):
        #for gri in range(ntr,ntr+1):
        #Read u and w from files
        filu=horspfname_date(gri,dates[0],varname)
        ds=xr.open_dataset(data_dir+filu)
        varu=ds["ampls_real"]+ds["ampls_imag"]*1j #time, z, ntri,l,k
        varu=varu[ti,zi,...].data 
        k=ds['k'].data #ntri,nk
        l=ds['l'].data #ntri,nl

        maxind=ds["max_ind"].data #ntri,dims
        #varw=xr.open_dataset(data_dir+filsw[0])["v"][0,0,...].data

        #Read average variables
        #avu=xr.open_dataset(av_dir+av_filename_beg+"u"+av_filename_end)['var'][10,0,...].data
        #avw=xr.open_dataset(av_dir+av_filename_beg+"v"+av_filename_end)['var'][0,0,...].data

        #Average flux on the triangles
        for tri in range(0,varu.shape[0]):
            [maxik,maxil]=maxind[tri,:]
            k1=k[tri,:maxik]
            l1=l[tri,:maxil]
            kk,ll=np.meshgrid(k1,l1)
            mask=np.logical_not(np.logical_and(kk==0,ll==0))
            #result[gri*ntri+tri]=np.sum(np.absolute(varu[tri,:maxil,:maxik])**2)
            #print("Number of components: ",varu[:,tri,:maxil,:maxik].shape)
            result[gri*varu.shape[0]+tri]=np.sum(np.absolute(varu[tri,:maxil,:maxik][mask])**2)#+(varw[subtri]-avw[tri])**2)
    #totsum=totsum+result[0]
    
    pars_mult=nk*nl
    return result*pars_mult

if __name__ == "__main__":
    res=horizontal_sp_flux()
    print("Sum for triangle 5000 is "+str(res)+" ms^-2.")
    print("Sum for triangle 5000 times 16 is "+str(res*16)+" ms^-2.")
    #ta.test_averages_noread(fn_grid_low,result*pars_mult*144,figpath,label) #Approximately *144 to have all days
    

