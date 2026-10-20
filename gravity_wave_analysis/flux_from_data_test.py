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

def flux_from_data_temporal():
    data_dir='/work/bb1097/DYAMOND_winter_ten_minute_output/'
    fn_mask="triangles_list-sphere"
    
    savedir="/work/bb1097/b381585/data/averaged/"

    #figpath="/work/bb1097/b381585/images/averages/"+"testuw-data-sum.png"
    
    
    av_dir="/work/bb1097/b381585/data/averaged/"
    av_filename_beg="average_"
    av_filename_end="-sphere.nc"

    #ti=10
    #zi=0
    ti=0
    zi=0

    #Read triangle mask:
    tri_list=np.load(savedir+fn_mask+".npy",allow_pickle=True)
    tri_list2,tri_len=av.fillnovalues(tri_list)
    del tri_list

    #Read u and w from files
    filsu=dysub.listfiles(data_dir,"u")
    #filsw=dysub.listfiles(data_dir,"w")
    totsum=0

    #Read average variables
    avu=xr.open_dataset(av_dir+av_filename_beg+"u"+av_filename_end)['var'][:,zi,...].data
    #avw=xr.open_dataset(av_dir+av_filename_beg+"w"+av_filename_end)['var'][ti,zi,...].data
    #Create field for results
    #result=np.empty_like(avu)
    result=np.zeros(avu.shape[-1]) #Triangles are at the last dimension
    
    for fil in filsu[0:12]:
        
        varu=xr.open_dataset(data_dir+fil)['u'][:,zi,...].data 
        #varw=xr.open_dataset(data_dir+filsw[0])["w"][ti,zi,...].data

        #Average flux on the triangles
        for tri in range(0,avu.shape[-1]):#[1]):
            subtri=tri_list2[tri][0:tri_len[tri]]
            #result[tri]=np.sum((varu[subtri]-avu[tri])*(varu[subtri]-avu[tri])) #s1=np.sum((varu[:,subtri]-avu[ti:ti+subtri.shape[0],tri])**2,axis=0)
            r=0
            for ti1 in range(ti,ti+varu.shape[0]):
                r=r+np.sum((varu[ti1-ti,subtri]-avu[ti1,tri])**2)
            result[tri]=r
            #if tri==5000:
            #    print("Number of components: ",varu[ti,...][subtri].shape)
        #totsum=totsum+result[5000]
        ti=ti+varu.shape[0]
        print("File "+fil+" finished with sum "+str(result[5000]))
    print("Number of time indices is ",str(ti+1))
    #Plot the averages:
    #print("Sum for triangle for all times 5000 is "+str(totsum)+" m^2s^-2.")
    #grid=xr.open_dataset(fn_grid_low)
    print("End.")
    return result

def flux_from_data_one(ti=None):
    data_dir='/work/bb1097/DYAMOND_winter_ten_minute_output/'
    fn_mask="triangles_list-sphere"
    
    savedir="/work/bb1097/b381585/data/averaged/"

    #figpath="/work/bb1097/b381585/images/averages/"+"testuw-data-sum.png"
    
    
    av_dir="/work/bb1097/b381585/data/averaged/"
    av_filename_beg="average_"
    av_filename_end="-sphere.nc"

    #ti=10
    #zi=0
    if ti==None:
        ti=10 #Have to be inside the first file
    zi=0

    #Read triangle mask:
    tri_list=np.load(savedir+fn_mask+".npy",allow_pickle=True)
    tri_list2,tri_len=av.fillnovalues(tri_list)
    del tri_list

    #Read u and w from files
    filsu=dysub.listfiles(data_dir,"w")
    #filsw=dysub.listfiles(data_dir,"w")
    totsum=0

    #Read average variables
    avu=xr.open_dataset(av_dir+av_filename_beg+"w"+av_filename_end)['var'][:,zi,...].data
    #avw=xr.open_dataset(av_dir+av_filename_beg+"w"+av_filename_end)['var'][ti,zi,...].data
    #Create field for results
    #result=np.empty_like(avu)
    result=np.zeros(avu.shape[-1]) #Triangles are at the last dimension
    fil=filsu[0] #Read only the first file
    
        
    varu=xr.open_dataset(data_dir+fil)['w'][:,zi,...].data 
    #varw=xr.open_dataset(data_dir+filsw[0])["w"][ti,zi,...].data

    #Average flux on the triangles
    for tri in range(0,avu.shape[-1]):#[1]):
        subtri=tri_list2[tri][0:tri_len[tri]]
        #result[tri]=np.sum((varu[subtri]-avu[tri])*(varu[subtri]-avu[tri])) #s1=np.sum((varu[:,subtri]-avu[ti:ti+subtri.shape[0],tri])**2,axis=0)
  
        result[tri]=np.sum((varu[ti,subtri]-avu[ti,tri])**2) #np.sum((varu[ti,subtri]/len(subtri)))**2
        #if tri==5000:
        #    print("Number of components: ",varu[ti,...][subtri].shape)
    #totsum=totsum+result[5000]
    print("File "+fil+" finished with sum "+str(result[5000]))
    print("Number of time indices is ",str(ti+1))
    #Plot the averages:
    #print("Sum for triangle for all times 5000 is "+str(totsum)+" m^2s^-2.")
    #grid=xr.open_dataset(fn_grid_low)
    print("End.")
    return result

if __name__ == "__main__":
    figpath="/work/bb1097/b381585/images/horizontal-tests/"+"testuu-data-sum-last.png"
    fn_grid_low="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    label=r"$u'u'$ [m$^2$ s$^{-2}$]"
    result=flux_from_data_temporal()
    ta.test_averages_noread(fn_grid_low,result,figpath,label) #Approximately *144 to have all days


