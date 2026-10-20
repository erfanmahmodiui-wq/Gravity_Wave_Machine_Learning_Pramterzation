import numpy as np
from pyjokes import get_joke
import average_variables as av
import xarray as xr
import os,sys
import test_average as ta
from functools import partial
import numba as nb
from scipy.optimize import minimize
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(os.path.join(dname, "../spec_appx/"))
#sys.path.append(os.path.join(dname, "../spec_appx/src/"))
import inputs.dyamond_functions.subsetdatafunctions_d as dysub # type: ignore


def plane_gpt(x, y, params):
    a, b, c = params
    return a * x + b * y + c

def error_gpt(params, x, y, z):
    plane_z = plane_gpt(x, y, params)
    return np.sum((plane_z - z) ** 2)

def interpolateplane_2Dfield_gpt(x,y, data2D):
    z=data2D
    fun = partial(error_gpt, x=x, y=y, z=z)
    params0 = [0, 0, np.mean(z)]
    res = minimize(fun, params0,method="COBYLA")
    return plane_gpt(x, y, res.x)

def plane_fixav(x, y, params):
    a, b = params
    return a * x + b * y 

def error_fixav(params, x, y, z):
    plane_z = plane_fixav(x, y, params)
    return np.sum((plane_z - z) ** 2)

def interpolateplane_2Dfield_fixav(x,y, data2D):
    z=data2D
    fun = partial(error_fixav, x=x, y=y, z=z)
    params0 = [0, 0]
    res = minimize(fun, params0)
    return plane_fixav(x, y, res.x)


def window_tukey_1d(x,xmin,xmax,alpha):
    window=np.ones(np.shape(x))
    maskmin=np.where(x<=xmin+alpha*(xmax-xmin)/2)
    maskmax=np.where(x>=xmax-alpha*(xmax-xmin)/2)
    window[maskmin]=1/2*(1-np.cos(2*3.14159/alpha*(x[maskmin]-xmin)/(xmax-xmin)))
    window[maskmax]=1/2*(1-np.cos(2*3.14159/alpha*(x[maskmax]-xmin)/(xmax-xmin)))
    return window

def taper_tukey(data,x,y,alpha):
    xwindow=window_tukey_1d(x,x.min(),x.max(),alpha)
    ywindow=window_tukey_1d(y,y.min(),y.max(),alpha)
    return data*xwindow*ywindow

def distance_line_points(x1,y1,x2,y2,x,y):
    s=np.sqrt((x1-x2)**2+(y1-y2)**2)
    c=np.abs((x2-x1)*(y1-y)-(x1-x)*(y2-y1))
    return c/s

def taper_tukey_tri(data,x,y,vort_x,vort_y,alpha):
    y1,y2,y3=vort_y[:]
    x1,x2,x3=vort_x[:]
    da=distance_line_points(x1,y1,x2,y2,x,y)
    awindow=window_tukey_1d(da,da.min(),da.max(),alpha)
    db=distance_line_points(x2,y2,x3,y3,x,y)
    bwindow=window_tukey_1d(db,db.min(),db.max(),alpha)
    dc=distance_line_points(x3,y3,x1,y1,x,y)
    cwindow=window_tukey_1d(dc,dc.min(),dc.max(),alpha)
    taper2D=awindow*bwindow*cwindow
    if data.ndim==2:
        return data*taper2D
    if data.ndim==3:
        return data*taper2D[None,...]
    return data*taper2D[None,None,...]

@nb.jit(nopython=True)
def temp_to_potential_temp(pres,temp):
    p0=100000 #1000 hPa
    Rcp=0.286 #R/c_p
    temp[:]=temp*(p0/pres)**Rcp

@nb.jit(nopython=True)
def all_pottemp_to_b(T,thetabar):
    g=9.81
    T[...]=-g*(T-thetabar)/thetabar

def compute_buoyancy_global(p,T):
    #Compute the buoyancy from pressure and temperature by -g theta'/theta_bar, where theta_bar is average over the level
    temp_to_potential_temp(p,T)
    thetabar=np.mean(T,axis=-1)[...,None]
    all_pottemp_to_b(T,thetabar)
    return T

def flux_from_data_deplane(ti=None,var="u"):
    #var also p,b
    data_dir='/work/bb1097/DYAMOND_winter_ten_minute_output/'
    fn_mask="triangles_list-sphere"
    
    savedir="/work/bb1097/b381585/data/averaged/"

    #figpath="/work/bb1097/b381585/images/averages/"+"testuw-data-sum.png"
    
    
    av_dir="/work/bb1097/b381585/data/averaged/"
    av_filename_beg="average_"
    av_filename_end="-sphere.nc"

    
    fn_coord_high="/work/bb1097/b381585/data/averaged/coordinates_projection.nc"
    #ti=10
    #zi=0
    if ti==None:
        ti=0#10
    zi=0
    #trii=5000

    #Read triangle mask:
    tri_list=np.load(savedir+fn_mask+".npy",allow_pickle=True)
    tri_list2,tri_len=av.fillnovalues(tri_list)
    del tri_list

    #Read the coordinates:
    coordf=xr.open_dataset(fn_coord_high)
    x_all=coordf.coord_x.data
    y_all=coordf.coord_y.data

    #Read coordinates of vertices for taper
    fn_grid_low="/work/bb1097/b381585/data/averaged/projection-vertices.nc"
    low_file=xr.open_dataset(fn_grid_low)
    vert_x_all=low_file.coord_x.data
    vert_y_all=low_file.coord_y.data


    #Read u and w from files
    if var=="b":
        fils_p=dysub.listfiles(data_dir,"pres")
        fils_t=dysub.listfiles(data_dir,"temp")
    else:
        fils_p=dysub.listfiles(data_dir,var)
    #filsw=dysub.listfiles(data_dir,"w")
    totsum=0
    if var=='p':
        varr="pres"
    else:
        varr=var
    for i,fil in enumerate(fils_p[:1]):# filsu[40:41]:#filsu[:12]:
        
        if var=="b":
            #Compute b from p and t
            var_p=xr.open_dataset(data_dir+fil)["pres"][:,zi,...].data 
            var_t=xr.open_dataset(data_dir+fils_t[i])["temp"][:,zi,...].data
            var_p=compute_buoyancy_global(var_p,var_t)
        else: 
            var_p=xr.open_dataset(data_dir+fil)[varr][:,zi,...].data 
        #varw=xr.open_dataset(data_dir+filsw[0])["w"][ti,zi,...].data

        #Read average variables
        avu=xr.open_dataset(av_dir+av_filename_beg+var+av_filename_end)['var'][:,zi,...].data
        #avw=xr.open_dataset(av_dir+av_filename_beg+"w"+av_filename_end)['var'][ti,zi,...].data
        #Create field for results
        #result=np.empty_like(avu)
        result=np.empty(avu.shape[1])
        #result=np.zeros(1)

        #Average flux on the triangles
        for tri in range(0,avu.shape[1]):# range(trii,trii+avu.shape[1]):
            subtri=tri_list2[tri][0:tri_len[tri]]
            #result[tri]=np.sum((varu[subtri]-avu[tri])*(varw[subtri]-avw[tri])) #s1=np.sum((varu[:,subtri]-avu[ti:ti+subtri.shape[0],tri])**2,axis=0)
            coord_x=x_all[subtri]
            coord_y=y_all[subtri]
            vert_x=vert_x_all[tri,:]
            vert_y=vert_y_all[tri,:]
            r=0
            for ti1 in range(ti,ti+1):#varu.shape[0]):
                subvar=var_p[ti1-ti,subtri]
                avv=np.average(subvar)
                #dat=taper_tukey_tri(subvar-avv,coord_x,coord_y,vert_x,vert_y,0.2)
                plane=interpolateplane_2Dfield_gpt(coord_x,coord_y,subvar-avv)+avv
                #Test also tapering:
                dat2=taper_tukey_tri(subvar-plane,coord_x,coord_y,vert_x,vert_y,0.2)
                #dat=dat-np.average(dat)
                #v2=subvar-avv-plane
                #v2=taper_tukey_tri(v,coord_x,coord_y,vert_x,vert_y,0.2)
                plane2=interpolateplane_2Dfield_gpt(coord_x,coord_y,dat2)
                r=r+np.sum((dat2-plane2)**2)
                #r=r+np.sum(v**2)
                #r=r+np.average((plane)**2)
                del plane,subvar
                #del subvar,plane,dat
                #r=r+np.sum((varu[ti1-ti,subtri]-avu[ti1,tri])**2)
                ##r=r+np.sum((varu[ti1-ti,subtri]-np.mean(avu[ti1,tri]))**2) #remove average
            result[tri]=r
            #if tri==5000:
            #    print("Number of components: ",varu[ti,...][subtri].shape)
        totsum=totsum+result[0]
        ti=ti+var_p.shape[0]
        print("File "+fil+" finished with sum "+str(result[0]))
    #print("Sum for triangle for all times 5000 is "+str(totsum)+" ms^-2.")    
    return result

def flux_from_data_deplane_temporal():
    data_dir='/work/bb1097/DYAMOND_winter_ten_minute_output/'
    fn_mask="triangles_list-sphere"
    
    savedir="/work/bb1097/b381585/data/averaged/"

    #figpath="/work/bb1097/b381585/images/averages/"+"testuw-data-sum.png"
    
    fn_coord_high="/work/bb1097/b381585/data/averaged/coordinates_projection.nc"

    av_dir="/work/bb1097/b381585/data/averaged/"
    av_filename_beg="average_"
    av_filename_end="-sphere.nc"

    ti=0
    zi=0

    #Read triangle mask:
    tri_list=np.load(savedir+fn_mask+".npy",allow_pickle=True)
    tri_list2,tri_len=av.fillnovalues(tri_list)
    del tri_list

    #Read the coordinates:
    coordf=xr.open_dataset(fn_coord_high)
    x_all=coordf.coord_x.data
    y_all=coordf.coord_y.data

    #Read coordinates of vertices for taper
    fn_grid_low="/work/bb1097/b381585/data/averaged/projection-vertices.nc"
    low_file=xr.open_dataset(fn_grid_low)
    vert_x_all=low_file.coord_x.data
    vert_y_all=low_file.coord_y.data


    #Read u and w from files
    filsu=dysub.listfiles(data_dir,"u")
    #filsw=dysub.listfiles(data_dir,"w")

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
            if tri==20000:
                print("Starting with triangle 20000")
            subtri=tri_list2[tri][0:tri_len[tri]]
            #result[tri]=np.sum((varu[subtri]-avu[tri])*(varu[subtri]-avu[tri])) #s1=np.sum((varu[:,subtri]-avu[ti:ti+subtri.shape[0],tri])**2,axis=0)
            subtri=tri_list2[tri][0:tri_len[tri]]
            #result[tri]=np.sum((varu[subtri]-avu[tri])*(varw[subtri]-avw[tri])) #s1=np.sum((varu[:,subtri]-avu[ti:ti+subtri.shape[0],tri])**2,axis=0)
            coord_x=x_all[subtri]
            coord_y=y_all[subtri]
            vert_x=vert_x_all[tri,:]
            vert_y=vert_y_all[tri,:]
            r=0
            for ti1 in range(0,varu.shape[0]):
                subvar=varu[ti1,subtri]
                avv=np.average(subvar)
                #dat=taper_tukey_tri(subvar-avv,coord_x,coord_y,vert_x,vert_y,0.2)
                plane=interpolateplane_2Dfield_gpt(coord_x,coord_y,subvar-avv)+avv
                #Test also tapering:
                dat2=taper_tukey_tri(subvar-plane,coord_x,coord_y,vert_x,vert_y,0.2)
                #dat=dat-np.average(dat)
                #v2=subvar-avv-plane
                #v2=taper_tukey_tri(v,coord_x,coord_y,vert_x,vert_y,0.2)
                plane2=interpolateplane_2Dfield_gpt(coord_x,coord_y,dat2)
                r=r+np.sum((dat2-plane2)**2)
                #r=r+np.sum(v**2)
                #r=r+np.average((plane)**2)
                del plane,subvar
            result[tri]=result[tri]+r
            #if tri==5000:
            #    print("Number of components: ",varu[ti,...][subtri].shape)
        #totsum=totsum+result[5000]
        print("Another file finished.")
    print("End.")
    return result

if __name__ == "__main__":
    figpath="/work/bb1097/b381585/images/horizontal-tests/"+"testuu-data-sum.png"
    label=r"$u'u'$ [m^2 s$^{-2}$]"
    fn_grid_low="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"

    #result=flux_from_data_deplane()
    result=flux_from_data_deplane_temporal()

    
    #Plot the averages:
    #grid=xr.open_dataset(fn_grid_low)
    
    ta.test_averages_noread(fn_grid_low,result,figpath,label) #Approximately result*144 to have all days from one timestep
    print("End.")

