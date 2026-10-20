import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from   matplotlib.collections import PolyCollection

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def test_averages(fn_grid_low,resultpath,figpath,label):
    results=xr.open_dataset(resultpath)
    var=results['var'][0,0,:].values
    var=np.ones_like(var)
    var[100]=2
    test_averages_noread(fn_grid_low,var,figpath,label)

def test_averages_noread(fn_grid_low,var,figpath,label):
    gridfil=xr.open_dataset(fn_grid_low)
    #lats=gridfil["clat"]*180/3.14159
    #lons=gridfil["clon"]*180/3.14159
    clon = np.rad2deg(gridfil.clon.values)
    clat = np.rad2deg(gridfil.clat.values)
    clon_vertices = np.rad2deg(gridfil.clon_vertices.values)#[128:,:]
    clat_vertices = np.rad2deg(gridfil.clat_vertices.values)#[128:,:]
    #m = Basemap(projection='moll',lon_0=0,resolution='c')
    #plt.tricontourf(lons,lats,var[0,0,:])

    #-- set contour levels, labels
    ncells, nv = clon_vertices.shape[0], clon_vertices.shape[1]
    #varMin, varMax, varInt = -25, 60, 5
    varMin=np.nanmin(var)
    varMax=np.nanmax(var)
    print("Minimum ",varMin)
    print("Maximum ",varMax)
    varMin=0
    varMax=6.251184730440418
    #varMin=-1783137.0769935423/3 #Fix the same axis as before
    #varMax=55782.43605156628
    #varMax=4.932e6
    #varMin=np.quantile(var,0.005)
    #varMax=np.quantile(var,0.9995)
    #m=max(abs(varMax),abs(varMin))
    #m=50 #For comparison with high-data
    #varMin=-m
    #varMax=m
    #varInt=int((varMax-varMin)/16)
    #levels = np.arange(varMin, varMax+varInt, varInt)
    varNum=24+8
    dx=(varMax-varMin)/varNum
    levels=np.linspace(varMin-dx,varMax+dx,varNum+1)
    nlevs  = levels.size
    #labels = ['{:.2f}'.format(x) for x in levels]
    #-- set projection
    projection = ccrs.PlateCarree()
    #-- create figure and axes instances; we need subplots for plot and colorbar
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(projection=projection))
    ax.set_global()
    #-- plot land areas at last to get rid of the contour lines at land
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='dimgray', alpha=0.4,
                 zorder=2)
    #-- define color map
    #cmap     = plt.get_cmap('Spectral_r', nlevs)        #-- read the color map
    #cmap     = plt.get_cmap('RdYlBu_r', nlevs)        #-- read the color map
    cmap     = plt.get_cmap('Reds', nlevs)        #-- read the color map
    #cmap     = plt.get_cmap('Reds_r', nlevs-1)        #From red (left) to white (right)
    cmaplist = [i for i in range(cmap.N)]               #-- color bar indices
    ncol     = len(cmaplist)+1                            #-- number of colors, one additional is the leftmost
    colors   = np.zeros([ncells,4], np.float32)       #-- assign color array for triangles
    #-- set color index of all cells in between levels
    for m in range(0+1,ncol-1):
        vind = []
        for i in range(0,len(var), 1):
            if (var[i] >= levels[m] and var[i] < levels[m+1]):
               colors[i,:] = cmap(cmaplist[m])
               vind.append(i)
        print('set colors: finished level %3d' % m ,
              ' -- %5d ' % len(vind) ,
              ' polygons considered')
        del vind
    ##Get black color
    #cmap2     = plt.get_cmap('binary_r', nlevs-1)        #From red (left) to white (right) (in a nasty way)
    #Get white color
    cmap2     = plt.get_cmap('binary', nlevs-1)        #From white (left) to red (right) (in a nasty way)
    
    cmaplist2 = [i for i in range(cmap2.N)] 
    for i in range(0,len(var), 1): #Add values that are lower than minimum
        if var[i]<varMin:
            colors[i,:]=cmap2(cmaplist2[0])
    #-- create the triangles
    clon_vertices = np.where(clon_vertices < -180., clon_vertices + 360., clon_vertices)
    clon_vertices = np.where(clon_vertices >  180., clon_vertices - 360., clon_vertices)

    triangles = np.zeros((ncells, nv, 2), np.float32)

    for i in range(0, ncells, 1):
        triangles[i,:,0] = np.array(clon_vertices[i,:])
        triangles[i,:,1] = np.array(clat_vertices[i,:])

    print('')
    print('--> triangles done')
    #-- create polygon/triangle collection
    #coll = PolyCollection(triangles, array=None, fc=colors, edgecolors='black', linewidth=0.05, transform=ccrs.Geodetic(), zorder=0)
    coll = PolyCollection(triangles, array=None, fc=colors, edgecolors='none', linewidth=0.05, transform=ccrs.Geodetic(), zorder=0)
    ax.add_collection(coll)

    print('--> polygon collection done')
    #-- add a color bar
    cb = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=varMin, vmax=varMax))
    cb.set_array([])
    cbar = plt.colorbar(cb,
                        orientation='horizontal',
                        ticks=levels,
                        boundaries=levels,
                        format='%0.1f', #For u
                        #format="%.0E", #Reasonable for N
                        shrink=0.8,
                        pad=0.04,
                        aspect=30,
                        )
    plt.setp(cbar.ax.get_xticklabels()[::2], visible=False)
    #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    cbar.ax.tick_params(axis='x', labelrotation=30)
    cbar.set_label(label)
    #cbar.ax.invert_yaxis()

    plt.savefig(figpath,dpi=300)



if __name__ == "__main__":

    savedir="/work/bb1097/b381585/data/averaged/"
    fn_grid_low="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    resultpath=savedir+"average_u-sphere.nc"
    #figpath="/work/bb1097/b381585/images/averages/"+"testuav-new2-time0-15km-sphere.png"
    figpath="/work/bb1097/b381585/images/averages/"+"testav-ones.pdf"
    label=r"$u$ [m s$^{-1}$]"
    #label=r"$N^2$ [s$^{-2}$]"
    test_averages(fn_grid_low,resultpath,figpath,label)