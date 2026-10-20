from pyjokes import get_joke
from flux_from_horsp_test import horizontal_sp_flux
from flux_from_data_test import flux_from_data_one
from flux_from_data_deplane_test import flux_from_data_deplane
import test_average as ta
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import ks_2samp,ttest_ind



def plot_alpha_hist(data,figpath,label,mult=16):
    plt.hist(data*mult,density=True,bins=20,color="darkorange")
    av=np.average(data*mult)
    med=np.median(data*mult)
    plt.axvline(x=med, label="median "+str(round(med,2)),color="black")
    plt.axvline(x=av, label="average "+str(round(av,2)),color="black",ls='--')
    plt.legend()
    plt.xlabel(str(mult)+label)
    plt.savefig(figpath)
    plt.close()
    print("Plot finished.")


def save_alphas(alpha,path):
    ds=xr.Dataset( data_vars=dict(
        alpha=(["ntri"], alpha),),
        attrs=dict(description="Fraction between fluxes from spectra and fluxes from data."),
        )
    ds.to_netcdf(path)

def compute_alphas():
    #Compute fraction between flux from horisontal sp
    # 
    # ectrum and from data
    #Time instant (=time file) is hardcoded in the flux computation functions. 
    var="p" #u,v,w,p,b
    alphas=[]
    for ti in range(0,1):
        # Second time at horsp-u-ny4_group-46_20200126T080000Z.nc
        #res_data=flux_from_data_one(ti)

        res_data=flux_from_data_deplane(ti,var)
        print("Fluxes from data calculated.")

        res_horsp=horizontal_sp_flux(ti,var)  
        print("Fluxes from horizontal spectra calculated.")
        
        alpha2=res_horsp/res_data
        #alphas.append(alpha2)
    savepath='/work/bb1097/b381585/data/alphas/'+'alpha_'+var+var+'_t0_notaper.nc'
    fn_grid_low="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    save_alphas(alpha2,savepath)

    figpath="/work/bb1097/b381585/images/horizontal-tests/"+"test_"+var+var+"_hist_corrhorsp_t0-COBYLA.png"
    fn_grid_low="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    label=r"$\alpha$"
    #label=r"$p'p'$"
    plot_alpha_hist(alpha2,figpath,label)
    figpath="/work/bb1097/b381585/images/horizontal-tests/"+"test_"+var+var+"_alpha_corrhorsp_t0-COBYLA.png"
    ta.test_averages_noread(fn_grid_low,np.sqrt(alpha2),figpath,label) #Approximately *144 to have all days
    #ta.test_averages_noread(fn_grid_low,np.sqrt(res_horsp),figpath,label)

def hist_alphas_nocompute():
    var="b"
    file1='/work/bb1097/b381585/data/alphas/'+'alpha_'+var+var+'_t0_notaper.nc'
    alpha2=xr.open_dataset(file1).alpha.data
    print("Var1 read.")
    figpath="/work/bb1097/b381585/images/horizontal-tests/"+"test_"+var+var+"_hist_corrhorsp_t0-COBYLA-nomult.png"
    label=r"$\alpha$"
    #label=r"$p'p'$"
    plot_alpha_hist(np.sqrt(alpha2),figpath,label,mult=1)


def test_distributions():
    file1='/work/bb1097/b381585/data/alphas/'+'alpha_pp_t0_notaper.nc'
    file2='/work/bb1097/b381585/data/alphas/'+'alpha_pp_t40_notaper.nc'
    var1=xr.open_dataset(file1).alpha.data
    print("Var1 read.")
    var2=xr.open_dataset(file2).alpha.data
    print("Var2 read.")
    stat=ks_2samp(var1, var2)
    print("Kolmogorov test: ",stat)
    stat=ttest_ind(var1,var2,equal_var="True")
    print("Student t-test: ",stat)
    stat=ttest_ind(var1,var2,equal_var="False")
    print("Welch t-test: ",stat)
    print("Mean of var1 is: ",np.mean(var1))
    print("Mean of var2 is: ",np.mean(var2))
    
    

if __name__ == "__main__": 
    compute_alphas()
    hist_alphas_nocompute()
    test_distributions()
    print(get_joke())