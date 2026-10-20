import numpy  as np
import xarray as xr
import constants
import gc
from decomposition import coriolis_parameter,read_averages,average_averages
from energy_from_u import N_from_u
import scipy.interpolate as interpolate
import multiprocessing as mp

def read_decomposition(datadir,filename):
    #Read data
    ds=xr.open_dataset(datadir+filename)
    real=ds["ampls_real"]
    imag=ds["ampls_imag"]
    ampls=(real+imag*1j).values #time,ntri,omega,l,k
    del real,imag
    gc.collect()
    ntri=ds["ntri"].values
    k=ds["k"].values#/1000#ntri,nk
    l=ds["l"].values#/1000 #Convert from 2pi/km to 2pi/m
    omega=ds['omega'].values
    maxind=ds["max_ind"][:,:].values#ntri,dims
    ti  = ds["time"].values
    return ampls,k,l,omega,maxind,ntri, ti

def select_averaged_time_tri(data,file_slide,ntri, ti):
    new_dict = {}
    beg = file_slide * 12
    if file_slide == 73:
        end = file_slide *12 + 1
    else:
        end = file_slide*12 + 12
    if len(ti) == 2:
        index_t = [0, 6]
    elif len(ti):
        index_t = [0] 
    for key, value in data.items():
        A = value[beg : end, ntri]
        new_dict[key] = A[index_t, :]

    return new_dict

def read_decomposition_averages(dir_dec, filename_beg, filename_end, groupnr, av_dir, av_filename_beg, av_filename_end, file_slide):
    fn_dec=filename_beg+str(groupnr)+filename_end
    ampls,k,l,omega,maxind,ntri=read_decomposition(dir_dec,fn_dec)
    ampls=np.abs(ampls)#Convert complex values to amplitudes only
    #l[l==0]=np.nan #Invalid values that are also described by maxind #l has shape (ntri,nl)
    #k[k==0]=np.nan #k has shape (ntri,nk)
    print("Amplitudes read.")
    maxl=maxind[:,1]
    mask = np.arange(l.shape[1])[None, :] >= np.array(maxl)[:, None]
    l[mask] = np.nan

    maxk=maxind[:,0]
    mask = np.arange(k.shape[1])[None, :] >= np.array(maxk)[:, None]
    k[mask] = np.nan

    #Read averages
    #av_var=read_averages_uvNH(av_dir,av_filename_beg,av_filename_end) #Less averaged variables to compute just m
    av_var=read_averages(av_dir,av_filename_beg,av_filename_end) #Read more averaged variables so that the computation of wave action density is possible
    av_var=average_averages("","","","",av_var,path_f=dir_dec+filename_beg + "plus" + "-ny4_window_group-" + str(groupnr) + ".nc")
    
    av_var=select_averaged_time_tri(av_var,file_slide,ntri)
    return ampls,k,l,omega,maxind,ntri, av_var

def read_decomposition_averages_plusminus(dir_dec, filename_beg,filename_mid, filename_end, groupnr, av_dir, av_filename_beg, av_filename_end, file_slide):
    fn_dec=filename_beg+"plus"+filename_mid+str(groupnr)+filename_end
    ampls,k,l,omega,maxind,ntri, ti=read_decomposition(dir_dec,fn_dec)
    ampls_plus=np.abs(ampls)#Convert complex values to amplitudes only
    fn_dec=filename_beg+"minus"+filename_mid+str(groupnr)+filename_end
    ampls,k2,l,omega,maxind,ntri, _=read_decomposition(dir_dec,fn_dec)
    ampls_minus=np.abs(ampls)#Convert complex values to amplitudes only
    assert (k==k2).all(), "Different grids for plus and minus"
    #l[l==0]=np.nan #Invalid values that are also described by maxind #l has shape (ntri,nl)
    #k[k==0]=np.nan #k has shape (ntri,nk)
    print("Amplitudes read.")
    maxl=maxind[:,1]
    mask = np.arange(l.shape[1])[None, :] >= np.array(maxl)[:, None]
    l[mask] = np.nan

    maxk=maxind[:,0]
    mask = np.arange(k.shape[1])[None, :] >= np.array(maxk)[:, None]
    k[mask] = np.nan
    mask = (
    np.isnan(k)[None, :, None, None, :] |   # (1,128,1,23,1)
    np.isnan(l)[None, :, None, :, None])     # (1,128,1,1,23))

    ampls_plus = np.where(mask, np.nan, ampls_plus)
    ampls_minus = np.where(mask, np.nan, ampls_minus)
    #Read averages
    #av_var=read_averages_uvNH(av_dir,av_filename_beg,av_filename_end) #Less averaged variables to compute just m
    av_var=read_averages(av_dir,av_filename_beg,av_filename_end) #Read more averaged variables so that the computation of wave action density is possible
    av_var=average_averages("","","","",av_var,path_f=dir_dec+filename_beg + "plus" + filename_mid + str(groupnr) + filename_end)
    
    av_var=select_averaged_time_tri(av_var,file_slide,ntri, ti)
    return ampls_plus,ampls_minus,k,l,omega,maxind,ntri,av_var, ti

def intrinsic_frequency_alltri(freqs,u,v):
    #Intrinsic frequency from extrinsic frequency
    #Assumes that freqs=k,l,omega
    #both k and l have  a dimension of length tri, this dimension will be in resulting omega_hat, too
    k3d=freqs['k'][np.newaxis,:,np.newaxis,np.newaxis,:] #time,tri,omega,l,k,
    l3d=freqs['l'][np.newaxis,:,np.newaxis,:,np.newaxis] #time,tri,omega,l,k,
    omega=freqs['omega'][np.newaxis,np.newaxis,:,np.newaxis,np.newaxis] #time,tri,omega,l,k,
    u3d=u[:,:,np.newaxis,np.newaxis,np.newaxis]
    v3d=v[:,:,np.newaxis,np.newaxis,np.newaxis]
    return omega-l3d*v3d-k3d*u3d

def vertical_wavenumber_alltri_shape(kh2,omega_hat,N2,H,f):
    return kh2*(omega_hat**2-N2)/(f**2-omega_hat**2)-1/(4*H**2)

def vertical_wavenumber_alltri(av_var, k,l, omega_hat,f):
    print(f"k:{k.shape}")
    print(f"l:shape{l.shape}")
    k4d=k[np.newaxis,:,np.newaxis,np.newaxis,:]
    l4d=l[np.newaxis,:,np.newaxis,:,np.newaxis]
    kh2=k4d**2+l4d**2
    Hsh=av_var['H'][:,:,np.newaxis,np.newaxis,np.newaxis]
    N2sh=av_var['N2'][:,:,np.newaxis,np.newaxis,np.newaxis]
    fsh=f[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
    m=np.sqrt(vertical_wavenumber_alltri_shape(kh2,omega_hat,N2sh,Hsh,fsh))
    return m

def compute_m_nan(f,av_var,k,l,omega_hat,ampls,levth):
    #Find the maximal dimensions and create the output grid
    m=vertical_wavenumber_alltri(av_var, k,l, omega_hat,f) #for one group shape (6, 128,143, 21, 23)
    print(f"m: {m.shape}")
    print(f"ampls shape : {ampls.shape}")
    assert m.shape==ampls.shape,"Shapes do not seem to be consistent."
    mbnd=2*3.14159/(2*levth)
    #m[m>mbnd[0]]=np.nan #For testing with one altitude, modify to the code below afterward
    for li in range(0,len(mbnd)):
        m[li,np.abs(m[li,:])>mbnd[li]]=np.nan #Remove values that should not be resolved.
    return m

def group_velocity_vertical(av_var, k, l, f, m_plus):
    N2 = av_var["N2"]
    N2_3D = N2[:, :, np.newaxis, np.newaxis, np.newaxis]
    k_3d  = k[np.newaxis, :, np.newaxis, np.newaxis, :]
    l_3d  = l[np.newaxis, :, np.newaxis, : ,np.newaxis]
    kh2   = k_3d**2 + l_3d**2
    f2_3D = f[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]**2
    kappa2 = kh2 + m_plus**2
    omega_hat = np.sqrt((N2_3D * kh2 + f2_3D * m_plus**2) / kappa2)
    cgz       = (N2_3D - f2_3D) * kh2 * m_plus / (omega_hat * kappa2**2)
    return cgz

def wave_numbers():
    # ── 1. Define target grids ─────────────────────────────────────────────────

    # Horizontal: k, l can be positive and negative (direction of propagation)
    lambda_h = np.array([150e3, 100e3, 50e3])          # large to small (m)
    k_mag    = np.sort(2 * np.pi / lambda_h)                # small to large (increasing)
    k_target = np.concatenate([-k_mag[::-1], k_mag])        # symmetric 8 values
    l_target = k_target.copy()

    # Vertical: m always positive (upward waves), log-spaced
    lambda_z = np.array([12e3, 7e3, 3.5e3, 1.5e3])         # large to small (m)
    m_target = 2 * np.pi / lambda_z                         # small to large (increasing) ✓
    return m_target, l_target, k_target

def linear_edges(centers):
    """Midpoint edges for sorted array, works for negative values."""
    inner = 0.5 * (centers[:-1] + centers[1:])
    left  = centers[0]  - 0.5 * (centers[1]  - centers[0])
    right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[left], inner, [right]])

def log_edges(centers):
    """Geometric mean edges for positive log-spaced centers."""
    log_c = np.log(centers)
    inner = np.exp(0.5 * (log_c[:-1] + log_c[1:]))
    left  = centers[0]  ** 2 / inner[0]
    right = centers[-1] ** 2 / inner[-1]
    return np.concatenate([[left], inner, [right]])


def coarse_grain(k_target, l_target, m_target, m_plus, k, l , N_wadensity_minus, c_gz, time):
    N_cgz = N_wadensity_minus.real
    k_edges = linear_edges(k_target)   # (9,)  for 8 bins
    l_edges = linear_edges(l_target)   # (9,)  for 8 bins
    m_edges = log_edges(m_target)      # (5,)  for 4 bins

    print("\nk_edges (rad/m):", k_edges)
    print("l_edges (rad/m):", l_edges)
    print("m_edges (rad/m):", m_edges)

    # ── 5. Broadcast k, l to full shape ───────────────────────────────────────
    k_3d  = k[np.newaxis, :, np.newaxis, np.newaxis, :]
    l_3d  = l[np.newaxis, :, np.newaxis, : ,np.newaxis]
    k_full = np.broadcast_to(k_3d, m_plus.shape)   # (2,128,144,23,23)
    l_full = np.broadcast_to(l_3d, m_plus.shape)   # (2,128,144,23,23)

    # ── 6. Coarse-grain: sum N_cgz per bin ────────────────────────────────────
    # N_action_coarse = Σ(N_phys * dV_old) per bin
    #                 = Σ(N_cgz/dV_old * dV_old) per bin
    #                 = Σ(N_cgz) per bin  ← sum N_cgz directly
    
    
    # Frequency spacing in rad/s
    dt = 600.0
    n_omega = m_plus.shape[2]
    delta_omega = 2 * np.pi / (n_omega * dt)

    print("\n===== FREQUENCY SPACING =====")
    print("delta_omega =", delta_omega, "rad/s")

    # Numerical dm/domega along frequency axis
    dm_domega = np.gradient(
        np.real(m_plus),
        delta_omega,
        axis=2
    )

    # Numerical domega/dm
    with np.errstate(divide="ignore", invalid="ignore"):
        domega_dm = 1.0 / dm_domega

    # Compare against group velocity
    cgz_abs = np.abs(np.real(c_gz))

    valid = (
        np.isfinite(dm_domega) &
        np.isfinite(domega_dm) &
        np.isfinite(cgz_abs) &
        (np.abs(dm_domega) > 0) &
        (cgz_abs > 0) &
        (np.real(m_plus) > 0)
    )

    print("\n===== JACOBIAN DIAGNOSTICS =====")

    for name, arr in [
        ("|dm/domega| (s/m)", np.abs(dm_domega)),
        ("|domega/dm| (m/s)", np.abs(domega_dm)),
        ("|c_gz| (m/s)", cgz_abs),
    ]:
        values = arr[valid]

        print(f"\n{name}")
        print("  min:   ", np.min(values))
        print("  median:", np.median(values))
        print("  mean:  ", np.mean(values))
        print("  max:   ", np.max(values))

    # Test the dispersion-relation identity
    ratio = np.abs(domega_dm[valid]) / cgz_abs[valid]

    print("\n===== JACOBIAN / GROUP VELOCITY =====")
    print("median |domega/dm| / |c_gz|:", np.median(ratio))
    print("mean   |domega/dm| / |c_gz|:", np.mean(ratio))
    print("5th percentile:", np.percentile(ratio, 5))
    print("95th percentile:", np.percentile(ratio, 95))
    
    

    N_action_coarse = np.full((len(time), 128, len(k_target), len(l_target), len(m_target)), np.nan)

    for t in range(len(time)):
        for tri in range(128):
            k_flat = np.real(k_full[t, tri].ravel())
            l_flat = np.real(l_full[t, tri].ravel())
            m_flat = np.real(m_plus[t, tri].ravel())
            w_flat = np.real(N_cgz[t, tri].ravel())    # sum N_cgz directly

            valid = (
                np.isfinite(k_flat) & np.isfinite(l_flat) &
                np.isfinite(m_flat) & np.isfinite(w_flat) &
                (m_flat > 0)
            )

            if valid.sum() == 0:
                continue

            k_v, l_v, m_v, w_v = k_flat[valid], l_flat[valid], m_flat[valid], w_flat[valid]

            for ik, (k_lo, k_hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
                for il, (l_lo, l_hi) in enumerate(zip(l_edges[:-1], l_edges[1:])):
                    for im, (m_lo, m_hi) in enumerate(zip(m_edges[:-1], m_edges[1:])):
                        in_bin = (
                            (k_v >= k_lo) & (k_v < k_hi) &
                            (l_v >= l_lo) & (l_v < l_hi) &
                            (m_v >= m_lo) & (m_v < m_hi)
                        )
                        if in_bin.sum() > 0:
                            N_action_coarse[t, tri, ik, il, im] = np.nansum(w_v[in_bin])

    print("\nN_action_coarse shape:", N_action_coarse.shape)   # (2, 128, 8, 8, 4)
    print("N_action_coarse non-nan:", np.sum(np.isfinite(N_action_coarse)))
    print("Fill fraction:", np.isfinite(N_action_coarse).mean())
    print("Fill per m bin:", np.isfinite(N_action_coarse).mean(axis=(0,1,2,3)))
    print("Fill per k bin:", np.isfinite(N_action_coarse).mean(axis=(0,1,3,4)))

    # ── 7. Sanity check: Σ(N_cgz) before vs Σ(N_action_coarse) after ─────────
    # Both equal Σ(N_phys * dV_old) — losses only from k=0 gap and out-of-range

    print("\nSanity check — N_phys*dV conservation:")
    for t in range(len(time)):
        for tri in [0, 1, 2]:
            before = np.nansum(np.real(N_cgz[t, tri]))
            after  = np.nansum(N_action_coarse[t, tri][np.isfinite(N_action_coarse[t, tri])])
            print(f"  t={t} tri={tri}: Σ(N_cgz)={before:.3e}  "
                f"Σ(N_action_coarse)={after:.3e}  ratio={after/before:.3f}")
    return N_action_coarse


from datetime import datetime, time

def save_coarse_gw_spectrum(N_action_coarse, k_target, l_target, m_target, 
                             tri_coords=None, time_coords=None, 
                             output_path='coarse_gw_spectrum.nc'):
    """
    Save coarse-grained gravity wave spectrum to NetCDF.
    
    Parameters
    ----------
    N_action_coarse : np.ndarray (nt, ntri, nk, nl, nm)
        Coarse-grained wave action N·cgz summed per bin
    k_target : np.ndarray (nk,)
        Horizontal wavenumber targets in rad/m
    l_target : np.ndarray (nl,)
        Horizontal wavenumber targets in rad/m  
    m_target : np.ndarray (nm,)
        Vertical wavenumber targets in rad/m
    tri_coords : np.ndarray (ntri,), optional
        Triangle indices or IDs
    time_coords : np.ndarray (nt,), optional
        Time coordinates
    output_path : str
        Output .nc file path
    """
    
    nt, ntri, nk, nl, nm = N_action_coarse.shape
    
    # Default coords if not provided
    if tri_coords is None:
        tri_coords = np.arange(ntri)
    if time_coords is None:
        time_coords = np.arange(nt)
    
    # Derived variables — wavelengths for convenience
    lambda_h_k = 2 * np.pi / np.abs(k_target)   # (nk,) horizontal wavelength from k
    lambda_h_l = 2 * np.pi / np.abs(l_target)   # (nl,) horizontal wavelength from l
    lambda_z   = 2 * np.pi / m_target            # (nm,) vertical wavelength

    # Build xarray Dataset
    ds = xr.Dataset(
        {
            # Main variable
            'N_action_coarse': xr.DataArray(
                N_action_coarse.astype(np.float64),
                dims=['time', 'triangle', 'k', 'l', 'm'],
                attrs={
                    'long_name'    : 'Coarse-grained gravity wave action flux N·cgz',
                    'units'        : 'kg m-1 s-1',
                    'description'  : 'Sum of N·cgz from fine spectral grid into coarse (k,l,m) bins',
                }
            ),
        },
        coords={
            'time': xr.DataArray(
                time_coords,
                dims=['time'],
                attrs={'long_name': 'time'}
            ),
            'triangle': xr.DataArray(
                tri_coords,
                dims=['triangle'],
                attrs={'long_name': 'triangle index'}
            ),
            'k': xr.DataArray(
                k_target,
                dims=['k'],
                attrs={
                    'long_name' : 'zonal wavenumber',
                    'units'     : 'rad m-1',
                    'note'      : 'negative=westward, positive=eastward'
                }
            ),
            'l': xr.DataArray(
                l_target,
                dims=['l'],
                attrs={
                    'long_name' : 'meridional wavenumber',
                    'units'     : 'rad m-1',
                    'note'      : 'negative=southward, positive=northward'
                }
            ),
            'm': xr.DataArray(
                m_target,
                dims=['m'],
                attrs={
                    'long_name' : 'vertical wavenumber',
                    'units'     : 'rad m-1',
                    'note'      : 'positive=upward propagating waves only'
                }
            ),
            # Convenience wavelength coords
            'lambda_h_k': xr.DataArray(
                lambda_h_k,
                dims=['k'],
                attrs={
                    'long_name' : 'horizontal wavelength from k',
                    'units'     : 'm'
                }
            ),
            'lambda_h_l': xr.DataArray(
                lambda_h_l,
                dims=['l'],
                attrs={
                    'long_name' : 'horizontal wavelength from l',
                    'units'     : 'm'
                }
            ),
            'lambda_z': xr.DataArray(
                lambda_z,
                dims=['m'],
                attrs={
                    'long_name' : 'vertical wavelength',
                    'units'     : 'm'
                }
            ),
        },
        attrs={
            'title'         : 'Coarse-grained gravity wave spectrum from ICON triangular mesh',
            'description'   : 'Wave action N·cgz binned onto regular (k,l,m) grid per triangle',
            'created'       : datetime.now().isoformat(),
            'k_lambda_km'   : str((2*np.pi/np.abs(k_target)/1e3).round(1).tolist()),
            'm_lambda_km'   : str((2*np.pi/m_target/1e3).round(2).tolist()),
            'conventions'   : 'CF-1.8',
        }
    )
    
    # Encoding for compression
    encoding = {
        'N_action_coarse': {
            'zlib'       : True,
            'complevel'  : 4,
            'dtype'      : 'float32',
            '_FillValue' : np.nan,
        }
    }
    
    ds.to_netcdf(output_path, encoding=encoding)
    print(f"Saved: {output_path}")
    print(f"  Shape: {N_action_coarse.shape}  (nt={nt}, ntri={ntri}, nk={nk}, nl={nl}, nm={nm})")
    print(f"  k targets (rad/m):      {k_target}")
    print(f"  k wavelengths (km):     {(2*np.pi/np.abs(k_target)/1e3).round(1)}")
    print(f"  m targets (rad/m):      {m_target}")
    print(f"  m wavelengths (km):     {(2*np.pi/m_target/1e3).round(2)}")
    print(f"  Non-nan fraction:       {np.isfinite(N_action_coarse).mean():.3f}")
    print(f"  Total N_action (all t): {np.nansum(N_action_coarse):.3e}")
    
    return ds


def action_coarse(file_slide):
    levth=np.array([453.208, 400.000, 400.000, 400.000, 400.000, 373.779])[[0]] #Thickness of the levels in the ICON simulation
    nrproc=128 #Number of processes to be run in parallel
    dir_dec = f"/work/bb1097/b383521/GW_Data_Summer/decomposition/slide_{file_slide}/"
    save_dir = "/work/bb1097/b383521/GW_Data_Summer/coarsed_action/"
    filename_beg="sp3D-gw_"
    filename_mid = "-ny4_window_group-"
    filename_end = ".nc"
    av_dir  = "/work/bb1097/b383521/resolved_var/"
    av_filename_beg = "average_"
    av_filename_end = "-sphere_2016.nc"
    for groupnr in range(160): # change this
        print(f"groupnr {groupnr}")
        ampls_plus,ampls_minus,k,l,omega,maxind,ntri, av_var, ti= read_decomposition_averages_plusminus(dir_dec, filename_beg,filename_mid, filename_end, groupnr, av_dir, av_filename_beg, av_filename_end, file_slide=file_slide)
        freqs_alltri=dict(zip(["k",'l','omega'],[k,l,omega]))
        omega_hat=intrinsic_frequency_alltri(freqs_alltri,av_var['u'],av_var['v'])
        ampls_plus[omega_hat<0]=np.nan #Should be used if taking both plus and minus. (1, 10, 143, 17, 23)
        ampls_minus[omega_hat<0]=np.nan
        path_grid = "/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
        f=coriolis_parameter(xr.open_dataset(path_grid)['clat'].data)[ntri]
        m_plus=compute_m_nan(f,av_var,k,l,omega_hat,ampls_plus,levth)
        N_wadensity_minus = N_from_u(omega_hat,k,l,f,av_var,np.abs(ampls_minus))
        cgz = group_velocity_vertical(av_var, k, l, f, m_plus)
        m_target, l_target, k_target = wave_numbers()
        print("k_target (rad/m):", k_target)
        print("k_target → lambda_h (km):", 2*np.pi / np.abs(k_target[4:]) / 1e3)
        print("m_target (rad/m):", m_target)
        print("m_target → lambda_z (km):", 2*np.pi / m_target / 1e3)
        N_coarse_grain = coarse_grain(k_target, l_target, m_target, m_plus, k, l , N_wadensity_minus, cgz, ti)

        ds = save_coarse_gw_spectrum(
            N_action_coarse = N_coarse_grain,
            k_target        = k_target,
            l_target        = l_target,
            m_target        = m_target,
            tri_coords      = ntri,     # or your actual triangle IDs
            time_coords     = ti,       # or your actual time values
            output_path     = save_dir + f'coarse_gw_spectrum_group_{groupnr}_fileslide_{file_slide}_2.nc'
        )

        # ── Quick check ────────────────────────────────────────────────────────────
        print("\nDataset overview:")
        print(ds)

if __name__ == "__main__":
    action_coarse(72)