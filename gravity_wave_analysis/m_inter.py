import numpy as np
import xarray as xr
import gc
import constants
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import scipy.interpolate as interpolate
import multiprocessing as mp

def read_dimesions(data_dir, filename_beg, filename_mid, filename_end, groupnr):
    filename = filename_beg+"plus"+filename_mid+str(groupnr)+filename_end
    ds   = xr.open_dataset(data_dir+filename)
    ntri = ds.ntri.data
    k = ds.k.data
    l = ds.l.data
    omega = ds.omega.data
    time = ds.time.data
    ds.close()
    del ds
    return ntri, k, l, omega, time

def read_decomposition(data_dir, filename, ti):
    ds   = xr.open_dataset(data_dir+filename)
    real = ds.ampls_real.data[ti, ...]
    imag = ds.ampls_imag.data[ti, ...]
    ampls = real + 1j*imag
    ds.close()
    #ampls = ampls[np.newaxis, :, :, :]
    del real, imag
    gc.collect()
    return ampls

def read_av(av_dir, av_filename_beg, av_filename_end):
    varlist = ["u","v","N","T","p"]
    paths = [av_dir+av_filename_beg+v+av_filename_end for v in varlist]
    vars = [xr.open_dataset(path)["var"].data[:, 0, :] for path in paths] # each has index (1008, 1, 20480) (time, height TODO, large triangles)
    vars1 = vars[0:3]
    vars_mid = vars[3:5]
    alt = xr.open_dataset(paths[0]).z.data[0] # for now only the first height
    cs2= constants.cp/constants.cv*constants.R*vars_mid[0] # cp/cv*RT
    H=constants.R*vars_mid[0]/constants.g # RT/g
    H_pot=constants.cp/constants.R*H # cp/R*H
    rho=vars_mid[1]/vars_mid[0]/constants.R # T/(pR)
    rho0=rho*np.exp(alt/H) # rho_0 e^{z/H}
    vars2 = [cs2,H,H_pot,rho,rho0]
    names = ["u","v","N2","cs2","H","H_pot","rho","rho0"]
    di    = dict(zip(names, vars1+vars2))
    return di

def avarge_avrages(avarages, window_length): # this is very very crucial for in fact this forms your training, (866, 1, 20480) for each variables in the dictionary
    core = np.ones(window_length)/window_length
    new_dict = {}
    for key, value in avarages.items():
        new_dict[key] = np.apply_along_axis(lambda am: np.convolve(am, core, mode="valid"), axis=0, arr=value)
    return new_dict

def select_avarge_time_tri(data, ti, ntri, file_slide):
    new_dict = {}
    for key, value in data.items():
        new_dict[key] = value[12*file_slide + 6*ti, ntri] #TODO: if you were supposed to change the time window slide to an 1hout 6*ti
    return new_dict

def read_decomosition_avarge_plus_minus(dir_dec, filename_beg, groupnr, filename_mid, filename_end, av_var, ti, ntri, file_slide):
    fn_dec=filename_beg+"plus"+filename_mid+str(groupnr)+filename_end
    ampls = read_decomposition(data_dir=dir_dec, filename=fn_dec, ti=ti)
    ampls_plus = np.abs(ampls)
    fn_dec=filename_beg+"minus"+filename_mid+str(groupnr)+filename_end
    ampls = read_decomposition(data_dir=dir_dec, filename=fn_dec, ti=ti)
    ampls_minus = np.abs(ampls)
    av_var_triangle_time = select_avarge_time_tri(av_var, ti, ntri, file_slide)
    return ampls_plus, ampls_minus, av_var_triangle_time

def corilios_parameter(lat):
    Om=7.2921e-5 #2pi/24h in rad/s
    return 2*Om*np.sin(lat)

def vertical_wavenumber_alltri_shape(k, l, av_var_1time_triangles, omega_hat, f):
    k4d = k[np.newaxis, np.newaxis, np.newaxis, :] #(tri, omega, l, k)
    l4d = l[np.newaxis, np.newaxis, :, np.newaxis]
    kh2 = k4d**(2) + l4d**(2)
    Hsh = np.transpose(av_var_1time_triangles["H"])[:, np.newaxis, np.newaxis, np.newaxis]
    N2sh = np.transpose(av_var_1time_triangles["N2"])[:, np.newaxis, np.newaxis, np.newaxis]
    fsh = f[:, np.newaxis, np.newaxis, np.newaxis]
    m = np.sqrt(kh2*(omega_hat**2-N2sh)/(fsh**2-omega_hat**2)-1/(4*Hsh**2))
    return m

def intrinsic_freq_alltri(k, l, omega,  av_var, f, N2):
    k3d = k[np.newaxis, np.newaxis, np.newaxis, :] # (ntri, omega, l, k)
    l3d = l[np.newaxis, np.newaxis, :, np.newaxis]
    omega3d = omega[np.newaxis, :, np.newaxis, np.newaxis]
    u3d = np.transpose(av_var["u"])[:, np.newaxis, np.newaxis, np.newaxis]
    v3d = np.transpose(av_var["v"])[:, np.newaxis, np.newaxis, np.newaxis]
    omega_hat = omega3d - l3d*v3d - k3d*u3d

    # ── filter to valid GW range ───────────────────────────────────────────
    # N2 shape: (ntri,) -> broadcast
    N2sh = np.sqrt(N2)[:, np.newaxis, np.newaxis, np.newaxis]    # N [rad/s]
    f3d  = f[:, np.newaxis, np.newaxis, np.newaxis]              # f [rad/s]

    # valid GW: f < |omega_hat| < N
    invalid = (np.abs(omega_hat) <= np.abs(f3d)) | (np.abs(omega_hat) >= N2sh)
    omega_hat[invalid] = np.nan

    return omega_hat

def compute_m_nan(k, l, av_var_1time_triangles, omega_hat, ampls, levth, f):
    m = vertical_wavenumber_alltri_shape(k, l, av_var_1time_triangles, omega_hat, f)
    assert(m.shape == ampls.shape), "there is inconsistency in dimensions"
    mbnd = 2*np.pi/(2*levth)
    
    m[np.abs(m)>mbnd]=np.nan
    return m

def sign2mult(sign):
    if sign=="minus":
        mult = -1
    else:
        mult = +1
    return mult

def m_gw(omega_hat, k4d, l4d, N2, f2, H, sign, cs2sh):
    sign = "plus"
    mult = sign2mult(sign)
    kh2 = k4d**2 + l4d**2
    with np.errstate(invalid='ignore'):
        if type(cs2sh) is np.ndarray:
            m = mult*np.sqrt(kh2*(omega_hat**2-N2)/(f2-omega_hat**2)-1/(4*H**2))
        else:
            m = mult*np.sqrt(kh2*(omega_hat**2-N2)/(f2-omega_hat**2)-1/(4*H**2)+omega_hat**2/cs2sh)
    return m

def v_from_u(omega_hat, k, l, f, u):
    with np.errstate(invalid='ignore'):
        return (l*omega_hat - 1j*k*f) / (k*omega_hat + 1j*l*f) * u

def b_from_u(omega_hat, k, l, m, f, H, H_theta, cs2, N2, u):
    den = N2*(omega_hat**2 - f**2 - cs2*(k**2 + l**2))
    nom = cs2*(k*omega_hat + 1j*l*f) * (1.0/H_theta - 1.0/(2.0*H) + 1j*m)
    with np.errstate(invalid='ignore'):
        return den/nom * u

def p_from_u(omega_hat, k, l, f, rho_0, u):  # keep this order everywhere
    den = k*omega_hat + 1j*l*f
    with np.errstate(invalid='ignore'):
        return rho_0*(omega_hat**2 - f**2)/den * u

def w_from_u(omega_hat, k, l, m, f, N2, H, H_theta, cs2, u):
    b = b_from_u(omega_hat, k, l, m, f, H, H_theta, cs2, N2, u)
    return 1j*omega_hat/N2 * b

def N_from_u(k, l, omega_hat, av_var_1time_triangles, f, ampls, sign):
    k4d = k[np.newaxis, np.newaxis, np.newaxis, :] #(ntri, omega, l, k)
    l4d = l[np.newaxis, np.newaxis, :, np.newaxis]
    size_u = np.abs(ampls)
    size_u[np.where(np.logical_not(np.isfinite(k4d)))]=np.nan
    size_u[np.where(np.logical_not(np.isfinite(l4d)))]=np.nan 
    o2 = omega_hat**(2)
    f2 = np.array(f**2)[:, np.newaxis, np.newaxis, np.newaxis]*np.ones_like(omega_hat)
    rhosh = np.transpose(av_var_1time_triangles["rho"])[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    N2sh = np.transpose(av_var_1time_triangles["N2"])[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    cs2sh = np.transpose(av_var_1time_triangles["cs2"])[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    Hsh = np.transpose(av_var_1time_triangles["H"])[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    m = m_gw(omega_hat, k4d, l4d, N2sh, f2, Hsh, sign, cs2sh)
    H_theta = constants.cp/constants.R*Hsh
    v = v_from_u(omega_hat=omega_hat, k=k4d, l=l4d, f=np.sqrt(f2), u=size_u)
    w = w_from_u(omega_hat=omega_hat, k=k4d, l=l4d, m=m, f=np.sqrt(f2), N2=N2sh, H=Hsh, H_theta=H_theta, cs2=cs2sh, u=size_u)
    b = b_from_u(omega_hat=omega_hat, k=k4d, l=l4d, m=m, f=np.sqrt(f2), H=Hsh, H_theta=H_theta, cs2=cs2sh, N2=N2sh, u=size_u)
    E = (size_u**2 + v*v.conjugate() + w*w.conjugate() + b*b.conjugate()/N2sh)*rhosh/4
    N = E/omega_hat
    return N.real

def compute_wave_action_density(dir_dec, filename_beg, groupnr, filename_mid, filename_end, av_dir, av_filename_beg, av_filename_end, path_grid, levth, file_slide):
    ntri, k, l, omega, time = read_dimesions(dir_dec, filename_beg=filename_beg, filename_mid=filename_mid, filename_end=filename_end, groupnr=groupnr)
    av_var = read_av(av_dir=av_dir, av_filename_beg=av_filename_beg, av_filename_end=av_filename_end)
    av_var = avarge_avrages(avarages=av_var, window_length=144)
    f = corilios_parameter(xr.open_dataset(path_grid).clat.data)[ntri]
    N_wadensity_plus = np.zeros((len(time), len(ntri), len(omega), len(l), len(k)))
    N_wadensity_minus = np.zeros((len(time), len(ntri), len(omega), len(l), len(k)))
    m_plus_full = np.zeros((len(time), len(ntri), len(omega), len(l), len(k)))
    omega_hat_full = np.zeros((len(time), len(ntri), len(omega), len(l), len(k)))
    for ti in range(len(time)):
        print(f"computing wave action density at time: {ti} ")
        ampls_plus, ampls_minus, av_var_1time_triangles = read_decomosition_avarge_plus_minus(dir_dec=dir_dec, filename_beg=filename_beg, filename_end=filename_end, groupnr=groupnr, filename_mid=filename_mid, av_var=av_var,
                                                                                                  ti=ti, ntri=ntri, file_slide=file_slide)
        omega_hat = intrinsic_freq_alltri(k, l, omega, av_var_1time_triangles, f=f, N2=np.transpose(av_var_1time_triangles["N2"]))
        m_plus = compute_m_nan(k, l, av_var_1time_triangles, omega_hat, ampls_plus, levth, f) #TODO: if we only want to compute upward wave then you only need minus!
        ampls_plus[omega_hat<0]=np.nan
        ampls_minus[omega_hat<0]=np.nan
        N_wadensity_plus[ti, ...] = N_from_u(k=k, l=l, omega_hat=omega_hat, av_var_1time_triangles=av_var_1time_triangles, f=f, ampls=ampls_plus, sign="plus")
        N_wadensity_minus[ti, ...] = N_from_u(k=k, l=l, omega_hat=omega_hat, av_var_1time_triangles=av_var_1time_triangles, f=f, ampls=ampls_minus, sign="minus")
        m_plus_full[ti, ...] = m_plus.real
        omega_hat_full[ti, ...] = omega_hat.real

    return N_wadensity_plus, N_wadensity_minus, k, l, m_plus_full, ntri, omega_hat_full, time, omega, f



def jacobian(k, l, time, omega_hat_full, m_plus_full, N2, file_s, f, alpha=1.1):
    ksh  = k[np.newaxis, :, np.newaxis, np.newaxis, :]
    lsh  = l[np.newaxis, :, np.newaxis, :, np.newaxis]
    kh2  = ksh**2 + lsh**2

    if file_s != 72:
        start_file = file_s*12
        N2   = N2[[start_file, start_file + 6], ...]
    else:
        start_file = file_s*12
        N2   = N2[[start_file], ...]
    N2sh = N2[:, :, np.newaxis, np.newaxis, np.newaxis]

    f_3D  = f[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]   # (1, ntri, 1, 1, 1)
    f2_3D = f_3D**2

    kappa2 = kh2 + m_plus_full**2

    # use omega_hat_full directly — already computed consistently
    c_gz = (N2sh - f2_3D) * kh2 * m_plus_full / (omega_hat_full * kappa2**2)

    # near-inertial filter: remove waves too close to f
    mask = np.abs(omega_hat_full) > alpha * f_3D
    c_gz = np.where(mask, c_gz, np.nan)   # nan is safer than 0 to avoid dividing by ~0

    print(f"max |cgz|: {np.nanmax(np.abs(c_gz)):.6f}")
    print(f"min |cgz|: {np.nanmin(np.abs(c_gz)):.6f}")
    print(f"cgz shape: {c_gz.shape}")

    # jacobian = 1/|cgz|
    return 1.0 / np.abs(c_gz)


def new_klm(N_wadensity_plus, N_wadensity_minus, k, l, m_plus, m_minus):
    Ns=np.concatenate([N_wadensity_minus, N_wadensity_plus],axis=-3)
    ms=np.concatenate([m_minus,m_plus],axis=-3)
    minsm = np.nanmin(ms[0, ...], axis=(1, 2, 3))
    maxsm = np.nanmax(ms[0, ...], axis=(1, 2, 3))
    minsk = np.nanmin(k, axis=1)
    maxsk = np.nanmax(k, axis=1)
    minsl = np.nanmin(l, axis=1)
    maxsl = np.nanmax(l, axis=1)
    min_m, max_m = np.nanmax(minsm), np.nanmin(maxsm)
    min_l, max_l = np.nanmax(minsl), np.nanmin(maxsl)
    min_k, max_k = np.nanmax(minsk), np.nanmin(maxsk)
    nm = ms.shape[2]//6
    ntri = ms.shape[1]
    nz = ms.shape[0]
    nk = k.shape[1]
    nl = l.shape[1]
    k_new = np.linspace(min_k, max_k, nk//2+2)[1:-1]
    l_new = np.linspace(min_l, max_l, nl//2+2)[1:-1]
    if np.sign(min_m)*np.sign(max_m)>0:
        m_new = np.sign(min_m)*np.logspace(np.log10(np.abs(min_m)), np.log10(np.abs(max_m)), nm//4+2)[1:-1]
    else:
        mzer = np.nanmin(np.abs(ms[0, ...]), axis=(1,2,3)) # maximum of m at each triangles
        m_at_zer = np.nanmax(mzer)
        m_new_minus = -np.logspace(np.log10(np.abs(min_m)), np.log10(np.abs(m_at_zer)), nm//4+2)[1:-1]
        m_new_plus = +np.logspace(np.log10(np.abs(m_at_zer)), np.log10(np.abs(max_m)), nm//4+2)[1:-1]
        m_new = np.concatenate([m_new_minus, m_new_plus])

    l_new_3d, k_new_3d, m_new_3d = np.meshgrid(l_new, k_new, m_new)
    #intdata = np.zeros((nz, ntri, nk//2, nl//2, len(m_new)), np.float64)
    k1d = np.repeat(np.repeat(k[:, np.newaxis, np.newaxis, :], repeats=ms.shape[2], axis=1), repeats=nl, axis=2)
    l1d = np.repeat(np.repeat(l[:, np.newaxis, :, np.newaxis], repeats=ms.shape[2], axis=1), repeats=nk, axis=3)
    m1d = ms.reshape(*ms.shape[:-3], -1)
    k1d = k1d.reshape(*k1d.shape[:-3], -1)
    l1d = l1d.reshape(*l1d.shape[:-3], -1)
    ampls1d = Ns.reshape(*Ns.shape[:-3], -1)
    return l_new_3d, k_new_3d, m_new_3d, k1d, l1d, m1d, ampls1d, k_new, l_new, m_new

def new_m_interp_old(N_wadensity_plus, N_wadensity_minus, l, k, m_plus, m_minus, time,ntri):
    print("starting interopolation to a unique vertical wave numbers")
    Ns=np.concatenate([N_wadensity_minus, N_wadensity_plus], axis=-3) # (time, ntri, omega, l, k)
    ms=np.concatenate([m_minus,m_plus], axis=-3)
    minsm = np.nanmin(ms[:, :, ...], axis=(1, 2))
    maxsm = np.nanmax(ms[:, :, ...], axis=(1, 2))
    min_m, max_m = np.nanmax(minsm), np.nanmin(maxsm)
    print(f"m_min is {min_m}")
    print(f"max m is {max_m}")
    
    
    nm = ms.shape[2]//6
    if np.sign(min_m)*np.sign(max_m)>0:
        m_new = np.sign(min_m)*np.logspace(np.log10(np.abs(min_m)), np.log10(np.abs(max_m)), nm//4+2)[1:-1]
    else:
        mzer = np.nanmin(np.abs(ms[0, ...]), axis=(1,2,3)) # maximum of m at each triangles
        m_at_zer = np.nanmax(mzer)
        m_new_minus = -np.logspace(np.log10(np.abs(min_m)), np.log10(np.abs(m_at_zer)), nm//4+2)[1:-1]
        m_new_plus = +np.logspace(np.log10(np.abs(m_at_zer)), np.log10(np.abs(max_m)), nm//4+2)[1:-1]
        m_new = np.concatenate([m_new_minus, m_new_plus])
    print(f"m_new is {m_new}")
    

    ds = xr.open_dataset("/work/bb1097/b383521/slided_window_avrage/inter_wave_action_density_filled_log/projected_data_fileslide-0-group-4.nc")
    m_new = ds.m.data

    ms = np.transpose(ms, (0, 1, 3, 4, 2))
    maskm = np.isfinite(ms)
    Ns = np.transpose(Ns, (0, 1, 3, 4, 2))
    result_for_tri = np.empty((len(time), len(ntri), len(l), len(k), len(m_new)))
    for ti in range(len(time)):
        for tri_i in range(len(ntri)):
            print(f"interpolating to m for time : {ti} and triangle {ntri[tri_i]}")
            for j in range(len(l)):
                for i in range(len(k)):
                    points = ms[ti, tri_i, j, i, :][maskm[ti, tri_i, j, i, :]]
                    values = Ns[ti, tri_i, j, i, :][maskm[ti, tri_i, j, i, :]]

                    if len(points) == 0 or len(values) == 0:
                        print(f"No valid data tri_i={tri_i}, j={j}, i={i}")
                        result_for_tri[ti, tri_i, j, i, :] = 1e-20 #?
                        continue

                    if len(points) < 2:
                        print(f"Insufficient data points for interpolation: tri_i={tri_i}, j={j}, i={i}")
                        result_for_tri[ti, tri_i, j, i, :] = values[0] if len(values) > 0 else np.nan
                        continue

                    mask_invalid = ~np.isfinite(values)
                    values[mask_invalid] = 1e-20
                    #print(values)
                    interopolator = interpolate.interp1d(points, values, bounds_error=False, fill_value=np.nan)
                    new_points = m_new
                    interp_values = interopolator(new_points)
                    result_for_tri[ti , tri_i, j, i, :] = interp_values
    return m_new, result_for_tri

def new_m_interp_old2(N_wadensity_plus, N_wadensity_minus, l, k, m_plus, m_minus, time, ntri, dk, dl, domega):
    print("starting interpolation to unique vertical wavenumbers")

    Ns = np.concatenate([N_wadensity_minus, N_wadensity_plus], axis=2)  # (time, ntri, omega, l, k)
    ms = np.concatenate([m_minus, m_plus], axis=2)
    print(f"minimum wave action density spectrum {np.nanmin(Ns)}")
    # ── determine m_new grid from data ────────────────────────────────────────
    minsm = np.nanmin(ms, axis=(0, 2))
    maxsm = np.nanmax(ms, axis=(0, 2))
    min_m = np.nanmax(minsm)
    max_m = np.nanmin(maxsm)
    print(f"m_min={min_m:.4f}  m_max={max_m:.4f}")

    nm = ms.shape[2] // 6

    if np.sign(min_m) * np.sign(max_m) > 0:
        m_new = np.sign(min_m) * np.logspace(
            np.log10(np.abs(min_m)), np.log10(np.abs(max_m)), nm // 4 + 2
        )[1:-1]
    else:
        mzer = np.nanmin(np.abs(ms), axis=(0, 2))
        m_at_zer = np.nanmax(mzer)
        m_new_minus = -np.logspace(np.log10(np.abs(min_m)), np.log10(np.abs(m_at_zer)), nm // 4 + 2)[1:-1]
        m_new_plus = +np.logspace( np.log10(np.abs(m_at_zer)), np.log10(np.abs(max_m)), nm // 4 + 2)[1:-1]
        m_new = np.concatenate([m_new_minus, m_new_plus])

    dm = np.abs(np.gradient(m_new))
    print(f"m_new range: {m_new[0]:.4f} to {m_new[-1]:.4f}, n={len(m_new)}")

    # transpose so omega axis is last
    ms = np.transpose(ms, (0, 1, 3, 4, 2))   # (time, ntri, l, k, omega)
    Ns = np.transpose(Ns, (0, 1, 3, 4, 2))

    result_for_tri = np.full((len(time), len(ntri), len(l), len(k), len(m_new)), np.nan)

    for ti in range(len(time)):
        for tri_i in range(len(ntri)):
            print(f"interpolating time={ti} triangle={ntri[tri_i]}")
            for j in range(len(l)):
                for i in range(len(k)):
                    m_col = ms[ti, tri_i, j, i, :]
                    N_col = Ns[ti, tri_i, j, i, :].copy()   # copy to avoid modifying original

                    # ── same as old: set non-GW points to small number ─────
                    mask_invalid = ~np.isfinite(N_col)
                    N_col[mask_invalid] = 1e-20

                    valid = np.isfinite(m_col)   # only need m to be finite now
                    if valid.sum() < 2:
                        result_for_tri[ti, tri_i, j, i, :] = 1e-20
                        continue

                    m_valid = m_col[valid]
                    N_valid = N_col[valid]

                    # sort by m before interpolating
                    sort_idx = np.argsort(m_valid)
                    m_sorted = m_valid[sort_idx]
                    N_sorted = N_valid[sort_idx]

                    interp = interpolate.interp1d(
                        m_sorted, N_sorted,
                        kind='linear',
                        bounds_error=False,
                        fill_value=1e-20    # same as old: small number outside range
                    )
                    result_for_tri[ti, tri_i, j, i, :] = interp(m_new)

    # ── OUTSIDE loop: go back to discrete ─────────────────────────────────────
    result_for_tri = result_for_tri * dk * dl * dm[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    print(f"minimum wave action density spectrum after interpolation {np.nanmin(result_for_tri)}")
    return m_new, result_for_tri


def new_m_interp(N_wadensity_plus, N_wadensity_minus, l, k, m_plus, m_minus, time, ntri, dk, dl, domega):
    print("starting interpolation to unique vertical wavenumbers")

    Ns = np.concatenate([N_wadensity_minus, N_wadensity_plus], axis=2)  # (time, ntri, omega, l, k)
    ms = np.concatenate([m_minus, m_plus], axis=2)
    print(f"minimum wave action density spectrum {np.nanmin(Ns)}")

    # ── determine m_new grid from each branch separately ─────────────────────
    min_m    = np.nanmin(m_minus)   # most negative globally
    max_m    = np.nanmax(m_plus)    # most positive globally

    # closest to zero from each branch
    m_at_zer_minus = np.nanmax(m_minus)   # least negative
    m_at_zer_plus  = np.nanmin(m_plus)    # least positive
    m_at_zer = max(np.abs(m_at_zer_minus), np.abs(m_at_zer_plus))

    print(f"min_m={min_m:.6f}  max_m={max_m:.6f}  m_at_zero={m_at_zer:.6f}")

    nm = ms.shape[2] // 6

    # negative branch: from most negative to least negative (increasing order)
    m_new_minus = -np.logspace(np.log10(np.abs(min_m)), np.log10(m_at_zer), nm // 4 + 2)[1:-1]

    # positive branch: from least positive to most positive
    m_new_plus = +np.logspace(np.log10(m_at_zer), np.log10(np.abs(max_m)), nm // 4 + 2)[1:-1]

    m_new = np.concatenate([m_new_minus, m_new_plus])
    dm    = np.abs(np.gradient(m_new))
    print(f"m_new range: {m_new[0]:.6f} to {m_new[-1]:.6f}, n={len(m_new)}")

    # transpose so omega axis is last
    ms = np.transpose(ms, (0, 1, 3, 4, 2))   # (time, ntri, l, k, omega)
    Ns = np.transpose(Ns, (0, 1, 3, 4, 2))

    result_for_tri = np.full((len(time), len(ntri), len(l), len(k), len(m_new)), np.nan)
    for ti in range(len(time)):
        for tri_i in range(len(ntri)):
            print(f"interpolating time={ti} triangle={ntri[tri_i]}")
            for j in range(len(l)):
                for i in range(len(k)):
                    m_col = ms[ti, tri_i, j, i, :]
                    N_col = Ns[ti, tri_i, j, i, :].copy()

                    # valid means BOTH m and N are finite
                    valid = np.isfinite(m_col) & np.isfinite(N_col)

                    if valid.sum() < 2:
                        result_for_tri[ti, tri_i, j, i, :] = 1e-20
                        continue

                    m_valid = m_col[valid]
                    N_valid = N_col[valid]

                    # sort by m
                    sort_idx = np.argsort(m_valid)
                    m_sorted = m_valid[sort_idx]
                    N_sorted = N_valid[sort_idx]

                    interp = interpolate.interp1d(
                        m_sorted, N_sorted,
                        kind='linear',
                        bounds_error=False,
                        fill_value=1e-20
                    )
                    result_for_tri[ti, tri_i, j, i, :] = interp(m_new)
    """
    for ti in range(len(time)):
        for tri_i in range(len(ntri)):
            print(f"interpolating time={ti} triangle={ntri[tri_i]}")
            for j in range(len(l)):
                for i in range(len(k)):
                    m_col = ms[ti, tri_i, j, i, :]
                    N_col = Ns[ti, tri_i, j, i, :].copy()

                    # valid means BOTH m and N are finite
                    valid = np.isfinite(m_col) & np.isfinite(N_col)

                    if valid.sum() < 1:
                        result_for_tri[ti, tri_i, j, i, :] = 1e-20
                        continue

                    m_valid = m_col[valid]
                    N_valid = N_col[valid]

                    # ── bin onto m_new instead of interpolating ────────────────
                    # define bin edges as midpoints between m_new points
                    edges = np.concatenate([
                        [m_new[0] - (m_new[1]-m_new[0])/2],
                        (m_new[1:] + m_new[:-1]) / 2,
                        [m_new[-1] + (m_new[-1]-m_new[-2])/2]
                    ])

                    # sum N into each bin
                    bin_sums, _ = np.histogram(m_valid, bins=edges, weights=N_valid)
                    bin_counts, _ = np.histogram(m_valid, bins=edges)
                    bin_avg = np.where(bin_counts > 0, bin_sums / bin_counts, 1e-20)
                    # where bins have data use the sum, elsewhere 1e-20
                    result_for_tri[ti, tri_i, j, i, :] = bin_avg
    """""
    # ── OUTSIDE loop: go back to discrete ─────────────────────────────────────
    result_for_tri = result_for_tri * dk * dl * dm[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    print(f"minimum wave action density after interpolation {np.nanmin(result_for_tri)}")

    return m_new, result_for_tri


def save_interpolated(save_path,intdata, k, l, m_new, tri, varname, time):
    ds=xr.Dataset(data_vars=dict(
            data=(["ntime","ntri","k","l","m"], intdata),
            k=(["nk"], k),
            l=(["nl"], l),
            m=(["nm"], m_new),
        ),
        coords=dict(
            ntri=np.array(tri), #Indices of the original triangles
            time = time[:]
        ),
        attrs=dict(description="Interpolated data for the group",
                    varname=varname,),
        )
    ds.to_netcdf(save_path)
    ds.close()

def vertical_interp_old(group_id, file_slide):
    levth=np.array([453.208, 400.000, 400.000, 400.000, 400.000, 373.779])[[0]]
    dir_dec = "/work/bb1097/b383521/dummy/decomposition/"
    filename_beg="decomposed_u_gw_"#plus minus between beg and mid
    filename_end=".nc"
    filename_mid="_group-"
    path_grid="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    groupnr = group_id
    av_dir="/work/bb1097/b381585/data/averaged/"
    av_filename_beg="average_"
    av_filename_end="-sphere.nc"
    save_path = "/work/bb1097/b383521/dummy/interp_wave_action_density/"+ "interp-alt0-fileslide_" + str(file_slide) +"-regroup-"+str(groupnr)+".nc"
    print(f'save_path : {save_path}')
    av_var = read_av(av_dir, av_filename_beg, av_filename_end)
    avrgae_of_avrage = avarge_avrages(av_var,window_length=144)
    N2 = avrgae_of_avrage["N2"]
    N_wadensity_plus, N_wadensity_minus, k, l, m_plus_full, ntri, omega_hat_full, time = compute_wave_action_density(dir_dec=dir_dec, filename_beg=filename_beg, groupnr=groupnr, filename_mid=filename_mid, filename_end=filename_end, av_dir=av_dir,
                                                                                                                        av_filename_beg=av_filename_beg, av_filename_end=av_filename_end, path_grid=path_grid, levth=levth, file_slide=file_slide)
    N2 = N2[:, ntri]
    jacobi = jacobian(k, l, time=time,omega_hat_full=omega_hat_full, m_plus_full=m_plus_full, N2=N2, file_s=file_slide)
    N_wadensity_minus = N_wadensity_minus*jacobi
    N_wadensity_plus = N_wadensity_plus*jacobi
    m_new, result_for_tri = new_m_interp(N_wadensity_plus=N_wadensity_plus, N_wadensity_minus=N_wadensity_minus, m_plus=m_plus_full, m_minus=-m_plus_full, time=time, ntri=ntri, k=k, l=l)
    save_interpolated(save_path=save_path, intdata=result_for_tri ,k=k,l=l,m_new=m_new, tri=ntri, varname="Wave action density", time=time)

def vertical_interp(group_id, file_slide):
    levth = np.array([453.208, 400.000, 400.000, 400.000, 400.000, 373.779])[[0]]
    dir_dec = "/work/bb1097/b383521/dummy/decomposition/"
    filename_beg = "decomposed_u_gw_"
    filename_end = ".nc"
    filename_mid = "_group-"
    path_grid = "/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"
    groupnr = group_id
    av_dir = "/work/bb1097/b381585/data/averaged/"
    av_filename_beg = "average_"
    av_filename_end = "-sphere.nc"
    save_path = (
        "/work/bb1097/b383521/dummy/interp_wave_action_density/"
        + "interp-alt0-fileslide_" + str(file_slide)
        + "-regroup-" + str(groupnr) + ".nc"
    )
    print(f'save_path : {save_path}')

    av_var = read_av(av_dir, av_filename_beg, av_filename_end)
    avrgae_of_avrage = avarge_avrages(av_var, window_length=144)
    N2 = avrgae_of_avrage["N2"]

    # ── Step 1: compute N(k,l,omega) and m(k,l,omega) ────────────────────────
    (N_wadensity_plus, N_wadensity_minus,
     k, l, m_plus_full, ntri, omega_hat_full, time, omega, f) = compute_wave_action_density(
        dir_dec=dir_dec, filename_beg=filename_beg, groupnr=groupnr,
        filename_mid=filename_mid, filename_end=filename_end,
        av_dir=av_dir, av_filename_beg=av_filename_beg,
        av_filename_end=av_filename_end, path_grid=path_grid,
        levth=levth, file_slide=file_slide
    )

    A_old = np.concatenate([N_wadensity_minus, N_wadensity_plus], axis=2) 
    #np.save("/work/bb1097/b383521/spectrum_ml/A2_old_" + str(group_id) + ".npy", A_old)

    print(f"max wave action density : {np.nanmax(N_wadensity_plus)}")
    print(f"min wave action density : {np.nanmin(N_wadensity_plus)}")
    # shapes: (time, ntri, omega, l, k)

    N2 = N2[:, ntri]

    # ── Step 2: Jacobian converts discrete N(omega) -> continuous N(m) ───────
    # jacobi = |d_omega / d_m| = 1/|cgz|  so that:
    # N_continuous(k,l,m) = N_discrete(k,l,omega) * jacobi / (dk*dl*domega)
    # but since we only interpolate in m (k,l stay same) we just need jacobi

    # grid spacings
    dk = np.nanmean(np.diff(k))
    dl = np.nanmean(np.diff(l))
    domega = np.nanmean(np.diff(omega))  # you need omega here — add it to compute_wave_action_density return
    
    # ── Step 2: go to continuous ───────────────────────────────────────────────────
    N_wadensity_plus  = N_wadensity_plus  / (dk * dl * domega)
    N_wadensity_minus = N_wadensity_minus / (dk * dl * domega)
    print(f"min wave action density : {np.nanmin(N_wadensity_plus)}")
    # ── Step 3: Jacobian: continuous N(k,l,omega) -> continuous N(k,l,m) ──────────
    jacobi = jacobian(k=k, l=l, time=time, omega_hat_full=omega_hat_full,
                    m_plus_full=m_plus_full, N2=N2, file_s=file_slide, f=f)


    # ── Step 3: multiply BEFORE interpolation ─────────────────────────────────
    # this gives us N as a function of m (still on the irregular m grid
    # defined by the dispersion relation at each (k,l,omega) point)
    N_wadensity_plus  = N_wadensity_plus  / jacobi
    N_wadensity_minus = N_wadensity_minus / jacobi
    print(f"min wave action plus density : {np.nanmin(N_wadensity_plus)}")
    print(f"min wave action minus density : {np.nanmin(N_wadensity_minus)}")
    # ── Step 4: interpolate from irregular m(k,l,omega) onto regular m_new ───
    m_new, result_for_tri = new_m_interp(
        N_wadensity_plus=N_wadensity_plus,
        N_wadensity_minus=N_wadensity_minus,
        m_plus=m_plus_full,
        m_minus=-m_plus_full,
        time=time, ntri=ntri, k=k, l=l, dk=dk, dl=dl, domega=domega
    )

    #np.save("/work/bb1097/b383521/spectrum_ml/A2_new_" + str(group_id) + ".npy", result_for_tri)
    # result_for_tri shape: (time, ntri, l, k, m_new)

    # ── Step 5: save ──────────────────────────────────────────────────────────
    save_interpolated(
        save_path=save_path,
        intdata=result_for_tri,
        k=k, l=l, m_new=m_new,
        tri=ntri,
        varname="Wave action density",
        time=time
    )

if __name__ == "__main__":
    dir_dec="/work/bb1097/b383521/decomposed-correction-daily/"
    filename_beg="decomposed_u_gw_"#plus minus between beg and mid
    filename_end=".nc"
    filename_mid="_group-"
    path_grid="/work/bb1097/b381585/data/icon_grid_0012_R02B04_G.nc"

    #for i in range(7, 10):
    groupnr = 4
        
    #interp_path=dir_dec+"interp-alt0-regroup-"+str(groupnr)+"-test.nc" # when you want to save data
    levth=np.array([453.208, 400.000, 400.000, 400.000, 400.000, 373.779])[[0]]
    #Neig=2048
    #read=False
    av_dir="/work/bb1097/b381585/data/averaged/"
    av_filename_beg="average_"
    av_filename_end="-sphere.nc"
    save_path = "/work/bb1097/b383521/interp_wave_density_klm-daily/"+"interp-alt0-regroup-"+str(groupnr)+".nc"
    N_wadensity_plus, N_wadensity_minus, k, l, m_plus_full, ntri, omega_hat_full, time = compute_wave_action_density(dir_dec=dir_dec, filename_beg=filename_beg, groupnr=groupnr, filename_mid=filename_mid, filename_end=filename_end, av_dir=av_dir,
                                                                                                                        av_filename_beg=av_filename_beg, av_filename_end=av_filename_end, path_grid=path_grid, levth=levth)
    #l_new_3d, k_new_3d, m_new_3d, k1d, l1d, m1d, ampls1d, k_new, l_new, m_new = new_klm(N_wadensity_plus=N_wadensity_plus, N_wadensity_minus=N_wadensity_minus, k=k, l=l, m_plus=m_plus, m_minus=m_minus)
    #intdata, res_tri = multiprocess_intropolation(l_new_3d=l_new_3d, k_new_3d=k_new_3d, m_new_3d=m_new_3d, k1d=k1d, l1d=l1d, m1d=m1d, ampls1d=ampls1d, k_new=k_new, l_new=l_new, m_new=m_new, ntri=ntri,processes=128)


    jacob = jacobian(av_dir, av_filename_beg, av_filename_end, k, l, omega_hat_full, time=time, ntri=ntri)
    print(f"jacob shape: {jacob.shape}")
    N_wadensity_minus = N_wadensity_minus*jacob
    N_wadensity_plus = N_wadensity_plus*jacob
    del jacob
    m_new, result_for_tri = new_m_interp(N_wadensity_plus=N_wadensity_plus, N_wadensity_minus=N_wadensity_minus, m_plus=m_plus_full, m_minus=-m_plus_full, time=time, ntri=ntri)

    save_interpolated(save_path=save_path, intdata=result_for_tri ,k=k,l=l,m_new=m_new, tri=ntri, varname="Wave action density", time=time)
    #del  N_wadensity_plus, N_wadensity_minus, k, l, m_plus, m_minus, ntri, alt, maxind, l_new_3d, k_new_3d, m_new_3d, k1d, l1d, m1d, ampls1d, k_new, l_new, m_new, intdata, res_tri