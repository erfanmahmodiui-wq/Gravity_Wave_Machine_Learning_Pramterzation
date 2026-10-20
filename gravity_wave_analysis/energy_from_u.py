import numpy as np
import constants



def Ek_from_u(omega_hat,k,l,f,size_u):
    #Horizontal kinetic energy
    l2=l[np.newaxis,:,np.newaxis]**2
    k2=k[np.newaxis,np.newaxis,:]**2
    with np.errstate(invalid='ignore'):
        res=(1+(l2*omega_hat**2+k2*f**2)/(k2*omega_hat**2+l2*f**2))*size_u**2
    return res


def E_from_u(omega_hat,k,l,f,averages,size_u):
    #Not testing if the points have m^2>0, should be filtered elsewhere...
    #Functionaly not tested
    l2=l[np.newaxis,:,np.newaxis]**2
    k2=k[np.newaxis,np.newaxis,:]**2
    o2=omega_hat**2
    f2=np.array([f**2])[np.newaxis,np.newaxis,np.newaxis]
    mult=size_u/2/(k2*o2+l2*f2)
    t1=(k2+l2)*(o2+f2)
    t2=averages['rho0']**2/averages['cs2']/averages['rhobar']**2*(o2-f2)**2
    t3_mult=(o2-f2)/averages['cs2']
    t3_nom=(o2-f2-averages['cs2']*(k2+l2))**2
    t3_den=(k2+l2)*averages['cs2']-constants.R*constants.g/averages['H']*(o2-f2)/(averages['N2']-o2)
    t3=t3_mult*t3_nom/t3_den
    res= mult*(t1+t2+t3)
    assert (res>0).all(), print("Energy is negative, some problem.")
    return res/omega_hat

def N_from_u(omega_hat,k,l,f,averages,size_u):
    #Not testing if the points have m^2>0, should be filtered elsewhere...
    #For data of the shape (nz,ntri,nomega,nl,nk) 
    k4d=k[np.newaxis,:,np.newaxis,np.newaxis,:]
    l4d=l[np.newaxis,:,np.newaxis,:,np.newaxis]
    size_u[np.where(np.logical_not(np.isfinite(k4d)))]=np.nan
    size_u[np.where(np.logical_not(np.isfinite(l4d)))]=np.nan
    size_u[np.where(np.logical_not(np.isfinite(omega_hat)))]=np.nan
    #print("Number of infinite values in u inside calculation",np.sum(np.logical_not(np.isfinite(size_u))))
    o2=omega_hat**2
    f2=np.array(f**2)[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]*np.ones_like(omega_hat)
    rhosh=averages['rho'][:,:,np.newaxis,np.newaxis,np.newaxis]
    N2sh=averages['N2'][:,:,np.newaxis,np.newaxis,np.newaxis]
    cs2sh=averages['cs2'][:,:,np.newaxis,np.newaxis,np.newaxis]
    Hsh=averages['H'][:,:,np.newaxis,np.newaxis,np.newaxis]
    #res = N_from_u_sh(omega_hat, size_u, k4d, l4d, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    res = N_from_vars_sh(omega_hat, size_u, k4d, l4d, o2, f2, rhosh, N2sh, cs2sh, Hsh,"plus")
    return res

def N_from_u_sh(omega_hat, size_u, k4d2, l4d2, o2, f2, rhosh, N2sh, cs2sh, Hsh,k,l,rho0=None):
    mult=size_u**2/4/(k4d2*o2+l4d2*f2)*rhosh
    t1=(k4d2+l4d2)*(o2+f2)
    t3_mult=(N2sh-o2)/cs2sh**2
    t3_nom=(o2-f2-cs2sh*(k4d2+l4d2))**2
    t3_den=-(constants.R/constants.cp**2*constants.cv/Hsh**2)+(k4d2+l4d2)*(N2sh-o2)/(o2-f2)
    t3=t3_mult*t3_nom/t3_den
    res= mult*(t1+t3)
    #Remove b
    if not rho0==None:
        m=m_gw(omega_hat,np.sqrt(k4d2),np.sqrt(l4d2),N2sh,f2,Hsh,"plus",cs2sh)
        H_theta=constants.cp/constants.R*Hsh
        p=p_from_u(omega_hat,k,l,np.sqrt(f2),rho0,size_u)
        term=p*p.conjugate()/cs2sh/rhosh**2
        res=res-term/2
    res[((k4d2+l4d2)*(N2sh-o2)/(o2-f2)-1/4/Hsh**2)<0]=np.nan #Where there cannot be gravity waves
    assert (res[np.isfinite(res)]>=0).all(), print("Energy is negative, some problem.")
    return 2*res/omega_hat #Multiply by two to sum the two gravity wave modes

def N_from_vars_sh(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign,cs2sh)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    #res=k*N*cgz*o2/(o2-f2)
    #res=-k*m*omega_hat*(N2sh-o2)/(N2sh-f2)/(m**2+1/4/Hsh**2)*N #Containing m
    H_theta=constants.cp/constants.R*Hsh
    v=v_from_u(omega_hat,k,l,np.sqrt(f2),size_u)
    w=w_from_u(omega_hat,k,l,m,np.sqrt(f2),N2sh,Hsh,H_theta,cs2sh,size_u)
    b=b_from_u(omega_hat,k,l,m,np.sqrt(f2),N2sh,Hsh,H_theta,cs2sh,size_u)
    E=(size_u**2+v*v.conjugate()+w*w.conjugate()+b*b.conjugate()/N2sh)*rhosh/4
    return E/omega_hat

def Etot_from_vars_sh(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign,cs2sh)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    #res=k*N*cgz*o2/(o2-f2)
    #res=-k*m*omega_hat*(N2sh-o2)/(N2sh-f2)/(m**2+1/4/Hsh**2)*N #Containing m
    H_theta=constants.cp/constants.R*Hsh
    v=v_from_u(omega_hat,k,l,np.sqrt(f2),size_u)
    w=w_from_u(omega_hat,k,l,m,np.sqrt(f2),N2sh,Hsh,H_theta,cs2sh,size_u)
    b=b_from_u(omega_hat,k,l,m,np.sqrt(f2),N2sh,Hsh,H_theta,cs2sh,size_u)
    E=(size_u**2+v*v.conjugate()+w*w.conjugate()+b*b.conjugate()/N2sh)*rhosh/4
    return E

def Etot_from_u_lev(omega_hat,k,l,f,averages,size_u,sign="plus"):
    #Not testing if the points have m^2>0, should be filtered elsewhere...
    #For data of the shape (ntri,nomega,nl,nk) 
    l3d=l[np.newaxis,:,np.newaxis]
    k3d=k[np.newaxis,np.newaxis,:]
    o2=omega_hat**2
    f2=np.array(f**2)
    rhosh=np.transpose(averages['rho'])
    N2sh=np.transpose(averages['N2'])
    cs2sh=np.transpose(averages['cs2'])
    Hsh=np.transpose(averages['H'])
    res=Etot_from_vars_sh(omega_hat, size_u, k3d, l3d, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign) #Safer for catching bugs
    #res=N_from_vars_sh_bouss(omega_hat, size_u, k3d, l3d, o2, f2, rhosh, N2sh, cs2sh, Hsh,"plus") #Boussinesque version
    return res


def N_from_vars_sh_bouss(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    #Same as flux_zx_noint_sh, only less numerical operations...
    m=m_gw_bouss(omega_hat,k,l,N2sh,f2,sign)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    #res=k*N*cgz*o2/(o2-f2)
    #res=-k*m*omega_hat*(N2sh-o2)/(N2sh-f2)/(m**2+1/4/Hsh**2)*N #Containing m
    v=v_from_u_bouss(omega_hat,k,l,np.sqrt(f2),size_u)
    w=w_from_u_bouss(omega_hat,k,l,m,np.sqrt(f2),N2sh,size_u)
    b=b_from_u_bouss(omega_hat,k,l,m,np.sqrt(f2),N2sh,size_u)
    E=(size_u**2+v*v.conjugate()+w*w.conjugate()+b*b.conjugate()/N2sh)*rhosh/4
    return 2*E/omega_hat

def N_from_u_lev(omega_hat,k,l,f,averages,size_u,sign="plus"):
    #Not testing if the points have m^2>0, should be filtered elsewhere...
    #For data of the shape (ntri,nomega,nl,nk) 
    l3d=l[np.newaxis,:,np.newaxis]
    k3d=k[np.newaxis,np.newaxis,:]
    o2=omega_hat**2
    f2=np.array(f**2)
    rhosh=np.transpose(averages['rho'])
    N2sh=np.transpose(averages['N2'])
    cs2sh=np.transpose(averages['cs2'])
    Hsh=np.transpose(averages['H'])
    rho0=np.transpose(averages['rho0'])
    res = N_from_u_sh(omega_hat, size_u, k3d**2, l3d**2, o2, f2, rhosh, N2sh, cs2sh, Hsh,k,l,rho0) #old
    #res=N_from_vars_sh(omega_hat, size_u, k3d, l3d, o2, f2, rhosh, N2sh, cs2sh, Hsh,"plus") #Safer for catching bugs
    #res=N_from_vars_sh_bouss(omega_hat, size_u, k3d, l3d, o2, f2, rhosh, N2sh, cs2sh, Hsh,"plus") #Boussinesque version
    return res

def cgx_sh(omega_hat,k,l,m,N2,f2,Hsh):
    #Zonal component of the group velocity
    mult1=k*(N2-f2)/omega_hat
    mult2=(m**2+1/4/Hsh**2)/(k**2+l**2+1/4/Hsh**2)**2
    return mult1*mult2

def cgx_sh2(omega_hat,k,l,m,N2,f2,Hsh):
    #Zonal component of the group velocity
    #Simplified formula
    return k/(k**2+l**2)/omega_hat*(N2-omega_hat**2)*(omega_hat**2-f2)/(N2-f2)

def cgx_sh_gpt(omega_hat, k, l, m, N2, f2, Hsh):
    # Ensure omega_hat is not zero to avoid division by zero
    if np.any(np.isclose(omega_hat, 0)):
        raise ValueError("All values of omega_hat must not be zero.")

    # Calculate the denominator once
    denominator = k**2 + l**2 + 1/(4 * Hsh**2)
    
    # Use np.clip to avoid division by very small numbers
    safe_denominator = np.clip(denominator, 1e-10, None)

    # Calculate the group velocity directly
    group_velocity = (k * (N2 - f2) / omega_hat) * ((m**2 + 1/(4 * Hsh**2)) / safe_denominator**2)

    return group_velocity

def cgy_sh(omega_hat,k,l,m,N2,f2,Hsh):
    #Meridional component of the group velocity
    mult1=l*(N2-f2)/omega_hat
    mult2=(m**2+1/4/Hsh**2)/(k**2+l**2+1/4/Hsh**2)**2
    return mult1*mult2

def cgy_sh2(omega_hat,k,l,m,N2,f2,Hsh):
    #Zonal component of the group velocity
    #Simplified formula
    return l/(k**2+l**2)/omega_hat*(N2-omega_hat**2)*(omega_hat**2-f2)/(N2-f2)

def cgz_sh(omega_hat,k,l,m,N2,f2,Hsh):
    #Vertical component of the group velocity
    mult1=-m*(N2-f2)/omega_hat
    mult2=(k**2+l**2)/(k**2+l**2+m**2+1/4/Hsh**2)**2
    return mult1*mult2

def cgz_sh2(omega_hat,k,l,m,N2,f2,Hsh):
    #Zonal component of the group velocity
    #Simplified formula
    return -m/(k**2+l**2)/omega_hat*((f2-omega_hat**2)**2)/(N2-f2)

def m_gw(omega_hat,k,l,N2,f2,H,sign,cs2sh):
    #Compute vertical wavenumber from dispersion relation for gravity waves
    #Returns the root of m^2 with the given sign
    kh2=k**2+l**2
    mult=sign2mult(sign)
    with np.errstate(invalid='ignore'):
        if type(cs2sh) is np.ndarray:
            return mult*np.sqrt(kh2*(omega_hat**2-N2)/(f2-omega_hat**2)-1/(4*H**2))
        else:
            return mult*np.sqrt(kh2*(omega_hat**2-N2)/(f2-omega_hat**2)-1/(4*H**2)+omega_hat**2/cs2sh)
    
def m_gw_bouss(omega_hat,k,l,N2,f2,sign):
    #Compute vertical wavenumber from dispersion relation for gravity waves
    #Returns the root of m^2 with the given sign
    kh2=k**2+l**2
    mult=sign2mult(sign)
    with np.errstate(invalid='ignore'):
        return mult*np.sqrt(kh2*(omega_hat**2-N2)/(f2-omega_hat**2))

def dmdomega(omega_hat,m,k,l,o2,f2,N2sh):
    return -(k**2+l**2)*omega_hat/m*(N2sh-f2)/(o2-f2)


def flux_zx_noint_sh(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    cgz=cgz_sh(omega_hat,k,l,m,N2sh,f2,Hsh)
    N[o2==f2]=np.nan
    res=k*N*cgz*o2/(o2-f2)
    return res/rhosh

def flux_zx_noint_sh2(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    #Mid frequency approximation
    N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    N[o2==f2]=np.nan
    #res=k*N*cgz*o2/(o2-f2)
    res=-np.sqrt(k**2+l**2)/m*N*omega_hat 
    return res/rhosh

def flux_zx_noint_sh3(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    #Same as flux_zx_noint_sh, only less numerical operations...
    N=N_from_vars_sh(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign)
    #N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    N[o2==f2]=np.nan
    #res=k*N*cgz*o2/(o2-f2)
    #res=-k*m*omega_hat*(N2sh-o2)/(N2sh-f2)/(m**2+1/4/Hsh**2)*N #Containing m
    res=-k*m*omega_hat/(k**2+l**2)*(o2-f2)/(N2sh-f2)*N #Without m^2
    #dmdo=dmdomega(omega_hat,m,k,l,o2,f2,N2sh)
    return res/rhosh#*dmdo

def flux_zx_uw(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    #Same as flux_zx_noint_sh, only less numerical operations...
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign,cs2sh)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    #res=k*N*cgz*o2/(o2-f2)
    #res=-k*m*omega_hat*(N2sh-o2)/(N2sh-f2)/(m**2+1/4/Hsh**2)*N #Containing m
    H_theta=constants.cp/constants.R*Hsh
    w=w_from_u(omega_hat,k,l,m,np.sqrt(f2),N2sh,Hsh,H_theta,cs2sh,size_u)
    return w*size_u

def flux_zx_uw_test(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    #Same as flux_zx_noint_sh, only less numerical operations...
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign,cs2sh)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    #res=k*N*cgz*o2/(o2-f2)
    #res=-k*m*omega_hat*(N2sh-o2)/(N2sh-f2)/(m**2+1/4/Hsh**2)*N #Containing m
    H_theta=constants.cp/constants.R*Hsh
    w=w_from_u_test(omega_hat,k,l,m,np.sqrt(f2),N2sh,Hsh,H_theta,cs2sh,size_u)
    return w*size_u

def flux_zx_fromN_bouss(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    #Same as flux_zx_noint_sh, only less numerical operations...
    N=N_from_vars_sh_bouss(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign)
    #N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw_bouss(omega_hat,k,l,N2sh,f2,sign)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    N[o2==f2]=np.nan
    #res=k*N*cgz*o2/(o2-f2)
    #res=-k*m*omega_hat*(N2sh-o2)/(N2sh-f2)/(m**2+1/4/Hsh**2)*N #Containing m
    res=-k*m*omega_hat/(k**2+l**2)*(o2-f2)/(N2sh-f2)*N #Without m^2
    #dmdo=dmdomega(omega_hat,m,k,l,o2,f2,N2sh)
    return res/rhosh#*dmdo

def flux_zx_uw_bouss(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    #Same as flux_zx_noint_sh, only less numerical operations...
    m=m_gw_bouss(omega_hat,k,l,N2sh,f2,sign)
    #cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    #res=k*N*cgz*o2/(o2-f2)
    #res=-k*m*omega_hat*(N2sh-o2)/(N2sh-f2)/(m**2+1/4/Hsh**2)*N #Containing m
    w=w_from_u_bouss(omega_hat,k,l,m,np.sqrt(f2),N2sh,size_u)
    return w*size_u

def flux_zy_noint_sh(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    cgz=cgz_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    return l*N*cgz/(1-f2/o2)/rhosh

def sign2mult(sign):
    if sign=='minus':
        mult=-1
    else:
        mult=1
    return mult

def flux_xy_noint_sh(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    cgx=cgx_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    return k*l*N*cgx/rhosh


def flux_xx_noint_sh(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    cgx=cgx_sh_gpt(omega_hat,k,l,m,N2sh,f2,Hsh)
    cgy=cgy_sh(omega_hat,k,l,m,N2sh,f2,Hsh)
    #dmdo=dmdomega(omega_hat,m,k,l,o2,f2,N2sh)
    return (k*N*cgx/(1-f2/o2)+l*N*cgy/(o2/f2-1))/rhosh#*dmdo

def flux_xx_noint_sh2(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    #cgx=cgx_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    #cgy=cgy_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    #dmdo=dmdomega(omega_hat,m,k,l,o2,f2,N2sh)
    res=k**2/(m**2+1/4/Hsh**2)*N/omega_hat#*dmdo
    return res/rhosh

def flux_xx_noint_simplified(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    #m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    #dmdo=dmdomega(omega_hat,m,k,l,o2,f2,N2sh)
    return size_u**2

def flux_yy_noint_sh(omega_hat, size_u, k, l, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign):
    N=N_from_u_sh(omega_hat, size_u, k**2, l**2, o2, f2, rhosh, N2sh, cs2sh, Hsh)
    m=m_gw(omega_hat,k,l,N2sh,f2,Hsh,sign)
    cgx=cgx_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    cgy=cgy_sh2(omega_hat,k,l,m,N2sh,f2,Hsh)
    return l*N*cgy/(1-f2/o2)+k*N*cgx/(o2/f2-1)/rhosh

def switch_flux(flux):
    return {
        'xx':flux_xx_noint_simplified,
        #'xx':flux_xx_noint_sh2,
        #'xx':flux_xx_noint_sh,
        'yx':flux_xy_noint_sh,
        'zx':flux_zx_uw,#flux_zx_fromN_bouss,#flux_zx_uw_bouss,#flux_zx_noint_sh3, #flux_zx_uw, also sh3 and sh,uw_test
        'xy':flux_xy_noint_sh,
        'yy':flux_yy_noint_sh,
        'zy':flux_zy_noint_sh
    }[flux]

def flux_from_u_lev(omega_hat,k,l,f,averages,size_u,flux,sign):
    #For data of the shape (ntri,nomega,nl,nk) 
    #Flux is one of the combinations xx,yx,zx,xy,yy,zy
    l3d=l[np.newaxis,:,np.newaxis]
    k3d=k[np.newaxis,np.newaxis,:]
    o2=omega_hat**2
    f2=np.array(f**2)
    rhosh=np.transpose(averages['rho'])
    N2sh=np.transpose(averages['N2'])
    cs2sh=np.transpose(averages['cs2'])
    Hsh=np.transpose(averages['H'])
    func=switch_flux(flux)
    res = func(omega_hat, size_u, k3d, l3d, o2, f2, rhosh, N2sh, cs2sh, Hsh,sign)
    return res

def uv_from_u(omega_hat,k,l,f,size_u):
    #Computes the size of u_hat*v_hat
    l3d=l[np.newaxis,:,np.newaxis]
    k3d=k[np.newaxis,np.newaxis,:]
    with np.errstate(invalid='ignore'):
        #fr=(l3d*omega_hat-1j*k3d*f)/(k3d*omega_hat-1j*l3d*f)
        #size_fr=np.abs(fr)
        
        size_fr=np.sqrt((l3d**2*omega_hat**2+k3d**2*f**2)/(k3d**2*omega_hat**2+l3d**2*f**2))
        #Correct the values on positions that are not so well defined:
        ml=np.where(l3d==0)
        size_fr[ml]=np.abs(f/omega_hat[ml])
        mk=np.where(k3d==0)
        size_fr[mk]=np.abs(omega_hat[mk]/f)
    return size_fr*size_u**2

def v_from_u(omega_hat,k3d,l3d,f,u):
    #Same for Boussinesque and general
    #l3d=l[np.newaxis,:,np.newaxis]
    #k3d=k[np.newaxis,np.newaxis,:]
    with np.errstate(invalid='ignore'):
        v=(l3d*omega_hat-1j*k3d*f)/(k3d*omega_hat+1j*l3d*f)*u
    return v


def b_from_u(omega_hat,k3d,l3d,m,f,N2,H,H_theta,cs2,u):
    #Eq. 10.108 from Atmospheric dynamics
    #l3d=l[np.newaxis,:,np.newaxis]
    #k3d=k[np.newaxis,np.newaxis,:]
    den=N2*(omega_hat**2-f**2-cs2*(k3d**2+l3d**2))
    nom=cs2*(k3d*omega_hat+1j*l3d*f)*(1/H_theta-1/2/H+1j*m)
    with np.errstate(invalid='ignore'):
        res=den/nom*u
    return res

def p_from_u(omega_hat,k,l,f,rho_0,u):
    l3d=l[np.newaxis,:,np.newaxis]
    k3d=k[np.newaxis,np.newaxis,:]
    nom=rho_0*(omega_hat**2-f**2)
    den=k3d*omega_hat+1j*l3d*f
    with np.errstate(invalid='ignore'):
        res=nom/den*u
    return res

def w_from_u(omega_hat,k,l,m,f,N2,H,H_theta,cs2,u):
    b=b_from_u(omega_hat,k,l,m,f,N2,H,H_theta,cs2,u)
    return 1j*omega_hat/N2*b  

def v_from_u_bouss(omega_hat,k3d,l3d,f,u):
    #Same for Boussinesque and general
    v=v_from_u(omega_hat,k3d,l3d,f,u)
    return v

def b_from_u_bouss(omega_hat,k3d,l3d,m,f,N2,u):
    #Eq. 10.108 from Atmospheric dynamics
    #l3d=l[np.newaxis,:,np.newaxis]
    #k3d=k[np.newaxis,np.newaxis,:]
    den=1j*N2*(k3d**2+l3d**2)
    nom=(k3d*omega_hat+1j*l3d*f)*m
    with np.errstate(invalid='ignore'):
        res=den/nom*u
    return res

def b_from_u_Honly(omega_hat,k3d,l3d,m,f,N2,H,H_theta,cs2,u):
    #Eq. 10.108 from Atmospheric dynamics
    #l3d=l[np.newaxis,:,np.newaxis]
    #k3d=k[np.newaxis,np.newaxis,:]
    den=1j*N2*(k3d**2+l3d**2)
    nom=(k3d*omega_hat+1j*l3d*f)*(1/H_theta-1/2/H+1j*m)
    with np.errstate(invalid='ignore'):
        res=den/nom*u
    return res

def b_from_u_cs2_only(omega_hat,k3d,l3d,m,f,N2,H,H_theta,cs2,u):
    #Eq. 10.108 from Atmospheric dynamics
    #l3d=l[np.newaxis,:,np.newaxis]
    #k3d=k[np.newaxis,np.newaxis,:]
    den=N2*(omega_hat**2-f**2-cs2*(k3d**2+l3d**2))
    nom=cs2*(k3d*omega_hat+1j*l3d*f)*1j*m
    with np.errstate(invalid='ignore'):
        res=den/nom*u
    return res

def w_from_u_test(omega_hat,k3d,l3d,m,f,N2,H,H_theta,cs2,u):
    b=b_from_u_cs2_only(omega_hat,k3d,l3d,m,f,N2,H,H_theta,cs2,u)
    return 1j*omega_hat/N2*b  

def w_from_u_bouss(omega_hat,k,l,m,f,N2,u):
    b=b_from_u_bouss(omega_hat,k,l,m,f,N2,u)
    return 1j*omega_hat/N2*b  