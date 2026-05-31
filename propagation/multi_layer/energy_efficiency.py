import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Core utilities (H–Q convention, same as your TMM)
# ============================================================
def _csqrt(z): return np.lib.scimath.sqrt(z)

def _fix_branch(kz):
    kz = np.asarray(kz, dtype=complex)
    neg_im = np.imag(kz) < 0
    kz[neg_im] *= -1
    near = np.isclose(np.imag(kz), 0.0, atol=1e-12)
    kz[near & (np.real(kz) < 0)] *= -1
    return kz

def _M_tm(eps, kz, d):
    phi = kz*d
    return np.array([[np.cos(phi),           (eps/kz)*np.sin(phi)],
                     [-(kz/eps)*np.sin(phi), np.cos(phi)]], dtype=complex)

def _stack_matrix(eps_list, kz_list, d_list):
    M = np.eye(2, dtype=complex)
    for eps, kz, d in zip(eps_list, kz_list, d_list):
        if d is not None: M = _M_tm(eps, kz, d) @ M
    return M

# ============================================================
#  Baseline TMM (H–Q) — reference
# ============================================================
def tmm_tm_H_user(kx, wavelength, eps_incident, eps_layers, d_layers):
    kx = np.asarray(kx, dtype=float)
    k0 = 2*np.pi / wavelength
    assert len(eps_layers) == len(d_layers)
    assert d_layers[-1] is None
    epsN = eps_layers[-1]

    T = np.empty_like(kx, dtype=float)
    r = np.empty_like(kx, dtype=complex)
    for i, kxi in enumerate(kx):
        kz0 = _fix_branch(_csqrt(eps_incident*k0**2 - kxi**2))
        kzL = [_fix_branch(_csqrt(eps*k0**2 - kxi**2)) for eps in eps_layers]
        M = _stack_matrix(eps_layers[:-1], kzL[:-1], d_layers[:-1])
        A,B,C,D = M.ravel()
        a0 = 1j*kz0/eps_incident
        cN = 1j*kzL[-1]/epsN
        num = -((C - cN*A) + (D - cN*B)*a0)
        den =  ((C - cN*A) - (D - cN*B)*a0)
        r[i]  = num/den
        Ht = -2*a0/den
        T[i] = np.abs(Ht)**2
    return T, r

# ============================================================
#  Minimal one-pole absolute model (non-TMM):
#   Uses only: input admittances of metal+halfspace and the gap layer matrix.
#   NO stack multiplication, NO calibration.
# ============================================================
def _Yin_in_HQ(eps_m, d_m, eps_out, k, k0):
    """Input admittance seen from the gap side: Yin=(C+D*c_out)/(A+B*c_out)."""
    kz_m  = _fix_branch(_csqrt(eps_m*k0**2 - k**2))
    A,B,C,D = _M_tm(eps_m, kz_m, d_m).ravel()
    c_out = 1j*_fix_branch(_csqrt(eps_out*k0**2 - k**2))/eps_out
    return (C + D*c_out) / (A + B*c_out)

def _F_gap_plus(k, k0, eps1,eps2,eps3,eps4,eps5, d2,d3,d4):
    """Loaded-gap pole equation (H–Q, '+' form): YL m11 + YL YR m12 + m21 + YR m22."""
    YL = _Yin_in_HQ(eps2, d2, eps1, k, k0)
    YR = _Yin_in_HQ(eps4, d4, eps5, k, k0)
    kz3 = _fix_branch(_csqrt(eps3*k0**2 - k**2))
    m11,m12,m21,m22 = _M_tm(eps3, kz3, d3).ravel()
    return YL*m11 + YL*YR*m12 + m21 + YR*m22

def _newton_complex(f, z0, tol=1e-12, maxit=60):
    z = z0
    for _ in range(maxit):
        fz = f(z)
        h  = 1e-6*(1+abs(z))
        df = (f(z+h)-f(z-h))/(2*h)
        if abs(df) < 1e-22: break
        z_new = z - fz/df
        if abs(z_new - z) < tol*(1+abs(z)): return z_new
        z = z_new
    return z

def _solve_pole(k0, eps1,eps2,eps3,eps4,eps5, d2,d3,d4, kmin, kmax, nseed=60):
    seeds = np.linspace(kmin, kmax, nseed)
    best = None; score = -np.inf
    for s in seeds:
        try:
            kr = _newton_complex(lambda kk: _F_gap_plus(kk,k0,eps1,eps2,eps3,eps4,eps5,d2,d3,d4), s+1j*1e-8*s)
            if not (kmin <= np.real(kr) <= kmax): continue
            sc = np.real(kr)/(1e-12+np.imag(kr))
            if sc>score: score=sc; best=kr
        except: pass
    if best is None:
        best = 1.3*k0 + 1j*0.03*k0
    return best

def _Fprime_at(k_star, k0, eps1,eps2,eps3,eps4,eps5, d2,d3,d4):
    """Numerical derivative F'(k*) with a robust complex-step/central mix."""
    h = 1e-6*(1+abs(k_star))
    f = lambda kk: _F_gap_plus(kk,k0,eps1,eps2,eps3,eps4,eps5,d2,d3,d4)
    # central difference along real axis around complex k*
    return (f(k_star + h) - f(k_star - h)) / (2*h)

def _port_prefactor_at(k_eval, k0, eps1,eps2,eps4,eps5, d1,d2,d4):
    """
    |2 a0|^2 / |(A1+B1 a0)(A2+B2 a0)(A4+B4 c5)|^2  evaluated at a single k (use Re k*).
    """
    kz1 = _fix_branch(_csqrt(eps1*k0**2 - k_eval**2))
    kz2 = _fix_branch(_csqrt(eps2*k0**2 - k_eval**2))
    kz4 = _fix_branch(_csqrt(eps4*k0**2 - k_eval**2))
    kz5 = _fix_branch(_csqrt(eps5*k0**2 - k_eval**2))
    a0 = 1j*kz1/eps1
    c5 = 1j*kz5/eps5
    A1,B1,_,_ = _M_tm(eps1, kz1, d1).ravel()
    A2,B2,_,_ = _M_tm(eps2, kz2, d2).ravel()
    A4,B4,_,_ = _M_tm(eps4, kz4, d4).ravel()
    Pden = (A1+B1*a0)*(A2+B2*a0)*(A4+B4*c5)
    return (np.abs(2*a0)**2) / (np.abs(Pden)**2 + 1e-300)

def T_k_onepole_absolute(k, lam, eps1,eps2,eps3,eps4,eps5, d1,d2,d3,d4):
    """
    Absolute single-pole model (no calibration):
      T(k) ≈ Pref / ( |F'(k*)|^2 [ (k-k0*)^2 + (Γ/2)^2 ] )
    where k* solves F=0 in the loaded-gap model; Γ = 2 Im k*.
    """
    k0 = 2*np.pi/lam
    # 1) pole from loaded gap
    k_star = _solve_pole(k0, eps1,eps2,eps3,eps4,eps5, d2,d3,d4, kmin=1.02*k0, kmax=3.8*k0)
    k0_star = np.real(k_star); Gamma = 2.0*np.imag(k_star)
    # 2) slope at the pole
    Fp = _Fprime_at(k_star, k0, eps1,eps2,eps3,eps4,eps5, d2,d3,d4)
    # 3) port prefactor evaluated at Re k*
    Pref = _port_prefactor_at(k0_star, k0, eps1,eps2,eps4,eps5, d1,d2,d4)
    # 4) final Lorentzian
    k = np.asarray(k, dtype=float)
    denom = (k - k0_star)**2 + (Gamma/2.0)**2
    T = Pref / ( (np.abs(Fp)**2)*denom + 1e-300 )
    info = dict(k_star=k_star, k0_star=k0_star, Gamma=Gamma, Pref=Pref, Fprime=Fp)
    return T, info

# ============================================================
#  Demo with your unified parameters
# ============================================================
if __name__ == "__main__":
    # Parameters (units unified with thickness)
    wavelength = 365.0
    eps_layers = [2.0, -2.0+0.5j, 2.25, -2.0+0.5j, 2.0]   # eps1..eps5
    d_layers   = [30.0, 20.0,     30.0, 20.0,     None]   # d1..d4, last None
    eps1, eps2, eps3, eps4, eps5 = eps_layers
    d1,  d2,  d3,  d4,  _        = d_layers

    k0 = 2*np.pi/wavelength
    kx = np.linspace(1.0*k0, 4.0*k0, 2401)

    # Reference: TMM (H–Q)
    T_tmm, _ = tmm_tm_H_user(kx, wavelength, eps1, eps_layers, d_layers)

    # One-pole absolute model (no calibration)
    T_abs, info = T_k_onepole_absolute(kx, wavelength, eps1,eps2,eps3,eps4,eps5, d1,d2,d3,d4)

    print(f"Pole k*/k0 ~ {info['k_star']/k0:.4f} (Re={info['k0_star']/k0:.4f}, Im={np.imag(info['k_star'])/k0:.4e})")
    print(f"|F'(k*)| ~ {np.abs(info['Fprime']):.4e}, Pref ~ {info['Pref']:.4e}")

    # Plot
    plt.figure(figsize=(8,5.6))
    plt.semilogy(kx/k0, T_tmm + 1e-300, '--', label='TMM (H–Q)')
    plt.semilogy(kx/k0, T_abs + 1e-300,  '-',  label='One-pole absolute (no calib)')
    plt.axvline(info['k0_star']/k0, ls=':', color='k', alpha=0.6, label='Re k* (loaded gap)')
    plt.xlabel(r'$k_\parallel/k_0$'); plt.ylabel(r'$|H_{\rm trans}/H_{\rm inc}|^2$')
    plt.title('k-space TM transmission: one-pole absolute vs TMM')
    plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
