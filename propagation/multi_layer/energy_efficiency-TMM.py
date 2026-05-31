import numpy as np

# ---------------- Core utilities ----------------
def _csqrt(z):
    return np.lib.scimath.sqrt(z)

def _fix_branch(kz):
    """Choose physical branch: Im(kz) >= 0; if ~real then Re(kz) >= 0."""
    kz = np.asarray(kz, dtype=complex)
    neg_im = np.imag(kz) < 0
    kz[neg_im] *= -1
    near_real = np.isclose(np.imag(kz), 0.0, atol=1e-12)
    kz[near_real & (np.real(kz) < 0)] *= -1
    return kz

def _M_tm(eps, kz, d):
    """
    H–Q convention layer matrix over thickness d:
      S = [H; Q],  Q = (eps/(i kz)) dH/dz
      M = [[cos(phi),         (eps/kz) sin(phi)],
           [-(kz/eps) sin(phi), cos(phi)]],  phi = kz d
    """
    phi = kz*d
    return np.array([[np.cos(phi),           (eps/kz)*np.sin(phi)],
                     [-(kz/eps)*np.sin(phi), np.cos(phi)]], dtype=complex)

def _stack_matrix(eps_list, kz_list, d_list):
    """Product of finite layers (left→right). Last semi-infinite (d=None) excluded."""
    M = np.eye(2, dtype=complex)
    for eps, kz, d in zip(eps_list, kz_list, d_list):
        if d is not None:
            M = _M_tm(eps, kz, d) @ M
    return M

# ---------------- Minimal, reusable TMM (H–Q convention) ----------------
def tmm_tm_H_user(kx, wavelength, eps_incident, eps_layers, d_layers):
    """
    Minimal TMM for TM polarization in the H–Q convention.
    Returns:
      T (ndarray): |H_trans/H_inc|^2 evaluated at the right-interface (z = sum d_i)
      r (ndarray): complex reflection coefficient at z=0 (from incident side)
    Inputs:
      kx         : ndarray of transverse wave numbers (same length unit as 'wavelength'/thickness)
      wavelength : scalar, same length unit as thicknesses
      eps_incident: scalar permittivity on incident side (left semi-infinite medium)
      eps_layers : list/tuple of permittivities, including the right semi-infinite last one
                   e.g. [eps1, eps2, eps3, eps4, eps5] with eps5 semi-infinite
      d_layers   : list/tuple of thicknesses for the same list, last one must be None
                   e.g. [d1, d2, d3, d4, None]
    Convention & checks:
      - Uses S=[H;Q], Q=(eps/(i kz)) dH/dz
      - Port admittances: a0 = i kz0/eps_incident, cN = i kzN/epsN
      - All lengths and wavelength are in the same unit (no SI enforcement here).
    """
    kx = np.asarray(kx, dtype=float)
    k0 = 2*np.pi / wavelength

    # sanity
    assert len(eps_layers) == len(d_layers), "eps_layers and d_layers must have same length"
    assert d_layers[-1] is None, "The last layer must be semi-infinite (d=None)"
    epsN = eps_layers[-1]

    T = np.empty_like(kx, dtype=float)
    r = np.empty_like(kx, dtype=complex)

    for i, kxi in enumerate(kx):
        # kz in incident and each layer (physical branch)
        kz0 = _fix_branch(_csqrt(eps_incident*k0**2 - kxi**2))
        kzL = [_fix_branch(_csqrt(eps*k0**2 - kxi**2)) for eps in eps_layers]

        # global matrix of finite layers
        M = _stack_matrix(eps_layers[:-1], kzL[:-1], d_layers[:-1])
        A,B,C,D = M.ravel()

        # port "admittances" (H–Q convention)
        a0 = 1j*kz0/eps_incident
        cN = 1j*kzL[-1]/epsN

        # reflection (same algebra as your MultiLayerTM)
        num = -((C - cN*A) + (D - cN*B)*a0)
        den =  ((C - cN*A) - (D - cN*B)*a0)
        r[i]  = num/den

        # transmitted H at the right interface (absolute) — direct closed form
        # Ht = -2*a0 / [(C - cN*A) - (D - cN*B)*a0]
        Ht = -2*a0 / den
        T[i] = np.abs(Ht)**2

    return T, r

# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Unified parameters (same unit for wavelength and thickness)
    wavelength = 365.0
    eps_layers = [2.0, -2.0+0.5j, 2.25, -2.0+0.5j, 2.0]    # eps1..eps5
    d_layers   = [30.0, 20.0, 30.0, 20.0, None]            # d1..d4, last None
    eps_incident = eps_layers[0]

    k0 = 2*np.pi/wavelength
    kx = np.linspace(1.0*k0, 4.0*k0, 2001)                 # scan k>k0

    T, r = tmm_tm_H_user(kx, wavelength, eps_incident, eps_layers, d_layers)
    # 现在 T 即为右界面的 |H_trans/H_inc|^2（与你的 MultiLayerTM 相同口径）
    # visulization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.plot(kx/k0, T, label='Transmitted |H|^2')
    plt.xlabel(r'$k_x / k_0$')
    plt.ylabel(r'')
    plt.yscale('log')
    plt.title('TMM TM Transmission Spectrum (H–Q convention)')
    plt.legend()
    plt.grid()
    plt.show()
