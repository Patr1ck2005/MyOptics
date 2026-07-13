"""
Angle-resolved T/R spectra for: Air | ZnO(40nm) | Ag(10nm) | ZnO(10nm) | Film(100μm) | Air

Computes T(λ) and R(λ) at incidence angles 0, 20, 40, 60, 80° in air.
The 100μm organic film is treated incoherently (thick-substrate approximation):
  - Coating (ZnO/Ag/ZnO) is computed coherently via TMM
  - The 100μm film is added via the closed-form incoherent multiple-reflection formula

Both s and p polarizations are computed; the unpolarized result is their average.

Outputs (under rsl/angle_scan_spectra/):
  - spectrum_angle_<deg>.csv  : per-angle spectrum (wl, Ts, Tp, T_unpol, Rs, Rp, R_unpol)
  - all_angles.csv            : long-format combined table
  - spectra_plot.png          : T(λ) and R(λ) for all angles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from optimize_transmittance import build_nk_interpolators

OUT_DIR = Path(__file__).parent.parent / 'rsl' / 'angle_scan_spectra'

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
#  1. Coherent TMM at arbitrary angle (kx-driven)
# ============================================================
def _kz_physical(kz):
    """Pick the physical branch: Im(kz) >= 0; if ~real, Re(kz) >= 0."""
    if np.imag(kz) < 0:
        return -kz
    if np.isclose(np.imag(kz), 0.0, atol=1e-14) and np.real(kz) < 0:
        return -kz
    return kz


def tmm_kx(wl_um, N_layers, d_nm_list, N_inc, N_exit, kx, pol='s'):
    """
    Coherent TMM at fixed transverse wavevector kx (in 1/nm).

    Parameters
    ----------
    wl_um : float
        Wavelength in μm.
    N_layers : list of complex
        Complex refractive indices of finite layers, incident->exit order.
    d_nm_list : list of float
        Thicknesses (nm) of each finite layer.
    N_inc, N_exit : complex
        Complex refractive indices of incident and exit (semi-infinite) media.
    kx : float
        Transverse wavevector (1/nm), conserved across all layers.
    pol : 's' or 'p'

    Returns
    -------
    T, R : float
        Transmittance (Poynting-flux ratio) and reflectance (|r|^2).
    """
    k0 = 2.0 * np.pi / (wl_um * 1000.0)  # 1/nm

    def kz_of(N_j):
        return _kz_physical(np.lib.scimath.sqrt(N_j**2 * k0**2 - kx**2))

    kz_inc = kz_of(N_inc)
    kz_exit = kz_of(N_exit)
    kz_layers = [kz_of(N_j) for N_j in N_layers]

    if pol == 's':
        Y_inc = kz_inc / k0
        Y_exit = kz_exit / k0
        Y_layers = [kzj / k0 for kzj in kz_layers]
    elif pol == 'p':
        Y_inc = N_inc**2 * k0 / kz_inc
        Y_exit = N_exit**2 * k0 / kz_exit
        Y_layers = [Nj**2 * k0 / kzj for Nj, kzj in zip(N_layers, kz_layers)]
    else:
        raise ValueError(f"pol must be 's' or 'p', got {pol!r}")

    M = np.eye(2, dtype=complex)
    for Y_j, dj, kzj in zip(Y_layers, d_nm_list, kz_layers):
        delta = kzj * dj
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        Mj = np.array([[cos_d, -1j * sin_d / Y_j],
                       [-1j * Y_j * sin_d, cos_d]], dtype=complex)
        M = M @ Mj

    m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    Y0, Ys = Y_inc, Y_exit
    denom = Y0 * m11 + Y0 * Ys * m12 + m21 + Ys * m22
    r = (Y0 * m11 + Y0 * Ys * m12 - m21 - Ys * m22) / denom
    t = 2.0 * Y0 / denom

    reY0 = np.real(Y0)
    T = (np.real(Ys) / reY0) * np.abs(t)**2 if reY0 > 0 else 0.0
    R = np.abs(r)**2
    return float(T), float(R)


# ============================================================
#  2. Incoherent thick-film correction
# ============================================================
def incoherent_tr(T_front, R_front, T_internal, R_internal, R_back, tau):
    """
    Closed-form incoherent slab formula for a thick substrate between coating and air.

    Forward (T_total): T_front * tau * (1 - R_back) / (1 - R_back * tau^2 * R_internal)
    Reverse (R_total): R_front + T_front * tau^2 * R_back * T_internal
                                  / (1 - R_back * tau^2 * R_internal)

    Reciprocity gives T_front == T_internal for passive coatings at the same kx,
    but we use both explicitly so the formula stays correct if they drift.
    """
    denom = 1.0 - R_back * tau**2 * R_internal
    T_total = T_front * tau * (1.0 - R_back) / denom
    R_total = R_front + T_front * tau**2 * R_back * T_internal / denom
    return T_total, R_total


# ============================================================
#  3. Per-(wavelength, angle, polarization) computation
# ============================================================
def compute_at(wl_um, theta_deg, pol,
               n_ZnO, k_ZnO, n_Ag, k_Ag, n_film, k_film,
               d_ZnO1=40.0, d_Ag=10.0, d_ZnO2=10.0, d_film_nm=100000.0):
    """Compute (T_total, R_total) for one (wl, angle, pol) combination."""
    k0 = 2.0 * np.pi / (wl_um * 1000.0)  # 1/nm
    kx = k0 * np.sin(np.deg2rad(theta_deg))  # N_inc = 1 (air)

    N_inc = complex(1.0, 0.0)
    N_air = complex(1.0, 0.0)
    N_film = complex(n_film, k_film)

    # Coating from air side: ZnO1(40) | Ag(10) | ZnO2(10), exit = semi-inf film
    N_layers_fwd = [complex(n_ZnO, k_ZnO), complex(n_Ag, k_Ag), complex(n_ZnO, k_ZnO)]
    d_fwd = [d_ZnO1, d_Ag, d_ZnO2]
    T_front, R_front = tmm_kx(wl_um, N_layers_fwd, d_fwd, N_inc, N_film, kx, pol)

    # Coating from film side (reversed): ZnO2(10) | Ag(10) | ZnO1(40), exit = air
    N_layers_bwd = [complex(n_ZnO, k_ZnO), complex(n_Ag, k_Ag), complex(n_ZnO, k_ZnO)]
    d_bwd = [d_ZnO2, d_Ag, d_ZnO1]
    T_internal, R_internal = tmm_kx(wl_um, N_layers_bwd, d_bwd, N_film, N_air, kx, pol)

    # Bare film-air interface from film side (no coating layers)
    _, R_back = tmm_kx(wl_um, [], [], N_film, N_air, kx, pol)

    # Single-pass intensity transmission through the 100μm film at this kx
    kz_film = _kz_physical(np.lib.scimath.sqrt(N_film**2 * k0**2 - kx**2))
    tau = float(np.exp(-2.0 * np.imag(kz_film) * d_film_nm))

    return incoherent_tr(T_front, R_front, T_internal, R_internal, R_back, tau)


# ============================================================
#  4. Main
# ============================================================
def main():
    print("=" * 64)
    print("Angle-resolved T/R: Air|ZnO40|Ag10|ZnO10|Film100μm|Air")
    print("=" * 64)

    angles_deg = [0, 20, 40, 60, 80]
    wl_grid = np.linspace(0.30, 2.00, 171)  # 0.3-2.0 μm, 10 nm steps

    print(f"\n[1] Wavelength grid: {wl_grid[0]:.2f}-{wl_grid[-1]:.2f} μm, "
          f"{len(wl_grid)} points")
    print(f"    Angles: {angles_deg}")

    print("\n[2] Loading material data...")
    n_ZnO, k_ZnO = build_nk_interpolators('ZnO', wl_grid)
    n_Ag, k_Ag = build_nk_interpolators('Ag', wl_grid)
    n_film, k_film = build_nk_interpolators('Film', wl_grid)
    print(f"    ZnO  @0.55μm: n={np.interp(0.55, wl_grid, n_ZnO):.3f}, "
          f"k={np.interp(0.55, wl_grid, k_ZnO):.4f}")
    print(f"    Ag   @0.55μm: n={np.interp(0.55, wl_grid, n_Ag):.3f}, "
          f"k={np.interp(0.55, wl_grid, k_Ag):.3f}")
    print(f"    Film @0.55μm: n={np.interp(0.55, wl_grid, n_film):.3f}, "
          f"k={np.interp(0.55, wl_grid, k_film):.5f}")

    print("\n[3] Computing spectra...")
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}  # angle -> dict of arrays
    for ia, theta in enumerate(angles_deg):
        Ts = np.zeros_like(wl_grid)
        Tp = np.zeros_like(wl_grid)
        Rs = np.zeros_like(wl_grid)
        Rp = np.zeros_like(wl_grid)
        for i, wl in enumerate(wl_grid):
            Ts[i], Rs[i] = compute_at(wl, theta, 's',
                                      n_ZnO[i], k_ZnO[i], n_Ag[i], k_Ag[i],
                                      n_film[i], k_film[i])
            Tp[i], Rp[i] = compute_at(wl, theta, 'p',
                                      n_ZnO[i], k_ZnO[i], n_Ag[i], k_Ag[i],
                                      n_film[i], k_film[i])
        T_unpol = 0.5 * (Ts + Tp)
        R_unpol = 0.5 * (Rs + Rp)
        results[theta] = dict(wl=wl_grid, Ts=Ts, Tp=Tp, T_unpol=T_unpol,
                              Rs=Rs, Rp=Rp, R_unpol=R_unpol)
        print(f"    θ={theta:>2}°: "
              f"T_vis={T_unpol[(wl_grid>=0.38)&(wl_grid<=0.78)].mean():.4f}, "
              f"T@0.55={np.interp(0.55, wl_grid, T_unpol):.4f}, "
              f"R@0.55={np.interp(0.55, wl_grid, R_unpol):.4f}, "
              f"T+R@0.55={np.interp(0.55, wl_grid, T_unpol)+np.interp(0.55, wl_grid, R_unpol):.4f}")

    print("\n[4] Exporting CSVs...")
    for theta in angles_deg:
        r = results[theta]
        path = out_dir / f'spectrum_angle_{theta}.csv'
        with open(path, 'w') as f:
            f.write('wl_um,Ts,Tp,T_unpol,Rs,Rp,R_unpol\n')
            for i in range(len(r['wl'])):
                f.write(f"{r['wl'][i]:.4f},{r['Ts'][i]:.6f},{r['Tp'][i]:.6f},"
                        f"{r['T_unpol'][i]:.6f},{r['Rs'][i]:.6f},{r['Rp'][i]:.6f},"
                        f"{r['R_unpol'][i]:.6f}\n")
        print(f"    {path.name}")

    # Combined long-format CSV
    combined_path = out_dir / 'all_angles.csv'
    with open(combined_path, 'w') as f:
        f.write('angle_deg,wl_um,Ts,Tp,T_unpol,Rs,Rp,R_unpol\n')
        for theta in angles_deg:
            r = results[theta]
            for i in range(len(r['wl'])):
                f.write(f"{theta},{r['wl'][i]:.4f},{r['Ts'][i]:.6f},{r['Tp'][i]:.6f},"
                        f"{r['T_unpol'][i]:.6f},{r['Rs'][i]:.6f},{r['Rp'][i]:.6f},"
                        f"{r['R_unpol'][i]:.6f}\n")
    print(f"    {combined_path.name}")

    print("\n[5] Plotting...")
    fig, (axT, axR) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(angles_deg)))
    for c, theta in zip(colors, angles_deg):
        r = results[theta]
        axT.plot(r['wl'], r['T_unpol'], color=c, label=f'θ={theta}°', linewidth=1.4)
        axR.plot(r['wl'], r['R_unpol'], color=c, label=f'θ={theta}°', linewidth=1.4)

    for ax, ylabel, title in [(axT, 'Transmittance', 'Transmittance (unpolarized)'),
                              (axR, 'Reflectance', 'Reflectance (unpolarized)')]:
        ax.axvspan(0.38, 0.78, alpha=0.12, color='yellow')
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim([0.3, 2.0])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Air | ZnO(40nm) | Ag(10nm) | ZnO(10nm) | Film(100μm) | Air\n'
                 'Unpolarized T & R vs wavelength at oblique incidence',
                 fontsize=12)
    plt.tight_layout()
    plot_path = out_dir / 'spectra_plot.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    {plot_path.name}")

    print("\n" + "=" * 64)
    print(f"Done. Outputs in: {out_dir}")
    print("=" * 64)


if __name__ == '__main__':
    main()
