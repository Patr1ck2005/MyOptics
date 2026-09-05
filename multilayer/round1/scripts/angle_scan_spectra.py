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

import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from multilayer.common.materials import build_nk_interpolators
from multilayer.common.tmm import kz_of, tmm_kx

OUT_DIR = Path(__file__).parent.parent / 'rsl' / 'angle_scan_spectra'
DATA_DIR = Path(__file__).parent.parent / 'data'

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['axes.unicode_minus'] = False


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
    kz_film = kz_of(N_film, k0, kx)
    tau = float(np.exp(-2.0 * np.imag(kz_film) * d_film_nm))

    return incoherent_tr(T_front, R_front, T_internal, R_internal, R_back, tau)


def compute_at_no_abs(wl_um, theta_deg, pol,
                      n_ZnO, k_ZnO, n_Ag, k_Ag, n_film,
                      d_ZnO1=40.0, d_Ag=10.0, d_ZnO2=10.0):
    """Compute (T, R) with semi-infinite lossless film (no substrate absorption).

    Matches the convention of optimize_transmittance.compute_spectrum: the film
    is treated as semi-infinite with real refractive index (k_film = 0), so
    there is no back-surface reflection and no absorption inside the film.
    """
    k0 = 2.0 * np.pi / (wl_um * 1000.0)
    kx = k0 * np.sin(np.deg2rad(theta_deg))

    N_inc = complex(1.0, 0.0)
    N_film = complex(n_film, 0.0)  # k = 0

    N_layers = [complex(n_ZnO, k_ZnO), complex(n_Ag, k_Ag), complex(n_ZnO, k_ZnO)]
    d_list = [d_ZnO1, d_Ag, d_ZnO2]
    return tmm_kx(wl_um, N_layers, d_list, N_inc, N_film, kx, pol)


# ============================================================
#  4. Main
# ============================================================
def main(mode='with_abs'):
    """Run angle-resolved scan.

    mode = 'with_abs' : 100μm film with absorption (incoherent multiple reflections)
    mode = 'no_abs'  : semi-infinite lossless film (no substrate absorption)
    """
    assert mode in ('with_abs', 'no_abs')
    print("=" * 64)
    print(f"Angle-resolved T/R: Air|ZnO40|Ag10|ZnO10|Film|Air  (mode={mode})")
    print("=" * 64)

    angles_deg = [0, 20, 40, 60, 80]
    wl_grid = np.linspace(0.30, 2.00, 171)  # 0.3-2.0 μm, 10 nm steps

    print(f"\n[1] Wavelength grid: {wl_grid[0]:.2f}-{wl_grid[-1]:.2f} μm, "
          f"{len(wl_grid)} points")
    print(f"    Angles: {angles_deg}")

    print("\n[2] Loading material data...")
    n_ZnO, k_ZnO = build_nk_interpolators('ZnO', wl_grid, DATA_DIR)
    n_Ag, k_Ag = build_nk_interpolators('Ag', wl_grid, DATA_DIR)
    n_film, k_film = build_nk_interpolators('Film', wl_grid, DATA_DIR)
    print(f"    ZnO  @0.55μm: n={np.interp(0.55, wl_grid, n_ZnO):.3f}, "
          f"k={np.interp(0.55, wl_grid, k_ZnO):.4f}")
    print(f"    Ag   @0.55μm: n={np.interp(0.55, wl_grid, n_Ag):.3f}, "
          f"k={np.interp(0.55, wl_grid, k_Ag):.3f}")
    print(f"    Film @0.55μm: n={np.interp(0.55, wl_grid, n_film):.3f}, "
          f"k={np.interp(0.55, wl_grid, k_film):.5f}")

    print(f"\n[3] Computing spectra (mode={mode})...")
    if mode == 'with_abs':
        out_dir = OUT_DIR
        compute_fn = lambda wl, th, pol, i: compute_at(
            wl, th, pol, n_ZnO[i], k_ZnO[i], n_Ag[i], k_Ag[i], n_film[i], k_film[i])
    else:
        out_dir = OUT_DIR.parent / 'angle_scan_spectra_no_film_abs'
        compute_fn = lambda wl, th, pol, i: compute_at_no_abs(
            wl, th, pol, n_ZnO[i], k_ZnO[i], n_Ag[i], k_Ag[i], n_film[i])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}  # angle -> dict of arrays
    for ia, theta in enumerate(angles_deg):
        Ts = np.zeros_like(wl_grid)
        Tp = np.zeros_like(wl_grid)
        Rs = np.zeros_like(wl_grid)
        Rp = np.zeros_like(wl_grid)
        for i, wl in enumerate(wl_grid):
            Ts[i], Rs[i] = compute_fn(wl, theta, 's', i)
            Tp[i], Rp[i] = compute_fn(wl, theta, 'p', i)
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

    if mode == 'with_abs':
        subtitle = 'Film(100μm) | Air  —  with substrate absorption (incoherent)'
    else:
        subtitle = 'Film(semi-inf, lossless)  —  no substrate absorption (coherent)'
    fig.suptitle('Air | ZnO(40nm) | Ag(10nm) | ZnO(10nm) | ' + subtitle + '\n'
                 'Unpolarized T & R vs wavelength at oblique incidence',
                 fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / 'spectra_plot.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    {plot_path.name}")

    print("\n" + "=" * 64)
    print(f"Done. Outputs in: {out_dir}")
    print("=" * 64)


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'with_abs'
    main(mode)
