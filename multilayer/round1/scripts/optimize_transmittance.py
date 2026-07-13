"""
Optimize ZnO/Ag/ZnO coating on 100μm film for maximum visible transmittance.

Structure: Air | ZnO1 | Ag | ZnO2 | Film(100μm) | Air

Step 1: Coherent TMM treating film as semi-infinite substrate,
         scan Ag(10-20nm) × ZnO1(0-40nm) × ZnO2(0-40nm).
Step 2: Incoherent correction for the 100μm finite film on best candidates.

Data sources (../data/):
  - Ag.csv   : n,k for silver
  - ZnO.csv  : n,k for zinc oxide
  - 薄膜.xlsx : n,k for the 100μm polymer/glass film
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
#  0. Paths
# ============================================================
DATA_DIR = Path(__file__).parent.parent / 'data'
OUT_DIR = Path(__file__).parent.parent / 'rsl' / 'optimize_transmittance'

# ============================================================
#  1. Load material data
# ============================================================
def load_csv_two_section(filepath):
    """Load a CSV that has two sections: wl,n then wl,k, separated by a header line."""
    with open(filepath) as f:
        lines = f.readlines()

    sec1_start = None
    sec2_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('wl,n') or line.strip().startswith('wl,n'):
            sec1_start = i + 1
        elif line.strip().startswith('wl,k') or line.strip().startswith('wl,k'):
            sec2_start = i + 1

    wl_n = []; n_vals = []
    for line in lines[sec1_start:]:
        line = line.strip()
        if not line or line.startswith('wl'):
            break
        parts = line.split(',')
        if len(parts) >= 2:
            wl_n.append(float(parts[0]))
            n_vals.append(float(parts[1]))

    wl_k = []; k_vals = []
    for line in lines[sec2_start:]:
        line = line.strip()
        if not line or line.startswith('wl'):
            break
        parts = line.split(',')
        if len(parts) >= 2:
            wl_k.append(float(parts[0]))
            k_vals.append(float(parts[1]))

    return (np.array(wl_n), np.array(n_vals)), (np.array(wl_k), np.array(k_vals))


def load_material(name):
    """Load n,k for a material. Returns (wl_n, n, wl_k, k) with full overlap range only."""
    if name == 'Ag':
        (wl_n, n), (wl_k, k) = load_csv_two_section(DATA_DIR / 'Ag.csv')
    elif name == 'ZnO':
        (wl_n, n), (wl_k, k) = load_csv_two_section(DATA_DIR / 'ZnO.csv')
    elif name == 'Film':
        df = pd.read_excel(DATA_DIR / '薄膜.xlsx', header=None)
        vals = df.values
        wl_n = vals[:, 0].astype(float)
        n = vals[:, 1].astype(float)
        k = vals[:, 2].astype(float)
        wl_k = wl_n.copy()
        return wl_n, n, wl_k, k
    else:
        raise ValueError(f"Unknown material: {name}")

    return wl_n, n, wl_k, k


def build_nk_interpolators(name, wl_grid):
    """Build n(λ) and k(λ) interpolators on a common wavelength grid (in μm)."""
    wl_n, n_raw, wl_k, k_raw = load_material(name)

    # Interpolate n and k to common grid, extrapolate with nearest
    n_interp = interp1d(wl_n, n_raw, kind='linear',
                        bounds_error=False, fill_value=(n_raw[0], n_raw[-1]))
    k_interp = interp1d(wl_k, k_raw, kind='linear',
                        bounds_error=False, fill_value=(k_raw[0], k_raw[-1]))

    return n_interp(wl_grid), k_interp(wl_grid)


# ============================================================
#  2. TMM — normal incidence (coherent)
# ============================================================
def tmm_normal_incidence(wl_um, n_list, k_list, d_nm_list, n_inc=1.0, n_exit=1.0):
    """
    Coherent TMM at normal incidence (kx=0).

    Parameters
    ----------
    wl_um : float
        Wavelength in μm.
    n_list, k_list : list of float
        Refractive index (real, imag) per layer.
    d_nm_list : list of float or None
        Thickness in nm. None = semi-infinite.
    n_inc, n_exit : float
        Incident and exit medium refractive indices (assumed real).

    Returns
    -------
    T : float  (transmittance, 0-1)
    R : float  (reflectance, 0-1)
    """
    N = [complex(n, k) for n, k in zip(n_list, k_list)]

    # Build stack matrix
    M = np.eye(2, dtype=complex)
    for Nj, dj in zip(N, d_nm_list):
        if dj is not None:
            delta = (2 * np.pi / (wl_um * 1000)) * Nj * dj  # wl_um→nm: *1000
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            Mj = np.array([
                [cos_d, -1j * sin_d / Nj],
                [-1j * Nj * sin_d, cos_d]
            ], dtype=complex)
            M = M @ Mj

    m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

    Y0 = n_inc
    Ys = n_exit

    denom = Y0 * m11 + Y0 * Ys * m12 + m21 + Ys * m22
    r = (Y0 * m11 + Y0 * Ys * m12 - m21 - Ys * m22) / denom
    t = 2 * Y0 / denom

    T = (np.real(Ys) / np.real(Y0)) * np.abs(t) ** 2
    R = np.abs(r) ** 2

    return T, R


def compute_spectrum(wl_grid, n_ZnO, k_ZnO, n_Ag, k_Ag, n_film, k_film,
                     d_ZnO1, d_Ag, d_ZnO2, n_inc=1.0, n_exit_is_film=True):
    """
    Compute T(λ) spectrum for Air|ZnO1|Ag|ZnO2|Film(semi-inf).

    d_ZnO1 or d_ZnO2 = 0 → skip that layer.
    If n_exit_is_film, exit medium = film (semi-infinite). Else exit = air.
    """
    T = np.zeros_like(wl_grid)
    for i, wl in enumerate(wl_grid):
        n_exit = n_film[i] if n_exit_is_film else 1.0

        n_list = []
        k_list = []
        d_list = []

        if d_ZnO1 > 0:
            n_list.append(n_ZnO[i]); k_list.append(k_ZnO[i])
            d_list.append(d_ZnO1)
        n_list.append(n_Ag[i]); k_list.append(k_Ag[i])
        d_list.append(d_Ag)
        if d_ZnO2 > 0:
            n_list.append(n_ZnO[i]); k_list.append(k_ZnO[i])
            d_list.append(d_ZnO2)

        Ti, _ = tmm_normal_incidence(wl, n_list, k_list, d_list,
                                     n_inc=n_inc, n_exit=n_exit)
        T[i] = Ti

    return T


# ============================================================
#  3. Incoherent correction for thick film
# ============================================================
def incoherent_film_correction(wl_grid, T_coating, R_coating_internal,
                                n_film, k_film, d_film_nm=100000):
    """
    Add the effect of a thick, incoherent film behind the coating.

    Coating (ZnO/Ag/ZnO) is treated coherently → gives T_coating(λ)
    and R_coating_internal(λ) (reflectance looking back from film into coating).

    Film: internal transmittance τ = exp(-4π·k·d/λ)
          back-surface reflectance R_back = |(N_film - 1)/(N_film + 1)|²

    Total: T = T_coating * τ * (1-R_back) / (1 - R_coating_internal * R_back * τ²)
    """
    tau = np.exp(-4 * np.pi * k_film * d_film_nm / (wl_grid * 1000))
    N_f = n_film + 1j * k_film
    R_back = np.abs((N_f - 1.0) / (N_f + 1.0)) ** 2

    T_total = T_coating * tau * (1 - R_back) / (1 - R_coating_internal * R_back * tau ** 2)
    return T_total


def compute_R_internal(wl_grid, n_ZnO, k_ZnO, n_Ag, k_Ag, n_film, k_film,
                       d_ZnO1, d_Ag, d_ZnO2):
    """
    Compute reflectance looking BACK from the film into the coating.
    Structure (reversed): Film | ZnO2 | Ag | ZnO1 | Air.

    d_ZnO1/d_ZnO2 = 0 → skip that layer.
    """
    R_int = np.zeros_like(wl_grid)
    for i, wl in enumerate(wl_grid):
        n_list = []
        k_list = []
        d_list = []

        # Reversed order: film side first
        # Film|ZnO2|Ag|ZnO1|Air
        # Incident = Film, Exit = Air
        # But we want R looking from film side, so the coating layers are reversed:
        if d_ZnO2 > 0:
            n_list.append(n_ZnO[i]); k_list.append(k_ZnO[i])
            d_list.append(d_ZnO2)
        n_list.append(n_Ag[i]); k_list.append(k_Ag[i])
        d_list.append(d_Ag)
        if d_ZnO1 > 0:
            n_list.append(n_ZnO[i]); k_list.append(k_ZnO[i])
            d_list.append(d_ZnO1)

        # Incident = film, Exit = air
        n_f = n_film[i]  # already real
        _, Ri = tmm_normal_incidence(wl, n_list, k_list, d_list,
                                     n_inc=n_f, n_exit=1.0)
        R_int[i] = Ri

    return R_int


# ============================================================
#  4. Parameter scan
# ============================================================
def run_scan(wl_grid, wl_vis_mask, n_ZnO, k_ZnO, n_Ag, k_Ag, n_film, k_film):
    """Scan Ag and ZnO thicknesses, return 3D array of average visible T."""
    d_Ag_vals = np.arange(10, 21, 1)       # 10-20 nm
    d_ZnO_vals = np.arange(0, 42, 2)       # 0-40 nm

    results = np.full((len(d_ZnO_vals), len(d_Ag_vals), len(d_ZnO_vals)), np.nan)

    total = len(d_ZnO_vals) * len(d_Ag_vals) * len(d_ZnO_vals)
    count = 0

    for i1, d1 in enumerate(d_ZnO_vals):
        for ia, da in enumerate(d_Ag_vals):
            for i2, d2 in enumerate(d_ZnO_vals):
                T = compute_spectrum(wl_grid, n_ZnO, k_ZnO, n_Ag, k_Ag,
                                     n_film, k_film, d1, da, d2)
                results[i1, ia, i2] = T[wl_vis_mask].mean()
                count += 1
                if count % 500 == 0:
                    print(f"  ... {count}/{total} ({100*count/total:.0f}%)")

    return d_ZnO_vals, d_Ag_vals, d_ZnO_vals, results


def find_top_candidates(d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results, n=10):
    """Return top N (ZnO1, Ag, ZnO2, T_avg) sorted descending."""
    candidates = []
    for i1, d1 in enumerate(d_ZnO1_vals):
        for ia, da in enumerate(d_Ag_vals):
            for i2, d2 in enumerate(d_ZnO2_vals):
                candidates.append((d1, da, d2, results[i1, ia, i2]))
    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates[:n]


# ============================================================
#  5. CSV Export
# ============================================================
def export_csv_results(d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results):
    """
    Export full scan results as CSV files, organized by Ag thickness.
    Directory structure:
      results/
        summary.csv          — all data in long format
        Ag=10nm/ZnO1_ZnO2_T.csv  — 2D matrix: rows=ZnO1, cols=ZnO2
        Ag=11nm/ZnO1_ZnO2_T.csv
        ...
    """
    results_dir = OUT_DIR
    results_dir.mkdir(exist_ok=True)

    # Summary CSV: long format with all combinations
    summary_path = results_dir / 'summary.csv'
    with open(summary_path, 'w') as f:
        f.write('Ag_nm,ZnO1_nm,ZnO2_nm,T_vis_avg\n')
        for ia, da in enumerate(d_Ag_vals):
            for i1, d1 in enumerate(d_ZnO1_vals):
                for i2, d2 in enumerate(d_ZnO2_vals):
                    f.write(f'{da:.0f},{d1:.0f},{d2:.0f},{results[i1,ia,i2]:.6f}\n')
    print(f"    Saved {summary_path}")

    # Per-Ag subdirectories with 2D matrix CSVs
    for ia, da in enumerate(d_Ag_vals):
        ag_dir = results_dir / f'Ag={da:.0f}nm'
        ag_dir.mkdir(exist_ok=True)

        # Matrix format: first column = ZnO1, header row = ZnO2
        csv_path = ag_dir / 'ZnO1_ZnO2_T.csv'
        with open(csv_path, 'w') as f:
            # Header: ZnO2 values
            header = 'ZnO1\\ZnO2,' + ','.join(f'{d2:.0f}' for d2 in d_ZnO2_vals)
            f.write(header + '\n')
            # Data rows
            for i1, d1 in enumerate(d_ZnO1_vals):
                row = f'{d1:.0f},' + ','.join(f'{results[i1,ia,i2]:.6f}'
                                              for i2 in range(len(d_ZnO2_vals)))
                f.write(row + '\n')
        print(f"    Saved {csv_path}")


# ============================================================
#  6. Plotting
# ============================================================
def plot_per_Ag_heatmaps(d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results):
    """
    For each Ag thickness, plot a 2D heatmap: ZnO1 × ZnO2 → T_vis_avg.
    Mark the best ZnO1/ZnO2 point on each subplot.
    """
    n_Ag = len(d_Ag_vals)
    n_cols = 4
    n_rows = int(np.ceil(n_Ag / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    fig.suptitle('T_vis_avg vs ZnO1 × ZnO2 — one panel per Ag thickness', fontsize=14)

    for ia, da in enumerate(d_Ag_vals):
        ax = axes.flat[ia]
        data = results[:, ia, :]  # [ZnO1, ZnO2]

        im = ax.pcolormesh(d_ZnO2_vals, d_ZnO1_vals, data,
                           shading='auto', cmap='nipy_spectral')
        plt.colorbar(im, ax=ax)

        # Find best for this Ag
        best_idx = np.unravel_index(np.nanargmax(data), data.shape)
        best_zno1 = d_ZnO1_vals[best_idx[0]]
        best_zno2 = d_ZnO2_vals[best_idx[1]]
        best_T = data[best_idx]
        ax.plot(best_zno2, best_zno1, 'go', markersize=5, markerfacecolor='none')
        ax.set_title(f'Ag = {da:.0f} nm\nBest: ZnO1={best_zno1:.0f} ZnO2={best_zno2:.0f} T={best_T:.4f}')
        ax.set_xlabel('ZnO2 (nm)')
        ax.set_ylabel('ZnO1 (nm)')

    # Hide unused axes
    for ax in axes.flat[n_Ag:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'heatmap_per_Ag.png', dpi=150)
    plt.close()


def print_best_per_Ag(d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results):
    """Print the best ZnO1/ZnO2 for each Ag thickness."""
    print(f"\n{'Ag (nm)':<10} {'ZnO1 (nm)':<12} {'ZnO2 (nm)':<12} {'T_vis_avg':<12}")
    print("-" * 46)
    for ia, da in enumerate(d_Ag_vals):
        data = results[:, ia, :]
        best_idx = np.unravel_index(np.nanargmax(data), data.shape)
        best_zno1 = d_ZnO1_vals[best_idx[0]]
        best_zno2 = d_ZnO2_vals[best_idx[1]]
        best_T = data[best_idx]
        print(f"{da:<10.0f} {best_zno1:<12.0f} {best_zno2:<12.0f} {best_T:<12.5f}")


def plot_top_spectra(wl_grid, wl_vis_mask, top_candidates,
                     n_ZnO, k_ZnO, n_Ag, k_Ag, n_film, k_film):
    """Plot T(λ) spectra for top candidates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_candidates)))

    for rank, (d1, da, d2, T_avg) in enumerate(top_candidates):
        T_coh = compute_spectrum(wl_grid, n_ZnO, k_ZnO, n_Ag, k_Ag,
                                  n_film, k_film, d1, da, d2)

        # Incoherent correction
        R_int = compute_R_internal(wl_grid, n_ZnO, k_ZnO, n_Ag, k_Ag,
                                    n_film, k_film, d1, da, d2)
        T_inc = incoherent_film_correction(wl_grid, T_coh, R_int,
                                            n_film, k_film, d_film_nm=100000)

        label = f'#{rank+1} ZnO1={d1:.0f} Ag={da:.0f} ZnO2={d2:.0f}'
        ax1.plot(wl_grid, T_coh, color=colors[rank], label=label, linewidth=1)
        ax2.plot(wl_grid, T_inc, color=colors[rank], label=label, linewidth=1)

    for ax, title in [(ax1, 'Coherent (film=semi-inf)'), (ax2, 'Incoherent (film=100μm)')]:
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Transmittance')
        ax.set_title(title)
        ax.axvspan(0.38, 0.78, alpha=0.1, color='yellow', label='Visible')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim([0.3, 2.5])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Top Candidates — Transmittance Spectra', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'top_spectra.png', dpi=150)
    plt.close()


def plot_best_detailed(wl_grid, wl_vis_mask, best, n_ZnO, k_ZnO, n_Ag, k_Ag,
                       n_film, k_film):
    """Detailed plot for the single best candidate."""
    d1, da, d2, T_avg = best

    T_coh = compute_spectrum(wl_grid, n_ZnO, k_ZnO, n_Ag, k_Ag,
                              n_film, k_film, d1, da, d2)
    R_int = compute_R_internal(wl_grid, n_ZnO, k_ZnO, n_Ag, k_Ag,
                                n_film, k_film, d1, da, d2)
    T_inc = incoherent_film_correction(wl_grid, T_coh, R_int,
                                        n_film, k_film, d_film_nm=100000)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl_grid, T_coh, 'b-', label='Coherent T (film=semi-inf)', linewidth=1.5)
    ax.plot(wl_grid, T_inc, 'r-', label='Incoherent T (film=100μm)', linewidth=1.5)
    ax.axvspan(0.38, 0.78, alpha=0.1, color='yellow')
    ax.text(0.58, 0.95, 'Visible band', transform=ax.get_xaxis_transform(),
            ha='center', fontsize=10, style='italic')

    T_vis_coh = T_coh[wl_vis_mask].mean()
    T_vis_inc = T_inc[wl_vis_mask].mean()
    ax.axhline(T_vis_coh, color='b', linestyle=':', alpha=0.6)
    ax.axhline(T_vis_inc, color='r', linestyle=':', alpha=0.6)

    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Transmittance')
    ax.set_title(f'Best: ZnO1={d1:.0f}nm, Ag={da:.0f}nm, ZnO2={d2:.0f}nm\n'
                 f'Avg T_vis (coh)={T_vis_coh:.4f}, (inc)={T_vis_inc:.4f}')
    ax.set_xlim([0.3, 2.5])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'best_detailed.png', dpi=150)
    plt.close()


# ============================================================
#  6. Main
# ============================================================
def main():
    print("=" * 60)
    print("ZnO/Ag/ZnO Transmittance Optimizer")
    print("=" * 60)

    # ---- 6a. Build wavelength grid ----
    print("\n[1] Building wavelength grid...")
    wl_full = np.linspace(0.30, 2.50, 2201)   # 0.3-2.5 μm, 1nm steps
    vis_mask = (wl_full >= 0.38) & (wl_full <= 0.78)
    print(f"    Full range: {wl_full[0]:.2f}-{wl_full[-1]:.2f} μm, {len(wl_full)} points")
    print(f"    Visible range: 0.38-0.78 μm, {vis_mask.sum()} points")

    # ---- 6b. Load & interpolate materials ----
    print("\n[2] Loading material data...")
    n_ZnO, k_ZnO = build_nk_interpolators('ZnO', wl_full)
    n_Ag, k_Ag = build_nk_interpolators('Ag', wl_full)
    n_film, k_film = build_nk_interpolators('Film', wl_full)
    print("    Done.")

    # Quick sanity check
    print(f"    ZnO  @ 0.55μm: n={np.interp(0.55, wl_full, n_ZnO):.3f}, k={np.interp(0.55, wl_full, k_ZnO):.4f}")
    print(f"    Ag   @ 0.55μm: n={np.interp(0.55, wl_full, n_Ag):.3f}, k={np.interp(0.55, wl_full, k_Ag):.3f}")
    print(f"    Film @ 0.55μm: n={np.interp(0.55, wl_full, n_film):.3f}, k={np.interp(0.55, wl_full, k_film):.5f}")

    # ---- 6c. Baseline: just film, no coating ----
    print("\n[3] Baseline: bare film substrate...")
    # Bare film as semi-infinite
    T_bare = np.zeros_like(wl_full)
    for i, wl in enumerate(wl_full):
        N_f = n_film[i] + 1j * k_film[i]
        R_bare = np.abs((1.0 - N_f) / (1.0 + N_f)) ** 2
        T_bare[i] = 1.0 - R_bare
    print(f"    Avg T_vis (bare film, semi-inf) = {T_bare[vis_mask].mean():.4f}")

    # ---- 6d. Scan ----
    print("\n[4] Scanning Ag/ZnO thicknesses...")
    print(f"    Ag:  10-20 nm, step 1 nm  ({len(np.arange(10,21,1))} values)")
    print(f"    ZnO: 0-40 nm,  step 2 nm  ({len(np.arange(0,42,2))} values)")
    print(f"    Total: {11*21*21} combinations")

    d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results = run_scan(
        wl_full, vis_mask, n_ZnO, k_ZnO, n_Ag, k_Ag, n_film, k_film)

    # ---- 6e. Results ----
    print("\n[5] Top candidates (coherent, film=semi-inf):")
    top = find_top_candidates(d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results, n=15)

    print(f"{'Rank':<5} {'ZnO1(nm)':<10} {'Ag(nm)':<8} {'ZnO2(nm)':<10} {'T_vis_avg':<10}")
    print("-" * 50)
    for rank, (d1, da, d2, T) in enumerate(top):
        print(f"{rank+1:<5} {d1:<10.0f} {da:<8.0f} {d2:<10.0f} {T:<10.5f}")

    best = (top[0][0], top[0][1], top[0][2], top[0][3])

    # ---- 6f. Best candidate with incoherent correction ----
    print(f"\n[6] Best candidate detailed analysis:")
    print(f"    ZnO1={best[0]:.0f}nm, Ag={best[1]:.0f}nm, ZnO2={best[2]:.0f}nm")

    T_best_coh = compute_spectrum(wl_full, n_ZnO, k_ZnO, n_Ag, k_Ag,
                                   n_film, k_film, best[0], best[1], best[2])
    R_int = compute_R_internal(wl_full, n_ZnO, k_ZnO, n_Ag, k_Ag,
                                n_film, k_film, best[0], best[1], best[2])
    T_best_inc = incoherent_film_correction(wl_full, T_best_coh, R_int,
                                              n_film, k_film, d_film_nm=100000)

    print(f"    Avg T_vis (coherent, film=semi-inf)  = {T_best_coh[vis_mask].mean():.5f}")
    print(f"    Avg T_vis (incoherent, film=100μm)   = {T_best_inc[vis_mask].mean():.5f}")

    # Also compute for top 15 with incoherent
    print(f"\n[7] Top 15 with incoherent correction:")
    print(f"{'Rank':<5} {'ZnO1':<8} {'Ag':<6} {'ZnO2':<8} {'T_vis_coh':<12} {'T_vis_inc':<12}")
    print("-" * 60)
    top_inc_results = []
    for rank, (d1, da, d2, T_avg) in enumerate(top):
        T_coh = compute_spectrum(wl_full, n_ZnO, k_ZnO, n_Ag, k_Ag,
                                  n_film, k_film, d1, da, d2)
        R_int_i = compute_R_internal(wl_full, n_ZnO, k_ZnO, n_Ag, k_Ag,
                                      n_film, k_film, d1, da, d2)
        T_inc_i = incoherent_film_correction(wl_full, T_coh, R_int_i,
                                               n_film, k_film, d_film_nm=100000)
        T_vis_inc = T_inc_i[vis_mask].mean()
        top_inc_results.append((d1, da, d2, T_avg, T_vis_inc))
        print(f"{rank+1:<5} {d1:<8.0f} {da:<6.0f} {d2:<8.0f} {T_avg:<12.5f} {T_vis_inc:<12.5f}")

    # ---- 6g. Export CSVs ----
    print("\n[8] Exporting CSV results...")
    export_csv_results(d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results)

    # ---- 6h. Plots ----
    print("\n[9] Generating plots...")
    plot_per_Ag_heatmaps(d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results)
    print_best_per_Ag(d_ZnO1_vals, d_Ag_vals, d_ZnO2_vals, results)
    plot_top_spectra(wl_full, vis_mask, top[:8], n_ZnO, k_ZnO, n_Ag, k_Ag,
                     n_film, k_film)
    plot_best_detailed(wl_full, vis_mask, best, n_ZnO, k_ZnO, n_Ag, k_Ag,
                       n_film, k_film)

    # ---- 6i. Best symmetric case ----
    print("\n[10] Best symmetric case (ZnO1 = ZnO2):")
    sym_best = None
    sym_T = -1
    for i1, d1 in enumerate(d_ZnO1_vals):
        if d1 == 0: continue
        for ia, da in enumerate(d_Ag_vals):
            i2 = i1  # symmetric
            T_avg = results[i1, ia, i2]
            if not np.isnan(T_avg) and T_avg > sym_T:
                sym_T = T_avg
                sym_best = (d1, da, d1, T_avg)
    if sym_best:
        print(f"    ZnO={sym_best[0]:.0f}nm, Ag={sym_best[1]:.0f}nm, T_vis={sym_T:.5f}")

    print("\n" + "=" * 60)
    print(f"Done. Outputs saved to {OUT_DIR}")
    print("  results/               — CSV data, per Ag subfolder")
    print("  heatmap_per_Ag.png     — 2D heatmap per Ag thickness")
    print("  top_spectra.png        — best candidate spectra")
    print("  best_detailed.png      — detailed best spectrum")
    print("=" * 60)


if __name__ == '__main__':
    main()
