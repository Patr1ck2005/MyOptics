"""Angle-resolved T/R spectra for several air-side ZnO thicknesses.

The calculation follows the existing round1 angle scan:
Air | ZnO(d_ZnO1) | Ag(10 nm) | ZnO(10 nm) | Film(semi-infinite, lossless)

Both s and p polarizations are calculated with coherent TMM. The exported
unpolarized values are the arithmetic mean of the two polarizations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from multilayer.common.materials import build_nk_interpolators
from multilayer.common.tmm import tmm_kx


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUT_ROOT = ROOT_DIR / "rsl" / "angle_scan_spectra_no_film_abs"
ANGLES_DEG = (0, 20, 40, 60, 80)
WL_GRID_UM = np.linspace(0.30, 2.00, 171)

# Keep the plot settings identical to the existing angle-scan result.
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["axes.unicode_minus"] = False


def build_nk_arrays(name: str, wl_grid_um: np.ndarray):
    """Linearly interpolate n and k, using endpoint values outside the data range."""
    return build_nk_interpolators(name, wl_grid_um, DATA_DIR)


def compute_at_no_abs(
    wl_um: float,
    theta_deg: float,
    pol: str,
    n_zno: float,
    k_zno: float,
    n_ag: float,
    k_ag: float,
    n_film: float,
    d_zno1_nm: float,
    d_ag_nm: float = 10.0,
    d_zno2_nm: float = 10.0,
):
    """Compute T and R for the lossless semi-infinite film convention."""
    k0 = 2.0 * np.pi / (wl_um * 1000.0)
    kx = k0 * np.sin(np.deg2rad(theta_deg))
    layers = [complex(n_zno, k_zno), complex(n_ag, k_ag), complex(n_zno, k_zno)]
    thicknesses = [d_zno1_nm, d_ag_nm, d_zno2_nm]
    return tmm_kx(
        wl_um,
        layers,
        thicknesses,
        complex(1.0, 0.0),
        complex(n_film, 0.0),
        kx,
        pol,
    )


def compute_variant(d_zno1_nm: float, n_zno, k_zno, n_ag, k_ag, n_film):
    """Compute all angles and return angle-keyed result arrays."""
    results = {}
    for theta in ANGLES_DEG:
        ts = np.zeros_like(WL_GRID_UM)
        tp = np.zeros_like(WL_GRID_UM)
        rs = np.zeros_like(WL_GRID_UM)
        rp = np.zeros_like(WL_GRID_UM)
        for index, wl_um in enumerate(WL_GRID_UM):
            ts[index], rs[index] = compute_at_no_abs(
                wl_um, theta, "s", n_zno[index], k_zno[index],
                n_ag[index], k_ag[index], n_film[index], d_zno1_nm,
            )
            tp[index], rp[index] = compute_at_no_abs(
                wl_um, theta, "p", n_zno[index], k_zno[index],
                n_ag[index], k_ag[index], n_film[index], d_zno1_nm,
            )
        results[theta] = {
            "wl": WL_GRID_UM.copy(),
            "Ts": ts,
            "Tp": tp,
            "T_unpol": 0.5 * (ts + tp),
            "Rs": rs,
            "Rp": rp,
            "R_unpol": 0.5 * (rs + rp),
        }
    return results


def write_spectrum_csv(path: Path, result: dict[str, np.ndarray]):
    """Write one per-angle CSV in the established column format."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["wl_um", "Ts", "Tp", "T_unpol", "Rs", "Rp", "R_unpol"])
        for row in zip(
            result["wl"], result["Ts"], result["Tp"], result["T_unpol"],
            result["Rs"], result["Rp"], result["R_unpol"],
        ):
            writer.writerow(
                [
                    f"{row[0]:.4f}", f"{row[1]:.6f}", f"{row[2]:.6f}",
                    f"{row[3]:.6f}", f"{row[4]:.6f}", f"{row[5]:.6f}",
                    f"{row[6]:.6f}",
                ]
            )


def write_all_angles_csv(path: Path, results: dict[int, dict[str, np.ndarray]]):
    """Write the long-format combined CSV used by the previous calculation."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["angle_deg", "wl_um", "Ts", "Tp", "T_unpol", "Rs", "Rp", "R_unpol"]
        )
        for theta in ANGLES_DEG:
            result = results[theta]
            for row in zip(
                result["wl"], result["Ts"], result["Tp"], result["T_unpol"],
                result["Rs"], result["Rp"], result["R_unpol"],
            ):
                writer.writerow(
                    [
                        theta, f"{row[0]:.4f}", f"{row[1]:.6f}",
                        f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}",
                        f"{row[5]:.6f}", f"{row[6]:.6f}",
                    ]
                )


def plot_results(path: Path, d_zno1_nm: float, results):
    """Create the 1950 x 750 px two-panel plot matching the reference."""
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ANGLES_DEG)))
    for color, theta in zip(colors, ANGLES_DEG):
        result = results[theta]
        label = f"\u03b8={theta}\u00b0"
        ax_t.plot(result["wl"], result["T_unpol"], color=color, label=label, linewidth=1.4)
        ax_r.plot(result["wl"], result["R_unpol"], color=color, label=label, linewidth=1.4)

    for axis, ylabel, title in (
        (ax_t, "Transmittance", "Transmittance (unpolarized)"),
        (ax_r, "Reflectance", "Reflectance (unpolarized)"),
    ):
        axis.axvspan(0.38, 0.78, alpha=0.12, color="yellow")
        axis.set_xlabel("Wavelength (\u03bcm)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xlim([0.3, 2.0])
        axis.set_ylim([0, 1])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=9)

    title = (
        f"Air | ZnO({d_zno1_nm:g}nm) | Ag(10nm) | ZnO(10nm) | "
        "Film(semi-inf, lossless)  \u2014  no substrate absorption (coherent)\n"
        "Unpolarized T & R vs wavelength at oblique incidence"
    )
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(path: Path, all_metrics: list[dict[str, float]]):
    """Write a compact numerical index for the three requested structures."""
    fields = [
        "ZnO1_nm", "angle_deg", "T_visible_avg", "T_at_0.55um",
        "R_at_0.55um", "TR_at_0.55um",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_metrics)


def run(d_zno1_values: list[float], out_root: Path):
    print("=" * 72)
    print("Angle-resolved T/R spectra: ZnO1 thickness variants")
    print("=" * 72)
    print(
        f"Wavelength: {WL_GRID_UM[0]:.2f}-{WL_GRID_UM[-1]:.2f} um, "
        f"{len(WL_GRID_UM)} points; angles: {list(ANGLES_DEG)}"
    )

    print("Loading material data...")
    n_zno, k_zno = build_nk_arrays("ZnO", WL_GRID_UM)
    n_ag, k_ag = build_nk_arrays("Ag", WL_GRID_UM)
    n_film, _ = build_nk_arrays("Film", WL_GRID_UM)
    print(
        f"ZnO @0.55 um: n={np.interp(0.55, WL_GRID_UM, n_zno):.3f}, "
        f"k={np.interp(0.55, WL_GRID_UM, k_zno):.4f}"
    )
    print(
        f"Ag  @0.55 um: n={np.interp(0.55, WL_GRID_UM, n_ag):.3f}, "
        f"k={np.interp(0.55, WL_GRID_UM, k_ag):.3f}"
    )
    print(f"Film @0.55 um: n={np.interp(0.55, WL_GRID_UM, n_film):.3f}")

    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for d_zno1_nm in d_zno1_values:
        name = f"ZnO1_{d_zno1_nm:g}nm"
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Computing {d_zno1_nm:g} nm air-side ZnO...")
        results = compute_variant(d_zno1_nm, n_zno, k_zno, n_ag, k_ag, n_film)

        for theta in ANGLES_DEG:
            write_spectrum_csv(out_dir / f"spectrum_angle_{theta}.csv", results[theta])
        write_all_angles_csv(out_dir / "all_angles.csv", results)
        plot_results(out_dir / "spectra_plot.png", d_zno1_nm, results)

        visible_mask = (WL_GRID_UM >= 0.38) & (WL_GRID_UM <= 0.78)
        for theta in ANGLES_DEG:
            result = results[theta]
            t055 = float(np.interp(0.55, result["wl"], result["T_unpol"]))
            r055 = float(np.interp(0.55, result["wl"], result["R_unpol"]))
            summary_rows.append(
                {
                    "ZnO1_nm": f"{d_zno1_nm:g}",
                    "angle_deg": theta,
                    "T_visible_avg": f"{result['T_unpol'][visible_mask].mean():.6f}",
                    "T_at_0.55um": f"{t055:.6f}",
                    "R_at_0.55um": f"{r055:.6f}",
                    "TR_at_0.55um": f"{t055 + r055:.6f}",
                }
            )
        print(f"  Saved: {out_dir}")

    write_summary(out_root / "summary.csv", summary_rows)
    print(f"Summary: {out_root / 'summary.csv'}")
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zno1", nargs="+", type=float, default=[60.0, 80.0, 100.0],
        help="air-side ZnO thickness values in nm (default: 60 80 100)",
    )
    parser.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="directory for grouped result folders",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.zno1, args.out_root)
