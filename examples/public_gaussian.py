"""Reproduce the public Gaussian-beam gallery; no external data required.

Run from the repository root: python examples/public_gaussian.py
All lengths are micrometres. This uses the repository's scalar Fresnel
propagator, not a full-vector Maxwell solver.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Import first so the repository's CUDA DLL discovery runs on Windows.
from optical_system.system import OpticalSystem  # noqa: F401
import cupy as cp
from propagation.propagator import AngularSpectrumPropagator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "showcase"
WAVELENGTH, WAIST, N, HALF_WINDOW = 0.532, 8.0, 512, 64.0
ZR = np.pi * WAIST**2 / WAVELENGTH


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.linewidth": 0.5,
                         "lines.linewidth": 1, "xtick.direction": "in",
                         "ytick.direction": "in", "svg.fonttype": "none"})
    x = np.linspace(-HALF_WINDOW, HALF_WINDOW, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    field = cp.asarray(np.exp(-(X**2 + Y**2) / WAIST**2), dtype=cp.complex128)
    propagator = AngularSpectrumPropagator(cp.asarray(x), cp.asarray(x), WAVELENGTH, "Fresnel")
    z_values = np.linspace(0, 2 * ZR, 101)
    center = N // 2
    cuts, widths, powers = [], [], []
    snapshots = {}
    for z in z_values:
        intensity = cp.asnumpy(cp.abs(propagator.propagate(field, float(z)))**2)
        cuts.append(intensity[center])
        widths.append(2 * np.sqrt(np.sum(intensity * X**2) / np.sum(intensity)))
        powers.append(float(np.sum(intensity)))
        if len(cuts) in (1, 51, 101):
            snapshots[len(cuts)] = intensity
    cuts = np.asarray(cuts)
    predicted = WAIST * np.sqrt(1 + (z_values / ZR)**2)
    width_error = float(np.max(np.abs(np.asarray(widths) / predicted - 1)))
    power_drift = float(np.max(np.abs(np.asarray(powers) / powers[0] - 1)))
    assert width_error < 1e-5, width_error
    assert power_drift < 1e-10, power_drift
    cmap = LinearSegmentedColormap.from_list("intensity", ["#00020b", "#00245e", "#075dff", "#73c9ff", "#ffffff"])
    def save(fig, name):
        fig.savefig(OUT / f"{name}.png", dpi=240)
        fig.savefig(OUT / f"{name}.svg")
        plt.close(fig)
    fig = plt.figure(figsize=(105 / 25.4, 73 / 25.4))
    ax = fig.add_axes([0.14, 0.17, 0.70, 0.76])
    cax = fig.add_axes([0.87, 0.17, 0.027, 0.76])
    image = ax.imshow(cuts.T, origin="lower", extent=[0, 2, -64, 64], aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.plot(z_values / ZR, predicted, color="#ff4545", lw=0.75, label=r"$\pm w(z)$ (paraxial theory)")
    ax.plot(z_values / ZR, -predicted, color="#ff4545", lw=0.75)
    ax.set(xlabel=r"Propagation $z/z_R$", ylabel=r"Transverse $x$ ($\mu$m)", ylim=(-35, 35))
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="white")
    cb = fig.colorbar(image, cax=cax); cb.set_label(r"$|U(x,0,z)|^2/|U(0,0,0)|^2$", fontsize=7); cb.ax.tick_params(labelsize=7)
    save(fig, "gaussian-propagation")
    fig, ax = plt.subplots(figsize=(105 / 25.4, 70 / 25.4))
    for index, color, label in [(1, "#0054ff", "0"), (51, "#e60028", "1"), (101, "#138149", "2")]:
        ax.plot(x, snapshots[index][center], color=color, label=rf"$z/z_R={label}$")
    ax.set(xlim=(-35, 35), ylim=(0, 1.04), xlabel=r"Transverse $x$ ($\mu$m)", ylabel=r"$|U(x,0,z)|^2/|U(0,0,0)|^2$")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(pad=0.7)
    save(fig, "gaussian-profiles")
    fig, ax = plt.subplots(figsize=(105 / 25.4, 70 / 25.4))
    ax.plot(z_values / ZR, predicted, color="#e60028", label="Paraxial Gaussian theory")
    ax.plot(z_values[::5] / ZR, np.asarray(widths)[::5], "o", color="#0054ff", ms=3, mfc="white", label="MyOptics Fresnel FFT")
    ax.set(xlim=(0, 2), xlabel=r"Propagation $z/z_R$", ylabel=r"Beam radius $w=2\sqrt{\langle x^2\rangle}$ ($\mu$m)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(pad=0.7)
    save(fig, "gaussian-width")
    manifest = {"kind": "synthetic numerical demonstration", "source": "examples/public_gaussian.py",
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "parameters": {"wavelength_um": WAVELENGTH, "amplitude_waist_um": WAIST, "grid": [N, N],
                               "window_um": [-64, 64], "endpoint": False, "dx_um": 0.25,
                               "z_range_um": [0, 2 * ZR], "z_samples": 101, "rayleigh_length_um": ZR,
                               "mode": "Fresnel", "dtype": "complex128", "normalization": "input peak intensity = 1"},
                "checks": {"max_relative_width_error": width_error, "max_relative_discrete_power_drift": power_drift},
                "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ["numpy", "matplotlib", "cupy-cuda12x", "myoptics"]}, "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()},
                "scope": "homogeneous scalar paraxial propagation, periodic FFT window; no material, device or experimental data",
                "style": {"width_mm": 105, "font": "Arial", "dpi": 240, "ticks": "in", "intensity_colormap": "custom black-blue-white (not COMSOL table)"},
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("gaussian-*"))}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
