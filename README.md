# MyOptics

**GPU Fourier optics in Python: build an optical path, propagate complex fields, and inspect the result.**

[中文](README.zh-CN.md) · [Quickstart](docs/quickstart.md) · [API](docs/api/system.md) · [Examples](examples/README.md) · [Visual gallery](docs/showcase/gallery.html)

![Scalar Gaussian beam spreading, computed with the MyOptics Fresnel propagator](docs/showcase/gaussian-propagation.png)

A synthetic Gaussian beam broadens as it propagates. The red curves show the paraxial prediction for the beam radius; the background is the computed intensity. [Reproduce this figure](examples/public_gaussian.py) · [Parameters, checks and environment](docs/showcase/manifest.json).

## What is included

| Area | Implemented building blocks |
| --- | --- |
| Scalar optics | Cached angular-spectrum propagation, Fresnel and `Rigorous` modes; lenses, apertures, gratings and spatial or momentum filters |
| Vector optics | Three-component fields, transverse spectral projection, Jones elements and Richards–Wolf focusing |
| Thin films | Transfer-matrix calculations and a multilayer momentum-space filter |
| Workflows | Field containers, parameter sweeps, coherent or incoherent spectral synthesis |
| Inspection | Cross sections, longitudinal cuts, intensity and phase; physical regression tests |

Use this for controlled Fourier-optics models. The scalar `Rigorous` mode uses the exact homogeneous-medium angular-spectrum dispersion; its name does not imply a full-vector Maxwell or arbitrary-geometry solver. See [physics guidance](docs/physics/angular-spectrum.md) and the source modules for each model's assumptions.

## Start here

Python **3.10+**, an NVIDIA GPU with a CUDA 12 compatible driver, and the declared CuPy dependencies are required for GPU propagation.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python examples/public_gaussian.py
```

The last command writes three PNG/SVG figures and a JSON manifest to `docs/showcase/`. It requires no downloaded beam, material or experimental data. On Windows, import `optical_system` before using CuPy kernels so the repository's CUDA DLL discovery runs. Dependency installation is shown for a fresh environment; the gallery was generated with the existing environment.

A minimal field propagation:

```python
import numpy as np
from optical_system.system import OpticalSystem
import cupy as cp
from propagation.propagator import AngularSpectrumPropagator

x = cp.linspace(-64, 64, 512, endpoint=False)  # micrometres
X, Y = cp.meshgrid(x, x)
u0 = cp.exp(-(X**2 + Y**2) / 8.0**2)
propagator = AngularSpectrumPropagator(x, x, wavelength=0.532, mode="Fresnel")
u1 = propagator.propagate(u0, z=np.pi * 8.0**2 / 0.532)
intensity = cp.asnumpy(cp.abs(u1)**2)
```

All lengths must share one unit. `OpticalSystem` normalizes its initial field to unit discrete squared norm; the minimal low-level example above preserves the input amplitude.

```mermaid
flowchart LR
    A["Coordinates + wavelength + initial field"] --> B["Optical elements / filters"]
    B --> C["Scalar or vector propagation"]
    C --> D["Complex field / Field container"]
    D --> E["Cuts, metrics and saved figures"]
    E --> F["Sampling and physical checks"]
```

## Read the public example

| Parameter | Value |
| --- | --- |
| Wavelength / input amplitude waist | 0.532 µm / 8 µm |
| Grid / sample spacing | 512 × 512 / 0.25 µm |
| Periodic FFT window | [−64, 64) µm on both axes |
| Propagation | Fresnel, 101 planes, 0 ≤ z ≤ 2zᵣ; zᵣ = πw₀²/λ ≈ 377.94 µm |
| Intensity normalization | Input on-axis peak = 1 for every displayed plane |
| Beam radius | w = 2√⟨x²⟩, weighted by the complete 2D intensity |

<table>
<tr><td><img src="docs/showcase/gaussian-profiles.png" alt="Intensity cuts at zero, one and two Rayleigh lengths"></td><td><img src="docs/showcase/gaussian-width.png" alt="Numerical second-moment width against paraxial Gaussian theory"></td></tr>
<tr><td>Beam spreading and peak reduction at fixed input normalization.</td><td>Fresnel FFT results compared with the analytic paraxial radius.</td></tr>
</table>

This is a homogeneous scalar, monochromatic, paraxial example (w₀ ≫ λ). FFT propagation assumes a periodic window; the beam must remain well inside it. Refine the grid and enlarge the window when changing the parameters. The comparison checks this specific discretization against its paraxial model; it does not measure error against a general Maxwell solution.

The recorded run used Python 3.11.9, MyOptics 0.5.1, NumPy 2.1.2, Matplotlib 3.9.2, CuPy 13.3.0 and an NVIDIA RTX 4060. Its maximum relative radius error was 1.17 × 10⁻¹² and discrete power drift 4.44 × 10⁻¹⁶. [Full run manifest](docs/showcase/manifest.json) records parameters and asset hashes.

## Explore further

- [Supported examples](examples/README.md): vector focusing, q-plates, multilayers and broadband workflows.
- [Quickstart](docs/quickstart.md), [documentation home](docs/index.md), [benchmarks](docs/benchmark.md) and [changelog](CHANGELOG.md).
- `examples/legacy/` contains historical scripts that are not maintained as current runnable examples.

```bash
python -m pytest tests/
```

GPU tests are marked and skip when CUDA is unavailable. A passing suite only covers the tested models and parameter cases.

## License

The package metadata currently declares **Proprietary** and this repository has no open-source license grant. Public source visibility does not grant general redistribution or reuse rights.
