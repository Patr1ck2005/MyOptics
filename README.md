# MyOptics: A Framework for Fourier Optics Simulation 🚀🔬

## Overview 🌟

MyOptics is a Python-based 🐍 framework for Fourier optics simulation, designed to model and
analyze light 💡 propagation through optical elements. The project uses the Angular Spectrum
Method for precise simulation, allowing for visualization of complex field distributions and
interactions in optical systems.

## Key Features ✨

- **Highly Modular Design** 🛠️: Components are easily interchangeable, making the framework intuitive and user-friendly.
- **High Performance** ⚡: Utilizes `cupy` for GPU-accelerated FFT calculations, significantly speeding up simulations compared to CPU-based methods.
- **Define Optical Elements** 🔍: Simulate lenses, phase plates, apertures, gratings, and other custom optical elements.
- **Angular Spectrum Propagation** 📐: Compute light field propagation in both real and Fourier spaces (`Fresnel` / `Rigorous` modes).
- **Vector Optics Engine** 🧭: Three-component (Ex, Ey, Ez) propagation with strict transversality projection; Richards-Wolf high-NA vectorial focusing; Jones polarization elements (waveplates, q-plates) with spin-orbit coupling (`vector/`).
- **Multilayer TMM** 🧱: Transfer-matrix computations for thin-film stacks; multilayer transfer functions embedded as momentum-space filters for superlens simulation (`multilayer/`).
- **Simulation Workflows** 🔁: Result containers with persistence, parameter sweeps with image metrics, and broadband (polychromatic) synthesis (`workflows/`).
- **Visualization Tools** 📊: Generate plots for intensity and phase distribution of light fields.
- **Physical Regression Tests** ✅: pytest suite covering energy conservation, focusing, TMM analytic checks, and vector-optics invariants.

## Requirements 📋

- Python 3.10+
- CUDA 12 capable GPU (drivers only — the CUDA runtime libraries are installed via pip wheels)
- See `requirements.txt` for pinned dependencies

## Installation 🛠️

```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies (includes CUDA 12 runtime wheels — no CUDA Toolkit needed)
pip install -r requirements.txt

# 3. Install the framework in editable mode
pip install -e .
```

> **GPU note**: CuPy 13 + the `nvidia-*-cu12` pip wheels provide the full CUDA runtime
> (NVRTC, cuFFT, cuBLAS…). You only need a working NVIDIA driver. On Windows, cupy's wheel
> does not auto-discover the pip-installed libraries — this repository handles it via
> `utils/cuda_path.py`, which runs automatically when `optical_system` is imported.

## Project Layout 🗂️

```
optical_system/      核心仿真框架（OpticalSystem + 光学元件）
  elements/          元件实现（透镜/光阑/光栅/轴棱锥/动量空间元件/多层膜平板）
propagation/         角谱传播算法
  multi_layer/       H-Q 形式多层膜场分布工具（MultiLayerTM，研究遗留区）
vector/              全矢量引擎（M2）：三分量场、矢量角谱传播、
                     Richards-Wolf 高NA焦场、Jones 偏振元件、矢量系统
workflows/           工作流层（M3）：Field 结果容器、参数扫描、
                     宽谱合成、标量→矢量桥接
utils/               光束模型、常量、CUDA DLL 路径
visualization/       截面/纵截面绘图（Plotter）
analytical/          解析近似计算脚本
examples/            受支持示例（见 examples/README.md）
  legacy/            历史探索脚本（未随版本验证）
multilayer/          薄膜 TMM 计算工作区
  common/            公共库（物料 nk 加载 + kx 驱动 TMM）
  round1/, round2/   计算轮次（脚本 + 数据 + 结果）
tests/               pytest 物理回归套件
AUDIT.md             全面审计报告（问题清单与处置结论）
```

## Usage 📝

### Initial Light Field and Simulation Setup 💡

The initial light field must be specified at `z=0`. This field can be a Gaussian beam or any
arbitrary distribution. Once defined, the program calculates the propagation of the light field
through various optical elements placed at different `z` positions.

Specify the simulation range and resolution carefully, as these directly affect accuracy.

### Fourier Optics and FFT Overview 🔍

- **Spatial Sampling (Δx, Δy)** 📏: To avoid aliasing, typically Δx, Δy < 1 / (2 ∗ f_max).
- **Simulation Window Size (Lx, Ly)** 📐: Determines the frequency resolution, Δf = 1 / L.
- **Frequency Coverage (Lf)** 🌌: Lf = N ∗ Δf.

Proper selection of the sampling interval and window size is essential for accurate results.
Note that the angular spectrum propagator does not band-limit the spectrum: for long
propagation distances keep the simulation window larger than the beam path to avoid
wrap-around aliasing.

### Step-by-Step Workflow 🛠️

1. **Define Initial Light Field** 💡
2. **Define Optical Elements** 🔍 and their `z_position`s
3. **Propagate Light Field** ➡️ via `OpticalSystem`
4. **Visualize Results** 📊 via `Plotter`

> **Modulation functions** (e.g. `SpatialPlate(modulation_function=...)`) receive **CuPy
> arrays** — use `cupy` ufuncs (`cp.exp`, `cp.arctan2`, …) inside them, not `numpy`.

### Numerical Precision

`OpticalSystem` defaults to `complex128`/`float64` (research precision). For very large meshes
that would exhaust GPU memory, pass `dtype=cp.complex64` explicitly.

## Example Code 💻

Below is a simplified example workflow demonstrating a 4f system using lenses and phase plates:

```python
import cupy as cp
import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import Lens, SpatialPlate
from visualization.plotter import Plotter

# Define parameters
wavelength = 0.5  # Wavelength in micrometers
sim_size = 100  # Simulation size
mesh = 1024 + 1  # Mesh size ( +1 to maintain central symmetry)
w_0 = 10.0  # Beam waist
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)

# Define initial light field (Gaussian beam)
initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)
f = 100  # Focal length
optical_system.add_element(
    # 调制函数接收 cupy 数组：必须用 cp ufunc
    SpatialPlate(z_position=1, modulation_function=lambda X, Y: cp.exp(1j * cp.arctan2(Y, X))))
optical_system.add_element(Lens(z_position=f + 1, focal_length=f))
optical_system.add_element(Lens(z_position=3 * f + 1, focal_length=f))

# Create plotter
plotter = Plotter(x, y)

# Compute cross sections
cross_z_positions = [0, 1, 2 * f + 1, 4 * f + 1]
cross_sections = optical_system.propagate_to_cross_sections(
    cross_z_positions,
    return_momentum_space_spectrum=True,
    propagation_mode='Fresnel')  # Fresnel | Rigorous

# Plot
plotter.plot_cross_sections(cross_sections, save_label='test-cross_section', show=False)

# Compute and plot a longitudinal section
coord_axis, z_coords, intensity, phase = optical_system.propagate_to_longitudinal_section(
    direction='x', position=0.0, num_z=100, z_max=4 * f + 1,
    propagation_mode='Rigorous')
plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase,
                                  save_label='test-longitudinal_section', show=False)
```

More complete experiments live in `examples/` — run any of them from the repository root:

```bash
python examples/radial_polarization_focusing.py    # 矢量：径向偏振高NA聚焦
python examples/qplate_vector_vortex.py            # 矢量：q-plate 矢量涡旋
python examples/broadband_chromatic_shift.py       # 工作流：宽谱色差焦移
python examples/multilayer_slab_superlens.py       # 多层膜：超透镜传函
```

See `examples/README.md` for the full index (historical exploration
scripts live under `examples/legacy/`).

## Testing ✅

```bash
python -m pytest tests/
```

Tests covering the import chain, element construction, and physics (energy conservation,
plane-wave phase, lens focusing, TMM analytic limits) run automatically; GPU-dependent tests
are skipped when CUDA is unavailable.

## Benchmark (Performance Data) ⏱️

| Framework           | Small 2D FFT (256x256) ⚡ | Medium 2D FFT (1024x1024) ⚡ | Large 2D FFT (8192x8192) ⚡ |
|---------------------|--------------------------|-----------------------------|----------------------------|
| NumPy (CPU)         | 0.003 s                  | 0.038 s                     | 3.438 s                    |
| SciPy (CPU)         | 0.001 s                  | 0.023 s                     | 1.919 s                    |
| CuPy (GPU)          | 0.155 s                  | 0.018 s                     | 0.094 s                    |

See `docs/benchmark.md` for end-to-end propagation benchmarks
(cached propagator: 2.68× on 513² × 256-step longitudinal scans).

## License 📜

This project has no license.
