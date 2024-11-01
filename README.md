# MyOptics: A Framework for Fourier Optics Simulation 🚀🔬

## Overview 🌟

MyOptics is a Python-based 🐍 framework for Fourier optics simulation, designed to model and analyze light 💡 propagation through optical elements. The project uses the Angular Spectrum Method for precise simulation, allowing for visualization of complex field distributions and interactions in optical systems.

## Key Features ✨

- **Highly Modular Design** 🛠️: Components are easily interchangeable, making the framework intuitive and user-friendly.
- **High Performance** ⚡: Utilizes `cupy` for GPU-accelerated FFT calculations, significantly speeding up simulations compared to CPU-based methods.
- **Benchmarking** 📊: Includes performance benchmarking for different configurations (benchmark data will be added).
- **Define Optical Elements** 🔍: Simulate lenses, phase plates, and other custom optical elements.
- **Angular Spectrum Propagation** 📐: Compute light field propagation in both real and Fourier spaces.
- **Visualization Tools** 📊: Generate plots for intensity and phase distribution of light fields.

## Calculation Method 🧠

This program employs the Angular Spectrum Method to calculate the light field distribution at different spatial positions, given the light field at a known plane. By transforming the known plane's light field into the frequency domain, the method allows propagation to any other plane, accurately capturing the evolution of light through free space or optical elements.

## Requirements 📋

- Python 3.7+ 🐍
- `numpy` 📦
- `cupy` (for GPU acceleration) 💨
- `matplotlib` 📈

## Installation 🛠️

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install the correct version of `cupy` for GPU acceleration (based on your CUDA version). For example:

   ```bash
   pip install cupy-cuda11x  # Replace '11x' with your CUDA version
   ```

3. To check your CUDA version, run:

   ```bash
   nvcc --version
   ```

## Benchmark (Performance Data) ⏱️

| Framework           | Small 2D FFT (256x256) ⚡ | Medium 2D FFT (1024x1024) ⚡ | Large 2D FFT (8192x8192) ⚡ |
|---------------------|--------------------------|-----------------------------|----------------------------|
| MATLAB              | TBD                      | TBD                         | TBD                        |
| NumPy (CPU)         | 0.003 s                  | 0.038 s                     | 3.438 s                    |
| SciPy (CPU)         | 0.001 s                  | 0.023 s                     | 1.919 s                    |
| CuPy (GPU)          | 0.155 s                  | 0.018 s                     | 0.094 s                    |

## Usage 📝

### Initial Light Field and Simulation Setup 💡

The initial light field must be specified at `z=0`. This field can be a Gaussian beam or any arbitrary distribution. Once defined, the program calculates the propagation of the light field through various optical elements placed at different `z` positions.

Specify the simulation range and resolution carefully, as these directly affect accuracy. The Angular Spectrum Method uses Fourier transforms, so both spatial extent and resolution (mesh size) are critical for accurate results.

### Fourier Optics and FFT Overview 🔍

Fourier optics represents any light field in terms of its spatial frequency components, computed using the Fast Fourier Transform (FFT). The Angular Spectrum Method operates in the frequency (momentum) space, making understanding the relationship between real space and frequency space crucial for accurate simulation.

- **Spatial Sampling (Δx, Δy)** 📏: The sampling interval affects how well high-frequency components are captured. To avoid aliasing, typically Δx, Δy < 1 / (2 ∗ f_max), where f_max is the maximum spatial frequency.
- **Simulation Window Size (Lx, Ly)** 📐: The extent of the simulation region determines the frequency resolution, Δf = 1 / L, where L is the size of the simulation window. A larger window size results in finer frequency resolution.
- **Frequency Coverage (Lf)** 🌌: The frequency coverage is Lf = N ∗ Δf, where N is the number of sampling points, and Δf is the frequency resolution.

Proper selection of the sampling interval and window size is essential for accurate Fourier transform calculations and light field propagation.

### Step-by-Step Workflow 🛠️

1. **Define Initial Light Field** 💡: Specify the initial light field at `z=0`. This could be a Gaussian beam or any other arbitrary field.
2. **Define Optical Elements** 🔍: Create various optical elements (e.g., lenses, phase plates) and specify their positions.
3. **Propagate Light Field** ➡️: Use `OpticalSystem` to set up the initial field, add elements, and simulate propagation.
4. **Visualize Results** 📊: Use the `Plotter` class to visualize the intensity and phase distribution of the propagated field.

The workflow is illustrated below:

- **Define Initial Light Field** ➡️ **Add Optical Elements** ➡️ **Propagate Light Field** ➡️ **Visualize Results**

## Example Code 💻

Below is a simplified example workflow demonstrating a 4f system using lenses and phase plates. Users can modify this example to suit their needs.

### Cross-Section and Longitudinal Section Plotting 📊

```python
import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import Lens, PhasePlate
from visualization.plotter import Plotter

# Define parameters
wavelength = 0.5  # Wavelength in micrometers
sim_size = 100  # Simulation size
mesh = 1024+1  # Mesh size ( +1 to maintain central symmetry)
w_0 = 10.0  # Beam waist
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)

# ----------------------------------------------------------------------------------------------------------------------
# Define initial light field (Gaussian beam)
initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# ----------------------------------------------------------------------------------------------------------------------
# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)
f = 100  # Focal length
optical_system.add_element(PhasePlate(z_position=1, phase_function=lambda X, Y: np.exp(1j * np.arctan2(Y, X))))
optical_system.add_element(Lens(z_position=f + 1, focal_length=f))
optical_system.add_element(Lens(z_position=3 * f + 1, focal_length=f))

# ----------------------------------------------------------------------------------------------------------------------
# Create plotter
plotter = Plotter(x, y)

# ----------------------------------------------------------------------------------------------------------------------
# Compute and Visualization
plot_cross_sections = True
plot_longitudinal_section = True

if plot_cross_sections:
    # Compute
    cross_z_positions = [0, 1, 2 * f + 1, 4 * f + 1]
    cross_sections \
        = optical_system.propagate_to_cross_sections(cross_z_positions,
                                                     return_momentum_space_spectrum=True,
                                                     propagation_mode='Fresnel')  # Fresnel | Rigorous
    # Plot
    plotter.plot_cross_sections(cross_sections, save_label='test-cross_section', show=False)

# another visualization mode
if plot_longitudinal_section:  # independently of cross_sections
    # Compute
    coord_axis, z_coords, intensity, phase = (
        optical_system.propagate_to_longitudinal_section(direction='x',
                                                         position=0.0,
                                                         num_z=100,
                                                         z_max=4 * f + 1,
                                                         propagation_mode='Rigorous'))  # Fresnel | Rigorous

    # Plot
    plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase, save_label='test-longitudinal_section', show=False)
```

### Explanation 📖

- **Define Parameters** ⚙️: Set basic parameters like wavelength, simulation size, and mesh size.
- **Initialize System** 🛠️: Create initial Gaussian beam and define spatial coordinates.
- **Add Elements** 🔍: Insert elements like lenses and phase plates to form a 4f system.
- **Propagate and Plot** 📊: Simulate field propagation and visualize both cross-sections and longitudinal sections independently.

## License 📜

This project is licensed under the MIT License.

