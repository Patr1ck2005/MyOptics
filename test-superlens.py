import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import *
from visualization.plotter import Plotter
from utils.constants import PI

# Define parameters
wavelength = 1  # Wavelength in micrometers
w_0 = 0.1
sim_size = 10  # Simulation size
mesh = 1024 * 2 + 1  # Mesh size ( +1 to maintain central symmetry)
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)

# ----------------------------------------------------------------------------------------------------------------------
# Define initial light field (Gaussian beam)
initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# ----------------------------------------------------------------------------------------------------------------------
# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)

# ----------------------------------------------------------------------------------------------------------------------
# Create plotter
plotter = Plotter(x, y)

# ----------------------------------------------------------------------------------------------------------------------
# Compute and Visualization
plot_cross_sections = False
plot_longitudinal_section = True

if plot_cross_sections:
    # Compute
    cross_z_positions = [0, 0.5, 1]
    cross_sections = optical_system.propagate_to_cross_sections(
        cross_z_positions,
        return_momentum_space_spectrum=True,
        propagation_mode='Rigorous'
    )  # Rigorous | Rigorous
    # Plot
    plotter.plot_cross_sections(cross_sections, save_label='superlens_test', show=False)

# another visualization mode
if plot_longitudinal_section:  # independently of cross_sections
    # Compute
    coord_axis, z_coords, intensity, phase = (
        optical_system.propagate_to_longitudinal_section_direct(
            direction='x',
            position=0.0,
            num_z=16,
            z_max=1./100,
            propagation_mode='Rigorous')
    )  # Fresnel | Rigorous

    # Plot
    plotter.plot_longitudinal_section(
        coord_axis, z_coords, intensity, phase,
        save_label='study',
        show=True,
        # norm_vmin=1
    )
