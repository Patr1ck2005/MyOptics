import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import *
from visualization.plotter import Plotter

# Global configuration
project_name = 'vortex_beam-exp_system'

# Define parameters
wavelength = 1.5  # Wavelength in um
sim_size = 1024*6+1  # Simulation size
mesh = 1024*6+1  # Mesh size ( +1 to maintain central symmetry)
theta = np.deg2rad(12)
w_0 = wavelength/PI/theta  # Beam waist
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)

# ----------------------------------------------------------------------------------------------------------------------
# Define initial light field (Gaussian beam)
initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# ----------------------------------------------------------------------------------------------------------------------
# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)
# optical_system.add_element(PhasePlate(z_position=1, phase_function=lambda X, Y: np.exp(1j * np.arctan2(Y, X))))
f1 = 4*1e3
d1 = 500*1e3
f2 = 50*1e3
# f2 = 500
# |--f1--|ObjectLens|--f1--|------------d1----------------|----f2----Lens1----f2----|----f2----Lens2----f2----|--------------------------------------
optical_system.add_element(MomentumSpacePlate(z_position=0, modulation_function=lambda X, Y: np.exp(1j * 2 * np.arctan2(Y, X))))
optical_system.add_element(Lens(z_position=f1, focal_length=f1, NA=0.42))
optical_system.add_element(Lens(z_position=2*f1+d1+f2, focal_length=f2))
optical_system.add_element(Lens(z_position=2*f1+d1+3*f2, focal_length=f2))

# ----------------------------------------------------------------------------------------------------------------------
# Create plotter
plotter = Plotter(x, y)

# ----------------------------------------------------------------------------------------------------------------------
# Compute and Visualization
plot_cross_sections = True
plot_longitudinal_section = True

if plot_cross_sections:
    # Compute
    cross_z_positions = [0, 4*f2+2*f1+d1]
    cross_sections \
        = optical_system.propagate_to_cross_sections(cross_z_positions,
                                                     return_momentum_space_spectrum=True,
                                                     propagation_mode='Rigorous')  # Fresnel | Rigorous
    # Plot
    plotter.plot_cross_sections(cross_sections, save_label=f'{project_name}', show=False)

# another visualization mode
if plot_longitudinal_section:  # independently of cross_sections
    # Compute
    coord_axis, z_coords, intensity, phase = (
        optical_system.propagate_to_longitudinal_section(direction='x',
                                                         position=0.0,
                                                         num_z=512,
                                                         z_max=f1*2+d1+f2*4,
                                                         propagation_mode='Rigorous'))  # Fresnel | Rigorous

    # Plot
    plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase, save_label=f'{project_name}', show=False)
