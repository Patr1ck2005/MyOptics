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
