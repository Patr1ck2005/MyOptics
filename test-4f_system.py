import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import *
from visualization.plotter import Plotter

# Define parameters
wavelength = 0.5  # Wavelength in micrometers
theta = np.deg2rad(2)
w_0 = wavelength/PI/theta  # Beam waist
f1 = 100
f2 = 3000
w_oj = f1*np.tan(45)
sim_size = 10*w_oj  # Simulation size
mesh = 1024*10+1  # Mesh size ( +1 to maintain central symmetry)
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)

# ----------------------------------------------------------------------------------------------------------------------
# Define initial light field (Gaussian beam)
initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# ----------------------------------------------------------------------------------------------------------------------
# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)

optical_system.add_element(Grating(z_position=0, period=1, amplitude=1))
optical_system.add_element(ObjectLens(z_position=f1, focal_length=f1))
optical_system.add_element(Lens(z_position=f2+2*f1, focal_length=f2))
optical_system.add_element(Lens(z_position=f2*2+2*f1+f2, focal_length=f2))

# ----------------------------------------------------------------------------------------------------------------------
# Create plotter
plotter = Plotter(x, y)

# ----------------------------------------------------------------------------------------------------------------------
# Compute and Visualization
plot_cross_sections = True
plot_longitudinal_section = True

if plot_cross_sections:
    # Compute
    cross_z_positions = [0, 2*f1, f2*2+2*f1, 4*f2+2*f1]
    cross_sections \
        = optical_system.propagate_to_cross_sections(cross_z_positions,
                                                     return_momentum_space_spectrum=True,
                                                     propagation_mode='Fresnel')  # Rigorous | Rigorous
    # Plot
    plotter.plot_cross_sections(cross_sections, save_label='test', show=False)

# another visualization mode
if plot_longitudinal_section:  # independently of cross_sections
    # Compute
    coord_axis, z_coords, intensity, phase = (
        optical_system.propagate_to_longitudinal_section(direction='x',
                                                         position=0.0,
                                                         num_z=512,
                                                         z_max=f2*2+2*f1+f2*2,
                                                         propagation_mode='Rigorous'))  # Fresnel | Rigorous

    # Plot
    plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase,
                                      save_label='test',
                                      show=True,
                                      norm_vmin=(1/np.e)/(w_oj/w_0)**2)
