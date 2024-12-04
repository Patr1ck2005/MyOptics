import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import *
from visualization.plotter import Plotter

# Define parameters
wavelength = 0.550  # Wavelength in micrometers
f1 = 100*1e3
f2 = 300*1e3
d1 = 100*1e3
w_f = 0.1*1e3  # Beam waist
initial_angle = wavelength/PI/w_f
focal_angle = np.arctan(w_f/f1)
w_0 = wavelength/PI/focal_angle

w_m = w_f*20
sim_size = w_m*5  # Simulation size
mesh = 1024*10+1  # Mesh size ( +1 to maintain central symmetry)
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)

# ----------------------------------------------------------------------------------------------------------------------
# Define initial light field (Gaussian beam)
initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# ----------------------------------------------------------------------------------------------------------------------
# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)
# optical_system.add_element(PhasePlate(z_position=1, phase_function=lambda X, Y: np.exp(1j * np.arctan2(Y, X))))

# |--f1--|ObjectLens|--f1--|d1|----f2----|Lens1|----f2----|----f2--|Lens2|----f2--|-----d2---------------------------------
# optical_system.add_element(BlazedGrating(z_position=0, period=1000/600, blaze_angle=np.deg2rad(17.46)))
optical_system.add_element(RectangularAperture(z_position=0, width=w_0/2, height=w_0/2))
optical_system.add_element(ObjectLens(z_position=f1, focal_length=f1))
optical_system.add_element(len1 := Lens(z_position=f2+2*f1+d1, focal_length=f2))
optical_system.add_element(len2 := Lens(z_position=f2*2+2*f1+f2+d1, focal_length=f2))

# ----------------------------------------------------------------------------------------------------------------------
# Create plotter
plotter = Plotter(x, y)

# ----------------------------------------------------------------------------------------------------------------------
# Compute and Visualization
plot_cross_sections = True
plot_longitudinal_section = True

if plot_cross_sections:
    # Compute
    cross_z_positions = [0, 2*f1, len1.z_position-f2, len2.z_position+f2]
    cross_sections \
        = optical_system.propagate_to_cross_sections(cross_z_positions,
                                                     return_momentum_space_spectrum=True,
                                                     propagation_mode='Rigorous')  # Rigorous | Rigorous
    # Plot
    plotter.plot_cross_sections(cross_sections, save_label='test', show=False)

# another visualization mode
if plot_longitudinal_section:  # independently of cross_sections
    # Compute
    coord_axis, z_coords, intensity, phase = (
        optical_system.propagate_to_longitudinal_section(direction='x',
                                                         position=0.0,
                                                         num_z=128*5,
                                                         z_max=len2.z_position+f2,
                                                         propagation_mode='Rigorous'))  # Fresnel | Rigorous

    # Plot
    plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase,
                                      save_label='test',
                                      show=True,
                                      norm_vmin=(1/np.e)/(w_m/w_0)**2)
