import matplotlib.pyplot as plt
import numpy as np

from optical_system.specific_elements import MSPP
from optical_system.system import OpticalSystem
from optical_system.elements import *
from visualization.plotter import Plotter

# Global configuration
project_name = 'vortex_beam-exp_system'

# Define parameters
# d0 = 45*1e3  # for no aperture
d0 = 25*1e3  # for aperture position
D_measure = 1000e3
f0 = 50*1e3
# fol = 4e3*5
fol = f0
f1 = 50*1e3
d1 = 500*1e3*1
f2 = 300*1e3
d3 = 100*1e3
f3 = 100*1e3


wavelength = 1.550*1  # Wavelength in um
mesh = 1024*4+1  # Mesh size ( +1 to maintain central symmetry)
theta0 = np.deg2rad(15)
w_0 = wavelength/PI/theta0  # Beam waist
w_ol = 25.4/2*1e3
sim_size = w_ol * 0.6
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)
mesh_size = x[1] - x[0]

# ----------------------------------------------------------------------------------------------------------------------
# Define initial light field (Gaussian beam)
from utils.beams import GaussianBeam
beam = GaussianBeam(wavelength=wavelength, waist_radius=w_0)
initial_field = beam.compute_field(z_position=d0, x=x, y=y)
# plt.imshow(np.abs(initial_field)**2*np.angle(initial_field))
# plt.show()
# initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# ----------------------------------------------------------------------------------------------------------------------
# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)
optical_system.add_element(aperture := CircularAperture(z_position=d0-d0, radius=0.75*1e3*2))
optical_system.add_element(lens0 := Lens(z_position=f0-d0, focal_length=f0, D=25.4*1e3))
# optical_system.add_element(PhasePlate(z_position=1, phase_function=lambda X, Y: np.exp(1j * np.arctan2(Y, X))))
# source|--f0-A-|Lens0|-f0-|-----d1-----|--4--|ObjectLens|--4--|Sample|--4--|ObjectLens|--4--|----f2----|Lens1|----f2----|--d3--|Lens2|--d3--|
optical_system.add_element(obj_lens1 := ObjectLens(z_position=lens0.back_position+d1+fol, focal_length=fol, NA=0.42))
optical_system.add_element(MSPP(z_position=obj_lens1.z_position+1e3, wavelength=wavelength))
optical_system.add_element(obj_lens2 := ObjectLens(z_position=obj_lens1.back_position+fol, focal_length=fol, NA=0.42))
# optical_system.add_element(obj_lens := Lens(z_position=lens0.back_position+d1, focal_length=f1, D=25.4*1e3))
# optical_system.add_element(lens3 := Lens(z_position=obj_lens.z_position+f1+f2, focal_length=f2, D=25.4*1e3))
# lens_focus_position = lens3.z_position+f2
# optical_system.add_element(lens4 := Lens(z_position=lens_focus_position+d3, focal_length=f3, D=25.4*1e3))

# ----------------------------------------------------------------------------------------------------------------------
# Create plotter
plotter = Plotter(x, y, wavelength=wavelength)

# ----------------------------------------------------------------------------------------------------------------------
z_max = optical_system.elements[-1].back_position
# z_max = lens0.back_position+1.0*d1
# Compute and Visualization
plot_cross_sections = True
plot_longitudinal_section = False

if plot_cross_sections:
    # Compute
    cross_z_positions = [lens0.z_position, lens0.back_position, obj_lens1.forw_position,
                         obj_lens1.back_position, obj_lens2.back_position, obj_lens2.back_position+D_measure]
    # cross_z_positions = [lens0.back_position, lens0.back_position+0.3*d1, lens0.back_position+0.6*d1, lens0.back_position+1.0*d1]
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
                                                         # num_z=64,
                                                         z_max=z_max,
                                                         propagation_mode='Rigorous'))  # Fresnel | Rigorous

    # Plot
    plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase,
                                      save_label=f'{project_name}',
                                      show=True, ref_position_min=0.6, ref_position_max=0.6,
                                      ref_multiplier_max=3)
