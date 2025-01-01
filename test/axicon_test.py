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
D_measure = 1000e3*1
f0 = 50*1e3
fol = 4e3*10
# fol = f0
d1 = 50*1e3*0
f1 = 200*1e3
d2 = 30*1e3*1
f2 = 300*1e3
d3 = 100*1e3
f3 = 100*1e3


wavelength = 1.550*1  # Wavelength in um
mesh = 1024*4+1  # Mesh size ( +1 to maintain central symmetry)
# theta0 = np.deg2rad(15)
# w_0 = wavelength/PI/theta0  # Beam waist
w_ol = 25.4/2*1e3
aperture_radius = 1*1e3*0.5
sim_size = aperture_radius * 10
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)
mesh_size = x[1] - x[0]

# ----------------------------------------------------------------------------------------------------------------------
# Define initial light field (Gaussian beam)
from utils.beams import GaussianBeam
beam = GaussianBeam(wavelength=wavelength, waist_radius=25.7*1e3/2)
initial_field = beam.compute_field(z_position=d0, x=x, y=y)
# plt.imshow(np.abs(initial_field)**2*np.angle(initial_field))
# plt.show()
# initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# ----------------------------------------------------------------------------------------------------------------------
# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)
optical_system.add_element(aperture := CircularAperture(z_position=0, radius=aperture_radius))
optical_system.add_element(axicon := Axicon(z_position=d1, base_angle=np.deg2rad(1)))
# A-----d1-----|Axicon|---d2---|--f1--|Lens|--f1--|--4--|ObjectLens|--4--|Sample|--4--|ObjectLens|--4--|----f2----|Lens1|----f2----|--d3--|Lens2|--d3--|
optical_system.add_element(lens1 := Lens(z_position=axicon.z_position+d2, focal_length=f1))
optical_system.add_element(obj_lens1 := ObjectLens(z_position=lens1.back_position+fol, focal_length=fol, NA=0.42))
# optical_system.add_element(mspp := MSPP(z_position=obj_lens1.z_position+1e3, wavelength=wavelength))
optical_system.add_element(obj_lens2 := ObjectLens(z_position=obj_lens1.back_position+fol, focal_length=fol, NA=0.42))

# ----------------------------------------------------------------------------------------------------------------------
# Create plotter
plotter = Plotter(x, y, wavelength=wavelength)

# ----------------------------------------------------------------------------------------------------------------------
# z_max = obj_lens2.back_position
z_max = obj_lens1.forw_position
# Compute and Visualization
plot_cross_sections = True
plot_longitudinal_section = False

if plot_cross_sections:
    # Compute
    cross_z_positions = [0, axicon.z_position, obj_lens1.forw_position,
                         obj_lens1.back_position, obj_lens2.back_position]
    cross_sections \
        = optical_system.propagate_to_cross_sections(cross_z_positions,
                                                     return_momentum_space_spectrum=True,
                                                     propagation_mode='Rigorous')  # Fresnel | Rigorous
    # Plot
    plotter.plot_cross_sections(cross_sections,
                                save_label=f'{project_name}',
                                show=False,
                                cmap_for_spatial_intensity='grey',
                                vmax_for_spatial_intensity=0.2e-7,
                                )

# another visualization mode
if plot_longitudinal_section:  # independently of cross_sections
    # Compute
    coord_axis, z_coords, intensity, phase = (
        optical_system.propagate_to_longitudinal_section(direction='x',
                                                         position=0.0,
                                                         num_z=128,
                                                         # num_z=64,
                                                         z_max=z_max,
                                                         propagation_mode='Rigorous'))  # Fresnel | Rigorous

    # Plot
    plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase,
                                      save_label=f'{project_name}',
                                      show=True, ref_position_min=0.6, ref_position_max=0.6,
                                      ref_multiplier_max=3)
