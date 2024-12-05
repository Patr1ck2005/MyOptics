import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import *
from visualization.plotter import Plotter

# Define parameters
wavelength = 1.550  # Wavelength in micrometers
f1 = 4*1e3
f2 = 10*1e3
f3 = f2/2
w_0 = 200  # exp: 3000
initial_angle = wavelength/PI/w_0
# w_f = 10  # Beam waist
# focal_angle = np.arctan(w_f/f1)

w_m = 400
# sim_size = w_m*40  # Simulation size
sim_size = 4000  # Simulation size
mesh = 1024*4+1  # Mesh size ( +1 to maintain central symmetry)
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)

# ----------------------------------------------------------------------------------------------------------------------
# Define initial light field (Gaussian beam)
initial_field = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / w_0 ** 2)

# ----------------------------------------------------------------------------------------------------------------------
# Create optical system and add elements
optical_system = OpticalSystem(wavelength, x, y, initial_field)
# optical_system.add_element(PhasePlate(z_position=1, phase_function=lambda X, Y: np.exp(1j * np.arctan2(Y, X))))

# |--f1--|ObjectLens|--f1--*sample*--f1--|ObjectLens|--f1--|----f2----|Lens1|----f2----|----f2----|Lens2|----f2----|
# optical_system.add_element(SinePhaseGrating(z_position=0, period=10, amplitude=1))
optical_system.add_element(RectAmplitudeGrating(z_position=0, period=40, slit_width=20))
optical_system.add_element(OL1 := ObjectLens(z_position=f1, focal_length=f1))
# optical_system.add_element(PhasePlate(z_position=f1*2, phase_function=lambda X, Y: np.exp(1j * 2 * np.arctan2(Y, X))))
optical_system.add_element(MomentumSpacePhasePlate(z_position=f1*2, phase_function=lambda X, Y: np.exp(1j * 2 * np.arctan2(Y, X))))
optical_system.add_element(OL2 := ObjectLens(z_position=OL1.z_position+f1*2, focal_length=f1))
optical_system.add_element(L1 := Lens(z_position=OL2.z_position+f2*2, focal_length=f2))
optical_system.add_element(L2 := Lens(z_position=L1.back_position+f3, focal_length=f3))

# ----------------------------------------------------------------------------------------------------------------------
# Create plotter
plotter = Plotter(x, y)

# ----------------------------------------------------------------------------------------------------------------------
# Compute and Visualization
plot_cross_sections = True
plot_longitudinal_section = True

if plot_cross_sections:
    # Compute
    cross_z_positions = [0, OL1.back_position, OL2.back_position, L2.z_position+f2]
    cross_sections \
        = optical_system.propagate_to_cross_sections(cross_z_positions,
                                                     return_momentum_space_spectrum=True,
                                                     propagation_mode='Rigorous')  # Rigorous | Rigorous
    # Plot
    plotter.plot_cross_sections(cross_sections, save_label='exp-vortex_array', show=False)

# another visualization mode
if plot_longitudinal_section:  # independently of cross_sections
    # Compute
    coord_axis, z_coords, intensity, phase = (
        optical_system.propagate_to_longitudinal_section(direction='x',
                                                         position=0.0,
                                                         num_z=128*1,
                                                         z_max=L2.z_position+f2,
                                                         # z_max=+2*f1+f2,
                                                         propagation_mode='Rigorous'))  # Fresnel | Rigorous

    # Plot
    plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase,
                                      save_label='exp-vortex_array',
                                      show=False,
                                      norm_vmin=(1/np.e)/(w_m/w_0)**2)
