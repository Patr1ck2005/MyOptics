import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

def generate_universal_wave(
    X, Y,
    wavelength=0.1,
    curvature_radius=0.3,
    tilt_zenith=0.0,      # Tilt angle in radians (zenith angle)
    tilt_azimuth=0.0,     # Tilt angle in radians (azimuth angle)
    curvature_center_x=0.0,
    curvature_center_y=0.0,
    amplitude_noise_level=0.3,
    phase_noise_sigma=1.0,
    amplitude_correlation_length=10,
    phase_correlation_length=30,
    A=1.0
):
    """
    Generates a universal noise wave with adjustable tilt, curvature, and Gaussian speckle noise.

    Parameters:
    - X, Y: 2D meshgrid coordinates.
    - wavelength:
    - curvature_radius: Curvature radius of the spherical wave.
    - tilt_zenith: Tilt angle in radians from the zenith direction.
    - tilt_azimuth: Tilt angle in radians from the azimuth direction.
    - curvature_center_x, curvature_center_y: Relative offsets for the curvature center.
    - amplitude_noise_level: Scaling factor for amplitude noise (dimensionless).
    - phase_noise_sigma: Standard deviation for phase noise scaling.
    - amplitude_correlation_length: Spatial correlation length for amplitude Gaussian filtering (controls speckle size).
    - phase_correlation_length: Spatial correlation length for phase Gaussian filtering (controls speckle size).
    - A: Amplitude constant of the base wave.

    Returns:
    - E_noise: Complex field representing the noise wave.
    """
    # Calculate shifted coordinates for curvature center
    X_shifted = X - curvature_center_x
    Y_shifted = Y - curvature_center_y

    # Compute the radial distance from the curvature center
    R = np.sqrt(X_shifted**2 + Y_shifted**2)

    # Avoid division by zero at the curvature center
    R[R == 0] = 1e-10

    # Compute tilt components based on zenith and azimuth angles
    # Tilt is represented in spherical coordinates
    # tilt_phase = k * (sin(theta) * cos(phi) * X + sin(theta) * sin(phi) * Y)
    k = 2*np.pi/wavelength
    tilt_phase = k * (np.sin(tilt_zenith) * np.cos(tilt_azimuth) * X +
                     np.sin(tilt_zenith) * np.sin(tilt_azimuth) * Y)

    # Compute curvature phase
    curvature_phase = (k / (2 * curvature_radius)) * (X_shifted**2 + Y_shifted**2)

    # Total base phase
    base_phase = tilt_phase + curvature_phase

    # Generate amplitude noise: Gaussian speckle
    amplitude_noise = np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    amplitude_noise = gaussian_filter(amplitude_noise, sigma=amplitude_correlation_length)
    amplitude_noise = 1 + amplitude_noise_level * amplitude_noise  # Multiplicative factor

    # Generate phase noise: Gaussian speckle
    phase_noise = np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    phase_noise = gaussian_filter(phase_noise, sigma=phase_correlation_length)
    # Normalize phase noise to have unit variance and scale by phase_noise_sigma
    phase_noise = phase_noise_sigma * phase_noise / np.std(phase_noise)

    # Apply amplitude and phase noise
    E_noise = amplitude_noise * A * np.exp(1j * (base_phase + phase_noise))

    return E_noise

# ---------------------------
# Example Usage
# ---------------------------

def main():
    # ---------------------------
    # Step 1: Setup Grid and Parameters
    # ---------------------------
    N = 1024  # Image size (N x N)
    x = np.linspace(-1/2, 1/2, N)
    y = np.linspace(-1/2, 1/2, N)
    X, Y = np.meshgrid(x, y)

    # ---------------------------
    # Step 2: Define Main Interference Fields
    # ---------------------------

    # Standard spherical wave
    ref_wave = generate_universal_wave(
        X, Y,
        wavelength=0.01,
        curvature_radius=4,
        tilt_zenith=np.deg2rad(0),   # Example tilt angles
        tilt_azimuth=0,
        curvature_center_x=-1.0,  # Example curvature center offsets
        curvature_center_y=0.0,
        amplitude_noise_level=0.5,
        phase_noise_sigma=0.1,
        amplitude_correlation_length=10,
        phase_correlation_length=2,
        A=0.1
    )

    # Vortex beam parameters
    topological_charge = 2
    aperture_radius = 0.2
    R = np.sqrt((X)**2 + (Y)**2)
    Theta = np.arctan2((Y), (X))

    # Vortex beam definition
    wavelength = 0.01
    k = 2*np.pi/wavelength
    standard_spherical_phase = (k / 2) * R**2
    sigma = 0.15
    # radial_intensity = (R / sigma) ** 2 * np.exp(- (R / sigma) ** 2)
    radial_intensity = (R / sigma) ** 2
    vortex_beam_intensity = 0.2
    radial_intensity[radial_intensity > vortex_beam_intensity] = vortex_beam_intensity
    vortex_phase = topological_charge * Theta - standard_spherical_phase/10
    vortex_beam = radial_intensity * np.exp(1j * vortex_phase)
    vortex_beam[R > aperture_radius] = 0

    # ---------------------------
    # Step 3: Generate Universal Noise Wave
    # ---------------------------
    noise_wave_1 = generate_universal_wave(
        X, Y,
        wavelength=0.01,
        curvature_radius=1,
        tilt_zenith=0,   # Example tilt angles
        tilt_azimuth=0,
        curvature_center_x=0.5,  # Example curvature center offsets
        curvature_center_y=0.5,
        amplitude_noise_level=0.3,
        phase_noise_sigma=0.1,
        amplitude_correlation_length=3,
        phase_correlation_length=20,
        A=0.05
    )

    noise_wave_2 = generate_universal_wave(
        X, Y,
        wavelength=0.01,
        curvature_radius=np.inf,
        tilt_zenith=np.deg2rad(10),   # Example tilt angles
        tilt_azimuth=np.deg2rad(10),
        curvature_center_x=0.5,  # Example curvature center offsets
        curvature_center_y=0.5,
        amplitude_noise_level=0.3,
        phase_noise_sigma=0.5,
        amplitude_correlation_length=3,
        phase_correlation_length=7,
        A=0.05
    )

    # ---------------------------
    # Step 4: Combine Fields
    # ---------------------------
    interfered_field_noisy = ref_wave + vortex_beam + noise_wave_1 + noise_wave_2
    # interfered_field_noisy = ref_wave + vortex_beam
    # interfered_field_noisy = vortex_beam
    # interfered_field_noisy = ref_wave
    # interfered_field_noisy = noise_wave

    # ---------------------------
    # Step 5: Calculate Interference Intensity
    # ---------------------------
    interference_intensity_noisy = np.abs(interfered_field_noisy) ** 2
    # before_interference = np.abs(vortex_beam) ** 2
    # np.save('interference_pattern', interference_intensity_noisy)
    normalised_intensity = np.astype(255*interference_intensity_noisy/interference_intensity_noisy.max(), np.int8)
    Image.fromarray(normalised_intensity, mode='L').save("artificial_pattern.png")
    # interference_intensity_noisy = np.angle(ref_wave)

    # ---------------------------
    # Step 6: Visualization
    # ---------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(interference_intensity_noisy, cmap='gray', extent=(-1, 1, -1, 1), vmin=0)
    plt.title('Interference Intensity with Universal Noise Wave')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Intensity')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
