import numpy as np
import matplotlib.pyplot as plt

class Layer:
    """Represents a single layer with complex permittivity and finite or semi-infinite thickness."""
    def __init__(self, eps: complex, thickness: float = None):
        """
        Args:
            eps: Complex dielectric constant of the layer.
            thickness: Physical thickness (None for semi-infinite).
        """
        self.eps = eps
        self.d = thickness

class MultiLayerTM:
    """Handles TM‐polarized wave propagation through a stack of layers."""
    def __init__(self, layers, eps_incident=1.0, wavelength=365, kx=2*2*np.pi/365):
        """
        Args:
            layers: list of Layer instances (last layer may be semi-infinite if d=None).
            eps_incident: Permittivity of the incidence medium.
            wavelength: Free-space wavelength.
            kx: Transverse wavevector component.
        """
        self.layers = layers
        self.eps0 = eps_incident
        self.wl = wavelength
        self.k0 = 2*np.pi / wavelength
        self.kx = kx
        self._compute_kzs()

    def _compute_kzs(self):
        """Compute longitudinal wavevector k_z in each layer (including incident side)."""
        self.kz0 = np.sqrt(self.eps0*self.k0**2 - self.kx**2, dtype=complex)
        self.kz = [np.sqrt(layer.eps*self.k0**2 - self.kx**2, dtype=complex)
                   for layer in self.layers]

    @staticmethod
    def _M_tm(eps, kz, d):
        """Single‐layer TM transfer matrix over thickness d."""
        phi = kz * d
        c = np.cos(phi)
        s = np.sin(phi)
        return np.array([
            [  c,         eps/kz * s],
            [-kz/eps * s,      c     ]
        ], dtype=complex)

    def _compute_global_M(self):
        """Multiply all finite‐thickness layer matrices to get M_total."""
        M = np.eye(2, dtype=complex)
        for layer, kz in zip(self.layers, self.kz):
            if layer.d is not None:
                M = self._M_tm(layer.eps, kz, layer.d) @ M
        return M

    def reflection_coefficient(self):
        """Compute TM reflection coefficient r."""
        # 1. Incident‐side admittance
        a0 = 1j*self.kz0/self.eps0
        # 2. Global transfer matrix
        M00, M01, M10, M11 = self._compute_global_M().ravel()
        # 3. Semi-infinite last layer boundary condition: Q/H = i*kz/eps
        epsN, kzN = self.layers[-1].eps, self.kz[-1]
        cN = 1j * kzN/epsN
        # 4. Build alpha, beta
        alpha = M10 - cN*M00
        beta  = M11 - cN*M01
        # 5. Solve for r
        r = -(alpha + beta*a0)/(alpha - beta*a0)
        return r

    def field_profile(self, num_points=600, z_max_factor=1.5):
        """
        Sample |H(z)| over the structure + into the semi-infinite last layer.
        Returns (z_array, H_array).
        """
        # Build cumulative interfaces
        thicknesses = [layer.d for layer in self.layers if layer.d is not None]
        interfaces = np.concatenate(([0], np.cumsum(thicknesses)))
        z_max = interfaces[-1] * z_max_factor
        z = np.linspace(0, z_max, num_points)

        # First compute A_i, B_i
        r = self.reflection_coefficient()
        a0 = 1j*self.kz0/self.eps0
        # S0 = [H; Q] at z=0: H=1+r, Q=a0*(1-r)
        S = np.array([1+r, a0*(1-r)], dtype=complex)

        # propagate to each interface
        Ss = [S]
        for layer, kz in zip(self.layers[:-1], self.kz[:-1]):
            M = (self._M_tm(layer.eps, kz, layer.d) if layer.d is not None else np.eye(2))
            S = M @ S
            Ss.append(S)

        # solve for A,B in each finite layer
        As, Bs = [], []
        for (layer, kz, S_prev) in zip(self.layers[:-1], self.kz[:-1], Ss[:-1]):
            H0, Q0 = S_prev
            eps, kzi = layer.eps, kz
            A = 0.5*(H0 + eps/(1j*kzi)*Q0)
            B = 0.5*(H0 - eps/(1j*kzi)*Q0)
            As.append(A); Bs.append(B)
        # last semi-infinite layer
        As.append(Ss[-1][0]); Bs.append(0)

        # now sample H(z)
        Hz = np.zeros_like(z, dtype=complex)
        for idx, zi in enumerate(z):
            if zi < interfaces[-1]:
                # 有限层
                layer_idx = np.searchsorted(interfaces[1:], zi, side='right')
                zloc = zi - interfaces[layer_idx]
                H = (As[layer_idx] * np.exp(1j * self.kz[layer_idx] * zloc)
                     + Bs[layer_idx] * np.exp(-1j * self.kz[layer_idx] * zloc))
            else:
                # 半无限层
                zloc = zi - interfaces[-1]
                H = As[-1] * np.exp(1j * self.kz[-1] * zloc)
            Hz[idx] = H
        return z, Hz

    def plot_field(self, **plt_kwargs):
        """Convenience to plot |H(z)| vs z."""
        z, H = self.field_profile()
        plt.figure()
        plt.plot(z, np.abs(H), **plt_kwargs)
        plt.xlabel('z')
        plt.ylabel(r'$|H(z)|$')
        plt.title('Field Magnitude Profile')
        plt.show()

    def scan_parameter(self, param_name, param_values):
        """
        Scan reflectance over a parameter (e.g. 'kx' or 'wl').
        Returns (values, reflectances).
        """
        results = []
        orig = getattr(self, param_name)
        for val in param_values:
            setattr(self, param_name, val)
            self._compute_kzs()
            results.append(abs(self.reflection_coefficient())**2)
        setattr(self, param_name, orig)
        self._compute_kzs()
        return np.array(param_values), np.array(results)
