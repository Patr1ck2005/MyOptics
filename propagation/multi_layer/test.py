from cls import *

# 1) Define your stack
layers = [
    Layer(2.56, 10),
    Layer(-2.6115+0.4431j, 10),
    Layer(2.7640+0.1808j, 15),
    Layer(-2.6194+0.4551j, 40),
    Layer(2.43, None),  # last layer semi-infinite
]

# 2) Create the solver
solver = MultiLayerTM(layers, eps_incident=1.0, wavelength=365, kx=2*2*np.pi/365)

# 3) Get reflection coefficient
r = solver.reflection_coefficient()
print("r =", r)

# 4) Plot |H(z)|
solver.plot_field()

# 5) Scan reflectance vs kx
kx_vals = np.linspace(0, 2.5*solver.k0, 200)
kx_vals, R_vals = solver.scan_parameter('kx', kx_vals)

plt.figure()
plt.plot(kx_vals, R_vals)
plt.xlabel(r'$k_x$')
plt.ylabel('Reflectance R')
plt.show()
