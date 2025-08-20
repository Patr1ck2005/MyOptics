# 一维 LSF（来自 1D 逆傅里叶）
import matplotlib.pyplot as plt
import numpy as np
from cls import *

# layers = [
#         Layer(2.56, 10),
#         Layer(-2.6115 + 0.4431j, 10),
#         Layer(2.7640 + 0.1808j, 15),
#         Layer(-2.6194 + 0.4551j, 40),
#         Layer(2.43, None),
#     ]
#
# solver = MultiLayerTM(layers, eps_incident=2.56, wavelength=365)
#
# z0 = 75.0
# kx_vals = np.linspace(- 30 * solver.k0, 30 * solver.k0, 128)
# x, l_amp, lsf_int, fwhm_lsf, xL_lsf, xR_lsf = solver.lsf_from_Hkx(z0, kx_vals, window='hann', normalize='peak')
# print("LSF FWHM =", fwhm_lsf)
#
# # 径向 PSF（极坐标、0 阶 Hankel 逆变换）
# r, h_amp_r, psf_r, fwhm_psf, rL, rR = solver.psf_from_kx_polar(z0, kx_vals, window='hann', normalize='peak',
#                                                                num_r=4096)
# print("PSF (radial) FWHM =", fwhm_psf)
#
# # 简单画图
# plt.figure()
# plt.plot(x, lsf_int, marker='+')
# plt.xlabel('x')
# plt.ylabel('LSF (intensity)')
# plt.title('LSF at z={}'.format(z0))
# # plt.xlim((-100, 100))
#
#
# plt.figure()
# plt.plot(r, psf_r, marker='+')
# plt.xlabel('r')
# plt.ylabel('PSF (intensity)')
# plt.title('PSF (radial) at z={}'.format(z0))
# plt.show()

fwhm_psf_lst = []
fwhm_lsf_lst = []

die_epsilon = 2.7640
die_thickness = 15
plas1_thickness = 10
plas2_thickness = 40
die_epsilon_lst = np.linspace(2.0, 3.0, 32)
plas_epsilon_real_lst = np.linspace(-3.0, -2.2, 32)
plas_epsilon_imag_lst = np.linspace(0.3, 0.6, 32)
die_thickness_lst = np.linspace(0, 40, 32)
plas1_thickness_lst = np.linspace(0, 20, 32)
plas2_thickness_lst = np.linspace(0, 60, 32)

# para_lst = die_epsilon_lst
# para_lst = plas_epsilon_real_lst
# para_lst = plas_epsilon_imag_lst
# para_lst = die_thickness_lst
# para_lst = plas1_thickness_lst
para_lst = plas2_thickness_lst

# para_label = 'die_epsilon'
# para_label = 'plas_epsilon_real'
# para_label = 'plas_epsilon_imag'
# para_label = 'die_thickness_lst'
# para_label = 'plas1_thickness'
para_label = 'plas2_thickness'

# for die_epsilon in die_epsilon_lst:
# for plas_epsilon_real in plas_epsilon_real_lst:
# for plas_epsilon_imag in plas_epsilon_imag_lst:
# for die_thickness in die_thickness_lst:
# for plas1_thickness in plas1_thickness_lst:
for plas2_thickness in plas2_thickness_lst:

    layers = [
        Layer(2.56, 10),
        # Layer(plas_epsilon_real + 0.4431j, plas1_thickness),
        Layer(-2.6115 + 0.4431j, plas1_thickness),
        # Layer(-2.6115 + plas_epsilon_imag*1j, plas1_thickness),
        Layer(die_epsilon + 0.1808j, die_thickness),
        # Layer(plas_epsilon_real + 0.4551j, plas2_thickness),
        Layer(-2.6194 + 0.4551j, plas2_thickness),
        # Layer(-2.6194 + plas_epsilon_imag*1j, plas2_thickness),
        Layer(2.43, None),
    ]

    solver = MultiLayerTM(layers, eps_incident=2.56, wavelength=365)

    z0 = 75.0
    kx_vals = np.linspace(-30 * solver.k0, 30 * solver.k0, 128)
    x, l_amp, lsf_int, fwhm_lsf, xL_lsf, xR_lsf = solver.lsf_from_Hkx(z0, kx_vals, window='hann', normalize='peak')
    print("LSF FWHM =", fwhm_lsf)

    # 径向 PSF（极坐标、0 阶 Hankel 逆变换）
    r, h_amp_r, psf_r, fwhm_psf, rL, rR = solver.psf_from_kx_polar(z0, kx_vals, window='hann', normalize='peak',
                                                                   num_r=4096)
    print("PSF (radial) FWHM =", fwhm_psf)

    fwhm_psf_lst.append(fwhm_psf)
    fwhm_lsf_lst.append(fwhm_lsf)

plt.figure(figsize=(4, 6))
# 第一个子图
plt.subplot(111)
plt.plot(para_lst, fwhm_lsf_lst, label='FWHM of LSF', color='red', marker='+')
plt.plot(para_lst, fwhm_psf_lst, label='FWHM of PSF', color='blue', marker='+')
plt.legend(loc='upper right')
plt.xlabel(f'{para_label}')
plt.ylabel('FWHM')
plt.title(f'LSF&PSF FWHM vs. {para_label}')
plt.grid()
# plt.ylim((0, 100))
# 显示图形
plt.tight_layout()  # 自动调整子图之间的间距
plt.savefig(f'FWHM-vs.-{para_label}.png', dpi=300)
plt.show()