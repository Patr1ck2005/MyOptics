"""超透镜传函仿真示例：ZnO/Ag/ZnO 膜堆作为动量空间滤波器。

演示 MultilayerSlab 的完整工作流：
1. 查看 1D 径向传函 H(kr)（传播带透过率 + 倏逝带增强）；
2. 亚波长缝隙（含高频倏逝分量）的场透过膜堆后的近场分布；
3. 与"理想无膜堆传播"对比，展示膜堆对高频分量的选择性。

物理背景：365nm 波长下 ZnO/Ag/ZnO 膜堆的 p 偏振倏逝带增强可把
亚衍射极限的空间频率信息从物面搬运到像面——超透镜成像的核心机制。
"""
import numpy as np

from optical_system.elements import MultilayerSlab
from optical_system.elements_cls import SpatialPlate
from optical_system.system import OpticalSystem
from utils.constants import PI

import cupy as cp

project_name = 'multilayer-superlens-transfer'

# ----------------------------------------------------------------------------
# 参数
wavelength = 0.365  # μm（i-line 紫外）
wl_nm = wavelength * 1000.0

# 空气 | ZnO | Ag | ZnO | 空气（365nm 材料参数，可替换为实测 nk）
N_ZNO = complex(2.56, 0.0)
N_AG = complex(-2.6115, 0.4431)  # 365nm 银的复折射率（n<0 约定见 TMM）
stack = dict(
    N_layers=[N_ZNO, N_AG, N_ZNO],
    d_nm_list=[10.0, 40.0, 10.0],  # ZnO10/Ag40/ZnO10（与历史超透镜工作同量级）
)

# 仿真网格（单位 μm）
sim_size = 6.0
mesh = 1024 + 1
x = np.linspace(-sim_size, sim_size, mesh)
y = np.linspace(-sim_size, sim_size, mesh)

# ----------------------------------------------------------------------------
# 1. 1D 径向传函分析
slab_s = MultilayerSlab(z_position=1.0, wavelength=wavelength, polarization='s', **stack)
slab_p = MultilayerSlab(z_position=1.0, wavelength=wavelength, polarization='p', **stack)
kr, H_s = slab_s.radial_H()
_, H_p = slab_p.radial_H()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

k0 = 2 * PI / wl_nm
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.semilogx(kr / k0, np.abs(H_s), label='|H_s|')
ax1.semilogx(kr / k0, np.abs(H_p), label='|H_p|')
ax1.axvline(1.0, color='gray', linestyle=':', label='k = k0 (光锥边界)')
ax1.set_xlabel('kr / k0')
ax1.set_ylabel('|H(kr)| (field amplitude ratio)')
ax1.set_title('ZnO10/Ag40/ZnO10 @ 365nm radial transfer function')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.semilogx(kr / k0, 20 * np.log10(np.abs(H_p) + 1e-12), label='p')
ax2.semilogx(kr / k0, 20 * np.log10(np.abs(H_s) + 1e-12), label='s')
ax2.axvline(1.0, color='gray', linestyle=':')
ax2.set_xlabel('kr / k0')
ax2.set_ylabel('|H| (dB)')
ax2.set_title('dB view (evanescent enhancement visible)')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'./img/{project_name}-transfer_function.png', dpi=150)
plt.close()
print(f"传函图已保存：{project_name}-transfer_function.png")
print(f"p 偏振倏逝带最大 |H| = {np.abs(H_p[(kr > k0) & (kr < 2.2 * k0)]).max():.2f}")

# ----------------------------------------------------------------------------
# 2. 亚波长结构成像演示：物面 = 双缝（缝宽 0.3λ，间距 0.8λ，均亚波长）
slit_w = 0.3 * wavelength
slit_sep = 0.8 * wavelength
X, Y = np.meshgrid(x, y)


def double_slat(X):
    return ((np.abs(X + slit_sep / 2) < slit_w / 2) |
            (np.abs(X - slit_sep / 2) < slit_w / 2)).astype(float)


object_field = double_slat(X)

system = OpticalSystem(wavelength, x, y, object_field)
# 物面贴膜堆：场直接经传函滤波（膜堆在 z=1.0，前面留 1μm 传播）
system.add_element(slab_p)
system.add_element(SpatialPlate(z_position=1.0 + 0.01,  # 占位：无操作
                                modulation_function=lambda X_, Y_: cp.ones_like(X_)))

# 观察 z=0（物面）、z=1.0（膜堆后紧贴）、z=1.0+0.2（近场演化）
cross = system.propagate_to_cross_sections(
    [0.0, 1.0, 1.2], return_momentum_space_spectrum=True, propagation_mode='Rigorous')

fig2, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, z in zip(axes, [0.0, 1.0, 1.2]):
    (U, xr, yr), _ = cross[z]
    if z == 0.0:
        img = np.abs(U) ** 2
    else:
        img = np.abs(U) ** 2
    vmax = img.max() if z == 0.0 else max(img.max(), 1e-12)
    im = ax.imshow(img, extent=[xr.min(), xr.max(), yr.min(), yr.max()],
                   origin='lower', cmap='inferno',
                   aspect='equal', vmax=vmax)
    ax.set_title(f'|U|^2 at z={z:.2f} um (p-pol)')
    ax.set_xlabel('x (μm)')
    plt.colorbar(im, ax=ax)
axes[0].set_ylabel('y (μm)')
plt.tight_layout()
plt.savefig(f'./img/{project_name}-nearfield_evolution.png', dpi=150)
plt.close()
print(f"近场演化图已保存：{project_name}-nearfield_evolution.png")
print("完成。观察要点：亚波长双缝的高频倏逝分量在透过膜堆后是否被部分保留，")
print("以及两缝间的近场耦合随 z 的演化。")
