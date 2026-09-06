"""径向偏振高NA聚焦示例：纵向分量主导的紧焦点。

物理背景：径向偏振光束经高NA物镜聚焦后，纵向电场分量 Ez 在焦点
占据主导（Dorn/Quabis/Leuchs PRL 2003）——这是矢量光学最标志性的
效应，标量框架无法描述。

演示两条等价物理路径并交叉对比：
1. RichardsWolfFocuser（Debye-Wolf 积分，光瞳坐标 f·sinθ）；
2. VectorLens（薄透镜相位 + apodization）+ 矢量角谱传播。
NA=0.85 时两者同为高NA聚焦模型，焦斑强度结构高度一致。

运行: python examples/radial_polarization_focusing.py
输出: img/radial_polarization_focusing.png + 控制台验证数字
"""
import numpy as np

from utils.cuda_path import ensure_cuda_dll_dirs

ensure_cuda_dll_dirs()   # pip 轮子 CUDA DLL 发现（Windows 必需），再导入 cupy

import cupy as cp

from vector.elements import VectorLens
from vector.propagator import VectorAngularSpectrumPropagator
from vector.richards_wolf import RichardsWolfFocuser
from vector.sources import vector_gaussian

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

project_name = 'radial-polarization-focusing'

# ----------------------------------------------------------------------------
# 参数
wavelength = 0.5      # μm
focal_length = 10.0   # μm
NA = 0.85
waist = 8.0           # μm（光瞳内近均匀照明的宽高斯）
mesh = 256
extent = 17.0

x = cp.arange(mesh, dtype=cp.float64) * (extent / mesh) - extent / 2
y = x

# ----------------------------------------------------------------------------
# 1. Richards-Wolf 路径
src = vector_gaussian(x, y, wavelength, waist, 'radial').normalized()
rw = RichardsWolfFocuser(x, y, wavelength, focal_length, NA)
E_rw = rw.focus(src)

# 2. VectorLens + 矢量角谱传播路径
sys_field = src.copy()
lens = VectorLens(0.0, focal_length, NA, apodization=True)
pupil = lens.apply(sys_field)
prop = VectorAngularSpectrumPropagator(x, y, wavelength)
E_vp = prop.propagate(pupil, focal_length)

# ----------------------------------------------------------------------------
# 3. 验证数字
for name, E in (('Richards-Wolf', E_rw), ('VectorLens+ASP', E_vp)):
    Px = float(cp.sum(cp.abs(E.ex) ** 2).get())
    Py = float(cp.sum(cp.abs(E.ey) ** 2).get())
    Pz = float(cp.sum(cp.abs(E.ez) ** 2).get())
    print(f"[{name}] 焦平面能量分数: Ex={Px:.3f} Ey={Py:.3f} Ez={Pz:.3f}"
          f"  (Ez 占比 {Pz / (Px + Py + Pz):.1%})")
    I = E.intensity()
    ny, nx = I.shape
    print(f"[{name}] 轴上 |Ez|={float(cp.abs(E.ez[ny // 2, nx // 2]).get()):.3f}"
          f"  轴上 |Ex|={float(cp.abs(E.ex[ny // 2, nx // 2]).get()):.2e}"
          f"（对称性归零）")

I_rw = cp.asnumpy(E_rw.intensity())
I_vp = cp.asnumpy(E_vp.intensity())
corr = float(np.corrcoef(I_rw.ravel(), I_vp.ravel())[0, 1])
print(f"两路径焦斑强度图相关系数: {corr:.4f}")

# ----------------------------------------------------------------------------
# 4. 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
ex_np = cp.asnumpy(cp.abs(E_rw.ex) ** 2)
ey_np = cp.asnumpy(cp.abs(E_rw.ey) ** 2)
ez_np = cp.asnumpy(cp.abs(E_rw.ez) ** 2)
vmax = max(ex_np.max(), ey_np.max(), ez_np.max())

for ax, img, title in zip(
        axes[0],
        (ex_np, ey_np, ez_np),
        (r'$|E_x|^2$ RW', r'$|E_y|^2$ RW', r'$|E_z|^2$ RW')):
    im = ax.imshow(img, extent=[float(x[0]), float(x[-1])] * 2,
                   origin='lower', cmap='inferno', vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)

axes[1, 0].imshow(I_vp, extent=[float(x[0]), float(x[-1])] * 2,
                  origin='lower', cmap='inferno')
axes[1, 0].set_title('Total intensity, VectorLens+ASP path')
axes[1, 1].plot(cp.asnumpy(x), I_rw[I_rw.shape[0] // 2, :], label='RW total (x-line)')
axes[1, 1].plot(cp.asnumpy(x), I_vp[I_vp.shape[0] // 2, :], '--',
                label='VectorLens+ASP total (x-line)')
axes[1, 1].set_xlabel('x (μm)')
axes[1, 1].legend()
axes[1, 1].set_title('Focal-plane cross section')
axes[1, 2].axis('off')
axes[1, 2].text(0.05, 0.6,
                f'NA = {NA}, f = {focal_length} μm, λ = {wavelength} μm\n'
                f'radial polarization focus\n'
                f'RW vs VectorLens corr = {corr:.4f}\n'
                f'Ez energy fraction (RW) = '
                f'{float(cp.sum(cp.abs(E_rw.ez) ** 2).get()) / I_rw.sum():.1%}',
                fontsize=12, family='monospace')

fig.suptitle('Radially polarized high-NA focusing (longitudinal-field dominant)')
fig.tight_layout()
import os
os.makedirs('img', exist_ok=True)
fig.savefig(f'img/{project_name}.png', dpi=150)
print(f"图已保存: img/{project_name}.png")
