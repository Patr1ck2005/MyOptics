"""q-plate 矢量涡旋示例：自旋-轨道耦合生成径向/方位偏振涡旋光束。

物理背景：q-plate 是快轴角 α(φ) = qφ + α0 的空间变向半波片。圆偏振
光入射时发生自旋-轨道耦合——自旋角动量 ±ħ 转换为轨道角动量 ±2qħ，
输出相反手性、拓扑荷 ±2q 的涡旋（|L⟩ → e^{+2iα}|R⟩）；线偏振入射
（|L⟩+|R⟩ 叠加）则得到径向/方位型矢量涡旋。

演示（q = 1/2, α0 = 0）：
1. x 线偏振输入 → 径向型矢量涡旋（局部偏振平行于 r̂，含 e^{iφ} 几何相位）；
2. RCP 输入 → 拓扑荷 −1 的标量涡旋（相位缠绕验证）；
3. 矢量传播后的远场衍射图案（矢量涡旋的自相关甜甜圈结构）。

运行: python examples/qplate_vector_vortex.py
输出: img/qplate_vector_vortex.png + 控制台验证数字
"""
import numpy as np

from utils.cuda_path import ensure_cuda_dll_dirs

ensure_cuda_dll_dirs()   # pip 轮子 CUDA DLL 发现（Windows 必需），再导入 cupy

import cupy as cp

from vector.elements import QPlate
from vector.propagator import VectorAngularSpectrumPropagator
from vector.sources import vector_gaussian

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

project_name = 'qplate-vector-vortex'

# ----------------------------------------------------------------------------
# 参数
wavelength = 0.5      # μm
waist = 4.0           # μm
q = 0.5               # 拓扑荷（q=1/2 → 输出涡旋荷 ±1）
mesh = 256
extent = 16.0
z_prop = 15.0         # 衍射距离

x = cp.arange(mesh, dtype=cp.float64) * (extent / mesh) - extent / 2
y = x
X, Y = cp.meshgrid(x, y)
r = cp.sqrt(X ** 2 + Y ** 2)

qplate = QPlate(z_position=0.0, q=q, alpha0=0.0)

# ----------------------------------------------------------------------------
# 1. x 线偏振输入 → 径向型矢量涡旋
src_x = vector_gaussian(x, y, wavelength, waist, 'x')
out_x = qplate.apply(src_x)
ex_x, ey_x = out_x.ex, out_x.ey

# 验证：局部偏振与 r̂ 的对齐度（r > 0.5 waist 的环带）
sel = r > 0.5 * waist
rx, ry = X / r, Y / r
dot = cp.abs(ex_x * cp.conj(rx) + ey_x * cp.conj(ry)) / \
    cp.sqrt(cp.abs(ex_x) ** 2 + cp.abs(ey_x) ** 2)
align = float(cp.mean(dot[sel]).get())
print(f"径向对齐度 <E|r_hat>（r>0.5w0）: {align:.6f}（理想 1.0）")

# 2. RCP 输入 → 拓扑荷 −1 标量涡旋
src_r = vector_gaussian(x, y, wavelength, waist, 'RCP')
out_r = qplate.apply(src_r)
u_vortex = out_r.ex
# 相位缠绕（r = 0.5w₀ 环路）
ring = cp.abs(r - 0.5 * waist) < 0.03
ys, xs = cp.where(ring)
ang = cp.arctan2(Y[ys, xs], X[ys, xs])
ph = cp.angle(u_vortex[ys, xs])
order = cp.argsort(ang)
ph_sorted = ph[order]
dph = cp.angle(cp.exp(1j * cp.diff(ph_sorted)))
closure = cp.angle(cp.exp(1j * (ph_sorted[0] - ph_sorted[-1])))
charge = float((cp.sum(dph) + closure).get() / (2 * cp.pi))
print(f"RCP 输出拓扑荷（相位缠绕）: {charge:.2f}（理论 −2q = −1）")

# 3. 矢量传播后的衍射（甜甜圈强度）
prop = VectorAngularSpectrumPropagator(x, y, wavelength)
E_far = prop.propagate(out_x, z_prop)
I_far = cp.asnumpy(E_far.intensity())
ny, nx = I_far.shape
line = I_far[ny // 2, :]
# 中心暗度：中心单像素 / 截线环峰（同口径单像素比，环峰即截线最大值）
center_pixel = float(I_far[ny // 2, nx // 2])
ring_peak = float(line.max())
print(f"传播 {z_prop} μm 后: 中心像素/环峰 = {center_pixel / ring_peak:.4f}"
      f"（矢量涡旋中心暗，远 < 1）")

# ----------------------------------------------------------------------------
# 4. 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
amp_x = cp.asnumpy(cp.abs(ex_x))
ph_vortex = cp.asnumpy(cp.angle(u_vortex))

im0 = axes[0, 0].imshow(amp_x, extent=[float(x[0]), float(x[-1])] * 2,
                        origin='lower', cmap='viridis')
axes[0, 0].set_title(r'$|E_x|$ after q-plate (x input)')
fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

# 局部偏振方向（椭圆取向角，归一化单位箭头，只画有光的区域）
ex_np = cp.asnumpy(ex_x)
ey_np = cp.asnumpy(ey_x)
amp2_x = np.abs(ex_np) ** 2
amp2_y = np.abs(ey_np) ** 2
psi = 0.5 * np.arctan2(2 * np.real(ex_np * np.conj(ey_np)),
                       amp2_x - amp2_y)          # 椭圆取向角（mod π）
amp_max = amp2_x.max() + amp2_y.max()
bright = (amp2_x + amp2_y) > 0.02 * amp_max        # 幅值阈值内的像素
step = 12
Xs, Ys = cp.asnumpy(X)[::step, ::step], cp.asnumpy(Y)[::step, ::step]
Us = 0.35 * np.cos(psi[::step, ::step])
Vs = 0.35 * np.sin(psi[::step, ::step])
Us[~bright[::step, ::step]] = 0.0
Vs[~bright[::step, ::step]] = 0.0
axes[0, 1].quiver(Xs, Ys, Us, Vs, angles='xy', scale_units='xy', scale=1,
                  width=0.004, color='darkorange')
axes[0, 1].imshow(amp_x, extent=[float(x[0]), float(x[-1])] * 2,
                  origin='lower', cmap='Greys', alpha=0.25)
axes[0, 1].set_xlim(float(x[0]), float(x[-1]))
axes[0, 1].set_ylim(float(x[0]), float(x[-1]))
axes[0, 1].set_title('Local polarization direction (radial-type)')

im2 = axes[0, 2].imshow(ph_vortex, extent=[float(x[0]), float(x[-1])] * 2,
                        origin='lower', cmap='twilight',
                        vmin=-np.pi, vmax=np.pi)
axes[0, 2].set_title(r'arg($E_x$) after q-plate (RCP input, charge $-1$)')
fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

axes[1, 0].imshow(I_far, extent=[float(x[0]), float(x[-1])] * 2,
                  origin='lower', cmap='inferno')
axes[1, 0].set_title(f'Vector vortex diffraction, z = {z_prop} μm')

axes[1, 1].plot(cp.asnumpy(x), line)
axes[1, 1].set_xlabel('x (μm)')
axes[1, 1].set_title('Far-field cross section (donut)')

axes[1, 2].axis('off')
axes[1, 2].text(0.05, 0.55,
                f'q = {q}, λ = {wavelength} μm\n'
                f'radial alignment: {align:.4f}\n'
                f'RCP vortex charge: {charge:.2f}\n'
                f'center / ring-peak: {center_pixel / ring_peak:.4f}',
                fontsize=12, family='monospace')

fig.suptitle('q-plate spin-orbit coupling → vector vortex beam')
fig.tight_layout()
import os
os.makedirs('img', exist_ok=True)
fig.savefig(f'img/{project_name}.png', dpi=150)
print(f"图已保存: img/{project_name}.png")
