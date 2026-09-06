"""宽谱色差示例：衍射透镜的焦移（chromatic focal shift）与宽谱焦斑。

物理背景：衍射光学元件（DOE/菲涅尔透镜）的相位 profile 是加工固定
的 Φ(r) = −k₀(√(f₀²+r²) − f₀)（k₀ 对应设计波长 λ₀）。对波长 λ 的光，
聚焦本领 ∝ 1/λ₀ 而衍射角 ∝ 1/λ，因此焦距 f(λ) = f₀·λ₀/λ——衍射
元件的色差比折射元件强一个量级（阿贝数 ~ −3.45 vs 玻璃 ~ 30-60），
是计算全息/超表面透镜设计的核心限制。

演示 M3 工作流层的完整协作：
1. ParameterSweep 扫 (λ, z) 网格，用峰值指标定位每个波长的焦面
   → 焦移曲线 f(λ)（与理论 f₀·λ₀/λ 对比）；
2. SimSpectrum 非相干合成宽谱焦斑（LED 型宽谱，权重高斯型）；
3. evaluate_metrics 量化：焦移范围、宽谱焦斑的 FWHM 恶化。

运行: python examples/broadband_chromatic_shift.py
输出: img/broadband-chromatic-shift.png + 控制台验证数字
"""
import numpy as np

from utils.cuda_path import ensure_cuda_dll_dirs

ensure_cuda_dll_dirs()   # pip 轮子 CUDA DLL 发现（Windows 必需），再导入 cupy

import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from optical_system.elements_cls import SpatialPlate  # noqa: E402
from optical_system.system import OpticalSystem  # noqa: E402
from workflows.spectrum import SimSpectrum  # noqa: E402
from workflows.sweep import ParameterSweep  # noqa: E402

project_name = 'broadband-chromatic-shift'

# ----------------------------------------------------------------------------
# 参数：设计波长 λ₀=0.5μm，f₀=10μm 的衍射透镜；宽谱 0.42-0.58μm
WL0, F0 = 0.5, 10.0
LAMBDAS = np.array([0.42, 0.46, 0.50, 0.54, 0.58])
WEIGHTS = np.exp(-0.5 * ((LAMBDAS - WL0) / 0.05) ** 2)   # 高斯型光谱

MESH, EXTENT = 256, 16.0
WAIST = 4.0          # 入射高斯束腰（覆盖光瞳）
Z_SCAN = np.round(np.arange(8.4, 12.01, 0.2), 2)   # 焦面扫描网格


def build_diffactive_system(wavelength):
    """λ 处的仿真系统：固定相位衍射透镜（加工后与 λ 无关）+ 高斯入射。"""
    x = cp.arange(MESH, dtype=cp.float64) * (EXTENT / MESH) - EXTENT / 2
    X, Y = cp.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    k0_design = 2 * np.pi / WL0
    phase_fixed = cp.asarray(-k0_design * (cp.sqrt(F0 ** 2 + r2) - F0))
    system = OpticalSystem(float(wavelength), x, x,
                           cp.exp(-r2 / WAIST ** 2))
    system.add_element(SpatialPlate(
        0.0, modulation_function=lambda Xg, Yg: cp.exp(1j * phase_fixed)))
    return system


# ----------------------------------------------------------------------------
# 1. ParameterSweep: (λ, z) 网格 → 每个波长的焦面定位
sweep = ParameterSweep(
    # 扫描参数全集 (wavelength, z) 都会传入；系统只依赖 wavelength
    build_system=lambda wavelength, z: build_diffactive_system(wavelength),
    metrics=('peak_intensity', 'fwhm_x'),
)
grid = {'wavelength': LAMBDAS.tolist(), 'z': Z_SCAN.tolist()}
result = sweep.run(grid, z_of=lambda wavelength, z: float(z),
                   keep='metrics', progress=True)
df = result.to_dataframe()

focal_z, fwhm_at_focus = [], []
for wl in LAMBDAS:
    sub = df[np.isclose(df['wavelength'], wl)]
    best = sub.loc[sub['peak_intensity'].idxmax()]
    focal_z.append(float(best['z']))
    fwhm_at_focus.append(float(best['fwhm_x']))
focal_z = np.array(focal_z)
theory = F0 * WL0 / LAMBDAS

print("\n衍射透镜色差焦移（f₀=10μm @ λ₀=0.5μm）:")
for wl, fz, th, fw in zip(LAMBDAS, focal_z, theory, fwhm_at_focus):
    print(f"  λ={wl:.2f}μm: 焦面 z={fz:.1f}μm (理论 {th:.2f})  "
          f"焦斑 FWHM={fw:.2f}μm")
print(f"焦移范围: {focal_z.max() - focal_z.min():.1f}μm "
      f"（理论 {theory.max() - theory.min():.2f}μm；"
      f"测量值来自 {Z_SCAN[1] - Z_SCAN[0]:.1f}μm 步长网格上的峰值定位，"
      f"含 ±1 格量化）")

# ----------------------------------------------------------------------------
# 2. SimSpectrum: 非相干合成（各 λ 在公称焦面 z=F0 的宽谱焦斑）
def simulate_component(wl):
    system = build_diffactive_system(wl)
    from workflows.field import Field
    return Field.from_scalar_system(system, z_positions=[F0])[0]


spec = SimSpectrum(LAMBDAS, weights=WEIGHTS, simulate=simulate_component)
broad = spec.run(coherent=False, progress=False)

mono = simulate_component(WL0)   # 单色对照（中心波长）
from workflows.sweep import evaluate_metrics  # noqa: E402
m_broad = evaluate_metrics(broad.field, ('fwhm_x', 'peak_intensity'))
m_mono = evaluate_metrics(mono, ('fwhm_x', 'peak_intensity'))
print(f"\n公称焦面 z={F0}μm 处:")
print(f"  单色 (λ={WL0}): FWHM={m_mono['fwhm_x']:.2f}μm, "
      f"峰值={m_mono['peak_intensity']:.4f}")
print(f"  宽谱:         FWHM={m_broad['fwhm_x']:.2f}μm, "
      f"峰值={m_broad['peak_intensity']:.4f}（色差模糊 + 峰值下降）")

# ----------------------------------------------------------------------------
# 3. 可视化
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

axes[0].plot(LAMBDAS, theory, 'k--', label=r'theory $f_0\lambda_0/\lambda$')
axes[0].plot(LAMBDAS, focal_z, 'o-', color='crimson', label='measured (peak z)')
axes[0].set_xlabel('wavelength λ (μm)')
axes[0].set_ylabel('focal distance z (μm)')
axes[0].set_title('Chromatic focal shift (diffractive lens)')
axes[0].legend()

cut_m = mono.line_cut(axis='x')
cut_b = broad.field.line_cut(axis='x')
axes[1].plot(cut_m['coord'], cut_m['total'] / cut_m['total'].max(),
             label=f'mono λ={WL0} (FWHM {m_mono["fwhm_x"]:.2f}μm)')
axes[1].plot(cut_b['coord'], cut_b['total'] / cut_b['total'].max(),
             label=f'broadband {LAMBDAS[0]}–{LAMBDAS[-1]} '
                   f'(FWHM {m_broad["fwhm_x"]:.2f}μm)')
axes[1].set_xlabel('x (μm)')
axes[1].set_title(f'Lateral profiles at z={F0} μm')
axes[1].legend()

wl_plot, z_plot = np.meshgrid(LAMBDAS, Z_SCAN, indexing='ij')
peak = np.zeros_like(wl_plot)
for i, wl in enumerate(LAMBDAS):
    sub = df[np.isclose(df['wavelength'], wl)]
    for j, z in enumerate(Z_SCAN):
        row = sub[np.isclose(sub['z'], z)]
        peak[i, j] = row['peak_intensity'].iloc[0] if len(row) else 0
im = axes[2].pcolormesh(z_plot, wl_plot, peak, shading='auto', cmap='inferno')
axes[2].plot(focal_z, LAMBDAS, 'c.-', label='focal ridge')
axes[2].axvline(F0, color='w', linestyle=':', alpha=0.6)
axes[2].set_xlabel('z (μm)')
axes[2].set_ylabel('wavelength λ (μm)')
axes[2].set_title('Peak intensity vs (z, λ)')
fig.colorbar(im, ax=axes[2], shrink=0.85)
axes[2].legend()

fig.suptitle('Diffractive lens chromatic aberration (M3 SimSpectrum + ParameterSweep)')
fig.tight_layout()
import os
os.makedirs('img', exist_ok=True)
fig.savefig(f'img/{project_name}.png', dpi=150)
print(f"图已保存: img/{project_name}.png")
