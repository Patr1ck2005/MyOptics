"""多层膜平板元件：把 TMM 传递函数嵌入角谱传播（动量空间 H(kx, ky) 滤波）。

物理图像：层状膜堆对横向波矢 k=(kx, ky) 的响应由 TMM 给出——每个平面波
分量透过膜堆时获得复振幅 H(k)（含相移、反射衰减与倏逝增强）。
在角谱框架下：

    U_out(x, y) = IFFT{ H(kx, ky) · FFT{ U_in } }

这正是超透镜传函仿真所需的"膜堆作为动量空间滤波器"模型。

约定与口径（重要）：
- **H 是场振幅比**：透射侧出射平面波振幅 / 入射侧平面波振幅（含膜堆相位），
  不做能流归一、不做 |H(k=0)|=1 归一。传播带内 |H|² 即功率透过率 T(k)；
  倏逝带内 |H| 可以 > 1（超透镜倏逝增强的物理来源——倏逝分量不携带净能流）。
- **框架一致性要求 N_inc = N_exit = 1（空气）**：本框架的自由传播按真空
  k0 展开，只有空气包覆的膜堆与前后传播严格自洽（超透镜、空气中的镀膜
  元件均属此类）。N_inc/N_exit ≠ 1 时 H 仍是 TMM 场比，但出射介质内部的
  传播相位/衰减不被框架表达——需要时应使用 MultiLayerTM 的场分布工具。
- 倏逝区由 TMM 物理分支自然给出 kz = i·γ 与相应振幅比，无需人为带限。
- 标量框架的偏振：polarization 只能选 's' 或 'p'（非偏振光 = 两种偏振
  各跑一次后对强度取平均，复振幅平均无物理意义）。
"""
import numpy as np

import cupy as cp

from multilayer.common.tmm import tmm_k2_amplitudes
from optical_system.elements_cls import OpticalElement
from utils.constants import PI


class MultilayerSlab(OpticalElement):
    """多层膜平板：以 TMM 场传函 H(kx, ky) 调制光场的动量空间元件。

    参数:
    z_position (float): 元件在光轴上的位置。
    N_layers (list[complex]): 有限层复折射率（入射侧 → 出射侧）。
    d_nm_list (list[float]): 各层厚度（nm）。
    wavelength (float): 真空波长（μm）——传函按单一波长预计算。
    N_inc / N_exit (complex): 入射/出射半空间复折射率；框架一致性
        要求保持 1（空气），见模块 docstring。
    polarization (str): 's' 或 'p'。
    n_kr (int): 1D 径向 H 采样点数（自动非均匀加密）。
    """

    def __init__(self, z_position, N_layers, d_nm_list, wavelength,
                 N_inc=1.0 + 0j, N_exit=1.0 + 0j,
                 polarization='s', n_kr=4096):
        super().__init__(z_position)
        if polarization not in ('s', 'p'):
            raise ValueError("polarization 必须是 's' 或 'p'")
        self.N_layers = [complex(N) for N in N_layers]
        self.d_nm_list = [float(d) for d in d_nm_list]
        self.wavelength = float(wavelength)
        self.N_inc = complex(N_inc)
        self.N_exit = complex(N_exit)
        self.polarization = polarization
        self.n_kr = int(n_kr)
        # 注意：含金属膜的膜堆在传播带内就有类 F-P 快变结构（|t| 尺度 ~1e-4
        # rad/nm），kr 采样密度直接决定插值精度；默认 4096 点对应 ~1e-4 相对
        # 插值误差（一次性构造成本 ~0.2s）。

        # 1D 径向传函（构造时算一次并缓存）
        self._kr_1d, self._H_1d = self._compute_radial_H()
        self.peak_transmittance = float(abs(self._H_1d[0]) ** 2)  # k=0 功率透过率

    # ------------------------------------------------------------------
    def _compute_radial_H(self):
        """1D 径向传函 H(kr)：非均匀 kr 采样（两端加密）。"""
        k0_nm = 2.0 * PI / (self.wavelength * 1000.0)  # 1/nm
        n_max = max(abs(self.N_inc), abs(self.N_exit),
                    max(abs(N) for N in self.N_layers))
        kr_edge = 1.2 * n_max * k0_nm   # 覆盖光锥附近的关键区
        kr_far = 4.0 * kr_edge          # 深倏逝区（强衰减，粗采样即可）

        n_a, n_b = self.n_kr // 3, self.n_kr // 3
        kr = np.concatenate([
            np.linspace(0.0, 0.25 * kr_edge, n_a, endpoint=False),
            np.linspace(0.25 * kr_edge, kr_edge, n_b, endpoint=False),
            np.geomspace(kr_edge, kr_far, self.n_kr - n_a - n_b),
        ])
        kr = np.unique(kr)

        H = np.empty_like(kr, dtype=complex)
        for i, kr_i in enumerate(kr):
            t_amp, _ = tmm_k2_amplitudes(self.wavelength, self.N_layers,
                                         self.d_nm_list, self.N_inc, self.N_exit,
                                         kx2=float(kr_i) ** 2,
                                         pol=self.polarization)
            H[i] = t_amp
        return kr, H

    # ------------------------------------------------------------------
    def radial_H(self):
        """返回 (kr [1/nm], H [复数]) 1D 传函（分析/绘图用）。"""
        return self._kr_1d.copy(), self._H_1d.copy()

    # ------------------------------------------------------------------
    def transfer_function_grid(self, x, y):
        """在坐标网格上构造 2D 传函 H(kx, ky)（cupy 复数数组）。

        索引约定（必须与 AngularSpectrumPropagator / fft2 一致）：
        返回数组 H[iy, ix]，行方向是 y、列方向是 x。
        kr 网格（1/μm 坐标 → 1/nm）线性插值 1D 传函；超出采样范围的
        极高频取最外侧值（深倏逝区，|H| 已衰减到很小）。
        """
        dx = float(cp.asnumpy(x[1]) - cp.asnumpy(x[0]))
        dy = float(cp.asnumpy(y[1]) - cp.asnumpy(y[0]))
        nx, ny = int(x.size), int(y.size)
        # 构造-移位法：先在 fftshift 排序（单调、含 DC 于中心）的频率轴上
        # 构造 H(kx, ky)，再 ifftshift 回 fftfreq 的非移位顺序与 fft2 对齐。
        # 这避免了奇数 N 下 fftfreq 的 Nyquist 负值排序问题。
        fx = cp.fft.fftshift(cp.fft.fftfreq(nx, d=dx))   # cycles/μm，中心为 0
        fy = cp.fft.fftshift(cp.fft.fftfreq(ny, d=dy))
        # indexing='ij'：FX 沿列变化、FY 沿行变化
        FX, FY = cp.meshgrid(fx, fy, indexing='ij')
        # cycles/μm → rad/nm: k = 2π·f / 1000（TMM 的 k0=2π/λ_nm 同口径）
        KR_nm = (2 * PI) * cp.sqrt(FX ** 2 + FY ** 2) / 1000.0

        kr_t = cp.asarray(self._kr_1d)
        H_t = cp.asarray(self._H_1d)
        idx = cp.searchsorted(kr_t, KR_nm.ravel())
        idx = cp.clip(idx, 1, kr_t.size - 1)
        kr_lo = kr_t[idx - 1]
        kr_hi = kr_t[idx]
        w = (KR_nm.ravel() - kr_lo) / (kr_hi - kr_lo)
        H_flat = H_t[idx - 1] * (1 - w) + H_t[idx] * w
        H_shifted = H_flat.reshape(nx, ny)
        # 转置回 [iy, ix] 并 ifftshift 两个轴回到 fftfreq 顺序
        return cp.fft.ifftshift(H_shifted.T)

    # ------------------------------------------------------------------
    def apply(self, U, x, y, wavelength=None):
        """动量空间滤波：FFT → ×H(kx,ky) → IFFT。"""
        H = self.transfer_function_grid(x, y)
        return cp.fft.ifft2(cp.fft.fft2(U) * H)

    # ------------------------------------------------------------------
    @property
    def config(self):
        cfg = super().config
        cfg.update({
            'N_layers': self.N_layers,
            'd_nm_list': self.d_nm_list,
            'wavelength': self.wavelength,
            'N_inc': self.N_inc,
            'N_exit': self.N_exit,
            'polarization': self.polarization,
            'peak_transmittance': self.peak_transmittance,
        })
        return cfg
