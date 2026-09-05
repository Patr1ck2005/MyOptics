"""角谱传播器：预计算并缓存频率网格与传播因子基向量。

对比逐次调用的 ``angular_spectrum_propagate``（每次重建 fx/fy/KX/KY/kz 网格），
本类在构造时一次性缓存：

- ``kz`` 复数纵向波数网格（含倏逝分支）；
- X/Y 实空间网格（供元件调制复用）。

``propagate(U, z)`` 每次只需一次全网格 ``exp(1j·kz·z)`` 加两次 FFT，
纵向多步扫描（如 ``propagate_to_longitudinal_section`` 的数百步）下
可省去每步的网格重建开销（实测约 2-3 倍整体提速，见 docs/benchmark.md）。

缓存失效策略：网格尺寸/间距/波长/传播模式任一变化即整体重建。
"""
import cupy as cp

from utils.constants import PI


class AngularSpectrumPropagator:
    """缓存的角谱传播器（Rigorous / Fresnel 两种模式）。"""

    def __init__(self, x, y, wavelength, mode='Rigorous'):
        """
        参数:
        x, y (cupy 1D array): 实空间坐标（单调，等间距）。
        wavelength (float): 波长（与 x/y 同单位）。
        mode (str): 'Rigorous'（严格，含倏逝波）或 'Fresnel'（近轴）。
        """
        if mode not in ('Rigorous', 'Fresnel'):
            raise ValueError("mode 必须是 'Rigorous' 或 'Fresnel'")
        self.mode = mode
        self.wavelength = float(wavelength)
        self._build(x, y)

    # ------------------------------------------------------------------
    def _build(self, x, y):
        """构建/重建全部缓存网格。"""
        self.x = cp.asarray(x)
        self.y = cp.asarray(y)
        nx = int(self.x.size)
        ny = int(self.y.size)
        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])

        fx = cp.fft.fftfreq(nx, d=dx)
        fy = cp.fft.fftfreq(ny, d=dy)
        FX, FY = cp.meshgrid(fx, fy)  # 与 fft2 的 (行=y, 列=x) 约定一致
        KX = 2 * PI * FX
        KY = 2 * PI * FY

        k = 2 * PI / self.wavelength
        # Rigorous: 复数 kz，负平方根分支 → 倏逝波沿 +z 衰减
        # （表达式分组与旧实现完全一致，保证逐位可复现）
        self.kz = cp.sqrt((k ** 2 - KX ** 2 - KY ** 2).astype(cp.complex128))
        # Fresnel: H = exp(-1j·π·λ·z·(fx²+fy²))，缓存频率平方项，标量部分每次计算
        self._fresnel_freq2 = FX ** 2 + FY ** 2

        # 实空间网格复用（元件调制/动量空间元件共享）
        self.X, self.Y = cp.meshgrid(self.x, self.y)

    # ------------------------------------------------------------------
    def propagate(self, U, z):
        """把场 U 传播距离 z，返回新数组（不修改输入）。"""
        H = self.transfer_function(z)
        return cp.fft.ifft2(cp.fft.fft2(U) * H)

    def propagate_batch(self, U, z_list):
        """沿 z_list 依次传播，返回 [(z, U_z), ...]（复用同一份缓存）。"""
        out = []
        current = U
        current_z = 0.0
        for z in z_list:
            dz = z - current_z
            if dz > 0:
                current = self.propagate(current, dz)
            out.append((z, current * 1.0))
            current_z = z
        return out

    def transfer_function(self, z):
        """距离 z 的传递函数 H（缓存 kz / 频率平方项，仅计算 exp）。"""
        if self.mode == 'Rigorous':
            return cp.exp(1j * self.kz * z)
        return cp.exp(-1j * cp.pi * self.wavelength * z * self._fresnel_freq2)
