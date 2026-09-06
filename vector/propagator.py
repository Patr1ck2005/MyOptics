"""矢量角谱传播器：横场投影（P = I − k̂k̂ᵀ）+ 标量角谱相位 + Ez 重构。

物理模型（平面波叠加，时谐约定 e^{i(k·r − ωt)}）：
- 任意横向输入 (Ex, Ey) 并**不是**自由空间麦克斯韦解（k·E ≠ 0）。
  传播器对每个谱点施加 3D 横场投影算子 P = I − k̂k̂ᵀ（k̂ = (kx,ky,kz)/k0）：

      Aex' = Aex − kx·D/k0²
      Aey' = Aey − ky·D/k0²        其中 D = kx·Aex + ky·Aey
      Aez' = −kz·D/k0²

  投影后的三元组严格满足 k·E = 0，且对**任意**输入处处有限：
  Ez' ∝ kz（掠入射 kz→0 时 Ez'→0，无除零奇异性）；倏逝区 kz = iγ
  自然给出衰减的倏逝纵向场。被投影剔除的 k̂ 方向分量正是"把横向迹
  误当作自由场"所产生的非物理部分（近轴时为 O(θ₀²) 小量）。
- 投影后各分量乘 exp(i·kz·z) 传播（kz 取 Im ≥ 0 分支，与标量传播器
  同一表达式）。

与标量传播器的关系：近轴极限（谱远低于光锥）下投影是恒等操作，
横向分量与标量传播一致到 O(θ₀²)；差别本身是物理（横向迹 → 自由空间
解需要投影修正），已由 propagate(z) 后再 project 的幂等性与近轴
标量极限测试分别覆盖。
"""
import cupy as cp

from utils.constants import PI
from vector.field import VectorField


class VectorAngularSpectrumPropagator:
    """缓存矢量角谱传播器（3D 横场投影 + Ez 重构）。"""

    def __init__(self, x, y, wavelength):
        """
        参数:
        x, y (cp.ndarray): 实空间坐标（单调，等间距）。
        wavelength (float): 波长（与 x/y 同单位）。
        """
        self.wavelength = float(wavelength)
        self._build(x, y)

    # ------------------------------------------------------------------
    def _build(self, x, y):
        """构建/重建全部缓存网格（行主序：轴0=y，轴1=x，与 fft2 一致）。"""
        self.x = cp.asarray(x)
        self.y = cp.asarray(y)
        nx = int(self.x.size)
        ny = int(self.y.size)
        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])

        fx = cp.fft.fftfreq(nx, d=dx)
        fy = cp.fft.fftfreq(ny, d=dy)
        FX, FY = cp.meshgrid(fx, fy)
        self.KX = 2 * PI * FX   # rad/unit，沿轴1
        self.KY = 2 * PI * FY   # 沿轴0

        k = 2 * PI / self.wavelength
        self.k0 = k
        self.k0sq = k * k
        # 复 kz，负平方根分支 → 倏逝波沿 +z 衰减（与标量传播器同一表达式）
        self.kz = cp.sqrt((k ** 2 - self.KX ** 2 - self.KY ** 2).astype(cp.complex128))

    # ------------------------------------------------------------------
    def _spectrum(self, field):
        """输入场的横向谱 (Aex, Aey)。"""
        return cp.fft.fft2(field.ex), cp.fft.fft2(field.ey)

    def _assemble(self, Aex, Aey, Aez, phase):
        """3D 横场投影 + 相位 + 逆变换。phase 为 exp(i·kz·z) 或 1。

        投影用完整 3D 内积（含输入 Ez）：输入已是横场时 D₃ = kx·Aex +
        ky·Aey + kz·Aez = 0，投影是恒等操作（幂等）；输入只有横向迹时
        剔除 k̂ 方向的非物理分量。"""
        D3 = self.KX * Aex + self.KY * Aey + self.kz * Aez
        Dk = D3 / self.k0sq
        ex = cp.fft.ifft2((Aex - self.KX * Dk) * phase)
        ey = cp.fft.ifft2((Aey - self.KY * Dk) * phase)
        ez = cp.fft.ifft2((Aez - self.kz * Dk) * phase)
        return ex, ey, ez

    # ------------------------------------------------------------------
    def project(self, field):
        """零距离投影：施加 3D 横场投影算子（幂等），不传播。"""
        Aex, Aey = self._spectrum(field)
        Aez = cp.fft.fft2(field.ez) if field.ez is not None else 0.0
        ex, ey, ez = self._assemble(Aex, Aey, Aez, 1.0)
        return VectorField(ex, ey, ez, self.x, self.y, self.wavelength)

    def propagate(self, field, z):
        """把矢量场传播距离 z，返回新 VectorField（含投影后的三分量）。

        横向迹在传播**前**投影到横场子空间，随后谱域逐点纯相位——
        自由传播保持 k·E = 0，传播后的场天然满足横场条件（幂等）。"""
        phase = cp.exp(1j * self.kz * z)
        Aex, Aey = self._spectrum(field)
        Aez = cp.fft.fft2(field.ez) if field.ez is not None else 0.0
        ex, ey, ez = self._assemble(Aex, Aey, Aez, phase)
        return VectorField(ex, ey, ez, self.x, self.y, self.wavelength)

    def propagate_batch(self, field, z_list):
        """沿 z_list 依次传播，返回 [(z, VectorField), ...]（复用同一份缓存）。"""
        out = []
        current = field.copy()
        current_z = 0.0
        for z in z_list:
            dz = z - current_z
            if dz > 0:
                current = self.propagate(current, dz)
            else:
                current = self.project(current)   # 保证输出含 Ez
            out.append((z, current))
            current = current.copy()
            current_z = z
        return out
