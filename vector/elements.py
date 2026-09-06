"""矢量光学元件：Jones 矩阵薄元件 + 高NA矢量物镜。

元件分两类：
- **Jones 薄元件**（WavePlate/QPlate/Polarizer/VectorAperture）：只作用在
  横向 Jones 矢量 (ex, ey) 上，2×2 矩阵或标量逐点乘法；输出的 ez=None，
  由下一次矢量传播从横场条件严格重构（薄元件面上的纵向分量物理上由
  衍射决定，传播器重构的是元件后的自由空间解）。
- **VectorLens**（高NA物镜）：与标量 ObjectLens 完全同构的薄透镜模型
  （球面相位 exp(−ik(√(f²+r²)−f)) + NA 光瞳 mask r ≤ f·tan(arcsin NA)），
  对 ex/ey 同时作用；可选 aplanatic 能量 apodization √cosθ。
  注意与 RichardsWolfFocuser（经典 Debye 约定，pupil 坐标 f·sinθ）的区别：
  两者在近轴极限一致，高NA下是两种不同的物理模型（见模块测试）。

偏振约定（与 sources 一致，时谐 e^{−iωt}）：
- Jones 基矢 (x̂, ŷ)，迎着 +z 传播方向看（从 +z 往 −z 看）逆时针为 LCP，
  Jones 矢量 |L⟩ = (1, +i)/√2，|R⟩ = (1, −i)/√2；
- 线延迟器（快轴角 α，延迟 δ>0，快轴本征值 e^{−iδ/2}）：
  J = R(α)·diag(e^{−iδ/2}, e^{+iδ/2})·R(−α)，R 为平面旋转矩阵；
  验证：QWP(45°) 把 x̂ 变为 LCP；HWP(α) 把 |L⟩ → e^{−iδ/2}e^{+2iα}|R⟩
  （旋光性翻转 + 几何相位 2α——q-plate 自旋-轨道耦合的来源）。
"""
import cupy as cp

from utils.constants import PI
from vector.field import VectorField


class VectorElement:
    """矢量元件基类：z_position 用于 VectorOpticalSystem 排序。"""

    def __init__(self, z_position):
        self.z_position = float(z_position)

    def apply(self, field: VectorField) -> VectorField:
        raise NotImplementedError

    @property
    def config(self):
        return {'type': type(self).__name__, 'z_position': self.z_position}


class JonesElement(VectorElement):
    """Jones 矩阵薄元件：子类实现 jones_matrix(X, Y) → (jxx, jxy, jyx, jyy)。"""

    def jones_matrix(self, X, Y):
        raise NotImplementedError

    def apply(self, field: VectorField) -> VectorField:
        X, Y = cp.meshgrid(field.x, field.y)
        jxx, jxy, jyx, jyy = self.jones_matrix(X, Y)
        ex, ey = field.ex, field.ey
        new_ex = jxx * ex + jxy * ey
        new_ey = jyx * ex + jyy * ey
        return VectorField(new_ex, new_ey, None, field.x, field.y,
                           field.wavelength)


def _retarder_jones(axis_angle, retardance, X):
    """线延迟器 Jones 矩阵（快轴 axis_angle，快轴本征值 e^{−iδ/2}）。"""
    c, s = cp.cos(axis_angle), cp.sin(axis_angle)
    c2, s2, cs = c * c, s * s, c * s
    ph_f = cp.exp(-1j * retardance / 2)
    ph_s = cp.exp(+1j * retardance / 2)
    jxx = ph_f * c2 + ph_s * s2
    jxy = (ph_f - ph_s) * cs
    jyx = jxy
    jyy = ph_f * s2 + ph_s * c2
    return jxx, jxy, jyx, jyy


class WavePlate(JonesElement):
    """理想波片：延迟 δ（rad），快轴角 fast_axis_angle（rad）。

    δ=π/2 → 四分之一波片；δ=π → 半波片。
    """

    def __init__(self, z_position, retardance, fast_axis_angle=0.0):
        super().__init__(z_position)
        self.retardance = float(retardance)
        self.fast_axis_angle = float(fast_axis_angle)

    def jones_matrix(self, X, Y):
        return _retarder_jones(self.fast_axis_angle, self.retardance, X)

    @property
    def config(self):
        cfg = super().config
        cfg.update(retardance=self.retardance,
                   fast_axis_angle=self.fast_axis_angle)
        return cfg


class QuarterWavePlate(WavePlate):
    def __init__(self, z_position, fast_axis_angle=0.0):
        super().__init__(z_position, PI / 2, fast_axis_angle)


class HalfWavePlate(WavePlate):
    def __init__(self, z_position, fast_axis_angle=0.0):
        super().__init__(z_position, PI, fast_axis_angle)


class QPlate(JonesElement):
    """q-plate：半波片，快轴角随方位角线性变化 α(r,φ) = q·φ + α0。

    自旋-轨道耦合（δ=π 全转换）：|L⟩ → e^{+2iα}|R⟩，|R⟩ → e^{−2iα}|L⟩，
    即输入圆偏振高斯 → 输出相反手性、拓扑荷 ±2q 的涡旋；线偏振输入 →
    径向/方位偏振矢量涡旋（q=1/2, α0=0 时）。
    中心 r=0 处 φ 无定义，取 α = α0（高斯中心场幅非零时有单像素相位
    缺陷，物理 q-plate 同样存在，量级可忽略）。
    """

    def __init__(self, z_position, q, alpha0=0.0, retardance=PI):
        super().__init__(z_position)
        self.q = float(q)
        self.alpha0 = float(alpha0)
        self.retardance = float(retardance)

    def jones_matrix(self, X, Y):
        phi = cp.arctan2(Y, X)
        axis = self.q * phi + self.alpha0
        return _retarder_jones(axis, self.retardance, X)

    @property
    def config(self):
        cfg = super().config
        cfg.update(q=self.q, alpha0=self.alpha0, retardance=self.retardance)
        return cfg


class Polarizer(JonesElement):
    """理想线偏振片：透过轴角 angle（rad）。Malus 定律成立。"""

    def __init__(self, z_position, angle=0.0):
        super().__init__(z_position)
        self.angle = float(angle)

    def jones_matrix(self, X, Y):
        c, s = cp.cos(self.angle), cp.sin(self.angle)
        jxx = cp.ones_like(X) * (c * c)
        jxy = cp.ones_like(X) * (c * s)
        jyy = cp.ones_like(X) * (s * s)
        return jxx, jxy, jxy, jyy

    @property
    def config(self):
        cfg = super().config
        cfg.update(angle=self.angle)
        return cfg


class VectorAperture(JonesElement):
    """圆孔/环孔光阑：r ∈ [inner_radius, radius] 内透光（标量掩模）。"""

    def __init__(self, z_position, radius, inner_radius=0.0):
        super().__init__(z_position)
        self.radius = float(radius)
        self.inner_radius = float(inner_radius)

    def jones_matrix(self, X, Y):
        r = cp.sqrt(X ** 2 + Y ** 2)
        mask = ((r <= self.radius) & (r >= self.inner_radius)).astype(cp.complex128)
        zero = cp.zeros_like(mask)
        return mask, zero, zero, mask

    @property
    def config(self):
        cfg = super().config
        cfg.update(radius=self.radius, inner_radius=self.inner_radius)
        return cfg


class VectorLens(VectorElement):
    """高NA薄物镜（矢量版 ObjectLens）：球面相位 + NA 光瞳 + 可选 apodization。

    相位与 mask 约定与标量 ObjectLens 完全一致（保留框架历史约定）：
    phase = k·(√(f²+r²) − f)，exp(−i·phase)；mask r ≤ f·tan(arcsin NA)。
    apodization=True 时乘以 √cosθ（aplanatic 能量因子，
    cosθ = f/√(f²+r²)）——经典 RW 强度函数使用它。
    """

    def __init__(self, z_position, focal_length, NA=0.0, apodization=False):
        super().__init__(z_position)
        self.focal_length = float(focal_length)
        self.NA = float(NA)
        self.apodization = bool(apodization)
        if NA > 0:
            if not (0 < NA < 1):
                raise ValueError("NA 必须在 (0, 1)（空气）")
            self.max_radius = self.focal_length * cp.tan(cp.arcsin(self.NA))
        else:
            self.max_radius = None

    def apply(self, field: VectorField) -> VectorField:
        X, Y = cp.meshgrid(field.x, field.y)
        r2 = X ** 2 + Y ** 2
        r = cp.sqrt(r2)
        k = 2 * PI / field.wavelength
        phase = cp.exp(-1j * k * (cp.sqrt(self.focal_length ** 2 + r2)
                                  - self.focal_length))
        factor = phase
        if self.max_radius is not None:
            factor = factor * (r <= self.max_radius)
        if self.apodization:
            cos_t = self.focal_length / cp.sqrt(self.focal_length ** 2 + r2)
            factor = factor * cp.sqrt(cos_t)
        ez = None if field.ez is None else field.ez * factor
        return VectorField(field.ex * factor, field.ey * factor, ez,
                           field.x, field.y, field.wavelength)

    @property
    def config(self):
        cfg = super().config
        cfg.update(focal_length=self.focal_length, NA=self.NA,
                   apodization=self.apodization)
        return cfg
