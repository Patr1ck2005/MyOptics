"""矢量光场容器：三分量复场 (ex, ey, ez) + 横向网格 + 波长。

设计要点：
- **ez 可为 None**：薄元件（Jones 型偏振元件）只作用于横向 Jones 矢量，
  此后 ez 由下一次传播从横场性条件 k·E = 0 严格重构；
- **后端无关**：容器只持有数组（cupy 或 numpy 均可），转换在后端方法里做；
  传播器/元件内部用 cupy；``VectorOpticalSystem`` 返回 numpy 化的结果。
- 功率约定与标量框架一致：``power() = Σ|E|²·dx·dy``（不含 kz 加权），
  系统初始化时把总功率归一到 1（OpticalSystem 同款约定 Σ|U|²=1）。
"""
from dataclasses import dataclass

import cupy as cp


@dataclass
class VectorField:
    """三分量矢量光场。

    参数:
    ex, ey (cp.ndarray): 横向分量，形状 (ny, nx)。
    ez (cp.ndarray | None): 纵向分量；None 表示"待传播重构"。
    x, y (cp.ndarray): 1D 坐标（len(x)=nx 与轴 1 对齐，len(y)=ny 与轴 0 对齐）。
    wavelength (float): 真空波长（与坐标同单位）。
    """

    ex: cp.ndarray
    ey: cp.ndarray
    ez: object | None
    x: cp.ndarray
    y: cp.ndarray
    wavelength: float

    def __post_init__(self):
        self.ex = cp.asarray(self.ex)
        self.ey = cp.asarray(self.ey)
        if self.ex.shape != self.ey.shape:
            raise ValueError(f"ex/ey 形状不一致: {self.ex.shape} vs {self.ey.shape}")
        if self.ez is not None:
            self.ez = cp.asarray(self.ez)
            if self.ez.shape != self.ex.shape:
                raise ValueError(f"ez 形状不一致: {self.ez.shape} vs {self.ex.shape}")
        self.x = cp.asarray(self.x)
        self.y = cp.asarray(self.y)

    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.ex.shape

    @property
    def nx(self):
        return int(self.x.size)

    @property
    def ny(self):
        return int(self.y.size)

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def dy(self):
        return float(self.y[1] - self.y[0])

    # ------------------------------------------------------------------
    def intensity(self):
        """总光强 |ex|²+|ey|²+(|ez|² if ez else 0)，形状 (ny, nx)。"""
        inten = cp.abs(self.ex) ** 2 + cp.abs(self.ey) ** 2
        if self.ez is not None:
            inten = inten + cp.abs(self.ez) ** 2
        return inten

    def power(self):
        """总功率 Σ|E|²·dx·dy（与标量框架 Σ|U|²·dx·dy 同口径）。"""
        return float(cp.sum(self.intensity()).get()) * self.dx * self.dy

    def copy(self):
        return VectorField(
            self.ex.copy(), self.ey.copy(),
            None if self.ez is None else self.ez.copy(),
            self.x, self.y, self.wavelength,
        )

    def normalized(self):
        """总功率（Σ|E|²·dx·dy）归一到 1（返回新对象，与标量框架同口径）。"""
        out = self.copy()
        norm = cp.sqrt(cp.sum(out.intensity()) * self.dx * self.dy)
        out.ex = out.ex / norm
        out.ey = out.ey / norm
        if out.ez is not None:
            out.ez = out.ez / norm
        return out

    def transverse(self):
        """返回横向分量元组 (ex, ey)（Jones 矢量输入用）。"""
        return self.ex, self.ey
