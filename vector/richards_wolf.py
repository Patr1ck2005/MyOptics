"""Richards-Wolf 高NA矢量焦场（Debye-Wolf 积分的 FFT 实现）。

物理模型（Debye 近似，时谐 e^{i(k·r−ωt)}，与框架传播器一致）：

    E(r) = C · ∫∫_{k⊥ ≤ k0·NA} [E∞(kx,ky)/kz] · e^{−i(kx·x + ky·y)} · e^{+i kz·z} dkx dky

- **光瞳映射**（aplanatic / 正弦条件）：谱点 (kx,ky) ↔ 光瞳点
  ρ = f·sinθ（sinθ = k⊥/k0），焦平面 z=0 过几何焦点，z 从焦点起量；
- **强度函数**（Novotny-Hecht 标准的横场投影形式）：入瞳横向场 E_in=(a,b)
  折射到方向 k̂ = (−sinθcosφ, −sinθsinφ, cosθ) 后投影到横场平面：
  E∞ = √cosθ·[E_in − k̂(k̂·E_in)]，√cosθ 为 aplanatic 能量 apodization；
- **权重 1/kz** 来自立体角换算 dΩ = dkx·dky/(k0·kz)；
- **全局相位常数** C = (f/2π)·e^{i(k0 f − π/2)}：由近轴极限与标量框架
  （ObjectLens + AngularSpectrumPropagator）严格对齐定出——近轴时标量
  焦场 U_f(0) = −i·e^{ikf}·E0·k f NA²/2，Debye 积分给 E0·f k NA²/2。

与 VectorLens + 矢量角谱传播的关系：两者是同一物理问题的两种模型——
RW 是经典 Debye 近似（光瞳坐标 f·sinθ，谱权重几何化）；VectorLens 是
薄透镜相位 + 严格矢量角谱传播（光瞳坐标 f·tanθ，谱由衍射精确给出）。
近轴极限下二者一致；高NA下的差异是模型差异而非数值误差（测试中以
松容差交叉验证防呆）。

仅支持空气像方（NA < 1）；浸没/高NA像差校正不在本模块范围。
"""
import cupy as cp

from utils.constants import PI
from vector.field import VectorField


def _bilinear_sample(F, qx, qy, x0, y0, dx, dy, nx, ny):
    """双线性采样 F (ny,nx) 于查询坐标 (qx,qy)，界外为 0。"""
    fx = (qx - x0) / dx
    fy = (qy - y0) / dy
    inside = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    fx = cp.clip(fx, 0.0, nx - 1)
    fy = cp.clip(fy, 0.0, ny - 1)
    j0 = cp.floor(fx).astype(cp.int64)
    i0 = cp.floor(fy).astype(cp.int64)
    j1 = cp.minimum(j0 + 1, nx - 1)
    i1 = cp.minimum(i0 + 1, ny - 1)
    tx = fx - j0
    ty = fy - i0
    v00 = F[i0, j0]
    v10 = F[i1, j0]
    v01 = F[i0, j1]
    v11 = F[i1, j1]
    v = (v00 * (1 - ty) + v10 * ty) * (1 - tx) + (v01 * (1 - ty) + v11 * ty) * tx
    return cp.where(inside, v, 0.0)


class RichardsWolfFocuser:
    """高NA aplanatic 矢量聚焦器（缓存全部谱域网格）。"""

    def __init__(self, x, y, wavelength, focal_length, NA):
        """
        参数:
        x, y (cp.ndarray): 输出/光瞳共用网格（焦斑落在此网格上）。
        wavelength (float): 波长（真空，与坐标同单位）。
        focal_length (float): 物镜焦距。
        NA (float): 数值孔径（0 < NA < 1，空气像方）。
        """
        if not (0.0 < NA < 1.0):
            raise ValueError("Richards-Wolf 聚焦要求 0 < NA < 1（空气像方）")
        self.wavelength = float(wavelength)
        self.focal_length = float(focal_length)
        self.NA = float(NA)
        self._build(x, y)

    # ------------------------------------------------------------------
    def _build(self, x, y):
        self.x = cp.asarray(x)
        self.y = cp.asarray(y)
        nx, ny = int(self.x.size), int(self.y.size)
        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])
        x0 = float(self.x[0])
        y0 = float(self.y[0])

        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.x0, self.y0 = x0, y0

        fx = cp.fft.fftfreq(nx, d=dx)
        fy = cp.fft.fftfreq(ny, d=dy)
        FX, FY = cp.meshgrid(fx, fy)
        self.KX = 2 * PI * FX
        self.KY = 2 * PI * FY

        k0 = 2 * PI / self.wavelength
        self.k0 = k0
        kt = cp.sqrt(self.KX ** 2 + self.KY ** 2)
        self.mask = kt <= k0 * self.NA

        kz = cp.sqrt(cp.maximum(k0 ** 2 - kt ** 2, 0.0))
        sin_t = cp.where(self.mask, kt / k0, 0.0)
        cos_t = cp.where(self.mask, kz / k0, 1.0)
        phi = cp.arctan2(self.KY, self.KX)

        self.kz = cp.where(self.mask, kz, 1.0)
        self.sqrt_cos = cp.where(self.mask, cp.sqrt(cos_t), 0.0)

        # 光瞳采样坐标 ρ = f·sinθ（正弦条件）
        rho = self.focal_length * sin_t
        self._px = rho * cp.cos(phi)
        self._py = rho * cp.sin(phi)

        # 强度函数投影系数（E∞ = √cosθ·[E_in − k̂(k̂·E_in)], E_in=(a,b,0)）
        s2 = sin_t ** 2
        cf, sf = cp.cos(phi), cp.sin(phi)
        scf = s2 * cf * sf
        # Ex∞ = √cosθ·[a(1−s²cos²φ) + b(−s²sinφcosφ)]
        self._cxx = self.sqrt_cos * (1.0 - s2 * cf * cf)
        self._cxy = -self.sqrt_cos * scf
        # Ey∞ = √cosθ·[a(−s²sinφcosφ) + b(1−s²sin²φ)]
        self._cyy = self.sqrt_cos * (1.0 - s2 * sf * sf)
        # Ez∞ = √cosθ·sinθcosθ·(a cosφ + b sinφ)
        self._czx = self.sqrt_cos * sin_t * cos_t * cf
        self._czy = self.sqrt_cos * sin_t * cos_t * sf

        # 权重：C·dkx·dky/kz，C = (f/2π)·e^{i(k0 f − π/2)}（近轴对齐全局相位）
        dkx = 2 * PI / (nx * dx)
        dky = 2 * PI / (ny * dy)
        C = (self.focal_length / (2 * PI)) * cp.exp(1j * (k0 * self.focal_length - PI / 2))
        self._W = C * dkx * dky / self.kz

        # 网格原点相位（任意 x0/y0 的精确 FFT 采样）
        self._chi = cp.exp(-1j * (self.KX * x0 + self.KY * y0))

    # ------------------------------------------------------------------
    def _spectrum(self, field):
        """入瞳场 → 三分量 Debye 谱（未乘 z 相位）。"""
        a = _bilinear_sample(field.ex, self._px.ravel(), self._py.ravel(),
                             self.x0, self.y0, self.dx, self.dy,
                             self.nx, self.ny)
        b = _bilinear_sample(field.ey, self._px.ravel(), self._py.ravel(),
                             self.x0, self.y0, self.dx, self.dy,
                             self.nx, self.ny)
        a = a.reshape(self.KX.shape)
        b = b.reshape(self.KX.shape)
        m = self.mask
        Sex = (self._cxx * a + self._cxy * b) * m
        Sey = (self._cxy * a + self._cyy * b) * m
        Sez = (self._czx * a + self._czy * b) * m
        W = self._W * self._chi
        return Sex * W, Sey * W, Sez * W

    def _emit(self, Sex, Sey, Sez):
        ex = cp.fft.fft2(Sex)
        ey = cp.fft.fft2(Sey)
        ez = cp.fft.fft2(Sez)
        return VectorField(ex, ey, ez, self.x, self.y, self.wavelength)

    # ------------------------------------------------------------------
    def focus(self, field: VectorField) -> VectorField:
        """入瞳矢量场 → 焦平面（z=0 过焦点）三分量矢量场。"""
        Sex, Sey, Sez = self._spectrum(field)
        return self._emit(Sex, Sey, Sez)

    def focus_scan(self, field: VectorField, z_list):
        """离焦扫描：返回 [(z, VectorField), ...]，谱只算一次。"""
        Sex, Sey, Sez = self._spectrum(field)
        out = []
        for z in z_list:
            ph = cp.exp(1j * self.kz * z) * self.mask
            out.append((float(z), self._emit(Sex * ph, Sey * ph, Sez * ph)))
        return out
