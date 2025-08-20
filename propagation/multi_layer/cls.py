import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0  # 用于 Hankel 变换

class Layer:
    """Represents a single layer with complex permittivity and finite or semi-infinite thickness."""
    def __init__(self, eps: complex, thickness: float = None):
        self.eps = eps
        self.d = thickness

class MultiLayerTM:
    """Handles TM‐polarized wave propagation through a stack of layers."""
    def __init__(self, layers, eps_incident=1.0, wavelength=365, kx=None):
        self.layers = layers
        self.eps0 = eps_incident
        self.wl  = wavelength
        self.k0  = 2*np.pi / wavelength
        # 默认 kx = 2*(2π/λ)
        self.kx  = kx if kx is not None else 2*2*np.pi/self.wl
        self._update()

    def _update(self):
        """Recompute all dependent quantities."""
        # kz in each region
        self.kz0 = np.sqrt(self.eps0*self.k0**2 - self.kx**2, dtype=complex)
        self.kz  = [np.sqrt(layer.eps*self.k0**2 - self.kx**2, dtype=complex)
                    for layer in self.layers]
        # interface positions [0, d1, d1+d2, …]
        th = [L.d for L in self.layers if L.d is not None]
        self.interfaces = np.concatenate(([0], np.cumsum(th)))

        # compute As, Bs one time
        self._compute_As_Bs()

    def _M_tm(self, eps, kz, d):
        """Single‐layer TM transfer matrix over thickness d."""
        φ = kz * d
        return np.array([
            [ np.cos(φ),         (eps/kz)*np.sin(φ)],
            [-(kz/eps)*np.sin(φ), np.cos(φ)        ]
        ], dtype=complex)

    def _global_M(self):
        """Global 2×2 transfer matrix of all finite layers."""
        M = np.eye(2, dtype=complex)
        for L, kz in zip(self.layers, self.kz):
            if L.d is not None:
                M = self._M_tm(L.eps, kz, L.d) @ M
        return M

    def reflection_coefficient(self):
        """Compute TM reflection coefficient r."""
        a0 = 1j * self.kz0 / self.eps0
        M00, M01, M10, M11 = self._global_M().ravel()
        epsN, kzN = self.layers[-1].eps, self.kz[-1]
        cN = 1j * kzN/epsN
        num = -( (M10 - cN*M00) + (M11 - cN*M01)*a0 )
        den =   (M10 - cN*M00) - (M11 - cN*M01)*a0
        return num/den

    def _compute_As_Bs(self):
        """
        Compute forward/back coefficients A_i, B_i in each layer
        and store them for later field_eval.
        """
        # 1. initial state S = [H; Q] at z=0
        r = self.reflection_coefficient()
        a0 = 1j*self.kz0/self.eps0
        S = np.array([1+r, a0*(1-r)], dtype=complex)

        # 2. propagate across each finite layer, store state at each interface
        Ss = [S]
        for L, kz in zip(self.layers[:-1], self.kz[:-1]):
            M = self._M_tm(L.eps, kz, L.d)
            S = M @ S
            Ss.append(S)

        # 3. from each S compute A,B in that layer
        As, Bs = [], []
        for (L, kz, S0) in zip(self.layers[:-1], self.kz[:-1], Ss[:-1]):
            H0, Q0 = S0
            # A = (H + (ε/(i kz)) Q)/2, B = (H - …)/2
            factor = L.eps/(1j*kz)
            As.append(0.5*(H0 + factor*Q0))
            Bs.append(0.5*(H0 - factor*Q0))
        # last semi‐infinite: only forward
        As.append(Ss[-1][0]); Bs.append(0)

        self._As = As
        self._Bs = Bs

    def field_at(self, z):
        """Return complex H(z) at arbitrary z."""
        # locate layer index
        if z < self.interfaces[-1]:
            idx = np.searchsorted(self.interfaces[1:], z, side='right')
            zloc = z - self.interfaces[idx]
        else:
            idx = len(self.layers)-1
            zloc = z - self.interfaces[-1]
        A, B, kz = self._As[idx], self._Bs[idx], self.kz[idx]
        # if semi‐infinite, B=0 already
        return A*np.exp(1j*kz*zloc) + B*np.exp(-1j*kz*zloc)

    def field_intensity_at(self, z):
        """Return |H(z)|^2."""
        return np.abs(self.field_at(z))**2

    def field_profile(self, num=600, zmax_factor=1.5):
        """Return (z_array, H_array) over entire structure."""
        zmax = self.interfaces[-1]*zmax_factor
        z = np.linspace(0, zmax, num)
        H = np.vectorize(self.field_at)(z)
        return z, H

    def plot_field(self, **plt_kwargs):
        """Plot |H(z)| vs z."""
        z, H = self.field_profile()
        plt.figure()
        plt.plot(z, np.abs(H), **plt_kwargs)
        plt.xlabel('z')
        plt.ylabel(r'$|H(z)|$')
        plt.title('Field Magnitude Profile')
        plt.show()

    def scan_field(self, param, values, z):
        """
        Scan a parameter (e.g. 'kx' or 'wl') and return (values, |H(z)|^2).
        """
        orig = getattr(self, param)
        out = []
        for v in values:
            setattr(self, param, v)
            if param=='wl':  # adjust k0[object Object]
                self.k0 = 2*np.pi/self.wl
            self._update()
            out.append(self.field_intensity_at(z))
        # restore
        setattr(self, param, orig)
        if param=='wl':
            self.k0 = 2*np.pi/orig
        self._update()
        return np.array(values), np.array(out)

# ------- 新增：计算 H(kx) 扫描（复场） -------
    def transfer_vs_kx(self, z, kx_vals):
        """
        返回 z 处随 kx 的复场 H(kx)（或强度 |H|^2）。
        注意：为了逐个 kx 计算，这里会暂改 self.kx 并在结束后复原。
        """
        orig_kx = self.kx
        Hk = np.empty_like(kx_vals, dtype=complex)
        for i, kx in enumerate(kx_vals):
            self.kx = kx
            self._update()
            Hk[i] = self.field_at(z)
        # 复原
        self.kx = orig_kx
        self._update()
        return kx_vals, Hk

    # ------- 新增：由 H(kx) 得到 PSF（幅度和强度） -------
    @staticmethod
    def _ifft_continuous_pair(Hk, kx_vals):
        """
        连续对偶的 IFFT 实现:
          h(x) = (Δk / 2π) * IFFT_kx{ H(kx) }，并配套给出 x 轴。
        要求 kx_vals 为等间隔采样。
        """
        # 等间隔检查 & Δk
        dks = np.diff(kx_vals)
        if not np.allclose(dks, dks[0], rtol=1e-6, atol=1e-12):
            raise ValueError("kx_vals 需要均匀采样以正确进行 FFT。")
        dk = dks[0]
        N  = kx_vals.size

        # 空间域采样间隔与坐标
        dx = 2*np.pi/(N*dk)
        x = (np.arange(N) - N//2) * dx

        # 连续对偶的尺度：Δk / (2π)
        h = (dk/(2*np.pi)) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Hk)))
        return x, h

    # ------- 新增：通用 FWHM 计算工具 -------
    @staticmethod
    def _fwhm(x, y):
        """
        计算曲线 y(x) 的半高宽（FWHM）。
        返回: fwhm, x_left, x_right
          - fwhm = x_right - x_left
          - 若某侧无交点，则对应的 x_* 与 fwhm 置为 np.nan
        """
        y = np.asarray(y)
        x = np.asarray(x)

        if y.size < 3 or not np.isfinite(y).any():
            return np.nan, np.nan, np.nan

        imax = np.nanargmax(y)
        ymax = y[imax]
        if not np.isfinite(ymax) or ymax <= 0:
            return np.nan, np.nan, np.nan

        half = 0.5 * ymax

        # 左侧交点（从峰值往左找第一个低于 half 的点）
        x_left = np.nan
        if imax > 0:
            left_mask = np.where(y[:imax] < half)[0]
            if left_mask.size > 0:
                i1 = left_mask[-1]  # 最靠近峰值且低于 half 的点
                i2 = i1 + 1  # 其右侧点（>= half）
                # 线性插值：在 (x[i1], y[i1]) 与 (x[i2], y[i2]) 间解 y=half
                x1, y1 = x[i1], y[i1]
                x2, y2 = x[i2], y[i2]
                if y2 != y1:
                    x_left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)

        # 右侧交点（从峰值往右找第一个低于 half 的点）
        x_right = np.nan
        if imax < y.size - 1:
            right_rel = np.where(y[imax + 1:] < half)[0]
            if right_rel.size > 0:
                j1 = imax + right_rel[0]  # 第一个低于 half 的点
                j0 = j1 - 1  # 其左侧点（>= half）
                x1, y1 = x[j0], y[j0]
                x2, y2 = x[j1], y[j1]
                if y2 != y1:
                    x_right = x1 + (half - y1) * (x2 - x1) / (y2 - y1)

        if np.isnan(x_left) or np.isnan(x_right):
            return np.nan, x_left, x_right
        return (x_right - x_left), x_left, x_right

    def lsf_from_Hkx(self, z, kx_vals, window=None, normalize='peak'):
        """
        由一维 H(kx) 计算 LSF：
          1) H(kx)（复数，相干） -> 一维逆傅里叶得到幅度 l(x)
          2) 强度 LSF(x) = |l(x)|^2
        返回:
          x, l_amp(x), LSF_intensity(x), fwhm, x_left, x_right
        """
        kx_vals = np.asarray(kx_vals)
        # 得到复场传递 H(kx)
        _, Hk = self.transfer_vs_kx(z, kx_vals)

        # 可选窗（频域）
        if window is not None:
            if isinstance(window, str):
                if window.lower() == 'hann':
                    w = np.hanning(kx_vals.size)
                else:
                    raise ValueError("window 仅支持 'hann' 或自定义 ndarray。")
            else:
                w = np.asarray(window)
                if w.shape != Hk.shape:
                    raise ValueError("窗函数尺寸需与 Hk 相同。")
            Hk = Hk * w

        # 连续对偶尺度的 IFFT： l(x) = (Δk/2π) * IFFT{H(k)}
        dks = np.diff(kx_vals)
        if not np.allclose(dks, dks[0], rtol=1e-6, atol=1e-12):
            raise ValueError("kx_vals 需要等间隔采样。")
        dk = dks[0]; N = kx_vals.size
        dx = 2*np.pi/(N*dk)
        x  = (np.arange(N) - N//2)*dx
        l_amp = (dk/(2*np.pi))*np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Hk)))

        lsf = np.abs(l_amp)**2

        # 归一化
        if normalize is None:
            pass
        elif normalize == 'peak':
            m = lsf.max()
            if m>0: lsf = lsf/m
        elif normalize == 'area':
            A = np.trapz(lsf, x)
            if A>0: lsf = lsf/A
        else:
            raise ValueError("normalize 仅支持 None/'peak'/'area'。")

        fwhm, xL, xR = self._fwhm(x, lsf)
        return x, l_amp, lsf, fwhm, xL, xR

    def psf_from_kx_polar(self, z, kx_vals, window=None, normalize='peak',
                          r_max=None, num_r=1024):
        """
        由一维 H(kx) 构造各向同性的 H(kr)，再做 2D 逆傅里叶（0 阶 Hankel 逆变换）得到 PSF(r)。
        参数:
          kx_vals : 一维频域采样（可包含负值）；我们只取 kr>=0 段
          window  : None/'hann'/ndarray，作用在 H(kr) 上，缓解截断振铃
          normalize: None/'peak'/'area'，用于 PSF 强度的归一化
          r_max   : r 轴上限；默认 ~ π/dk
          num_r   : r 采样点数
        返回:
          r, h_amp(r), PSF_intensity(r), fwhm, r_left, r_right
        """
        kx_vals = np.asarray(kx_vals)
        # 原始一维 H(kx)
        _, Hk_all = self.transfer_vs_kx(z, kx_vals)

        # 构造 kr>=0 的网格与 H(kr)（对复数分实部/虚部插值）
        # 先确保按 kx 升序
        order = np.argsort(kx_vals)
        kx_s = kx_vals[order]
        Hk_s = Hk_all[order]

        # 仅取非负频率段
        mask_pos = kx_s >= 0
        if not np.any(mask_pos):
            raise ValueError("kx_vals 中必须包含非负频率以构造 kr。")
        kr = kx_s[mask_pos]
        Hkr = Hk_s[mask_pos].astype(complex)

        # 频域加窗（可选）
        if window is not None:
            if isinstance(window, str):
                if window.lower() == 'hann':
                    w = np.hanning(kr.size)
                else:
                    raise ValueError("window 仅支持 'hann' 或自定义 ndarray。")
            else:
                w = np.asarray(window)
                if w.shape != Hkr.shape:
                    raise ValueError("窗函数尺寸需与 H(kr) 相同。")
            Hkr = Hkr * w

        # 采样检查
        dks = np.diff(kr)
        if not np.allclose(dks, dks[0], rtol=1e-6, atol=1e-12):
            raise ValueError("kr 采样需等间隔（请用等步长的 kx_vals）。")
        dk = dks[0]
        kmax = kr[-1]

        # r 网格：默认取到 ~ π/dk（与 1D 情况类似的 Nyquist 尺度）
        if r_max is None:
            r_max = np.pi/dk
        r = np.linspace(0.0, r_max, num_r)

        # 数值 Hankel 逆变换（梯形积分类比）： h(r) = (1/2π)∫ H(kr) J0(kr r) kr dkr
        # 向量化实现：对每个 r，计算 J0(kr*r) 并积分
        # [num_r, num_k] = outer
        KR = np.outer(r, kr)               # [Nr, Nk]
        J  = j0(KR)                         # Bessel J0
        integrand = (Hkr * kr)              # [Nk]
        # 梯形法：等间距 dk
        # 对每个 r: sum( J[r,:]*integrand[:] ) * dk
        h_amp = (1.0/(2*np.pi)) * (J @ integrand) * dk

        psf = np.abs(h_amp)**2

        # 归一化
        if normalize is None:
            pass
        elif normalize == 'peak':
            m = psf.max()
            if m>0: psf = psf/m
        elif normalize == 'area':
            A = np.trapz( psf * 2*np.pi*r, r )   # 2πr 权重：面积归一
            if A>0: psf = psf/A
        else:
            raise ValueError("normalize 仅支持 None/'peak'/'area'。")

        full_r = np.concatenate((-r[::-1], r))
        full_psf = np.concatenate((psf[::-1], psf))
        fwhm, rL, rR = self._fwhm(full_r, full_psf)
        return r, h_amp, psf, fwhm, rL, rR

# ---------------- Demo ----------------
if __name__=='__main__':
    layers = [
        Layer(2.56, 10),
        Layer(-2.6115+0.4431j, 10),
        Layer(2.7640+0.1808j, 15),
        Layer(-2.6194+0.4551j, 40),
        Layer(2.43, None),
    ]
    solver_test = MultiLayerTM(layers, eps_incident=2.56, wavelength=365)

    print("r =", solver_test.reflection_coefficient())

    # 画场分布
    solver_test.plot_field()

    # 绘图
    plt.figure(figsize=(10, 6))
    # for die_thickness in [0, 5, 10, 15, 20]:
    # for plas_thickness in [0, 5, 10, 15, 20]:
    for plas_thickness in [0, 10, 20, 40, 60]:
    # for plas_epsilon in [-2.2, -2.4, -2.6, -2.8, -3.0]:
    # for die_epsilon in [2.1, 2.3, 2.5, 2.7, 2.9]:
        layers = [
            Layer(2.56, 10),
            # Layer(plas_epsilon + 0.4431j, 10),
            Layer(-2.6115+0.4431j, 10),
            Layer(2.7640 + 0.1808j, 15),
            # Layer(plas_epsilon + 0.4551j, 40),
            Layer(-2.6194+0.4551j, plas_thickness),
            Layer(2.43, None),
        ]

        solver = MultiLayerTM(layers, eps_incident=2.56, wavelength=365)

        # 单点场强
        z0 = 75.0
        print(f"|H({z0})|^2 =", solver.field_intensity_at(z0))

        # 随 kx 扫描 z=75 处的场强
        kx_vals = np.linspace(-10*solver.k0, 10*solver.k0, 300)
        ks, Is = solver.scan_field('kx', kx_vals, z0)

        # plt.plot(ks, Is, label=f'die thickness = {die_thickness}nm')
        plt.plot(ks/solver.k0, Is, label=f'plas thickness 2 = {plas_thickness}nm')
        # plt.plot(ks, Is, label=f'plas epsilon (real) = {plas_epsilon}')
        # plt.plot(ks, Is, label=f'die epsilon (real) = {die_epsilon}')
        plt.xlabel(r'$k_x/k_0$')
        plt.ylabel(rf'$|H(z={z0})|^2$')
        plt.title(f'Field Intensity at z={z0} vs $k_x$')
    plt.legend(loc='upper left')
    plt.show()

