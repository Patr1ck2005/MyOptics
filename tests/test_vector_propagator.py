"""M2b 矢量角谱传播物理验证测试。"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _grid(n=64, extent=20.0):
    x = np.linspace(-extent / 2, extent / 2, n, endpoint=False)
    return x, x


def _gaussian_np(x, y, waist):
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X ** 2 + Y ** 2) / waist ** 2)


def test_vector_field_power_and_normalization():
    import cupy as cp

    from vector.field import VectorField
    x, y = _grid()
    ex = cp.array(_gaussian_np(x, y, 5.0))
    ey = 0.5 * ex
    f = VectorField(ex, ey, None, cp.array(x), cp.array(y), 0.5)
    # ∫|G|²dxdy = πw²/2（离散化近似到高精度）
    expected = 1.25 * np.pi * 5.0 ** 2 / 2
    assert abs(f.power() - expected) / expected < 1e-3
    f1 = f.normalized()
    assert abs(f1.power() - 1.0) < 1e-12
    # normalized 不修改原对象
    assert abs(f.power() - expected) / expected < 1e-3


def test_propagation_power_conservation_gaussian():
    """横向高斯自由传播：谱域逐点纯相位 + 入谱一次投影 → Σ|E|²·dx·dy
    在不同 z 之间严格守恒（离散 Parseval，1e-10）；相对初始横向迹的
    差为被投影剔除的 k̂ 分量，O(θ₀²) 小量（θ₀≈0.04）。"""
    import cupy as cp

    from vector.propagator import VectorAngularSpectrumPropagator
    from vector.sources import vector_gaussian
    x, y = _grid()
    f0 = vector_gaussian(cp.array(x), cp.array(y), 0.5, 4.0, 'x').normalized()
    prop = VectorAngularSpectrumPropagator(cp.array(x), cp.array(y), 0.5)
    p0 = f0.power()
    powers = []
    for z in (3.0, 17.0):
        f1 = prop.propagate(f0, z)
        powers.append(f1.power())
        # Ez 重构非平凡：场含非零纵向分量（衍射的横向波矢分量所致）
        assert float(cp.max(cp.abs(f1.ez)).get()) > 0
    assert abs(powers[0] - powers[1]) / powers[0] < 1e-10
    assert abs(powers[0] - p0) / p0 < 5e-3


def test_transversality_after_propagation():
    """投影一致性：propagate(z) 后再做 project，结果不变（Ez 是横场的导出量，
    重构路径幂等到 FFT 往返精度）。"""
    import cupy as cp

    from vector.propagator import VectorAngularSpectrumPropagator
    from vector.sources import vector_gaussian
    x, y = _grid()
    f0 = vector_gaussian(cp.array(x), cp.array(y), 0.5, 4.0, '45').normalized()
    prop = VectorAngularSpectrumPropagator(cp.array(x), cp.array(y), 0.5)
    f1 = prop.propagate(f0, 7.0)
    f2 = prop.project(f1)
    dz_ex = float(cp.max(cp.abs(f2.ex - f1.ex)).get())
    dz_ey = float(cp.max(cp.abs(f2.ey - f1.ey)).get())
    dz_ez = float(cp.max(cp.abs(f2.ez - f1.ez)).get())
    scale = float(cp.max(cp.abs(f1.ex)).get())
    assert max(dz_ex, dz_ey) / scale < 1e-6   # 相对偏差 ~8e-4：输入迹含 O(θ₀²)
    # 的 k̂ 分量，单次投影剔除后传播保持横场（幂等到 FFT 往返精度）
    assert dz_ez < 1e-8


def test_scalar_limit_matches_scalar_propagator():
    """x 线偏振输入（近轴，θ₀≈0.04）：矢量传播器的 ex 与标量传播器
    一致到 O(θ₀²)（横场投影修正为 (kx/k0)² 量级），截线相对偏差 < 5e-3。"""
    import cupy as cp

    from propagation.propagator import AngularSpectrumPropagator
    from vector.propagator import VectorAngularSpectrumPropagator
    from vector.sources import vector_gaussian
    x, y = _grid()
    wl = 0.5
    f0 = vector_gaussian(cp.array(x), cp.array(y), wl, 4.0, 'x')
    xv, yv = cp.array(x), cp.array(y)
    vp = VectorAngularSpectrumPropagator(xv, yv, wl)
    sp = AngularSpectrumPropagator(xv, yv, wl, mode='Rigorous')
    f1 = vp.propagate(f0, 7.5)
    u1 = sp.propagate(cp.array(f0.ex), 7.5)
    dev = float(cp.max(cp.abs(f1.ex - u1) / cp.max(cp.abs(u1))).get())
    assert dev < 5e-3, f"近轴标量极限偏差 {dev:.2e}"
    assert bool(cp.all(cp.isfinite(f1.ez)))


def test_circular_polarization_field_structure():
    """LCP 传播后 |ex|≈|ey|（圆偏振的横向分量等幅）+ 功率守恒。"""
    import cupy as cp

    from vector.propagator import VectorAngularSpectrumPropagator
    from vector.sources import vector_gaussian
    x, y = _grid(96, 24.0)
    wl = 0.5
    f0 = vector_gaussian(cp.array(x), cp.array(y), wl, 4.0, 'LCP').normalized()
    prop = VectorAngularSpectrumPropagator(cp.array(x), cp.array(y), wl)
    f1 = prop.propagate(f0, 9.0)
    nx, ny = f1.nx, f1.ny
    ix, iy = nx // 2, ny // 2
    # 束腰中心区域 3×3 平均（避免单像素数值噪声）
    ex_c = cp.abs(f1.ex[iy - 1:iy + 2, ix - 1:ix + 2]).mean()
    ey_c = cp.abs(f1.ey[iy - 1:iy + 2, ix - 1:ix + 2]).mean()
    assert abs(ex_c - ey_c) / ex_c < 0.2   # 轻微椭圆化来自传播的偏振耦合
    assert abs(f1.power() - 1.0) < 5e-3    # O(θ₀²) 口径，见功率守恒测试


def test_evanescent_decay_consistency():
    """倏逝区：给定谱点的衰减率 exp(-γ·z) 与传播器输出一致。"""
    import cupy as cp

    from vector.propagator import VectorAngularSpectrumPropagator
    from vector.sources import vector_gaussian
    wl = 0.5
    n = 128
    dx = 20.0 / n          # Nyquist = π/dx ≈ 20.1 > 1.2·k0 ≈ 15.1
    x = cp.arange(n) * dx
    f0 = vector_gaussian(x, x, wl, 0.2, 'x')   # 紧束腰：谱含倏逝区能量
    prop = VectorAngularSpectrumPropagator(x, x, wl)
    # 找一个倏逝谱点：kx = 1.2·k0（谱幅 ~e^{-2.3}，数值可用）
    kx_target = 1.2 * prop.k0
    fx_grid = 2 * cp.pi * cp.fft.fftfreq(n, d=dx)
    jx = int(cp.argmin(cp.abs(fx_grid - kx_target)).get())
    A0 = cp.fft.fft2(f0.ex)[0, jx]          # ky=0 行
    assert float(cp.abs(A0).get()) > 1e-6
    z1, z2 = 0.2, 0.5
    f1 = prop.propagate(f0, z1)
    f2 = prop.propagate(f0, z2)
    A1 = cp.fft.fft2(f1.ex)[0, jx]
    A2 = cp.fft.fft2(f2.ex)[0, jx]
    kz = prop.kz[0, jx]
    expected_ratio = cp.exp(1j * kz * (z2 - z1))
    assert abs(abs(A2 / A1) - abs(expected_ratio)) < 1e-10
    # 该谱点确实是倏逝波（γ>0，衰减）
    assert float(cp.imag(kz).get()) > 0
