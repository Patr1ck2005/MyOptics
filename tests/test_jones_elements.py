"""M2c Jones 偏振元件与 q-plate 自旋-轨道耦合验证测试。"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _flat_field(n=32, wl=0.5):
    """单位均匀横向场（Jones 测试不涉及传播）。"""
    import cupy as cp
    x = cp.linspace(-1, 1, n)
    ones = cp.ones((n, n), dtype=cp.complex128)
    from vector.field import VectorField
    return VectorField(ones, cp.zeros_like(ones), None, x, x, wl)


def _amp_phase(u):
    import cupy as cp
    return float(cp.abs(u).get()), float(cp.angle(u).get())


def test_qwp_45_converts_x_to_rcp():
    """QWP(45°)：x̂ → RCP（(1,−i)/√2，本框架约定，见 elements 模块 docstring），
    幅度守恒。"""
    import cupy as cp

    from vector.elements import QuarterWavePlate
    f0 = _flat_field()
    qwp = QuarterWavePlate(0.0, np.pi / 4)
    f1 = qwp.apply(f0)
    ex = cp.asarray(f1.ex)[8, 8]
    ey = cp.asarray(f1.ey)[8, 8]
    ax, px = _amp_phase(ex)
    ay, py = _amp_phase(ey)
    assert abs(ax - np.sqrt(0.5)) < 1e-12
    assert abs(ay - np.sqrt(0.5)) < 1e-12
    # 相对相位 −π/2（RCP）
    dph = (py - px) % (2 * np.pi)
    assert abs(dph - 3 * np.pi / 2) < 1e-10


def test_qwp_axis_x_gives_linear():
    """QWP(0°) 对 x 偏振只给全局相位（本征态），仍是线偏振。"""
    import cupy as cp

    from vector.elements import QuarterWavePlate
    f1 = QuarterWavePlate(0.0, 0.0).apply(_flat_field())
    ey = cp.asarray(f1.ey)[8, 8]
    assert abs(ey) < 1e-12
    ex = cp.asarray(f1.ex)[8, 8]
    assert abs(abs(ex) - 1.0) < 1e-12


def test_hwp_flips_helicity_and_geometric_phase():
    """HWP：|L⟩ → e^{2iα}·|R⟩。用 α 与 α=0 的输出比值提取几何相位 2α
    （约定无关的验证），并验证手性翻转。"""
    import cupy as cp

    from vector.elements import HalfWavePlate
    lcp = _flat_field()
    lcp.ex = lcp.ex / np.sqrt(2)
    lcp.ey = 1j * lcp.ex                    # |L⟩ = (1, i)/√2
    out0 = HalfWavePlate(0.0, 0.0).apply(lcp)
    out45 = HalfWavePlate(0.0, np.pi / 4).apply(lcp)
    ex0 = cp.asarray(out0.ex)[8, 8]
    ex45 = cp.asarray(out45.ex)[8, 8]
    # 手性翻转：输出 ∝ (1, −i)，即 ey0 = −i·ex0
    ey0 = cp.asarray(out0.ey)[8, 8]
    assert abs(ey0 + 1j * ex0) / abs(ex0) < 1e-12
    # 几何相位：out45/out0 = e^{2iα}
    ratio = ex45 / ex0
    assert abs(ratio - np.exp(2j * np.pi / 4)) < 1e-12


def test_polarizer_malus_and_extinction():
    import cupy as cp

    from vector.elements import Polarizer
    f0 = _flat_field()
    f45 = Polarizer(0.0, np.pi / 4).apply(f0)
    ex = cp.asarray(f45.ex)[8, 8]
    ey = cp.asarray(f45.ey)[8, 8]
    assert abs(abs(ex) ** 2 + abs(ey) ** 2 - 0.5) < 1e-12   # Malus（总功率）
    assert abs(ey - ex) < 1e-12                             # 沿 45° 轴
    f90 = Polarizer(0.0, np.pi / 2).apply(f0)
    assert abs(cp.asarray(f90.ex)[8, 8]) < 1e-12
    assert abs(cp.asarray(f90.ey)[8, 8]) < 1e-12  # x 输入被 90° 偏振片完全截止


def test_vector_aperture_mask():
    import cupy as cp

    from vector.elements import VectorAperture
    f0 = _flat_field(n=64)
    ap = VectorAperture(0.0, radius=0.4, inner_radius=0.2)
    f1 = ap.apply(f0)
    X, Y = np.meshgrid(cp.asnumpy(f1.x), cp.asnumpy(f1.y))
    r = np.sqrt(X ** 2 + Y ** 2)
    ex = cp.asnumpy(f1.ex)
    assert np.all(ex[r < 0.19] == 0)
    assert np.all(ex[r > 0.41] == 0)
    assert np.allclose(ex[(r > 0.21) & (r < 0.39)], 1.0)


def test_qplate_spin_to_orbit_conversion():
    """q=1/2 q-plate + RCP 输入 → LCP 输出携带拓扑荷 −1 涡旋（自旋-轨道
    耦合，J|R⟩ = −i·e^{−2iqφ}|L⟩）。约定无关验证：
    (a) |ex| = |ey| = 1/2 处处成立（圆偏振无振幅奇点）；
    (b) 相位沿环路缠绕 = 2π×(−1)（荷守恒），LCP 输入反号。"""
    import cupy as cp

    from vector.elements import QPlate
    n = 128
    x = cp.linspace(-1, 1, n)
    X, Y = cp.meshgrid(x, x)
    r = cp.sqrt(X ** 2 + Y ** 2)

    def charge_of(sign):
        f0 = _flat_field(n=n)
        f0.ex = f0.ex / np.sqrt(2)
        f0.ey = sign * 1j * f0.ex          # RCP: sign=−1；LCP: sign=+1
        out = QPlate(0.0, q=0.5).apply(f0)
        u = cp.asarray(out.ex)
        amp = cp.abs(u)
        assert float(cp.max(cp.abs(amp - np.sqrt(0.5))).get()) < 1e-9
        sel = cp.abs(r - 0.45) < 0.02
        ys, xs = cp.where(sel)
        ang = cp.arctan2(Y[ys, xs], X[ys, xs])
        ph = cp.angle(u[ys, xs])
        order = cp.argsort(ang)
        ph = ph[order]
        dph = cp.angle(cp.exp(1j * cp.diff(ph)))
        closure = cp.angle(cp.exp(1j * (ph[0] - ph[-1])))
        return float((cp.sum(dph) + closure).get()) / (2 * np.pi)

    assert abs(charge_of(-1) + 1.0) < 0.3, "RCP 输入应得拓扑荷 −1"
    assert abs(charge_of(+1) - 1.0) < 0.3, "LCP 输入应得拓扑荷 +1"


def test_qplate_linear_to_cylindrical_vector_vortex():
    """q=1/2 + x 线偏振 → 径向型矢量涡旋：域内每点的 Jones 矢量
    平行于 r̂（q>0, x 输入；允许逐点整体复相位），即 |⟨E | r̂⟩|/|E| = 1
    （约定无关的不变量）。"""
    import cupy as cp

    from vector.elements import QPlate
    n = 96
    x = cp.linspace(-1, 1, n)
    X, Y = cp.meshgrid(x, x)
    r = cp.sqrt(X ** 2 + Y ** 2)
    f0 = _flat_field(n=n)
    out = QPlate(0.0, q=0.5).apply(f0)
    ex = cp.asarray(out.ex)
    ey = cp.asarray(out.ey)
    sel = (r > 0.3) & (r < 0.7)
    ys, xs = cp.where(sel)
    ex_s = ex[ys, xs]
    ey_s = ey[ys, xs]
    rx = X[ys, xs] / r[ys, xs]
    ry = Y[ys, xs] / r[ys, xs]
    amp = cp.sqrt(cp.abs(ex_s) ** 2 + cp.abs(ey_s) ** 2)
    # 归一化内积 <E | r̂>：平行 → 单位模（允许逐点几何相位 e^{iφ}）
    dot = (ex_s * cp.conj(rx) + ey_s * cp.conj(ry)) / amp
    assert float(cp.max(cp.abs(cp.abs(dot) - 1.0)).get()) < 1e-9


def test_jones_output_transverse_and_lossless():
    """Jones 元件输出 ez=None 且功率不变（无损幺正性）。"""
    from vector.elements import HalfWavePlate
    f0 = _flat_field(n=48)
    f0 = f0.normalized()
    out = HalfWavePlate(0.0, 0.7).apply(f0)
    assert out.ez is None
    assert abs(out.power() - 1.0) < 1e-10
