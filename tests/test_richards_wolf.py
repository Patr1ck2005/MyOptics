"""M2a Richards-Wolf 高NA矢量焦场验证测试。"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _centerline(intensity, axis='x', half=8):
    """过焦斑中心的强度截线（输入为强度数组，cp 或 np）。"""
    ny, nx = intensity.shape[-2:]
    if axis == 'x':
        line = intensity[ny // 2, nx // 2 - half: nx // 2 + half]
    else:
        line = intensity[nx // 2 - half: nx // 2 + half, ny // 2]
    import cupy as cp
    return cp.asnumpy(line) if hasattr(line, 'get') else np.asarray(line)


def test_rw_paraxial_alignment_with_scalar_model():
    """近轴极限（NA=0.1）：RW 焦场与标量模型（ObjectLens+角谱传播）强度
    截线相对偏差 < 2e-2（模型等价区）。"""
    import cupy as cp

    from optical_system.elements.lens import ObjectLens
    from propagation.propagator import AngularSpectrumPropagator
    from vector.richards_wolf import RichardsWolfFocuser
    from vector.sources import vector_gaussian
    wl, f, NA, w0 = 0.5, 50.0, 0.1, 40.0
    n = 512
    x = cp.arange(n, dtype=cp.float64) * (128.0 / n) - 64.0
    src = vector_gaussian(x, x, wl, w0, 'x')
    pupil = src.normalized()

    rw = RichardsWolfFocuser(x, x, wl, f, NA)
    E_rw = rw.focus(pupil)
    I_rw = _centerline(E_rw.intensity())

    # 标量模型：ObjectLens 相位 + NA mask，传播 f 到焦点
    sys_field = cp.array(pupil.ex) * 1.0
    lens = ObjectLens(0.0, f, NA)
    u_pupil = lens.apply(sys_field, x, x, wl)
    prop = AngularSpectrumPropagator(x, x, wl, mode='Rigorous')
    u_focus = prop.propagate(u_pupil, f)
    I_sc = cp.abs(u_focus) ** 2
    ny, nx = I_sc.shape
    line_sc = cp.asnumpy(I_sc[ny // 2, nx // 2 - 8: nx // 2 + 8])

    rel = float(np.max(np.abs(I_rw - line_sc) / np.max(line_sc)))
    assert rel < 2e-2, f"近轴 RW/标量 截线偏差 {rel:.2e}"


def test_rw_radial_focus_strong_longitudinal():
    """径向偏振 + NA=0.85：场积分口径下 Ez 能量分数 ≈ 0.265（解析值
    ∫sin³θ√cosθ/∫(sin³θ+sinθ)√cosθ，容差 ±0.06）；轴上横向场对称归零，
    焦点中心由纵向分量主导；总强度为中心峰。"""
    import cupy as cp

    from vector.richards_wolf import RichardsWolfFocuser
    from vector.sources import vector_gaussian
    wl, f, NA, w0 = 0.5, 10.0, 0.85, 8.0
    n = 256
    x = cp.arange(n, dtype=cp.float64) * (17.0 / n) - 8.5
    src = vector_gaussian(x, x, wl, w0, 'radial').normalized()
    rw = RichardsWolfFocuser(x, x, wl, f, NA)
    E = rw.focus(src)
    Ix = float(cp.sum(cp.abs(E.ex) ** 2).get())
    Iy = float(cp.sum(cp.abs(E.ey) ** 2).get())
    Iz = float(cp.sum(cp.abs(E.ez) ** 2).get())
    frac = Iz / (Ix + Iy + Iz)
    assert 0.20 < frac < 0.33, f"径向偏振 Ez 能量分数 {frac:.3f}（期望 0.265）"
    # 轴上（r=0 单点）：横向场严格为 0（对称性），Ez 主导
    ny, nx = E.ex.shape
    t_center = max(float(cp.abs(E.ex[ny // 2, nx // 2]).get()),
                   float(cp.abs(E.ey[ny // 2, nx // 2]).get()))
    z_center = float(cp.abs(E.ez[ny // 2, nx // 2]).get())
    assert z_center > 10 * max(t_center, 1e-30), \
        f"轴上 |Ez|={z_center:.3e} 应主导横向 {t_center:.3e}"
    inten = E.intensity()
    center = float(inten[ny // 2, nx // 2].get())
    ring = float(inten[ny // 2, nx // 2 - 12].get())
    assert center > ring, "径向偏振焦斑应为中心峰（非环形）"


def test_rw_azimuthal_no_longitudinal_hollow():
    """方位偏振 + NA=0.85：Ez 几乎为零（<1e-3 相对），焦斑中空（环形）。"""
    import cupy as cp

    from vector.richards_wolf import RichardsWolfFocuser
    from vector.sources import vector_gaussian
    wl, f, NA, w0 = 0.5, 10.0, 0.85, 8.0
    n = 256
    x = cp.arange(n, dtype=cp.float64) * (17.0 / n) - 8.5
    src = vector_gaussian(x, x, wl, w0, 'azimuthal').normalized()
    rw = RichardsWolfFocuser(x, x, wl, f, NA)
    E = rw.focus(src)
    Iz = float(cp.sum(cp.abs(E.ez) ** 2).get())
    Itot = float(cp.sum(E.intensity()).get())
    assert Iz / Itot < 1e-3
    inten = E.intensity()
    ny, nx = inten.shape
    center = float(inten[ny // 2, nx // 2].get())
    ring = float(inten[ny // 2, nx // 2 - 10].get())
    assert center < 0.2 * ring, "方位偏振焦斑应中空"


def test_rw_circular_polarization_symmetric_split():
    """圆偏振 + NA=0.85：|Ex|² 与 |Ey|² 总量近似相等（<5%）；
    Ez 能量分数 ≈ 0.115（解析期望，容差 ±0.04）。"""
    import cupy as cp

    from vector.richards_wolf import RichardsWolfFocuser
    from vector.sources import vector_gaussian
    wl, f, NA, w0 = 0.5, 10.0, 0.85, 8.0
    n = 256
    x = cp.arange(n, dtype=cp.float64) * (17.0 / n) - 8.5
    src = vector_gaussian(x, x, wl, w0, 'LCP').normalized()
    rw = RichardsWolfFocuser(x, x, wl, f, NA)
    E = rw.focus(src)
    Ix = float(cp.sum(cp.abs(E.ex) ** 2).get())
    Iy = float(cp.sum(cp.abs(E.ey) ** 2).get())
    Iz = float(cp.sum(cp.abs(E.ez) ** 2).get())
    assert abs(Ix - Iy) / (Ix + Iy) < 0.05
    frac = Iz / (Ix + Iy + Iz)
    assert 0.08 < frac < 0.16, f"圆偏振 Ez 能量分数 {frac:.3f}（期望 0.115）"


def test_rw_defocus_parseval_exact():
    """轴外平面 z≠0：各谱点只乘纯相位 → 总 Σ|E|² 网格和逐位不变
    （离散 Parseval 恒等式，数值精度内精确）。"""
    import cupy as cp

    from vector.richards_wolf import RichardsWolfFocuser
    from vector.sources import vector_gaussian
    wl, f, NA, w0 = 0.5, 10.0, 0.85, 8.0
    n = 256
    x = cp.arange(n, dtype=cp.float64) * (17.0 / n) - 8.5
    src = vector_gaussian(x, x, wl, w0, 'radial').normalized()
    rw = RichardsWolfFocuser(x, x, wl, f, NA)
    planes = rw.focus_scan(src, [0.0, 2.0, -1.5])
    sums = [float((cp.sum(cp.abs(E.ex) ** 2) + cp.sum(cp.abs(E.ey) ** 2)
                   + cp.sum(cp.abs(E.ez) ** 2)).get()) for _, E in planes]
    assert abs(sums[1] - sums[0]) / sums[0] < 1e-12
    assert abs(sums[2] - sums[0]) / sums[0] < 1e-12


def test_rw_rejects_super_na():
    import cupy as cp

    from vector.richards_wolf import RichardsWolfFocuser
    x = cp.linspace(-1, 1, 8)
    try:
        RichardsWolfFocuser(x, x, 0.5, 10.0, 1.2)
        raise AssertionError("应当拒绝 NA >= 1")
    except ValueError:
        pass
