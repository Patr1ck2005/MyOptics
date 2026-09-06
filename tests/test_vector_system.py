"""M2c VectorOpticalSystem 端到端验证测试。"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _gaussian_cp(n, extent, waist):
    import cupy as cp
    x = cp.arange(n, dtype=cp.float64) * (extent / n) - extent / 2
    X, Y = cp.meshgrid(x, x)
    return x, cp.exp(-(X ** 2 + Y ** 2) / waist ** 2)


def test_system_matches_direct_propagation_no_elements():
    """无元件系统：cross_section(z) 与直接传播逐位一致。"""
    import cupy as cp

    from vector.propagator import VectorAngularSpectrumPropagator
    from vector.sources import vector_gaussian
    from vector.system import VectorOpticalSystem
    x, env = _gaussian_cp(64, 20.0, 5.0)
    wl = 0.5
    f0 = vector_gaussian(x, x, wl, 5.0, 'x')
    sysv = VectorOpticalSystem(wl, x, x, f0)
    out = sysv.propagate_to_cross_sections([7.5])
    prop = VectorAngularSpectrumPropagator(x, x, wl)
    direct = prop.propagate(sysv.E, 7.5)
    assert cp.array_equal(cp.asarray(out[7.5].ex), direct.ex)
    assert cp.array_equal(cp.asarray(out[7.5].ez), direct.ez)


def test_system_element_ordering_and_blocking():
    """偏振片(90°) 放在 z=5：z=2 处场未受影响，z=10 处 x 偏振被完全截止。"""
    import cupy as cp

    from vector.elements import Polarizer
    from vector.sources import vector_gaussian
    from vector.system import VectorOpticalSystem
    x, env = _gaussian_cp(64, 20.0, 5.0)
    wl = 0.5
    f0 = vector_gaussian(x, x, wl, 5.0, 'x')
    sysv = VectorOpticalSystem(wl, x, x, f0, normalize=False)
    sysv.add_element(Polarizer(5.0, np.pi / 2))
    out = sysv.propagate_to_cross_sections([2.0, 10.0])
    assert float(cp.max(cp.abs(cp.asarray(out[2.0].ex))).get()) > 0.9
    # 截止量级 = 偏振片 (1e-15) × 传播数值噪声（掠入射纯相位 ~1e-6），
    # 相对入射场幅 ~1.4 已被抑制 6 个量级
    assert float(cp.max(cp.abs(cp.asarray(out[10.0].ex))).get()) < 1e-5


def test_system_radial_focus_end_to_end():
    """径向高斯 → VectorLens(NA=0.85) → 矢量传播到焦点：轴上 Ez 主导
    （与 RW 一致的物理结论），总功率归一守恒。"""
    import cupy as cp

    from vector.elements import VectorLens
    from vector.sources import vector_gaussian
    from vector.system import VectorOpticalSystem
    wl, f, NA, w0 = 0.5, 10.0, 0.85, 8.0
    n = 256
    x = cp.arange(n, dtype=cp.float64) * (17.0 / n) - 8.5
    f0 = vector_gaussian(x, x, wl, w0, 'radial').normalized()
    sysv = VectorOpticalSystem(wl, x, x, f0)
    sysv.add_element(VectorLens(0.0, f, NA, apodization=True))
    out = sysv.propagate_to_cross_sections([f])
    E = out[f]
    ny, nx = E.ex.shape
    ex_c = complex(E.ex[ny // 2, nx // 2])
    ey_c = complex(E.ey[ny // 2, nx // 2])
    ez_c = complex(E.ez[ny // 2, nx // 2])
    assert abs(ex_c) < 0.05 * abs(ez_c)
    assert abs(ey_c) < 0.05 * abs(ez_c)
    # 功率量级合理（透镜 apodization 损失 + 横场投影剔除，均为 O(NA²)）
    assert 0.6 < E.power() < 1.05


def test_system_longitudinal_section_shapes():
    """纵向截面的数组形状与总量正确性。"""
    from vector.sources import vector_gaussian
    from vector.system import VectorOpticalSystem
    x, env = _gaussian_cp(64, 20.0, 5.0)
    wl = 0.5
    f0 = vector_gaussian(x, x, wl, 5.0, 'x')
    sysv = VectorOpticalSystem(wl, x, x, f0)
    coord, zs, I_total, I_comp = sysv.propagate_to_longitudinal(
        direction='x', position=0.0, num_z=25, z_max=12.0)
    assert coord.shape == (64,)
    assert zs.shape == (25,)
    assert I_total.shape == (64, 25)
    assert set(I_comp) == {'ex', 'ey', 'ez'}
    assert np.allclose(I_total, I_comp['ex'] + I_comp['ey'] + I_comp['ez'])
    assert I_total.max() > 0
