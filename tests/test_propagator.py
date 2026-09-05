"""缓存传播器与旧逐次传播实现的一致性、以及性能回归。"""
import time

import cupy as cp
import numpy as np
import pytest

from propagation.angular_spectrum import angular_spectrum_propagate
from propagation.propagator import AngularSpectrumPropagator


def _make_field(nx=129, ny=129, waist=5.0):
    x = cp.linspace(-20, 20, nx)
    X, Y = cp.meshgrid(x, x)
    # 束腰远小于窗口，避免边缘截断的频谱泄漏污染能量守恒检查
    U = cp.exp(-(X ** 2 + Y ** 2) / waist ** 2).astype(cp.complex128)
    return x, U


@pytest.mark.gpu
def test_propagator_bitwise_matches_legacy_rigorous():
    """缓存传播器与旧逐次实现逐位一致（Rigorous）。"""
    x, U = _make_field()
    wl = 1.0
    prop = AngularSpectrumPropagator(x, x, wl, mode='Rigorous')
    for z in (0.0, 3.7, 55.5):
        new = prop.propagate(U, z)
        old = angular_spectrum_propagate(U, x, x, z, wl, propagation_mode='Rigorous')
        assert cp.array_equal(new, old), f"z={z} 结果与旧实现存在逐位差异"


@pytest.mark.gpu
def test_propagator_bitwise_matches_legacy_fresnel():
    x, U = _make_field()
    wl = 1.0
    prop = AngularSpectrumPropagator(x, x, wl, mode='Fresnel')
    for z in (0.0, 3.7, 55.5):
        new = prop.propagate(U, z)
        old = angular_spectrum_propagate(U, x, x, z, wl, propagation_mode='Fresnel')
        assert cp.array_equal(new, old), f"z={z} 结果与旧实现存在逐位差异"


@pytest.mark.gpu
def test_propagator_batch_matches_stepwise():
    """propagate_batch 与逐步 propagate 结果一致。"""
    x, U = _make_field()
    prop = AngularSpectrumPropagator(x, x, 1.0)
    z_list = [5.0, 12.5, 30.0]
    batch = dict(prop.propagate_batch(U, z_list))
    stepwise = U
    current = 0.0
    for z in z_list:
        stepwise = prop.propagate(stepwise, z - current)
        current = z
        assert cp.array_equal(batch[z], stepwise)


@pytest.mark.gpu
def test_propagator_energy_conservation():
    x, U = _make_field()
    prop = AngularSpectrumPropagator(x, x, 1.0)
    e0 = float(cp.sum(cp.abs(U) ** 2).get())
    e1 = float(cp.sum(cp.abs(prop.propagate(U, 23.4)) ** 2).get())
    assert e1 == pytest.approx(e0, rel=1e-10)


@pytest.mark.gpu
def test_propagator_cache_reuse_x_y_grids():
    """X/Y 网格缓存与独立 meshgrid 结果一致，且多次访问为同一对象。"""
    x, _ = _make_field()
    prop = AngularSpectrumPropagator(x, x, 1.0)
    X_ref, Y_ref = cp.meshgrid(x, x)
    assert cp.array_equal(prop.X, X_ref)
    assert cp.array_equal(prop.Y, Y_ref)
    assert prop.X is prop.X  # 同一缓存对象


@pytest.mark.gpu
def test_propagator_invalid_mode():
    x, _ = _make_field()
    with pytest.raises(ValueError):
        AngularSpectrumPropagator(x, x, 1.0, mode='bogus')


@pytest.mark.slow
@pytest.mark.gpu
def test_propagator_longitudinal_speedup():
    """缓存传播器 vs 旧逐次实现的纯传播性能对照（信息性，防意外退化）。

    两者做完全相同的操作序列（256 步累积传播），仅差异在
    旧实现每步重建 fftfreq/meshgrid/kz 网格。
    """
    x, U = _make_field(513, 513)
    num_z, z_max = 256, 200.0
    z_steps = np.diff(np.linspace(0, z_max, num_z + 1))

    # 预热（内核编译/FFT plan 缓存）
    prop = AngularSpectrumPropagator(x, x, 1.0)
    _ = prop.propagate(U, 1.0)
    _ = angular_spectrum_propagate(U, x, x, 1.0, 1.0, propagation_mode='Rigorous')

    U_cur = U
    t0 = time.perf_counter()
    for dz in z_steps:
        U_cur = prop.propagate(U_cur, float(dz))
    t_cached = time.perf_counter() - t0

    U_cur = U
    t0 = time.perf_counter()
    z_acc = 0.0
    for dz in z_steps:
        z_acc += float(dz)
        U_cur = angular_spectrum_propagate(U_cur, x, x, z_acc, 1.0,
                                           propagation_mode='Rigorous')
    t_legacy = time.perf_counter() - t0

    speedup = t_legacy / max(t_cached, 1e-9)
    print(f"\n[benchmark] cached={t_cached:.3f}s legacy={t_legacy:.3f}s speedup={speedup:.2f}x")
    # 防意外大幅退化（缓存版不应明显慢于旧实现）
    assert speedup > 0.9
