"""角谱传播与 OpticalSystem 的物理回归。"""
import cupy as cp
import numpy as np
import pytest

from optical_system.elements import CircularAperture, Lens, SpatialPlate
from optical_system.system import OpticalSystem
from utils.constants import PI


@pytest.mark.gpu
def test_free_propagation_energy_conservation():
    """自由空间传播守恒 sum|U|^2（角谱传播是幺正的）。"""
    wl = 0.8
    x = np.linspace(-50, 50, 257)
    X, Y = np.meshgrid(x, x)
    U0 = np.exp(-(X ** 2 + Y ** 2) / 20 ** 2).astype(np.complex128)
    system = OpticalSystem(wl, x, x, U0)
    e0 = float(cp.sum(cp.abs(system.U) ** 2).get())
    results = system.propagate_to_cross_sections([0.0, 25.0])
    U_25 = results[25.0][0][0]
    e1 = float(np.sum(np.abs(U_25) ** 2))
    assert e1 == pytest.approx(e0, rel=1e-6)


@pytest.mark.gpu
def test_plane_wave_longitudinal_phase():
    """平面波传播 z 后应获得相位 exp(i k z)。"""
    wl, z = 1.0, 7.3
    x = np.linspace(-20, 20, 129)
    U0 = np.ones((129, 129), dtype=np.complex128)
    system = OpticalSystem(wl, x, x, U0)
    results = system.propagate_to_cross_sections([z])
    U_z = results[z][0][0]
    expected_phase = (2 * PI / wl * z) % (2 * np.pi)
    measured = float(np.angle(U_z[64, 64])) % (2 * np.pi)
    assert measured == pytest.approx(expected_phase, abs=1e-6)


@pytest.mark.gpu
def test_lens_focus_position():
    """平行光经透镜后应在 z≈f 处聚焦（强度峰值位置）。"""
    wl, f = 1.0, 200.0
    x = np.linspace(-40, 40, 257)
    X, Y = np.meshgrid(x, x)
    U0 = np.exp(-(X ** 2 + Y ** 2) / 50 ** 2).astype(np.complex128)
    system = OpticalSystem(wl, x, x, U0)
    system.add_element(Lens(z_position=f / 2, focal_length=f, NA=0.35))
    z_coords = np.linspace(f / 2, f / 2 + f * 1.4, 40)
    on_axis = []
    for z in z_coords:
        U = system.propagate_to_cross_sections([float(z)])[float(z)][0][0]
        on_axis.append(np.abs(U[128, 128]) ** 2)
    focus_z = z_coords[int(np.argmax(on_axis))]
    assert abs(focus_z - (f / 2 + f)) < 0.06 * f


@pytest.mark.gpu
def test_element_sorting_by_z():
    """元件应按 z_position 排序，且同 z 保持插入顺序。"""
    x = np.linspace(-10, 10, 33)
    U0 = np.ones((33, 33), dtype=np.complex128)
    system = OpticalSystem(1.0, x, x, U0)
    a1 = CircularAperture(z_position=5.0, radius=1.0)
    a2 = SpatialPlate(z_position=1.0, modulation_function=lambda X, Y: cp.exp(1j * cp.arctan2(Y, X)))
    a3 = CircularAperture(z_position=5.0, radius=2.0)
    for el in (a1, a2, a3):
        system.add_element(el)
    system.sort_elements()
    assert [e.z_position for e in system.elements] == [1.0, 5.0, 5.0]
    assert system.elements[1] is a1 and system.elements[2] is a3  # 同 z 稳定排序


@pytest.mark.gpu
def test_cross_section_at_zero_returns_normalized_initial_field():
    x = np.linspace(-10, 10, 65)
    X, Y = np.meshgrid(x, x)
    U0 = (np.exp(-(X ** 2 + Y ** 2) / 25 ** 2)).astype(np.complex128)
    system = OpticalSystem(1.0, x, x, U0)
    results = system.propagate_to_cross_sections([0.0])
    U, xr, yr = results[0.0][0]
    assert np.allclose(U, cp.asnumpy(system.U))
    assert np.allclose(xr, x) and np.allclose(yr, x)


@pytest.mark.gpu
def test_dtype_parameter():
    """默认 complex128；显式 complex64 生效。"""
    x = np.linspace(-10, 10, 33)
    U0 = np.ones((33, 33), dtype=np.complex64)
    s128 = OpticalSystem(1.0, x, x, U0)
    assert s128.U.dtype == cp.complex128 and s128.x.dtype == cp.float64
    s64 = OpticalSystem(1.0, x, x, U0, dtype=cp.complex64)
    assert s64.U.dtype == cp.complex64 and s64.x.dtype == cp.float32
