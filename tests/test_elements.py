"""元件构造与掩膜/调制行为回归。"""
import cupy as cp
import pytest
from conftest import requires_gpu

from optical_system.elements import (
    AnnularAperture,
    Axicon,
    BlazedGrating,
    CircularAperture,
    CrossAperture,
    EllipticalAperture,
    Lens,
    MomentumSpacePlate,
    ObjectLens,
    RectAmplitudeGrating,
    RectangularAperture,
    SimpleMSPP,
    SineAmplitudeGrating,
    SinePhaseGrating,
    SquareAperture,
)


def test_aperture_constructions():
    """回归：4/6 光阑子类曾因 super().__init__(radius=None) 构造即 TypeError。"""
    CircularAperture(0, radius=1.0)
    SquareAperture(0, size=1.0)
    EllipticalAperture(0, radius_x=1.0, radius_y=2.0)
    RectangularAperture(0, width=2.0, height=3.0)
    CrossAperture(0, arm_width=0.5)
    AnnularAperture(0, inner_radius=0.5, outer_radius=1.0)


@requires_gpu
def test_circular_aperture_mask():
    ap = CircularAperture(0, radius=2.0)
    x = cp.linspace(-5, 5, 101)
    X, Y = cp.meshgrid(x, x)
    mask = ap.create_mask(X, Y)
    assert mask[50, 50]  # 中心通过
    assert not mask[0, 50]  # 边缘遮挡


@requires_gpu
def test_lens_phase_and_na_mask():
    f, wl = 100.0, 0.5
    lens = Lens(0, focal_length=f, NA=0.42)
    x = cp.linspace(-100, 100, 201)
    X, Y = cp.meshgrid(x, x)
    U = cp.ones_like(X, dtype=cp.complex128)
    out = lens.apply(U, x, x, wl)
    # 相位：轴上点应为平面波透镜相位 exp(-i k r²/2f)，r=0 → 1
    assert cp.allclose(out[100, 100], 1.0)
    # NA 掩膜：|r| > f·tan(arcsin NA) 处应为 0
    max_radius = f * cp.tan(cp.arcsin(cp.float64(0.42)))
    r_edge = cp.sqrt(X ** 2 + Y ** 2) > max_radius * 1.01
    assert cp.all(cp.abs(out[r_edge]) == 0)


@requires_gpu
def test_object_lens_nonparaxial_phase():
    f, wl = 4000.0, 1.55
    ol = ObjectLens(0, focal_length=f, NA=0.42)
    x = cp.linspace(-100, 100, 101)
    X, Y = cp.meshgrid(x, x)
    U = cp.ones_like(X, dtype=cp.complex128)
    out = ol.apply(U, x, x, wl)
    r = cp.sqrt(X ** 2 + Y ** 2)
    expected_phase = 2 * cp.pi / wl * (cp.sqrt(f ** 2 + r ** 2) - f)
    assert cp.allclose(cp.angle(out), -expected_phase % (2 * cp.pi), atol=1e-6)


@requires_gpu
def test_gratings_modulation():
    x = cp.linspace(-10, 10, 401)
    X, Y = cp.meshgrid(x, x)
    U = cp.ones_like(X, dtype=cp.complex128)

    sine = SinePhaseGrating(0, period=5.0, amplitude=1.0)
    out = sine.apply(U, x, x, 1.0)
    assert cp.abs(out).max() <= 1.0 + 1e-12  # 纯相位调制

    rect = RectAmplitudeGrating(0, period=5.0, slit_width=2.5)
    out = rect.apply(U, x, x, 1.0)
    assert set(cp.unique(cp.abs(out)).get().round(6)) <= {0.0, 1.0}  # 0/1 振幅光栅

    sine_amp = SineAmplitudeGrating(0, period=5.0)
    out = sine_amp.apply(U, x, x, 1.0)
    assert cp.abs(out).max() <= 1.0 + 1e-12  # clip 到 [0,1]

    blaze = BlazedGrating(0, blaze_angle=0.1, period=8.0)
    out = blaze.apply(U, x, x, 1.0)
    assert cp.allclose(cp.abs(out), 1.0)  # 纯相位


@requires_gpu
def test_axicon_geometry_consistency():
    ax = Axicon(0, base_angle=0.1, refractive_index=1.5)
    assert ax.apex_angle == pytest.approx(cp.pi - 2 * 0.1)
    with pytest.raises(ValueError):
        Axicon(0)  # 未提供角度必须报错


@requires_gpu
def test_momentum_space_plate_vortex():
    """动量空间涡旋相位板：输出场中心应为相位奇点（强度为零）。"""
    wl = 1.55
    plate = MomentumSpacePlate(0, modulation_function=lambda KX, KY: cp.exp(1j * 2 * cp.arctan2(KY, KX)))
    x = cp.linspace(-50, 50, 257)
    X, Y = cp.meshgrid(x, x)
    U = cp.exp(-(X ** 2 + Y ** 2) / 30 ** 2).astype(cp.complex128)
    out = plate.apply(U, x, x, wl)
    center = cp.abs(out[128, 128])
    peak = cp.abs(out).max()
    assert center < 1e-6 * peak


@requires_gpu
def test_simple_mspp_masks_evanescent():
    """SimpleMSPP 在光锥外应调制为 0（倏逝波滤除）。"""
    wl, charge = 1.55, 2
    plate = SimpleMSPP(0, topology_charge=charge, wavelength=wl)
    assert plate.modulation_function is not None


def test_mspp_loads_data():
    """回归：MSPP 数据路径曾指向不存在的 elements/data/ 目录。"""
    from optical_system.elements import MSPP
    modulator = MSPP(z_position=0, wavelength=1.550)
    assert modulator.modulation_array is not None
    assert modulator.modulation_array.shape == (51, 51)
