"""M3 workflows 层验证测试（C4 Field / C3 sweep / A3 SimSpectrum / 桥接）。"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _gaussian_field(sigma=1.5, n=128, extent=16.0, wl=0.5):
    """解析可控的标量高斯场容器。"""
    x = np.linspace(-extent / 2, extent / 2, n, endpoint=False)
    X, Y = np.meshgrid(x, x)
    U = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    from workflows.field import Field
    return Field(
        intensities={'U': np.abs(U) ** 2}, x=x, y=x, wavelength=wl, z=0.0,
        complex_fields={'U': U}, meta={'tag': 'gauss'},
    ), x


def _scalar_system(wavelength=0.5, n=128, extent=16.0, waist=3.0, defocus=0.0):
    """自由传播 + 可选离焦的高斯标量系统（每调用全新实例）。"""
    import cupy as cp

    from optical_system.system import OpticalSystem
    x = cp.arange(n, dtype=cp.float64) * (extent / n) - extent / 2
    X, Y = cp.meshgrid(x, x)
    U0 = cp.exp(-(X ** 2 + Y ** 2) / waist ** 2)
    system = OpticalSystem(wavelength, x, x, U0)
    system.propagation_mode = 'Rigorous'
    _ = defocus   # 占位：调用方通过 z_positions 控制传播距离
    return system


# ---------------------------------------------------------------------------
# C4 Field
# ---------------------------------------------------------------------------
def test_field_save_load_roundtrip(tmp_path):
    f0, _ = _gaussian_field()
    path = tmp_path / 'field.npz'
    f0.save(path)
    f1 = type(f0).load(path)
    assert np.allclose(f1.x, f0.x)
    assert f1.wavelength == 0.5 and f1.z == 0.0
    assert f1.meta == {'tag': 'gauss'}
    assert np.allclose(f1.intensities['U'], f0.intensities['U'])
    assert np.allclose(f1.component('U'), f0.component('U'))
    assert set(f1.intensities) == {'U'}


def test_field_total_and_line_cut():
    f0, x = _gaussian_field(sigma=1.0)
    T = f0.total_intensity
    peak_idx = int(np.unravel_index(np.argmax(T), T.shape)[0])
    cut = f0.line_cut(axis='x')          # 过峰值的截线
    assert np.allclose(cut['total'], T[peak_idx, :])
    assert cut['coord'] is f0.x or np.allclose(cut['coord'], f0.x)
    # Gaussian 截线峰值在网格中心附近
    assert abs(cut['coord'][int(np.argmax(cut['total']))]) < f0.dx * 1.5


def test_field_shape_validation():
    from workflows.field import Field
    x = np.linspace(-1, 1, 8)
    try:
        Field(intensities={'a': np.ones((8, 8)), 'b': np.ones((4, 8))},
              x=x, y=x)
        raise AssertionError('形状不一致应当抛错')
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 指标库
# ---------------------------------------------------------------------------
def test_metrics_fwhm_matches_analytic_gaussian():
    """复振幅 U=exp(−r²/2σ²) → 强度 I=exp(−r²/σ²) 的 FWHM = 2√(ln2)·σ
    （解析锚点）。"""
    from workflows.sweep import evaluate_metrics
    sigma = 1.3
    f0, _ = _gaussian_field(sigma=sigma, n=256, extent=24.0)
    m = evaluate_metrics(f0, ('fwhm_x', 'fwhm_y', 'peak_intensity', 'centroid'))
    expected = 2.0 * np.sqrt(np.log(2.0)) * sigma
    assert abs(m['fwhm_x'] - expected) / expected < 1e-2
    assert abs(m['fwhm_y'] - expected) / expected < 1e-2
    cx, cy = m['centroid']
    assert abs(cx) < f0.dx and abs(cy) < f0.dx


# ---------------------------------------------------------------------------
# C3 ParameterSweep
# ---------------------------------------------------------------------------
def test_parameter_sweep_gaussian_defocus():
    """无透镜高斯（腰在 z=0，w0=3, zR≈56）远场扫描：z ≪ zR 区间内
    束宽随 z 单调增大，最小 FWHM 在最近的 z=7。"""
    from workflows.sweep import ParameterSweep
    sweep = ParameterSweep(
        build_system=lambda z: _scalar_system(),
        metrics=('fwhm_x', 'peak_intensity'),
    )
    z_grid = [7.0, 9.0, 11.0]
    result = sweep.run(param_grid={'z': z_grid},
                       z_of=lambda z: z,
                       keep='fields', progress=False)
    # 显式传 z_of 才是离焦扫描（z 参数同时进入 build_system 忽略之）
    assert len(result.records) == len(z_grid)
    df = result.to_dataframe().sort_values('z')
    assert set(df.columns) >= {'z', 'fwhm_x', 'peak_intensity'}
    assert df['fwhm_x'].is_monotonic_increasing   # 无透镜 → 单调展宽
    best = result.best('fwhm_x', mode='min')
    assert best['z'] == z_grid[0]
    assert len(result.fields) == len(z_grid)
    assert result.fields[(9.0,)].z == 9.0


def test_parameter_sweep_wavelength_defocus_on_focus():
    """波长扫描 + z_of：离焦面 fwhm 单调随 |z−z0| 增大。"""
    from workflows.sweep import ParameterSweep
    sweep = ParameterSweep(
        build_system=lambda wavelength: _scalar_system(wavelength=wavelength),
        metrics=('fwhm_x',),
    )
    result = sweep.run(param_grid={'wavelength': [0.4, 0.5, 0.6]},
                       z_of=lambda wavelength: 14.0,
                       keep='metrics', progress=False)
    df = result.to_dataframe().sort_values('wavelength')
    assert len(df) == 3
    # 更长波长衍射更强 → 远场束腰更大
    assert df['fwhm_x'].is_monotonic_increasing


# ---------------------------------------------------------------------------
# A3 SimSpectrum
# ---------------------------------------------------------------------------
def test_sim_spectrum_single_lambda_identity():
    """单波长、权重 1：非相干/相干合成与单色分量一致（精确恒等）。"""
    from workflows.spectrum import SimSpectrum
    sim = SimSpectrum(wavelengths=[0.5],
                      simulate=lambda wl: _make_component(wl))
    r_inc = sim.run(coherent=False, progress=False)
    r_coh = sim.run(coherent=True, progress=False)
    comp = r_inc.components[0.5]
    assert np.allclose(r_inc.field.intensities['broadband'],
                       comp.total_intensity)
    assert np.allclose(r_coh.field.intensities['U'],
                       comp.total_intensity, atol=1e-12)
    assert r_inc.field.wavelength is None


def test_sim_spectrum_weights_incoherent_sum():
    """两波长非相干合成 = 权重加权的强度和（构造性验证）。"""
    from workflows.spectrum import SimSpectrum
    sim = SimSpectrum(wavelengths=[0.4, 0.6], weights=[0.25, 0.75],
                      simulate=lambda wl: _make_component(wl))
    result = sim.run(coherent=False, progress=False)
    c1 = result.components[0.4].total_intensity
    c2 = result.components[0.6].total_intensity
    expect = 0.25 * c1 + 0.75 * c2
    assert np.allclose(result.field.intensities['broadband'], expect)
    assert abs(result.weights.sum() - 1.0) < 1e-12


def test_sim_spectrum_component_grid_mismatch():
    from workflows.spectrum import SimSpectrum
    sim = SimSpectrum(
        wavelengths=[0.4, 0.6],
        simulate=lambda wl: _make_component(wl, n=256 if wl > 0.5 else 128),
    )
    try:
        sim.run(coherent=False, progress=False)
        raise AssertionError('网格不一致应当抛错')
    except ValueError:
        pass


def _make_component(wl, n=128, extent=16.0):
    """独立单色仿真：高斯传播到固定平面（模拟 build_system(λ)）。"""
    import cupy as cp

    from optical_system.system import OpticalSystem
    x = cp.arange(n, dtype=cp.float64) * (extent / n) - extent / 2
    X, Y = cp.meshgrid(x, x)
    U0 = cp.exp(-(X ** 2 + Y ** 2) / 3.0 ** 2)
    system = OpticalSystem(float(wl), x, x, U0)
    fields = type(_make_component)  # noqa: F841 (占位避免误用)
    from workflows.field import Field
    return Field.from_scalar_system(system, z_positions=[14.0])[0]


# ---------------------------------------------------------------------------
# 桥接
# ---------------------------------------------------------------------------
def test_lift_to_vector_x_matches_scalar_propagation():
    """x 偏振提升后矢量传播 ex 与标量传播一致到 O(θ₀²)。"""
    import cupy as cp

    from propagation.propagator import AngularSpectrumPropagator
    from vector.propagator import VectorAngularSpectrumPropagator
    from workflows.bridge import lift_to_vector
    n, extent, wl = 128, 20.0, 0.5
    x = cp.arange(n, dtype=cp.float64) * (extent / n) - extent / 2
    X, Y = cp.meshgrid(x, x)
    U0 = cp.exp(-(X ** 2 + Y ** 2) / 4.0 ** 2)
    vec = lift_to_vector(cp.asnumpy(U0), cp.asnumpy(x), cp.asnumpy(x), wl, 'x')
    vp = VectorAngularSpectrumPropagator(x, x, wl)
    sp = AngularSpectrumPropagator(x, x, wl, mode='Rigorous')
    out_v = vp.propagate(vec, 8.0)
    out_s = sp.propagate(U0, 8.0)
    dev = float(cp.max(cp.abs(out_v.ex - out_s) / cp.max(cp.abs(out_s))).get())
    assert dev < 5e-3, f"桥接后矢量/标量偏差 {dev:.2e}"


def test_lift_to_vector_circular_polarization():
    import cupy as cp

    from vector.sources import vector_gaussian
    from workflows.bridge import lift_to_vector
    n, wl = 64, 0.5
    x = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, x)
    U = np.exp(-(X ** 2 + Y ** 2) / 2.0)
    vec = lift_to_vector(U, x, x, wl, 'LCP')
    ref = vector_gaussian(x, x, wl, np.sqrt(2.0), 'LCP')
    # envelope: vector_gaussian 用 exp(-r²/w²)，此处 U=exp(-r²/2) → w=√2
    # （VectorField 容器统一持 cupy 数组，比较前两侧都转 numpy）
    dev_ex = float(np.max(np.abs(cp.asnumpy(vec.ex) - cp.asnumpy(ref.ex))))
    dev_ey = float(np.max(np.abs(cp.asnumpy(vec.ey) - cp.asnumpy(ref.ey))))
    assert dev_ex < 1e-12
    assert dev_ey < 1e-12
