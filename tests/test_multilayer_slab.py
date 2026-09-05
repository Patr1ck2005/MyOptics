"""MultilayerSlab：多层膜 × 角谱耦合的验收测试链。"""
import numpy as np
import pytest

from conftest import GPU_OK

if GPU_OK:
    import cupy as cp
    from optical_system.elements import MultilayerSlab
    from optical_system.system import OpticalSystem
    from utils.constants import PI

pytestmark = pytest.mark.gpu

# 测试膜堆：空气 | ZnO(40nm) | Ag(10nm) | ZnO(10nm) | 空气（研究线真实结构）
ZNOK_AG_STACK = {
    "N_layers": [complex(1.9, 0.0), complex(0.18, 3.4), complex(1.9, 0.0)],
    "d_nm_list": [40.0, 10.0, 10.0],
}


def _grid(nx=129):
    x = cp.linspace(-10, 10, nx)
    return x


@pytest.mark.skipif(not GPU_OK, reason="GPU 不可用")
class TestMultilayerSlab:
    def test_air_slab_degenerates_to_free_propagation(self):
        """锚点 1：单层空气板 H(k) ≡ exp(i·kz·d)，元件输出 ≡ 标准角谱传播。"""
        wl, d_nm = 1.0, 5000.0  # 5 μm 空气板
        slab = MultilayerSlab(0, [complex(1.0)], [d_nm], wl)
        x = _grid()
        X, Y = cp.meshgrid(x, x)
        U = cp.exp(-(X ** 2 + Y ** 2) / 9.0).astype(cp.complex128)

        out_slab = slab.apply(U, x, x)
        # 标准角谱传播同样厚度
        k0 = 2 * PI / wl
        dx = float(cp.asnumpy(x[1]) - cp.asnumpy(x[0]))
        fx = cp.fft.fftfreq(x.size, d=dx)
        FY, FX = cp.meshgrid(fx, fx)
        kz = cp.sqrt((k0 ** 2 - (2 * PI * FX) ** 2 - (2 * PI * FY) ** 2).astype(cp.complex128))
        d_um = d_nm / 1000.0
        out_ref = cp.fft.ifft2(cp.fft.fft2(U) * cp.exp(1j * kz * d_um))
        # 元件插值引入 O(1e-3) 误差（1D kr 采样密度决定），非逐位但需高精度一致
        err = float(cp.abs(out_slab - out_ref).max() / cp.abs(out_ref).max())
        assert err < 5e-3, f"空气板退化测试误差 {err:.2e}"

    def test_radial_symmetry_of_H(self):
        """锚点 2：H(kx, ky) 严格径向对称。"""
        wl = 0.55
        slab = MultilayerSlab(0, **ZNOK_AG_STACK, wavelength=wl)
        x = _grid(65)
        H = slab.transfer_function_grid(x, x)
        # fftfreq 顺序下 DC 位于 [0,0]；等 kr 的两点 H 应一致
        H_axis_x = H[0, 10]   # (kx=10Δf, ky=0)
        H_axis_y = H[10, 0]   # (kx=0, ky=10Δf)
        H_diag = H[7, 7]      # 7√2 ≈ 9.9·Δf ≈ 10·Δf
        assert abs(H_axis_x - H_axis_y) < 1e-9 * (abs(H_axis_x) + 1e-30)
        assert abs(H_diag - H_axis_x) < 5e-3 * (abs(H_axis_x) + abs(H_diag) + 1e-30)

    def test_H_matches_tmm_kx_cross_section(self):
        """锚点 3：H(kx, ky=0) 与 tmm_k2_amplitudes 逐点一致（振幅口径）。"""
        from multilayer.common.tmm import tmm_k2_amplitudes

        wl = 0.55
        slab = MultilayerSlab(0, **ZNOK_AG_STACK, wavelength=wl)
        x = _grid(129)
        dx = float(cp.asnumpy(x[1]) - cp.asnumpy(x[0]))
        fx = cp.fft.fftfreq(129, d=dx)  # cycles/μm
        H = slab.transfer_function_grid(x, x)
        H_row = H[0, :]  # fftfreq 顺序下 ky=0 是第 0 行
        k_conv = 2 * PI / 1000.0  # 元件内部 cycles/μm → rad/nm 的换算
        # 抽查若干频率点
        for j in (0, 5, 20, 40, 60, 64, 100):
            kx_nm = float(fx[j]) * k_conv
            t_amp, _ = tmm_k2_amplitudes(wl, slab.N_layers, slab.d_nm_list,
                                         1.0 + 0j, 1.0 + 0j, kx2=kx_nm ** 2, pol='s')
            # 1D 线性插值误差 ~3e-5（n_kr=768），容差取 1e-4
            assert abs(H_row[j] - t_amp) < 1e-4 * (abs(t_amp) + 1e-30), (j, kx_nm)

    def test_evanescent_enhancement_present(self):
        """超透镜物理：p 偏振倏逝带 |H| 可显著大于 1（倏逝增强）。"""
        # Ag 邻近场增强结构：ZnO/Ag/ZnO 在波长 0.55 的 p 倏逝带
        slab = MultilayerSlab(0, **ZNOK_AG_STACK, wavelength=0.55, polarization='p')
        kr, H = slab.radial_H()
        k0 = 2 * np.pi / (0.55 * 1000.0)
        band = (kr > 1.05 * k0) & (kr < 1.6 * k0)
        assert band.any()
        assert float(np.abs(H[band]).max()) > 1.0, "p 偏振倏逝带未出现增强"

    def test_propagation_band_power_matches_tmm_T(self):
        """传播带内 |H(kr)|² 应等于 TMM 功率透过率 T(kr)。"""
        from multilayer.common.tmm import tmm_k2
        wl = 0.55
        slab = MultilayerSlab(0, **ZNOK_AG_STACK, wavelength=wl, polarization='s')
        kr, H = slab.radial_H()
        k0 = 2 * np.pi / (wl * 1000.0)
        mask = kr < 0.9 * k0
        for i in np.where(mask)[0][::40]:
            T, _ = tmm_k2(wl, slab.N_layers, slab.d_nm_list,
                          1.0 + 0j, 1.0 + 0j, kx2=float(kr[i]) ** 2, pol='s')
            assert abs(abs(H[i]) ** 2 - T) < 5e-3, (kr[i], abs(H[i]) ** 2, T)

    def test_zero_thickness_slab_is_identity(self):
        """零厚度膜堆：t→1（无界面反射的空膜堆），H≡1，输出≡输入。"""
        slab = MultilayerSlab(0, [complex(1.0)], [0.0], 1.0)
        x = _grid()
        X, Y = cp.meshgrid(x, x)
        U = cp.exp(-(X ** 2 + Y ** 2) / 9.0).astype(cp.complex128)
        out = slab.apply(U, x, x)
        assert float(cp.abs(out - U).max()) < 1e-12

    def test_in_system_end_to_end(self):
        """集成：OpticalSystem 中平板 + 膜堆按 z 顺序应用。"""
        wl = 1.0
        x = np.linspace(-10, 10, 129)
        X, Y = np.meshgrid(x, x)
        U0 = np.exp(-(X ** 2 + Y ** 2) / 9.0).astype(np.complex128)
        system = OpticalSystem(wl, x, x, U0)
        slab = MultilayerSlab(5.0, **ZNOK_AG_STACK, wavelength=wl)
        system.add_element(slab)
        results = system.propagate_to_cross_sections([0.0, 10.0])
        assert set(results.keys()) == {0.0, 10.0}
        U_after = results[10.0][0][0]
        assert np.isfinite(np.abs(U_after)).all()
        # 能量应明显衰减（10nm Ag @1μm 透过率有限）
        e_in = float(np.sum(np.abs(U0) ** 2))
        e_out = float(np.sum(np.abs(U_after) ** 2))
        assert 0 < e_out < e_in
