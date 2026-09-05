"""多层膜 TMM 的物理回归。"""
import numpy as np
import pytest


def test_single_interface_fresnel_s_and_p():
    """单界面反射系数对照 Fresnel 解析公式（从 multilayer 公共 TMM 计算）。"""
    pytest.importorskip("multilayer.common.tmm")
    from multilayer.common.tmm import tmm_kx

    wl_um = 0.55
    n1, n2 = 1.0, 1.5
    # 正入射：r = (n1 - n2)/(n1 + n2)，s 与 p 相同
    Ts, Rs = tmm_kx(wl_um, [], [], complex(n1), complex(n2), kx=0.0, pol='s')
    Tp, Rp = tmm_kx(wl_um, [], [], complex(n1), complex(n2), kx=0.0, pol='p')
    r_expected = (n1 - n2) / (n1 + n2)
    assert Rs == pytest.approx(r_expected ** 2, rel=1e-10)
    assert Rp == pytest.approx(r_expected ** 2, rel=1e-10)
    assert Ts == pytest.approx(1.0 - r_expected ** 2, rel=1e-10)


def test_absorbing_stack_energy_conservation():
    """含吸收金属膜的膜堆：R + T ≤ 1（吸收占余量）。"""
    pytest.importorskip("multilayer.common.tmm")
    from multilayer.common.tmm import tmm_kx

    wl_um = 0.55
    layers = [complex(1.9, 0.0), complex(0.18, 3.4), complex(1.9, 0.0)]  # ZnO/Ag/ZnO 近似
    d_list = [30.0, 12.0, 10.0]
    for pol in ('s', 'p'):
        T, R = tmm_kx(wl_um, layers, d_list, complex(1.0), complex(1.5), kx=0.0, pol=pol)
        assert R + T <= 1.0 + 1e-9
        assert 0.0 <= R <= 1.0 and 0.0 <= T <= 1.0


def test_total_internal_reflection():
    """密介质到疏介质、超临界角：R≈1，T≈0。"""
    pytest.importorskip("multilayer.common.tmm")
    from multilayer.common.tmm import tmm_kx

    wl_um = 0.55
    n1, n2 = 1.5, 1.0
    k0 = 2.0 * np.pi / (wl_um * 1000.0)  # 1/nm
    kx = 0.8 * n1 * k0  # sinθ = 0.8 > n2/n1 → 全反射
    T, R = tmm_kx(wl_um, [], [], complex(n1), complex(n2), kx=kx, pol='s')
    assert R == pytest.approx(1.0, abs=1e-9)
    assert T == pytest.approx(0.0, abs=1e-9)


def test_tmm_k2_matches_tmm_kx():
    """D1: kx2 = kx²+ky² 驱动入口与 kx 驱动入口完全一致（径向对称性）。"""
    from multilayer.common.tmm import tmm_k2, tmm_kx

    wl_um = 0.55
    layers = [complex(1.9, 0.0), complex(0.18, 3.4), complex(1.9, 0.0)]
    d_list = [30.0, 12.0, 10.0]
    k0 = 2.0 * np.pi / (wl_um * 1000.0)
    for kx in (0.0, 0.2 * k0, 0.9 * k0, 1.3 * k0):  # 含传播区与倏逝区
        for pol in ('s', 'p'):
            T_kx, R_kx = tmm_kx(wl_um, layers, d_list, complex(1), complex(1.5),
                                kx=kx, pol=pol)
            T_k2, R_k2 = tmm_k2(wl_um, layers, d_list, complex(1), complex(1.5),
                                kx2=kx ** 2, pol=pol)
            assert T_k2 == pytest.approx(T_kx, rel=1e-12)
            assert R_k2 == pytest.approx(R_kx, rel=1e-12)


def test_tmm_k2_rejects_negative_kx2():
    """D1: 非法的负 kx2 应报错。"""
    from multilayer.common.tmm import tmm_k2

    with pytest.raises(ValueError):
        tmm_k2(0.55, [], [], complex(1), complex(1.5), kx2=-1e-12, pol='s')


def test_tmm_k2_energy_conservation_over_band():
    """D1: 传播区 T+R+A=1；临界掠入射 R=1；倏逝区 T=0（R 为场振幅比，可 >1）。"""
    from multilayer.common.tmm import tmm_k2

    layers = [complex(1.9, 0.0), complex(0.18, 3.4), complex(1.9, 0.0)]
    d_list = [30.0, 12.0, 10.0]
    for wl_um in (0.3, 0.55, 1.0, 2.0):
        k0 = 2.0 * np.pi / (wl_um * 1000.0)
        for frac in (0.0, 0.3, 0.8, 0.99, 1.0, 1.2, 1.6):
            for pol in ('s', 'p'):
                T, R = tmm_k2(wl_um, layers, d_list, complex(1), complex(1.5),
                              kx2=(frac * k0) ** 2, pol=pol)
                assert np.isfinite(T) and np.isfinite(R), (wl_um, frac, pol)
                if frac < 1.0:
                    # 传播区：能流守恒 T+R+A=1（吸收占余量）
                    assert -1e-9 <= T + R <= 1.0 + 1e-9, (wl_um, frac, pol)
                    assert 0.0 <= 1.0 - T - R <= 1.0 + 1e-9, (wl_um, frac, pol)
                else:
                    # 临界与倏逝：无净透过功率
                    assert T == pytest.approx(0.0, abs=1e-12), (wl_um, frac, pol)
                if frac == 1.0:
                    assert R == pytest.approx(1.0, abs=1e-9), (wl_um, pol)
