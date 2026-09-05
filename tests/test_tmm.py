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
