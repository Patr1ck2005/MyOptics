"""相干传输矩阵方法（TMM），N-导纳形式，kx 驱动，s/p 双偏振。

本实现收敛自 round2/angle_scan_spectra_variants.py（四份历史实现中最完整，
含 kz 物理分支选取），round1/round2 脚本统一改为从这里导入。

约定：
- 长度单位 nm，波长单位 μm（k0 = 2π/(wl_um·1000)）
- kx 为跨层守恒的横向波矢（1/nm）
- 膜堆顺序为入射介质 → 有限层列表 → 出射介质（半无限）
"""
import numpy as np


def _kz_physical(kz):
    """Pick the physical branch: Im(kz) >= 0; if ~real, Re(kz) >= 0."""
    if np.imag(kz) < 0:
        return -kz
    if np.isclose(np.imag(kz), 0.0, atol=1e-14) and np.real(kz) < 0:
        return -kz
    return kz


def kz_of(N_j, k0, kx):
    """Layer longitudinal wavevector with physical-branch selection."""
    return _kz_physical(np.lib.scimath.sqrt(N_j ** 2 * k0 ** 2 - kx ** 2))


def tmm_kx(wl_um, N_layers, d_nm_list, N_inc, N_exit, kx, pol="s"):
    """
    Coherent TMM at fixed transverse wavevector kx (in 1/nm).

    Parameters
    ----------
    wl_um : float
        Wavelength in μm.
    N_layers : list of complex
        Complex refractive indices of finite layers, incident->exit order.
    d_nm_list : list of float
        Thicknesses (nm) of each finite layer.
    N_inc, N_exit : complex
        Complex refractive indices of incident and exit (semi-infinite) media.
    kx : float
        Transverse wavevector (1/nm), conserved across all layers.
    pol : 's' or 'p'

    Returns
    -------
    T, R : float
        Transmittance (Poynting-flux ratio) and reflectance (|r|^2).
    """
    if pol not in ("s", "p"):
        raise ValueError(f"pol must be 's' or 'p', got {pol!r}")

    k0 = 2.0 * np.pi / (wl_um * 1000.0)  # 1/nm

    kz_inc = kz_of(N_inc, k0, kx)
    kz_exit = kz_of(N_exit, k0, kx)
    kz_layers = [kz_of(N_j, k0, kx) for N_j in N_layers]

    if pol == "s":
        Y_inc = kz_inc / k0
        Y_exit = kz_exit / k0
        Y_layers = [kzj / k0 for kzj in kz_layers]
    else:  # 'p'
        Y_inc = N_inc ** 2 * k0 / kz_inc
        Y_exit = N_exit ** 2 * k0 / kz_exit
        Y_layers = [Nj ** 2 * k0 / kzj for Nj, kzj in zip(N_layers, kz_layers)]

    M = np.eye(2, dtype=complex)
    for Y_j, dj, kzj in zip(Y_layers, d_nm_list, kz_layers):
        delta = kzj * dj
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        Mj = np.array([[cos_d, -1j * sin_d / Y_j],
                       [-1j * Y_j * sin_d, cos_d]], dtype=complex)
        M = M @ Mj

    m11, m12, m21, m22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    Y0, Ys = Y_inc, Y_exit
    denom = Y0 * m11 + Y0 * Ys * m12 + m21 + Ys * m22
    r = (Y0 * m11 + Y0 * Ys * m12 - m21 - Ys * m22) / denom
    t = 2.0 * Y0 / denom

    reY0 = np.real(Y0)
    T = (np.real(Ys) / reY0) * np.abs(t) ** 2 if reY0 > 0 else 0.0
    R = np.abs(r) ** 2
    return float(T), float(R)
