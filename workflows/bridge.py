"""标量 → 矢量场桥接：把标量场提升为 VectorField 的入口。

典型工作流：标量 OpticalSystem 里用宽谱/扫描算完传播链，到了需要
偏振/纵向场的环节切到矢量引擎——用 ``lift_to_vector`` 把标量场升为
指定偏振态的 VectorField，再交给 VectorOpticalSystem / 矢量传播器
（后者会在首次传播时做横场投影，剔除非横场残量）。

约定：x 线偏振提升后的场与标量传播器的输出在近轴极限下直接可比；
圆偏振提升包含 ±i 相位（LCP=(1,+i)/√2，与 vector.sources 一致）。
"""
from __future__ import annotations

import numpy as np

from vector.field import VectorField


def lift_to_vector(U, x, y, wavelength, polarization='x',
                   backend='numpy'):
    """把标量复场 U 提升为指定偏振态的 VectorField。

    参数:
    U (np.ndarray): 标量复场 (ny, nx)。
    x, y (np.ndarray): 1D 坐标。
    wavelength (float): 波长。
    polarization (str): 'x' | 'y' | '45' | '135' | 'LCP' | 'RCP'
        | 'radial' | 'azimuthal'（radial/azimuthal 按位置生成横向
        单位矢，其余按固定 Jones 矢量；归一化系数与
        ``vector.sources.vector_gaussian`` 完全一致）。
    backend (str): 'numpy'（默认，VectorOpticalSystem 内部会转 cupy）
        或 'cupy'（当场转入 GPU）。

    返回:
    VectorField（ez=None）。
    """
    U = np.asarray(U)
    if backend == 'cupy':
        import cupy as cp
        U = cp.asarray(U)
        xp = cp
    else:
        xp = np

    env = xp.asarray(U)
    a = 1.0 / xp.sqrt(2.0)
    if polarization == 'x':
        ex, ey = env * 1.0, xp.zeros_like(env)
    elif polarization == 'y':
        ex, ey = xp.zeros_like(env), env * 1.0
    elif polarization == '45':
        ex, ey = env * a, env * a
    elif polarization == '135':
        ex, ey = env * a, env * (-a)
    elif polarization == 'LCP':
        ex, ey = env * a, env * (1j * a)
    elif polarization == 'RCP':
        ex, ey = env * a, env * (-1j * a)
    elif polarization in ('radial', 'azimuthal'):
        xg = xp.asarray(np.meshgrid(np.asarray(x), np.asarray(y))[0])
        yg = xp.asarray(np.meshgrid(np.asarray(x), np.asarray(y))[1])
        r = xp.sqrt(xg ** 2 + yg ** 2)
        r_safe = xp.where(r > 0, r, 1.0)
        if polarization == 'radial':
            ex = xp.where(r > 0, env * xg / r_safe, 0.0)
            ey = xp.where(r > 0, env * yg / r_safe, 0.0)
        else:
            ex = xp.where(r > 0, env * (-yg / r_safe), 0.0)
            ey = xp.where(r > 0, env * (xg / r_safe), 0.0)
    else:
        raise ValueError(f"未知偏振: {polarization!r}")

    return VectorField(ex, ey, None, x, y, float(wavelength))
