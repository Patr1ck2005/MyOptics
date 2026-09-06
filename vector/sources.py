"""矢量高斯光源：线/圆/径向/方位偏振的横向场生成。

约定与标量框架 GaussianBeam 一致：束腰处（z=0）平面截面——高斯包络 +
偏振方向单位矢量（径向/方位偏振为位置相关的横向单位矢量）。

径向 (radial) 与方位 (azimuthal) 偏振光是矢量光学的标准验证光源：
- 径向偏振经高NA聚焦后产生强纵向分量（焦点处 Ez 占主导）；
- 方位偏振经高NA聚焦后产生中空环强度（无纵向分量，纯横向）。
两者经矢量传播器的横场投影后是严格自由空间解。
"""
import cupy as cp

from vector.field import VectorField


def vector_gaussian(x, y, wavelength, waist_radius, polarization='x',
                    z_position=0.0):
    """生成偏振受控的高斯光束横向场（束腰附近平面截面）。

    参数:
    x, y (cp.ndarray): 1D 坐标。
    wavelength (float): 波长。
    waist_radius (float): 束腰半径 w0（振幅 1/e 半径）。
    polarization (str): 'x' | 'y' | '45' | '135' | 'LCP' | 'RCP'
        | 'radial' | 'azimuthal'。
    z_position (float): 光源所在的初始平面（仅记录，不改变场）。

    返回:
    VectorField（ez=None，等待传播器投影重构；LCP/RCP 圆偏振的 Jones
    约定：从 +z 迎着来光方向看逆时针为 LCP，(x̂ ± i·ŷ)/√2，此约定与
    波片元件一致）。
    """
    x = cp.asarray(x)
    y = cp.asarray(y)
    X, Y = cp.meshgrid(x, y)
    envelope = cp.exp(-(X ** 2 + Y ** 2) / waist_radius ** 2)

    ex = cp.zeros_like(envelope)
    ey = cp.zeros_like(envelope)
    amp = 1.0 / cp.sqrt(2.0)   # 圆偏振分量归一

    if polarization == 'x':
        ex = envelope * 1.0
    elif polarization == 'y':
        ey = envelope * 1.0
    elif polarization == '45':
        ex = envelope * amp
        ey = envelope * amp
    elif polarization == '135':
        ex = envelope * amp
        ey = envelope * (-amp)
    elif polarization == 'LCP':
        ex = envelope * amp
        ey = envelope * (1j * amp)
    elif polarization == 'RCP':
        ex = envelope * amp
        ey = envelope * (-1j * amp)
    elif polarization == 'radial':
        r = cp.sqrt(X ** 2 + Y ** 2)
        r_safe = cp.where(r > 0, r, 1.0)
        ex = cp.where(r > 0, envelope * X / r_safe, 0.0)
        ey = cp.where(r > 0, envelope * Y / r_safe, 0.0)
    elif polarization == 'azimuthal':
        r = cp.sqrt(X ** 2 + Y ** 2)
        r_safe = cp.where(r > 0, r, 1.0)
        ex = cp.where(r > 0, envelope * (-Y / r_safe), 0.0)
        ey = cp.where(r > 0, envelope * (X / r_safe), 0.0)
    else:
        raise ValueError(f"未知偏振: {polarization!r}")

    return VectorField(ex, ey, None, x, y, float(wavelength))
