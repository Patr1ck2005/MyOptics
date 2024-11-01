import numpy as np

def angular_spectrum_propagate(U, x, y, z, wavelength, return_spectrum=False):
    """
    使用严格角谱法传播光场，并可选返回动量空间光谱。

    参数:
    U (ndarray): 输入光场。
    x (ndarray): x轴坐标。
    y (ndarray): y轴坐标。
    z (float): 传播距离。
    wavelength (float): 波长。
    return_spectrum (bool): 是否返回动量空间光谱。

    返回:
    ndarray: 传播后的光场。
    ndarray (可选): 动量空间光谱。
    """
    # 常量定义
    k = 2 * np.pi / wavelength
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    nx, ny = U.shape

    # 频率分量（不使用 fftshift，以保持与 FFT 一致的排列）
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    # 严格的传播因子 H，考虑复数 k_z
    k_squared = k**2
    KX = 2 * np.pi * FX
    KY = 2 * np.pi * FY
    kz = np.sqrt(k_squared - KX**2 - KY**2 + 0j)

    # 传播因子
    H = np.exp(1j * kz * z)
    # # 近似传播因子 H (可选)
    # H = np.exp(-1j * np.pi * wavelength * z * (FX ** 2 + FY ** 2))

    # 正向傅里叶变换
    U_fft = np.fft.fft2(U)
    U_propagated_fft = U_fft * H

    # 逆向傅里叶变换
    U_propagated = np.fft.ifft2(U_propagated_fft)

    if return_spectrum:
        # 可选返回频域光谱
        return U_propagated, U_propagated_fft
    return U_propagated
