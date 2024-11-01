# propagation/angular_spectrum_longitudinal.py

import numpy as np
from utils.constants import PI

def angular_spectrum_propagate_longitudinal(U_initial, x_axis, z_axis, wavelength):
    """
    使用角谱法计算纵截面（x-z平面）上的光场分布。

    参数:
    U_initial (ndarray): 初始平面光场，1D数组（x方向）。
    x_axis (ndarray): x坐标。
    z_axis (ndarray): z方向传播距离数组。
    wavelength (float): 波长。

    返回:
    ndarray: 纵截面（x-z平面）上的光场分布，2D数组。
    """
    k = 2 * PI / wavelength
    dx = x_axis[1] - x_axis[0]
    nx = len(x_axis)

    # 计算频率坐标
    fx = np.fft.fftfreq(nx, d=dx)

    # 初始化纵截面上的光场分布
    U_longitudinal = np.zeros((len(z_axis), nx), dtype=complex)

    # 在起始位置初始化光场
    U_longitudinal[0, :] = U_initial

    # 逐步计算每个z位置的光场分布
    for i, z in enumerate(z_axis[1:], start=1):
        # 构建传播算子
        H = np.exp(-1j * PI * wavelength * z * (fx**2))

        # 执行FFT和传播计算
        U_fft = np.fft.fft(U_longitudinal[i-1, :])
        U_propagated_fft = U_fft * H
        U_propagated = np.fft.ifft(U_propagated_fft)

        # 保存当前z位置的光场
        U_longitudinal[i, :] = U_propagated

    return U_longitudinal


def calculate_field_on_grid(U_initial, x_axis, y_axis, dx, dy, X, Y, Z, wavelength):
    """
    计算纵截面上的场强分布。

    参数:
    - U_initial: 初始平面光场分布，二维复数数组
    - x_axis, y_axis: x 和 y 方向的坐标数组
    - dx, dy: x 和 y 方向的采样间隔
    - X, Y, Z: 实空间中的 x、y 和 z 网格
    - wavelength: 波长

    返回:
    - field: 纵截面网格 (X, Z) 上的复振幅分布
    """
    # 计算波数 k
    k = 2 * PI / wavelength

    # 构建动量空间坐标
    kx_values = np.fft.fftfreq(len(x_axis), d=dx) * 2 * PI
    ky_values = np.fft.fftfreq(len(y_axis), d=dy) * 2 * PI
    KX, KY = np.meshgrid(kx_values, ky_values, indexing='ij')

    # 对初始平面光场做2D傅里叶变换
    U_fft = np.fft.fft2(U_initial)

    # 计算 kz，以满足 kz^2 + kx^2 + ky^2 = k^2
    kz_square = k**2 - KX**2 - KY**2
    kz_square[kz_square < 0] = 0  # 负数取零，避免消失波
    KZ = np.sqrt(kz_square)

    # 展开 KX, KY, KZ
    KX_flat = KX.ravel()
    KY_flat = KY.ravel()
    KZ_flat = KZ.ravel()
    U_fft_flat = U_fft.ravel()

    # 计算每个点的场
    exponent = 1j * (np.outer(KX_flat, X.ravel()) + np.outer(KY_flat, Y.ravel()) + np.outer(KZ_flat, Z.ravel()))
    field_k = U_fft_flat[:, None] * np.exp(exponent)

    # 对动量空间进行积分
    field = np.sum(field_k, axis=0).reshape(X.shape)

    return field
