import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, ifft2


def calculate_fourier(image):
    """
    计算图像的傅里叶变换，并返回傅里叶谱。
    """
    F = fftshift(fft2(image))
    return F


def apply_filter(F, loc_x, loc_y, radius, ny, nx):
    """
    根据给定的滤波参数对傅里叶谱进行滤波。

    参数:
        F: 傅里叶谱
        loc_x: 滤波中心 x 坐标
        loc_y: 滤波中心 y 坐标
        radius: 滤波半径
        ny, nx: 图像尺寸

    返回:
        F_filtered: 滤波后的傅里叶谱
        interference_filtered: 逆变换后的空间域图像
    """
    # 创建掩码
    y, x = np.ogrid[:ny, :nx]
    mask = np.zeros((ny, nx))
    mask_area = (y - loc_y) ** 2 + (x - loc_x) ** 2 <= radius ** 2
    mask[mask_area] = 1

    # 应用掩码
    F_filtered = F * mask
    F_filtered = np.roll(F_filtered, shift=(ny // 2 - loc_y, nx // 2 - loc_x))

    # 逆傅里叶变换
    interference_filtered = ifft2(ifftshift(F_filtered))
    return F_filtered, interference_filtered
