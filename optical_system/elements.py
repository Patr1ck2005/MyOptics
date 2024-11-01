# optical_system/elements.py

import numpy as np
from utils.constants import PI

class OpticalElement:
    def __init__(self, z_position):
        """
        初始化光学元件。

        参数:
        z_position (float): 光学元件在z轴上的位置。
        """
        self.z_position = z_position

    def apply(self, U, x, y, wavelength):
        """
        应用光学元件对光场的影响。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        wavelength (float): 波长。

        返回:
        ndarray: 处理后的光场。
        """
        raise NotImplementedError("每个光学元件必须实现apply方法")

class Lens(OpticalElement):
    def __init__(self, z_position, focal_length):
        super().__init__(z_position)
        self.focal_length = focal_length

    def apply(self, U, x, y, wavelength):
        """
        透镜的相位调制。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        wavelength (float): 波长。

        返回:
        ndarray: 处理后的光场。
        """
        X, Y = np.meshgrid(x, y)
        k = 2 * PI / wavelength
        phase = np.exp(-1j * k / (2 * self.focal_length) * (X**2 + Y**2))
        return U * phase

class PhasePlate(OpticalElement):
    def __init__(self, z_position, phase_function):
        super().__init__(z_position)
        self.phase_function = phase_function

    def apply(self, U, x, y, wavelength):
        """
        相位板的相位调制。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        wavelength (float): 波长。

        返回:
        ndarray: 处理后的光场。
        """
        X, Y = np.meshgrid(x, y)
        phase_factor = self.phase_function(X, Y)
        return U * phase_factor
