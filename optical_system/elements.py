# optical_system/elements.py

import cupy as cp
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

    def __init__(self, z_position, focal_length, NA=0):
        """
        初始化考虑数值孔径(NA)的透镜。

        参数:
        z_position (float): 透镜在z轴上的位置。
        focal_length (float): 透镜的焦距。
        NA (float): 数值孔径。
        """
        super().__init__(z_position)
        self.focal_length = focal_length
        self.NA = NA

    def apply(self, U, x, y, wavelength):
        """
        透镜的相位调制，考虑数值孔径的影响。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        wavelength (float): 波长。

        返回:
        ndarray: 处理后的光场。
        """
        X, Y = cp.meshgrid(x, y)
        k = 2 * PI / wavelength
        phase = cp.exp(-1j * k / (2 * self.focal_length) * (X ** 2 + Y ** 2))

        if self.NA == 0:
            return U * phase
        # Calculate the spatial cutoff frequency based on NA
        max_angle = cp.arcsin(self.NA)  # Max angle given by NA
        max_radius = self.focal_length * cp.tan(max_angle)  # Corresponding radius in focal plane
        NA_mask = cp.sqrt(X ** 2 + Y ** 2) <= max_radius  # Mask for NA limitation

        return U * phase * NA_mask


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
        X, Y = cp.meshgrid(x, y)
        phase_factor = self.phase_function(X, Y)
        return U * phase_factor


class MomentumSpacePhasePlate(OpticalElement):
    def __init__(self, z_position, phase_function):
        """
        初始化动量空间的相位板。

        参数:
        z_position (float): 相位板在z轴上的位置。
        phase_function_k (function): 一个接受 kx 和 ky 的函数，定义了动量空间的相位调制。
        """
        super().__init__(z_position)
        self.phase_function = phase_function

    def apply(self, U, x, y, wavelength):
        """
        应用动量空间相位板的相位调制。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        wavelength (float): 波长。

        返回:
        ndarray: 处理后的光场。
        """
        # 计算动量空间坐标 (kx, ky)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        kx = cp.fft.fftfreq(x.size, dx) * 2 * PI
        ky = cp.fft.fftfreq(y.size, dy) * 2 * PI
        KX, KY = cp.meshgrid(kx, ky)

        # 进入动量空间 (傅里叶变换)
        U_k = cp.fft.fft2(U)

        # 应用动量空间相位调制
        phase_factor_k = self.phase_function(KX, KY)
        U_k_modified = U_k * phase_factor_k

        # 返回到实空间 (逆傅里叶变换)
        U_modified = cp.fft.ifft2(U_k_modified)

        return U_modified


class Grating(OpticalElement):
    def __init__(self, z_position, period, amplitude):
        """
        初始化光栅。

        参数:
        z_position (float): 光栅在z轴上的位置。
        period (float): 光栅的周期。
        amplitude (float): 光栅的相位调制幅度。
        """
        super().__init__(z_position)
        self.period = period
        self.amplitude = amplitude

    def apply(self, U, x, y, wavelength):
        """
        光栅的相位调制。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        wavelength (float): 波长。

        返回:
        ndarray: 处理后的光场。
        """
        # 创建y方向的相位调制
        _, Y = cp.meshgrid(x, y)
        phase = cp.exp(1j * self.amplitude * cp.sin(2 * PI * Y / self.period))

        # 应用相位调制
        return U * phase


class Aperture(OpticalElement):
    def __init__(self, z_position, radius):
        """
        初始化光阑。

        参数:
        z_position (float): 光阑在z轴上的位置。
        radius (float): 光阑的半径。
        """
        super().__init__(z_position)
        self.radius = radius

    def apply(self, U, x, y, wavelength):
        """
        应用光阑对光场的影响。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        wavelength (float): 波长。

        返回:
        ndarray: 处理后的光场。
        """
        X, Y = cp.meshgrid(x, y)
        aperture_mask = cp.sqrt(X**2 + Y**2) <= self.radius
        return U * aperture_mask


