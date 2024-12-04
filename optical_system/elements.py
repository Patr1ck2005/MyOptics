# optical_system/elements.py
from abc import abstractmethod, ABC

import cupy as cp
from utils.constants import PI


class OpticalElement(ABC):
    def __init__(self, z_position):
        """
        初始化光学元件。

        参数:
        z_position (float): 光学元件在z轴上的位置。
        """
        self.z_position = z_position

    @property
    def config(self):
        """
        获取光学元件的配置信息。

        返回:
        dict: 包含对象所有属性的字典。
        """
        return self.__dict__

    @abstractmethod
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
        pass


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


class ObjectLens(OpticalElement):

    def __init__(self, z_position, focal_length, NA=0):
        """
        初始化非近轴条件下的理想物镜。

        参数:
        z_position (float): 物镜在z轴上的位置。
        focal_length (float): 物镜的焦距。
        NA (float): 数值孔径。
        """
        super().__init__(z_position)
        self.focal_length = focal_length
        self.NA = NA

    def apply(self, U, x, y, wavelength):
        """
        应用非近轴相位调制，实现高NA物镜的仿真。

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

        # 非近轴相位函数
        r_squared = X ** 2 + Y ** 2
        phase = cp.exp(-1j * k * (cp.sqrt(self.focal_length ** 2 + r_squared) - self.focal_length))

        if self.NA == 0:
            return U * phase

        # 数值孔径的限制
        max_angle = cp.arcsin(self.NA)  # 最大入射角，由NA确定
        max_radius = self.focal_length * cp.tan(max_angle)  # 焦平面中的最大半径
        NA_mask = cp.sqrt(r_squared) <= max_radius  # 应用NA的限制

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


class PhaseGrating(OpticalElement):
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
        phase_factor = cp.exp(1j * self.amplitude * cp.sin(2 * PI * Y / self.period))

        # 应用相位调制
        return U * phase_factor


class BlazedGrating(OpticalElement):
    def __init__(self, z_position, blaze_angle, period):
        """
        初始化闪耀光栅。

        参数:
        z_position (float): 光栅在z轴上的位置。
        blaze_angle (float): 闪耀角度（以弧度为单位）。
        period (float): 光栅的周期。
        """
        super().__init__(z_position)
        self.blaze_angle = blaze_angle
        self.period = period

    def apply(self, U, x, y, wavelength):
        """
        闪耀光栅的周期性相位调制。

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

        # 计算相位调制
        phase = k * (cp.tan(self.blaze_angle)*(Y % self.period))
        # phase = k * self.period * (cp.sin(self.blaze_angle*Y))

        # 应用相位因子
        phase_factor = cp.exp(1j * phase)

        return U * phase_factor


class Aperture(OpticalElement):
    def __init__(self, z_position, radius):
        """
        初始化光阑基类。

        参数:
        z_position (float): 光阑在z轴上的位置。
        radius (float): 光阑的半径（或半边长）。
        """
        super().__init__(z_position)
        self.radius = radius

    @abstractmethod
    def create_mask(self, X, Y):
        """
        创建光阑的遮挡掩膜。

        参数:
        X (ndarray): x轴坐标网格。
        Y (ndarray): y轴坐标网格。

        返回:
        ndarray: 遮挡掩膜。
        """
        pass

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
        aperture_mask = self.create_mask(X, Y)
        return U * aperture_mask


class CircularAperture(Aperture):
    def create_mask(self, X, Y):
        """
        创建圆形光阑的遮挡掩膜。

        参数:
        X (ndarray): x轴坐标网格。
        Y (ndarray): y轴坐标网格。

        返回:
        ndarray: 圆形遮挡掩膜。
        """
        return cp.sqrt(X ** 2 + Y ** 2) <= self.radius


class SquareAperture(Aperture):
    def create_mask(self, X, Y):
        """
        创建方形光阑的遮挡掩膜。

        参数:
        X (ndarray): x轴坐标网格。
        Y (ndarray): y轴坐标网格。

        返回:
        ndarray: 方形遮挡掩膜。
        """
        return (cp.abs(X) <= self.radius) & (cp.abs(Y) <= self.radius)


class EllipticalAperture(Aperture):
    def __init__(self, z_position, radius_x, radius_y):
        """
        初始化椭圆形光阑。

        参数:
        z_position (float): 光阑在z轴上的位置。
        radius_x (float): 椭圆形光阑在x轴的半径。
        radius_y (float): 椭圆形光阑在y轴的半径。
        """
        super().__init__(z_position, radius=None)
        self.radius_x = radius_x
        self.radius_y = radius_y

    def create_mask(self, X, Y):
        """
        创建椭圆形光阑的遮挡掩膜。

        返回:
        ndarray: 椭圆形遮挡掩膜。
        """
        return (X / self.radius_x) ** 2 + (Y / self.radius_y) ** 2 <= 1


class RectangularAperture(Aperture):
    def __init__(self, z_position, width, height):
        """
        初始化矩形光阑。

        参数:
        z_position (float): 光阑在z轴上的位置。
        width (float): 矩形光阑的宽度。
        height (float): 矩形光阑的高度。
        """
        super().__init__(z_position, radius=None)
        self.width = width
        self.height = height

    def create_mask(self, X, Y):
        """
        创建矩形光阑的遮挡掩膜。

        返回:
        ndarray: 矩形遮挡掩膜。
        """
        return (cp.abs(X) <= self.width / 2) & (cp.abs(Y) <= self.height / 2)


class CrossAperture(Aperture):
    def __init__(self, z_position, arm_width):
        """
        初始化十字形光阑。

        参数:
        z_position (float): 光阑在z轴上的位置。
        arm_width (float): 十字形光阑的臂宽。
        """
        super().__init__(z_position, radius=None)
        self.arm_width = arm_width

    def create_mask(self, X, Y):
        """
        创建十字形光阑的遮挡掩膜。

        返回:
        ndarray: 十字形遮挡掩膜。
        """
        return (cp.abs(X) <= self.arm_width / 2) | (cp.abs(Y) <= self.arm_width / 2)


class AnnularAperture(Aperture):
    def __init__(self, z_position, inner_radius, outer_radius):
        """
        初始化环形光阑。

        参数:
        z_position (float): 光阑在z轴上的位置。
        inner_radius (float): 环形光阑的内半径。
        outer_radius (float): 环形光阑的外半径。
        """
        super().__init__(z_position, radius=None)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def create_mask(self, X, Y):
        """
        创建环形光阑的遮挡掩膜。

        返回:
        ndarray: 环形遮挡掩膜。
        """
        r = cp.sqrt(X ** 2 + Y ** 2)
        return (r >= self.inner_radius) & (r <= self.outer_radius)

