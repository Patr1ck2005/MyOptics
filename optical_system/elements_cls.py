# optical_system/elements.py
from abc import abstractmethod, ABC

import cupy as cp
from scipy.interpolate import RegularGridInterpolator

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

    def __init__(self, z_position, focal_length, NA=0, D=None):
        """
        初始化考虑数值孔径(NA)的透镜。

        参数:
        z_position (float): 透镜在z轴上的位置。
        focal_length (float): 透镜的焦距。
        NA (float): 数值孔径。
        """
        super().__init__(z_position)
        self.focal_length = focal_length
        if D is not None:
            self.NA = D/2/cp.sqrt((D/2)**2+self.focal_length**2)
        else:
            self.NA = NA
        # self.D = D

    @property
    def back_position(self):
        return self.z_position+self.focal_length

    @property
    def forw_position(self):
        return self.z_position-self.focal_length

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


class SpatialPlate(OpticalElement):
    def __init__(self, z_position, modulation_function):
        super().__init__(z_position)
        self.modulation_function = modulation_function

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
        modulate_factor = self.modulation_function(X, Y)
        return U * modulate_factor


class MomentumSpacePlate(OpticalElement):
    def __init__(self, z_position, modulation_function):
        """
        初始化动量空间的调制器。

        参数:
        z_position (float): 相位板在z轴上的位置。
        modulation_function (function): 一个接受 kx 和 ky 的函数，定义了动量空间的调制。
        """
        super().__init__(z_position)
        self.modulation_function = modulation_function

    def apply(self, U, x, y, wavelength):
        """
        应用动量空间调制板的调制。

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

        # 应用动量空间调制
        phase_factor_k = self.modulation_function(KX, KY)
        U_k_modified = U_k * phase_factor_k

        # 返回到实空间 (逆傅里叶变换)
        U_modified = cp.fft.ifft2(U_k_modified)

        return U_modified


class SpatialLightModulator(SpatialPlate):
    def __init__(self, z_position, modulation_function=None,
                 modulation_array=None, mod_x=None, mod_y=None):
        """
        初始化空间光调制器。

        参数:
        z_position (float): 光调制器在z轴上的位置。
        modulation_function (function): 接受 x, y 的函数，用于产生实空间调制因子。
        modulation_array (ndarray): 已定义好的调制数组。
        mod_x (ndarray): 与 modulation_array 对应的 x 坐标。
        mod_y (ndarray): 与 modulation_array 对应的 y 坐标。
        """
        super().__init__(z_position=z_position,
                         modulation_function=modulation_function)
        # self.modulation_function = modulation_function
        self.modulation_array = modulation_array
        self.mod_x = mod_x
        self.mod_y = mod_y

    def _get_modulation(self, x, y):
        # 如果有函数定义，则直接计算
        if self.modulation_function is not None:
            X, Y = cp.meshgrid(x, y)
            return self.modulation_function(X, Y)

        # 否则，如果有调制数组和坐标，则使用插值
        if self.modulation_array is not None and self.mod_x is not None and self.mod_y is not None:
            # 由于插值在CPU上，需先将数据转移到CPU
            mod_array_cpu = cp.asnumpy(self.modulation_array)
            mod_x_cpu = cp.asnumpy(self.mod_x)
            mod_y_cpu = cp.asnumpy(self.mod_y)
            interp = RegularGridInterpolator((mod_y_cpu, mod_x_cpu), mod_array_cpu, bounds_error=False, fill_value=0)

            # 创建待插值点阵列
            X, Y = cp.meshgrid(x, y)
            points = cp.stack([Y.ravel(), X.ravel()], axis=-1)
            points_cpu = cp.asnumpy(points)

            # 插值到目标坐标
            mod_values = interp(points_cpu).reshape(Y.shape)
            return cp.asarray(mod_values)

        # 如果既没有函数也没有数组，则不做调制
        return cp.ones((y.size, x.size), dtype=cp.float32)

    def apply(self, U, x, y, wavelength):
        modulation = self._get_modulation(x, y)
        # 对输入光场施加调制
        U_modified = U * modulation
        return U_modified


class MomentumSpaceModulator(MomentumSpacePlate):
    def __init__(self, z_position, modulation_function=None,
                 modulation_array=None, mod_kx=None, mod_ky=None):
        """
        初始化动量空间光调制器。

        参数:
        z_position (float): 调制器在z轴的位置。
        modulation_function (function): 接受 kx, ky 的函数，用于产生动量空间调制因子。
        modulation_array (ndarray): 已定义好的动量空间调制数组。
        mod_kx (ndarray): 与 modulation_array 对应的 kx 坐标。
        mod_ky (ndarray): 与 modulation_array 对应的 ky 坐标。
        """
        super().__init__(z_position=z_position,
                         modulation_function=modulation_function)
        # self.modulation_function = modulation_function
        self.modulation_array = modulation_array
        self.mod_kx = mod_kx
        self.mod_ky = mod_ky

    def _get_modulation_k(self, KX, KY):
        # 如果有函数定义，则直接计算
        if self.modulation_function is not None:
            return self.modulation_function(KX, KY)

        # 如果有给定的调制数组和对应动量坐标，则插值
        if self.modulation_array is not None and self.mod_kx is not None and self.mod_ky is not None:
            # # 频域坐标变换
            # KX = cp.fft.fftshift(KX)
            # KY = cp.fft.fftshift(KY)

            mod_array_cpu = cp.asnumpy(self.modulation_array)
            mod_kx_cpu = cp.asnumpy(self.mod_kx)
            mod_ky_cpu = cp.asnumpy(self.mod_ky)
            interp = RegularGridInterpolator((mod_ky_cpu, mod_kx_cpu), mod_array_cpu,
                                             bounds_error=False,
                                             method='cubic',
                                             # method='linear',
                                             fill_value=1.0)

            points = cp.stack([KY.ravel(), KX.ravel()], axis=-1)
            points_cpu = cp.asnumpy(points)
            mod_values = interp(points_cpu).reshape(KY.shape)

            # 频域坐标变换
            mod_values = cp.asarray(mod_values)
            # mod_values = cp.fft.ifftshift(mod_values)
            return mod_values

        # 如果既没有函数也没有数组，则不做调制
        return cp.ones_like(KX)

    def apply(self, U, x, y, wavelength):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        kx = cp.fft.fftfreq(x.size, dx) * 2 * PI
        ky = cp.fft.fftfreq(y.size, dy) * 2 * PI
        KX, KY = cp.meshgrid(kx, ky)

        U_k = cp.fft.fft2(U)

        # 获取动量空间调制因子
        mod_factor = self._get_modulation_k(KX, KY)
        U_k_modified = U_k * mod_factor

        U_modified = cp.fft.ifft2(U_k_modified)
        return U_modified


class Aperture(OpticalElement):
    def __init__(self, z_position, size):
        """
        初始化光阑基类。

        参数:
        z_position (float): 光阑在z轴上的位置。
        radius (float): 光阑的半径（或半边长）。
        """
        super().__init__(z_position)
        self.size = size

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



from optical_system.elements.grating import *
from optical_system.elements.apertures import *
from optical_system.elements.lens import *
