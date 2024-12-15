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


class ObjectLens(Lens):

    def __init__(self, z_position, focal_length, NA=0):
        """
        初始化非近轴条件下的理想物镜。

        参数:
        z_position (float): 物镜在z轴上的位置。
        focal_length (float): 物镜的焦距。
        NA (float): 数值孔径。
        """
        super().__init__(z_position, focal_length, NA)
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


class SpatialLightModulator(OpticalElement):
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
        super().__init__(z_position)
        self.modulation_function = modulation_function
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


class MomentumSpaceModulator(OpticalElement):
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
        super().__init__(z_position)
        self.modulation_function = modulation_function
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


class SinePhaseGrating(OpticalElement):
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

    def apply(self, U, x, y, **kwargs):
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


class RectAmplitudeGrating(OpticalElement):
    def __init__(self, z_position, period, slit_width):
        """
        初始化01型振幅光栅。

        参数:
        z_position (float): 光栅在z轴上的位置。
        period (float): 光栅的周期。
        slit_width (float): 狭缝的宽度。
        """
        super().__init__(z_position)
        self.period = period
        self.slit_width = slit_width

    def apply(self, U, x, y, **kwargs):
        """
        光栅的振幅调制。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。

        返回:
        ndarray: 处理后的光场。
        """
        # 创建y方向的振幅调制
        _, Y = cp.meshgrid(x, y)
        # 按周期生成狭缝位置，使用矩形函数模拟
        grating_pattern = ((cp.mod(Y, self.period) < self.slit_width)).astype(cp.float32)

        # 应用振幅调制
        return U * grating_pattern


class SineAmplitudeGrating(OpticalElement):
    def __init__(self, z_position, period, amplitude=1.0, bias=0.5):
        """
        初始化正弦振幅光栅。

        参数:
        z_position (float): 光栅在z轴上的位置。
        period (float): 光栅的周期。
        amplitude (float): 振幅调制的最大值（默认为1.0）。
        bias (float): 振幅调制的偏置（默认为0.5）。
        """
        super().__init__(z_position)
        self.period = period
        self.amplitude = amplitude
        self.bias = bias

    def apply(self, U, x, y, **kwargs):
        """
        光栅的振幅调制。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。

        返回:
        ndarray: 处理后的光场。
        """
        # 创建y方向的振幅调制
        _, Y = cp.meshgrid(x, y)
        grating_pattern = self.bias + self.amplitude * cp.sin(2 * PI * Y / self.period)

        # 确保振幅在[0, 1]范围内
        grating_pattern = cp.clip(grating_pattern, 0, 1)

        # 应用振幅调制
        return U * grating_pattern


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


class CircularAperture(Aperture):
    def __init__(self, z_position, radius):
        """
        初始化圆形光阑。

        参数:
        z_position (float): 光阑在z轴上的位置。
        radius (float): 光阑的半径。
        """
        super().__init__(z_position=z_position, size=radius)

    def create_mask(self, X, Y):
        """
        创建圆形光阑的遮挡掩膜。

        参数:
        X (ndarray): x轴坐标网格。
        Y (ndarray): y轴坐标网格。

        返回:
        ndarray: 圆形遮挡掩膜。
        """
        return cp.sqrt(X ** 2 + Y ** 2) <= self.size


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
        return (cp.abs(X) <= self.size) & (cp.abs(Y) <= self.size)


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


if __name__ == '__main__':
    import cupy as cp
    import matplotlib.pyplot as plt

    # 定义输入光场和坐标
    x = cp.linspace(-10, 10, 500)  # x 坐标
    y = cp.linspace(-10, 10, 500)  # y 坐标
    U_in = cp.ones((500, 500))  # 平面波光场

    # 创建光栅实例
    z_position = 0
    period = 5.0
    slit_width = 2.0
    grating = RectAmplitudeGrating(z_position, period, slit_width)

    # 应用光栅
    U_out = grating.apply(U_in, x, y)

    # 将结果转换为 numpy 以便绘图
    U_out_np = cp.asnumpy(U_out)

    # 可视化结果
    plt.figure(figsize=(10, 5))
    plt.imshow(U_out_np.real, extent=(-10, 10, -10, 10), cmap='gray', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.title('01型振幅光栅调制后的光场')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
