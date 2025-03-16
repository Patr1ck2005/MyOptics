import cupy as cp

from optical_system.elements import OpticalElement, SpatialPlate
from utils.constants import PI

class SinePhaseGrating(SpatialPlate):
    def __init__(self, z_position, period, amplitude):
        """
        初始化光栅。

        参数:
        z_position (float): 光栅在z轴上的位置。
        period (float): 光栅的周期。
        amplitude (float): 光栅的相位调制幅度。
        """
        self.period = period
        self.amplitude = amplitude
        super().__init__(z_position, modulation_function=lambda X, Y: cp.exp(1j * self.amplitude * cp.sin(2 * PI * Y / self.period)))


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