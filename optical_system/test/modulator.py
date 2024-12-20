import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

from optical_system.elements import OpticalElement
from utils.constants import PI


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
                                             # method='cubic',
                                             method='linear',
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

        # DEBUG
        plt.imshow(cp.abs(U).get())
        plt.show()
        # DEBUG
        U_k_plot = cp.fft.fftshift(U_k)
        plt.imshow(cp.abs(U_k_plot).get())
        plt.show()

        # 获取动量空间调制因子
        mod_factor = self._get_modulation_k(KX, KY)
        U_k_modified = U_k * mod_factor

        # DEBUG
        mod_factor_plot = cp.fft.fftshift(mod_factor)
        plt.imshow((cp.abs(mod_factor_plot)**2).get(), cmap='hot', vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
        # DEBUG
        U_k_modified_plot = cp.fft.fftshift(U_k_modified)
        plt.imshow(cp.abs(U_k_modified_plot).get())
        plt.show()

        U_modified = cp.fft.ifft2(U_k_modified)
        return U_modified


# 生成测试数据
def generate_gaussian_beam(x, y, w0):
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2
    U = np.exp(-r2 / w0**2)
    return U

# 主程序
def main():
    wavelength = 1.550
    # 加载动量空间调制器数据
    efficiency = np.load("efficiency.npy")  # 加载效率数组
    phase = np.load("phase.npy")            # 加载相位数组

    # 计算总调制因子：sqrt(efficiency) * exp(i * phase)
    modulation_array = np.sqrt(efficiency) * np.exp(1j * phase)
    # DEBUG
    # plt.imshow(phase, cmap='twilight', alpha=efficiency)
    # plt.show()

    # 动量空间坐标网格定义
    data_NA = np.sin(np.deg2rad(40))  # 数值孔径
    data_k_max = 2 * PI * data_NA / wavelength
    n_k = efficiency.shape[0]
    data_kx = np.linspace(-data_k_max, data_k_max, n_k)
    data_ky = np.linspace(-data_k_max, data_k_max, n_k)

    # 初始化光调制器
    z_position = 0
    modulator = MomentumSpaceModulator(z_position, modulation_array=modulation_array, mod_kx=data_kx, mod_ky=data_ky)

    # 生成输入光场（高斯光束）
    sim_NA = 0.03
    w0 = wavelength/PI/sim_NA
    sim_size = 25.4/2*1e3*0.6
    x = np.linspace(-sim_size, sim_size, 1024*4)
    y = np.linspace(-sim_size, sim_size, 1024*4)
    U_in = generate_gaussian_beam(x, y, w0=w0)

    # 应用动量空间光调制器
    U_out = modulator.apply(cp.array(U_in), cp.array(x), cp.array(y), wavelength=1)

    # 显示结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Input Gaussian Beam (Amplitude)")
    plt.imshow(np.abs(U_in), extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Output Beam After Modulation (Amplitude)")
    plt.imshow(cp.abs(U_out).get(), extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
