import numpy as np

class GaussianBeam:
    """
    高斯光束生成器类，用于计算高斯光束的复振幅光场。

    参数：
    wavelength (float): 光波波长 (单位：与 x, y 坐标单位一致，例如米)
    waist_radius (float): 束腰半径 \( w_0 \) (单位：与 x, y 坐标单位一致，例如米)
    divergence_angle (float, optional): 发散角 \(\theta_0\) (单位：弧度，默认根据波长和束腰半径计算)
    """

    def __init__(self, wavelength, waist_radius, divergence_angle=None):
        self.wavelength = wavelength
        self.waist_radius = waist_radius
        self.divergence_angle = (
            divergence_angle if divergence_angle else wavelength / (np.pi * waist_radius)
        )
        self.z_rayleigh = np.pi * waist_radius**2 / wavelength  # 瑞利长度

    def compute_field(self, z_position, x, y):
        """
        计算指定传播距离 \( z \) 和网格上的高斯光束复振幅光场。

        参数：
        z_position (float): 偏离束腰处的传播距离 \( z \) (单位：与 x, y 坐标一致)
        x (np.ndarray): x 网格 (1D 数组)
        y (np.ndarray): y 网格 (1D 数组)

        返回：
        field (np.ndarray): 给定 \( x, y \) 网格上光场的复振幅数组
        """
        X, Y = np.meshgrid(x, y)
        R_squared = X**2 + Y**2  # 径向坐标平方

        # z 处的光斑半径
        beam_radius_z = self.waist_radius * np.sqrt(1 + (z_position / self.z_rayleigh)**2)

        # 曲率半径和 Gouy 相位
        if z_position != 0:
            curvature_radius = z_position * (1 + (self.z_rayleigh / z_position)**2)
        else:
            curvature_radius = np.inf  # 在束腰处，曲率无穷大
        gouy_phase = np.arctan(z_position / self.z_rayleigh)

        # 振幅因子和相位
        amplitude = np.exp(-R_squared / beam_radius_z**2)
        phase = (-np.pi * R_squared / (self.wavelength * curvature_radius) + gouy_phase)

        # 返回复振幅
        return amplitude * np.exp(1j * phase)

    def info(self):
        """
        打印高斯光束的主要参数。
        """
        print(f"Wavelength: {self.wavelength} m")
        print(f"Waist Radius: {self.waist_radius} m")
        print(f"Divergence Angle: {self.divergence_angle:.4f} rad")
        print(f"Rayleigh Length: {self.z_rayleigh:.4f} m")
