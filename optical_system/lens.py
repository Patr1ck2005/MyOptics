import cupy as cp

from optical_system.elements import Lens, OpticalElement
from utils.constants import PI

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
        phase = k * (cp.sqrt(self.focal_length ** 2 + r_squared) - self.focal_length)
        modulation_factor = cp.exp(-1j * phase)

        if self.NA == 0:
            return U * modulation_factor

        # 数值孔径的限制
        max_angle = cp.arcsin(self.NA)  # 最大入射角，由NA确定
        max_radius = self.focal_length * cp.tan(max_angle)  # 焦平面中的最大半径
        NA_mask = cp.sqrt(r_squared) <= max_radius  # 应用NA的限制

        return U * modulation_factor * NA_mask

class Axicon(OpticalElement):
    def __init__(self, z_position, radius=cp.inf, apex_angle=None, base_angle=None, refractive_index=1.544):
        """
        初始化轴棱锥(axicon)光学元件。

        参数:
        z_position (float): 轴棱锥在z轴上的位置。
        radius (float): 轴棱锥的有效半径。
        apex_angle (float, optional): 轴棱锥的顶角大小（弧度）。
        base_angle (float, optional): 轴棱锥的底角大小（弧度）。
        refractive_index (float, optional): 轴棱锥材料的折射率。默认为1.0（真空或空气）。

        注意:
        需要提供顶角或底角中的一个。如果同时提供，则优先使用顶角。
        几何关系: apex_angle + 2 * base_angle = π（180度）
        """
        super().__init__(z_position)
        self.radius = radius

        # 验证折射率
        if refractive_index <= 0:
            raise ValueError("refractive_index must be a positive number.")
        self.refractive_index = refractive_index

        if apex_angle is not None and base_angle is not None:
            print("Warning: Both apex_angle and base_angle provided. apex_angle will be used.")

        if apex_angle is not None:
            if not (0 < apex_angle < PI):
                raise ValueError("apex_angle must be between 0 and π radians.")
            self.apex_angle = apex_angle
            self.base_angle = (PI - apex_angle) / 2
        elif base_angle is not None:
            if not (0 < base_angle < PI / 2):
                raise ValueError("base_angle must be between 0 and π/2 radians.")
            self.base_angle = base_angle
            self.apex_angle = PI - 2 * base_angle
        else:
            raise ValueError("Must provide either apex_angle or base_angle.")

    def apply(self, U, x, y, wavelength):
        """
        轴棱锥的相位调制。

        参数:
        U (ndarray): 输入光场。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        wavelength (float): 波长。

        返回:
        ndarray: 处理后的光场。
        """
        X, Y = cp.meshgrid(x, y)
        R = cp.sqrt(X ** 2 + Y ** 2)
        k = 2 * PI / wavelength
        # 计算相位延迟，h(r) = (n - 1) * r * tan(base_angle)
        phase_delay = -(self.refractive_index - 1) * R * cp.tan(self.base_angle) * k
        phase_factor = cp.exp(-1j * phase_delay)
        # 应用半径限制
        mask = R <= self.radius
        phase_factor = cp.where(mask, phase_factor, 1.0)
        return U * phase_factor

    @property
    def config(self):
        """
        获取轴棱锥的配置信息。

        返回:
        dict: 包含轴棱锥所有属性的字典。
        """
        config = super().config
        config.update({
            'radius': self.radius,
            'apex_angle': self.apex_angle,
            'base_angle': self.base_angle,
            'refractive_index': self.refractive_index
        })
        return config