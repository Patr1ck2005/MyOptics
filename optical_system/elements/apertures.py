from optical_system.elements_cls import Aperture
import cupy as cp

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