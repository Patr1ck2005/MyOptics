# optical_system/system.py

import numpy as np

from optical_system.elements import Lens, PhasePlate
from propagation.angular_spectrum import angular_spectrum_propagate
from propagation.angular_spectrum_longitudinal import angular_spectrum_propagate_longitudinal, calculate_field_on_grid

from utils.constants import PI


class OpticalSystem:
    def __init__(self, wavelength, x, y, initial_field):
        """
        初始化光学系统。

        参数:
        wavelength (float): 波长。
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        initial_field (ndarray): 初始光场。
        """
        self.wavelength = wavelength
        self.x = x
        self.y = y
        self.U = initial_field
        self.elements = []
        self.element_positions = []
        self.sorted = False

    def add_element(self, element):
        """
        添加光学元件到系统中。

        参数:
        element (OpticalElement): 光学元件实例。
        """
        self.elements.append(element)
        self.element_positions.append(element.z_position)
        self.sorted = False

    def sort_elements(self):
        """
        根据z_position对光学元件进行排序。
        """
        if not self.sorted:
            sorted_pairs = sorted(zip(self.element_positions, self.elements), key=lambda pair: pair[0])
            if sorted_pairs:
                self.element_positions, self.elements = zip(*sorted_pairs)
                self.element_positions = list(self.element_positions)
                self.elements = list(self.elements)
            else:
                self.element_positions = []
                self.elements = []
            self.sorted = True

    def propagate_to_cross_sections(self, z_positions, return_spectrum=False):
        """
        计算指定z位置的横截面光场。

        参数:
        z_positions (list or ndarray): 需要计算的z坐标列表。
        return_spectrum (bool): 是否返回动量空间光谱。默认值为 False。

        返回:
        dict: 包含z坐标为键，值为元组的字典。
              - 如果 return_spectrum=False，元组为 (U, x, y)。
              - 如果 return_spectrum=True，元组为 ((U, x, y), (U_k, kx, ky))。
        """
        self.sort_elements()
        results = {}
        current_U = self.U.copy()
        current_z = 0
        element_index = 0
        x = self.x
        y = self.y
        X, Y = np.meshgrid(x, y)
        wavelength = self.wavelength

        sorted_z = np.sort(z_positions).tolist()

        for z in sorted_z:
            # 传播到下一个光学元件或目标z位置
            while element_index < len(self.elements) and self.element_positions[element_index] <= z:
                z_prop = self.element_positions[element_index] - current_z
                if z_prop > 0:
                    current_U = angular_spectrum_propagate(current_U, x, y, z_prop, wavelength)
                # 应用光学元件
                current_U = self.elements[element_index].apply(current_U, x, y, wavelength)
                current_z = self.element_positions[element_index]
                element_index += 1

            # 传播到目标z位置
            z_prop = z - current_z
            if z_prop > 0:
                current_U = angular_spectrum_propagate(current_U, x, y, z_prop, wavelength)
                current_z = z

            if return_spectrum:
                # 计算动量空间光谱
                U_k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(current_U)))
                dkx = 2 * PI / (x[-1] - x[0])
                dky = 2 * PI / (y[-1] - y[0])
                kx = np.fft.fftshift(np.fft.fftfreq(len(x), d=(x[1] - x[0]))) * 2 * PI
                ky = np.fft.fftshift(np.fft.fftfreq(len(y), d=(y[1] - y[0]))) * 2 * PI
                # results[z] = (U_k.copy(), kx.copy(), ky.copy())
                results[z] = ((current_U.copy(), x.copy(), y.copy()), (U_k.copy(), kx.copy(), ky.copy()))
            else:
                results[z] = (current_U.copy(), x.copy(), y.copy()), ()

        return results
    def propagate_to_longitudinal_section(self, direction='x', position=0.0, num_z=500, z_max=100):
        """
        计算指定方向和位置的纵截面光场。

        参数:
        direction (str): 'x' 或 'y'，指定沿哪个轴进行纵截面。
        position (float): 在指定轴上的固定位置。
        num_z (int): z轴采样点数。
        z_max (float): 最大传播距离。

        返回:
        tuple: (coord_axis, z_coords, intensity, phase)
            coord_axis (ndarray): x或y坐标数组。
            z_coords (ndarray): z轴坐标数组。
            intensity (ndarray): 光场强度二维数组，形状为 (len(coord_axis), len(z_coords))。
            phase (ndarray): 光场相位二维数组，形状为 (len(coord_axis), len(z_coords))。
        """
        if direction not in ['x', 'y']:
            raise ValueError("direction必须是 'x' 或 'y'")

        self.sort_elements()

        # 定义纵截面坐标轴
        coord_axis = self.y if direction == 'x' else self.x
        # 定义z坐标
        z_coords = np.linspace(0, z_max, num_z)
        intensity = np.zeros((len(coord_axis), len(z_coords)))
        phase = np.zeros((len(coord_axis), len(z_coords)))

        # 初始化变量
        current_z = 0
        element_index = 0
        x, y = self.x, self.y
        wavelength = self.wavelength

        U_cross = self.U.copy()

        for i, z in enumerate(z_coords):
            z_prop = z - current_z

            # 检查是否有光学元件需要应用
            while element_index < len(self.elements) and self.element_positions[element_index] <= z:
                z_elem = self.element_positions[element_index]
                if z_elem > current_z and z_elem <= z:
                    # 传播到光学元件位置
                    z_prop_to_elem = z_elem - current_z
                    if z_prop_to_elem > 0:
                        U_cross = angular_spectrum_propagate(U_cross, x, y, z_prop, wavelength)
                        current_z = z_elem

                # 应用光学元件
                element = self.elements[element_index]
                U_cross = element.apply(U_cross, x, y, wavelength)
                # 处理下一个元件
                element_index += 1

            if z_prop > 0:
                # 使用横截面传播函数进行纵向传播
                U_cross = angular_spectrum_propagate(U_cross, x, y, z_prop, wavelength)
                current_z = z

            # 切片纵截面光场
            if direction == 'x':
                # y固定在position，提取U[:, y_idx]
                y_idx = np.argmin(np.abs(self.y - position))
                U_longitudinal_segment = U_cross[:, y_idx].copy()
            else:
                # x固定在position，提取U[x_idx, :]
                x_idx = np.argmin(np.abs(self.x - position))
                U_longitudinal_segment = U_cross[x_idx, :].copy()

            # 保存光场强度和相位
            intensity[:, i] = np.abs(U_longitudinal_segment) ** 2
            phase[:, i] = np.angle(U_longitudinal_segment)

        return coord_axis, z_coords, intensity, phase

    # def propagate_to_longitudinal_section(self, direction='x', position=0.0, num_z=500, z_max=100):
    #     """
    #     计算指定方向和位置的纵截面光场。
    #
    #     参数:
    #     direction (str): 'x' 或 'y'，指定沿哪个轴进行纵截面。
    #     position (float): 在指定轴上的固定位置。
    #     num_z (int): z轴采样点数。
    #     z_max (float): 最大传播距离。
    #
    #     返回:
    #     tuple: (coord_axis, z_coords, intensity, phase)
    #         coord_axis (ndarray): x或y坐标数组。
    #         z_coords (ndarray): z轴坐标数组。
    #         intensity (ndarray): 光场强度二维数组，形状为 (len(coord_axis), len(z_coords))。
    #         phase (ndarray): 光场相位二维数组，形状为 (len(coord_axis), len(z_coords))。
    #     """
    #     if direction not in ['x', 'y']:
    #         raise ValueError("direction必须是 'x' 或 'y'")
    #
    #     self.sort_elements()
    #
    #     # 定义纵截面坐标轴
    #     coord_axis = self.y if direction == 'x' else self.x
    #     # 定义z坐标
    #     z_coords = np.linspace(0, z_max, num_z)
    #     intensity = np.zeros((1, len(coord_axis)))
    #     phase = np.zeros((1, len(coord_axis)))
    #
    #     # 初始化变量
    #     current_z = 0
    #     element_index = 0
    #     wavelength = self.wavelength
    #     x = self.x
    #     y = self.y
    #     dx = x[1] - x[0]
    #     dy = y[1] - y[0]
    #
    #     # 初始化纵截面光场
    #     if direction == 'x':
    #         # y固定在position，提取U[:, y_idx]
    #         y_idx = np.argmin(np.abs(self.y - position))
    #         U_initial = self.U[:, y_idx].copy()
    #         coord_dir = self.x.copy()
    #     else:
    #         # x固定在position，提取U[x_idx, :]
    #         x_idx = np.argmin(np.abs(self.x - position))
    #         U_initial = self.U[x_idx, :].copy()
    #         coord_dir = self.y.copy()
    #
    #     # 将光学元件的位置添加到 z 分段中
    #     z_segments = [0] + self.element_positions + [z_max]
    #     z_segments = [z for z in z_segments if z <= z_max]
    #     num_segments = len(z_segments) - 1
    #
    #     # 初始化 z 位置索引
    #     z_indices = np.searchsorted(z_coords, z_segments)
    #
    #     # 复制初始光场
    #     U_cross = U_initial.copy()
    #
    #     # 遍历每个段
    #     for seg_idx in range(num_segments):
    #         z_start = z_segments[seg_idx]
    #         z_end = z_segments[seg_idx + 1]
    #         z_range = z_coords[z_indices[seg_idx]:z_indices[seg_idx + 1]]
    #
    #         # 应用当前区域的光学元件
    #         if element_index < len(self.elements) and self.element_positions[element_index] < z_end:
    #             # 应用光学元件
    #             element = self.elements[element_index]
    #             U_cross = element.apply(U_cross, x, y, wavelength)
    #             # 处理下一个元件
    #             element_index += 1
    #
    #         # 如果段内没有 z 坐标点，跳过
    #         if len(z_range) == 0:
    #             continue
    #
    #         # 计算在该段内的传播距离数组
    #         z_propagate = z_range
    #         # 在该段内逐步传播
    #         X, Z = np.meshgrid(x, z_propagate)
    #         Y = np.full_like(X, 0)
    #         U_longitudinal = calculate_field_on_grid(
    #             U_cross, x, y, dx, dy, X, Y, Z, wavelength)
    #
    #         # 存储当前区域的光场
    #         intensity = np.concatenate((intensity, np.abs(U_longitudinal) ** 2), axis=0)
    #         phase = np.concatenate((phase, np.angle(U_longitudinal)), axis=0)
    #
    #     return coord_dir, z_coords, intensity, phase