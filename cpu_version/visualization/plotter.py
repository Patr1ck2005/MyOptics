# visualization/plotter.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm


class Plotter:
    def __init__(self, x, y):
        """
        初始化绘图器。

        参数:
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        """
        self.x = x
        self.y = y

    def plot_field(self, U, x_coords, y_coords, title="radical field"):
        """
        绘制光场的intensity和phase。

        参数:
        U (ndarray): 光场复数数组。
        x_coords (ndarray): x轴坐标。
        y_coords (ndarray): y轴坐标。
        title (str): 图标题。
        """
        intensity = np.abs(U) ** 2
        phase = np.angle(U)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

        im0 = axes[0].imshow(intensity, extent=extent, cmap='inferno', origin='lower')
        axes[0].set_title('intensity')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(phase, extent=extent, cmap='twilight', origin='lower')
        axes[1].set_title('phase')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1])

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_cross_sections(self, cross_sections):
        """
        绘制多个z位置的横截面光场，包括空间域和momentum space 。

        参数:
        cross_sections (dict): 包含z坐标为键，值为元组的字典。
                               - 如果 return_spectrum=False，元组为 (U, x, y)。
                               - 如果 return_spectrum=True，元组为 ((U, x, y), (U_k, kx, ky))。
        """
        num_sections = len(cross_sections)
        if num_sections == 0:
            print("没有横截面数据可绘制。")
            return

        # 计算总行数：每个z位置有2个（空间域）或4个（空间域+momentum space ）子图
        total_plots_per_section = 2  # 默认空间域的intensity和phase
        has_spectrum = any(len(value) == 2 for value in cross_sections.values())
        if has_spectrum:
            total_plots_per_section = 4  # 包含momentum space 的intensity和phase
        # 设置图像布局
        fig, axes = plt.subplots(4, num_sections, figsize=(6 * num_sections, 20))
        if num_sections == 1:
            axes = np.expand_dims(axes, axis=1)

        for i, (z, data) in enumerate(sorted(cross_sections.items())):
            # 绘制空间域
            U, x, y = data[0]
            intensity = np.abs(U) ** 2
            phase = np.angle(U)
            extent = [x.min(), x.max(), y.min(), y.max()]

            im0 = axes[0][i].imshow(intensity, extent=extent, cmap='inferno', origin='lower')
            axes[0][i].set_title(f'intensity at z = {z:.2f}')
            axes[0][i].set_xlabel('x')
            axes[0][i].set_ylabel('y')
            plt.colorbar(im0, ax=axes[0][i])

            im1 = axes[1][i].imshow(phase, extent=extent, cmap='twilight', origin='lower')
            axes[1][i].set_title(f'phase at z = {z:.2f}')
            axes[1][i].set_xlabel('x')
            axes[1][i].set_ylabel('y')
            plt.colorbar(im1, ax=axes[1][i])

            if data[1]:
                # 绘制momentum space 
                (U_k, kx, ky) = data[1]
                intensity_k = np.abs(U_k) ** 2
                phase_k = np.angle(U_k)
                extent_k = [kx.min(), kx.max(), ky.min(), ky.max()]

                im2 = axes[2][i].imshow(intensity_k, extent=extent_k, cmap='inferno', origin='lower')
                axes[2][i].set_title(f'momentum space intensity at z = {z:.2f}')
                axes[2][i].set_xlabel('$k_x$ (rad/μm)')
                axes[2][i].set_ylabel('$k_y$ (rad/μm)')
                plt.colorbar(im2, ax=axes[2][i])

                im3 = axes[3][i].imshow(phase_k, extent=extent_k, cmap='twilight', origin='lower')
                axes[3][i].set_title(f'momentum space phase at z = {z:.2f}')
                axes[3][i].set_xlabel('$k_x$ (rad/μm)')
                axes[3][i].set_ylabel('$k_y$ (rad/μm)')
                plt.colorbar(im3, ax=axes[3][i])
            else:
                # 如果没有momentum space 数据，隐藏额外的子图
                axes[2][i].axis('off')
                axes[3][i].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_longitudinal_section(self, coord_axis, z_coords, intensity, phase, direction='x', position=0.0):
        """
        绘制纵截面光场的intensity和phase。

        参数:
        coord_axis (ndarray): x或y坐标。
        z_coords (ndarray): z轴坐标。
        intensity (ndarray): 光场intensity二维数组，形状为 (len(coord_axis), len(z_coords))。
        phase (ndarray): 光场phase二维数组，形状为 (len(coord_axis), len(z_coords))。
        direction (str): 'x' 或 'y'，指定纵截面方向。
        position (float): 在指定方向上的固定位置。
        """
        if direction not in ['x', 'y']:
            raise ValueError("direction必须是 'x' 或 'y'")

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # intensity图
        im0 = axes[0].imshow(intensity, extent=[z_coords.min(), z_coords.max(), coord_axis.min(), coord_axis.max()],
                             aspect='auto', cmap='rainbow', origin='lower', interpolation=None, norm=SymLogNorm(linthresh=1/np.e, linscale=1))  # norm=LogNorm()
        axes[0].set_title(f'longitudinal intensity at {direction} = {position}')
        axes[0].set_xlabel('z')
        xlabel = 'y' if direction == 'x' else 'x'
        axes[0].set_ylabel(xlabel)
        plt.colorbar(im0, ax=axes[0])

        # phase图
        im1 = axes[1].imshow(phase, extent=[z_coords.min(), z_coords.max(), coord_axis.min(), coord_axis.max()],
                             aspect='auto', cmap='twilight', origin='lower', interpolation=None)
        axes[1].set_title(f'longitudinal phase at {direction} = {position}')
        axes[1].set_xlabel('z')
        axes[1].set_ylabel(xlabel)
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()
