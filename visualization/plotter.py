# visualization/plotter.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm


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
        intensity, phase = np.abs(U) ** 2, np.angle(U)
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, data, cmap, label in zip(axes, [intensity, phase], ['inferno', 'twilight'], ['intensity', 'phase']):
            im = ax.imshow(data, extent=extent, cmap=cmap, origin='lower', interpolation='nearest')
            ax.set(title=label, xlabel='x', ylabel='y')
            plt.colorbar(im, ax=ax)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_cross_sections(self, cross_sections, save_label='default', show=False):
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

        total_plots_per_section = 4 if any(len(value) == 2 for value in cross_sections.values()) else 2
        fig, axes = plt.subplots(total_plots_per_section, num_sections, figsize=(4 * num_sections, 12))
        axes = np.atleast_2d(axes)  # 保证axes是2维，即使只有一个截面

        for i, (z, data) in enumerate(sorted(cross_sections.items())):
            U, x, y = data[0]
            intensity, phase = np.abs(U) ** 2, np.angle(U)
            extent = [x.min(), x.max(), y.min(), y.max()]

            im0 = axes[0][i].imshow(intensity, extent=extent, cmap='inferno', origin='lower', interpolation='nearest')
            axes[0][i].set(title=f'intensity at z = {z:.2f}', xlabel='x', ylabel='y')
            plt.colorbar(im0, ax=axes[0][i])

            im1 = axes[1][i].imshow(phase, extent=extent, cmap='twilight', origin='lower', interpolation='nearest')
            axes[1][i].set(title=f'phase at z = {z:.2f}', xlabel='x', ylabel='y')
            plt.colorbar(im1, ax=axes[1][i])

            if len(data) == 2:
                U_k, kx, ky = data[1]
                intensity_k, phase_k = np.abs(U_k) ** 2, np.angle(U_k)
                extent_k = [kx.min(), kx.max(), ky.min(), ky.max()]

                im2 = axes[2][i].imshow(intensity_k, extent=extent_k, cmap='inferno', origin='lower',
                                        interpolation='nearest')
                axes[2][i].set(title=f'momentum space intensity at z = {z:.2f}', xlabel='$k_x$ (rad/μm)',
                               ylabel='$k_y$ (rad/μm)')
                plt.colorbar(im2, ax=axes[2][i])

                im3 = axes[3][i].imshow(phase_k, extent=extent_k, cmap='twilight', origin='lower',
                                        interpolation='nearest')
                axes[3][i].set(title=f'momentum space phase at z = {z:.2f}', xlabel='$k_x$ (rad/μm)',
                               ylabel='$k_y$ (rad/μm)')
                plt.colorbar(im3, ax=axes[3][i])

        plt.tight_layout()
        plt.savefig(f'{save_label}-cross_sections.png', dpi=1000)
        if show:
            plt.show()
        plt.close(fig)

    def plot_longitudinal_section(self, coord_axis, z_coords, intensity, phase, direction='x', position=0.0,
                                  save_label='default'):
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
        xlabel = 'y' if direction == 'x' else 'x'

        im0 = axes[0].imshow(intensity, extent=[z_coords.min(), z_coords.max(), coord_axis.min(), coord_axis.max()],
                             aspect='auto', cmap='rainbow', origin='lower', interpolation='nearest',
                             norm=SymLogNorm(linthresh=1 / np.e, linscale=1))
        axes[0].set(title=f'longitudinal intensity at {direction} = {position}', xlabel='z', ylabel=xlabel)
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(phase, extent=[z_coords.min(), z_coords.max(), coord_axis.min(), coord_axis.max()],
                             aspect='auto', cmap='twilight', origin='lower', interpolation='nearest')
        axes[1].set(title=f'longitudinal phase at {direction} = {position}', xlabel='z', ylabel=xlabel)
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(f'{save_label}-longitudinal_section.png', dpi=1000)
        plt.show()
        plt.close(fig)