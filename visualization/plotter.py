# visualization/plotter.py
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Plotter:
    def __init__(self, x, y, max_dpi=1024, wavelength=1.550, output_dir='./img'):
        """
        初始化绘图器。

        参数:
        x (ndarray): x轴坐标。
        y (ndarray): y轴坐标。
        max_dpi (int): 保存图像的最大 DPI。
        wavelength (float): 波长（仅用于动量空间角度刻度换算，需与仿真一致）。
        output_dir (str): 图像输出目录（相对当前工作目录或绝对路径）。
        """
        self.x = x
        self.y = y
        self.max_dpi = max_dpi
        self.wavelength = wavelength
        self.output_dir = output_dir
        logging.info("Plotter initialized with max_dpi=%d", max_dpi)
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_dynamic_dpi(self, data_shape, fig_size, sub_fig_num=16, up_sampling=1):
        """
        动态计算保存图像的 DPI，确保所有数据点都被清晰展示。

        参数:
        data_shape (tuple): 数据的形状 (高度, 宽度)。
        fig_size (tuple): 图像的尺寸 (宽度英寸, 高度英寸)。
        max_dpi (int): 最大 DPI 值。

        返回:
        int: 计算得到的 DPI。
        """
        height, width = data_shape
        fig_width, fig_height = fig_size
        dpi_width = width / fig_width
        dpi_height = height / fig_height
        dpi = int(max(dpi_width, dpi_height))
        dpi *= np.sqrt(sub_fig_num)
        dpi *= up_sampling
        if dpi >= self.max_dpi:
            logging.info(f"DPI {dpi} reaching limit {self.max_dpi}")
            dpi = self.max_dpi
        logging.info("Calculated dynamic DPI: %d", dpi)
        return int(dpi)

    def plot_field(self, U, x_coords, y_coords, title="radical field", show=False):
        """
        绘制光场的intensity和phase。

        参数:
        U (ndarray): 光场复数数组。
        x_coords (ndarray): x轴坐标。
        y_coords (ndarray): y轴坐标。
        title (str): 图标题。
        """
        logging.info("Starting plot_field with title: %s", title)
        intensity, phase = np.abs(U) ** 2, np.angle(U)
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, data, cmap, label in zip(axes, [intensity, phase], ['rainbow', 'twilight'], ['intensity', 'phase']):
            im = ax.imshow(data, extent=extent, cmap=cmap, origin='lower', interpolation='nearest')
            ax.set(title=label, xlabel='x', ylabel='y')
            plt.colorbar(im, ax=ax)
        logging.info("Field plot completed")

        plt.suptitle(title)
        plt.tight_layout()
        # 先保存后显示：show 会阻塞并在窗口关闭后清理 figure，先 savefig 可避免存出空白图
        plt.savefig(os.path.join(self.output_dir, f'{title.replace(" ", "_")}.png'), dpi=self.max_dpi)
        if show:
            plt.show()
        plt.close(fig)

    def plot_cross_sections(
            self,
            cross_sections,
            save_label='default',
            show=False,
            cmap_for_spatial_intensity='rainbow',
            # cmap_for_spatial_phase='twilight',
            vmax_for_spatial_intensity=None,
    ):
        """
        绘制多个z位置的横截面光场，包括空间域和momentum space 。

        参数:
        cross_sections (dict): 包含z坐标为键，值为元组的字典。
                               - 如果 return_spectrum=False，元组为 (U, x, y)。
                               - 如果 return_spectrum=True，元组为 ((U, x, y), (U_k, kx, ky))。
        save_label (str): 保存图像的标签前缀。
        show (bool): 是否显示图像。
        cmap_for_intensity (str): 空间域光场的颜色映射。
        cmap_for_phase (str): 空间域相位图的颜色映射。
        vmax_for_intensity (Optional[float]): 空间域光场的颜色映射最大值。如果为 None，则自适应。
        """
        logging.info("Starting plot_cross_sections with %d sections", len(cross_sections))
        num_sections = len(cross_sections)
        if num_sections == 0:
            logging.warning("No cross-section data available to plot.")
            return

        total_plots_per_section = 4 if any(len(value) == 2 for value in cross_sections.values()) else 2
        fig, axes = plt.subplots(total_plots_per_section, num_sections, figsize=(4 * num_sections, 12))
        # 统一为 (行, 列) 二维索引：单截面时 subplots 返回一维数组，atleast_2d 会错误地变成 (1, N)
        axes = np.asarray(axes).reshape(total_plots_per_section, -1)


        for i, (z, data) in enumerate(sorted(cross_sections.items())):
            U, x, y = data[0]
            logging.info("Plotting cross-section at z=%.2f", z)
            intensity, phase = np.abs(U) ** 2, np.angle(U)
            extent = [x.min(), x.max(), y.min(), y.max()]

            im0 = axes[0][i].imshow(intensity,
                                    extent=extent,
                                    cmap=cmap_for_spatial_intensity,
                                    origin='lower',
                                    interpolation='nearest',
                                    vmax=vmax_for_spatial_intensity)  # 自适应或手动颜色映射范围
            axes[0][i].set(title=f'intensity at z = {z:.2f}', xlabel='x', ylabel='y')
            plt.colorbar(im0, ax=axes[0][i])

            im1 = axes[1][i].imshow(phase, extent=extent, cmap='twilight', origin='lower', interpolation='nearest')
            axes[1][i].set(title=f'phase at z = {z:.2f}', xlabel='x', ylabel='y')
            plt.colorbar(im1, ax=axes[1][i])

            if len(data) == 2:
                U_k, kx, ky = data[1]
                logging.info("Including momentum space data for z=%.2f", z)
                intensity_k, phase_k = np.abs(U_k) ** 2, np.angle(U_k)
                extent_k = [kx.min(), kx.max(), ky.min(), ky.max()]
                yticks = np.linspace(ky.min(), ky.max(), 10)
                # 角度换算 sinθ = ky/k0；|ky| 超出光锥（倏逝区）无实角度，标签置空
                k0 = 2 * np.pi / self.wavelength
                sin_theta = np.clip(yticks / k0, -1.0, 1.0)
                angles = np.degrees(np.arcsin(sin_theta))
                angles[~(np.abs(yticks) <= k0)] = np.nan  # 倏逝区刻度留空

                im2 = axes[2][i].imshow(intensity_k, extent=extent_k, cmap='rainbow',
                                        origin='lower', interpolation='nearest')
                axes[2][i].set(title=f'momentum space intensity at z = {z:.2f}',
                               xlabel='$k_x$ (rad/μm)', ylabel=r'angle (deg)')
                # 设置 y 轴的刻度和标签
                axes[2][i].set_yticks(yticks)  # 设置刻度位置
                axes[2][i].set_yticklabels([f'{a:.1f}' if np.isfinite(a) else '' for a in angles])  # 设置刻度标签
                plt.colorbar(im2, ax=axes[2][i])

                im3 = axes[3][i].imshow(phase_k, extent=extent_k, cmap='twilight',
                                        origin='lower', interpolation='nearest')
                axes[3][i].set(title=f'momentum space phase at z = {z:.2f}',
                               xlabel='$k_x$ (rad/μm)', ylabel=r'angle (deg)')
                # 设置 y 轴的刻度和标签
                axes[3][i].set_yticks(yticks)  # 设置刻度位置
                axes[3][i].set_yticklabels([f'{a:.1f}' if np.isfinite(a) else '' for a in angles])  # 设置刻度标签
                plt.colorbar(im3, ax=axes[3][i])

        plt.tight_layout()
        # 动态计算整体图像的 DPI
        # 以第一个截面的数据形状作为示例计算 DPI
        first_z, first_data = sorted(cross_sections.items())[0]
        first_U, first_x, first_y = first_data[0]
        fig_dpi = self.calculate_dynamic_dpi(first_U.shape, (4 * num_sections, 12))
        plt.savefig(os.path.join(self.output_dir, f'{save_label}-cross_sections.png'), dpi=fig_dpi)
        if show:
            plt.show()
        plt.close(fig)
        logging.info("Cross-section plots saved as %s-cross_sections.png\n", save_label)

    def plot_longitudinal_section(
            self,
            coord_axis: np.ndarray,
            z_coords: np.ndarray,
            intensity: np.ndarray,
            phase: np.ndarray,
            direction: str = 'x',
            position: float = 0.0,
            save_label: str = 'default',
            show: bool = False,
            norm_vmin: float | None = None,
            ref_position_min: float | None = None,  # 归一化参考位置，取值范围 [0, 1]，用于动态确定 norm_vmin
            ref_multiplier_min: float | None = 1.0,  # 用于 norm_vmin 的倍数
            norm_vmax: float | None = None,
            ref_position_max: float | None = None,  # 归一化参考位置，取值范围 [0, 1]，用于动态确定 norm_vmax
            ref_multiplier_max: float | None = 1.0,  # 用于 norm_vmax 的倍数
            figsize: tuple[int, int] | None = None,
            dpi: int | None = None
    ):
        """
        绘制纵截面光场的 intensity 和 phase。

        参数:
        coord_axis (ndarray): x或y坐标。
        z_coords (ndarray): z轴坐标。
        intensity (ndarray): 光场 intensity 二维数组，形状为 (len(coord_axis), len(z_coords))。
        phase (ndarray): 光场 phase 二维数组，形状为 (len(coord_axis), len(z_coords))。
        direction (str): 'x' 或 'y'，指定纵截面方向。
        position (float): 在指定方向上的固定位置。
        save_label (str): 保存图像的标签前缀。
        show (bool): 是否显示图像。
        norm_vmin (float, optional): 颜色映射的最小值。如果提供了 ref_position_min，将忽略此参数。
        ref_position_min (float, optional): 归一化参考位置，取值范围 [0, 1]，用于动态确定 norm_vmin。
        ref_multiplier_min (float, optional): 用于 norm_vmin 的倍数。
        norm_vmax (float, optional): 颜色映射的最大值。如果提供了 ref_position_max，将忽略此参数。
        ref_position_max (float, optional): 归一化参考位置，取值范围 [0, 1]，用于动态确定 norm_vmax。
        ref_multiplier_max (float, optional): 用于 norm_vmax 的倍数。
        figsize (tuple, optional): 图像的尺寸。
        dpi (int, optional): 图像的分辨率。
        """
        logging.info("Starting plot_longitudinal_section at %s = %.2f", direction, position)

        if direction not in ['x', 'y']:
            logging.error("Invalid direction: %s. Must be 'x' or 'y'", direction)
            raise ValueError("direction 必须是 'x' 或 'y'")

        if figsize is None:
            figsize = (10, 8)

        # 动态确定 norm_vmin
        if ref_position_min is not None:
            if not (0.0 <= ref_position_min <= 1.0):
                logging.error("ref_position_min %.2f is out of bounds [0.0, 1.0]", ref_position_min)
                raise ValueError("ref_position_min 必须在 [0.0, 1.0] 范围内")

            # 计算参考位置对应的索引
            ref_idx_min = int(ref_position_min * (len(coord_axis) - 1))
            ref_intensity_min = intensity[ref_idx_min, :]
            max_ref_intensity_min = np.max(ref_intensity_min)

            # 设置 norm_vmin 基于参考位置的强度
            norm_vmin_dynamic = (max_ref_intensity_min / np.e ** 2) * ref_multiplier_min
            logging.info(
                "Reference position min (normalized): %.2f, max intensity at ref: %.3f, norm_vmin=%.3f",
                ref_position_min, max_ref_intensity_min, norm_vmin_dynamic)
            norm_vmin = norm_vmin_dynamic
        else:
            # 使用用户提供的 norm_vmin 或默认值
            if norm_vmin is None:
                norm_vmin = 1 / np.e ** 2
            logging.info("Using provided norm_vmin: %.3f", norm_vmin)

        # 动态确定 norm_vmax
        if ref_position_max is not None:
            if not (0.0 <= ref_position_max <= 1.0):
                logging.error("ref_position_max %.2f is out of bounds [0.0, 1.0]", ref_position_max)
                raise ValueError("ref_position_max 必须在 [0.0, 1.0] 范围内")

            # 计算参考位置对应的索引
            ref_idx_max = int(ref_position_max * (len(coord_axis) - 1))
            ref_intensity_max = intensity[ref_idx_max, :]
            max_ref_intensity_max = np.max(ref_intensity_max)

            # 设置 norm_vmax 基于参考位置的强度
            norm_vmax_dynamic = (max_ref_intensity_max) * ref_multiplier_max
            logging.info(
                "Reference position max (normalized): %.2f, max intensity at ref: %.3f, norm_vmax=%.3f",
                ref_position_max, max_ref_intensity_max, norm_vmax_dynamic)
            norm_vmax = norm_vmax_dynamic
        else:
            # 使用用户提供的 norm_vmax 或默认值
            if norm_vmax is None:
                norm_vmax = np.max(intensity)
            logging.info("Using provided norm_vmax: %.3f", norm_vmax)

        fig, axes = plt.subplots(2, 1, figsize=figsize)
        xlabel = 'y' if direction == 'x' else 'x'

        # 获取并设置 colormap 的 'under' 颜色（LogNorm 低于 vmin 的区域显示为黑色）
        cmap = plt.colormaps['rainbow'].copy()  # 复制以避免修改全局 colormap
        cmap.set_under('black')  # 设置 'under' 颜色

        # 确保 norm_vmin < norm_vmax（LogNorm 要求 vmin > 0）
        if not (0 < norm_vmin < norm_vmax):
            logging.error("Invalid LogNorm range (vmin=%.3g, vmax=%.3g)，回退为线性显示", norm_vmin, norm_vmax)
            norm = None
        else:
            norm = LogNorm(vmin=norm_vmin, vmax=norm_vmax)

        im0 = axes[0].imshow(
            intensity,
            extent=[z_coords.min(), z_coords.max(), coord_axis.min(), coord_axis.max()],
            aspect='auto',
            cmap=cmap,
            origin='lower',
            interpolation='nearest',
            norm=norm,  # 对数显示：低于 1/e² (或参考阈值) 的区域压暗
        )
        axes[0].set(title=f'Longitudinal Intensity at {direction} = {position}', xlabel='z', ylabel=xlabel)
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(
            phase,
            extent=[z_coords.min(), z_coords.max(), coord_axis.min(), coord_axis.max()],
            aspect='auto',
            cmap='twilight',
            origin='lower',
            interpolation='nearest'
        )
        axes[1].set(title=f'Longitudinal Phase at {direction} = {position}', xlabel='z', ylabel=xlabel)
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        save_dpi = dpi if dpi is not None else self.calculate_dynamic_dpi(intensity.shape, figsize)
        save_path = os.path.join(self.output_dir, f'{save_label}-longitudinal_section.png')
        plt.savefig(save_path, dpi=save_dpi)
        if show:
            plt.show()
        plt.close(fig)
        logging.info("Longitudinal section plot saved as %s\n", save_path)
