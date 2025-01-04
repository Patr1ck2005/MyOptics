# core/data_processing.py

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, laplace
from skimage.filters import sobel
from scipy.fft import fft2, ifft2, fftshift, ifftshift

import logging
from pathlib import Path
from PIL import Image
from matplotlib import cm
from matplotlib.colors import Normalize


class DataProcessor:
    """
    负责处理数据的类，包括裁剪、坐标重置和计算平均光强。
    """
    def __init__(self, data: pd.DataFrame, pixel_size_um: float = 5.0):
        self.pixel_size_um = pixel_size_um
        self._prepare_data(data)

    def _prepare_data(self, data: pd.DataFrame):
        """
        预处理数据，包括去除行列索引并归一化。
        """
        if 'row' in data.columns and 'col' in data.index.names:
            data = data.drop(['row'], axis=1)
            data.index = data['row']
            data = data.drop(['row'], axis=1)
        self.data = data.astype(float) / 255.0  # 归一化到0-1
        logging.info(f"数据预处理完成，数据形状: {self.data.shape}")

    def crop_by_shape(
        self,
        center_row: float,
        center_col: float,
        radius: float,
        inner_radius: float = 0,
        shape: str = 'circle',
        relative: bool = False,
    ) -> 'DataProcessor':
        """
        按指定形状（方形或圆形）裁剪数据，并可选择保存裁剪后的图像。

        :param center_row: 中心行坐标（相对比例或绝对像素）
        :param center_col: 中心列坐标（相对比例或绝对像素）
        :param radius: 裁剪外半径（相对比例或绝对像素）
        :param inner_radius: 裁剪内半径（相对比例或绝对像素）
        :param shape: 裁剪形状，'square' 或 'circle'
        :param relative: 是否使用相对坐标
        :return: 返回自身以支持链式调用
        """
        if shape not in {'square', 'circle'}:
            raise ValueError("形状必须为 'square' 或 'circle'。")

        total_rows, total_cols = self.data.shape

        if relative:
            if not (0 <= center_row <= 1) or not (0 <= center_col <= 1):
                raise ValueError("当 relative=True 时，center_row 和 center_col 必须在 0 到 1 之间。")
            if not (0 <= radius <= 1):
                raise ValueError("当 relative=True 时，radius 必须在 0 到 1 之间。")
            center_row = int(total_rows * center_row)
            center_col = int(total_cols * center_col)
            radius = int(min(total_rows, total_cols) * radius)
            inner_radius = int(min(total_rows, total_cols) * inner_radius)
            logging.info("使用相对坐标进行裁剪。")
        else:
            if not (0 <= center_row < total_rows) or not (0 <= center_col < total_cols):
                raise ValueError("center_row 和 center_col 必须在数据范围内。")
            if radius <= 0:
                raise ValueError("radius 必须是正数。")

        row_start = max(center_row - radius, 0)
        row_end = min(center_row + radius, total_rows)
        col_start = max(center_col - radius, 0)
        col_end = min(center_col + radius, total_cols)

        logging.info(
            f"裁剪参数: center=({center_row}, {center_col}), radius={radius}, shape={shape}，裁剪区域: rows({row_start}-{row_end}), cols({col_start}-{col_end})"
        )

        cropped_data = self.data.iloc[row_start:row_end, col_start:col_end].copy()

        if shape == 'circle':
            num_rows, num_cols = cropped_data.shape
            y, x = np.ogrid[:num_rows, :num_cols]
            center_y = center_row - row_start
            center_x = center_col - col_start
            r2 = (x - center_x) ** 2 + (y - center_y) ** 2
            mask = r2 < inner_radius ** 2
            cropped_data = cropped_data.mask(mask)
            mask = r2 > radius ** 2
            cropped_data = cropped_data.mask(mask)
            logging.info("应用圆形遮罩，圆形外区域设置为 NaN。")

        self.data = cropped_data
        logging.info(f"裁剪完成，数据形状: {self.data.shape}")

        return self

    def apply_image_filter(self, filter_type='gaussian', **kwargs) -> 'DataProcessor':
        """
        对裁剪后的数据应用图像滤波器。

        :param filter_type: 滤波器类型，支持 'gaussian'（高斯滤波）、'median'（中值滤波）、'mean'（均值滤波）、
                            'sobel'（边缘检测）、'laplace'（拉普拉斯滤波）、'lowpass'（频域低通滤波）、'highpass'（频域高通滤波）。
        :param kwargs: 滤波器的参数。
        :return: 返回自身以支持链式调用。
        """
        if self.data is None:
            raise ValueError("数据为空，无法应用滤波器。")

        # 将数据转换为 NumPy 数组
        data_array = self.data.values

        if filter_type == 'gaussian':
            sigma = kwargs.get('sigma', 1)  # 高斯滤波的标准差
            filtered_data = gaussian_filter(data_array, sigma=sigma)
            logging.info(f"高斯滤波已应用，sigma={sigma}")

        elif filter_type == 'median':
            size = kwargs.get('size', 3)  # 中值滤波的窗口大小
            filtered_data = median_filter(data_array, size=size)
            logging.info(f"中值滤波已应用，窗口大小={size}")

        elif filter_type == 'mean':
            size = kwargs.get('size', 3)  # 均值滤波的窗口大小
            filtered_data = uniform_filter(data_array, size=size)
            logging.info(f"均值滤波已应用，窗口大小={size}")

        elif filter_type == 'sobel':
            filtered_data = sobel(data_array)
            logging.info("Sobel 边缘检测已应用。")

        elif filter_type == 'laplace':
            filtered_data = laplace(data_array)
            logging.info("拉普拉斯滤波已应用。")

        elif filter_type == 'lowpass':
            cutoff = kwargs.get('cutoff', 30)  # 低通滤波的截止频率
            filtered_data = self._apply_frequency_filter(data_array, filter_type='lowpass', cutoff=cutoff)
            logging.info(f"频域低通滤波已应用，截止频率={cutoff}")

        elif filter_type == 'highpass':
            cutoff = kwargs.get('cutoff', 30)  # 高通滤波的截止频率
            filtered_data = self._apply_frequency_filter(data_array, filter_type='highpass', cutoff=cutoff)
            logging.info(f"频域高通滤波已应用，截止频率={cutoff}")

        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")

        # 更新数据
        self.data = pd.DataFrame(filtered_data, index=self.data.index, columns=self.data.columns)
        logging.info("滤波完成，数据已更新。")

        return self

    def _apply_frequency_filter(self, image, filter_type='lowpass', cutoff=30):
        """
        在频域中应用低通或高通滤波器。

        :param image: 输入图像数据。
        :param filter_type: 滤波器类型，'lowpass' 或 'highpass'。
        :param cutoff: 截止频率。
        :return: 滤波后的图像数据。
        """
        f = fft2(image)
        fshift = fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # 创建滤波器
        mask = np.zeros((rows, cols), dtype=np.float32)
        if filter_type == 'lowpass':
            mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
        elif filter_type == 'highpass':
            mask[:, :] = 1
            mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0
        else:
            raise ValueError("filter_type 必须是 'lowpass' 或 'highpass'")

        # 应用滤波器
        fshift_filtered = fshift * mask
        f_ishift = ifftshift(fshift_filtered)
        filtered_image = np.abs(ifft2(f_ishift))

        return filtered_image

    def save_processed_image(self, save_path: Path, colormap='viridis'):
        """
        保存裁剪后的图像。

        :param save_path: 保存图像的路径。
        :param colormap: 保存图片的重颜色映射
        :return: 返回自身以支持链式调用。
        """
        if self.data is None:
            raise ValueError("数据为空，无法保存图像。")

        try:
            # 创建一个布尔掩码，True 表示数据有效，False 表示需要透明
            mask = self.data.notna().values.astype(np.uint8) * 255  # 255 表示不透明，0 表示透明

            # 将 DataFrame 转换为 NumPy 数组，替换 NaN 为 0
            cropped_array = self.data.fillna(0).values

            # 归一化数据到 0-1
            norm = Normalize(vmin=cropped_array.min(), vmax=cropped_array.max())
            normalized_data = norm(cropped_array)

            # 获取颜色映射函数
            if isinstance(colormap, str):
                cmap = cm.get_cmap(colormap)
            elif callable(colormap):
                cmap = colormap
            else:
                raise ValueError("colormap 必须是有效的 matplotlib colormap 名称或可调用的映射函数")

            # 使用颜色映射生成 RGBA 数据
            rgba_data = (cmap(normalized_data) * 255).astype(np.uint8)

            # 替换 Alpha 通道为自定义的透明度（基于 mask）
            rgba_data[..., 3] = mask

            # 创建 RGBA 图像
            img = Image.fromarray(rgba_data, mode='RGBA')

            # 确保保存路径的父目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存图像为 PNG 格式
            img.save(save_path, format='PNG')
            logging.info(f"当前的图像已保存至: {save_path}")
        except Exception as e:
            logging.error(f"保存当前的图像失败 ({save_path}): {e}")
            raise

    def reset_coordinates(self) -> 'DataProcessor':
        """
        重置数据的行列坐标。
        :return: 返回自身以支持链式调用
        """
        self.data.reset_index(drop=True, inplace=True)
        logging.info("坐标已重置。")
        return self

    def calculate_average_intensity(self) -> float:
        """
        计算数据的平均光强。
        :return: 平均光强值
        """
        avg_intensity = self.data.mean().mean()
        logging.info(f"平均光强计算完成: {avg_intensity}")
        return avg_intensity

    def rescale(self) -> 'DataProcessor':
        """
        将数据重新缩放到 0-1 范围。
        :return: 返回自身以支持链式调用
        """
        min_val = self.data.min().min()
        max_val = self.data.max().max()
        if max_val - min_val == 0:
            raise ValueError("数据的最大值和最小值相同，无法归一化。")
        self.data = (self.data - min_val) / (max_val - min_val)
        logging.info("数据已归一化。")
        return self

