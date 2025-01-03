# core/data_processing.py

import pandas as pd
import numpy as np
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
        save_path: Path = None,
        colormap = 'viridis',
    ) -> 'DataProcessor':
        """
        按指定形状（方形或圆形）裁剪数据，并可选择保存裁剪后的图像。

        :param center_row: 中心行坐标（相对比例或绝对像素）
        :param center_col: 中心列坐标（相对比例或绝对像素）
        :param radius: 裁剪外半径（相对比例或绝对像素）
        :param inner_radius: 裁剪内半径（相对比例或绝对像素）
        :param shape: 裁剪形状，'square' 或 'circle'
        :param relative: 是否使用相对坐标
        :param save_path: 可选的裁剪后图像保存路径
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

        if save_path:
            try:
                # 创建一个布尔掩码，True 表示数据有效，False 表示需要透明
                mask = self.data.notna().values.astype(np.uint8) * 255  # 255 表示不透明，0 表示透明

                # 将 DataFrame 转换为 NumPy 数组，替换 NaN 为 0
                cropped_array = self.data.fillna(0).values

                # 归一化数据到 0-1
                norm = Normalize(vmin=cropped_array.min(), vmax=cropped_array.max()/3)
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
                logging.info(f"裁剪后的图像已保存至: {save_path}")
            except Exception as e:
                logging.error(f"保存裁剪后的图像失败 ({save_path}): {e}")
                raise

        return self

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

