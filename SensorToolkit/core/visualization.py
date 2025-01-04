# core/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

class Visualizer:
    """
    负责数据可视化的类。
    """
    @staticmethod
    def visualize(
        data: pd.DataFrame,
        color_map: str = 'gray',
        title: str = "Intensity Distribution",
        save_path: Path = None
    ):
        """
        可视化数据并显示或保存图像。

        :param data: 需要可视化的数据
        :param color_map: 使用的颜色映射
        :param title: 图像标题
        :param save_path: 如果指定，将图像保存到该路径
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap=color_map)
        plt.colorbar(label='Intensity')
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logging.info(f"可视化图像已保存至: {save_path}")
        else:
            plt.show()
        plt.close()
