# core/data_saving.py
import os

import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import numpy as np
from PIL import Image

class DataSaver(ABC):
    """
    抽象基类，用于定义数据保存的接口。
    """
    @abstractmethod
    def save(self, data: pd.DataFrame, output_path: Path):
        """
        保存数据到指定路径。
        """
        pass


class CSVSaver(DataSaver):
    """
    将数据保存为 CSV 文件的保存器。
    """
    def save(self, data: pd.DataFrame, output_path: Path):
        os.makedirs(output_path.parent, exist_ok=True)
        try:
            data_to_save = data.copy()
            data_to_save.index.name = 'row'
            data_to_save.columns.name = 'col'
            data_to_save.to_csv(output_path, index=True)
            logging.info(f"CSV 文件已保存至: {output_path}")
        except Exception as e:
            logging.error(f"保存 CSV 文件失败 ({output_path}): {e}")
            raise


class ImageSaver(DataSaver):
    """
    将数据保存为图像文件的保存器。
    """
    def save(self, data: pd.DataFrame, output_path: Path):
        try:
            img_array = (data.fillna(0) * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(output_path)
            logging.info(f"图像文件已保存至: {output_path}")
        except Exception as e:
            logging.error(f"保存图像文件失败 ({output_path}): {e}")
            raise
