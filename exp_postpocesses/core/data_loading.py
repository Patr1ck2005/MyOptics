# core/data_loading.py

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path
import logging

class DataLoader(ABC):
    """
    抽象基类，用于定义数据加载的接口。
    """
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        加载数据并返回为 Pandas DataFrame。
        """
        pass


class CSVDataLoader(DataLoader):
    """
    用于加载 CSV 文件的数据加载器。
    """
    def __init__(self, file_path: Path, encoding: str = 'latin1'):
        if not file_path.is_file():
            raise FileNotFoundError(f"CSV 文件未找到: {file_path}")
        self.file_path = file_path
        self.encoding = encoding

    def load_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.file_path, encoding=self.encoding, index_col=0)
            logging.info(f"成功加载 CSV 文件: {self.file_path}，数据形状: {data.shape}")
            return data
        except Exception as e:
            logging.error(f"加载 CSV 文件失败 ({self.file_path}): {e}")
            raise


class ImageDataLoader(DataLoader):
    """
    用于加载图像文件的数据加载器，支持多种格式。
    """
    SUPPORTED_FORMATS = {'.png', '.bmp', '.jpg', '.jpeg', '.tiff'}

    def __init__(self, file_path: Path):
        if not file_path.is_file():
            raise FileNotFoundError(f"图像文件未找到: {file_path}")
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的图像格式: {file_path.suffix}")
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        try:
            with Image.open(self.file_path) as img:
                img = img.convert('L')  # 转换为灰度图像
                img_array = np.array(img)
                df = pd.DataFrame(img_array)
                df.index.name = 'row'
                df.columns.name = 'col'
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'row'}, inplace=True)
                logging.info(f"成功加载图像文件: {self.file_path}，数据形状: {df.shape}")
                return df
        except Exception as e:
            logging.error(f"加载图像文件失败 ({self.file_path}): {e}")
            raise
