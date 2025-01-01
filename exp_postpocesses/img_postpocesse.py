import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod
from PIL import Image  # 用于图像处理
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass


class CSVDataLoader(DataLoader):
    def __init__(self, file_path, encoding='latin1'):
        self.file_path = file_path
        self.encoding = encoding

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path, encoding=self.encoding, index_col=0)
            logging.info(f"CSV数据成功加载！数据形状：{data.shape}")
            return data
        except Exception as e:
            logging.error(f"加载CSV数据失败: {e}")
            raise


class ImageDataLoader(DataLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            with Image.open(self.file_path) as img:
                img = img.convert('L')  # 转换为灰度图像
                img_array = np.array(img)
                # 创建DataFrame，添加行和列坐标
                df = pd.DataFrame(img_array)
                df.index.name = 'row'
                df.columns.name = 'col'
                # 添加坐标列
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'row'}, inplace=True)
                logging.info(f"图像数据成功加载！数据形状：{df.shape}")
                return df
        except Exception as e:
            logging.error(f"加载图像数据失败: {e}")
            raise


class DataProcessor:
    def __init__(self, data, pixel_size_um=5):
        # 如果DataFrame有行和列索引，则移除
        if 'row' in data.columns and 'col' in data.index.names:
            data = data.drop(['row'], axis=1)
            data.index = data['row']
            data = data.drop(['row'], axis=1)
        self.data = data.astype(float) / 255  # 归一化到0-1
        self.pixel_size_um = pixel_size_um

    def crop(self, row_start, row_end, col_start, col_end):
        """
        按像素索引裁剪数据
        """
        self.data = self.data.iloc[row_start:row_end, col_start:col_end]
        logging.info(f"已裁剪数据：新数据形状 {self.data.shape}")
        return self

    def crop_by_ratio(self, row_start_ratio, row_end_ratio, col_start_ratio, col_end_ratio):
        """
        按比例裁剪数据
        """
        total_rows, total_cols = self.data.shape
        row_start = int(total_rows * row_start_ratio)
        row_end = int(total_rows * row_end_ratio)
        col_start = int(total_cols * col_start_ratio)
        col_end = int(total_cols * col_end_ratio)
        logging.info(
            f"按比例裁剪参数: rows({row_start_ratio}-{row_end_ratio}) -> ({row_start}-{row_end}), "
            f"cols({col_start_ratio}-{col_end_ratio}) -> ({col_start}-{col_end})")
        return self.crop(row_start, row_end, col_start, col_end)

    def crop_by_shape(self, center_row, center_col, radius, shape='square', relative=False):
        """
        按指定形状（方形或圆形）裁剪数据

        :param center_row: 中心行坐标（绝对像素或相对比例）
        :param center_col: 中心列坐标（绝对像素或相对比例）
        :param radius: 裁剪半径（绝对像素或相对比例）
        :param shape: 裁剪形状，'square' 或 'circle'
        :param relative: 布尔值，指示是否使用相对坐标
        """
        if shape not in ['square', 'circle']:
            raise ValueError("形状必须是 'square' 或 'circle'。")

        total_rows, total_cols = self.data.shape

        if relative:
            if not (0 <= center_row <= 1) or not (0 <= center_col <= 1):
                raise ValueError("当 relative=True 时，center_row 和 center_col 必须在 0 到 1 之间。")
            if not (0 <= radius <= 1):
                raise ValueError("当 relative=True 时，radius 必须在 0 到 1 之间。")
            center_row = int(total_rows * center_row)
            center_col = int(total_cols * center_col)
            # 将 radius 定义为相对于最小维度的比例
            radius = int(min(total_rows, total_cols) * radius)
            logging.info("使用相对坐标进行裁剪。")
        else:
            if not (0 <= center_row < total_rows) or not (0 <= center_col < total_cols):
                raise ValueError("center_row 和 center_col 必须在数据范围内。")
            if radius <= 0:
                raise ValueError("radius 必须是正数。")

        # 计算裁剪区域的边界
        row_start = max(center_row - radius, 0)
        row_end = min(center_row + radius, total_rows)
        col_start = max(center_col - radius, 0)
        col_end = min(center_col + radius, total_cols)

        logging.info(f"按形状裁剪参数: center=({center_row}, {center_col}), radius={radius}, shape={shape}")
        logging.info(f"裁剪区域: rows({row_start}-{row_end}), cols({col_start}-{col_end})")

        cropped_data = self.data.iloc[row_start:row_end, col_start:col_end].copy()

        if shape == 'circle':
            # 生成遮罩
            cropped_array = cropped_data.values
            num_rows, num_cols = cropped_array.shape
            y, x = np.ogrid[:num_rows, :num_cols]
            center_y = center_row - row_start
            center_x = center_col - col_start
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2
            cropped_array[mask] = np.nan
            cropped_data = pd.DataFrame(cropped_array, index=cropped_data.index, columns=cropped_data.columns)
            logging.info("已应用圆形遮罩，圆形外区域设置为 NaN。")

        self.data = cropped_data
        logging.info(f"已裁剪数据：新数据形状 {self.data.shape}")
        return self

    def reset_coordinates(self):
        self.data.reset_index(drop=True, inplace=True)
        logging.info("已重置坐标。")
        return self

    def calculate_average_intensity(self):
        avg_intensity = self.data.mean().mean()
        logging.info(f"平均光强：{avg_intensity}")
        return avg_intensity

    def rescale(self):
        norm_data = (self.data - self.data.min().min()) / (self.data.max().max() - self.data.min().min())
        self.data = norm_data
        logging.info("数据已归一化。")
        return self


class Visualizer:
    @staticmethod
    def visualize(data, color_map='gray', title="Normalized Intensity Visualization"):
        plt.imshow(data, cmap=color_map)
        plt.colorbar()
        plt.title(title)
        plt.show()


class DataSaver(ABC):
    @abstractmethod
    def save(self, data, output_path):
        pass


class CSVSaver(DataSaver):
    def save(self, data, output_path):
        try:
            # 添加行和列坐标
            data_to_save = data.copy()
            data_to_save.index.name = 'row'
            # data_to_save.reset_index(inplace=True)
            data_to_save.columns.name = 'col'
            # 添加 "coordinate" 标记
            data_to_save.to_csv(output_path, index=False)
            logging.info(f"处理后的CSV数据已保存至 {output_path}")
        except Exception as e:
            logging.error(f"保存CSV数据失败: {e}")
            raise


class ImageSaver(DataSaver):
    def save(self, data, output_path):
        try:
            # 移除坐标列和行
            if 'row' in data.columns and 'col' in data.index.names:
                data = data.drop(['row'], axis=1)
                data.index = data['row']
                data = data.drop(['row'], axis=1)
            # 处理 NaN 值，设为0
            img_array = data.fillna(0).values * 255
            img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(output_path)
            logging.info(f"处理后的图像已保存至 {output_path}")
        except Exception as e:
            logging.error(f"保存图像数据失败: {e}")
            raise


class SensorDataProcessor:
    def __init__(self, loader: DataLoader, visualizer: Visualizer, saver: DataSaver):
        self.loader = loader
        self.visualizer = visualizer
        self.saver = saver
        self.data = self.loader.load_data()
        self.processor = DataProcessor(self.data)

    def process(self, crop_params=None, crop_by_ratio=False, crop_by_shape_params=None):
        """
        处理数据，包括裁剪、重置坐标、计算平均光强和归一化。

        :param crop_params: 如果按像素裁剪，传入 (row_start, row_end, col_start, col_end)
                            如果按比例裁剪，传入 (row_start_ratio, row_end_ratio, col_start_ratio, col_end_ratio)
        :param crop_by_ratio: 布尔值，指示是否按比例裁剪
        :param crop_by_shape_params: 如果按形状裁剪，传入一个字典，例如:
                                     {
                                         'center_row': 0.5,  # 相对坐标为0.5，即50%
                                         'center_col': 0.5,  # 相对坐标为0.5，即50%
                                         'radius': 0.1,      # 相对半径为10%
                                         'shape': 'circle',
                                         'relative': True    # 指示是否使用相对坐标
                                     }
        """
        if crop_params:
            if crop_by_ratio:
                self.processor.crop_by_ratio(*crop_params)
            else:
                self.processor.crop(*crop_params)

        if crop_by_shape_params:
            self.processor.crop_by_shape(**crop_by_shape_params)

        self.processor.reset_coordinates()
        avg_intensity = self.processor.calculate_average_intensity()
        # self.processor.rescale()  # 注释掉以保持原始数据
        return avg_intensity

    def visualize(self, color_map='gray'):
        self.visualizer.visualize(self.processor.data, color_map=color_map)

    def save(self, output_path):
        self.saver.save(self.processor.data*255, output_path)


def convert_image_to_csv(image_path, csv_path):
    """
    将PNG图像转换为CSV文件。

    :param image_path: 输入的图像文件路径
    :param csv_path: 输出的CSV文件路径
    """
    image_loader = ImageDataLoader(image_path)
    visualizer = Visualizer()
    saver = CSVSaver()
    sensor_processor = SensorDataProcessor(image_loader, visualizer, saver)
    sensor_processor.save(csv_path)
    logging.info(f"图像已成功转换为CSV文件：{csv_path}")


def convert_csv_to_image(csv_path, image_path):
    """
    将CSV文件转换为PNG图像。

    :param csv_path: 输入的CSV文件路径
    :param image_path: 输出的图像文件路径
    """
    csv_loader = CSVDataLoader(csv_path)
    visualizer = Visualizer()
    saver = ImageSaver()
    sensor_processor = SensorDataProcessor(csv_loader, visualizer, saver)
    sensor_processor.save(image_path)
    logging.info(f"CSV文件已成功转换为图像：{image_path}")


# 示例使用
if __name__ == "__main__":
    # 配置加载器、可视化器和保存器
    csv_loader = CSVDataLoader("./1550/Au-fake.csv")
    image_loader = ImageDataLoader("./1550/Au-fake.png")  # 假设有对应的PNG文件
    visualizer = Visualizer()
    csv_saver = CSVSaver()
    image_saver = ImageSaver()

    # 初始化处理类
    # 示例1: 按比例裁剪
    sensor_processor_csv = SensorDataProcessor(csv_loader, visualizer, csv_saver)
    crop_parameters_ratio = (0.0, 1.0, 0.0, 1.0)  # 行从0%到100%，列从0%到100%
    avg_intensity_ratio = sensor_processor_csv.process(crop_params=crop_parameters_ratio, crop_by_ratio=True)

    print(f"按比例裁剪后的平均光强：{avg_intensity_ratio}")
    sensor_processor_csv.visualize(color_map='gray')
    sensor_processor_csv.save("processed_sensor_data_ratio.csv")

    # 示例2: 按形状裁剪（圆形）使用相对坐标
    sensor_processor_csv = SensorDataProcessor(csv_loader, visualizer, csv_saver)
    crop_shape_params_relative = {
        'center_col': 0.46,  # 相对坐标
        'center_row': 0.38,  # 相对坐标
        'radius': 0.32,      # 相对半径
        'shape': 'circle',
        'relative': True     # 使用相对坐标
    }
    avg_intensity_circle_relative = sensor_processor_csv.process(crop_by_shape_params=crop_shape_params_relative)

    print(f"圆形裁剪（相对坐标）后的平均光强：{avg_intensity_circle_relative}")
    sensor_processor_csv.visualize(color_map='gray')
    sensor_processor_csv.save("processed_sensor_data_circle_relative.csv")

    # 示例3: 按形状裁剪（方形）使用绝对像素坐标
    sensor_processor_csv = SensorDataProcessor(csv_loader, visualizer, csv_saver)
    crop_shape_params_square_absolute = {
        'center_col': 300,    # 绝对像素坐标
        'center_row': 500,    # 绝对像素坐标
        'radius': 100,        # 绝对像素半径
        'shape': 'square',
        'relative': False     # 使用绝对坐标
    }
    avg_intensity_square_absolute = sensor_processor_csv.process(crop_by_shape_params=crop_shape_params_square_absolute)

    print(f"方形裁剪（绝对坐标）后的平均光强：{avg_intensity_square_absolute}")
    sensor_processor_csv.visualize(color_map='gray')
    sensor_processor_csv.save("processed_sensor_data_square_absolute.csv")

    # 示例4: 将图像转换为CSV
    # 假设有一个名为 "Au.png" 的图像文件
    image_path = "./1550/Au-fake.png"
    csv_output_path = "./1550/Au-fake.csv"
    if os.path.exists(image_path):
        convert_image_to_csv(image_path, csv_output_path)
    else:
        logging.warning(f"图像文件 {image_path} 不存在，跳过图像到CSV的转换。")

    # 示例5: 将CSV转换为图像
    csv_input_path = "./1550/Au.csv"
    image_output_path = "./1550/Au.png"
    if os.path.exists(csv_input_path):
        convert_csv_to_image(csv_input_path, image_output_path)
    else:
        logging.warning(f"CSV文件 {csv_input_path} 不存在，跳过CSV到图像的转换。")
