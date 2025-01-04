from pathlib import Path
import logging
from ..core.data_loading import ImageDataLoader
from ..core.data_processing import DataProcessor
from ..core.data_saving import CSVSaver
from ..core.visualization import Visualizer

class BatchProcessor:
    """
    负责批量处理输入目录中的所有图像文件。
    """
    def __init__(self, input_dir: Path, output_dir: Path, crop_shape_params: dict):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.crop_shape_params = crop_shape_params
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"输出目录已设置为: {self.output_dir}")

    def process_all(self):
        """
        批量处理所有支持格式的图像文件。
        """
        image_files = list(self.input_dir.glob("*"))
        if not image_files:
            logging.warning(f"在 {self.input_dir} 中未找到任何图像文件。")
            return

        for image_file in image_files:
            try:
                # 提取波长信息，假设文件名格式为 "{wavelength}-*.bmp"
                wavelength_str = image_file.stem.split('-')[0]
                wavelength = float(wavelength_str)
                logging.info(f"开始处理文件: {image_file.name} (波长: {wavelength} nm)")

                # 数据加载
                loader = ImageDataLoader(image_file)
                data = loader.load_data()

                # 数据处理
                processor = DataProcessor(data)
                processor.crop_by_shape(**self.crop_shape_params)
                processor.reset_coordinates()
                avg_intensity = processor.calculate_average_intensity()
                logging.info(f"文件 {image_file.name} 的平均光强: {avg_intensity}")

                # 数据保存
                saver = CSVSaver()
                output_csv = self.output_dir / f"{wavelength}.csv"
                saver.save(processor.data, output_csv)

                # 可视化（可选）
                # visualizer = Visualizer()
                # visualizer.visualize(processor.data, title=f"Wavelength {wavelength} nm")

            except Exception as e:
                logging.error(f"处理文件 {image_file.name} 时发生错误: {e}")
                continue
